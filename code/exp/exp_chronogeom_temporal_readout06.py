"""Mean-preserving temporal readout with retained TimeBase. No new LLM or encoder."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,csv,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import exp.exp_chronogeom_temporal_kd04 as old
from exp.exp_chronogeom_downstream import predict
from exp.exp_chronogeom_residual03 import frozen_base,features
from exp.exp_chronogeom_qwen_capital import windows,setup,tensorpack
from scripts.chronogeom.task_context02_analysis import read,save,csvsave
sha=old.sha;git=old.git
def center(x):return x-x.mean(1,keepdim=True)
def components(pred,y):
 e=np.asarray(pred,dtype=float)-np.asarray(y,dtype=float);mu=e.mean(1,keepdims=True);l=(mu*mu).mean((1,2));s=((e-mu)**2).mean((1,2));m=(e*e).mean((1,2));assert np.max(abs(m-l-s))<1e-12
 return m,l,s,abs(e).mean((1,2))
def ix(c,t):return int((np.datetime64(t,"h")-np.datetime64(c["base"],"h"))/np.timedelta64(1,"h"))
def context(c):
 rr=Path(c["reuse_run"]);oc=read(rr/"config.json");tr=Path(oc["teacher_run"]);tc=read(tr/"config.json");gc=read(Path(tc["source_run"])/"config.json")
 return rr,oc,tr,tc,gc
class TimeReadout(nn.Module):
 def __init__(self,base,table,seed,mode):
  super().__init__();self.base=copy.deepcopy(base);self.mode=mode
  for p in self.base.parameters():p.requires_grad_(False)
  self.base.eval()
  with torch.no_grad():
   q=self.base.readout(self.base.codes());assert torch.equal(q,q[:1].expand_as(q))
  self.register_buffer("old_effect",q[0].clone());self.register_buffer("table",torch.tensor(table,dtype=torch.float32,device=q.device))
  torch.manual_seed(seed+10000);self.time_mlp=nn.Sequential(nn.Linear(168,32),nn.GELU(),nn.Linear(32,32)).to(q.device);self.readout=nn.Linear(32,2,bias=False).to(q.device);nn.init.zeros_(self.readout.weight)
 def train(self,mode=True):
  super().train(mode);self.base.eval();return self
 def baseline(self,raw):return raw*(1+torch.tanh(self.old_effect[0]))+self.old_effect[1]
 def from_raw(self,raw,ids):
  z=self.readout(self.time_mlp(self.table))[ids];delta=raw*(torch.tanh(self.old_effect[0]+z[...,:1])-torch.tanh(self.old_effect[0]))+z[...,1:]
  delta=delta.double();p0=self.baseline(raw).double()
  return p0+(center(delta) if self.mode=="level" else delta)
 def forward(self,x,ids):
  with torch.no_grad():raw=self.base.body(x)
  return self.from_raw(raw,ids)
def raw_predict(base,p,device):
 pp=[]
 with torch.no_grad():
  for j in range(0,len(p["x"]),128):pp.append(base.body(torch.tensor(p["x"][j:j+128],device=device)).cpu().numpy())
 return np.concatenate(pp)
def forecast(m,raw,ids,device):
 out=[];m.eval()
 with torch.no_grad():
  for j in range(0,len(raw),128):out.append(m.from_raw(torch.tensor(raw[j:j+128],device=device),torch.tensor(ids[j:j+128],device=device)).cpu().numpy())
 return np.concatenate(out)
def prepare(c,r):
 assert not (r/"config.json").exists();save(r/"config.json",c);rr,oc,tr,tc,gc=context(c);b=np.load(rr/"power/known_data.npz");p=old.packed(b,b["train_origins"]);w=b["state_weights"].astype(float);tabs=np.load(tr/"power/tables.npz")
 sources=read(rr/"source.json")["teacher_sources"]
 for q in sources:assert sha(q["path"])==q["sha256"]
 tables={}
 for name in {v["table"] for v in c["supervised_groups"].values()}:tables[name]=old.center_rms(tabs[name],w)[0]
 np.savez_compressed(r/"tables.npz",**tables);source=dict(commit=git(),source_sha=sha(__file__),config_sha=sha("docs/chronogeom/temporal_readout06_config.json"),timebase_sha=sha("layers/ChronoGeomTimeBaseOfficial.py"),teacher_sources=sources,known_data_sha=sha(rr/"power/known_data.npz"),new_targets_unread=True)
 save(r/"source.json",source);print("PREPARED",len(p["x"]),"training origins",flush=True)
def groupdefs(c,kd):
 return c["kd_groups"] if kd else c["supervised_groups"]
def fit(c,r,seed,device,kd=False):
 setup(device,c["gpu_cap_gib"]);tick=time.monotonic();rr,oc,tr,tc,gc=context(c);b=np.load(rr/"power/known_data.npz");p=old.packed(b,b["train_origins"]);v,_=windows(b["values"],b["ids"],ix(c,c["validation"][0]),ix(c,c["validation"][1]));tabs=np.load(r/"tables.npz")
 base,rec=frozen_base(tc,tr,"power",seed,device);raw=raw_predict(base,p,device);vr=raw_predict(base,v,device);rt=torch.tensor(raw,device=device);idt=torch.tensor(p["ids"],device=device);yt=torch.tensor(p["y"],dtype=torch.float64,device=device)
 out=r/(("kd_" if kd else "sup_")+f"seed{seed}");out.mkdir(exist_ok=False);defs=groupdefs(c,kd);records=[];cache=np.load(rr/"power"/f"seed{seed}/teacher_training_cache.npz") if kd else None
 if kd:assert read(r/"gate.json")["passed"]
 for name,desc in defs.items():
  if time.monotonic()-tick>c["fit_cap_seconds"]:raise TimeoutError("fit budget")
  table=tabs["Calendar32" if kd else desc["table"]];mode="level" if kd else desc["mode"];m=TimeReadout(base,table,seed,mode).to(device);params=[p for p in m.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==6528 and sum(p.numel() for p in m.parameters())==13098
  target=torch.tensor(cache[desc["teacher"]],dtype=torch.float64,device=device) if kd else None;lam=0.;cal=None;rng=np.random.default_rng(seed+271);opt=torch.optim.Adam(params,lr=c["lr"]);curve=[];best=float("inf");start=time.monotonic();grad0=None;grad1=None
  initial=forecast(m,vr,v["ids"],device);reference=m.baseline(torch.tensor(vr,device=device)).double().detach().cpu().numpy();assert np.max(abs(initial-reference))<2e-6
  for step in range(c["steps"]+1):
   if time.monotonic()-tick>c["fit_cap_seconds"]:raise TimeoutError("fit budget")
   if step%c["validation_every"]==0:
    vp=forecast(m,vr,v["ids"],device);mse,level,shape,mae=components(vp,v["y"]);curve.append(dict(step=step,mse=float(mse.mean()),mae=float(mae.mean()),level=float(level.mean()),shape=float(shape.mean())))
    if mse.mean()<best:best=float(mse.mean());beststep=step;state=copy.deepcopy(m.state_dict())
   if step==c["steps"]:break
   ii=torch.tensor(rng.integers(len(raw),size=c["batch_size"]),device=device);m.train();yp=m.from_raw(rt[ii],idt[ii]);sup=F.mse_loss(yp,yt[ii]);loss=sup
   if kd:
    shape_aux=F.mse_loss(center(yp),center(target[ii]));output_aux=F.mse_loss(yp,target[ii]);aux=shape_aux if desc["loss"]=="shape" else output_aux
    if step==0:
     gs=torch.cat([z.reshape(-1) if z is not None else torch.zeros_like(q).reshape(-1) for z,q in zip(torch.autograd.grad(sup,params,retain_graph=True,allow_unused=True),params)])
     ga=torch.cat([z.reshape(-1) if z is not None else torch.zeros_like(q).reshape(-1) for z,q in zip(torch.autograd.grad(shape_aux,params,retain_graph=True,allow_unused=True),params)])
     go=torch.cat([z.reshape(-1) if z is not None else torch.zeros_like(q).reshape(-1) for z,q in zip(torch.autograd.grad(output_aux,params,retain_graph=True,allow_unused=True),params)])
     lam=.25*float(gs.norm()/ga.norm()) if ga.norm()>1e-10 else 0.
     constant=float(((yp.mean(1)-target[ii].mean(1))**2).mean());cal=dict(lambda_value=lam,supervised_grad=float(gs.norm()),shape_grad=float(ga.norm()),output_shape_grad_max_difference=float((go-ga).abs().max()),output_shape_grad_relative_difference=float((go-ga).norm()/ga.norm().clamp_min(1e-12)),loss_constant_identity_error=float(abs(float(output_aux-shape_aux)-constant)))
     assert cal["output_shape_grad_relative_difference"]<1e-5
    loss=loss+lam*aux
   opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(loss);torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
   if step==0:grad0=float(m.readout.weight.grad.norm());assert grad0>1e-10
   if step==1:grad1=float(m.time_mlp[0].weight.grad.norm());assert grad1>1e-12
   opt.step()
  m.load_state_dict(state);path=out/(name+".pt");torch.save(state,path);csvsave(out/(name+"_curve.csv"),curve);pp=forecast(m,vr,v["ids"],device);np.save(out/(name+"_validation.npy"),pp)
  basepred=m.baseline(torch.tensor(vr,device=device)).double().detach().cpu().numpy();meanerr=float(np.max(abs(pp.mean(1)-basepred.mean(1))))
  if mode=="level":assert meanerr<2e-6
  record=dict(name=name,seed=seed,kind="kd" if kd else "supervised",mode=mode,table="Calendar32" if kd else desc["table"],teacher=desc.get("teacher"),kd_loss=desc.get("loss"),parameters=13098,trainable=6528,best_step=beststep,val_mse=best,path=str(path),sha256=sha(path),steps=600,seconds=time.monotonic()-start,head_initial_grad=grad0,upstream_after_update_grad=grad1,initial_prediction_maxdiff=float(np.max(abs(initial-reference))),selected_mean_maxdiff=meanerr,calibration=cal)
  records.append(record);save(out/"records.json",records);print("FIT",json.dumps(record),flush=True);assert all(not p.requires_grad and p.grad is None for p in m.base.parameters());del m,opt
 assert sha(rec["path"])==rec["sha256"];save(out/"completion.json",dict(seconds=time.monotonic()-tick,models=len(records),steps_each=600,base_unchanged=True,peak_gib=torch.cuda.max_memory_allocated(device)/2**30))
def gate(c,r):
 rr,oc,tr,tc,gc=context(c);b=np.load(rr/"power/known_data.npz");v,_=windows(b["values"],b["ids"],ix(c,c["validation"][0]),ix(c,c["validation"][1]));baseline=[];values=[];details=[]
 for seed in c["seeds"]:
  rows=read(r/f"sup_seed{seed}/records.json");rec=next(q for q in rows if q["name"]=="B_Calendar");values.append(rec["val_mse"]);base,_=frozen_base(tc,tr,"power",seed,"cpu")
  oldpred=predict(base,tensorpack(v,"cpu"));baseline.append(float(np.mean((oldpred-v["y"])**2)))
  details.append(dict(seed=seed,baseline=baseline[-1],B_Calendar=values[-1],improvement=baseline[-1]-values[-1],B_Task=next(q["val_mse"] for q in rows if q["name"]=="B_Task")))
 decomp=list(csv.DictReader((r/"kd05_error_decomposition.csv").open()));te=next(q for q in decomp if q["domain"]=="power" and q["stage"]=="development_validation" and q["model"]=="Teacher__TaskKD");be=next(q for q in decomp if q["domain"]=="power" and q["stage"]=="development_validation" and q["model"]=="FrozenBase")
 gains=np.array(baseline)-values;passed=bool(gains.mean()>1e-4 and (gains>1e-6).sum()>=2 and (gains>=-1e-6).all() and float(te["shape_error"])<float(be["shape_error"]))
 save(r/"gate.json",dict(passed=passed,details=details,mean_gain=float(gains.mean()),teacher_validation_shape=float(te["shape_error"]),base_validation_shape=float(be["shape_error"]),only_development_used=True,new_targets_unread=True));print("GATE",read(r/"gate.json"),flush=True)
def lock(c,r):
 records=[]
 for seed in c["seeds"]:
  records+=read(r/f"sup_seed{seed}/records.json")
  if read(r/"gate.json")["passed"]:records+=read(r/f"kd_seed{seed}/records.json")
 for q in records:assert sha(q["path"])==q["sha256"]
 save(r/"lock.json",dict(commit=git(),models=records,new_targets_unread=True,gate=read(r/"gate.json")));print("LOCKED",len(records),flush=True)
def evaluate(c,r,device):
 setup(device,c["gpu_cap_gib"]);rr,oc,tr,tc,gc=context(c);b=np.load(rr/"power/known_data.npz");tabs=np.load(r/"tables.npz");locked=read(r/"lock.json")
 stages=dict(validation=c["validation"],**c["development_intervals"],confirmation=c["confirmation"])
 for stage,bounds in stages.items():
  out=r/stage;out.mkdir(exist_ok=False)
  for q in locked["models"]:assert sha(q["path"])==q["sha256"]
  save(out/"before_targets.json",dict(commit=git(),models_locked=True,independent=stage=="confirmation"))
  if stage=="validation":values,ids=b["values"],b["ids"]
  else:
   raw,ids,_=old.grid(gc,"power",bounds[1]);values=(raw-b["mean"])/b["sd"]
  lo,hi=[ix(c,t) for t in bounds];p,counts=windows(values,ids,lo,hi);preds={};checks=[]
  for seed in c["seeds"]:
   base,_=frozen_base(tc,tr,"power",seed,device);raw=raw_predict(base,p,device);preds[f"FrozenBase_s{seed}"]=predict(base,tensorpack(p,device)).astype(float)
   for rec in [q for q in locked["models"] if q["seed"]==seed]:
    m=TimeReadout(base,tabs[rec["table"]],seed,rec["mode"]).to(device);m.load_state_dict(torch.load(rec["path"],map_location=device,weights_only=True));pp=forecast(m,raw,p["ids"],device);preds[rec["name"]+f"_s{seed}"]=pp
    meanerr=float(np.max(abs(pp.mean(1)-preds[f"FrozenBase_s{seed}"].mean(1))))
    if rec["mode"]=="level":assert meanerr<2e-6
    checks.append(dict(name=rec["name"],seed=seed,max_mean_difference=meanerr));del m
   normal=np.load(tr/"power"/f"seed{seed}/numeric_features.npz");ft=features(base,p,device,normal["mean"],normal["sd"])
   for teacher in ["TaskKD","GenericKD","ShuffleKD","FourierKD"]:
    tm=old.load_teacher(oc,rr,"power",seed,teacher,device);preds["Teacher_"+teacher+f"_s{seed}"]=old.teacher_predict(tm,ft,p,device).astype(float);del tm
   del base
  assert all(np.isfinite(v).all() and v.shape==p["y"].shape for v in preds.values());np.savez_compressed(out/"predictions.npz",truth=p["y"],origin=p["origin"],**preds);csvsave(out/"mean_preservation.csv",checks);save(out/"completion.json",dict(counts=counts,lo=lo,hi=hi,independent=stage=="confirmation"))
  print("EVALUATED",stage,counts,flush=True)
def summarize(c,r):
 out=r/"summary";out.mkdir(exist_ok=False);scores=[];cis=[];windowsrows=[]
 for stage in ["validation","spring","summer","confirmation"]:
  p=np.load(r/stage/"predictions.npz");truth=p["truth"].astype(float);orig=p["origin"];co=read(r/stage/"completion.json");days=np.arange(co["lo"],co["hi"]-96+1,24);pos=(orig-co["lo"])//24;errors={}
  for name in sorted(set(k.rsplit("_s",1)[0] for k in p.files if "_s" in k)):
   mm=[];ll=[];ss=[];aa=[]
   for seed in c["seeds"]:
    mse,lev,shape,mae=components(p[name+f"_s{seed}"],truth);mm.append(mse);ll.append(lev);ss.append(shape);aa.append(mae)
    for i,o in enumerate(orig):windowsrows.append(dict(stage=stage,name=name,seed=seed,origin=int(o),mse=float(mse[i]),level=float(lev[i]),shape=float(shape[i]),mae=float(mae[i])))
   mm,ll,ss,aa=map(np.stack,[mm,ll,ss,aa]);errors[name]=dict(mse=mm,level=ll,shape=ss,mae=aa);row=dict(stage=stage,name=name,mse=float(mm.mean()),mae=float(aa.mean()),level=float(ll.mean()),shape=float(ss.mean()),n_origins=len(orig))
   for j,seed in enumerate(c["seeds"]):row.update({f"seed{seed}_mse":float(mm[j].mean()),f"seed{seed}_mae":float(aa[j].mean()),f"seed{seed}_level":float(ll[j].mean()),f"seed{seed}_shape":float(ss[j].mean())})
   scores.append(row)
  pairs=[("B_Task",b) for b in ["FrozenBase","A_Task","B_Fourier","B_Generic","B_Shuffle","B_Calendar"]]
  if "KD_Task_Output" in errors:
   pairs += [("KD_Task_Output",b) for b in ["FrozenBase","B_Calendar","KD_Generic_Output","KD_Shuffle_Output","KD_Fourier_Output","KD_Task_Shape"]]
  rng=np.random.default_rng(c["bootstrap_seed"]);n=c["bootstrap_replicates"];si=rng.integers(3,size=(n,3));st=rng.integers(len(days),size=(n,int(np.ceil(len(days)/c["block_days"]))));ti=((st[:,:,None]+np.arange(c["block_days"]))%len(days)).reshape(n,-1)[:,:len(days)]
  for a,b in pairs:
   row=dict(stage=stage,comparison=a+"-"+b)
   for key in ["mse","level","shape","mae"]:
    diff=errors[a][key]-errors[b][key];ar=np.full((3,len(days)),np.nan);ar[:,pos]=diff;draw=np.nanmean(ar[si[:,:,None],ti[:,None,:]],axis=(1,2));ci=np.nanquantile(draw,[.025,.975]);row[key+"_difference"]=float(diff.mean());row[key+"_lo95"]=float(ci[0]);row[key+"_hi95"]=float(ci[1])
   row["all3_better"]=bool(np.all((errors[a]["mse"]-errors[b]["mse"]).mean(1)<0));cis.append(row)
 csvsave(out/"scores.csv",scores);csvsave(out/"contrasts.csv",cis);csvsave(out/"window_errors.csv",windowsrows)
 if read(r/"gate.json")["passed"]:
  q=np.load(r/"confirmation/predictions.npz");eq=[]
  for seed in c["seeds"]:eq.append(dict(seed=seed,output_shape_prediction_max_difference=float(np.max(abs(q[f"KD_Task_Output_s{seed}"]-q[f"KD_Task_Shape_s{seed}"])))))
  save(out/"output_shape_equivalence.json",dict(theoretical_equivalence="For frozen mean, full MSE = centered MSE + constant; gradients equal",confirmation=eq))
 print("RESULTS",json.dumps([s for s in scores if s["stage"]=="confirmation"]),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["prepare","sup","gate","kd","lock","evaluate","summarize"]);ap.add_argument("--seed",type=int);ap.add_argument("--device",default="cuda:0");a=ap.parse_args();c=read("docs/chronogeom/temporal_readout06_config.json");r=Path(c["run"])
 try:
  if a.stage=="prepare":prepare(c,r)
  elif a.stage in ["sup","kd"]:fit(c,r,a.seed,a.device,a.stage=="kd")
  elif a.stage=="gate":gate(c,r)
  elif a.stage=="lock":lock(c,r)
  elif a.stage=="evaluate":evaluate(c,r,a.device)
  else:summarize(c,r)
 except BaseException as e:
  save(r/f"failure_{a.stage}_{a.seed}.json",dict(error=str(e),traceback=traceback.format_exc()));raise
if __name__=="__main__":main()
