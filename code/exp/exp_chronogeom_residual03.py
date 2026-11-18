"""Frozen numerical TimeBase plus matched residual readouts. No LLM or KD training."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,hashlib,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from exp.exp_chronogeom_downstream import CalendarTimeBase,predict
from exp.exp_chronogeom_task_context02 import grid as source_grid,ix as source_ix
from exp.exp_chronogeom_qwen_capital import windows,setup,tensorpack
from scripts.chronogeom.task_context02_analysis import read,save,csvsave
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git():return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
def ix(c,domain,date):return int((np.datetime64(date,"h")-np.datetime64(c["domains"][domain]["base"],"h"))/np.timedelta64(1,"h"))
class ResidualReadout(nn.Module):
 def __init__(self,seed,table):
  super().__init__();torch.manual_seed(seed+20000);self.register_buffer("table",torch.tensor(table,dtype=torch.float32))
  self.mlp=nn.Sequential(nn.Linear(200,32),nn.GELU(),nn.Linear(32,1));self.numeric=nn.Linear(8,1);self.lead_bias=nn.Parameter(torch.zeros(96))
  self.register_buffer("lead_onehot",F.one_hot(torch.arange(96)%24,24).float())
  nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias);nn.init.zeros_(self.numeric.weight);nn.init.zeros_(self.numeric.bias)
 def forward(self,num,ids):
  lead=self.lead_onehot[None].expand(len(num),-1,-1);timecodes=self.table[ids];z=torch.cat([num,lead,timecodes],-1)
  return self.mlp(z)+self.numeric(num)+self.lead_bias[None,:,None]
def numeric_summary(x,p0):
 n=len(x);repeated=lambda v:v[:,None,None].expand(n,96,1)
 early=x[:,:24,0].mean(1);late=x[:,-24:,0].mean(1)
 cols=[repeated(x[:,:,0].mean(1)),repeated(x[:,:,0].std(1,unbiased=False)),repeated(x[:,-1,0]),repeated(late),repeated(x[:,-24:,0].std(1,unbiased=False)),repeated(late-early),x[:,-24:].repeat(1,4,1),p0]
 return torch.cat(cols,-1)
def prepare(c,r):
 old=Path(c["source_run"]);sc=read(old/"config.json");lock=read(old/"lock.json");source_tables=np.load(old/"tables.npz")
 for domain,dc in c["domains"].items():
  dr=r/domain;dr.mkdir(exist_ok=False);end=dc["residual_validation"][1]
  raw,ids,dates=source_grid(sc,domain,end);b=np.load(old/domain/"train_val.npz");values=(raw-b["mean"])/b["sd"]
  lo,hi=[ix(c,domain,t) for t in dc["residual_fit"]];origins=np.arange(lo,hi-96+1);good=np.array([np.isfinite(values[o-96:o+96]).all() for o in origins]);origins=origins[good];assert len(origins)>100
  va,counts=windows(values,ids,ix(c,domain,dc["residual_validation"][0]),ix(c,domain,end))
  frequency=np.bincount(np.concatenate([ids[o:o+96] for o in origins]),minlength=168).astype(float);weights=frequency/frequency.sum()
  native={"XOnly":np.zeros((168,168)),"Fourier32":source_tables["Fourier32"],"Calendar32":source_tables["Calendar32"],"Calendar168":source_tables["Joint168"],"QwenGeneric32":source_tables["Generic32"],"QwenTask32":source_tables[dc["correct_context"]],"QwenWrong32":source_tables[dc["wrong_context"]],"QwenContextShuffle32":source_tables[dc["shuffle_context"]]}
  rng=np.random.default_rng(c["feature_seed"]);random=rng.normal(size=(168,32));padded=np.zeros((168,168));padded[:,:32]=random;mix=np.linalg.qr(rng.normal(size=(168,168)))[0]
  native["Random32"]=padded@mix;perm=rng.permutation(168);native["TaskIdentityShuffle32"]=native["QwenTask32"][perm]
  tables={};diagnostics=[]
  for name,z in native.items():
   z=z.astype(float);center=(weights[:,None]*z).sum(0);z=z-center;scale=np.sqrt(np.sum(weights[:,None]*z*z)/168);z=z/max(scale,1e-12);tables[name]=z.astype("float32")
   sv=np.linalg.svd(z,compute_uv=False);diagnostics.append(dict(arm=name,train_weighted_rms=float(np.sqrt(np.sum(weights[:,None]*z*z)/168)),rank=int(np.sum(sv>max(float(sv[0]),1e-12)*1e-6)),min_row_norm=float(np.linalg.norm(z,axis=1).min()),max_row_norm=float(np.linalg.norm(z,axis=1).max())))
  np.savez_compressed(dr/"tables.npz",**tables);csvsave(dr/"feature_diagnostics.csv",diagnostics)
  np.savez_compressed(dr/"known_data.npz",values=values,ids=ids,mean=b["mean"],sd=b["sd"],train_origins=origins)
  backbones=[m for m in lock["domains"][domain]["models"] if m["arm"]=="NoCalendar"];assert len(backbones)==3
  for m in backbones:assert sha(m["path"])==m["sha256"]
  save(dr/"prepared.json",dict(commit=git(),domain=domain,backbones=backbones,data_sha=sha(dc["path"]),train_origins=len(origins),train_candidates=len(good),train_excluded_missing=int((~good).sum()),validation=counts,residual_targets_disjoint_backbone_fit_and_selection=dc["residual_fit"][0]>=dc["backbone_selection"][1],old_scaler_preserved=True,tables_sha=sha(dr/"tables.npz"),identity_permutation=perm.tolist(),state_frequency=frequency.tolist()))
  h=ResidualReadout(c["seeds"][0],tables["QwenTask32"]);assert sum(p.numel() for p in h.parameters())==6570;assert torch.count_nonzero(h(torch.randn(2,96,8),torch.arange(96).repeat(2,1)))==0
  print("PREPARED",domain,read(dr/"prepared.json")["train_origins"],counts,flush=True)
 save(r/"source.json",dict(commit=git(),source_sha=sha(__file__),config_sha=sha("docs/chronogeom/residual03_config.json"),source_run=c["source_run"],timebase_sha=sha("layers/ChronoGeomTimeBaseOfficial.py")))
def frozen_base(c,r,domain,seed,device):
 rec=next(m for m in read(r/domain/"prepared.json")["backbones"] if m["seed"]==seed)
 assert sha(rec["path"])==rec["sha256"]
 m=CalendarTimeBase(c,seed,np.zeros((168,168),dtype="float32")).to(device);m.load_state_dict(torch.load(rec["path"],map_location=device,weights_only=True));m.eval()
 for p in m.parameters():p.requires_grad_(False)
 return m,rec
def packed_train(b):
 origins=b["train_origins"];v,ids=b["values"],b["ids"]
 return dict(x=np.stack([v[o-96:o] for o in origins]),y=np.stack([v[o:o+96] for o in origins]),ids=np.stack([ids[o:o+96] for o in origins]),origin=origins)
def features(m,p,device,mu=None,sd=None):
 p0=predict(m,tensorpack(p,device));x=torch.tensor(p["x"],device=device);p0t=torch.tensor(p0,device=device)
 with torch.no_grad():num=numeric_summary(x,p0t).cpu().numpy()
 if mu is None:mu=num.mean((0,1));sd=num.std((0,1));sd=np.maximum(sd,1e-3)
 num=(num-mu)/sd
 return dict(num=num,p0=p0,mu=mu,sd=sd)
def head_predict(h,packed,device):
 out=[];h.eval()
 with torch.no_grad():
  for j in range(0,len(packed["num"]),128):out.append((h(torch.tensor(packed["num"][j:j+128],device=device),torch.tensor(packed["ids"][j:j+128],device=device))+torch.tensor(packed["p0"][j:j+128],device=device)).cpu().numpy())
 return np.concatenate(out)
def fit(c,r,domain,seed,device):
 setup(device);dr=r/domain;out=dr/f"seed{seed}";out.mkdir(exist_ok=False);b=np.load(dr/"known_data.npz");dc=c["domains"][domain];tables=np.load(dr/"tables.npz")
 tr=packed_train(b);va,_=windows(b["values"],b["ids"],ix(c,domain,dc["residual_validation"][0]),ix(c,domain,dc["residual_validation"][1]))
 base,base_rec=frozen_base(c,r,domain,seed,device);started=time.monotonic();trf=features(base,tr,device);vaf=features(base,va,device,trf["mu"],trf["sd"])
 np.savez_compressed(out/"numeric_features.npz",mean=trf["mu"],sd=trf["sd"],train_p0=trf["p0"],validation_p0=vaf["p0"])
 np.save(out/"NoCorrection_validation.npy",vaf["p0"]);baseline_mse=float(np.mean((vaf["p0"]-va["y"])**2))
 records=[dict(arm="NoCorrection",seed=seed,parameters=0,total_parameters=6570,best_step=None,val_mse=baseline_mse,path=base_rec["path"],sha256=base_rec["sha256"],seconds=0.)]
 vt=dict(num=vaf["num"],p0=vaf["p0"],ids=va["ids"]);nt=torch.tensor(trf["num"],device=device);p0t=torch.tensor(trf["p0"],device=device);yt=torch.tensor(tr["y"],device=device);idt=torch.tensor(tr["ids"],device=device);xt=torch.tensor(tr["x"],device=device);vpt=tensorpack(va,device)
 for arm in c["arms"][1:]:
  if time.monotonic()-started>c["fit_cap_seconds"]:raise TimeoutError("Per seed fitting cap")
  ensemble=arm=="IndependentEnsemble"
  m=CalendarTimeBase(c,seed+1000,np.zeros((168,168),dtype="float32")).to(device) if ensemble else ResidualReadout(seed,tables[arm]).to(device)
  assert sum(p.numel() for p in m.parameters())==6570
  opt=torch.optim.Adam(m.parameters(),lr=c["lr"]);rng=np.random.default_rng(seed+271);curve=[];best=float("inf");tick=time.monotonic()
  for step in range(c["steps"]+1):
   if time.monotonic()-started>c["fit_cap_seconds"]:raise TimeoutError("Per seed fitting cap")
   if step%100==0:
    pred=(predict(m,vpt)+vaf["p0"])/2 if ensemble else head_predict(m,vt,device)
    mse=float(np.mean((pred-va["y"])**2));mae=float(np.mean(abs(pred-va["y"])));assert np.isfinite(mse)
    curve.append(dict(step=step,mse=mse,mae=mae))
    if mse<best:best=mse;beststep=step;state=copy.deepcopy(m.state_dict())
   if step==c["steps"]:break
   ids=torch.tensor(rng.integers(len(tr["x"]),size=32),device=device);m.train()
   # Ensemble member is trained independently against actual y; no residual/KD target.
   prediction=m(xt[ids],idt[ids]) if ensemble else p0t[ids]+m(nt[ids],idt[ids])
   loss=F.mse_loss(prediction,yt[ids]);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
  m.load_state_dict(state);path=out/(arm+".pt");torch.save(state,path);csvsave(out/(arm+"_curve.csv"),curve)
  pred=(predict(m,vpt)+vaf["p0"])/2 if ensemble else head_predict(m,vt,device);np.save(out/(arm+"_validation.npy"),pred)
  rec=dict(arm=arm,seed=seed,parameters=6570,total_parameters=13140,best_step=beststep,val_mse=best,path=str(path),sha256=sha(path),seconds=time.monotonic()-tick,initial_val_mse=curve[0]["mse"],independent_member_seed=seed+1000 if ensemble else None)
  records.append(rec);save(out/"records.json",records);print("FIT",domain,json.dumps(rec),flush=True);del m,opt
 assert all(p.grad is None and not p.requires_grad for p in base.parameters());assert sha(base_rec["path"])==base_rec["sha256"]
 save(out/"completion.json",dict(commit=git(),domain=domain,seed=seed,seconds=time.monotonic()-started,trained_models=11,updates_each=600,base_parameters_frozen=6570,base_checkpoint_unchanged=True,peak_gib=torch.cuda.max_memory_allocated(device)/2**30))
def lock(c,r):
 allmodels={};seconds=0.
 for domain in c["domains"]:
  recs=[]
  for seed in c["seeds"]:
   recs+=read(r/domain/f"seed{seed}/records.json");seconds+=read(r/domain/f"seed{seed}/completion.json")["seconds"]
  assert len(recs)==36
  for rec in recs:assert sha(rec["path"])==rec["sha256"]
  means={arm:float(np.mean([m["val_mse"] for m in recs if m["arm"]==arm])) for arm in c["arms"]};strong=min(c["strong_candidates"],key=means.get)
  allmodels[domain]=dict(records=recs,validation_means=means,strong=strong)
 assert seconds<c["total_fit_worker_cap_seconds"];save(r/"lock.json",dict(commit=git(),domains=allmodels,fit_worker_seconds=seconds,new_power_targets_opened=False));print("LOCKED",json.dumps({k:{"validation":v["validation_means"],"strong":v["strong"]} for k,v in allmodels.items()}),flush=True)
def evaluate(c,r,device):
 setup(device);lockdata=read(r/"lock.json");source_cfg=read(Path(c["source_run"])/"config.json")
 for domain,dc in c["domains"].items():
  b=np.load(r/domain/"known_data.npz");tables=np.load(r/domain/"tables.npz")
  stages=["development_validation"]+(["new_confirmation"] if dc["test"] is not None else [])
  for stage in stages:
   out=r/domain/stage;out.mkdir(exist_ok=False)
   for rec in lockdata["domains"][domain]["records"]:assert sha(rec["path"])==rec["sha256"]
   save(out/"before_targets.json",dict(commit=git(),all_domain_models_locked=True,new_confirmation=stage=="new_confirmation"))
   if stage=="development_validation":values,ids=b["values"],b["ids"];lo,hi=[ix(c,domain,x) for x in dc["residual_validation"]]
   else:
    raw,ids,dates=source_grid(source_cfg,domain,dc["test"][1]);values=(raw-b["mean"])/b["sd"];lo,hi=[ix(c,domain,x) for x in dc["test"]]
   p,counts=windows(values,ids,lo,hi);preds={};vp=tensorpack(p,device);rows=[]
   for seed in c["seeds"]:
    base,base_rec=frozen_base(c,r,domain,seed,device);normal=np.load(r/domain/f"seed{seed}/numeric_features.npz");ft=features(base,p,device,normal["mean"],normal["sd"]);packed=dict(num=ft["num"],p0=ft["p0"],ids=p["ids"])
    for arm in c["arms"]:
     rec=next(q for q in lockdata["domains"][domain]["records"] if q["arm"]==arm and q["seed"]==seed)
     if stage=="development_validation":pred=np.load(r/domain/f"seed{seed}"/(arm+"_validation.npy"))
     elif arm=="NoCorrection":pred=ft["p0"]
     else:
      ensemble=arm=="IndependentEnsemble";m=CalendarTimeBase(c,seed+1000,np.zeros((168,168),dtype="float32")).to(device) if ensemble else ResidualReadout(seed,tables[arm]).to(device)
      m.load_state_dict(torch.load(rec["path"],map_location=device,weights_only=True));pred=(predict(m,vp)+ft["p0"])/2 if ensemble else head_predict(m,packed,device);del m
     assert np.isfinite(pred).all();preds[f"{arm}_s{seed}"]=pred
     e=pred-p["y"];rows.append(dict(arm=arm,seed=seed,mse=float(np.mean(e**2)),mae=float(np.mean(abs(e))),raw_mse=float(np.mean(e**2))*float(b["sd"][0])**2,raw_mae=float(np.mean(abs(e)))*float(b["sd"][0]),n_origins=len(p["origin"])))
    del base
   csvsave(out/"scores.csv",rows);np.savez_compressed(out/"predictions.npz",**preds,truth=p["y"],origin=p["origin"])
   save(out/"completion.json",dict(commit=git(),domain=domain,stage=stage,counts=counts,lo=lo,hi=hi,fresh=stage=="new_confirmation",models_unchanged=all(sha(rec["path"])==rec["sha256"] for rec in lockdata["domains"][domain]["records"])))
   print("EVALUATED",domain,stage,json.dumps(rows),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=["prepare","fit","lock","evaluate"]);p.add_argument("--domain",choices=["capital","power"]);p.add_argument("--seed",type=int);p.add_argument("--device",default="cuda:0");args=p.parse_args();c=read("docs/chronogeom/residual03_config.json");r=Path(c["run"])
 try:
  if args.stage=="prepare":prepare(c,r)
  elif args.stage=="fit":fit(c,r,args.domain,args.seed,args.device)
  elif args.stage=="lock":lock(c,r)
  else:evaluate(c,r,args.device)
 except BaseException as e:
  save(r/f"failure_{args.stage}_{args.domain}_{args.seed}.json",dict(error=str(e),traceback=traceback.format_exc()));raise
if __name__=="__main__":main()
