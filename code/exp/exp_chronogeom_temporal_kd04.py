"""Standard feature/output KD into unchanged TimeBase Calendar Adapter. No LLM calls."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,hashlib,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from exp.exp_chronogeom_downstream import CalendarTimeBase,predict
from exp.exp_chronogeom_residual03 import ResidualReadout,frozen_base,features
from exp.exp_chronogeom_task_context02 import grid
from exp.exp_chronogeom_qwen_capital import windows,setup,tensorpack
from scripts.chronogeom.task_context02_analysis import save,read,csvsave
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git():return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
def ix(c,d,t):return int((np.datetime64(t,"h")-np.datetime64(c["domains"][d]["base"],"h"))/np.timedelta64(1,"h"))
def pack(v,ids,lo,hi,stride=1):
 oo=np.arange(lo,hi-96+1,stride);good=np.array([np.isfinite(v[o-96:o+96]).all() for o in oo]);keep=oo[good]
 assert len(keep)>5
 return dict(x=np.stack([v[o-96:o] for o in keep]),y=np.stack([v[o:o+96] for o in keep]),ids=np.stack([ids[o:o+96] for o in keep]),origin=keep),dict(candidate=len(oo),retained=len(keep),excluded_missing=int((~good).sum()))
def packed(b,origins):
 v,ids=b["values"],b["ids"]
 return dict(x=np.stack([v[o-96:o] for o in origins]),y=np.stack([v[o:o+96] for o in origins]),ids=np.stack([ids[o:o+96] for o in origins]),origin=origins)
def center_rms(z,w):
 mean=(w[:,None]*z).sum(0,keepdims=True);a=z-mean;s=np.sqrt(np.sum(w[:,None]*a*a)/a.shape[-1]);assert s>1e-8
 return ((a/max(s,1e-8)).astype("float32"),float(s))
def student_codes(m,w):
 z=m.codes();z=z-(w[:,None]*z).sum(0,keepdim=True)
 return z/torch.sqrt((w[:,None]*z.square()).sum()/z.shape[-1]).clamp_min(1e-6)
def prepare(c,r):
 r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c)
 tr=Path(c["teacher_run"]);tc=read(tr/"config.json");oldcfg=read(Path(tc["source_run"])/"config.json");lock=read(tr/"lock.json")
 used=[]
 for d,dc in c["domains"].items():
  dr=r/d;dr.mkdir();old=np.load(tr/d/"known_data.npz");v,ids,dates=grid(oldcfg,d,dc["validation"][1]);v=(v-old["mean"])/old["sd"]
  p,traincount=pack(v,ids,ix(c,d,dc["train"][0]),ix(c,d,dc["train"][1]));va,vcount=windows(v,ids,ix(c,d,dc["validation"][0]),ix(c,d,dc["validation"][1]))
  assert p["origin"].max()+96<=ix(c,d,dc["validation"][0])
  freq=np.bincount(p["ids"].ravel(),minlength=168).astype(float);assert np.all(freq>0);w=freq/freq.sum()
  source=np.load(tr/d/"tables.npz");stable,_=center_rms(source["Calendar32"],w)
  np.savez_compressed(dr/"known_data.npz",values=v,ids=ids,mean=old["mean"],sd=old["sd"],train_origins=p["origin"],state_weights=w.astype("float32"),student_table=stable)
  recs=[q for q in lock["domains"][d]["records"] if q["arm"] in [a for a in c["arms"].values() if a]]
  assert len(recs)==15
  bases=read(tr/d/"prepared.json")["backbones"]
  for q in recs+bases:
   assert sha(q["path"])==q["sha256"];used.append(dict(domain=d,arm=q["arm"],seed=q["seed"],path=q["path"],sha256=q["sha256"]))
  sv=np.linalg.svd(stable.astype(float),compute_uv=False);save(dr/"prepared.json",dict(train=traincount,validation=vcount,student_train_start=dc["train"][0],student_train_end=dc["train"][1],student_table_rank=int((sv>sv[0]*1e-6).sum()),student_table_train_rms=float(np.sqrt(np.sum(w[:,None]*stable*stable)/168)),student_table_sha=sha(dr/"known_data.npz"),teacher_records=recs,teacher_base_records=bases,teacher_data_seen_through=tc["domains"][d]["residual_validation"][1],student_source_is_consumed_development=True,new_confirmation_numerically_unread=True))
  print("PREPARED",d,traincount,vcount,flush=True)
 save(r/"source.json",dict(commit=git(),config_sha=sha("docs/chronogeom/temporal_kd04_config.json"),source_sha=sha(__file__),timebase_sha=sha("layers/ChronoGeomTimeBaseOfficial.py"),teacher_sources=used,tslib_upstream="4e938a1767106324dd753b2a44832bf870a0252e"))
def teacher_cache(c,r,d,seed,base,p,device,mu,sd):
 tr=Path(c["teacher_run"]);tb=np.load(tr/d/"tables.npz");records=read(r/d/"prepared.json")["teacher_records"];ft=features(base,p,device,mu,sd);cache={};protos={};stats=[]
 for arm,source in c["arms"].items():
  if source is None:continue
  rec=next(x for x in records if x["seed"]==seed and x["arm"]==source);assert sha(rec["path"])==rec["sha256"]
  m=ResidualReadout(seed,tb[source]).to(device);m.load_state_dict(torch.load(rec["path"],map_location=device,weights_only=True));m.eval()
  for q in m.parameters():q.requires_grad_(False)
  predictions=[];summed=np.zeros((168,32));counts=np.zeros(168)
  with torch.inference_mode():
   for j in range(0,len(p["x"]),128):
    num=torch.tensor(ft["num"][j:j+128],device=device);ids=torch.tensor(p["ids"][j:j+128],device=device);lead=m.lead_onehot[None].expand(len(num),-1,-1)
    h=m.mlp[1](m.mlp[0](torch.cat([num,lead,m.table[ids]],-1)));pred=m.mlp[-1](h)+m.numeric(num)+m.lead_bias[None,:,None]+torch.tensor(ft["p0"][j:j+128],device=device)
    predictions.append(pred.cpu().numpy());ii=ids.cpu().numpy().ravel();hh=h.cpu().numpy().reshape(-1,32);np.add.at(summed,ii,hh);counts+=np.bincount(ii,minlength=168)
  assert np.all(counts>0);proto=summed/counts[:,None];w=counts/counts.sum();proto,energy=center_rms(proto,w)
  predictions=np.concatenate(predictions);cache[arm]=predictions;protos[arm]=proto
  sv=np.linalg.svd(proto.astype(float),compute_uv=False);stats.append(dict(arm=arm,source_arm=source,prototype_rank=int((sv>sv[0]*1e-6).sum()),raw_prototype_centered_rms=energy,normalized_prototype_rms=float(np.sqrt(np.sum(w[:,None]*proto*proto)/32)),train_teacher_mse=float(np.mean((predictions-p["y"])**2)),teacher_sha=rec["sha256"]))
  assert all(q.grad is None for q in m.parameters());del m
 return cache,protos,stats,ft["p0"]
def load_teacher(c,r,d,seed,arm,device):
 src=Path(c["teacher_run"]);table=np.load(src/d/"tables.npz")[c["arms"][arm]];rec=next(x for x in read(r/d/"prepared.json")["teacher_records"] if x["seed"]==seed and x["arm"]==c["arms"][arm])
 assert sha(rec["path"])==rec["sha256"];m=ResidualReadout(seed,table).to(device);m.load_state_dict(torch.load(rec["path"],map_location=device,weights_only=True));m.eval()
 for p in m.parameters():p.requires_grad_(False)
 return m
def teacher_predict(m,ft,p,device):
 out=[]
 with torch.inference_mode():
  for j in range(0,len(p["x"]),128):
   n=torch.tensor(ft["num"][j:j+128],device=device);ii=torch.tensor(p["ids"][j:j+128],device=device)
   out.append((m(n,ii)+torch.tensor(ft["p0"][j:j+128],device=device)).cpu().numpy())
 return np.concatenate(out)
def gradnorm(loss,params):
 gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);v=sum(g.square().sum() for g in gs if g is not None)
 return float(torch.sqrt(v)) if not isinstance(v,int) else 0.
def fit(c,r,d,seed,device):
 setup(device,c["gpu_cap_gib"]);start=time.monotonic();out=r/d/f"seed{seed}";out.mkdir(exist_ok=False);b=np.load(r/d/"known_data.npz");dc=c["domains"][d];p=packed(b,b["train_origins"]);va,_=windows(b["values"],b["ids"],ix(c,d,dc["validation"][0]),ix(c,d,dc["validation"][1]))
 tr=Path(c["teacher_run"]);tc=read(tr/"config.json");base,brec=frozen_base(tc,tr,d,seed,device);normal=np.load(tr/d/f"seed{seed}/numeric_features.npz")
 cache,protos,diag,p0=teacher_cache(c,r,d,seed,base,p,device,normal["mean"],normal["sd"]);csvsave(out/"teacher_prototypes.csv",diag);np.savez_compressed(out/"teacher_training_cache.npz",**cache,**{a+"_prototype":z for a,z in protos.items()})
 tx,ty,ti=[torch.tensor(p[k],device=device) for k in ["x","y","ids"]];vp=tensorpack(va,device);w=torch.tensor(b["state_weights"],device=device);records=[];initials=[]
 for arm in c["arms"]:
  if time.monotonic()-start>c["per_seed_cap_seconds"]:raise TimeoutError("fit budget")
  m=CalendarTimeBase(c,seed,b["student_table"]).to(device);m.body.load_state_dict(copy.deepcopy(base.body.state_dict()));assert sum(q.numel() for q in m.parameters())==6570
  target=torch.tensor(cache[arm],device=device) if arm!="NoKD" else None;proto=torch.tensor(protos[arm],device=device) if arm!="NoKD" else None
  opt=torch.optim.Adam(m.parameters(),lr=c["lr"]);rng=np.random.default_rng(seed+271);curve=[];best=float("inf");tick=time.monotonic();lamf=lamy=0.;gradstats=None
  for step in range(c["steps"]+1):
   if time.monotonic()-start>c["per_seed_cap_seconds"]:raise TimeoutError("fit budget")
   if step%c["validation_every"]==0:
    pp=predict(m,vp);vm=float(np.mean((pp-va["y"])**2));assert np.isfinite(vm)
    curve.append(dict(step=step,mse=vm,mae=float(np.mean(abs(pp-va["y"])))))
    if vm<best:best=vm;beststep=step;state=copy.deepcopy(m.state_dict())
   if step==c["steps"]:break
   ii=torch.tensor(rng.integers(len(p["x"]),size=c["batch_size"]),device=device);m.train();yp=m(tx[ii],ti[ii]);sup=F.mse_loss(yp,ty[ii]);lf=lo=torch.zeros((),device=device)
   if arm!="NoKD":
    lf=F.smooth_l1_loss(student_codes(m,w)[ti[ii]],proto[ti[ii]]);lo=F.mse_loss(yp,target[ii])
    if step==0:
     params=list(m.parameters());ng=gradnorm(sup,params);nf=gradnorm(lf,params);no=gradnorm(lo,params)
     lamf=c["feature_initial_gradient_ratio"]*ng/nf if nf>1e-10 else 0.;lamy=c["output_initial_gradient_ratio"]*ng/no if no>1e-10 else 0.
     gradstats=dict(supervised=ng,feature=nf,output=no,lambda_feature=lamf,lambda_output=lamy,feature_disabled=nf<=1e-10,output_disabled=no<=1e-10)
   loss=sup+lamf*lf+lamy*lo;assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
  m.load_state_dict(state);path=out/(arm+".pt");torch.save(state,path);csvsave(out/(arm+"_curve.csv"),curve);pred=predict(m,vp);np.save(out/(arm+"_validation.npy"),pred)
  initial=CalendarTimeBase(c,seed,b["student_table"]).to(device);initial.body.load_state_dict(base.body.state_dict());initials.append(predict(initial,vp));del initial
  with torch.inference_mode():
   for _ in range(5):m(vp["x"][:1],vp["ids"][:1])
   torch.cuda.synchronize(device);t=time.monotonic()
   for _ in range(50):m(vp["x"][:1],vp["ids"][:1])
   torch.cuda.synchronize(device);ms=(time.monotonic()-t)*20
  rec=dict(arm=arm,seed=seed,parameters=6570,best_step=beststep,val_mse=best,seconds=time.monotonic()-tick,path=str(path),sha256=sha(path),gradient_calibration=gradstats,inference_batch1_ms=ms,updates=600,teacher_calls_at_inference=0)
  records.append(rec);save(out/"records.json",records);print("FIT",d,json.dumps(rec),flush=True);del m,opt
 assert all(np.array_equal(initials[0],z) for z in initials)
 assert all(not q.requires_grad and q.grad is None for q in base.parameters());assert sha(brec["path"])==brec["sha256"]
 save(out/"completion.json",dict(commit=git(),seed=seed,domain=d,seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,student_initial_predictions_identical=True,all6_students600updates=True,teacher_base_unchanged=True,teacher_prototype_training_inputs_only=True))
def lock(c,r):
 dd={};tt=0
 for d in c["domains"]:
  models=[]
  for seed in c["seeds"]:models+=read(r/d/f"seed{seed}/records.json");tt+=read(r/d/f"seed{seed}/completion.json")["seconds"]
  assert len(models)==18
  for q in models:assert sha(q["path"])==q["sha256"]
  means={a:float(np.mean([q["val_mse"] for q in models if q["arm"]==a])) for a in c["arms"]}
  strong=min(["FourierKD","CalendarKD"],key=means.get);dd[d]=dict(models=models,validation_means=means,strong_analytic=strong)
 assert tt<c["total_worker_cap_seconds"];save(r/"lock.json",dict(commit=git(),domains=dd,worker_seconds=tt,new_power_confirmation_unread=True));print("LOCK",json.dumps({d:v["validation_means"] for d,v in dd.items()}),flush=True)
def evaluate(c,r,device):
 setup(device,c["gpu_cap_gib"]);locked=read(r/"lock.json");tr=Path(c["teacher_run"]);tc=read(tr/"config.json");oldcfg=read(Path(tc["source_run"])/"config.json")
 for d,dc in c["domains"].items():
  b=np.load(r/d/"known_data.npz")
  for stage in ["development_validation"]+(["new_confirmation"] if dc["test"] else []):
   out=r/d/stage;out.mkdir(exist_ok=False)
   for m in locked["domains"][d]["models"]:assert sha(m["path"])==m["sha256"]
   save(out/"before_targets.json",dict(models_locked=True,new_confirmation=stage=="new_confirmation",commit=git()))
   if stage=="new_confirmation":
    raw,ids,_=grid(oldcfg,d,dc["test"][1]);values=(raw-b["mean"])/b["sd"];a,z=[ix(c,d,t) for t in dc["test"]]
   else:values=b["values"];ids=b["ids"];a,z=[ix(c,d,t) for t in dc["validation"]]
   p,counts=windows(values,ids,a,z);vp=tensorpack(p,device);preds={};rows=[]
   for seed in c["seeds"]:
    base,_=frozen_base(tc,tr,d,seed,device);normal=np.load(tr/d/f"seed{seed}/numeric_features.npz");ft=features(base,p,device,normal["mean"],normal["sd"]);preds[f"FrozenBase_s{seed}"]=ft["p0"]
    for arm in c["arms"]:
     rec=next(q for q in locked["domains"][d]["models"] if q["seed"]==seed and q["arm"]==arm);m=CalendarTimeBase(c,seed,b["student_table"]).to(device);m.load_state_dict(torch.load(rec["path"],map_location=device,weights_only=True))
     pred=predict(m,vp) if stage=="new_confirmation" else np.load(r/d/f"seed{seed}"/(arm+"_validation.npy"));preds[f"{arm}_s{seed}"]=pred
     # A time-input replacement check shows whether the final student uses calendar states.
     if arm in ["TaskKD","NoKD"]:
      perm=np.random.default_rng(20241028).permutation(168)
      shuffled=predict(m,dict(x=vp["x"],ids=torch.tensor(perm,device=device)[vp["ids"]]))
      rows.append(dict(arm=arm,seed=seed,calendar_identity_shuffle_prediction_rms=float(np.sqrt(np.mean((shuffled-pred)**2)))))
     del m
     if arm!="NoKD":
      teacher=load_teacher(c,r,d,seed,arm,device);preds[f"Teacher_{arm}_s{seed}"]=teacher_predict(teacher,ft,p,device);del teacher
    del base
   assert all(np.isfinite(v).all() and v.shape==p["y"].shape for v in preds.values())
   np.savez_compressed(out/"predictions.npz",truth=p["y"],origin=p["origin"],**preds);csvsave(out/"calendar_sensitivity.csv",rows)
   sc=[]
   for arm in ["FrozenBase"]+list(c["arms"])+["Teacher_"+a for a in c["arms"] if a!="NoKD"]:
    ee=np.stack([preds[f"{arm}_s{s}"].astype(float)-p["y"] for s in c["seeds"]]);ms=(ee*ee).mean((1,2,3));ma=abs(ee).mean((1,2,3))
    sc.append(dict(arm=arm,mse=float(ms.mean()),mae=float(ma.mean()),seed2024_mse=float(ms[0]),seed2025_mse=float(ms[1]),seed2026_mse=float(ms[2]),seed2024_mae=float(ma[0]),seed2025_mae=float(ma[1]),seed2026_mae=float(ma[2]),raw_mse=float(ms.mean())*float(b["sd"][0])**2,raw_mae=float(ma.mean())*float(b["sd"][0]),n_origins=len(p["origin"])))
   csvsave(out/"scores.csv",sc);save(out/"completion.json",dict(commit=git(),domain=d,stage=stage,counts=counts,lo=a,hi=z));print("EVALUATED",d,stage,json.dumps(sc),flush=True)
def summarize(c,r):
 out=r/"summary";out.mkdir(exist_ok=False);allsc=[];cc=[];originrows=[];geometry=[];locked=read(r/"lock.json")
 for d,dc in c["domains"].items():
  for stage in ["development_validation"]+(["new_confirmation"] if dc["test"] else []):
   path=r/d/stage;pp=np.load(path/"predictions.npz");truth=pp["truth"];orig=pp["origin"];co=read(path/"completion.json");lo,hi=co["lo"],co["hi"];days=np.arange(lo,hi-96+1,24);positions=(orig-lo)//24
   for row in list(__import__("csv").DictReader((path/"scores.csv").open())):allsc.append(dict(domain=d,stage=stage,**row))
   err={a:np.stack([((pp[f"{a}_s{s}"].astype(float)-truth)**2).mean((1,2)) for s in c["seeds"]]) for a in c["arms"]}
   ae={a:np.stack([abs(pp[f"{a}_s{s}"].astype(float)-truth).mean((1,2)) for s in c["seeds"]]) for a in c["arms"]}
   rng=np.random.default_rng(c["bootstrap_seed"]);n=c["bootstrap_replicates"];si=rng.integers(3,size=(n,3));st=rng.integers(len(days),size=(n,int(np.ceil(len(days)/c["block_days"]))));ti=((st[:,:,None]+np.arange(c["block_days"]))%len(days)).reshape(n,-1)[:,:len(days)]
   for other in [a for a in c["arms"] if a!="TaskKD"]:
    diff=err["TaskKD"]-err[other];ad=ae["TaskKD"]-ae[other];full=np.full((3,len(days)),np.nan);full[:,positions]=diff;draw=np.nanmean(full[si[:,:,None],ti[:,None,:]],axis=(1,2));ci=np.nanquantile(draw,[.025,.975]);full[:,positions]=ad;ac=np.nanquantile(np.nanmean(full[si[:,:,None],ti[:,None,:]],axis=(1,2)),[.025,.975])
    cc.append(dict(domain=d,stage=stage,comparison="TaskKD-"+other,mse_difference=float(diff.mean()),mse_lo95=float(ci[0]),mse_hi95=float(ci[1]),mae_difference=float(ad.mean()),mae_lo95=float(ac[0]),mae_hi95=float(ac[1]),improvement_percent=float(-100*diff.mean()/err[other].mean()),seed2024_difference=float(diff[0].mean()),seed2025_difference=float(diff[1].mean()),seed2026_difference=float(diff[2].mean()),all3_better=bool((diff.mean(1)<0).all()),n_origins=len(orig)))
    for j,o in enumerate(orig):originrows.append(dict(domain=d,stage=stage,comparison="TaskKD-"+other,origin=int(o),date=str(np.datetime64(dc["base"],"h")+np.timedelta64(int(o),"h")),mse_difference=float(diff[:,j].mean())))
   b=np.load(r/d/"known_data.npz");w=torch.tensor(b["state_weights"]);protoarm="TaskKD"
   for seed in c["seeds"]:
    cached=np.load(r/d/f"seed{seed}/teacher_training_cache.npz");target=cached[protoarm+"_prototype"];target=target/np.linalg.norm(target,axis=1,keepdims=True).clip(1e-8);tg=target@target.T
    for arm in c["arms"]:
     rec=next(q for q in locked["domains"][d]["models"] if q["seed"]==seed and q["arm"]==arm);m=CalendarTimeBase(c,seed,b["student_table"]);m.load_state_dict(torch.load(rec["path"],map_location="cpu",weights_only=True))
     with torch.no_grad():z=student_codes(m,w).numpy()
     z=z/np.linalg.norm(z,axis=1,keepdims=True).clip(1e-8);gg=z@z.T;tri=np.triu_indices(168,1);rho=np.corrcoef(gg[tri],tg[tri])[0,1];geometry.append(dict(domain=d,seed=seed,arm=arm,task_teacher_pairwise_similarity_correlation=float(rho),task_teacher_gram_mse=float(np.mean((gg-tg)**2))))
   # Geometry is one fixed model measurement, not new samples for each evaluation interval.
   break_geometry=False
 csvsave(out/"scores.csv",allsc);csvsave(out/"contrasts.csv",cc);csvsave(out/"per_origin_differences.csv",originrows)
 # Same model geometry was collected once per stage; deduplicate repeated rows.
 geometry=list({(q["domain"],q["seed"],q["arm"]):q for q in geometry}.values());csvsave(out/"representation_transfer.csv",geometry)
 print("SUMMARY",json.dumps(allsc),flush=True);print("CONTRASTS",json.dumps(cc),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["prepare","fit","lock","evaluate","summarize"]);ap.add_argument("--domain",choices=["capital","power"]);ap.add_argument("--seed",type=int);ap.add_argument("--device",default="cuda:0");args=ap.parse_args();c=read("docs/chronogeom/temporal_kd04_config.json");r=Path(c["run"])
 try:
  if args.stage=="prepare":prepare(c,r)
  elif args.stage=="fit":fit(c,r,args.domain,args.seed,args.device)
  elif args.stage=="lock":lock(c,r)
  elif args.stage=="evaluate":evaluate(c,r,args.device)
  else:summarize(c,r)
 except BaseException as e:
  r.mkdir(parents=True,exist_ok=True);save(r/f"failure_{args.stage}_{args.domain}_{args.seed}.json",dict(error=str(e),traceback=traceback.format_exc()));raise
if __name__=="__main__":main()
