"""Existing dynamic teachers with ONE train-internal shared stopping rule."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,csv,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from exp import exp_chronogeom_dynamic_teacher01 as old
rt=old.rt
read,save,csvsave=old.read,old.save,old.csvsave

def daily_mse(pp,y,origin):
 e=((pp-y)**2).mean((1,2));return float(np.mean([e[origin//24==day].mean() for day in np.unique(origin//24)]))
def code_for(features,arm):
 if arm!="PlainFourier":return features[arm]
 f=features["StaticQwen"][:,:,:32];z=np.zeros_like(features["StaticQwen"]);z[:,:,:32]=f;z[:,:,32:64]=f;return z
def select_common(curves,c):
 best=float("inf");stepbest=0;stale=0;decision=[];stopped_at=c["max_steps"]
 for step in range(0,c["max_steps"]+1,c["check_every"]):
  qq=[q for q in curves if q["step"]==step];assert len(qq)==len(c["arms"])*len(c["seeds"])
  score=float(np.mean([q["relative_day_mse"] for q in qq]));improved=score<best-c["min_relative_improvement"]
  if improved:best=score;stepbest=step;stale=0
  elif step>0:stale+=1
  decision.append(dict(step=step,pooled_relative_day_mse=score,improved=bool(improved),stale=stale))
  if stale>=c["patience"]:stopped_at=step;break
 return dict(selected_step=stepbest,stopped_at=stopped_at,best_score=best,trace=decision,includes_zero=True,all_arms_use_same_step=True,outer_validation_used_for_selection=False)
def prepare(c,r):
 source=Path(c["source_run"]);p=np.load(source/"data.npz");n=int(p["n_train"]);o=p["origin"][:n]
 cut=rt.ix(c,c["inner_cut"]);fit=np.flatnonzero(o+96<=cut);selection=np.flatnonzero(o>=cut);purged=np.flatnonzero((o+96>cut)&(o<cut))
 assert len(fit)>=32 and len(selection)>=8 and np.intersect1d(fit,selection).size==0
 assert np.max(o[fit]+96)<=np.min(o[selection])
 # With this fixed split, innerfit remains a prefix, allowing old feature builder without changes.
 assert np.array_equal(fit,np.arange(len(fit)))
 inner=r/"inner_preprocessing";inner.mkdir()
 data={k:p[k] for k in p.files};data["n_train"]=np.array(len(fit));data["summary_mean"]=p["statistics"][fit].mean(0);data["summary_sd"]=np.maximum(p["statistics"][fit].std(0),1e-3)
 np.savez_compressed(inner/"data.npz",**data)
 src=read(source/"source.json");save(inner/"source.json",src)
 (inner/"dynamic_raw_layer24.npy").symlink_to(source/"dynamic_raw_layer24.npy")
 oc=read(c["source_config"]);old.makefeatures(oc,inner)
 split=dict(fit_indices=fit.tolist(),selection_indices=selection.tolist(),purged_indices=purged.tolist(),n_fit=len(fit),n_selection=len(selection),n_purged=len(purged),fit_days=len(np.unique(o[fit]//24)),selection_days=len(np.unique(o[selection]//24)),cut=c["inner_cut"],max_fit_target_end=int(np.max(o[fit]+96)),min_selection_origin=int(np.min(o[selection])),selection_rule_frozen_before_evaluation=True)
 save(r/"split.json",split)
 assets=[source/"data.npz",source/"features.npz",source/"dynamic_raw_layer24.npy",Path(src["static_path"])]
 assets += [Path(q["path"]) for q in read(source/"readout_completion.json")["base_sources"]]
 assets += [source/f"numeric_seed{s}.npz" for s in c["seeds"]]
 save(r/"source.json",dict(git=rt.git(),source_sha=rt.sha(__file__),configuration_sha=rt.sha("docs/chronogeom/dynamic_earlystop02_config.json"),assets={str(f):rt.sha(f) for f in assets},timebase_source_sha=rt.sha("layers/ChronoGeomTimeBaseOfficial.py"),zero_new_llm_calls=True,all_outer_data_previously_viewed=True))
 print("PREPARED",json.dumps(split),flush=True)
def numeric(p,p0,indices,device="cpu"):
 with torch.no_grad():a=old.numeric_summary(torch.tensor(p["x"],device=device),torch.tensor(p0,dtype=torch.float32,device=device)).cpu().numpy()
 mu=a[indices].mean((0,1));sd=np.maximum(a[indices].std((0,1)),1e-3)
 return ((a-mu)/sd).astype("float32"),mu,sd
def head(seed,device):return old.ResidualReadout(seed,np.zeros((168,168),dtype="float32")).to(device)
def update(h,opt,nt,ct,pt,yt,ii):
 h.train();yp=old.forward(h,nt[ii],ct[ii],pt[ii]);loss=F.mse_loss(yp,yt[ii]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.,error_if_nonfinite=True);opt.step()
def run(c,device):
 r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c);tick=time.monotonic();rt.setup(device,c["gpu_cap_gib"]);prepare(c,r)
 src=Path(c["source_run"]);p=np.load(src/"data.npz");n=int(p["n_train"]);split=read(r/"split.json");fit=np.array(split["fit_indices"]);sel=np.array(split["selection_indices"]);outer=np.arange(n,len(p["x"]));innerfeatures=np.load(r/"inner_preprocessing/features.npz")
 allcurves=[];innerrecords=[];fit_seconds=0.
 for seed in c["seeds"]:
  p0=np.load(src/f"numeric_seed{seed}.npz")["p0"];num,mu,sd=numeric(p,p0,fit,device)
  np.savez_compressed(r/f"inner_numeric_seed{seed}.npz",mu=mu,sd=sd)
  baseline=daily_mse(p0[sel],p["y"][sel],p["origin"][sel]);assert baseline>1e-8
  for arm in c["arms"]:
   start=time.monotonic();h=head(seed,device);code=code_for(innerfeatures,arm);rng=np.random.default_rng(seed+271);opt=torch.optim.Adam(h.parameters(),lr=c["lr"]);states={}
   nt=torch.tensor(num[fit],device=device);ct=torch.tensor(code[fit],device=device);pt=torch.tensor(p0[fit],device=device);yt=torch.tensor(p["y"][fit],dtype=torch.float64,device=device)
   for step in range(c["max_steps"]+1):
    if time.monotonic()-tick>c["workflow_cap_seconds"]:raise TimeoutError("workflowcap")
    if step%c["check_every"]==0:
     pp=old.prediction(h,num[sel],code[sel],p0[sel],device);day=daily_mse(pp,p["y"][sel],p["origin"][sel]);m,l,s,ma=rt.components(pp,p["y"][sel])
     if step==0:assert np.array_equal(pp,p0[sel])
     allcurves.append(dict(arm=arm,seed=seed,step=step,inner_day_mse=day,relative_day_mse=day/baseline,window_mse=float(m.mean()),mae=float(ma.mean()),level=float(l.mean()),shape=float(s.mean())))
     states[step]={k:v.detach().cpu().clone() for k,v in h.state_dict().items()}
    if step==c["max_steps"]:break
    ii=torch.tensor(rng.integers(len(fit),size=c["batch_size"]),device=device);update(h,opt,nt,ct,pt,yt,ii)
   path=r/f"inner_{arm}_s{seed}_states.pt";torch.save(states,path);seconds=time.monotonic()-start;fit_seconds+=seconds
   innerrecords.append(dict(arm=arm,seed=seed,path=str(path),sha256=rt.sha(path),updates=c["max_steps"],seconds=seconds));csvsave(r/"inner_curves.csv",allcurves);save(r/"inner_records.json",innerrecords)
   print("INNER_FIT",arm,seed,round(seconds,2),flush=True);del h,opt,nt,ct,pt,yt,states
 chosen=select_common(allcurves,c);save(r/"common_selection.json",dict(**chosen,git=rt.git(),split=split,outer_validation_accessed_for_selection=False));steps=chosen["selected_step"];decision_sha=rt.sha(r/"common_selection.json")
 print("COMMON_SELECTION",json.dumps(chosen),flush=True)
 # Refit on all128 training windows using a common step count and full-training-only preprocessing.
 ff=np.load(src/"features.npz");records=[];pred={"truth":p["y"][outer],"origin":p["origin"][outer]};scores=[];windowrows=[];replays=[]
 for seed in c["seeds"]:
  cache=np.load(src/f"numeric_seed{seed}.npz");p0=cache["p0"];num,mu,sd=numeric(p,p0,np.arange(n),device);assert np.array_equal(mu,cache["mu"]) and np.array_equal(sd,cache["sd"])
  np.savez_compressed(r/f"refit_numeric_seed{seed}.npz",mu=mu,sd=sd)
  pred[f"FrozenBase_s{seed}"]=p0[outer]
  for arm in c["arms"]:
   start=time.monotonic();h=head(seed,device);code=code_for(ff,arm);opt=torch.optim.Adam(h.parameters(),lr=c["lr"]);rng=np.random.default_rng(seed+271)
   nt=torch.tensor(num[:n],device=device);ct=torch.tensor(code[:n],device=device);pt=torch.tensor(p0[:n],device=device);yt=torch.tensor(p["y"][:n],dtype=torch.float64,device=device)
   init=old.prediction(h,num[outer],code[outer],p0[outer],device);assert np.array_equal(init,p0[outer])
   for step in range(steps):
    if time.monotonic()-tick>c["workflow_cap_seconds"]:raise TimeoutError("workflowcap")
    ii=torch.tensor(rng.integers(n,size=c["batch_size"]),device=device);update(h,opt,nt,ct,pt,yt,ii)
   path=r/f"refit_{arm}_s{seed}.pt";torch.save(h.state_dict(),path);pp=old.prediction(h,num[outer],code[outer],p0[outer],device);pred[f"{arm}_s{seed}"]=pp
   seconds=time.monotonic()-start;fit_seconds+=seconds;rec=dict(arm=arm,seed=seed,updates=steps,parameters=sum(z.numel() for z in h.parameters()),frozen_base_parameters=6570,path=str(path),sha256=rt.sha(path),initial_max_error=float(abs(init-p0[outer]).max()),seconds=seconds)
   records.append(rec);save(r/"refit_records.json",records);print("REFIT",json.dumps(rec),flush=True);del h,opt,nt,ct,pt,yt
 # All model weights and selection are locked before outer metrics.
 save(r/"refit_lock.json",dict(git=rt.git(),decision_sha=decision_sha,models=records,all_same_updates=steps,outer_checkpoint_selection=False))
 for seed in c["seeds"]:
  for arm in ["FrozenBase"]+c["arms"]:
   pp=pred[f"{arm}_s{seed}"];base=pred[f"FrozenBase_s{seed}"];meanerr=float(abs(pp.mean(1)-base.mean(1)).max());assert meanerr<1e-12
   m,l,s,ma=rt.components(pp,pred["truth"]);scores.append(dict(arm=arm,seed=seed,steps=0 if arm=="FrozenBase" else steps,mse=float(m.mean()),mae=float(ma.mean()),level=float(l.mean()),shape=float(s.mean()),max_mean_error=meanerr))
   for j,o in enumerate(pred["origin"]):windowrows.append(dict(arm=arm,seed=seed,origin=int(o),mse=float(m[j]),mae=float(ma[j]),level=float(l[j]),shape=float(s[j])))
 csvsave(r/"scores_by_seed.csv",scores);csvsave(r/"scores_by_window.csv",windowrows);np.savez_compressed(r/"predictions.npz",**pred)
 means=[]
 for arm in ["FrozenBase"]+c["arms"]:
  qq=[q for q in scores if q["arm"]==arm];out=dict(arm=arm,steps=0 if arm=="FrozenBase" else steps)
  out.update({k:float(np.mean([q[k] for q in qq])) for k in ["mse","mae","level","shape"]});out.update({f"seed{s}_mse":next(q["mse"] for q in qq if q["seed"]==s) for s in c["seeds"]});means.append(out)
 csvsave(r/"scores_mean.csv",means)
 # CPU replay all selected refits.
 for rec in records:
  seed=rec["seed"];cache=np.load(src/f"numeric_seed{seed}.npz");p0=cache["p0"];normal=np.load(r/f"refit_numeric_seed{seed}.npz")
  with torch.no_grad():num=old.numeric_summary(torch.tensor(p["x"]),torch.tensor(p0,dtype=torch.float32)).numpy()
  num=((num-normal["mu"])/normal["sd"]).astype("float32");code=code_for(ff,rec["arm"])
  h=head(seed,"cpu");h.load_state_dict(torch.load(rec["path"],map_location="cpu",weights_only=True));pp=old.prediction(h,num[outer],code[outer],p0[outer],"cpu");error=float(abs(pp-pred[f"{rec['arm']}_s{seed}"]).max());assert error<2e-5
  replays.append(dict(arm=rec["arm"],seed=seed,max_prediction_error=error))
 csvsave(r/"cpu_replays.csv",replays)
 # Paired7day block uncertainty,3seeds averaged before bootstrap.
 bc=c["bootstrap"];start=rt.ix(c,"2010-03-01");end=rt.ix(c,"2010-03-29");days=len(np.arange(start,end-96+1,24));b=bc["block_days"];rng=np.random.default_rng(bc["seed"])
 draws=(rng.integers(days,size=(bc["replicates"],int(np.ceil(days/b))))[:,:,None]+np.arange(b))%days;draws=draws.reshape(bc["replicates"],-1)[:,:days];cis=[]
 for other in ["FrozenBase","PlainFourier","Fourier","StaticQwen","HistoryShuffleQwen"]:
  diff=np.full(days,np.nan)
  for j,o in enumerate(pred["origin"]):
   vals=[float(np.mean((pred[f"DynamicQwen_s{s}"][j]-pred["truth"][j])**2)-np.mean((pred[f"{other}_s{s}"][j]-pred["truth"][j])**2)) for s in c["seeds"]];diff[(int(o)-start)//24]=np.mean(vals)
  v=diff[draws];den=np.isfinite(v).sum(1);boot=np.nansum(v,1)/np.maximum(den,1);lo,hi=np.quantile(boot[den>0],[.025,.975])
  cis.append(dict(comparison="DynamicQwen-"+other,mse_difference=float(np.nanmean(diff)),lo95=float(lo),hi95=float(hi),block_days=b,unique_origins=len(pred["origin"])))
 csvsave(r/"paired_timeblock_intervals.csv",cis)
 source=read(r/"source.json");assert all(rt.sha(path)==h for path,h in source["assets"].items());assert rt.sha("layers/ChronoGeomTimeBaseOfficial.py")==source["timebase_source_sha"];assert rt.sha(r/"common_selection.json")==decision_sha
 assert fit_seconds<c["fit_cap_seconds"] and all(q["parameters"]==6570 for q in records)
 a={q["arm"]:q for q in means};controls=["FrozenBase","PlainFourier","StaticQwen","HistoryShuffleQwen"];gate=all(a["DynamicQwen"]["mse"]<a[z]["mse"] and sum(a["DynamicQwen"][f"seed{s}_mse"]<a[z][f"seed{s}_mse"] for s in c["seeds"])>=2 for z in controls)
 save(r/"completion.json",dict(seconds=time.monotonic()-tick,fit_seconds=fit_seconds,selected_step=steps,new_inner_models=len(innerrecords),new_refit_models=len(records),actual_updates=len(innerrecords)*c["max_steps"]+len(records)*steps,new_llm_calls=0,new_distilled_students=0,old_assets_unchanged=True,selection_unchanged=True,max_mean_error=max(q["max_mean_error"] for q in scores),max_cpu_replay_error=max(q["max_prediction_error"] for q in replays),peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,teacher_value_gate=bool(gate)))
 print("RESULT",json.dumps(dict(means=means,completion=read(r/"completion.json"))),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda:0");args=ap.parse_args();c=read("docs/chronogeom/dynamic_earlystop02_config.json");r=Path(c["run"])
 try:run(c,args.device)
 except BaseException:
  if r.exists():save(r/"failure.json",dict(traceback=traceback.format_exc()))
  raise
if __name__=="__main__":main()
