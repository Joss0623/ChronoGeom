"""Descriptive Capital reversal analysis and fixed temporal-kernel diagnostics. No training."""
import json,csv
from pathlib import Path
import numpy as np
import pandas as pd
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def read(p):return json.loads(Path(p).read_text())
def csvsave(p,rows):
 with Path(p).open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def capital(c,r):
 old=Path(c["old_run"]);out=r/"capital_exploratory";out.mkdir(parents=True,exist_ok=False)
 b=np.load(old/"train_val.npz");ix=lambda x:int((np.datetime64(x,"h")-np.datetime64("2011-01-01","h"))/np.timedelta64(1,"h"))
 train=b["values"][ix(c["domains"]["capital"]["train_start"]):ix(c["domains"]["capital"]["train_end"]),0];edges=np.quantile(train,[1/3,2/3,.9]);groups=[];concentration=[];complement=[];origins_all=[]
 for stage in ["validation","pilot","confirmation"]:
  if stage=="validation":
   from exp.exp_chronogeom_qwen_capital import windows
   p,_=windows(b["values"],b["ids"],ix(c["domains"]["capital"]["train_end"]),ix(c["domains"]["capital"]["val_end"]));truth=p["y"];origin=p["origin"]
   preds={n:np.load(old/"fit"/(n+"_val.npy")) for n in ["QwenTask32","QwenGeneric32","Fourier32","NoCalendar"]}
  else:
   p=np.load(old/stage/"predictions.npz");truth=p["truth"];origin=p["origin"];preds={n:p[n] for n in ["QwenTask32","QwenGeneric32","Fourier32","NoCalendar"]}
  targethours=origin[:,None]+np.arange(96)[None,:];times=pd.DatetimeIndex((np.datetime64("2011-01-01T00","h")+targethours).ravel());shape=targethours.shape
  hour=times.hour.to_numpy().reshape(shape);weekday=times.dayofweek.to_numpy().reshape(shape);month=np.array(times.strftime("%Y-%m")).reshape(shape);seasons=np.array(["winter","winter","spring","spring","spring","summer","summer","summer","autumn","autumn","autumn","winter"])[times.month.to_numpy()-1].reshape(shape)
  y=truth[:,:,0];q=preds["QwenTask32"][:,:,0];qe=q-y
  masks=[("all","all",np.ones(shape,bool))]
  for kind,values in [("hour",hour),("weekday",weekday),("weekpart",np.where(weekday<5,"weekday","weekend")),("month",month),("season",seasons),("demand_bin",np.array(["q0_33","q33_66","q66_90","q90_100"])[np.digitize(y,edges)])]:
   masks += [(kind,str(v),values==v) for v in np.unique(values)]
  masks += [("clock_band",label,np.isin(hour,hs)) for label,hs in [("night",range(0,6)),("morning",range(6,10)),("day",range(10,16)),("evening",range(16,20)),("late",range(20,24))]]
  for other in ["QwenGeneric32","Fourier32","NoCalendar"]:
   oe=preds[other][:,:,0]-y
   for kind,label,mask in masks:
    if not mask.any():continue
    groups.append(dict(stage=stage,comparison="QwenTask32-"+other,group_type=kind,group=label,n_forecast_cells=int(mask.sum()),n_unique_target_hours=int(len(np.unique(targethours[mask]))),n_origins=int(np.any(mask,axis=1).sum()),qwen_mse=float(np.mean(qe[mask]**2)),other_mse=float(np.mean(oe[mask]**2)),mse_difference=float(np.mean(qe[mask]**2-oe[mask]**2)),qwen_mae=float(np.mean(abs(qe[mask]))),other_mae=float(np.mean(abs(oe[mask]))),qwen_mean_signed_error=float(qe[mask].mean()),other_mean_signed_error=float(oe[mask].mean()),qwen_large_overprediction_rate=float(np.mean(qe[mask]>1)),other_large_overprediction_rate=float(np.mean(oe[mask]>1)),qwen_large_underprediction_rate=float(np.mean(qe[mask]<-1)),other_large_underprediction_rate=float(np.mean(oe[mask]<-1))))
   gain=(oe**2-qe**2).mean(1);order=np.argsort(-gain);gross=np.maximum(gain,0).sum();net=gain.sum()
   for k in [1,3,5]:
    keep=np.ones(len(gain),bool);keep[order[:k]]=False
    concentration.append(dict(stage=stage,other=other,k=k,n_origins=len(gain),n_qwen_better=int((gain>0).sum()),mean_qwen_minus_other=float(-gain.mean()),topk_positive_share=float(np.maximum(gain[order[:k]],0).sum()/gross) if gross>0 else 0.,topk_net_gain_share=float(gain[order[:k]].sum()/net) if abs(net)>1e-12 else 0.,leave_topk_out_qwen_minus_other=float(-gain[keep].mean()),topk_origins=origin[order[:k]].tolist()))
   for i,o in enumerate(origin):origins_all.append(dict(stage=stage,origin=int(o),date=str(np.datetime64("2011-01-01T00","h")+np.timedelta64(int(o),"h")),other=other,qwen_mse=float(np.mean(qe[i]**2)),other_mse=float(np.mean(oe[i]**2)),mse_gain=float(gain[i])))
  n=preds["NoCalendar"][:,:,0];ne=n-y;mix=(q+n)/2;errmix=mix-y;qwin=(qe**2).mean(1);nwin=(ne**2).mean(1)
  complement.append(dict(stage=stage,n_origins=len(origin),task_mse=float((qe**2).mean()),no_calendar_mse=float((ne**2).mean()),fixed_half_mse=float((errmix**2).mean()),task_mae=float(abs(qe).mean()),no_calendar_mae=float(abs(ne).mean()),fixed_half_mae=float(abs(errmix).mean()),residual_correlation=float(np.corrcoef(qe.ravel(),ne.ravel())[0,1]),task_better_origin_fraction=float(np.mean(qwin<nwin)),retrospective_origin_oracle_mse=float(np.minimum(qwin,nwin).mean()),oracle_is_not_deployable=True))
 csvsave(out/"group_errors.csv",groups);csvsave(out/"concentration.csv",concentration);csvsave(out/"complementarity.csv",complement);csvsave(out/"all_origin_contrasts.csv",origins_all)
 save(out/"definition.json",dict(demand_edges_train_scaled=edges.tolist(),demand_groups_are_postoutcome_not_forecast_inputs=True,intervals_are_already_consumed=True,no_model_choices_from_analysis=True,counts_include_overlap_and_unique_hours=True))
 print("CAPITAL_COMPLEMENT",json.dumps(complement),flush=True)
 print("CAPITAL_CONFIRM_CONCENTRATION",json.dumps([x for x in concentration if x["stage"]=="confirmation"]),flush=True)
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--config",default="docs/chronogeom/task_context02_config.json");args=p.parse_args();c=read(args.config);r=Path(c["run"]);r.mkdir(parents=True,exist_ok=True)
 if not (r/"config.json").exists():save(r/"config.json",c)
 capital(c,r)


def summarize(c,r):
 import hashlib
 from itertools import combinations
 out=r/"summary";out.mkdir(exist_ok=False);lock=read(r/"lock.json");tables=np.load(r/"tables.npz")
 native=np.load(r/"native_contexts.npz");names=["Generic32","Bicycle32","Electricity32"]
 def normalized(z):
  z=z-z.mean(0);return z/np.sqrt(np.mean(z*z))
 zn={n:normalized(native[n]) for n in names};cos={n:(z/np.linalg.norm(z,axis=1,keepdims=True))@(z/np.linalg.norm(z,axis=1,keepdims=True)).T for n,z in zn.items()}
 geometry=[]
 for a,b in combinations(names,2):
  x,y=zn[a],zn[b];u,sv,vt=np.linalg.svd(x.T@y,full_matrices=False);rotation=u@vt
  ga,gb=cos[a],cos[b];ea=np.linalg.eigvalsh(ga);eb=np.linalg.eigvalsh(gb);ea=ea/ea.sum();eb=eb/eb.sum()
  geometry.append(dict(a=a,b=b,centered_coordinate_nrmse=float(np.linalg.norm(x-y)/np.linalg.norm(y)),orthogonal_procrustes_nrmse=float(np.linalg.norm(x@rotation-y)/np.linalg.norm(y)),cosine_kernel_similarity=float(np.sum(ga*gb)/np.linalg.norm(ga)/np.linalg.norm(gb)),cosine_kernel_relative_difference=float(np.linalg.norm(ga-gb)/np.linalg.norm(gb)),trace_normalized_eigenspectrum_difference=float(np.linalg.norm(ea-eb)/np.linalg.norm(eb))))
 csvsave(out/"geometry.csv",geometry)
 d=np.repeat(np.arange(7),24);h=np.tile(np.arange(24),7);same=d[:,None]==d;hd=np.abs(h[:,None]-h);hd=np.minimum(hd,24-hd);nonself=~np.eye(168,dtype=bool)
 w=d<5;wk=d>=5;morn=np.isin(h,[7,8,9]);even=np.isin(h,[16,17,18]);night=h<6
 masks={"adjacent_hours_same_day":same&(hd==1),"far_hours_same_day":same&(hd>=8),"same_hour_weekday_pairs":(h[:,None]==h)&(w[:,None]&w)&nonself,"same_hour_weekday_weekend":(h[:,None]==h)&(w[:,None]!=w),"same_day_morning_evening":same&((morn[:,None]&even)|(even[:,None]&morn)),"same_day_evening_night":same&((even[:,None]&night)|(night[:,None]&even))}
 relations=[]
 for name in names:
  for label,mask in masks.items():relations.append(dict(context=name,relation=label,n_ordered_pairs=int(mask.sum()),mean_cosine=float(cos[name][mask].mean())))
 csvsave(out/"temporal_relations.csv",relations)
 f4=np.c_[np.cos(2*np.pi*h/24),np.sin(2*np.pi*h/24),np.cos(2*np.pi*d/7),np.sin(2*np.pi*d/7)];cols=[]
 for z,p,ks in [(h,24,range(1,12)),(d,7,range(1,4))]:
  for k in ks:cols.extend([np.cos(2*np.pi*k*z/p),np.sin(2*np.pi*k*z/p)])
 cols.extend(f4[:,i]*f4[:,j] for i in [0,1] for j in [2,3]);f=np.column_stack(cols);f-=f.mean(0)
 explanation=[]
 for name,z in zn.items():
  pred=f@np.linalg.lstsq(f,z,rcond=1e-10)[0];explanation.append(dict(context=name,fourier32_relative_residual_energy=float(np.sum((z-pred)**2)/np.sum(z*z)),joint168_residual="numericalzero_by_complete_basis"))
 csvsave(out/"analytical_explanation.csv",explanation)
 allscores=[];contrasts=[];statecorr=[];case_rows=[];all_errors={};gate_checks={}
 for domain,dc in c["domains"].items():
  strong=lock["domains"][domain]["strong"];correct=dc["correct"]
  for stage in ["validation","pilot","confirmation"]:
   root=r/domain/stage;z=np.load(root/"predictions.npz");completion=read(root/"completion.json");truth=z["truth"];origin=z["origin"];n=len(origin);errors={}
   for arm in c["arms"]:
    ee=np.stack([(z[f"{arm}_s{s}"]-truth)[:,:,0] for s in c["seeds"]]);errors[arm]=(ee**2).mean(2)
    mse=np.mean(ee**2,axis=(1,2));mae=np.mean(abs(ee),axis=(1,2))
    allscores.append(dict(domain=domain,stage=stage,arm=arm,mse=float(mse.mean()),mae=float(mae.mean()),mse_seed_sd=float(mse.std(ddof=1)),seed2024_mse=float(mse[0]),seed2025_mse=float(mse[1]),seed2026_mse=float(mse[2]),n_origins=n,n_seeds=3,parameters=40 if arm=="Original" else 6570))
   all_errors[(domain,stage)]=errors
   lo,hi=completion["lo"],completion["hi"];days=np.arange(lo,hi-96+1,24);daysix=(origin-lo)//24
   rng=np.random.default_rng(c["bootstrap_seed"]);reps=c["bootstrap_replicates"];seedix=rng.integers(3,size=(reps,3));starts=rng.integers(len(days),size=(reps,int(np.ceil(len(days)/14))));dayix=((starts[:,:,None]+np.arange(14))%len(days)).reshape(reps,-1)[:,:len(days)]
   for other in ["Generic32",dc["wrong"],strong,"NoCalendar",dc["shuffle"],"Original"]:
    diff=errors[correct]-errors[other];full=np.full((3,len(days)),np.nan);full[:,daysix]=diff;draw=full[seedix[:,:,None],dayix[:,None,:]];bs=np.nanmean(draw,axis=(1,2));interval=np.nanquantile(bs,[.025,.975]);seed_d=diff.mean(1)
    contrasts.append(dict(domain=domain,stage=stage,correct=correct,other=other,mse_difference=float(diff.mean()),improvement_percent=float(-100*diff.mean()/errors[other].mean()),lo95=float(interval[0]),hi95=float(interval[1]),seed2024_difference=float(seed_d[0]),seed2025_difference=float(seed_d[1]),seed2026_difference=float(seed_d[2]),all3_better=bool(np.all(seed_d<0)),n_origins=n,calendar_span_days=len(days),retained_days=n))
    if domain=="power" and stage=="confirmation" and other in ["Generic32",dc["wrong"],strong,"NoCalendar"]:gate_checks[other]=bool(np.all(seed_d<0) and interval[1]<0)
   # Descriptive correspondence, all periods retained. Calendar identities, never forecast selection.
   target_idx=origin[:,None]+np.arange(96);times=pd.DatetimeIndex((np.datetime64(dc["base"],"h")+target_idx).ravel());state=(times.dayofweek.to_numpy()*24+times.hour.to_numpy()).reshape(n,96)
   delta=np.sqrt(np.mean((cos[correct]-cos["Generic32"])**2,axis=1));cg=np.mean([(z[f"Generic32_s{s}"]-truth)[:,:,0]**2-(z[f"{correct}_s{s}"]-truth)[:,:,0]**2 for s in c["seeds"]],axis=0)
   gains=[];ds=[]
   for sid in range(168):
    mask=state==sid
    if mask.any():gains.append(float(cg[mask].mean()));ds.append(delta[sid])
   corr=float(np.corrcoef(pd.Series(gains).rank(),pd.Series(ds).rank())[0,1])
   statecorr.append(dict(domain=domain,stage=stage,correct=correct,calendar_states=len(gains),spearman_kernel_change_vs_mse_gain=corr,descriptive_not_independent_samples=True))
   for other in ["Generic32",dc["wrong"],strong,"NoCalendar"]:
    diff=(errors[correct]-errors[other]).mean(0)
    for i,o in enumerate(origin):
     case_rows.append(dict(domain=domain,stage=stage,origin=int(o),date=str(np.datetime64(dc["base"],"h")+np.timedelta64(int(o),"h")),other=other,correct_mse=float(errors[correct][:,i].mean()),other_mse=float(errors[other][:,i].mean()),mse_difference=float(diff[i])))
 csvsave(out/"scores.csv",allscores);csvsave(out/"contrasts.csv",contrasts);csvsave(out/"geometry_forecast_correspondence.csv",statecorr);csvsave(out/"all_origin_contrasts.csv",case_rows)
 capmatch=next(t for t in contrasts if t["domain"]=="capital" and t["stage"]=="confirmation" and t["other"]==c["domains"]["capital"]["wrong"])
 gate_checks["capital_matching_direction"]=capmatch["mse_difference"]<0
 save(out/"teacher_value_gate.json",dict(passed=all(gate_checks.values()),checks=gate_checks,kd_this_round=False,next_round_only=True,capital_confirmation_is_not_new=True))
 # Fair fixed50/50 diagnostic controls for the consumed seed2024 Capital result, never used in Power selection.
 old=Path(c["old_run"]);ensemble=[]
 for stage in ["validation","pilot","confirmation"]:
  if stage=="validation":
   b=np.load(old/"train_val.npz");from exp.exp_chronogeom_qwen_capital import windows,index
   p,_=windows(b["values"],b["ids"],index(c["domains"]["capital"]["train_end"]),index(c["domains"]["capital"]["val_end"]));truth=p["y"];pred={n:np.load(old/"fit"/(n+"_val.npy")) for n in ["QwenTask32","QwenGeneric32","Fourier32","NoCalendar"]}
  else:
   p=np.load(old/stage/"predictions.npz");truth=p["truth"];pred={n:p[n] for n in ["QwenTask32","QwenGeneric32","Fourier32","NoCalendar"]}
  for name in ["QwenTask32","QwenGeneric32","Fourier32"]:
   e=(pred[name]+pred["NoCalendar"])/2-truth;ensemble.append(dict(stage=stage,half_time_model=name,mse=float(np.mean(e**2)),mae=float(np.mean(abs(e)))))
 csvsave(out/"capital_fixed_ensemble_controls.csv",ensemble)
 print("MEAN_CONFIRMATION",json.dumps([x for x in allscores if x["stage"]=="confirmation"]),flush=True)
 print("TEACHER_VALUE",read(out/"teacher_value_gate.json"),flush=True)
 print("GEOMETRY",json.dumps(geometry),flush=True)
 return allscores,contrasts
