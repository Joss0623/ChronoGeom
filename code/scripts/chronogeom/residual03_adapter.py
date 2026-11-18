"""Selected household Calendar Adapter measurements; no saved model updates."""
import copy,csv,json
from pathlib import Path
import numpy as np
import torch
from exp.exp_chronogeom_downstream import CalendarTimeBase
from exp.exp_chronogeom_qwen_capital import windows
from scripts.chronogeom.task_context02_analysis import read,save,csvsave
def normgrads(params):
 gs=[p.grad.square().sum() for p in params if p.grad is not None]
 return float(torch.sqrt(sum(gs))) if gs else 0.
def grads(m,x,ids,y):
 m.zero_grad(set_to_none=True);loss=((m(x,ids)-y)**2).mean();loss.backward()
 return dict(loss=float(loss),body=normgrads(m.body.parameters()),first_time_layer=normgrads(m.time_mlp[0].parameters()),last_time_layer=normgrads(m.time_mlp[2].parameters()),readout=normgrads(m.readout.parameters()))
def run(c,r):
 torch.set_num_threads(2);src=Path(c["source_run"]);cfg=read(src/"config.json");dc=cfg["domains"]["power"];models=read(src/"lock.json")["domains"]["power"]["models"];tabs=np.load(src/"tables.npz");b=np.load(src/"power/train_val.npz")
 ix=lambda s:int((np.datetime64(s,"h")-np.datetime64(dc["base"],"h"))/np.timedelta64(1,"h"))
 rng=np.random.default_rng(20241028);oo=rng.choice(b["train_origins"],size=32);x=torch.tensor(np.stack([b["values"][o-96:o] for o in oo]));y=torch.tensor(np.stack([b["values"][o:o+96] for o in oo]));ids=torch.tensor(np.stack([b["ids"][o:o+96] for o in oo]))
 vp,_=windows(b["values"],b["ids"],ix(dc["train_end"]),ix(dc["val_end"]));vx,vy,vi=[torch.tensor(vp[k]) for k in ["x","y","ids"]];perm=rng.permutation(168)
 out=r/"adapter_check";out.mkdir(parents=True,exist_ok=False);rows=[];details=[];pointwise=[]
 stored=np.load(src/"power/confirmation/predictions.npz")
 for seed in c["seeds"]:
  fa=stored[f"Fourier32_s{seed}"];ca=stored[f"Calendar32_s{seed}"];ja=stored[f"Joint168_s{seed}"]
  pointwise.append(dict(seed=seed,fourier_calendar32_maxabs=float(np.max(abs(fa-ca))),fourier_calendar168_maxabs=float(np.max(abs(fa-ja))),n_forecast_cells=int(fa.size)))
  for arm in ["NoCalendar","Fourier32","Calendar32","Joint168","Generic32","Electricity32","Bicycle32","ElectricityShuffle32","BicycleShuffle32"]:
   rec=next(q for q in models if q["seed"]==seed and q["arm"]==arm);m=CalendarTimeBase(cfg,seed,tabs[arm]);m.load_state_dict(torch.load(rec["path"],map_location="cpu",weights_only=True));m.eval()
   with torch.no_grad():
    p=m(vx,vi);effect=p-m.body(vx);raw=m.readout(m.codes());ps=m(vx,vi,torch.tensor(tabs[arm][perm]));rms=float(effect.square().mean().sqrt());shuffle=float((p-ps).square().mean().sqrt())
   train_g=grads(m,x,ids,y);val_g=grads(m,vx,vi,vy)
   clone=copy.deepcopy(m);opt=torch.optim.Adam(clone.parameters(),lr=.001);clone.zero_grad();loss=((clone(x,ids)-y)**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_(clone.parameters(),1.);opt.step()
   after_g=grads(clone,x,ids,y)
   with torch.no_grad():after_rms=float((clone(vx,vi)-clone.body(vx)).square().mean().sqrt())
   curve=list(csv.DictReader((Path(rec["path"]).parent/(arm+"_curve.csv")).open()));v0=float(curve[0]["mse"]);vn=min(float(row["mse"]) for row in curve[1:])
   rows.append(dict(seed=seed,arm=arm,selected_step=rec["best_step"],time_effect_rms=rms,raw_gain_rms=float(raw[:,0].square().mean().sqrt()),raw_bias_rms=float(raw[:,1].square().mean().sqrt()),shuffled_identity_prediction_rms=shuffle,shuffled_identity_prediction_maxabs=float((p-ps).abs().max()),train_first_layer_grad=train_g["first_time_layer"],train_readout_grad=train_g["readout"],validation_readout_grad=val_g["readout"],after_one_update_effect_rms=after_rms,after_one_update_first_layer_grad=after_g["first_time_layer"],step0_val=v0,best_nonzero_step_val=vn,nonzero_step_best_minus_step0=vn-v0))
   details.append(dict(seed=seed,arm=arm,train=train_g,validation=val_g,disposable_clone_after_one_update=after_g))
 csvsave(out/"measurements.csv",rows);csvsave(out/"pointwise_analytical.csv",pointwise);save(out/"gradient_details.json",details);save(out/"scope.json",dict(used_old_data_only=True,original_checkpoints_unchanged=True,clone_updates_not_forecast_candidates=True,numerical_train_origins=oo.tolist(),permutation=perm.tolist()))
 print("POINTWISE",pointwise,flush=True)
 print("QWEN_TASK",json.dumps([x for x in rows if x["arm"]=="Electricity32"]),flush=True)
if __name__=="__main__":
 c=read("docs/chronogeom/residual03_config.json");r=Path(c["run"]);r.mkdir(parents=True,exist_ok=True);save(r/"config.json",c);run(c,r)


def summarize(c,r):
 import pandas as pd
 out=r/"summary";out.mkdir(exist_ok=False);lock=read(r/"lock.json");scores=[];correlations=[];contrasts=[];concentration=[];grouped=[];all_origins=[]
 def correlation(x,y):
  x=np.asarray(x).ravel();y=np.asarray(y).ravel()
  if x.std()<1e-12 or y.std()<1e-12:return None
  return float(np.corrcoef(x,y)[0,1])
 def idx(dc,t):return int((np.datetime64(t,"h")-np.datetime64(dc["base"],"h"))/np.timedelta64(1,"h"))
 for domain,dc in c["domains"].items():
  known=np.load(r/domain/"known_data.npz");tr=known["values"][idx(dc,dc["residual_fit"][0]):idx(dc,dc["residual_fit"][1]),0];edges=np.nanquantile(tr,[1/3,2/3,.9])
  for stage in ["development_validation"]+(["new_confirmation"] if dc["test"] else []):
   dr=r/domain/stage;p=np.load(dr/"predictions.npz");truth=p["truth"].astype(float);origin=p["origin"];comp=read(dr/"completion.json");errors={};abs_errors={}
   for arm in c["arms"]:
    ee=[]
    for seed in c["seeds"]:
     pred=p[f"{arm}_s{seed}"].astype(float);e=pred-truth;base=p[f"NoCorrection_s{seed}"].astype(float);be=base-truth;delta=pred-base;ee.append(e[:,:,0])
     correlations.append(dict(domain=domain,stage=stage,arm=arm,seed=seed,error_vs_base_correlation=correlation(e,be),correction_vs_true_residual_correlation=correlation(delta,truth-base),correction_rms=float(np.sqrt(np.mean(delta**2)))))
    ee=np.stack(ee);errors[arm]=(ee*ee).mean(2);abs_errors[arm]=abs(ee).mean(2);mse=(ee*ee).mean((1,2));mae=abs(ee).mean((1,2))
    scores.append(dict(domain=domain,stage=stage,arm=arm,mse=float(mse.mean()),mae=float(mae.mean()),raw_mse=float(mse.mean()*float(known["sd"][0])**2),raw_mae=float(mae.mean()*float(known["sd"][0])),seed_sd_mse=float(mse.std(ddof=1)),seed2024_mse=float(mse[0]),seed2025_mse=float(mse[1]),seed2026_mse=float(mse[2]),seed2024_mae=float(mae[0]),seed2025_mae=float(mae[1]),seed2026_mae=float(mae[2]),n_origins=len(origin),trainable_parameters=0 if arm=="NoCorrection" else 6570,total_parameters=6570 if arm=="NoCorrection" else 13140))
   lo,hi=comp["lo"],comp["hi"];days=np.arange(lo,hi-96+1,24);pos=(origin-lo)//24
   rng=np.random.default_rng(c["bootstrap_seed"]);B=c["bootstrap_replicates"];si=rng.integers(3,size=(B,3));starts=rng.integers(len(days),size=(B,int(np.ceil(len(days)/14))));ti=((starts[:,:,None]+np.arange(14))%len(days)).reshape(B,-1)[:,:len(days)]
   for other in [a for a in c["arms"] if a!="QwenTask32"]:
    diff=errors["QwenTask32"]-errors[other];ad=abs_errors["QwenTask32"]-abs_errors[other]
    full=np.full((3,len(days)),np.nan);full[:,pos]=diff;draw=np.nanmean(full[si[:,:,None],ti[:,None,:]],axis=(1,2));ci=np.nanquantile(draw,[.025,.975])
    full[:,pos]=ad;drawa=np.nanmean(full[si[:,:,None],ti[:,None,:]],axis=(1,2));cia=np.nanquantile(drawa,[.025,.975]);sd=diff.mean(1)
    contrasts.append(dict(domain=domain,stage=stage,comparison="QwenTask32-"+other,mse_difference=float(diff.mean()),mse_lo95=float(ci[0]),mse_hi95=float(ci[1]),mae_difference=float(ad.mean()),mae_lo95=float(cia[0]),mae_hi95=float(cia[1]),improvement_percent=float(-100*diff.mean()/errors[other].mean()),seed2024_difference=float(sd[0]),seed2025_difference=float(sd[1]),seed2026_difference=float(sd[2]),all3_better=bool(np.all(sd<0)),n_origins=len(origin),calendar_origin_days=len(days)))
    gain=-diff.mean(0);order=np.argsort(-gain);gross=np.maximum(gain,0).sum();net=gain.sum()
    for k in [1,3,5]:
     keep=np.ones(len(gain),bool);keep[order[:k]]=False
     concentration.append(dict(domain=domain,stage=stage,other=other,k=k,n_task_better=int((gain>0).sum()),n_origins=len(origin),topk_gross_positive_share=float(np.maximum(gain[order[:k]],0).sum()/gross) if gross>0 else 0.,topk_net_share=float(gain[order[:k]].sum()/net) if abs(net)>1e-12 else None,leave_topk_out_task_minus_other=float(-gain[keep].mean()),removed_origins=origin[order[:k]].tolist()))
    for j,o in enumerate(origin):all_origins.append(dict(domain=domain,stage=stage,origin=int(o),date=str(np.datetime64(dc["base"],"h")+np.timedelta64(int(o),"h")),other=other,task_mse=float(errors["QwenTask32"][:,j].mean()),other_mse=float(errors[other][:,j].mean()),mse_difference=float(diff[:,j].mean())))
   target=origin[:,None]+np.arange(96);shape=target.shape;dates=pd.DatetimeIndex((np.datetime64(dc["base"],"h")+target).ravel());hour=dates.hour.to_numpy().reshape(shape);weekday=dates.dayofweek.to_numpy().reshape(shape);month=np.array(dates.strftime("%Y-%m")).reshape(shape)
   masks=[("all","all",np.ones(shape,bool)),("clock","evening16_19",(hour>=16)&(hour<=19)),("clock","other_hours",(hour<16)|(hour>19))]
   for kind,x in [("hour",hour),("weekday",weekday),("weekpart",np.where(weekday<5,"weekday","weekend")),("month",month),("demand_bin",np.array(["q0_33","q33_66","q66_90","q90_100"])[np.digitize(truth[:,:,0],edges)])]:
    masks += [(kind,str(v),x==v) for v in np.unique(x)]
   strong=lock["domains"][domain]["strong"]
   for arm in list(dict.fromkeys(["NoCorrection","XOnly","QwenTask32","QwenGeneric32",strong,"IndependentEnsemble"])):
    e=np.stack([(p[f"{arm}_s{s}"].astype(float)-truth)[:,:,0] for s in c["seeds"]])
    for kind,label,mask in masks:
     q=e[:,mask];grouped.append(dict(domain=domain,stage=stage,arm=arm,group_type=kind,group=label,n_forecast_cells_per_seed=int(mask.sum()),n_unique_target_hours=int(len(np.unique(target[mask]))),n_origins=int(np.any(mask,axis=1).sum()),mse=float(np.mean(q*q)),mae=float(np.mean(abs(q))),mean_signed_error=float(q.mean()),overprediction_rate=float(np.mean(q>0))))
 csvsave(out/"scores.csv",scores);csvsave(out/"error_correlations.csv",correlations);csvsave(out/"contrasts.csv",contrasts);csvsave(out/"concentration.csv",concentration);csvsave(out/"group_errors.csv",grouped);csvsave(out/"all_origin_contrasts.csv",all_origins)
 checks={}
 domain="power";stage="new_confirmation";strong=lock["domains"][domain]["strong"]
 for other in list(dict.fromkeys(["NoCorrection","XOnly",strong,"IndependentEnsemble","QwenGeneric32","QwenWrong32","QwenContextShuffle32","Random32","TaskIdentityShuffle32"])):
  row=next(x for x in contrasts if x["domain"]==domain and x["stage"]==stage and x["comparison"]=="QwenTask32-"+other)
  checks[other]=row["all3_better"] and row["mse_hi95"]<0
 save(out/"residual_teacher_value.json",dict(stable_specific_increment=all(checks.values()),checks=checks,distillation_tested=False,capital_is_development=True,new_power_confirmation=True))
 print("SCORES",json.dumps(scores),flush=True);print("TEACHER_RESIDUAL_VALUE",read(out/"residual_teacher_value.json"),flush=True)
 return scores,contrasts
