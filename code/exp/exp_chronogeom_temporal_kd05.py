"""Loss-only ablation of existing temporal KD. No new model or teacher."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,csv,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import exp.exp_chronogeom_temporal_kd04 as old
from exp.exp_chronogeom_downstream import CalendarTimeBase,predict
from exp.exp_chronogeom_qwen_capital import windows,setup,tensorpack
from exp.exp_chronogeom_residual03 import frozen_base,features
from exp.exp_chronogeom_task_context02 import grid
from scripts.chronogeom.task_context02_analysis import read,save,csvsave
sha=old.sha
git=old.git
def ix(c,d,t):return old.ix(c,d,t)
def context(c):
 rr=Path(c["reuse_run"]);oc=read(rr/"config.json");tr=Path(oc["teacher_run"]);tc=read(tr/"config.json");gc=read(Path(tc["source_run"])/"config.json")
 return rr,oc,tr,tc,gc
def prepare(c,r):
 r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c);rr,oc,tr,tc,gc=context(c);source=read(rr/"source.json")
 for a in source["teacher_sources"]:assert sha(a["path"])==a["sha256"]
 refs=[]
 for d in c["domains"]:
  (r/d).mkdir()
  for seed in c["seeds"]:
   dd=rr/d/f"seed{seed}";assert (dd/"teacher_training_cache.npz").exists()
   for q in read(dd/"records.json"):
    assert sha(q["path"])==q["sha256"];refs.append(dict(domain=d,**q))
 save(r/"source.json",dict(commit=git(),source_sha=sha(__file__),config_sha=sha("docs/chronogeom/temporal_kd05_config.json"),reuse_config_sha=sha(rr/"config.json"),old_main_script_sha=sha("exp/exp_chronogeom_temporal_kd04.py"),old_students=refs,teacher_sources=source["teacher_sources"],new_confirmation_unread=True))
 print("PREPARED",len(refs),"old models reused",flush=True)
def train_probe(m,w,x,ids,y,target,proto):
 yp=m(x,ids)
 return F.mse_loss(yp,y),F.smooth_l1_loss(old.student_codes(m,w)[ids],proto[ids]),F.mse_loss(yp,target)
def fit(c,r,d,seed,device):
 setup(device,c["gpu_cap_gib"]);start=time.monotonic();rr,oc,tr,tc,gc=context(c);out=r/d/f"seed{seed}";out.mkdir(exist_ok=False)
 b=np.load(rr/d/"known_data.npz");p=old.packed(b,b["train_origins"]);dc=c["domains"][d];va,_=windows(b["values"],b["ids"],ix(c,d,dc["validation"][0]),ix(c,d,dc["validation"][1]));vp=tensorpack(va,device)
 cached=np.load(rr/d/f"seed{seed}/teacher_training_cache.npz");refs=read(rr/d/f"seed{seed}/records.json");base,brec=frozen_base(tc,tr,d,seed,device)
 tx,ty,ti=[torch.tensor(p[k],device=device) for k in ["x","y","ids"]];w=torch.tensor(b["state_weights"],device=device);probe=torch.tensor(np.random.default_rng(seed+271).integers(len(p["x"]),size=c["batch_size"]),device=device)
 records=[]
 for rec in refs:
  q=copy.deepcopy(rec);q.update(model="NoKD" if q["arm"]=="NoKD" else "Joint__"+q["arm"],mode="NoKD" if q["arm"]=="NoKD" else "Joint",teacher=None if q["arm"]=="NoKD" else q["arm"],new_training=False);records.append(q)
 for mode in c["modes"]:
  for teacher in c["teachers"]:
   if time.monotonic()-start>c["per_seed_cap_seconds"]:raise TimeoutError("per-seed budget")
   ref=next(q for q in refs if q["arm"]==teacher);cal=ref["gradient_calibration"];lf=cal["lambda_feature"] if mode=="FeatureOnly" else 0.;lo=cal["lambda_output"] if mode=="OutputOnly" else 0.
   model=mode+"__"+teacher;m=CalendarTimeBase(c,seed,b["student_table"]).to(device);m.body.load_state_dict(copy.deepcopy(base.body.state_dict()));assert sum(q.numel() for q in m.parameters())==6570
   opt=torch.optim.Adam(m.parameters(),lr=c["lr"]);rng=np.random.default_rng(seed+271);curve=[];best=float("inf");tick=time.monotonic();target=torch.tensor(cached[teacher],device=device);proto=torch.tensor(cached[teacher+"_prototype"],device=device)
   for step in range(c["steps"]+1):
    if time.monotonic()-start>c["per_seed_cap_seconds"]:raise TimeoutError("per-seed budget")
    if step%c["validation_every"]==0:
     pp=predict(m,vp);vm=float(np.mean((pp-va["y"])**2))
     with torch.no_grad():ls,lfeat,lout=train_probe(m,w,tx[probe],ti[probe],ty[probe],target[probe],proto)
     curve.append(dict(step=step,mse=vm,mae=float(np.mean(abs(pp-va["y"]))),fixed_training_supervised=float(ls),fixed_training_feature=float(lfeat),fixed_training_imitation=float(lout)))
     if vm<best:best=vm;beststep=step;state=copy.deepcopy(m.state_dict())
    if step==c["steps"]:break
    ii=torch.tensor(rng.integers(len(p["x"]),size=c["batch_size"]),device=device);m.train();yp=m(tx[ii],ti[ii]);loss=F.mse_loss(yp,ty[ii])
    if mode=="FeatureOnly":loss=loss+lf*F.smooth_l1_loss(old.student_codes(m,w)[ti[ii]],proto[ti[ii]])
    else:loss=loss+lo*F.mse_loss(yp,target[ii])
    assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
   last=out/(model+".last.pt");torch.save(m.state_dict(),last);m.load_state_dict(state);path=out/(model+".pt");torch.save(state,path);csvsave(out/(model+"_curve.csv"),curve);np.save(out/(model+"_validation.npy"),predict(m,vp))
   oldcurve=list(csv.DictReader((rr/d/f"seed{seed}"/(teacher+"_curve.csv")).open()));assert abs(curve[0]["mse"]-float(oldcurve[0]["mse"]))<1e-6
   rec=dict(model=model,mode=mode,teacher=teacher,seed=seed,parameters=6570,updates=600,new_training=True,best_step=beststep,val_mse=best,lambda_feature=lf,lambda_output=lo,path=str(path),sha256=sha(path),last_path=str(last),last_sha=sha(last),seconds=time.monotonic()-tick)
   records.append(rec);save(out/"records.json",records);print("FIT",d,json.dumps(rec),flush=True);del m,opt
 assert sha(brec["path"])==brec["sha256"] and all(q.grad is None and not q.requires_grad for q in base.parameters())
 save(out/"completion.json",dict(commit=git(),seconds=time.monotonic()-start,new_models=10,updates_each=600,old_models_reused=6,teacher_unchanged=True,peak_gib=torch.cuda.max_memory_allocated(device)/2**30))
def lock(c,r):
 dd={};seconds=0.
 for d in c["domains"]:
  records=[]
  for seed in c["seeds"]:records+=read(r/d/f"seed{seed}/records.json");seconds+=read(r/d/f"seed{seed}/completion.json")["seconds"]
  assert len(records)==48
  for q in records:assert sha(q["path"])==q["sha256"]
  means={model:float(np.mean([q["val_mse"] for q in records if q["model"]==model])) for model in sorted(set(q["model"] for q in records))}
  dd[d]=dict(models=records,validation_means=means)
 assert seconds<c["total_worker_cap_seconds"];save(r/"lock.json",dict(commit=git(),domains=dd,fit_worker_seconds=seconds,all_new_models_locked=True,consumed_Q2_not_independent=True,new_Q3_unread=True));print("LOCKED",seconds,flush=True)
def getpack(c,r,d,stage):
 rr,oc,tr,tc,gc=context(c);dc=c["domains"][d];b=np.load(rr/d/"known_data.npz")
 if stage=="development_validation":values,ids=b["values"],b["ids"];bounds=dc["validation"]
 else:
  bounds=dc["previous_test"] if stage=="matched_previous" else dc["test"];raw,ids,_=grid(gc,d,bounds[1]);values=(raw-b["mean"])/b["sd"]
 lo,hi=[ix(c,d,t) for t in bounds];p,counts=windows(values,ids,lo,hi);return b,p,counts,lo,hi
def metric(pred,y):
 e=pred.astype(float)-y
 return dict(mse=float(np.mean(e*e)),mae=float(np.mean(abs(e))))
def evaluate(c,r,device):
 setup(device,c["gpu_cap_gib"]);rr,oc,tr,tc,gc=context(c);locked=read(r/"lock.json");maxreplay=0.
 for d,dc in c["domains"].items():
  for stage in ["development_validation"]+(["matched_previous","new_confirmation"] if dc["test"] else []):
   out=r/d/stage;out.mkdir(exist_ok=False)
   for rec in locked["domains"][d]["models"]:assert sha(rec["path"])==rec["sha256"]
   save(out/"before_targets.json",dict(all_models_locked=True,commit=git(),independent=stage=="new_confirmation",consumed_previous=stage=="matched_previous"))
   b,p,counts,lo,hi=getpack(c,r,d,stage);vp=tensorpack(p,device);preds={};mixrows=[]
   previous=np.load(rr/d/"new_confirmation/predictions.npz") if stage=="matched_previous" else None
   if previous is not None:assert np.array_equal(previous["truth"],p["y"]) and np.array_equal(previous["origin"],p["origin"])
   for seed in c["seeds"]:
    base,brec=frozen_base(tc,tr,d,seed,device);normal=np.load(tr/d/f"seed{seed}/numeric_features.npz");ft=features(base,p,device,normal["mean"],normal["sd"]);preds[f"FrozenBase_s{seed}"]=ft["p0"]
    with torch.no_grad():
     oldbody=base.body(vp["x"]);q0=base.readout(base.codes())[0];alpha=1+torch.tanh(q0[0]);beta=q0[1]
     assert torch.allclose(alpha*oldbody+beta,torch.tensor(ft["p0"],device=device),atol=2e-6)
    mixrows.append(dict(model="InitialBodyOnly",seed=seed,condition="BodyOnly",**metric(oldbody.cpu().numpy(),p["y"]),old_gain=float(alpha),old_bias=float(beta)))
    for rec in [a for a in locked["domains"][d]["models"] if a["seed"]==seed]:
     m=CalendarTimeBase(c,seed,b["student_table"]).to(device);m.load_state_dict(torch.load(rec["path"],map_location=device,weights_only=True));pp=predict(m,vp);preds[rec["model"]+f"_s{seed}"]=pp
     if previous is not None and not rec["new_training"]:
      original="NoKD" if rec["mode"]=="NoKD" else rec["teacher"];maxreplay=max(maxreplay,float(np.max(abs(pp-previous[original+f"_s{seed}"]))))
     if rec["model"] in ["NoKD","Joint__TaskKD","FeatureOnly__TaskKD","OutputOnly__TaskKD"]:
      with torch.no_grad():
       body=m.body(vp["x"]);eff=m.readout(m.codes())[vp["ids"]];calgain=1+torch.tanh(eff[...,:1]);calbias=eff[...,1:]
       variants={"Full":torch.tensor(pp,device=device),"TrainedBodyOnly":body,"TrainedBody_OldConstant":alpha*body+beta,"OldBody_TrainedCalendar":calgain*oldbody+calbias}
      for condition,pr in variants.items():mixrows.append(dict(model=rec["model"],seed=seed,condition=condition,**metric(pr.cpu().numpy(),p["y"]),old_gain=float(alpha),old_bias=float(beta)))
     del m
    for teacher in c["teachers"]:
     m=old.load_teacher(oc,rr,d,seed,teacher,device);preds["Teacher__"+teacher+f"_s{seed}"]=old.teacher_predict(m,ft,p,device);del m
    del base
   assert maxreplay<2e-6
   np.savez_compressed(out/"predictions.npz",truth=p["y"],origin=p["origin"],**preds);csvsave(out/"component_interventions.csv",mixrows)
   rows=[]
   for model in sorted(set(k.rsplit("_s",1)[0] for k in preds)):
    e=np.stack([preds[model+f"_s{s}"].astype(float)-p["y"] for s in c["seeds"]]);mse=(e*e).mean((1,2,3));mae=abs(e).mean((1,2,3))
    rows.append(dict(model=model,mse=float(mse.mean()),mae=float(mae.mean()),seed2024_mse=float(mse[0]),seed2025_mse=float(mse[1]),seed2026_mse=float(mse[2]),seed2024_mae=float(mae[0]),seed2025_mae=float(mae[1]),seed2026_mae=float(mae[2]),n_origins=len(p["origin"]),raw_mse=float(mse.mean())*float(b["sd"][0])**2,raw_mae=float(mae.mean())*float(b["sd"][0])))
   csvsave(out/"scores.csv",rows);save(out/"completion.json",dict(commit=git(),domain=d,stage=stage,counts=counts,lo=lo,hi=hi,independent=stage=="new_confirmation",max_old_prediction_replay_difference=maxreplay));print("EVALUATED",d,stage,json.dumps(rows),flush=True)
def flatgrads(loss,m):
 params=list(m.parameters());g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);return torch.cat([(a.detach() if a is not None else torch.zeros_like(b)).reshape(-1) for a,b in zip(g,params)])
def cosine(a,b):
 z=float(a.norm()*b.norm());return float((a@b)/z) if z>1e-12 else None
def diagnostics(c,r,device):
 setup(device,c["gpu_cap_gib"]);rr,oc,tr,tc,gc=context(c);locked=read(r/"lock.json");rows=[];geometry=[];predfit=[]
 for d,dc in c["domains"].items():
  b=np.load(rr/d/"known_data.npz");p=old.packed(b,b["train_origins"]);va,_=windows(b["values"],b["ids"],ix(c,d,dc["validation"][0]),ix(c,d,dc["validation"][1]));tp=tensorpack(p,device);vp=tensorpack(va,device);w=torch.tensor(b["state_weights"],device=device)
  for seed in c["seeds"]:
   cache=np.load(rr/d/f"seed{seed}/teacher_training_cache.npz");probe=torch.tensor(np.random.default_rng(seed+271).integers(len(p["x"]),size=32),device=device);base,brec=frozen_base(tc,tr,d,seed,device);normal=np.load(tr/d/f"seed{seed}/numeric_features.npz");vft=features(base,va,device,normal["mean"],normal["sd"]);vt={}
   for teacher in c["teachers"]:
    m=old.load_teacher(oc,rr,d,seed,teacher,device);vt[teacher]=old.teacher_predict(m,vft,va,device);del m
   for rec in [a for a in locked["domains"][d]["models"] if a["seed"]==seed]:
    teacher=rec["teacher"] or "TaskKD";target=torch.tensor(cache[teacher],device=device);proto=torch.tensor(cache[teacher+"_prototype"],device=device);m=CalendarTimeBase(c,seed,b["student_table"]).to(device);m.load_state_dict(torch.load(rec["path"],map_location=device,weights_only=True));m.eval()
    sup,feat,output=train_probe(m,w,tp["x"][probe],tp["ids"][probe],tp["y"][probe],target[probe],proto);gs,gf,go=[flatgrads(z,m) for z in [sup,feat,output]]
    if rec["mode"]=="Joint":cal=rec["gradient_calibration"];lf,lo=cal["lambda_feature"],cal["lambda_output"]
    elif rec["mode"]=="NoKD":lf=lo=0.
    else:lf,lo=rec["lambda_feature"],rec["lambda_output"]
    rows.append(dict(domain=d,seed=seed,model=rec["model"],teacher=teacher,feature_forecast_cosine=cosine(gf,gs),output_forecast_cosine=cosine(go,gs),feature_output_cosine=cosine(gf,go),active_kd_forecast_cosine=cosine(lf*gf+lo*go,gs),forecast_grad_norm=float(gs.norm()),feature_grad_norm=float(gf.norm()),output_grad_norm=float(go.norm())))
    with torch.no_grad():
     zs=old.student_codes(m,w).cpu().numpy();zt=proto.cpu().numpy();za=zs/np.linalg.norm(zs,axis=1,keepdims=True).clip(1e-8);zb=zt/np.linalg.norm(zt,axis=1,keepdims=True).clip(1e-8);tri=np.triu_indices(168,1)
    geometry.append(dict(domain=d,seed=seed,model=rec["model"],teacher=teacher,feature_smoothl1=float(feat),own_teacher_similarity_correlation=float(np.corrcoef((za@za.T)[tri],(zb@zb.T)[tri])[0,1]),fixed_training_imitation=float(output),fixed_training_forecast=float(sup)))
    trainpred=predict(m,tp);valpred=predict(m,vp);predfit.append(dict(domain=d,seed=seed,model=rec["model"],teacher=teacher,train_teacher_mse=float(np.mean((trainpred-cache[teacher])**2)),val_teacher_mse=float(np.mean((valpred-vt[teacher])**2)),train_forecast_mse=float(np.mean((trainpred-p["y"])**2)),val_forecast_mse=float(np.mean((valpred-va["y"])**2))))
    del m,gs,gf,go
   del base
 out=r/"analysis";out.mkdir(exist_ok=True);csvsave(out/"gradient_conflict.csv",rows);csvsave(out/"representation_alignment.csv",geometry);csvsave(out/"prediction_imitation.csv",predfit)
 print("DIAGNOSTICS_COMPLETE",len(rows),flush=True)
def teacher_prototype_diagnostic(c,r,device):
 setup(device,c["gpu_cap_gib"]);rr,oc,tr,tc,gc=context(c);rows=[]
 for d,dc in c["domains"].items():
  b=np.load(rr/d/"known_data.npz");train=old.packed(b,b["train_origins"])
  for seed in c["seeds"]:
   base,_=frozen_base(tc,tr,d,seed,device);normal=np.load(tr/d/f"seed{seed}/numeric_features.npz");tf=features(base,train,device,normal["mean"],normal["sd"]);m=old.load_teacher(oc,rr,d,seed,"TaskKD",device)
   sums=np.zeros((168,32));count=np.zeros(168);sumh2=0.;N=0
   with torch.inference_mode():
    for j in range(0,len(train["x"]),128):
     num=torch.tensor(tf["num"][j:j+128],device=device);ids=torch.tensor(train["ids"][j:j+128],device=device);lead=m.lead_onehot[None].expand(len(num),-1,-1);h=m.mlp[1](m.mlp[0](torch.cat([num,lead,m.table[ids]],-1))).cpu().numpy().reshape(-1,32);ii=ids.cpu().numpy().ravel();np.add.at(sums,ii,h);count+=np.bincount(ii,minlength=168);sumh2+=float(np.sum(h.astype(float)**2));N+=len(h)
   rawproto=sums/count[:,None];mean=sums.sum(0)/N;total=sumh2/N-float(mean@mean);discarded=sumh2/N-float(np.sum(count[:,None]*rawproto**2)/N);normalized,_=old.center_rms(rawproto,count/N);cached=np.load(rr/d/f"seed{seed}/teacher_training_cache.npz")["TaskKD_prototype"];delta=float(np.max(abs(cached-normalized)));assert delta<1e-5
   np.savez_compressed(r/d/f"seed{seed}/task_raw_prototype.npz",prototype=rawproto)
   for stage in ["development_validation"]+(["matched_previous","new_confirmation"] if dc["test"] else []):
    _,p,counts,lo,hi=getpack(c,r,d,stage);ft=features(base,p,device,normal["mean"],normal["sd"]);out=[]
    with torch.inference_mode():
     for j in range(0,len(p["x"]),128):
      num=torch.tensor(ft["num"][j:j+128],device=device);ids=torch.tensor(p["ids"][j:j+128],device=device);h=torch.tensor(rawproto,dtype=torch.float32,device=device)[ids]
      out.append((torch.tensor(ft["p0"][j:j+128],device=device)+m.mlp[-1](h)+m.numeric(num)+m.lead_bias[None,:,None]).cpu().numpy())
    averaged=np.concatenate(out);full=np.load(r/d/stage/"predictions.npz")[f"Teacher__TaskKD_s{seed}"]
    rows.append(dict(domain=d,stage=stage,seed=seed,training_hidden_variance_discarded_fraction=discarded/max(total,1e-12),original_prototype_replay_maxabs=delta,full_teacher_mse=metric(full,p["y"])["mse"],calendar_mean_hidden_teacher_mse=metric(averaged,p["y"])["mse"],replacement_prediction_rms=float(np.sqrt(np.mean((averaged-full)**2)))))
   del base,m
 csvsave(r/"analysis/teacher_prototype_information.csv",rows);print("PROTOTYPE_DIAGNOSTIC_COMPLETE",flush=True)
def summarize(c,r):
 out=r/"summary";out.mkdir(exist_ok=False);scores=[];contrasts=[];components=[]
 for d,dc in c["domains"].items():
  for stage in ["development_validation"]+(["matched_previous","new_confirmation"] if dc["test"] else []):
   path=r/d/stage;pp=np.load(path/"predictions.npz");truth=pp["truth"];co=read(path/"completion.json");days=np.arange(co["lo"],co["hi"]-96+1,24);positions=(pp["origin"]-co["lo"])//24
   for q in csv.DictReader((path/"scores.csv").open()):scores.append(dict(domain=d,stage=stage,**q))
   for q in csv.DictReader((path/"component_interventions.csv").open()):components.append(dict(domain=d,stage=stage,**q))
   rng=np.random.default_rng(c["bootstrap_seed"]);n=c["bootstrap_replicates"];si=rng.integers(3,size=(n,3));st=rng.integers(len(days),size=(n,int(np.ceil(len(days)/c["block_days"]))));ti=((st[:,:,None]+np.arange(c["block_days"]))%len(days)).reshape(n,-1)[:,:len(days)]
   def errors(a,absolute=False):
    ee=np.stack([pp[a+f"_s{s}"].astype(float)-truth for s in c["seeds"]]);return (abs(ee) if absolute else ee*ee).mean((2,3))
   pairs=[]
   for mode in ["Joint","FeatureOnly","OutputOnly"]:
    a=mode+"__TaskKD";pairs += [(a,b) for b in ["NoKD","FrozenBase"]+[mode+"__"+t for t in c["teachers"] if t!="TaskKD"]]
   pairs += [("FeatureOnly__TaskKD","Joint__TaskKD"),("OutputOnly__TaskKD","Joint__TaskKD"),("OutputOnly__TaskKD","FeatureOnly__TaskKD")]
   for a,b in pairs:
    de=errors(a)-errors(b);da=errors(a,True)-errors(b,True);arr=np.full((3,len(days)),np.nan);arr[:,positions]=de;ci=np.nanquantile(np.nanmean(arr[si[:,:,None],ti[:,None,:]],axis=(1,2)),[.025,.975]);arr[:,positions]=da;ac=np.nanquantile(np.nanmean(arr[si[:,:,None],ti[:,None,:]],axis=(1,2)),[.025,.975])
    contrasts.append(dict(domain=d,stage=stage,comparison=a+"-"+b,mse_difference=float(de.mean()),mse_lo95=float(ci[0]),mse_hi95=float(ci[1]),mae_difference=float(da.mean()),mae_lo95=float(ac[0]),mae_hi95=float(ac[1]),improvement_percent=float(-100*de.mean()/errors(b).mean()),all3_better=bool(np.all(de.mean(1)<0)),seed2024_difference=float(de[0].mean()),seed2025_difference=float(de[1].mean()),seed2026_difference=float(de[2].mean())))
 csvsave(out/"scores.csv",scores);csvsave(out/"contrasts.csv",contrasts);csvsave(out/"component_interventions.csv",components)
 print("SUMMARY_COMPLETE",flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["prepare","fit","lock","evaluate","diagnostics","prototype","summarize"]);ap.add_argument("--domain");ap.add_argument("--seed",type=int);ap.add_argument("--device",default="cuda:1");args=ap.parse_args();c=read("docs/chronogeom/temporal_kd05_config.json");r=Path(c["run"])
 try:
  if args.stage=="prepare":prepare(c,r)
  elif args.stage=="fit":fit(c,r,args.domain,args.seed,args.device)
  elif args.stage=="lock":lock(c,r)
  elif args.stage=="evaluate":evaluate(c,r,args.device)
  elif args.stage=="diagnostics":diagnostics(c,r,args.device)
  elif args.stage=="prototype":teacher_prototype_diagnostic(c,r,args.device)
  else:summarize(c,r)
 except BaseException as e:
  r.mkdir(parents=True,exist_ok=True);save(r/f"failure_{args.stage}_{args.domain}_{args.seed}.json",dict(error=str(e),traceback=traceback.format_exc()));raise
if __name__=="__main__":main()
