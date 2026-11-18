"""Train-only magnitude shrinkage of frozen teacher shape corrections. No new model."""
from exp.exp_chronogeom_temporal_readout06 import *

def fit(c,r,seed,device,kd=False):
 setup(device,c["gpu_cap_gib"]);tick=time.monotonic();rr,oc,tr,tc,gc=context(c);b=np.load(rr/"power/known_data.npz");p=old.packed(b,b["train_origins"]);v,_=windows(b["values"],b["ids"],ix(c,c["validation"][0]),ix(c,c["validation"][1]));tabs=np.load(r/"tables.npz")
 base,rec=frozen_base(tc,tr,"power",seed,device);raw=raw_predict(base,p,device);vr=raw_predict(base,v,device);rt=torch.tensor(raw,device=device);idt=torch.tensor(p["ids"],device=device);yt=torch.tensor(p["y"],dtype=torch.float64,device=device)
 out=r/(("kd_" if kd else "sup_")+f"seed{seed}");out.mkdir(exist_ok=False);defs=groupdefs(c,kd);records=[];cache=np.load(rr/"power"/f"seed{seed}/teacher_training_cache.npz") if kd else None
 if kd:assert read(r/"gate.json")["passed"]
 for name,desc in defs.items():
  if time.monotonic()-tick>c["fit_cap_seconds"]:raise TimeoutError("fit budget")
  table=tabs["Calendar32" if kd else desc["table"]];mode="level" if kd else desc["mode"];m=TimeReadout(base,table,seed,mode).to(device);params=[p for p in m.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==6528 and sum(p.numel() for p in m.parameters())==13098
  target=torch.tensor(cache[desc["teacher"]],dtype=torch.float64,device=device) if kd else None;shrink=None;fixed_lam=None;lam=0.;cal=None;rng=np.random.default_rng(seed+271);opt=torch.optim.Adam(params,lr=c["lr"]);curve=[];best=float("inf");start=time.monotonic();grad0=None;grad1=None
  if kd:
   with torch.no_grad():
    pb=m.baseline(rt).double();direction=center(target-pb);residual=center(yt-pb)
    denominator=float((direction*direction).sum());unclipped=float((direction*residual).sum())/denominator if denominator>1e-20 else 0.;alpha=float(np.clip(unclipped,0.,1.))
    before=float(F.mse_loss(direction,residual));target=pb+alpha*direction;after=float(F.mse_loss(alpha*direction,residual))
    assert after<=before+1e-12
    shrink=dict(alpha=alpha,unclipped_alpha=unclipped,training_shape_mse_before=before,training_shape_mse_after=after,training_origins=len(raw),estimation="training-only scalar projection; clipped0to1; no validation/test labels")
   anchor=next(q for q in read(Path(c["reference_run"])/f"kd_seed{seed}/records.json") if q["teacher"]==desc["teacher"] and q["kd_loss"]=="output")
   fixed_lam=float(anchor["calibration"]["lambda_value"])
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
     lam=fixed_lam
     constant=float(((yp.mean(1)-target[ii].mean(1))**2).mean());cal=dict(shrinkage=shrink,lambda_source="retained original unshrunk initial-gradient coefficient",lambda_value=lam,supervised_grad=float(gs.norm()),shape_grad=float(ga.norm()),output_shape_grad_max_difference=float((go-ga).abs().max()),output_shape_grad_relative_difference=float((go-ga).norm()/ga.norm().clamp_min(1e-12)),loss_constant_identity_error=float(abs(float(output_aux-shape_aux)-constant)))
     assert cal["output_shape_grad_max_difference"]<1e-6
    loss=loss+lam*aux
   opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(loss);torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
   if step==0:grad0=float(m.readout.weight.grad.norm());assert grad0>1e-10
   if step==1:grad1=float(m.time_mlp[0].weight.grad.norm());assert grad1>1e-12
   opt.step()
  m.load_state_dict(state);path=out/(name+".pt");torch.save(state,path);csvsave(out/(name+"_curve.csv"),curve);pp=forecast(m,vr,v["ids"],device);np.save(out/(name+"_validation.npy"),pp)
  basepred=m.baseline(torch.tensor(vr,device=device)).double().detach().cpu().numpy();meanerr=float(np.max(abs(pp.mean(1)-basepred.mean(1))))
  if mode=="level":assert meanerr<2e-6
  record=dict(name=name,seed=seed,kind="kd" if kd else "supervised",mode=mode,table="Calendar32" if kd else desc["table"],teacher=desc.get("teacher"),kd_loss=desc.get("loss"),parameters=13098,trainable=6528,best_step=beststep,val_mse=best,path=str(path),sha256=sha(path),steps=c["steps"],seconds=time.monotonic()-start,head_initial_grad=grad0,upstream_after_update_grad=grad1,initial_prediction_maxdiff=float(np.max(abs(initial-reference))),selected_mean_maxdiff=meanerr,calibration=cal)
  records.append(record);save(out/"records.json",records);print("FIT",json.dumps(record),flush=True);assert all(not p.requires_grad and p.grad is None for p in m.base.parameters());del m,opt
 assert sha(rec["path"])==rec["sha256"];save(out/"completion.json",dict(seconds=time.monotonic()-tick,models=len(records),steps_each=c["steps"],base_unchanged=True,peak_gib=torch.cuda.max_memory_allocated(device)/2**30))
