"""Cached-only analysis, never changes models/selection."""
from pathlib import Path
import csv,json,math
import numpy as np
import torch
from exp import exp_chronogeom_ordinal_opd01 as o
import sys
r=Path(o.read(sys.argv[1] if len(sys.argv)>1 else "docs/chronogeom/ordinal_opd01_config.json")["run"])
c=o.read(r/"config.json");a=np.load(r/"predictions.npz");days=a["origin"]//24
grid=np.arange(days.min(),days.max()+1);rng=np.random.default_rng(c["bootstrap"]["seed"])
length=c["bootstrap"]["block_days"];starts=rng.integers(len(grid),size=(c["bootstrap"]["replicates"],math.ceil(len(grid)/length)))
resample=((starts[:,:,None]+np.arange(length))%len(grid)).reshape(len(starts),-1)[:,:len(grid)]
errs={}
for name in ["Start"]+c["families"]+[f+"_matched" for f in c["families"]]:
 per=[]
 for seed in c["seeds"]:
  m,l,s,ma=o.rt.components(a[f"{name}_s{seed}"],a["truth"]);per.append(np.stack([m,ma],axis=-1))
 errs[name]=np.mean(per,axis=0)
out=[]
for other in ["Start","NoKD","FourierOPD","QwenShuffle","NoKD_matched","FourierOPD_matched","QwenShuffle_matched"]:
 for k,metric in enumerate(["mse","mae"]):
  delta=errs["QwenOPD"][:,k]-errs[other][:,k]
  v=np.array([np.mean(delta[days==d]) if np.any(days==d) else np.nan for d in grid])
  dist=np.nanmean(v[resample],axis=1)
  out.append(dict(comparison="QwenOPD-"+other,metric=metric,point_difference=float(delta.mean()),ci_low=float(np.nanquantile(dist,.025)),ci_high=float(np.nanquantile(dist,.975)),origins=len(days),candidate_days=len(grid),bootstrap=3000,block_days=7,seeds_averaged_before_resampling=True))
o.csvsave(r/"paired_block_intervals.csv",out)
data=np.load(c["dynamic_run"]+"/data.npz")
feedback=[json.loads(s) for s in (r/"qwen_feedback.jsonl").read_text().splitlines()]
diag=[]
for phase in ["inner","full"]:
 statepath=r/f"analytic_{phase}.pt"
 m=torch.nn.Linear(102,3);m.load_state_dict(torch.load(statepath,map_location="cpu",weights_only=True));m.eval()
 v=o.read(r/f"analytic_{phase}.json");target_rows=[]
 for z in feedback:
  if z["phase"]!=phase or z["family"]!="QwenOPD":continue
  ii=np.array(z["rows"]);truth=o.true_states(data["y"][ii],c);qp=np.array(z["probabilities"])
  x=o.analytical_x(data["ids"][ii],data["statistics"][ii],np.array(z["states"]),np.array(v["mu"]),np.array(v["sd"]))
  with torch.no_grad():fp=m(torch.tensor(x)).softmax(-1).numpy()
  for name,p in [("Qwen",qp),("Fourier",fp)]:
   correct=(p.argmax(-1)==truth);nll=-np.log(np.maximum(np.take_along_axis(p,truth[...,None],axis=-1)[...,0],1e-12))
   target_rows.append(dict(name=name,accuracy=float(correct.mean()),nll=float(nll.mean()),low=float(p[...,0].mean()),middle=float(p[...,1].mean()),high=float(p[...,2].mean()),true_low=float((truth==0).mean()),true_middle=float((truth==1).mean()),true_high=float((truth==2).mean())))
 for name in ["Qwen","Fourier"]:
  rr=[z for z in target_rows if z["name"]==name]
  if rr:diag.append(dict(phase=phase,teacher=name,feedback_batches=len(rr),**{k:float(np.mean([z[k] for z in rr])) for k in rr[0] if k!="name"}))
if diag:o.csvsave(r/"teacher_training_diagnostics.csv",diag)
prior=np.load(Path(c["early_run"])/"predictions.npz")
parity=max(float(abs(a[f"Start_s{s}"]-prior[f"PlainFourier_s{s}"]).max()) for s in c["seeds"])
assert parity<1e-12
decomp=[]
for name in ["NoKD_matched","FourierOPD_matched","QwenOPD","QwenShuffle_matched"]:
 for seed in c["seeds"]:
  base=a[f"Start_s{seed}"];delta=a[f"{name}_s{seed}"]-base;error=base-a["truth"]
  variance=float(np.mean(delta**2));cross=float(2*np.mean(error*delta));difference=float(np.mean((base+delta-a["truth"])**2)-np.mean(error**2));assert abs(variance+cross-difference)<1e-12
  decomp.append(dict(arm=name,seed=seed,correction_rms=variance**.5,cross_error=cross,squared_correction=variance,mse_difference=difference,mean_window_std=float(np.mean((base+delta).std(1))),start_window_std=float(np.mean(base.std(1)))))
o.csvsave(r/"correction_decomposition.csv",decomp)
o.save(r/"start_parity.json",dict(max_absolute_error=parity,original_fourier_source=str(Path(c["early_run"])/"predictions.npz")))
o.save(r/"cached_analysis.json",dict(no_new_training=True,no_new_llm_calls=True,bootstrap_seed=c["bootstrap"]["seed"],overlapping_origins_not_independent=True,ordinal_training_diagnostics_not_heldout_forecast_scores=True))
print("INTERVALS",json.dumps(out));print("TEACHER_DIAGNOSTICS",json.dumps(diag))
