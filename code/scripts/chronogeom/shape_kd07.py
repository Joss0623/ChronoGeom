"""Same-budget shape-target shrinkage, reusing frozen teacher caches and TimeBase."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,csv,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from exp import exp_chronogeom_shape_kd07 as old
from scripts.chronogeom.task_context02_analysis import read,save,csvsave

def run(config,device):
 c=read(config);r=Path(c["run"]);ref=Path(c["reference_run"]);r.mkdir(parents=True,exist_ok=False);tick=time.monotonic()
 original_lock=read(ref/"lock.json");original_shas={q["path"]:old.sha(q["path"]) for q in original_lock["models"]}
 save(r/"config.json",c)
 np.savez_compressed(r/"tables.npz",**dict(np.load(ref/"tables.npz")))
 save(r/"gate.json",dict(passed=True,reason="Explicit user-requested budget sensitivity; this is not a new teacher-value gate",independent=False))
 save(r/"source.json",dict(commit=old.git(),source_sha=old.sha(__file__),config_sha=old.sha(config),reused_source_sha=old.sha(old.__file__),official_timebase_sha=old.sha("layers/ChronoGeomTimeBaseOfficial.py"),reference_run=str(ref),all_evaluation_intervals_previously_viewed=True,event_teacher_available=False,historical_event_template_available=False))
 records=[]
 for seed in c["seeds"]:
  for kd in [True]:
   if time.monotonic()-tick>c["workflow_cap_seconds"]:raise TimeoutError("total budget")
   old.fit(c,r,seed,device,kd)
   folder=r/(("kd_" if kd else "sup_")+f"seed{seed}")
   rr=read(folder/"records.json")
   for q in rr:
    curve=list(csv.DictReader((folder/(q["name"]+"_curve.csv")).open()))
    assert [int(x["step"]) for x in curve]==[0,c["steps"]]
    # Reused fit() loops over c["steps"] but hard-codes 600 in its metadata.
    # Correct this new-run metadata only; original sources/runs remain unchanged.
    q.update(steps=c["steps"],comparison_class="equal100_train_only_shape_shrinkage",reported_steps_in_reused_function=c["steps"])
   save(folder/"records.json",rr)
   co=read(folder/"completion.json");co["steps_each"]=c["steps"];save(folder/"completion.json",co)
   records+=rr
 save(r/"lock.json",dict(models=records,all_evaluation_intervals_previously_viewed=True,independent=False))
 rr,oc,tr,tc,gc=old.context(c);b=np.load(rr/"power/known_data.npz");tabs=np.load(r/"tables.npz")
 results=[];prediction_files=[];prefix_checks=[]
 for seed in c["seeds"]:
  for weak,full,kind in c["reference_names"]:
   weak_folder=r/(("sup_" if kind=="supervised" else "kd_")+f"seed{seed}")
   full_folder=ref/(("sup_" if kind=="supervised" else "kd_")+f"seed{seed}")
   wc=list(csv.DictReader((weak_folder/(weak+"_curve.csv")).open()))
   fc=list(csv.DictReader((full_folder/(full+"_curve.csv")).open()))
   for a in [z for z in wc if int(z["step"])==0]:
    d=next(x for x in fc if int(x["step"])==int(a["step"]))
    err=max(abs(float(a[k])-float(d[k])) for k in ["mse","mae","level","shape"])
    prefix_checks.append(dict(seed=seed,weak=weak,full=full,step=int(a["step"]),max_metric_difference=err));assert err<2e-6
 for stage,bounds in {"validation":c["validation"],"viewed_fall":c["confirmation"]}.items():
  if time.monotonic()-tick>c["workflow_cap_seconds"]:raise TimeoutError("total budget")
  if stage=="validation":values,ids=b["values"],b["ids"];refstage="validation"
  else:
   raw,ids,_=old.old.grid(gc,"power",bounds[1]);values=(raw-b["mean"])/b["sd"];refstage="confirmation"
  p,counts=old.windows(values,ids,old.ix(c,bounds[0]),old.ix(c,bounds[1]))
  original=np.load(ref/refstage/"predictions.npz");assert np.array_equal(p["origin"],original["origin"]);np.testing.assert_allclose(p["y"],original["truth"],atol=0,rtol=0)
  preds={"truth":p["y"],"origin":p["origin"]}
  for seed in c["seeds"]:
   base,_=old.frozen_base(tc,tr,"power",seed,device);raw=old.raw_predict(base,p,device)
   for name in ["FrozenBase","B_Calendar","KD_Fourier_Output","KD_Shuffle_Output","KD_Task_Output"]:
    pp=original[name+f"_s{seed}"];preds[name+f"_s{seed}"]=pp
    mse,level,shape,mae=old.components(pp,p["y"]);results.append(dict(stage=stage,name=name,seed=seed,steps=0 if name=="FrozenBase" else 600,budget_class="retained_reference",mse=float(mse.mean()),mae=float(mae.mean()),level=float(level.mean()),shape=float(shape.mean()),n_origins=len(p["origin"])))
   for q in [z for z in records if z["seed"]==seed]:
    m=old.TimeReadout(base,tabs[q["table"]],seed,q["mode"]).to(device);m.load_state_dict(torch.load(q["path"],map_location=device,weights_only=True));pp=old.forecast(m,raw,p["ids"],device)
    assert old.sha(q["path"])==q["sha256"];assert np.isfinite(pp).all()
    assert np.max(abs(pp.mean(1)-original[f"FrozenBase_s{seed}"].mean(1)))<2e-6
    preds[q["name"]+f"_s{seed}"]=pp;mse,level,shape,mae=old.components(pp,p["y"])
    results.append(dict(stage=stage,name=q["name"],seed=seed,steps=c["steps"],budget_class="equal100_train_only_shape_shrinkage",mse=float(mse.mean()),mae=float(mae.mean()),level=float(level.mean()),shape=float(shape.mean()),n_origins=len(p["origin"])))
    del m
   del base
  np.savez_compressed(r/(stage+"_predictions.npz"),**preds);prediction_files.append(stage+"_predictions.npz")
  save(r/(stage+"_counts.json"),dict(**counts,independent=False))
 csvsave(r/"scores_by_seed.csv",results)
 means=[]
 for stage in ["validation","viewed_fall"]:
  for name in sorted(set(q["name"] for q in results)):
   rows=[q for q in results if q["stage"]==stage and q["name"]==name]
   out={k:rows[0][k] for k in ["stage","name","steps","budget_class","n_origins"]}
   out.update({k:float(np.mean([q[k] for q in rows])) for k in ["mse","mae","level","shape"]});means.append(out)
 csvsave(r/"scores_mean.csv",means);csvsave(r/"prefix_equivalence.csv",prefix_checks)
 assert all(old.sha(p)==h for p,h in original_shas.items())
 ver=dict(new_fits=len(records),updates=len(records)*c["steps"],old_checkpoints_unchanged=True,official_timebase_unchanged=old.sha("layers/ChronoGeomTimeBaseOfficial.py")==read(r/"source.json")["official_timebase_sha"],prefix_checks=len(prefix_checks),prefix_max_error=max(q["max_metric_difference"] for q in prefix_checks),all_intervals_previously_viewed=True,independent_confirmation=False,event_experiment=False,seconds=time.monotonic()-tick,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30)
 assert ver["new_fits"]==9 and ver["official_timebase_unchanged"]
 save(r/"completion.json",ver);print(json.dumps(dict(completion=ver,scores=means),indent=2),flush=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--device",default="cuda:3");a=ap.parse_args()
 try:run(a.config,a.device)
 except BaseException:
  c=read(a.config);r=Path(c["run"])
  if r.exists():save(r/"failure.json",dict(traceback=traceback.format_exc()))
  raise
