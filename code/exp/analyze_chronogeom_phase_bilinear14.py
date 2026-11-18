"""Post-lock saved forecast analysis; no new fitting."""
from pathlib import Path
import csv,json
import numpy as np
from exp import exp_chronogeom_phase_bilinear14 as m
c=m.rt.read(m.CONFIG);r=Path(c["run"])
def read(name):
 with (r/name).open() as f:return list(csv.DictReader(f))
src=m.rt.read(r/"source_train.json");lock=m.rt.read(r/"selection_lock.json")
assert all(m.rt.sha(p)==v for p,v in lock["files"].items())
rows=read("cases.csv");rawpred=np.load(r/"predictions.npz")
lookup={(v["arm"],int(v["seed"]),v["stage"],int(v["origin"])):v for v in rows}
arms=sorted(set(v["arm"] for v in rows));replay=0;maxerr=0.;ablations=[];models=[]
pc=m.rt.read(Path(c["reliability_run"])/"config.json");raw=Path(pc["reuse_run"])
for seed in c["seeds"]:
 for stage in c["stages"]:
  data=np.load(raw/f"seed{seed}_{stage}.npz");y=data["y"]
  for arm in arms:
   if arm=="FrozenBase":continue
   teacher=arm.removeprefix("Teacher_")
   name="SeasonFourier32" if arm=="NoKD_Fourier" else teacher
   path=r/(f"student_s{seed}_{arm.removeprefix('Student_')}.npz" if arm.startswith("Student_") else f"teacher_s{seed}_{name}.npz")
   model=np.load(path);z=np.load(r/f"deployment_s{seed}_{stage}_{arm}.npz")
   angle=m.predict(model,z["x"],c)
   pred=z["b"]+z["u"]*(np.cos(angle)[:,None]-1)+z["v"]*np.sin(angle)[:,None]
   maxerr=max(maxerr,float(abs(pred-rawpred[f"s{seed}_{stage}_{arm}"]).max()));replay+=1
   xx=(z["x"]-model["mu"])/model["sd"]
   for ab,logits in [("Full",m.numpy_logits(model,z["x"])),("NoBilinear",xx@model["coef"]+float(model["bias"])),
                  ("BiasOnly",np.full(len(y),float(model["bias"])))]:
    delta=c["delta_max"]*np.tanh(logits)
    p=z["b"]+z["u"]*(np.cos(delta)[:,None]-1)+z["v"]*np.sin(delta)[:,None]
    ablations.append(dict(seed=seed,stage=stage,arm=arm,ablation=ab,mse=float(np.mean((p-y)**2)),mae=float(np.mean(abs(p-y)))))
   if stage==c["stages"][0]:
    models.append(dict(seed=seed,arm=arm,parameters=m.count(model),linear_norm=float(np.linalg.norm(model["coef"])),
       left_norm=float(np.linalg.norm(model["left"])),right_norm=float(np.linalg.norm(model["right"])),
       interaction_matrix_norm=float(np.linalg.norm(model["left"])*np.linalg.norm(model["right"])),
       bias=float(model["bias"])))
assert maxerr==0 and all(v["parameters"]==163 for v in models)
m.csvsave(r/"ablations.csv",ablations);m.csvsave(r/"model_norms.csv",models)
rng=np.random.default_rng(1414);boots=[]
for control in ["FrozenBase","NoKD_Fourier","Teacher_QwenTrainShuffle","Teacher_QwenWrong"]:
 groups=[]
 for stage in c["stages"]:
  origins=sorted({int(v["origin"]) for v in rows if v["stage"]==stage})
  diffs=np.array([np.mean([float(lookup["Teacher_QwenTask",seed,stage,o]["mse"])-float(lookup[control,seed,stage,o]["mse"]) for seed in c["seeds"]]) for o in origins])
  block=(np.array(origins)-min(origins))//(14*24)
  groups.append([diffs[block==k] for k in sorted(set(block))])
 res=[np.mean([np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))]).mean() for g in groups]) for _ in range(2000)]
 boots.append(dict(control=control,mse_difference=float(np.mean([np.concatenate(g).mean() for g in groups])),
   ci_low=float(np.quantile(res,.025)),ci_high=float(np.quantile(res,.975)),blocks_per_stage=[len(g) for g in groups],
   descriptive_development_only=True))
direction=[]
for arm in arms:
 if arm=="FrozenBase":continue
 for sign in [-1,1]:
  a=[v for v in rows if v["arm"]==arm and int(v["oracle_direction"])==sign and float(v["phase_energy"])>1e-12]
  g=np.array([float(v["gain"]) for v in a]);shift=np.array([float(v["shift_hours"]) for v in a])
  direction.append(dict(arm=arm,oracle_direction=sign,n_records=len(a),n_unique_origins=len({v["origin"] for v in a}),
    correct_direction_fraction=float(np.mean(np.sign(shift)==sign)),mean_shift_hours=float(shift.mean()),
    gross_benefit=float(np.maximum(g,0).mean()),gross_harm=float(np.maximum(-g,0).mean()),net_gain=float(g.mean())))
m.csvsave(r/"direction_diagnostics.csv",direction)
worst=sorted([v for v in rows if v["arm"]=="Teacher_QwenTask"],key=lambda v:float(v["gain"]))[:5]
m.csvsave(r/"worst_cases.csv",worst)
assert all(m.rt.sha(p)==v for p,v in lock["files"].items())
assert all(m.rt.sha(p)==v for p,v in src["sources"].items())
size=sum(p.stat().st_size for p in r.rglob("*") if p.is_file())/2**20
assert size<c["budget"]["max_raw_mib"]
m.rt.save(r/"analysis.json",dict(replays=replay,max_prediction_error=maxerr,model_parameters=163,raw_mib=size,
 bootstrap=boots,all_old_sources_and_lock_files_unchanged=True))
print("SUMMARY");print((r/"summary.csv").read_text())
print("SEEDS");print((r/"macro_by_seed.csv").read_text())
print("GATE");print((r/"teacher_gate.json").read_text())
print("ANALYSIS",json.dumps(dict(bootstrap=boots,replays=replay,raw_mib=size)))
print("DIRECTIONS",json.dumps(direction))
print("COMPLETION",(r/"completion.json").read_text())
