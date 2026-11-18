"""Saved-forecast verification and paired time-block description, no fitting."""
import csv,json
from pathlib import Path
import numpy as np
from exp import exp_chronogeom_phase_opd17 as m
c=m.rt.read(m.CONFIG);r=Path(c["run"]);lock=m.rt.read(r/"selection_lock.json")
assert all(m.rt.sha(p)==v for p,v in lock["files"].items())
def read(name):
 with (r/name).open() as f:return list(csv.DictReader(f))
pred=np.load(r/"predictions.npz");replays=0;err=0.
for seed in c["seeds"]:
 for stage in c["stages"]:
  z=np.load(r/f"deployment_s{seed}_{stage}.npz")
  for arm in c["student_arms"]:
   model=np.load(r/f"student_s{seed}_{arm}.npz");assert m.bil.count(model)==163
   a=m.bil.predict(model,z["x"],c)
   p=z["b"]+z["u"]*(np.cos(a)[:,None]-1)+z["v"]*np.sin(a)[:,None]
   err=max(err,float(abs(p-pred[f"s{seed}_{stage}_{arm}"]).max()));replays+=1
assert err==0
cases=read("cases.csv");lookup={(v["arm"],int(v["seed"]),v["stage"],int(v["origin"])):v for v in cases}
summary=read("summary.csv");routes=m.rt.read(r/"route_choice.json")
task=routes["Task"];controls=["FrozenBase","FrozenFourierKD","NoKD",routes["Fourier"],routes["Wrong"],routes["Shuffle"]]
rng=np.random.default_rng(1515);boot=[]
for control in controls:
 for metric in ["mse","mae"]:
  groups=[]
  for stage in c["stages"]:
   origins=sorted({int(v["origin"]) for v in cases if v["stage"]==stage})
   delta=np.array([np.mean([float(lookup[task,seed,stage,o][metric])-float(lookup[control,seed,stage,o][metric]) for seed in c["seeds"]]) for o in origins])
   block=(np.array(origins)-min(origins))//(14*24)
   groups.append([delta[block==j] for j in sorted(set(block))])
  reps=[np.mean([np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))]).mean() for g in groups]) for _ in range(2000)]
  boot.append(dict(control=control,metric=metric,difference=float(np.mean([np.concatenate(g).mean() for g in groups])),
   ci_low=float(np.quantile(reps,.025)),ci_high=float(np.quantile(reps,.975)),blocks_per_stage=[len(g) for g in groups],
   descriptive_development_only=True))
worst=sorted([v for v in cases if v["arm"]==task],key=lambda v:float(v["gain"]))[:5];m.csvsave(r/"worst_cases.csv",worst)
teachers=read("teacher_by_seed_period.csv");tm=[]
for arm in c["teacher_arms"]:
 a=[v for v in teachers if v["arm"]==arm];tm.append(dict(arm=arm,**{k:float(np.mean([float(v[k]) for v in a])) for k in ["mse","mae"]}))
m.csvsave(r/"teacher_summary.csv",tm)
models=list(r.glob("student_s*.npz"));assert len(models)==27
src=m.rt.read(r/"source_train.json")
assert all(m.rt.sha(p)==v for p,v in src["sources"].items())
assert all(m.rt.sha(p)==v for p,v in lock["files"].items())
older=Path("/home/home_new/qsmx/runs/ChronoGeom/downstream/phase_response_opd16_20261030_01")
parity={"NoKD":0.,"teachers":0.}
for seed in c["seeds"]:
 for kind,arms in [("student",["NoKD"]),("teacher",c["teacher_arms"])]:
  for arm in arms:
   now=np.load(r/f"{kind}_s{seed}_{arm}.npz");before=np.load(older/f"{kind}_s{seed}_{arm}.npz")
   target="NoKD" if kind=="student" else "teachers"
   parity[target]=max(parity[target],max(float(abs(now[k]-before[k]).max()) for k in now.files))
assert max(parity.values())==0
m.rt.save(r/"parity16.json",dict(parameter_max_abs_error=parity,NoKD_and_teacher_weights_unchanged=True))
size=sum(f.stat().st_size for f in r.rglob("*") if f.is_file())/2**20;assert size<c["budget"]["max_raw_mib"]
m.rt.save(r/"analysis.json",dict(replays=replays,max_prediction_error=err,raw_mib=size,routes=routes,
 bootstrap=boot,model_count=len(models),new_LLM_calls=0,all_sources_and_selection_unchanged=True))
print("SUMMARY",(r/"summary.csv").read_text())
print("SEEDS",(r/"macro_by_seed.csv").read_text())
print("TEACHERS",json.dumps(tm))
print("ANALYSIS",json.dumps(dict(bootstrap=boot,replays=replays,raw_mib=size,routes=routes)))
print("COMPLETION",(r/"completion.json").read_text())
