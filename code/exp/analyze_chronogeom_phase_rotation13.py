"""Saved-prediction checks and descriptive time-block uncertainty; no fitting."""
import csv,json
from pathlib import Path
import numpy as np
from exp import exp_chronogeom_phase_rotation13 as m
def read(p):
    with open(p) as f:return list(csv.DictReader(f))
c=m.rt.read("docs/chronogeom/phase_rotation13_config.json");r=Path(c["run"])
pred=np.load(r/"predictions.npz");maximum=0.;maximum_angle=0.;checks=0
for seed in c["seeds"]:
    for stage in c["stages"]:
        z=np.load(r/f"deployment_s{seed}_{stage}.npz")
        for arm in ["FrozenBase"]+c["student_arms"]:
            angle=np.zeros(len(z["x"])) if arm=="FrozenBase" else m.predict(np.load(r/f"student_s{seed}_{arm}.npz"),z["x"],c)
            p=z["b"]+z["u"]*(np.cos(angle)[:,None]-1)+z["v"]*np.sin(angle)[:,None]
            maximum=max(maximum,float(abs(p-pred[f"s{seed}_{stage}_{arm}"]).max()))
            maximum_angle=max(maximum_angle,float(abs(angle-pred[f"a{seed}_{stage}_{arm}"]).max()));checks+=1
assert maximum==0 and maximum_angle==0
summary=read(r/"summary.csv");base=next(float(x["mse"]) for x in summary if x["arm"]=="FrozenBase")
assert abs(base-.86457629661)<1e-10
cases=read(r/"cases.csv")
lookup={(v["arm"],int(v["seed"]),v["stage"],int(v["origin"])):v for v in cases}
bootstrap=[];rng=np.random.default_rng(1313)
for other in ["FrozenBase","NoKD","FourierKD","QwenWrongKD","QwenShuffleKD"]:
    groups=[];ns=[]
    for stage in c["stages"]:
        origins=sorted(set(int(v["origin"]) for v in cases if v["stage"]==stage))
        delta=np.array([np.mean([float(lookup["QwenTaskKD",seed,stage,o]["mse"])-float(lookup[other,seed,stage,o]["mse"]) for seed in c["seeds"]]) for o in origins])
        blocks=(np.array(origins)-min(origins))//(14*24)
        gs=[delta[blocks==b] for b in sorted(set(blocks))];groups.append(gs);ns.append(len(gs))
    samples=[]
    for _ in range(2000):
        samples.append(np.mean([np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))]).mean() for g in groups]))
    exact=np.mean([np.concatenate(g).mean() for g in groups])
    bootstrap.append(dict(comparison="QwenTaskKD-minus-"+other,mse_difference=float(exact),
        ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)),blocks_per_stage=ns,replicates=2000,descriptive_development_only=True))
teachers=read(r/"teacher_by_seed_period.csv");tm=[]
for arm in c["teacher_arms"]:
    a=[v for v in teachers if v["arm"]==arm]
    tm.append(dict(arm=arm,**{k:float(np.mean([float(v[k]) for v in a])) for k in ["mse","mae","shift_mean_hours"]}))
failures=[];periods=[]
for arm in c["student_arms"]:
    for stage in c["stages"]:
        a=[v for v in cases if v["arm"]==arm and v["stage"]==stage]
        gains=np.array([float(v["gain"]) for v in a])
        angles=np.array([float(v["shift_hours"]) for v in a])
        periods.append(dict(arm=arm,stage=stage,n_per_seed=len(a)//3,
          mse=float(np.mean([float(v["mse"]) for v in a])),mae=float(np.mean([float(v["mae"]) for v in a])),
          useful=int((gains>0).sum()),harmful=int((gains<0).sum()),unchanged=int((gains==0).sum()),
          benefit=float(np.maximum(gains,0).mean()),harm=float(np.maximum(-gains,0).mean()),
          shift_mean_hours=float(angles.mean()),shift_sd_hours=float(angles.std()),
          max_abs_shift_hours=float(abs(angles).max())))
    failures.extend(sorted([v for v in cases if v["arm"]==arm],key=lambda v:float(v["gain"]))[:5])
models=[]
for f in sorted(r.glob("student_s*.npz")):
    z=np.load(f);models.append(dict(file=f.name,parameters=int(z["coef"].size+z["bias"].size),coef_norm=float(np.linalg.norm(z["coef"])),bias=float(z["bias"])))
assert all(v["parameters"]==82 for v in models)
total=sum(p.stat().st_size for p in r.rglob("*") if p.is_file())
assert total/2**20<c["budget"]["max_raw_mib"]
m.rt.save(r/"analysis.json",dict(numpy_replays=checks,max_prediction_error=maximum,max_angle_error=maximum_angle,
 frozen_base_matches_prior=True,raw_mib_before_analysis=total/2**20,bootstrap=bootstrap,teacher_macro=tm,models=models,
 baseline_mean_preserved=True,diagnostic_only_not_independent_confirmation=True))
m.csvsave(r/"periods.csv",periods);m.csvsave(r/"worst_cases.csv",failures)
print(json.dumps(dict(bootstrap=bootstrap,teacher_macro=tm,raw_mib=total/2**20,replays=checks),indent=2))
print("MACRO_BY_SEED")
print((r/"macro_by_seed.csv").read_text())
print("PERIODS")
print((r/"periods.csv").read_text())
print("ORACLE", (r/"training_oracle.json").read_text())
print("COMPLETION", (r/"completion.json").read_text())
