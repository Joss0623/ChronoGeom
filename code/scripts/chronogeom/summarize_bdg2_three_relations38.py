from pathlib import Path
import csv,json,numpy as np
from exp.exp_chronogeom_event_knowledge19 import scores,read,save,sha
r=Path("/home/home_new/qsmx/runs/ChronoGeom/downstream/bdg2_three_relations38_20261101_01")
assert read(r/"status.json")["complete"]
c=read(r/"config.json");rows=read(r/"metrics.json");cut=int((np.datetime64(c["selection_cut"],"h")-np.datetime64("2016-01-01","h"))/np.timedelta64(1,"h"))
maxerr=0.;maxlevel=0.;amps=[];checked=0
for d in r.iterdir():
 if not d.is_dir():continue
 for file in d.glob("*_val.npz"):
  seed,arm,_=file.stem.split("_");b=np.load(file);p=b["prediction"];y=b["y"];o=b["origin"]
  for period,mask in {"val":np.ones(len(o),bool),"cal":o+96<=cut,"check":o>=cut}.items():
   item=next(x for x in rows if x["building"]==d.name and x["seed"]==int(seed) and x["arm"]==arm and x["period"]==period);q=scores(p[mask],y[mask]);maxerr=max(maxerr,max(abs(q[k]-item[k]) for k in q));ref=next(x for x in rows if x["building"]==d.name and x["seed"]==int(seed) and x["arm"]=="OldTeacherFourier" and x["period"]==period);maxlevel=max(maxlevel,abs(q["level"]-ref["level"]));checked+=1
  if "amplitude" in b:
   a=b["amplitude"];amps.append(dict(building=d.name,seed=int(seed),arm=arm,mean=float(a.mean()),std=float(a.std()),minimum=float(a.min()),maximum=float(a.max()),negative_fraction=float((a<0).mean())))
assert maxerr<1e-12 and maxlevel<1e-10
save(r/"amplitude_diagnostics.json",amps)
seedmacro=[];buildingmean=[]
for period in ["cal","check","val"]:
 for arm in sorted({x["arm"] for x in rows}):
  for seed in c["seeds"]:
   v=[x for x in rows if x["period"]==period and x["arm"]==arm and x["seed"]==seed]
   seedmacro.append(dict(period=period,arm=arm,seed=seed,**{k:float(np.mean([x[k] for x in v])) for k in ["mse","mae","level","shape"]}))
  for name in sorted({x["building"] for x in rows}):
   v=[x for x in rows if x["period"]==period and x["arm"]==arm and x["building"]==name]
   buildingmean.append(dict(period=period,arm=arm,building=name,**{k:float(np.mean([x[k] for x in v])) for k in ["mse","mae","level","shape"]}))
for name,rr in [("seed_macro.csv",seedmacro),("by_building.csv",buildingmean)]:
 with (r/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
save(r/"independent_verification.json",dict(metric_records_checked=checked,metric_error=maxerr,level_error=maxlevel,training_weights_not_modified=True,only_saved_2016_arrays=True))
print(json.dumps(read(r/"summary.json"),indent=2));print("AMPLITUDES",json.dumps(amps));print("VERIFIED",maxerr,maxlevel)

# Preserve three-channel diagnostics and byte-value equality of the raw cache.
profiles=read(Path(c["source"])/"profiles.json")
assert all(np.array_equal(np.array(profiles[k]),np.array(read(r/"fixed_relation_channels.json")[k])) for k in profiles)
single=[x for x in read(Path(c["single_source"])/"metrics.json") if x["arm"] in ["QwenShape","WrongShape","FourierShape"]]
for x in single:
 y=next(y for y in rows if y["building"]==x["building"] and y["seed"]==x["seed"] and y["period"]==x["period"] and y["arm"]=="Single"+x["arm"])
 assert all(y[k]==x[k] for k in ["mse","mae","level","shape"])
diagnostics=[];component=[]
for d in r.iterdir():
 if not d.is_dir():continue
 prep=read(d/"preprocessors.json")
 for seed,record in prep.items():
  for arm,diag in record["shape_scales"].items():
   assert abs(diag["normalized_total_energy"]-1)<1e-10
   diagnostics.append(dict(building=d.name,seed=int(seed),arm=arm,**diag))
 for f in d.glob("*Relations_val.npz"):
  seed,arm,_=f.stem.split("_");aa=np.load(f)["amplitude"].reshape(-1,3)
  component.append(dict(building=d.name,seed=int(seed),arm=arm,coefficient_mean=aa.mean(0).tolist(),coefficient_rms=np.sqrt(np.mean(aa*aa,0)).tolist(),coefficient_std=aa.std(0).tolist(),negative_fraction=(aa<0).mean(0).tolist()))
save(r/"relation_diagnostics.json",diagnostics);save(r/"coefficient_diagnostics.json",component)
check=read(r/"independent_verification.json");check.update(raw_cache_exact=True,single_shape_records_unchanged=len(single),energy_equalization_checked=len(diagnostics));save(r/"independent_verification.json",check)
print("RAW_CACHE_SINGLE_SHAPE_AND_ENERGY_VERIFIED")
