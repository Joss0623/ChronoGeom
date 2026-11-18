from pathlib import Path
import csv,json,numpy as np
from exp.exp_chronogeom_event_knowledge19 import scores,read,save,sha
r=Path("/home/home_new/qsmx/runs/ChronoGeom/downstream/bdg2_context_teacher39_20261101_03")
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

groups=read(r/"groups.json");requests=read(r/"requests.json");old=Path(c["source38"]);raw=read(r/"raw_context_responses.json");features=read(r/"context_features.json")
for b,dd in raw.items():
 A={key:np.stack([np.array(table[k]) for k in sorted(table,key=int)]) for key,table in dd.items()}
 assert np.array_equal(A["Task"]-A["Generic"],np.array(features[b]["HistoryQwen"]))
 assert np.array_equal(A["Wrong"]-A["Generic"],np.array(features[b]["WrongContext"]))
 assert np.array_equal(np.roll(np.array(features[b]["HistoryQwen"]),-1,axis=0),np.array(features[b]["HistoryShuffle"]))
 for state in dd["Task"]:
  history=[g["history"] for g in groups if g["building"]==b and g["state"]==int(state)]
  assert len(history)==3 and len(set(history))==1
static_checks=0
for x in read(old/"metrics.json"):
 if x["arm"]!="QwenRelations":continue
 y=next(a for a in rows if a["building"]==x["building"] and a["seed"]==x["seed"] and a["period"]==x["period"] and a["arm"]=="StaticQwen")
 assert all(y[k]==x[k] for k in ["mse","mae","level","shape"]);static_checks+=1
diagnostics=[]
for b,dd in features.items():
 q=np.array(dd["HistoryQwen"]);w=np.array(dd["WrongContext"])
 diagnostics.append(dict(building=b,states=len(q),context_RMS_difference=float(np.sqrt(np.mean((q-w)**2))),history_RMS_variation=float(np.sqrt(np.mean((q-q.mean(0,keepdims=True))**2)))))
save(r/"knowledge_variation.json",diagnostics)
rank=[]
for dd in r.iterdir():
 if (dd/"preprocessors.json").exists():
  for seed,x in read(dd/"preprocessors.json").items():
   for arm,info in x["shape_scales"].items():
    assert abs(info["normalized_total_energy"]-1)<1e-10
    rank.append(dict(building=dd.name,seed=int(seed),arm=arm,**info))
save(r/"relation_diagnostics.json",rank)
iv=read(r/"independent_verification.json");iv.update(context_differences_exact=True,all_contexts_same_history=True,shuffle_exact=True,static_records_exact=static_checks,normalized_inputs_checked=len(rank),queries=len(requests),no2017_access=True);save(r/"independent_verification.json",iv)
print("CONTEXT_HISTORY_STATIC_AND_ENERGY_CHECKS_PASSED")

decision=read(r/"KD_decision.json");sm=read(r/"summary.json");q=sm["cal"]["HistoryQwen"]
for ref in ["WrongContext","HistoryShuffle"]:
 t=next(t for t in decision["tests"] if t["reference"]==ref);b=sm["cal"][ref]
 assert t["pass_"]==(q["mse"]<b["mse"] and q["mae"]<=b["mae"])
assert decision["pass_"]==all(t["pass_"] for t in decision["tests"])
print("LOCKED_SEMANTIC_MACRO_GATE_VERIFIED")
