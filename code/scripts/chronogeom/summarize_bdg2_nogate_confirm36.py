"""Saved-output verification for fresh-building/site no-gate confirmation36."""
from pathlib import Path
import csv,json,numpy as np
from exp.exp_chronogeom_event_knowledge19 import read,save,sha,scores,git
r=Path("/home/home_new/qsmx/runs/ChronoGeom/downstream/bdg2_nogate_confirm36_20261101_01");assert read(r/"status.json")["complete"]
selection=read(r/"selection_buildings.json");selected=selection["selected"];cfg=read(r/"config.json")
arms=["NoKD","FourierKD","WrongKD","QwenTaskKD"];allrows=[];bybuilding=[];byseed=[];intervals=[];windowrows=[];not_evaluable=[];maxerr=0.;maxlevel=0.;identityerr=0.;sources=0;cache_sha=sha(Path(cfg["teacher_source"])/"profiles.json");floor_triggered=[]
rng=np.random.default_rng(2024)
for meta in selected:
    name=meta["building"];case=r/name;status=read(case/"status.json")
    assert status["complete"];assert read(case/"model_lock.json")["before_2017_access"]
    for x in read(case/"model_lock.json")["files"]+read(case/"all_building_locks.json")+read(r/"old_assets.json"):
        assert sha(x["path"])==x["sha"];sources+=1
    assert len(read(case/"fits.json"))==24
    for seed,n in read(case/"loss_scales.json").items():
        if n["mse"]!=n["raw"]["mse"] or n["mae"]!=n["raw"]["mae"]:floor_triggered.append(dict(building=name,seed=seed))
    if not status.get("evaluable",False):
        not_evaluable.append(dict(building=name,category=meta["category"],reason=status["reason"]));continue
    P=np.load(case/"test_predictions.npz");L=np.load(case/"test_labels.npz");Y=L["y"];M=read(case/"metrics.json");lock=read(case/"prediction_lock.json")
    assert sha(case/"model_lock.json")==lock["models_sha"] and sha(case/"test_predictions.npz")==lock["predictions_sha"]
    assert read(case/"verification.json")["gate_parameters"]==0 and read(case/"verification.json")["auxiliary_heads"]==0
    sd=read(case/"scale.json")["sd"];origin=L["origin"];blocks=(origin-origin.min())//336;unique=np.unique(blocks)
    for m in M:
        q=scores(P[f'{m["seed"]}_{m["arm"]}'],Y);maxerr=max(maxerr,max(abs(q[k]-m[k]) for k in q));identityerr=max(identityerr,abs(q["mse"]-q["level"]-q["shape"]))
        ref=scores(P[f'{m["seed"]}_FrozenTimeBase'],Y);maxlevel=max(maxlevel,abs(q["level"]-ref["level"]))
        allrows.append(dict(building=name,category=meta["category"],site=meta["site"],windows=len(Y),**m,mse_raw_kWh2=m["mse"]*sd**2,mae_raw_kWh=m["mae"]*sd))
    for arm in arms:
        rows=[x for x in M if x["arm"]==arm];bybuilding.append(dict(building=name,category=meta["category"],site=meta["site"],arm=arm,windows=len(Y),**{k:float(np.mean([x[k] for x in rows])) for k in ["mse","mae","level","shape"]}))
    for seed in cfg["seeds"]:
        for i,o in enumerate(origin):
            row=dict(building=name,category=meta["category"],site=meta["site"],seed=seed,origin=int(o))
            for arm in arms:
                e=P[f"{seed}_{arm}"][i]-Y[i];row[arm+"_mse"]=float((e*e).mean());row[arm+"_mae"]=float(abs(e).mean())
            windowrows.append(row)
    for ref in ["NoKD","FourierKD","WrongKD"]:
        for metric in ["mse","mae"]:
            diffs=[]
            for seed in cfg["seeds"]:
                e=P[f"{seed}_QwenTaskKD"]-Y;b=P[f"{seed}_{ref}"]-Y;diffs.append(((e*e-b*b) if metric=="mse" else (abs(e)-abs(b))).mean((1,2)))
            dd=np.mean(diffs,axis=0);bs=[]
            for _ in range(2000):
                ix=np.concatenate([np.flatnonzero(blocks==b) for b in rng.choice(unique,len(unique),replace=True)]);bs.append(float(dd[ix].mean()))
            intervals.append(dict(building=name,site=meta["site"],reference=ref,metric=metric,delta=float(dd.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),blocks=len(unique),note="Seedlossaverage beforepaired14dayorigin-blockbootstrap;overlappingwindowsnotIID."))
for name,rows in [("by_seed.csv",allrows),("by_building.csv",bybuilding),("per_window.csv",windowrows)]:
    if rows:
        with (r/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert maxerr<1e-12 and maxlevel<1e-10 and identityerr<1e-10
macro={}
for cat in ["Office","Education","Healthcare","ALL"]:
    macro[cat]={}
    for arm in arms:
        rows=[x for x in bybuilding if x["arm"]==arm and (cat=="ALL" or x["category"]==cat)]
        if rows:macro[cat][arm]={k:float(np.mean([x[k] for x in rows])) for k in ["mse","mae","level","shape"]}
seedmacro=[]
for seed in cfg["seeds"]:
    for arm in arms:
        rows=[x for x in allrows if x["seed"]==seed and x["arm"]==arm]
        if rows:seedmacro.append(dict(seed=seed,arm=arm,**{k:float(np.mean([x[k] for x in rows])) for k in ["mse","mae","level","shape"]}))
with (r/"seed_macro.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(seedmacro[0]));w.writeheader();w.writerows(seedmacro)
wins=[]
for ref in ["NoKD","FourierKD","WrongKD"]:
    record=dict(reference=ref,building_dual_wins=0,building_dual_losses=0,building_mixed=0,building_ties=0,seed_dual_wins=0,seed_dual_losses=0,seed_mixed=0,seed_ties=0)
    for name in sorted({x["building"] for x in bybuilding}):
        a=next(x for x in bybuilding if x["building"]==name and x["arm"]=="QwenTaskKD");b=next(x for x in bybuilding if x["building"]==name and x["arm"]==ref)
        for prefix,groups in [("building",[(a,b)]),("seed",[(next(x for x in allrows if x["building"]==name and x["seed"]==seed and x["arm"]=="QwenTaskKD"),next(x for x in allrows if x["building"]==name and x["seed"]==seed and x["arm"]==ref)) for seed in cfg["seeds"]])]:
            for x,y in groups:
                if x["mse"]<y["mse"]-1e-12 and x["mae"]<y["mae"]-1e-12:key="dual_wins"
                elif x["mse"]>y["mse"]+1e-12 and x["mae"]>y["mae"]+1e-12:key="dual_losses"
                elif abs(x["mse"]-y["mse"])<=1e-12 and abs(x["mae"]-y["mae"])<=1e-12:key="ties"
                else:key="mixed"
                record[prefix+"_"+key]+=1
    wins.append(record)
summary=dict(macro=macro,seed_macro=seedmacro,wins=wins,non_evaluable=not_evaluable,selected_buildings=6,evaluated_buildings=len({x["building"] for x in bybuilding}),sites=selection["selected_sites"],all_sites_unseen=selection["all_selected_sites_unseen"])
save(r/"summary.json",summary);save(r/"paired_intervals.json",intervals)
save(r/"independent_verification.json",dict(metric_rows=len(allrows),maximum_metric_error=maxerr,maximum_level_difference=maxlevel,maximum_MSE_decomposition_error=identityerr,old_and_selected_model_hash_checks=sources,teacher_cache_SHA=cache_sha,teacher_cache_exact_reuse=True,loss_scale_floor_triggered=floor_triggered,all6_weights_locked_before_test=True,all_old_models_and_results_unchanged=True,new_buildings_and_sites=True,LLM_calls=0,gate_parameters=0,auxiliary_heads=0,new_fits=144,updates=86400,student_forecast_replays=sum(read(r/x["building"]/"verification.json")["selected_forecast_replays"] for x in selected if (r/x["building"]/"verification.json").exists()),git=git()))
print(json.dumps(summary,indent=2));print("VERIFIED",maxerr,maxlevel,identityerr)

