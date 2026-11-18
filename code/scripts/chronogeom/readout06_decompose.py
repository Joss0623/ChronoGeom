from pathlib import Path
import json,csv,numpy as np
r=Path("/home/home_new/qsmx/runs/ChronoGeom/downstream/temporal_readout06_20261028_01");r.mkdir(exist_ok=True)
source=Path("/home/home_new/qsmx/runs/ChronoGeom/downstream/temporal_kd05_20261028_01");config=json.loads((source/"config.json").read_text());rows=[];windows=[]
for domain,dc in config["domains"].items():
 for stage in ["development_validation"]+(["matched_previous","new_confirmation"] if dc["test"] else []):
  p=np.load(source/domain/stage/"predictions.npz");y=p["truth"].astype("float64");origin=p["origin"]
  for model in sorted(set(k.rsplit("_s",1)[0] for k in p.files if "_s" in k)):
   ee=[];ll=[];ss=[];ma=[]
   for seed in config["seeds"]:
    pred=p[model+f"_s{seed}"].astype("float64");e=pred-y;mean=e.mean(1,keepdims=True);level=(mean*mean).mean((1,2));shape=((e-mean)**2).mean((1,2));mse=(e*e).mean((1,2));mae=abs(e).mean((1,2))
    assert np.max(abs(mse-level-shape))<1e-12
    ee.append(mse);ll.append(level);ss.append(shape);ma.append(mae)
    for i,o in enumerate(origin):
     windows.append(dict(domain=domain,stage=stage,model=model,seed=seed,origin=int(o),date=str(np.datetime64(dc["base"],"h")+np.timedelta64(int(o),"h")),mse=float(mse[i]),mae=float(mae[i]),level_error=float(level[i]),shape_error=float(shape[i]),window_mean_error=float(mean[i].mean())))
   ee,ll,ss,ma=map(np.stack,[ee,ll,ss,ma]);row=dict(domain=domain,stage=stage,model=model,mse=float(ee.mean()),mae=float(ma.mean()),level_error=float(ll.mean()),shape_error=float(ss.mean()),n_origins=len(origin))
   for j,seed in enumerate(config["seeds"]):row.update({f"seed{seed}_mse":float(ee[j].mean()),f"seed{seed}_level":float(ll[j].mean()),f"seed{seed}_shape":float(ss[j].mean())})
   rows.append(row)
for name,data in [("kd05_error_decomposition.csv",rows),("kd05_error_decomposition_windows.csv",windows)]:
 with (r/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
for domain,stage in [("capital","development_validation"),("power","development_validation"),("power","matched_previous"),("power","new_confirmation")]:
 print(domain,stage)
 for q in rows:
  if q["domain"]==domain and q["stage"]==stage and q["model"] in ["FrozenBase","NoKD","Teacher__TaskKD","Teacher__GenericKD","Teacher__ShuffleKD","Teacher__FourierKD","FeatureOnly__TaskKD","OutputOnly__TaskKD","Joint__TaskKD"]:
   print(q["model"],"MSE",round(q["mse"],6),"Level",round(q["level_error"],6),"Shape",round(q["shape_error"],6),"MAE",round(q["mae"],6))
print("All decompositions verified; no training performed")
