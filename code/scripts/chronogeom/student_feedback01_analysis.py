"""Dependence-aware descriptive analysis of cached Qwen action feedback."""
from pathlib import Path
import csv,json
from collections import Counter
import numpy as np
from scripts.chronogeom.task_context02_analysis import read,save,csvsave
def main():
 c=read("docs/chronogeom/student_feedback01_config.json");r=Path(c["run"]);p=np.load(r/"predictions_and_candidates.npz")
 answers=read(r/"answers.json");rows=list(csv.DictReader((r/"scores_by_window.csv").open()));means=list(csv.DictReader((r/"scores_mean.csv").open()))
 error=((p["candidates"]-p["truth"][:,None])**2).mean((2,3));ae=abs(p["candidates"]-p["truth"][:,None]).mean((2,3));uniform=dict(method="UniformActionExpectation",n_windows=len(error),n_unique_origins=len(np.unique(p["origin"])),mse=float(error.mean()),mae=float(ae.mean()),level=float(next(x["level"] for x in means if x["method"]=="Student")),shape=0)
 uniform["shape"]=uniform["mse"]-uniform["level"]
 for s in c["seeds"]:uniform[f"seed{s}_mse"]=float(error[p["seed"]==s].mean())
 comparisons=[{k:q[k] for k in uniform} for q in means]+[uniform];csvsave(r/"comparison.csv",comparisons)
 quality={}
 for context in ["Task","Generic"]:
  aa=[q for q in answers if q["context"]==context];chosen=np.array([q["action"] for q in aa]);indices=np.array([q["row"] for q in aa]);chosen_error=error[indices,chosen];noop=error[indices,0];oracle=error.min(1)[indices];labels=Counter(q["label"] for q in aa)
  quality[context]=dict(action_counts={c["actions"][k]:int((chosen==k).sum()) for k in range(7)},label_counts=dict(labels),improved_windows=int((chosen_error<noop-1e-12).sum()),harmed_windows=int((chosen_error>noop+1e-12).sum()),unchanged_windows=int((abs(chosen_error-noop)<=1e-12).sum()),oracle_action_matches=int((chosen==error.argmin(1)[indices]).sum()),mean_regret_to_oracle=float((chosen_error-oracle).mean()),unconstrained_top_is_action_letter=sum(q["unconstrained_top_token"] in list("ABCDEFG") for q in aa),n_seed_windows=len(aa))
 quality["contexts_same_action"]=sum(next(q["action"] for q in answers if q["row"]==i and q["context"]=="Task")==next(q["action"] for q in answers if q["row"]==i and q["context"]=="Generic") for i in range(len(error)))
 quality["caveat"]="48 seed-window rows share16time origins and overlapping96horizons; descriptive counts are not48independent trials. No sampling repetitions or new trajectories."
 save(r/"feedback_quality.json",quality)
 cases=list(csv.DictReader((r/"all_cases.csv").open()))
 contributions=[]
 for action in c["actions"]:
  qq=[x for x in cases if x["qwen_task_action"]==action]
  contributions.append(dict(action=action,n_seed_windows=len(qq),contribution_to_mean_mse_difference=sum(float(x["qwen_task_mse"])-float(x["student_mse"]) for x in qq)/len(cases)))
 csvsave(r/"action_contributions.csv",contributions)
 assert np.isfinite(p["candidates"]).all() and np.isfinite(p["truth"]).all()
 max_mean_error=float(abs(p["candidates"].mean(2)-p["candidates"][:,1].mean(1)[:,None]).max())
 assert max_mean_error<1e-12 and len(answers)==96 and len(np.unique(p["origin"]))==16
 from exp.exp_chronogeom_temporal_readout06 import sha
 source=read(r/"source.json")
 assert all(sha(path)==expected for path,expected in source["student_and_base_shas"].items())
 assert np.isclose(sum(x["contribution_to_mean_mse_difference"] for x in contributions),float(next(x["mse"] for x in means if x["method"]=="QwenTask"))-float(next(x["mse"] for x in means if x["method"]=="Student")))
 save(r/"analysis_verification.json",dict(finite=True,n_prompts=len(answers),n_seed_windows=len(error),unique_origins=16,max_mean_error=max_mean_error,old_models_unchanged=True,action_decomposition_exact=True))
 start=int((np.datetime64(c["validation"][0],"h")-np.datetime64(c["base"],"h"))/np.timedelta64(1,"h"))
 end=int((np.datetime64(c["validation"][1],"h")-np.datetime64(c["base"],"h"))/np.timedelta64(1,"h"))
 n_days=len(np.arange(start,end-96+1,24));origins=sorted(set(int(q["origin"]) for q in rows));rng=np.random.default_rng(20241029)
 draws=(rng.integers(n_days,size=(3000,int(np.ceil(n_days/7))))[:,:,None]+np.arange(7))%n_days;draws=draws.reshape(3000,-1)[:,:n_days];ci=[]
 for other in ["Student","FrozenBase","TrainGlobal","SeasonalRule","QwenGeneric"]:
  array=np.full(n_days,np.nan)
  for o in origins:
   a=np.mean([float(q["mse"]) for q in rows if q["method"]=="QwenTask" and int(q["origin"])==o]);b=np.mean([float(q["mse"]) for q in rows if q["method"]==other and int(q["origin"])==o]);array[(o-start)//24]=a-b
  vals=array[draws];den=np.isfinite(vals).sum(1);boot=np.nansum(vals,1)/np.maximum(den,1);boot=boot[den>0];lo,hi=np.quantile(boot,[.025,.975])
  ci.append(dict(comparison="QwenTask-"+other,mse_difference=float(np.nanmean(array)),lo95=float(lo),hi95=float(hi),block_days=7,replicates=len(boot),unique_origins=len(origins),seed_handling="paired three-seed average before time bootstrap"))
 csvsave(r/"paired_block_intervals.csv",ci);print(json.dumps(dict(quality=quality,intervals=ci),indent=2))
if __name__=="__main__":main()
