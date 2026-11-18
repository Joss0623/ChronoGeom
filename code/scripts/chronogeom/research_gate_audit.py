"""Read-only Research Gate summaries; never fit a model or modify prior runs."""
import argparse,csv,hashlib,json,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[2]
prior=Path("/home/home_new/qsmx/runs/ChronoGeom/phase1b/phase1b_20261024_01")
files=[prior/"train/metrics.csv",prior/"train/learning_curves.csv",prior/"test/metrics.csv",repo/"data_provider/chronogeom_identifiability.py",repo/"exp/exp_chronogeom_identifiability.py"]
files+=list((repo/"docs/chronogeom").glob("PHASE1*.md"))
files+=[repo/"docs/chronogeom/CHRONOGEOM_RESEARCH_DECISION.md",repo/"docs/chronogeom/REPRESENTATION_PROPERTIES_PROTOCOL.md"]
manifest={str(f):{"bytes":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in files}
rows=[]
for stage in ["train","test"]:
 with (prior/stage/"metrics.csv").open() as f:
  for r in csv.DictReader(f):
   if r["n_train"]=="256" and r["energy"]=="rms" and r["task"] in "BCD":
    if r["encoding"] in ["none","raw_timestamp","sincos","fourier","helix","ttw_phase"]:
     if r["cohort"] in ["train","val","iid","phase","new_system"]:rows.append(r)
groups={}
for r in rows:
 k=tuple(r[c] for c in ["task","encoding","decoder","cohort"]);groups.setdefault(k,[]).append(r)
summary=[]
for k,v in sorted(groups.items()):
 r=dict(zip(["task","encoding","decoder","cohort"],k));r["fit_seeds"]=len(v)
 for col in ["mse","clean_mse"]:
  x=[float(z[col]) for z in v];r[col+"_mean"]=statistics.mean(x);r[col+"_seed_sd"]=statistics.stdev(x) if len(x)>1 else None
 summary.append(r)
f=out/"prior_metrics_summary.csv"
with f.open("w") as h:
 w=csv.DictWriter(h,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
with (prior/"train/learning_curves.csv").open() as f:
 curves=[r for r in csv.DictReader(f) if r["task"]=="C" and r["n_train"]=="256" and r["energy"]=="rms" and r["encoding"] in ["none","sincos"]]
with (out/"c_learning_curves.csv").open("w") as f:
 w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
info={"base_commit":"f7e3cb1","audit_source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),"audit_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"kind":"read_only_existing_results","new_fits":0,"new_predictions":0,"new_random_seeds":0,"aggregation":"fixed seed mean; seed SD is not a confidence interval","input_files":manifest,"rows_in_selected_summary":len(rows),"groups":len(summary)}
(out/"prior_audit_manifest.json").write_text(json.dumps(info,indent=2))
for r in summary:
 if r["task"]=="C" and r["encoding"] in ["none","sincos"] and r["decoder"] in ["linear","mlp"] and r["cohort"] in ["train","val","iid"]:print(r)
