"""Replay real feedback updates without another LLM call and describe forecast performance."""
import csv,json,time,contextlib
from pathlib import Path
import numpy as np
import torch
from exp import exp_chronogeom_live_feedback18 as m
c=m.rt.read(m.CONFIG);r=Path(c["run"]);lock=m.rt.read(r/"selection_lock.json")
assert all(m.rt.sha(p)==v for p,v in lock["files"].items())
def read(name):
 with (r/name).open() as f:return list(csv.DictReader(f))
raw=[json.loads(line) for line in (r/"live_feedback.jsonl").open()]
assert len(raw)*3==m.rt.read(r/"qwen_execution.json")["forward_sequences"]==1299
logit_error=0.
for v in raw:
 z=np.array(v["letter_logits"])
 physical=np.stack([z[j,np.argsort(perm)] for j,perm in enumerate(m.PERMS)])
 a=physical.mean(0);a-=a.max();q=np.exp(a);q/=q.sum()
 logit_error=max(logit_error,float(abs(q-np.array(v["q_physical"])).max()))
assert logit_error<1e-12
lookup={(v["phase"],v["seed"],v["step"],v["family"],v["row"]):v for v in raw if v["phase"]!="noop_check"}
def package(phase,seed):
 z=np.load(r/f"package_{phase}_s{seed}.npz");p={k:z[k] for k in z.files}
 p.update(m.rt.read(r/f"package_{phase}_s{seed}.json"));p["initial"]=m.npmodel(np.load(r/f"initial_{phase}_s{seed}.npz"))
 return p
class Replay:
 def __init__(self):self.calls=3;self.action_error=0.;self.requests=0
 def preference(self,requests):
  out=[]
  for req in requests:
   q=lookup[req["phase"],req["seed"],req["step"],req["family"],req["row"]]
   error=float(abs(np.array(req["candidate_angles"])-np.array(q["candidate_angles"])).max())
   self.action_error=max(self.action_error,error)
   assert error<1e-11 and req["texts"]==q["texts"] and req["candidate_sha256"]==q["candidate_sha256"]
   out.append(q["q_physical"]);self.requests+=1
  self.calls+=3*len(requests);return np.array(out)
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
verify=r/"cpu_replay";verify.mkdir(exist_ok=False);replay=Replay();state_error=0.;states=0
with (r/"cpu_replay_stdout.log").open("w") as f,contextlib.redirect_stdout(f):
 for phase in ["inner","full"]:
  for seed in c["seeds"]:
   m.run_phase(c,verify,phase,seed,package(phase,seed),replay)
   for arm in m.ARMS:
    for step in [0]+c["checks"]:
     a=np.load(r/f"{phase}_s{seed}_{arm}_step{step}.npz");b=np.load(verify/f"{phase}_s{seed}_{arm}_step{step}.npz")
     state_error=max(state_error,max(float(abs(a[k]-b[k]).max()) for k in a.files));states+=1
assert state_error<1e-11 and replay.calls==1299
pred=np.load(r/"predictions.npz");prediction_error=0.;predictions=0;fixed=[];fixed_outputs={};feedback=[]
for seed in c["seeds"]:
 for stage in c["stages"]:
  z=np.load(r/f"deployment_s{seed}_{stage}.npz");p=m.pack(z["b"],z["y"])
  for arm in m.ARMS:
   model=np.load(r/f"student_s{seed}_{arm}.npz");delta=m.bil.predict(model,z["x"],c)
   got=m.p13.rotate(p,delta);prediction_error=max(prediction_error,float(abs(got-pred[f"s{seed}_{stage}_{arm}"]).max()));predictions+=1
   final=np.load(r/f"full_s{seed}_{arm}_step100.npz");d=m.bil.predict(final,z["x"],c)
   pp,met,ch=m.p13.forecast_metrics(p,d);fixed_outputs[f"s{seed}_{stage}_{arm}"]=pp
   fixed.append(dict(seed=seed,stage=stage,arm=arm,**{k:float(met[k].mean()) for k in ["mse","mae","level","shape"]}))
assert prediction_error==0
m.csvsave(r/"fixed100_by_seed_period.csv",fixed);np.savez_compressed(r/"fixed100_predictions.npz",**fixed_outputs)
fixedsummary=[]
for arm in m.ARMS:
 a=[v for v in fixed if v["arm"]==arm]
 fixedsummary.append(dict(arm=arm,**{k:float(np.mean([v[k] for v in a])) for k in ["mse","mae","level","shape"]}))
m.csvsave(r/"fixed100_summary.csv",fixedsummary)
for phase in ["inner","full"]:
 for seed in c["seeds"]:
  pkg=package(phase,seed)
  usage=m.rt.read(r/f"{phase}_s{seed}_feedback_usage.json")
  for v in usage:
   probs=np.array(v["q_used"]);qp=np.array(v["q_raw"])
   if v["arm"]=="QwenShuffle":assert np.array_equal(probs,np.roll(qp,1,axis=0))
   else:assert np.array_equal(probs,qp)
   for j,row in enumerate(v["rows"]):
    p=m.pack(pkg["b"][row:row+1],pkg["y"][row:row+1]);angles=np.array(v["candidate_angles"][j])
    cand=np.stack([m.p13.rotate(p,np.array([a]))[0] for a in angles])
    mse=((cand-p["y"])**2).mean(1);mae=abs(cand-p["y"]).mean(1)
    joint=.5*(mse/pkg["scales"]["squared_error_normalizer"]+mae/pkg["scales"]["absolute_error_normalizer"])
    choice=int(probs[j].argmax());oracle=int(joint.argmin());gain=float(joint[1]-joint[choice])
    entry=dict(phase=phase,seed=seed,step=v["step"],arm=v["arm"],row=row,origin=int(pkg["origin"][row]),chosen=choice,
       oracle=oracle,correct=choice==oracle,gain=gain,benefit=max(gain,0.),harm=max(-gain,0.),
       teacher_entropy=float(-np.sum(probs[j]*np.log(np.maximum(probs[j],1e-15)))),teacher_max_probability=float(probs[j].max()))
    if v["arm"]!="Fourier":
     rr=lookup[phase,seed,v["step"],v["arm"],row];entry["order_disagreement"]=rr["order_disagreement"]
    else:entry["order_disagreement"]=False
    feedback.append(entry)
m.csvsave(r/"feedback_quality_cases.csv",feedback);quality=[]
for phase in ["inner","full"]:
 for arm in m.ARMS[1:]:
  a=[v for v in feedback if v["phase"]==phase and v["arm"]==arm]
  quality.append(dict(phase=phase,arm=arm,n_records=len(a),accuracy=float(np.mean([v["correct"] for v in a])),
    useful=sum(v["gain"]>1e-12 for v in a),harmful=sum(v["gain"]<-1e-12 for v in a),noop=sum(v["chosen"]==1 for v in a),
    mean_joint_gain=float(np.mean([v["gain"] for v in a])),mean_gross_benefit=float(np.mean([v["benefit"] for v in a])),
    mean_gross_harm=float(np.mean([v["harm"] for v in a])),mean_entropy=float(np.mean([v["teacher_entropy"] for v in a])),
    mean_max_probability=float(np.mean([v["teacher_max_probability"] for v in a])),
    order_disagreement=float(np.mean([v["order_disagreement"] for v in a]))))
m.csvsave(r/"feedback_quality.csv",quality)
matched=[]
for k,t in lookup.items():
 phase,seed,step,family,row=k
 if step==8 and family=="QwenTask":
  w=lookup[phase,seed,step,"QwenWrong",row]
  assert np.array_equal(t["candidate_angles"],w["candidate_angles"]) and t["candidate_sha256"]==w["candidate_sha256"]
  matched.append(dict(phase=phase,seed=seed,row=row,q_l1=float(abs(np.array(t["q_physical"])-np.array(w["q_physical"])).sum()),
     different_choice=int(np.argmax(t["q_physical"]))!=int(np.argmax(w["q_physical"]))))
m.rt.save(r/"matched_first_feedback_context.json",matched)
cases=read("cases.csv");lk={(v["arm"],int(v["seed"]),v["stage"],int(v["origin"])):v for v in cases}
rng=np.random.default_rng(1818);boot=[]
for control in ["FrozenFourier","NoKD","Fourier","QwenWrong","QwenShuffle"]:
 for metric in ["mse","mae"]:
  groups=[]
  for stage in c["stages"]:
   origins=sorted({int(v["origin"]) for v in cases if v["stage"]==stage})
   delta=np.array([np.mean([float(lk["QwenTask",seed,stage,o][metric])-float(lk[control,seed,stage,o][metric]) for seed in c["seeds"]]) for o in origins])
   block=(np.array(origins)-min(origins))//(14*24)
   groups.append([delta[block==j] for j in sorted(set(block))])
  reps=[np.mean([np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))]).mean() for g in groups]) for _ in range(2000)]
  boot.append(dict(control=control,metric=metric,difference=float(np.mean([np.concatenate(g).mean() for g in groups])),
      ci_low=float(np.quantile(reps,.025)),ci_high=float(np.quantile(reps,.975)),blocks_per_stage=[len(g) for g in groups],descriptive_development_only=True))
worst=sorted([v for v in cases if v["arm"]=="QwenTask"],key=lambda v:float(v["gain"]))[:5]
m.csvsave(r/"worst_cases.csv",worst)
src=m.rt.read(r/"source_train.json")
assert all(m.rt.sha(p)==v for p,v in src["sources"].items())
assert all(m.rt.sha(p)==v for p,v in lock["files"].items())
size=sum(p.stat().st_size for p in r.rglob("*") if p.is_file())/2**20;assert size<c["budget"]["max_raw_mib"]
m.rt.save(r/"analysis.json",dict(real_Qwen_sequences=1299,physical_choice_sets=len(raw),letter_logit_reconstruction_error=logit_error,
 cpu_replayed_states=states,max_state_error=state_error,replayed_current_action_sets=replay.requests,max_action_error=replay.action_error,
 numpy_forecast_replays=predictions,max_prediction_error=prediction_error,raw_mib=size,bootstrap=boot,
 all_old_sources_and_locks_unchanged=True,live_model_grads=False,TimeBase_updates=0))
print("SUMMARY",(r/"summary.csv").read_text());print("SEEDS",(r/"macro_by_seed.csv").read_text())
print("FIXED100",(r/"fixed100_summary.csv").read_text());print("FEEDBACK",json.dumps(quality))
print("SELECTION",(r/"selection.json").read_text());print("QWEN",(r/"qwen_execution.json").read_text())
print("REPLAY",json.dumps(dict(states=states,state_error=state_error,predictions=predictions,prediction_error=prediction_error,raw_mib=size)))

# Fixed-budget100 sensitivity: pre-existing final checkpoint, never used to override early stopping.
import json,csv
from pathlib import Path
import numpy as np
from exp import exp_chronogeom_live_feedback18 as live
cfg=live.rt.read(live.CONFIG);root=Path(cfg["run"])
fixed=np.load(root/"fixed100_predictions.npz");boot100=[];rng=np.random.default_rng(1818100)
for control in ["FrozenFourier","NoKD","Fourier","QwenWrong","QwenShuffle"]:
 for metric in ["mse","mae"]:
  groups=[]
  for stage in cfg["stages"]:
   vals=[];origins=None
   for seed in cfg["seeds"]:
    data=np.load(root/f"deployment_s{seed}_{stage}.npz");target=data["y"];origins=data["origin"]
    task=fixed[f"s{seed}_{stage}_QwenTask"]
    other=(np.load(root/"predictions.npz")[f"s{seed}_{stage}_FrozenFourier"] if control=="FrozenFourier" else fixed[f"s{seed}_{stage}_{control}"])
    if metric=="mse":diff=((task-target)**2).mean(1)-((other-target)**2).mean(1)
    else:diff=abs(task-target).mean(1)-abs(other-target).mean(1)
    vals.append(diff)
   d=np.mean(vals,axis=0);block=(origins-origins.min())//(14*24)
   groups.append([d[block==k] for k in sorted(set(block))])
  reps=[np.mean([np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))]).mean() for g in groups]) for _ in range(2000)]
  boot100.append(dict(control=control,metric=metric,difference=float(np.mean([np.concatenate(g).mean() for g in groups])),
      ci_low=float(np.quantile(reps,.025)),ci_high=float(np.quantile(reps,.975)),descriptive_development_only=True))
live.rt.save(root/"fixed100_uncertainty.json",boot100)
with (root/"fixed100_by_seed_period.csv").open() as f:rows=list(csv.DictReader(f))
macro=[]
for seed in cfg["seeds"]:
 for arm in live.ARMS:
  a=[v for v in rows if int(v["seed"])==seed and v["arm"]==arm]
  macro.append(dict(seed=seed,arm=arm,mse=float(np.mean([float(v["mse"]) for v in a])),mae=float(np.mean([float(v["mae"]) for v in a]))))
live.csvsave(root/"fixed100_macro_by_seed.csv",macro)
previous=Path("/home/home_new/qsmx/runs/ChronoGeom/downstream/phase_response_opd17_20261030_01");maximum=0.
for seed in cfg["seeds"]:
 now=np.load(root/f"full_s{seed}_NoKD_step100.npz");old=np.load(previous/f"student_s{seed}_NoKD.npz")
 maximum=max(maximum,max(float(abs(now[k]-old[k]).max()) for k in now.files))
live.rt.save(root/"NoKD100_parity17.json",dict(max_parameter_difference=maximum,threshold=1e-11,passed=maximum<1e-11))
assert maximum<1e-11
print("FIXED100_BOOTSTRAP",json.dumps(boot100))
print("NOKD_PARITY17",maximum)

# Additional saved-data diagnostics; no parameter selection or new Qwen inference.
from pathlib import Path
import json,csv,numpy as np
from exp import exp_chronogeom_live_feedback18 as live
cfg=live.rt.read(live.CONFIG);root=Path(cfg["run"])
raw=[json.loads(v) for v in (root/"live_feedback.jsonl").read_text().splitlines()]
counts={}
for arm in ["QwenTask","QwenWrong","QwenShuffle"]:
 a=[v for v in raw if v["family"]==arm and v["phase"]!="noop_check"]
 counts[arm]=np.bincount(np.array([v["letter_logits"] for v in a]).argmax(2).reshape(-1),minlength=3).tolist()
live.rt.save(root/"raw_letter_counts.json",counts)
with (root/"inner_curves.csv").open() as f:curves=list(csv.DictReader(f))
diag={}
for arm in live.ARMS:
 grid=[dict(step=step,joint=float(np.mean([float(v["joint"]) for v in curves if v["arm"]==arm and int(v["step"])==step]))) for step in [0,25,50,75,100]]
 diag[arm]=dict(best_internal_step=min(grid,key=lambda v:(v["joint"],v["step"]))["step"],grid=grid,diagnostic_not_main_selection=True)
live.rt.save(root/"internal_arm_diagnostics.json",diag)
fp=np.load(root/"fixed100_predictions.npz");baseline=np.load(root/"predictions.npz");cases=[]
for seed in cfg["seeds"]:
 for stage in cfg["stages"]:
  data=np.load(root/f"deployment_s{seed}_{stage}.npz");y=data["y"]
  task=fp[f"s{seed}_{stage}_QwenTask"];none=fp[f"s{seed}_{stage}_NoKD"];frozen=baseline[f"s{seed}_{stage}_FrozenFourier"]
  em=((task-y)**2).mean(1);en=((none-y)**2).mean(1);ef=((frozen-y)**2).mean(1)
  am=abs(task-y).mean(1);an=abs(none-y).mean(1)
  for j,o in enumerate(data["origin"]):
   cases.append(dict(seed=seed,stage=stage,origin=int(o),task_mse=float(em[j]),NoKD_mse=float(en[j]),FrozenFourier_mse=float(ef[j]),
       task_minus_NoKD_mse=float(em[j]-en[j]),task_mae=float(am[j]),NoKD_mae=float(an[j]),task_minus_NoKD_mae=float(am[j]-an[j])))
live.csvsave(root/"fixed100_paired_cases.csv",cases)
live.csvsave(root/"fixed100_worst_cases.csv",sorted(cases,key=lambda v:v["task_minus_NoKD_mse"],reverse=True)[:8])
print("SAVED_DIAGNOSTICS",json.dumps(dict(raw_letters=counts,internal_best_steps={k:v["best_internal_step"] for k,v in diag.items()},fixed100_cases=len(cases))))
