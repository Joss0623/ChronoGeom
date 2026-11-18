"""Frozen Qwen feedback on current TimeBase curves; not autoregressive OPD."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from exp import exp_chronogeom_temporal_readout06 as rt
from scripts.chronogeom.task_context02_analysis import read,save,csvsave
def center(x):return x-x.mean(1,keepdims=True)
def candidates(base,pred):
 mean=base.mean(1,keepdims=True);n=pred.shape[1];t=np.arange(n)
 earlier=pred[:,np.minimum(t+3,n-1)]
 later=pred[:,np.maximum(t-3,0)]
 curves=[pred,base,base+.5*(pred-base),mean+.75*center(pred),mean+1.25*center(pred),earlier,later]
 return np.stack([mean+center(v) for v in curves],axis=1)
def arraytext(x):return "["+",".join(f"{v:.3f}" for v in x[:,0])+"]"
def prompt(row,context,order,actions):
 domain=("The series is hourly household electricity active power from one home in Sceaux, France. Use household daily/weekly activity knowledge cautiously; occupancy, appliances and future weather are unknown." if context=="Task" else "The series is an hourly numerical demand time series. Use its observed temporal patterns.")
 definitions=["keep the student forecast","use the original TimeBase forecast","use the midpoint of student and TimeBase","reduce the student's centered amplitude to75%","increase the student's centered amplitude to125%","move the student's peaks earlier by3hours (edge-clamped shift)","move the student's peaks later by3hours (edge-clamped shift)"]
 choices="\n".join(f"{chr(65+i)}: {definitions[a]}" for i,a in enumerate(order))
 return (domain+"\nYou are reviewing a lightweight model's CURRENT forecast, not generating a separate forecast. Select exactly one action expected to minimize future96-hour squared error. Every action preserves the original TimeBase96-hour forecast mean. All values use the same fixed historical-training standardization; negatives are valid. Future observed values are unavailable. Prefer evidence from the numerical history to unsupported assumptions.\n"
 +f"Forecast start: {row['date']}; weekday={row['weekday']}; hour={row['hour']}. Arrays are ordered hourly.\n"
 +"Observed previous96hours: "+arraytext(row["x"])+"\n"
 +"Original TimeBase next96 forecast: "+arraytext(row["base"])+"\n"
 +"Current student next96 forecast: "+arraytext(row["student"])+"\n"
 +"Actions (mean is re-centered after every transformation):\n"+choices+"\nReply with ONLY the single action letter A,B,C,D,E,F or G.")
def run(c,device):
 r=Path(c["run"]);r.mkdir(exist_ok=False,parents=True);tick=time.monotonic();save(r/"config.json",c)
 rt.setup(device,2);rr,oc,tr,tc,gc=rt.context(c);b=np.load(rr/"power/known_data.npz");train=rt.old.packed(b,b["train_origins"])
 p,counts=rt.windows(b["values"],b["ids"],rt.ix(c,c["validation"][0]),rt.ix(c,c["validation"][1]))
 assert len(p["x"])==16
 src=Path(c["student_run"]);tabs=np.load(src/"tables.npz");rows=[];trainrefs=[];model_shas={}
 for seed in c["seeds"]:
  base,brec=rt.frozen_base(tc,tr,"power",seed,device);rec=next(q for q in read(src/f"kd_seed{seed}/records.json") if q["name"]=="Shape_Task");model_shas[rec["path"]]=rt.sha(rec["path"]);model_shas[brec["path"]]=rt.sha(brec["path"])
  m=rt.TimeReadout(base,tabs[rec["table"]],seed,"level").to(device);m.load_state_dict(torch.load(rec["path"],map_location=device,weights_only=True));m.eval()
  raw=rt.raw_predict(base,train,device);sp=rt.forecast(m,raw,train["ids"],device)
  with torch.no_grad():bp=m.baseline(torch.tensor(raw,device=device)).double().cpu().numpy()
  cand=candidates(bp,sp);errs=((cand-train["y"][:,None])**2).mean((2,3));fixed=int(errs.mean(0).argmin());trainrefs.append(dict(seed=seed,best_action=fixed,mean_training_mse=errs.mean(0).tolist(),n_origins=len(train["x"])))
  raw=rt.raw_predict(base,p,device);sp=rt.forecast(m,raw,p["ids"],device)
  with torch.no_grad():bp=m.baseline(torch.tensor(raw,device=device)).double().cpu().numpy()
  cand=candidates(bp,sp);naive=np.tile(p['x'][:,-24:],(1,4,1))
  # Center candidate curves over their horizon, not action dimension.
  rule=((cand-cand.mean(2,keepdims=True)-center(naive)[:,None])**2).mean((2,3)).argmin(1)
  for i,o in enumerate(p["origin"]):
   dt=str(np.datetime64(c["base"],"h")+np.timedelta64(int(o),"h"));state=int(p["ids"][i,0])
   rows.append(dict(seed=seed,origin=int(o),date=dt,weekday=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][state//24],hour=state%24,x=p["x"][i],base=bp[i],student=sp[i],candidates=cand[i],truth=p["y"][i],global_action=fixed,rule_action=int(rule[i])))
  del m,base
 torch.cuda.empty_cache()
 save(r/"source.json",dict(commit=rt.git(),source_sha=rt.sha(__file__),config_sha=rt.sha("docs/chronogeom/student_feedback01_config.json"),student_and_base_shas=model_shas,all_validation_previously_viewed=True,student_training_this_stage=False))
 save(r/"train_action_reference.json",trainrefs)
 rng=np.random.default_rng(20241029);orders=[rng.permutation(7).tolist() for _ in rows];requests=[]
 for i,row in enumerate(rows):
  for context in ["Generic","Task"]:
   requests.append(dict(row=i,context=context,order=orders[i],prompt=prompt(row,context,orders[i],c["actions"])))
 save(r/"prompts.json",requests)
 from transformers import AutoTokenizer,AutoModelForCausalLM
 free,_=torch.cuda.mem_get_info(device);assert free>=22*2**30
 rt.setup(device,22);torch.manual_seed(2024);llmtick=time.monotonic()
 tok=AutoTokenizer.from_pretrained(c["model_path"],local_files_only=True);tok.padding_side="left"
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 letterids=[tok.encode(chr(65+i),add_special_tokens=False) for i in range(7)];assert all(len(x)==1 for x in letterids);letterids=[x[0] for x in letterids]
 chats=[tok.apply_chat_template([{"role":"user","content":q["prompt"]}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for q in requests]
 assert all("<think>\n\n</think>" in s for s in chats)
 lengths=[len(tok.encode(s,add_special_tokens=False)) for s in chats];assert max(lengths)<=c["prompt_max_tokens"]
 save(r/"prompt_lengths.json",lengths);print("LOAD_QWEN",len(chats),max(lengths),flush=True)
 llm=AutoModelForCausalLM.from_pretrained(c["model_path"],dtype=torch.bfloat16,device_map={"":device},local_files_only=True,attn_implementation="sdpa");llm.eval()
 for q in llm.parameters():q.requires_grad_(False)
 answers=[]
 with torch.inference_mode():
  for j in range(0,len(chats),c["batch_size"]):
   if time.monotonic()-llmtick>c["teacher_cap_seconds"]:raise TimeoutError("teacher time cap")
   if time.monotonic()-tick>c["workflow_cap_seconds"]:raise TimeoutError("workflow time cap")
   batch=tok(chats[j:j+c["batch_size"]],padding=True,add_special_tokens=False,return_tensors="pt").to(device)
   logits=llm(**batch,use_cache=False,logits_to_keep=1).logits[:,-1].float();subset=logits[:,letterids];probs=subset.softmax(-1)
   for k in range(len(subset)):
    req=requests[j+k];label=int(subset[k].argmax());ans=dict(request=j+k,row=req["row"],context=req["context"],label=chr(65+label),action=req["order"][label],action_name=c["actions"][req["order"][label]],action_logits=subset[k].cpu().tolist(),action_probabilities=probs[k].cpu().tolist(),unconstrained_top_token=tok.decode([int(logits[k].argmax())]))
    answers.append(ans)
   save(r/"answers.json",answers)
   if j%16==0:print("FEEDBACK",j+len(subset),len(chats),round(time.monotonic()-llmtick,2),flush=True)
 llmseconds=time.monotonic()-llmtick;del llm;torch.cuda.empty_cache()
 by={(a["row"],a["context"]):a["action"] for a in answers};scores=[];preds={};examples=[];maxmean=0
 for i,row in enumerate(rows):
  choices=dict(Student=0,FrozenBase=1,TrainGlobal=row["global_action"],SeasonalRule=row["rule_action"],QwenGeneric=by[i,"Generic"],QwenTask=by[i,"Task"],Oracle=int(((row["candidates"]-row["truth"])**2).mean((1,2)).argmin()))
  for name,a in choices.items():
   pp=row["candidates"][a];mse,level,shape,mae=rt.components(pp[None],row["truth"][None]);maxmean=max(maxmean,float(abs(pp.mean()-row["base"].mean())))
   scores.append(dict(seed=row["seed"],origin=row["origin"],date=row["date"],method=name,action=c["actions"][a],mse=float(mse[0]),mae=float(mae[0]),level=float(level[0]),shape=float(shape[0])))
  examples.append(dict(seed=row["seed"],origin=row["origin"],date=row["date"],qwen_task_action=c["actions"][choices["QwenTask"]],oracle_action=c["actions"][choices["Oracle"]],student_mse=float(((row["student"]-row["truth"])**2).mean()),qwen_task_mse=float(((row["candidates"][choices["QwenTask"]]-row["truth"])**2).mean())))
 csvsave(r/"scores_by_window.csv",scores);csvsave(r/"all_cases.csv",examples)
 means=[]
 for name in choices:
  qq=[q for q in scores if q["method"]==name];out=dict(method=name,n_windows=len(qq),n_unique_origins=len(p["x"]))
  out.update({k:float(np.mean([q[k] for q in qq])) for k in ["mse","mae","level","shape"]})
  out.update({f"seed{s}_mse":float(np.mean([q["mse"] for q in qq if q["seed"]==s])) for s in c["seeds"]});means.append(out)
 csvsave(r/"scores_mean.csv",means)
 a={x["method"]:x for x in means};passed=a["QwenTask"]["mse"]<.99*a["Student"]["mse"] and a["QwenTask"]["mse"]<.99*a["TrainGlobal"]["mse"] and a["QwenTask"]["mse"]<=a["FrozenBase"]["mse"] and sum(a["QwenTask"][f"seed{s}_mse"]<a["Student"][f"seed{s}_mse"] for s in c["seeds"])>=2
 save(r/"feedback_gate.json",dict(passed=bool(passed),student_fits=0,rule=c["gate"]))
 np.savez_compressed(r/"predictions_and_candidates.npz",truth=np.stack([q["truth"] for q in rows]),candidates=np.stack([q["candidates"] for q in rows]),seed=np.array([q["seed"] for q in rows]),origin=np.array([q["origin"] for q in rows]))
 assert maxmean<1e-12 and all(rt.sha(path)==h for path,h in model_shas.items())
 completion=dict(seconds=time.monotonic()-tick,llm_seconds=llmseconds,llm_prompts=len(answers),new_student_fits=0,model_training=False,nonthinking=True,model_revision=c["model_revision"],max_mean_error=maxmean,old_models_unchanged=True,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30,prompt_max_tokens=max(lengths),passed_feedback_gate=bool(passed))
 save(r/"completion.json",completion);print(json.dumps(dict(completion=completion,scores=means),indent=2),flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda:3");a=ap.parse_args();c=read("docs/chronogeom/student_feedback01_config.json")
 try:run(c,a.device)
 except BaseException:
  r=Path(c["run"])
  if r.exists():save(r/"failure.json",dict(traceback=traceback.format_exc()))
  raise
