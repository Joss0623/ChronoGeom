"""Teacher-only pairwise temporal preference task, frozen evaluation and optional approved LoRA."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import minimize
from scipy.special import expit
from exp import exp_chronogeom_ordinal_opd01 as old
def ct(a):return a-a.mean(axis=-1,keepdims=True)
def candidate(base,kind):
 x=ct(base);i=np.arange(96)
 if kind=="early3":x=x[np.minimum(i+3,95)]
 elif kind=="late3":x=x[np.maximum(i-3,0)]
 elif kind=="amplitude065":x=.65*x
 elif kind=="amplitude135":x=1.35*x
 elif kind=="flat":x=np.zeros_like(x)
 return ct(x)
def vector(a):return "["+",".join(f"{v:.2f}" for v in a)+"]"
def text(x,base,alt,state,swap):
 a,b=(alt,base) if swap else (base,alt)
 day=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][state//24]
 return ("Compare two candidate SHAPES for hourly household electricity demand from one home in Sceaux, France. Occupancy, appliance schedules and future weather are unknown. "
 "Choose the curve expected to have smaller mean squared error over the next96hours after removing each curve's96-hour mean. Both candidates are already zero-mean. Their absolute demand level is handled by the original forecaster. "
 "All values use fixed historical-training standardized power units; negatives are valid. Large and small amplitudes are both possible: use history and calendar rather than a default preference for flatness. "
 "Future observations are unavailable.\n"
 f"Forecast starts {day} {state%24:02d}:00. Following calendar hours advance normally. Arrays are hourly and ordered from earliest to latest.\n"
 "Observed previous96hours: "+vector(x)+"\n"
 "Candidate A next96hour relative shape: "+vector(a)+"\n"
 "Candidate B next96hour relative shape: "+vector(b)+"\n"
 "Which candidate has the more accurate future shape? Reply with only A or B.")
def prepare(c,r):
 p=np.load(c["data"]);base=np.load(c["base_prediction"])["p0"][:,:,0];sp=old.read(c["split"]);pairs=[]
 for part,ixs in [("fit",sp["fit_indices"]),("check",sp["check_indices"])]:
  for j,i in enumerate(ixs):
   types=c["transforms"] if part=="fit" else [c["transforms"][j%5]]
   for kind in types:
    a=ct(base[i]);b=candidate(base[i],kind);truth=ct(p["y"][i,:,0]);ea=float(np.mean((a-truth)**2));eb=float(np.mean((b-truth)**2))
    x=p["x"][i,:,0];state=int(p["ids"][i,0]);last24=ct(np.tile(x[-24:],4))
    pairs.append(dict(part=part,index=int(i),origin=int(p["origin"][i]),kind=kind,x=x,base=a,alt=b,truth=truth,base_error=ea,alt_error=eb,alt_better=eb<ea,offset=float(base[i].mean()-p["y"][i,:,0].mean()),level_error=float((base[i].mean()-p["y"][i,:,0].mean())**2),rule_alt=bool(np.mean((b-last24)**2)<np.mean((a-last24)**2)),state=state))
 requests=[]
 for j,q in enumerate(pairs):
  for swap in [False,True]:
   label=(not q["alt_better"]) if swap else q["alt_better"]
   requests.append(dict(pair=j,swap=swap,label=int(label),part=q["part"],text=text(q["x"],q["base"],q["alt"],q["state"],swap)))
 old.save(r/"requests.json",requests)
 np.savez_compressed(r/"pairs.npz",base=np.stack([q["base"] for q in pairs]),alt=np.stack([q["alt"] for q in pairs]),truth=np.stack([q["truth"] for q in pairs]),x=np.stack([q["x"] for q in pairs]),index=np.array([q["index"] for q in pairs]),origin=np.array([q["origin"] for q in pairs]),fit=np.array([q["part"]=="fit" for q in pairs]))
 old.save(r/"pair_labels.json",[{k:v for k,v in q.items() if k not in ["x","base","alt","truth"]} for q in pairs])
 # Strong supervised linear reference on same rounded observations/calendar; antisymmetric candidate difference.
 x=np.stack([np.r_[np.round(q["x"],2),old.old.fourier(np.array([[q["state"]]]))[0,0],np.round(q["base"],2),np.round(q["alt"],2),np.mean(q["alt"]**2)-np.mean(q["base"]**2)] for q in pairs])
 fit=np.array([q["part"]=="fit" for q in pairs]);mu=x[fit].mean(0);sd=np.maximum(x[fit].std(0),.001);x=np.c_[(x-mu)/sd,np.ones(len(x))];y=np.array([q["alt_better"] for q in pairs],float)
 def loss(w):
  z=x[fit]@w;pen=w.copy();pen[-1]=0;return np.mean(np.logaddexp(0,z)-y[fit]*z)+.1*(pen**2).sum(),x[fit].T@(expit(z)-y[fit])/sum(fit)+.2*pen
 res=minimize(loss,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",options=dict(maxiter=300,ftol=1e-12));assert res.success
 scores=x@res.x
 for j,q in enumerate(pairs):q["linear_alt"]=bool(scores[j]>0)
 prior={}
 for k in c["transforms"]:
  rows=[q for q in pairs if q["part"]=="fit" and q["kind"]==k];prior[k]=bool(np.mean([q["alt_error"] for q in rows])<np.mean([q["base_error"] for q in rows]))
 for q in pairs:q["prior_alt"]=prior[q["kind"]]
 old.save(r/"baseline_fit.json",dict(prior=prior,linear_parameters=len(res.x),regularizer=.1,train_only=True));return pairs,requests
def outputs(model,batch,ids):
 h=model.get_base_model().model(**batch,use_cache=False,return_dict=True).last_hidden_state[:,-1] if hasattr(model,"get_base_model") else model.model(**batch,use_cache=False,return_dict=True).last_hidden_state[:,-1]
 weight=model.get_output_embeddings().weight[ids];return F.linear(h,weight).float()
def evaluate(c,r,name,model,tok,tokens,ids,pairs,requests,device):
 model.eval();indices=[j for j,q in enumerate(requests) if q["part"]=="check"];margins={};tops={};tick=time.monotonic()
 with torch.inference_mode():
  for j in indices:
   if time.monotonic()-tick>c["frozen_cap_seconds"]:raise TimeoutError("inference cap")
   batch={"input_ids":torch.tensor([tokens[j]],device=device)};batch["attention_mask"]=torch.ones_like(batch["input_ids"])
   z=outputs(model,batch,ids)[0].cpu().numpy();margins[j]=float(z[1]-z[0]);tops[j]=int(z.argmax())
 bypair={}
 for j in indices:bypair.setdefault(requests[j]["pair"],[]).append(j)
 rows=[];bad=0
 for i,jj in bypair.items():
  q=pairs[i];normal=next(j for j in jj if not requests[j]["swap"]);swap=next(j for j in jj if requests[j]["swap"])
  raw=tops[normal]==1;reverse=tops[swap]==0;bad+=raw!=reverse;margin=(margins[normal]-margins[swap])/2;chosen=margin>0
  for arm,decision in [(name+"Raw",raw),(name,chosen),("Base",False),("TrainGlobal",q["prior_alt"]),("History24",q["rule_alt"]),("FourierHistoryLinear",q["linear_alt"]),("Oracle",q["alt_better"])]:
   pp=q["alt"] if decision else q["base"];err=q["alt_error"] if decision else q["base_error"]
   rows.append(dict(arm=arm,index=q["index"],origin=q["origin"],kind=q["kind"],alt=bool(decision),correct=bool(decision==q["alt_better"]),shape_mse=err,shape_mae=float(np.mean(abs(pp-q["truth"]))),mse=err+q["level_error"],mae=float(np.mean(abs(pp-q["truth"]+q["offset"])))))
 old.csvsave(r/f"{name}_cases.csv",rows);summary=[]
 for arm in dict.fromkeys(q["arm"] for q in rows):
  qq=[q for q in rows if q["arm"]==arm];summary.append(dict(arm=arm,n=len(qq),accuracy=float(np.mean([q["correct"] for q in qq])),shape_mse=float(np.mean([q["shape_mse"] for q in qq])),shape_mae=float(np.mean([q["shape_mae"] for q in qq])),mse=float(np.mean([q["mse"] for q in qq])),mae=float(np.mean([q["mae"] for q in qq]))))
 old.csvsave(r/f"{name}_scores.csv",summary);old.save(r/f"{name}_order.json",dict(inconsistent_pairs=int(bad),pairs=len(bypair),fraction=float(bad/len(bypair)),margins=margins,seconds=time.monotonic()-tick))
 print("EVALUATED",name,json.dumps(summary),flush=True)
def run(c,r,device,train):
 tick=time.monotonic();r.mkdir(parents=True,exist_ok=False);old.save(r/"config.json",c);old.save(r/"source.json",dict(git=old.rt.git(),sha=old.rt.sha(__file__),train_requested=train,assets={c[k]:old.rt.sha(c[k]) for k in ["data","base_prediction","split"]}))
 pairs,requests=prepare(c,r)
 from transformers import AutoTokenizer,AutoModelForCausalLM
 tok=AutoTokenizer.from_pretrained(c["model_path"],local_files_only=True);tok.pad_token=tok.eos_token
 chats=[tok.apply_chat_template([dict(role="user",content=q["text"])],tokenize=False,add_generation_prompt=True,enable_thinking=False) for q in requests]
 tokens=[tok.encode(s,add_special_tokens=False) for s in chats];ids=[tok.encode(a,add_special_tokens=False)[0] for a in "AB"]
 assert all(len(tok.encode(a,add_special_tokens=False))==1 for a in "AB");assert max(map(len,tokens))<=c["max_tokens"],max(map(len,tokens))
 old.save(r/"tokenization.json",dict(max_tokens=max(map(len,tokens)),letter_ids=ids,examples=len(tokens),test_prompts=sum(q["part"]=="check" for q in requests)))
 if device=="prepare":print("PREPARED",len(requests),max(map(len,tokens)),flush=True);return
 old.rt.setup(device,c["gpu_cap_gib"]);assert torch.cuda.mem_get_info(device)[0]>=22*2**30
 model=AutoModelForCausalLM.from_pretrained(c["model_path"],local_files_only=True,dtype=torch.bfloat16,device_map={"":device},attn_implementation="sdpa")
 for p in model.parameters():p.requires_grad_(False)
 evaluate(c,r,"FrozenQwen",model,tok,tokens,ids,pairs,requests,device)
 if train:
  from peft import LoraConfig,get_peft_model,TaskType
  torch.manual_seed(c["seed"]);layers=list(range(model.config.num_hidden_layers-c["last_layers"],model.config.num_hidden_layers))
  model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=c["lora_rank"],lora_alpha=c["lora_alpha"],lora_dropout=0.,target_modules=c["targets"],layers_to_transform=layers,bias="none"))
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
  opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c["lr"],weight_decay=.01)
  train_indices=np.array([j for j,q in enumerate(requests) if q["part"]=="fit"]);rng=np.random.default_rng(c["seed"]);logs=[];trainstart=time.monotonic()
  model.train()
  for step in range(c["steps"]):
   if time.monotonic()-trainstart>c["train_cap_seconds"]:raise TimeoutError("LoRA train cap")
   opt.zero_grad(set_to_none=True);running=0.
   for j in rng.choice(train_indices,c["accumulation"],replace=False):
    batch={"input_ids":torch.tensor([tokens[j]],device=device)};batch["attention_mask"]=torch.ones_like(batch["input_ids"])
    logits=outputs(model,batch,ids);loss=F.cross_entropy(logits,torch.tensor([requests[j]["label"]],device=device));(loss/c["accumulation"]).backward();running+=float(loss.detach())/c["accumulation"]
   grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step();logs.append(dict(step=step+1,loss=running,gradient=float(grad)))
   if step%8==0:print("TRAIN",step+1,running,round(time.monotonic()-trainstart,2),flush=True)
  assert all(p.grad is None for n,p in model.named_parameters() if "lora_" not in n)
  model.save_pretrained(r/"adapter");old.csvsave(r/"train.csv",logs);old.save(r/"lora.json",dict(trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),layers=layers,steps=len(logs),seconds=time.monotonic()-trainstart))
  evaluate(c,r,"AdaptedQwen",model,tok,tokens,ids,pairs,requests,device)
 old.save(r/"completion.json",dict(seconds=time.monotonic()-tick,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,student_updates=0,llm_adapted=bool(train),teacher_only=True,preconsumed_data=True));print("COMPLETE",flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda:0");ap.add_argument("--train-lora",action="store_true");ap.add_argument("--run");a=ap.parse_args();c=old.read("docs/chronogeom/teacher_pairwise02_config.json");r=Path(a.run or c["run"])
 try:run(c,r,a.device,a.train_lora)
 except BaseException:
  if r.exists():old.save(r/"failure.json",dict(traceback=traceback.format_exc()))
  raise
