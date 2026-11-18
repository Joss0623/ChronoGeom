"""OPD-style ordinal trajectory auxiliary for the unchanged direct TimeBase.
Student samples refreshed each gradient step; frozen Qwen gives causal next-state probabilities.
"""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,csv,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from exp import exp_chronogeom_dynamic_earlystop02 as es
old=es.old;rt=es.rt
read,save,csvsave=es.read,es.save,es.csvsave
def probs(pred,c):
 z=pred.reshape(len(pred),16,6).mean(-1);z=z-z.mean(1,keepdim=True)
 low=torch.sigmoid((-c["band"]-z)/c["softness"]);high=torch.sigmoid((z-c["band"])/c["softness"])
 out=torch.stack([low,1-low-high,high],-1).clamp_min(1e-7)
 return out/out.sum(-1,keepdim=True)
def true_states(y,c):
 z=y.reshape(len(y),16,6).mean(-1);z=z-z.mean(1,keepdims=True)
 return np.where(z<-c["band"],0,np.where(z>c["band"],2,1)).astype("int64")
def analytical_x(ids,stats,samples,mu,sd):
 n=len(ids);calendar=old.fourier(ids[:,3::6]);s=(stats-mu)/sd
 prefix=np.eye(3)[samples];mask=np.tril(np.ones((16,16)),k=-1)
 prefix=(prefix[:,None]*mask[None,:,:,None]).reshape(n,16,48)
 return np.concatenate([calendar,np.repeat(s[:,None],16,1),np.repeat(np.eye(16)[None],n,0),prefix],-1).astype("float32")
def fit_analytic(p,idx,c,device,r,phase):
 start=time.monotonic();st=true_states(p["y"][idx],c);mu=p["statistics"][idx].mean(0);sd=np.maximum(p["statistics"][idx].std(0),1e-3)
 x=analytical_x(p["ids"][idx],p["statistics"][idx],st,mu,sd);x=torch.tensor(x.reshape(-1,102),device=device);y=torch.tensor(st.reshape(-1),device=device)
 torch.manual_seed(20241029);m=torch.nn.Linear(102,3).to(device);opt=torch.optim.Adam(m.parameters(),lr=.03);rng=np.random.default_rng(20241029)
 for j in range(300):
  ii=torch.tensor(rng.integers(len(x),size=256),device=device);loss=F.cross_entropy(m(x[ii]),y[ii]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 m.eval()
 for q in m.parameters():q.requires_grad_(False)
 with torch.no_grad():acc=float((m(x).argmax(-1)==y).float().mean())
 path=r/f"analytic_{phase}.pt";torch.save(m.state_dict(),path)
 save(r/f"analytic_{phase}.json",dict(parameters=309,training_tokens=len(x),training_accuracy=acc,updates=300,mu=mu.tolist(),sd=sd.tolist(),seconds=time.monotonic()-start,sha256=rt.sha(path),no_validation_fit=True))
 return m,mu,sd
def build_prompt(p,index):
 days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
 labels=", ".join(f"{days[int(s)//24]} {int(s)%24:02d}:00" for s in p["ids"][index,3::6])
 stats=", ".join(f"{k}={v:.3f}" for k,v in zip(["mean","std","last_value","last24_mean","last24_std","last24_minus_first24_mean"],p["statistics"][index]))
 return ("Domain: hourly electricity active power of one household in Sceaux, France. Occupancy, appliances and future weather are unknown. "
 "Only observed historical statistics and legal calendar metadata are provided.\n"
 "Describe the SHAPE of demand over the next96 hours as16 consecutive6-hour block labels. All numerical values are in fixed historical-training standardized power units. "
 "A means the block average is more than0.25 BELOW the average over the entire future96-hour window; B means within plus/minus0.25 of that future-window average; C means more than0.25 ABOVE it. "
 "The future-window average is unobserved: infer relative daily/weekly shape from available context, not future data.\n"
 "Observed previous96-hour statistics: "+stats+".\n"
 "Calendar at the center of each of the16future6-hour blocks: "+labels+".\n"
 "Produce exactly16labels A,B,C separated by spaces, in block order. Continue the supplied partial label sequence using calendar and observed-history evidence. No explanations.")
class QwenScorer:
 def __init__(self,c,p,device,r):
  from transformers import AutoTokenizer,AutoModelForCausalLM
  self.c,self.p,self.device,self.r=c,p,device,r;self.tick=time.monotonic();self.calls=0;self.records=0
  self.tok=AutoTokenizer.from_pretrained(c["model_path"],local_files_only=True)
  if self.tok.pad_token_id is None:self.tok.pad_token=self.tok.eos_token
  self.prompts=[build_prompt(p,i) for i in range(len(p["x"]))]
  self.tokens=[]
  for text in self.prompts:
   chat=self.tok.apply_chat_template([dict(role="user",content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   assert "<think>\n\n</think>" in chat
   self.tokens.append(self.tok.encode(chat,add_special_tokens=False))
  ids=[self.tok.encode(" "+s,add_special_tokens=False) for s in "ABC"];assert all(len(a)==1 for a in ids);self.ids=[a[0] for a in ids]
  assert max(map(len,self.tokens))+16<=c["prompt_max_tokens"];save(r/"qwen_prompts.json",self.prompts);save(r/"qwen_tokenizer.json",dict(letter_ids=self.ids,decoded=[self.tok.decode([q]) for q in self.ids],max_tokens=max(map(len,self.tokens))+16))
  print("LOAD_QWEN",flush=True)
  self.model=AutoModelForCausalLM.from_pretrained(c["model_path"],dtype=torch.bfloat16,device_map={"":device},local_files_only=True,attn_implementation="sdpa");self.model.eval()
  for q in self.model.parameters():q.requires_grad_(False)
  self.load_seconds=time.monotonic()-self.tick;self.log=(r/"qwen_feedback.jsonl").open("w")
  s=np.zeros((1,16),dtype=int);a=self._score([0],s);s[0,-1]=2;b=self._score([0],s);error=float(abs(a-b).max());assert error<1e-4
  save(r/"qwen_causality_check.json",dict(changed_last_student_token_max_error_all_teacher_next_token_scores=error,passed=True))
 def _score(self,indices,sampled):
  if time.monotonic()-self.tick>self.c["teacher_cap_seconds"]:raise TimeoutError("Qwen timecap")
  bs=len(indices);seq=[self.tokens[int(i)]+[self.ids[int(s)] for s in states] for i,states in zip(indices,sampled)];width=max(map(len,seq));ids=torch.full((bs,width),self.tok.pad_token_id,dtype=torch.long,device=self.device);mask=torch.zeros_like(ids)
  for j,s in enumerate(seq):ids[j,-len(s):]=torch.tensor(s,device=self.device);mask[j,-len(s):]=1
  positions=torch.arange(width-17,width-1,device=self.device)
  with torch.inference_mode():
   h=self.model.model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state[:,positions]
   w=self.model.get_output_embeddings().weight[self.ids];z=F.linear(h,w).float();out=z.softmax(-1).cpu().numpy()
  self.calls+=bs;return out
 def score(self,indices,sampled,metadata):
  out=[]
  for j in range(0,len(indices),self.c["prompt_batch_size"]):out.append(self._score(indices[j:j+self.c["prompt_batch_size"]],sampled[j:j+self.c["prompt_batch_size"]]))
  ans=np.concatenate(out);self.records+=len(ans);self.log.write(json.dumps(dict(**metadata,rows=[int(i) for i in indices],states=sampled.tolist(),probabilities=ans.tolist()))+"\n");self.log.flush();return ans
def zerohead(seed,state,device):
 h=es.head(seed,device);h.load_state_dict(state);return h
def data_for_phase(c,p,src,early,phase,seed,device):
 n=int(p["n_train"]);split=read(early/"split.json")
 if phase=="inner":
  train=np.array(split["fit_indices"]);val=np.array(split["selection_indices"]);features=np.load(early/"inner_preprocessing/features.npz");mu_sd=np.load(early/f"inner_numeric_seed{seed}.npz")
  states=torch.load(early/f"inner_PlainFourier_s{seed}_states.pt",map_location=device,weights_only=True);initial=states[25]
 else:
  train=np.arange(n);val=np.arange(n,len(p["x"]));features=np.load(src/"features.npz");mu_sd=np.load(src/f"numeric_seed{seed}.npz")
  initial=torch.load(early/f"refit_PlainFourier_s{seed}.pt",map_location=device,weights_only=True)
 p0=np.load(src/f"numeric_seed{seed}.npz")["p0"]
 with torch.no_grad():num=old.numeric_summary(torch.tensor(p["x"],device=device),torch.tensor(p0,dtype=torch.float32,device=device)).cpu().numpy()
 num=((num-mu_sd["mu"])/mu_sd["sd"]).astype("float32");code=es.code_for(features,"PlainFourier")
 return train,val,num,code,p0,initial
def simulate(c,p,src,early,r,phase,scorer,device,selections=None):
 idx=np.array(read(early/"split.json")["fit_indices"]) if phase=="inner" else np.arange(int(p["n_train"]))
 analytic,am,asd=fit_analytic(p,idx,c,device,r,phase);allcurves=[];allrecords=[];predictions={};lambdas=[];feedback_summary=[];cpu_replay=[]
 strengths=c["gradient_fractions"]
 variants=[("NoKD",0.)]+[(family,f) for family in ["FourierOPD","QwenOPD","QwenShuffle"] for f in strengths] if phase=="inner" else [(family,selections[family]["fraction"]) for family in c["families"]]
 maxsteps=c["steps"] if phase=="inner" else max(q["step"] for q in selections.values())
 permutation=np.random.default_rng(20241029).permutation(16);save(r/"shuffle_permutation.json",permutation.tolist())
 for seed in c["seeds"]:
  train,val,num,code,p0,initial=data_for_phase(c,p,src,early,phase,seed,device)
  reference=old.prediction(zerohead(seed,initial,device),num[val],code[val],p0[val],device);base_day=es.daily_mse(reference,p["y"][val],p["origin"][val]) if phase=="inner" else None
  predictions[f"Start_s{seed}"]=reference
  heads={};opts={};params={};fixed_lambda={};states={}
  for fam,frac in variants:
   key=f"{fam}_{frac:g}";heads[key]=zerohead(seed,initial,device);opts[key]=torch.optim.Adam(heads[key].parameters(),lr=c["lr"]);params[key]=list(heads[key].parameters());states[key]={};fixed_lambda[key]=None
  nt=torch.tensor(num[train],device=device);ct=torch.tensor(code[train],device=device);pt=torch.tensor(p0[train],device=device);yt=torch.tensor(p["y"][train],dtype=torch.float64,device=device)
  rng=np.random.default_rng(seed+963)
  for step in range(maxsteps+1):
   if time.monotonic()-scorer.tick>c["workflow_cap_seconds"]:raise TimeoutError("workflow cap")
   check=step in c["checks"] or step==maxsteps
   if check:
    for fam,frac in variants:
     key=f"{fam}_{frac:g}";h=heads[key];states[key][step]={k:v.detach().cpu().clone() for k,v in h.state_dict().items()}
     # Outer labels are never evaluated during full optimization.
     if phase=="inner":
      pp=old.prediction(h,num[val],code[val],p0[val],device);m,l,s,ma=rt.components(pp,p["y"][val]);allcurves.append(dict(family=fam,fraction=frac,seed=seed,step=step,relative_day_mse=es.daily_mse(pp,p["y"][val],p["origin"][val])/base_day,mse=float(m.mean()),mae=float(ma.mean())))
   if step==maxsteps:break
   ii=rng.integers(len(train),size=c["batch_size"]);globalrows=train[ii];ti=torch.tensor(ii,device=device)
   for vindex,(fam,frac) in enumerate(variants):
    key=f"{fam}_{frac:g}";h=heads[key];h.train();yp=old.forward(h,nt[ti],ct[ti],pt[ti]);sup=F.mse_loss(yp,yt[ti]);loss=sup
    if fam!="NoKD":
     ps=probs(yp,c);g=torch.Generator(device=device).manual_seed(seed+100000*int(phase=="full")+1000*step)
     sampled=torch.multinomial(ps.detach().reshape(-1,3),1,generator=g).reshape(len(ii),16).cpu().numpy()
     if fam=="FourierOPD":
      features=analytical_x(p["ids"][globalrows],p["statistics"][globalrows],sampled,am,asd)
      with torch.no_grad():target=analytic(torch.tensor(features,device=device)).softmax(-1).cpu().numpy()
     else:
      target=scorer.score(globalrows,sampled,dict(phase=phase,family=fam,fraction=frac,seed=seed,step=step))
      if fam=="QwenShuffle":target=target[:,permutation]
     target=torch.tensor(target,dtype=ps.dtype,device=device).clamp_min(1e-7);target=target/target.sum(-1,keepdim=True)
     aux=(ps*(ps.log()-target.log())).sum(-1).mean()
     if fixed_lambda[key] is None:
      gs=torch.autograd.grad(sup,params[key],retain_graph=True,allow_unused=True);ga=torch.autograd.grad(aux,params[key],retain_graph=True,allow_unused=True)
      ns=torch.sqrt(sum(z.square().sum() for z in gs if z is not None));na=torch.sqrt(sum(z.square().sum() for z in ga if z is not None));lam=frac*float(ns/na) if na>1e-10 else 0.
      fixed_lambda[key]=lam;lambdas.append(dict(phase=phase,family=fam,fraction=frac,seed=seed,coefficient=lam,supervised_gradient=float(ns),kd_gradient=float(na)))
     loss=sup+fixed_lambda[key]*aux
     if step==0:feedback_summary.append(dict(phase=phase,family=fam,fraction=frac,seed=seed,mean_teacher_entropy=float(-(target*target.log()).sum(-1).mean()),initial_reverse_kl=float(aux.detach())))
    opts[key].zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params[key],1.,error_if_nonfinite=True);opts[key].step()
  for fam,frac in variants:
   key=f"{fam}_{frac:g}";path=r/f"{phase}_{fam}_{frac:g}_s{seed}_states.pt";torch.save(states[key],path);allrecords.append(dict(phase=phase,family=fam,fraction=frac,seed=seed,updates=maxsteps,path=str(path),sha256=rt.sha(path),trainable=6570))
   if phase=="full":
    chosen=selections[fam]["step"];head=heads[key];head.load_state_dict(states[key][chosen]);pp=old.prediction(head,num[val],code[val],p0[val],device);predictions[f"{fam}_s{seed}"]=pp
    cpu=zerohead(seed,states[key][chosen],"cpu")
    with torch.no_grad():nc=old.numeric_summary(torch.tensor(p["x"]),torch.tensor(p0,dtype=torch.float32)).numpy()
    normal=np.load(src/f"numeric_seed{seed}.npz");nc=((nc-normal["mu"])/normal["sd"]).astype("float32");cp=old.prediction(cpu,nc[val],code[val],p0[val],"cpu");diff=float(abs(cp-pp).max());assert diff<2e-5;cpu_replay.append(dict(family=fam,seed=seed,max_error=diff))
    qs=selections["QwenOPD"]["step"];head.load_state_dict(states[key][qs]);predictions[f"{fam}_matched_s{seed}"]=old.prediction(head,num[val],code[val],p0[val],device)
  save(r/f"{phase}_records.json",allrecords);print("PHASE_SEED_DONE",phase,seed,"queries",scorer.calls,flush=True)
 if phase=="inner":csvsave(r/"inner_curves.csv",allcurves)
 else:np.savez_compressed(r/"predictions.npz",**predictions,truth=p["y"][int(p["n_train"]):],origin=p["origin"][int(p["n_train"]):]);csvsave(r/"cpu_replays.csv",cpu_replay)
 if lambdas:csvsave(r/f"{phase}_kd_coefficients.csv",lambdas)
 if feedback_summary:csvsave(r/f"{phase}_teacher_diagnostics.csv",feedback_summary)
 if not lambdas:save(r/f"{phase}_no_updates.json",dict(updates=0,reason="all internally selected zero-step fallback"))
 return allcurves,allrecords
def select(c,curves,r):
 choices={};means=[]
 for fam in c["families"]:
  fractions=[0.] if fam=="NoKD" else c["gradient_fractions"];candidates=[]
  for f in fractions:
   for step in c["checks"]:
    qq=[q for q in curves if q["family"]==fam and q["fraction"]==f and q["step"]==step];assert len(qq)==3
    row=dict(family=fam,fraction=f,step=step,score=float(np.mean([q["relative_day_mse"] for q in qq])));candidates.append(row);means.append(row)
  # First candidate within1e-4 of minimum: earliest step, then lowerfraction.
  minimum=min(q["score"] for q in candidates);eligible=[q for q in candidates if q["score"]<=minimum+1e-4];best=min(eligible,key=lambda q:(q["step"],q["fraction"]));choices[fam]=best
 save(r/"inner_selection.json",dict(choices=choices,outer_metrics_used=False,selection_rule="sharedcriterion/identicalsearchgrid,perfamilychoice",git=rt.git()));csvsave(r/"inner_candidate_means.csv",means);print("INNER_SELECTION",json.dumps(choices),flush=True);return choices
def evaluate(c,r,choices):
 a=np.load(r/"predictions.npz");rows=[];windows=[]
 names=["Start"]+c["families"]+[s+"_matched" for s in c["families"]]
 for name in names:
  for seed in c["seeds"]:
   pp=a[f"{name}_s{seed}"];m,l,s,ma=rt.components(pp,a["truth"]);err=float(abs(pp.mean(1)-a[f"Start_s{seed}"].mean(1)).max());assert err<1e-12
   steps=0 if name=="Start" else choices["QwenOPD"]["step"] if name.endswith("_matched") else choices[name]["step"]
   rows.append(dict(arm=name,seed=seed,additional_updates=steps,mse=float(m.mean()),mae=float(ma.mean()),level=float(l.mean()),shape=float(s.mean()),max_mean_error=err))
   for j,o in enumerate(a["origin"]):windows.append(dict(arm=name,seed=seed,origin=int(o),mse=float(m[j]),mae=float(ma[j])))
 csvsave(r/"scores_by_seed.csv",rows);csvsave(r/"scores_by_window.csv",windows);means=[]
 for name in names:
  qq=[q for q in rows if q["arm"]==name];q=dict(arm=name,additional_updates=qq[0]["additional_updates"])
  q.update({k:float(np.mean([z[k] for z in qq])) for k in ["mse","mae","level","shape"]});q.update({f"seed{s}_mse":next(z["mse"] for z in qq if z["seed"]==s) for s in c["seeds"]});means.append(q)
 csvsave(r/"scores_mean.csv",means);by={z["arm"]:z for z in means};win=all(by["QwenOPD"]["mse"]<by[x]["mse"] for x in ["Start","NoKD","FourierOPD","QwenShuffle"])
 print("FORECAST_RESULT",json.dumps(dict(means=means,all_point_comparisons_win=win)),flush=True)
 return means,win,max(q["max_mean_error"] for q in rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda:5");ap.add_argument("--config",default="docs/chronogeom/ordinal_opd01_config.json");args=ap.parse_args();c=read(args.config);r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c);tick=time.monotonic();scorer=None
 try:
  rt.setup(args.device,c["gpu_cap_gib"]);free,_=torch.cuda.mem_get_info(args.device);assert free>=22*2**30
  src=Path(c["dynamic_run"]);early=Path(c["early_run"]);p=np.load(src/"data.npz")
  sources=dict(read(early/"source.json")["assets"])
  for seed in c["seeds"]:
   for name in [f"inner_PlainFourier_s{seed}_states.pt",f"refit_PlainFourier_s{seed}.pt",f"inner_numeric_seed{seed}.npz"]:
    path=early/name;sources[str(path)]=rt.sha(path)
  path=early/"inner_preprocessing/features.npz";sources[str(path)]=rt.sha(path)
  save(r/"source.json",dict(git=rt.git(),source_sha=rt.sha(__file__),configuration_sha=rt.sha(args.config),inherited_assets=sources,start_inner=str(early/"inner_PlainFourier_s2024_states.pt"),start_full=str(early/"refit_PlainFourier_s2024.pt"),timebase_source_sha=rt.sha("layers/ChronoGeomTimeBaseOfficial.py")))
  scorer=QwenScorer(c,p,args.device,r);curves,innerrec=simulate(c,p,src,early,r,"inner",scorer,args.device);choices=select(c,curves,r);decision_sha=rt.sha(r/"inner_selection.json")
  _,fullrec=simulate(c,p,src,early,r,"full",scorer,args.device,choices);save(r/"full_models_lock.json",dict(selection_sha=decision_sha,models=fullrec,outer_metrics_not_used=True))
  means,win,maxmean=evaluate(c,r,choices)
  assert rt.sha(r/"inner_selection.json")==decision_sha and all(rt.sha(k)==v for k,v in sources.items())
  assert all(rt.sha(z["path"])==z["sha256"] for z in innerrec+fullrec)
  scorer.log.close()
  completion=dict(seconds=time.monotonic()-tick,teacher_elapsed_seconds=time.monotonic()-scorer.tick,qwen_load_seconds=scorer.load_seconds,qwen_prompt_sequences=scorer.calls,feedback_sequences=scorer.records,teacher_weights_frozen=True,nonthinking=True,generated_tokens=0,new_llm_training=False,teacher_feedback_refreshed_every_gradient_step=True,inner_models=len(innerrec),full_models=len(fullrec),old_assets_unchanged=True,selection_unchanged=True,max_mean_error=maxmean,peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/2**30,all_primary_point_comparisons_win=bool(win),all_evaluation_previously_viewed=True)
  save(r/"completion.json",completion);print("COMPLETE",json.dumps(completion),flush=True)
 except BaseException:
  if scorer is not None:scorer.log.flush()
  save(r/"failure.json",dict(traceback=traceback.format_exc()));raise
if __name__=="__main__":main()
