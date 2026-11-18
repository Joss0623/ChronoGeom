"""Past-only shape opportunities and frozen teacher selection inside pinned TSLib."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,json,time,traceback
from pathlib import Path
import numpy as np,torch
from scipy.optimize import minimize
from scipy.special import expit
from exp import exp_chronogeom_temporal_readout06 as rt
from scripts.chronogeom.teacher_pairwise02 import text,outputs
read,save,csvsave=rt.read,rt.save,rt.csvsave
def center(x):return x-x.mean(-1,keepdims=True)
def pool(x,p0,ids,week):
 x=np.asarray(x,dtype=float);z=center(np.asarray(p0,dtype=float));daily=center(np.tile(x[:,-24:],(1,4)));weekly=center(week[ids]);ii=np.arange(96)
 shifts=[center(np.stack([np.interp(ii+s,ii,v) for v in z])) for s in [1.5,-1.5,3,-3]]
 means=x.reshape(len(x),4,24).mean(-1);slope=((means-means.mean(1,keepdims=True))*np.array([-1.5,-.5,.5,1.5])).sum(1)/5/24;slope=np.clip(slope,-3*x.std(1)/96,3*x.std(1)/96);trend=slope[:,None]*(ii-47.5)
 smooth=np.stack([np.convolve(np.pad(v,2,mode="reflect"),np.ones(5)/5,mode="valid") for v in z])
 a=[z,daily,weekly,.5*(z+daily),.5*(z+weekly),.75*z,1.25*z,1.5*z,*shifts,z+.5*trend,z+trend,center(smooth),.5*(daily+weekly),np.zeros_like(z)]
 result=np.stack(a,1);result-=result.mean(-1,keepdims=True);result+=np.asarray(p0,dtype=float).mean(-1)[:,None,None]
 assert result.shape==(len(x),17,96) and np.max(abs(result.mean(-1)-np.asarray(p0).mean(-1)[:,None]))<1e-12
 return result
def feature(x,p0,candidates,ids):
 n,k,h=candidates.shape;a=center(p0);b=center(candidates);day=center(np.tile(x[:,-24:],(1,4)));hstats=np.c_[x.mean(1),x.std(1),x[:,-24:].mean(1),x[:,-24:].std(1),x[:,-1],x[:,-24:].mean(1)-x[:,:24].mean(1)]
 from exp.exp_chronogeom_dynamic_teacher01 import fourier
 cal=fourier(ids[:,0:1])[:,0];fixed=np.c_[x,a,cal,hstats]
 repeated=np.repeat(fixed[:,None],k-1,axis=1);d=b[:,1:];q=np.stack([np.mean(d*d,2),np.mean((d-day[:,None])**2,2),np.repeat(np.mean(a*a,1)[:,None],k-1,1),np.repeat(np.mean((a-day)**2,1)[:,None],k-1,1),d[:,:,-1]-d[:,:,0]],-1)
 return np.concatenate([repeated,d,q],-1)
def datasets(c,r):
 cc=read(c["reference_config"]);rr,oc,tr,tc,gc=rt.context(cc);b=np.load(rr/"power/known_data.npz");raw,fullids,_=rt.old.grid(gc,"power",cc["confirmation"][1]);values=(raw-b["mean"])/b["sd"]
 lo=rt.ix(cc,gc["domains"]["power"]["train_start"]);hi=rt.ix(cc,gc["domains"]["power"]["train_end"]);v=b["values"][lo:hi,0];ids=b["ids"][lo:hi];week=np.array([v[(ids==i)&np.isfinite(v)].mean() for i in range(168)]);assert np.isfinite(week).all()
 save(r/"weekly_source.json",dict(fit_start=gc["domains"]["power"]["train_start"],fit_end=gc["domains"]["power"]["train_end"],forecast_history_len=96,no_extra_current_week_history=True));np.save(r/"weekly_template.npy",week)
 stages={"train":rt.old.packed(b,b["train_origins"])};counts={}
 for name,bounds in dict(validation=cc["validation"],**cc["development_intervals"],confirmation=cc["confirmation"]).items():stages[name],counts[name]=rt.windows(values,fullids,*[rt.ix(cc,q) for q in bounds])
 save(r/"window_counts.json",counts);allsets={};sources={str(rr/"power/known_data.npz"):rt.sha(rr/"power/known_data.npz")}
 for seed in c["seeds"]:
  model,rec=rt.frozen_base(tc,tr,"power",seed,"cpu");sources[rec["path"]]=rec["sha256"]
  for stage,p in stages.items():
   if stage=="train":base=rt.old.predict(model,rt.tensorpack(p,"cpu"))[:,:,0].astype(float)
   else:
    path=Path(cc["run"])/stage/"predictions.npz";cache=np.load(path);assert np.array_equal(cache["origin"],p["origin"]) and np.allclose(cache["truth"],p["y"],atol=1e-6)
    base=cache[f"FrozenBase_s{seed}"][:,:,0].astype(float);sources[str(path)]=rt.sha(path)
   assert p["origin"].min()>hi
   x=p["x"][:,:,0].astype(float);y=p["y"][:,:,0].astype(float);cand=pool(x,base,p["ids"],week);err=((cand-y[:,None])**2).mean(-1);allsets[seed,stage]=dict(x=x,y=y,base=base,ids=p["ids"],origin=p["origin"],candidates=cand,errors=err)
   np.savez_compressed(r/f"seed{seed}_{stage}.npz",**allsets[seed,stage])
  del model
 save(r/"source.json",dict(git=rt.git(),sha=rt.sha(__file__),assets=sources,all_periods_previously_viewed=True));return allsets
def opportunities(c,r,sets):
 rows=[];candidate_rows=[]
 for (seed,stage),q in sets.items():
  err=q["errors"];base=err[:,0];oracle=err.min(1);gain=base-oracle
  rows.append(dict(seed=seed,stage=stage,n=len(base),base_mse=float(base.mean()),oracle_mse=float(oracle.mean()),oracle_gain=float(1-oracle.mean()/base.mean()),beneficial_fraction=float(np.mean(gain>1e-8)),beneficial_1pct_fraction=float(np.mean(gain>.01*base)),median_perwindow_gain=float(np.median(gain/base))))
  for k,name in enumerate(c["pool"]):candidate_rows.append(dict(seed=seed,stage=stage,arm=name,mse=float(err[:,k].mean()),benefit_fraction=float(np.mean(err[:,k]<base)),oracle_chosen=int((err.argmin(1)==k).sum())))
 csvsave(r/"opportunity_by_seed.csv",rows);csvsave(r/"candidate_marginals.csv",candidate_rows)
 means=[]
 for stage in ["train","validation","spring","summer","confirmation"]:
  rr=[q for q in rows if q["stage"]==stage];means.append(dict(stage=stage,n=rr[0]["n"],**{k:float(np.mean([q[k] for q in rr])) for k in ["base_mse","oracle_mse","oracle_gain","beneficial_fraction","beneficial_1pct_fraction"]}))
 csvsave(r/"opportunity_mean.csv",means);g=c["gate"];passed=means[0]["oracle_gain"]>=g["train_mse_gain"] and means[0]["beneficial_1pct_fraction"]>=g["train_fraction_1pct_benefit"] and sum(q["oracle_gain"]>=g["later_gain"] for q in means[1:])>=g["min_later_periods"]
 save(r/"opportunity_gate.json",dict(passed=bool(passed),rule=g,no_candidate_selection_by_oracle=True));print("OPPORTUNITY",json.dumps(means),"GATE",passed,flush=True);return passed
def baseline(c,r,sets):
 seed=c["judge_seed"];q=sets[seed,"train"];x=feature(q["x"],q["base"],q["candidates"],q["ids"]);shape=x.shape;xf=x.reshape(-1,shape[-1]);mu=xf.mean(0);sd=np.maximum(xf.std(0),.001);xf=np.c_[(xf-mu)/sd,np.ones(len(xf))];gain=(q["errors"][:,0:1]-q["errors"][:,1:]).ravel();y=(gain>0).astype(float)
 def obj(w):
  z=xf@w;pen=w.copy();pen[-1]=0;return np.mean(np.logaddexp(0,z)-y*z)+.1*np.mean(pen*pen),xf.T@(expit(z)-y)/len(xf)+.2*pen/len(pen)
 rr=minimize(obj,np.zeros(xf.shape[1]),jac=True,method="L-BFGS-B",options=dict(maxiter=400,ftol=1e-10));assert rr.success,rr.message
 ridge=np.linalg.solve(xf.T@xf+len(xf)*np.eye(xf.shape[1]),xf.T@gain)
 np.savez_compressed(r/"history_models.npz",mu=mu,sd=sd,logistic=rr.x,utility=ridge)
 chosen={}
 for (s,stage),q in sets.items():
  if s!=seed:continue
  x=feature(q["x"],q["base"],q["candidates"],q["ids"]);xx=np.c_[((x-mu)/sd).reshape(-1,x.shape[-1]),np.ones(x.shape[0]*16)];n=len(q["x"])
  day=center(np.tile(q["x"][:,-24:],(1,4)));d=((center(q["candidates"])-day[:,None])**2).mean(-1)
  chosen[stage]=dict(Base=np.zeros(n,int),Oracle=q["errors"].argmin(1),History24=d.argmin(1),HistClassifier=np.c_[np.zeros(n), (xx@rr.x).reshape(n,16)].argmax(1),HistoryLinearUtility=np.c_[np.zeros(n),(xx@ridge).reshape(n,16)].argmax(1))
 save(r/"baseline_fit.json",dict(windows=shape[0],pair_examples=len(xf),features=xf.shape[1],classifier_converged=bool(rr.success),fixed_hyperparameters=True,fit_only_train=True));return chosen
def score(r,c,q,choices,stage,suffix):
 rows=[];means=[]
 for arm,ids in choices.items():
  pp=q["candidates"][np.arange(len(ids)),ids];e=((pp-q["y"])**2).mean(1);ma=abs(pp-q["y"]).mean(1);base=q["errors"][:,0];diff=base-e
  shape=((center(pp)-center(q["y"]))**2).mean(1)
  for j in range(len(ids)):rows.append(dict(stage=stage,arm=arm,origin=int(q["origin"][j]),candidate=c["pool"][int(ids[j])],mse=float(e[j]),mae=float(ma[j]),shape=float(shape[j]),gain=float(diff[j])))
  means.append(dict(stage=stage,arm=arm,n=len(ids),mse=float(e.mean()),mae=float(ma.mean()),shape=float(shape.mean()),relative_mse_gain=float(diff.mean()/base.mean()),gross_gain=float(np.maximum(diff,0).mean()),gross_harm=float(np.maximum(-diff,0).mean()),changed=int(np.sum(ids!=0))))
 csvsave(r/f"{stage}_{suffix}_scores.csv",means);csvsave(r/f"{stage}_{suffix}_cases.csv",rows);return means
def judge(c,r,sets,chosen,device):
 requests=[];sampled={}
 for stage in ["validation","spring","summer","confirmation"]:
  q=sets[c["judge_seed"],stage];ii=np.linspace(0,len(q["x"])-1,4,dtype=int);sampled[stage]=ii
  for j in ii:
   for k in range(1,17):
    for swap in [False,True]:
     requests.append(dict(stage=stage,row=int(j),candidate=k,swap=swap,text=text(q["x"][j],center(q["base"][j]),center(q["candidates"][j,k]),int(q["ids"][j,0]),swap)))
 save(r/"judge_sample.json",{s:v.tolist() for s,v in sampled.items()});save(r/"prompts.json",requests)
 from transformers import AutoTokenizer,AutoModelForCausalLM
 tok=AutoTokenizer.from_pretrained(c["model_path"],local_files_only=True);chats=[tok.apply_chat_template([dict(role="user",content=q["text"])],tokenize=False,add_generation_prompt=True,enable_thinking=False) for q in requests];tokens=[tok.encode(x,add_special_tokens=False) for x in chats];assert max(map(len,tokens))<=2304
 rt.setup(device,22);assert torch.cuda.mem_get_info(device)[0]>=22*2**30;tick=time.monotonic();llm=AutoModelForCausalLM.from_pretrained(c["model_path"],local_files_only=True,dtype=torch.bfloat16,device_map={"":device},attn_implementation="sdpa");llm.eval()
 for p in llm.parameters():p.requires_grad_(False)
 ids=[tok.encode(x,add_special_tokens=False)[0] for x in "AB"];teacher_margins={}
 for arm in ["QwenTask","FrozenLoRA"]:
  if arm=="FrozenLoRA":
   from peft import PeftModel
   llm=PeftModel.from_pretrained(llm,c["adapter"],is_trainable=False);llm.eval()
  margins=[]
  with torch.inference_mode():
   for j,t in enumerate(tokens):
    if time.monotonic()-tick>600:raise TimeoutError("frozen teacher600sec")
    b=dict(input_ids=torch.tensor([t],device=device));b["attention_mask"]=torch.ones_like(b["input_ids"]);z=outputs(llm,b,ids)[0];margins.append(float(z[1]-z[0]))
    if (j+1)%128==0:print("JUDGE",arm,j+1,len(tokens),round(time.monotonic()-tick,1),flush=True)
  teacher_margins[arm]=margins;save(r/"teacher_margins.json",teacher_margins)
 for stage,ii in sampled.items():
  q=sets[c["judge_seed"],stage];ss={k:v[ii] for k,v in q.items()};pick={arm:a[ii] for arm,a in chosen[stage].items()}
  for arm,margins in teacher_margins.items():
   vals=[]
   for row in ii:
    pp=[]
    for k in range(1,17):
     jj=[j for j,a in enumerate(requests) if a["stage"]==stage and a["row"]==row and a["candidate"]==k];assert len(jj)==2;pp.append((margins[jj[0]]-margins[jj[1]])/2)
    vals.append(int(np.argmax([0]+pp)))
   pick[arm]=np.array(vals)
  print("JUDGMENT",json.dumps(score(r,c,ss,pick,stage,"judge")),flush=True)
 save(r/"teacher_completion.json",dict(seconds=time.monotonic()-tick,teacher_prompts=2*len(tokens),max_tokens=max(map(len,tokens)),peak_gib=torch.cuda.max_memory_allocated(device)/2**30,new_llm_training=False,student_updates=0))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda:0");a=ap.parse_args();c=read("docs/chronogeom/shape_opportunity03_config.json");r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c);tick=time.monotonic();torch.set_num_threads(2)
 try:
  sets=datasets(c,r);passed=opportunities(c,r,sets);chosen=baseline(c,r,sets)
  for stage,p in chosen.items():score(r,c,sets[c["judge_seed"],stage],p,stage,"all")
  if passed:judge(c,r,sets,chosen,a.device)
  src=read(r/"source.json");assert all(rt.sha(p)==v for p,v in src["assets"].items())
  save(r/"completion.json",dict(seconds=time.monotonic()-tick,candidate_gate_passed=bool(passed),old_assets_unchanged=True,no_new_llm_or_forecaster_training=True));print("COMPLETE",flush=True)
 except BaseException:save(r/"failure.json",dict(traceback=traceback.format_exc()));raise
if __name__=="__main__":main()
