"""Minimal dynamic temporal Qwen teacher inside existing THUML TSLib.
No LLM training, new Encoder, or new distillation objective.
"""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,gc,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from exp import exp_chronogeom_temporal_readout06 as rt
from exp.exp_chronogeom_residual03 import ResidualReadout,numeric_summary
from scripts.chronogeom.task_context02_analysis import read,save,csvsave

def center(a):return a-a.mean(1,keepdims=True)
def statistics(x):
 a=x[:,:,0].astype(float)
 return np.column_stack([a.mean(1),a.std(1),a[:,-1],a[:,-24:].mean(1),a[:,-24:].std(1),a[:,-24:].mean(1)-a[:,:24].mean(1)])
def fourier(ids):
 h=ids%24;d=ids//24
 first=np.stack([np.cos(2*np.pi*h/24),np.sin(2*np.pi*h/24),np.cos(2*np.pi*d/7),np.sin(2*np.pi*d/7)],-1)
 cols=[]
 for x,p,ks in [(h,24,range(1,12)),(d,7,range(1,4))]:
  for k in ks:cols.extend([np.cos(2*np.pi*k*x/p),np.sin(2*np.pi*k*x/p)])
 cols.extend(first[...,i]*first[...,j] for i in [0,1] for j in [2,3])
 return np.stack(cols,-1)
def rmsfit(a):
 mu=a.mean(tuple(range(a.ndim-1)));out=a-mu;scale=max(float(np.sqrt(np.mean(out*out))),1e-12)
 return out/scale,mu,scale
def pack(c):
 rr,oc,tr,tc,gc=rt.context(c);b=np.load(rr/"power/known_data.npz")
 rng=np.random.default_rng(c["sampling_seed"]);choice=np.sort(rng.choice(len(b["train_origins"]),size=c["train_windows"],replace=False))
 train=rt.old.packed(b,b["train_origins"][choice])
 val,counts=rt.windows(b["values"],b["ids"],rt.ix(c,c["validation"][0]),rt.ix(c,c["validation"][1]))
 assert train["origin"].max()+96<=rt.ix(c,c["train"][1]) and len(val["x"])==16
 return train,val,counts,tr,tc,Path(tc["source_run"])
def prompt(state,stats,mu,sd):
 days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
 history="unavailable" if stats is None else ", ".join(f"{k}={v:.3f}" for k,v in zip(["mean","std","last_value","last24_mean","last24_std","last24_minus_first24_mean"],stats))
 return ("Domain: hourly household electricity active power, one home in Sceaux, France. Occupancy, appliances, weather and future demand are unknown. "
 "Only Hour/Weekday and the observed96-hour history are available. Represent how calendar context and observed activity state jointly relate to the next96-hour demand SHAPE, without guessing an absolute level. "
 f"Forecast begins {days[int(state)//24]}, hour {int(state)%24}. Historical statistics in fixed historical-training standardized power units: {history}. "
 "Do not infer holidays or future observations. Focus on daily/weekly activity patterns conditional on the observed history.")
def prep(c,r):
 r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c);train,val,counts,tr,tc,staticrun=pack(c)
 p={k:np.concatenate([train[k],val[k]]) for k in ["x","y","ids","origin"]};n=len(train["x"]);stats=statistics(p["x"])
 rng=np.random.default_rng(c["sampling_seed"]+1);perms=[];shuffled=[]
 for x in p["x"]:
  order=rng.permutation(4)
  while np.array_equal(order,np.arange(4)):order=rng.permutation(4)
  perms.append(order);shuffled.append(x.reshape(4,24,1)[order].reshape(96,1))
 ss=statistics(np.stack(shuffled));assert np.allclose(stats[:,:2],ss[:,:2])
 mu=stats[:n].mean(0);sd=np.maximum(stats[:n].std(0),1e-3);requests=[]
 for kind,values in [("DynamicQwen",stats),("HistoryShuffleQwen",ss)]:
  for j in range(len(stats)):
   requests.append(dict(kind=kind,row=j,origin=int(p["origin"][j]),split="train" if j<n else "validation",text=prompt(p["ids"][j,0],values[j],mu,sd)))
 np.savez_compressed(r/"data.npz",**p,statistics=stats,shuffled_statistics=ss,summary_mean=mu,summary_sd=sd,block_permutations=np.stack(perms),n_train=n)
 save(r/"prompts.json",requests);save(r/"source.json",dict(commit=rt.git(),source_sha=rt.sha(__file__),config_sha=rt.sha("docs/chronogeom/dynamic_teacher01_config.json"),known_data_sha=rt.sha(Path(c["reuse_run"])/"power/known_data.npz"),static_path=str(staticrun/"electricity_cache/layer24.npy"),static_sha=rt.sha(staticrun/"electricity_cache/layer24.npy"),timebase_source_sha=rt.sha("layers/ChronoGeomTimeBaseOfficial.py"),n_train=n,n_validation=len(val["x"]),validation_counts=counts,all_evaluation_previously_viewed=True))
 print("PREPARED",n,len(val["x"]),len(requests),flush=True)
def extract(c,r,device):
 from transformers import AutoTokenizer,AutoModelForCausalLM
 t=time.monotonic();rt.setup(device,c["gpu_cap_gib"]);free,_=torch.cuda.mem_get_info(device);assert free>=22*2**30
 requests=read(r/"prompts.json");tok=AutoTokenizer.from_pretrained(c["model_path"],local_files_only=True);tok.padding_side="left"
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 chats=[tok.apply_chat_template([dict(role="user",content=q["text"])],tokenize=False,add_generation_prompt=True,enable_thinking=False) for q in requests]
 lengths=[len(tok.encode(s,add_special_tokens=False)) for s in chats];assert max(lengths)<=c["prompt_max_tokens"];save(r/"prompt_lengths.json",lengths)
 print("LOAD_QWEN",len(chats),max(lengths),flush=True)
 model=AutoModelForCausalLM.from_pretrained(c["model_path"],dtype=torch.bfloat16,device_map={"":device},local_files_only=True,attn_implementation="sdpa");model.eval()
 for p in model.parameters():p.requires_grad_(False)
 loaded=time.monotonic();buf={};out=[]
 def hook(module,inputs,output):
  z=output[0] if isinstance(output,tuple) else output;buf["z"]=z[:,-1].detach().float().cpu().numpy()
 handle=model.model.layers[c["layer"]-1].register_forward_hook(hook)
 with torch.inference_mode():
  for j in range(0,len(chats),c["prompt_batch_size"]):
   if time.monotonic()-t>c["teacher_cap_seconds"]:raise TimeoutError("Qwen extraction cap")
   tokens=tok(chats[j:j+c["prompt_batch_size"]],padding=True,add_special_tokens=False,return_tensors="pt").to(device)
   buf.clear();model.model(**tokens,use_cache=False,return_dict=True);out.append(buf["z"])
   if j%64==0:print("EXTRACT",j+len(buf["z"]),len(chats),round(time.monotonic()-t,2),flush=True)
 handle.remove();raw=np.concatenate(out);assert raw.shape==(len(chats),4096) and np.isfinite(raw).all()
 np.save(r/"dynamic_raw_layer24.npy",raw)
 save(r/"extraction.json",dict(seconds=time.monotonic()-t,load_seconds=loaded-t,prompts=len(chats),layer=c["layer"],max_tokens=max(lengths),model_revision=c["model_revision"],nonthinking=True,llm_training=False,generated_tokens=0,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,raw_sha=rt.sha(r/"dynamic_raw_layer24.npy")))
 del model,p,tokens
 gc.collect();torch.cuda.synchronize(device);torch.cuda.empty_cache()
def makefeatures(c,r):
 p=np.load(r/"data.npz");n=int(p["n_train"]);N=len(p["x"]);source=read(r/"source.json");static=np.load(source["static_path"]).astype(float);raw=np.load(r/"dynamic_raw_layer24.npy").astype(float)
 dyn,shuf=raw[:N],raw[N:];weights=np.bincount(p["ids"][:n].ravel(),minlength=168).astype(float);weights/=weights.sum()
 means={"StaticQwen":(weights[:,None]*static).sum(0),"DynamicQwen":dyn[:n].mean(0),"HistoryShuffleQwen":shuf[:n].mean(0)}
 # Equal arm mass; static contribution weighted by actual TRAIN calendar frequencies.
 stack=np.concatenate([(static-means["StaticQwen"])*np.sqrt(n*weights[:,None]),dyn[:n]-means["DynamicQwen"],shuf[:n]-means["HistoryShuffleQwen"]])
 _,sing,basis=np.linalg.svd(stack,full_matrices=False);basis=basis[:32]
 native={"StaticQwen":((static-means["StaticQwen"])@basis.T)[p["ids"]],"DynamicQwen":np.repeat(((dyn-means["DynamicQwen"])@basis.T)[:,None],96,1),"HistoryShuffleQwen":np.repeat(((shuf-means["HistoryShuffleQwen"])@basis.T)[:,None],96,1)}
 f=fourier(p["ids"]);_,fmu,fscale=rmsfit(f[:n]);f=(f-fmu)/fscale
 stats=(p["statistics"]-p["summary_mean"])/p["summary_sd"]
 projection=np.random.default_rng(c["sampling_seed"]+2).normal(size=(6,32))/np.sqrt(6)
 native["Fourier"]=f*(stats@projection)[:,None]
 features={};diagnostics=[];normal={}
 for name in c["arms"]:
  z=native[name];_,mu,scale=rmsfit(z[:n]);zz=(z-mu)/scale
  code=np.zeros((N,96,168),dtype=np.float32);code[:,:,:32]=f;code[:,:,32:64]=zz;features[name]=code
  a=zz[:n].reshape(-1,32);sv=np.linalg.svd(a,compute_uv=False);mass=sv*sv;mass=mass[mass>1e-15];mass/=mass.sum()
  retention=float(np.sum(((dyn[:n]-means[name])@basis.T)**2)/np.sum((dyn[:n]-means[name])**2)) if name=="DynamicQwen" else float(np.sum(((shuf[:n]-means[name])@basis.T)**2)/np.sum((shuf[:n]-means[name])**2)) if name=="HistoryShuffleQwen" else float(np.sum(weights[:,None]*((static-means[name])@basis.T)**2)/np.sum(weights[:,None]*(static-means[name])**2)) if name=="StaticQwen" else 1.
  diagnostics.append(dict(arm=name,additional_width=32,common_fourier_width=32,padded_width=168,train_additional_rms=float(np.sqrt(np.mean(zz[:n]**2))),validation_additional_rms=float(np.sqrt(np.mean(zz[n:]**2))),effective_rank=int((sv>sv[0]*1e-6).sum()),entropy_rank=float(np.exp(-(mass*np.log(mass)).sum())),raw_variance_retention=retention))
  normal[name+"_mu"]=mu;normal[name+"_scale"]=scale
 np.savez_compressed(r/"features.npz",**features);np.savez_compressed(r/"projection.npz",basis=basis,singular_values=sing,static_mu=means["StaticQwen"],dynamic_mu=means["DynamicQwen"],shuffle_mu=means["HistoryShuffleQwen"],fourier_mu=fmu,fourier_scale=fscale,interaction_projection=projection,**normal)
 csvsave(r/"feature_diagnostics.csv",diagnostics);print("FEATURES",json.dumps(diagnostics),flush=True)
def forward(head,num,code,p0):
 lead=head.lead_onehot[None].expand(len(num),-1,-1);z=torch.cat([num,lead,code],-1)
 correction=head.mlp(z)+head.numeric(num)+head.lead_bias[None,:,None]
 correction=correction.double();return p0.double()+correction-correction.mean(1,keepdim=True)
def prediction(head,num,code,p0,device):
 ans=[];head.eval()
 with torch.no_grad():
  for j in range(0,len(num),64):
   args=[torch.tensor(x[j:j+64],device=device) for x in [num,code,p0]]
   ans.append(forward(head,*args).cpu().numpy())
 return np.concatenate(ans)
def fit(c,r,device):
 gc.collect();torch.cuda.synchronize(device);torch.cuda.empty_cache()
 rt.setup(device,2);t=time.monotonic();p=np.load(r/"data.npz");n=int(p["n_train"]);features=np.load(r/"features.npz");_,_,tr,tc,_=rt.context(c);scores=[];windowrows=[];records=[];outputs={"truth":p["y"][n:],"origin":p["origin"][n:]};base_sources=[]
 for seed in c["seeds"]:
  base,brec=rt.frozen_base(tc,tr,"power",seed,device);base_sources.append(brec)
  bpred=rt.old.predict(base,rt.tensorpack({k:p[k] for k in ["x","y","ids","origin"]},device)).astype(float)
  x=torch.tensor(p["x"],device=device);bp=torch.tensor(bpred,dtype=torch.float32,device=device)
  with torch.no_grad():num=numeric_summary(x,bp).cpu().numpy()
  mu=num[:n].mean((0,1));sd=np.maximum(num[:n].std((0,1)),1e-3);num=((num-mu)/sd).astype("float32")
  np.savez_compressed(r/f"numeric_seed{seed}.npz",mu=mu,sd=sd,p0=bpred)
  outputs[f"FrozenBase_s{seed}"]=bpred[n:]
  def score(name,pp,head=None):
   mse,level,shape,mae=rt.components(pp,p["y"][n:]);meanerr=float(abs(pp.mean(1)-bpred[n:].mean(1)).max());assert meanerr<1e-12
   result=dict(arm=name,seed=seed,mse=float(mse.mean()),mae=float(mae.mean()),level=float(level.mean()),shape=float(shape.mean()),max_mean_error=meanerr,n_origins=len(pp),trainable=0 if head is None else sum(z.numel() for z in head.parameters()),base_frozen=6570)
   scores.append(result);outputs[f"{name}_s{seed}"]=pp
   for j,o in enumerate(p["origin"][n:]):windowrows.append(dict(arm=name,seed=seed,origin=int(o),mse=float(mse[j]),mae=float(mae[j]),level=float(level[j]),shape=float(shape[j])))
   return result
  score("FrozenBase",bpred[n:])
  for arm in c["arms"]:
   head=ResidualReadout(seed,np.zeros((168,168),dtype="float32")).to(device);assert sum(z.numel() for z in head.parameters())==6570
   code=features[arm];init=prediction(head,num[n:],code[n:],bpred[n:],device);assert np.array_equal(init,bpred[n:])
   nt=torch.tensor(num[:n],device=device);ct=torch.tensor(code[:n],device=device);pt=torch.tensor(bpred[:n],dtype=torch.float64,device=device);yt=torch.tensor(p["y"][:n],dtype=torch.float64,device=device)
   rng=np.random.default_rng(seed+271);opt=torch.optim.Adam(head.parameters(),lr=c["lr"]);curve=[];norm0=None
   for step in range(c["steps"]):
    if time.monotonic()-t>150:raise TimeoutError("teacher readout budget150sec")
    ii=torch.tensor(rng.integers(n,size=c["batch_size"]),device=device);head.train();yp=forward(head,nt[ii],ct[ii],pt[ii]);loss=F.mse_loss(yp,yt[ii]);opt.zero_grad(set_to_none=True);loss.backward()
    if step==0:norm0=float(head.mlp[-1].weight.grad.norm());assert norm0>1e-12
    torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
    if (step+1)%100==0:
     vp=prediction(head,num[n:],code[n:],bpred[n:],device);a,l,s,ma=rt.components(vp,p["y"][n:]);curve.append(dict(step=step+1,mse=float(a.mean()),mae=float(ma.mean()),shape=float(s.mean())))
   pp=prediction(head,num[n:],code[n:],bpred[n:],device);row=score(arm,pp,head);path=r/f"teacher_{arm}_s{seed}.pt";torch.save(head.state_dict(),path)
   trainpp=prediction(head,num[:n],code[:n],bpred[:n],device);np.save(r/f"teacher_{arm}_s{seed}_train.npy",trainpp)
   rec=dict(**row,path=str(path),sha256=rt.sha(path),steps=c["steps"],first_grad_norm=norm0,initial_max_error=float(abs(init-bpred[n:]).max()))
   records.append(rec);save(r/"teacher_records.json",records);csvsave(r/f"teacher_{arm}_s{seed}_curve.csv",curve);print("TEACHER_FIT",json.dumps(rec),flush=True)
   del head,opt,nt,ct,pt,yt
  assert rt.sha(brec["path"])==brec["sha256"] and all(not z.requires_grad and z.grad is None for z in base.parameters())
  del base
 csvsave(r/"scores_by_seed.csv",scores);csvsave(r/"scores_by_window.csv",windowrows);np.savez_compressed(r/"predictions.npz",**outputs)
 means=[]
 for arm in ["FrozenBase"]+c["arms"]:
  qq=[z for z in scores if z["arm"]==arm];out=dict(arm=arm)
  out.update({k:float(np.mean([z[k] for z in qq])) for k in ["mse","mae","level","shape"]})
  out.update({f"seed{s}_mse":next(z["mse"] for z in qq if z["seed"]==s) for s in c["seeds"]});means.append(out)
 csvsave(r/"scores_mean.csv",means);a={z["arm"]:z for z in means};controls=["Fourier","StaticQwen","HistoryShuffleQwen"]
 passed="DynamicQwen" in a and a["DynamicQwen"]["mse"]<a["FrozenBase"]["mse"] and all(a["DynamicQwen"]["mse"]<a[k]["mse"] and sum(a["DynamicQwen"][f"seed{s}_mse"]<a[k][f"seed{s}_mse"] for s in c["seeds"])>=2 for k in controls)
 save(r/"teacher_gate.json",dict(passed=bool(passed),rule=c["gate"],new_distillation_fits=0))
 save(r/"readout_completion.json",dict(seconds=time.monotonic()-t,models=len(records),updates=len(records)*c["steps"],all_initial_predictions_exact=True,all_old_bases_unchanged=True,base_sources=base_sources,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30))
 print("TEACHER_RESULT",json.dumps(dict(means=means,gate=bool(passed))),flush=True)
def plain_fourier_control(c,r,device):
 import shutil
 out=r/"plain_fourier_control";out.mkdir(exist_ok=False)
 shutil.copyfile(r/"data.npz",out/"data.npz")
 p=np.load(r/"data.npz");n=int(p["n_train"]);f=np.load(r/"features.npz")["StaticQwen"][:,:,:32]
 z=np.zeros((len(f),96,168),dtype="float32");z[:,:,:32]=f;z[:,:,32:64]=f
 np.savez_compressed(out/"features.npz",PlainFourier=z)
 cc=dict(c,arms=["PlainFourier"],run=str(out));save(out/"config.json",cc)
 save(out/"control_definition.json",dict(reason="Supplement original multi-harmonic Fourier WITHOUT forced history interactions; retain all original arms, budgets and negative results.",additional32="duplicate common32 Fourier to match input RMS/nominal width; no extra calendar information",training_rms=float(np.sqrt(np.mean(f[:n]**2))),training_rank=32,history_summary_unchanged=True,no_qwen_calls=True,preplanned_primary=False))
 fit(cc,out,device)
 replay=[]
 arrays=np.load(out/"predictions.npz")
 for rec in read(out/"teacher_records.json"):
  seed=rec["seed"];nf=np.load(out/f"numeric_seed{seed}.npz")
  with torch.no_grad():num=numeric_summary(torch.tensor(p["x"]),torch.tensor(nf["p0"],dtype=torch.float32)).numpy()
  num=((num-nf["mu"])/nf["sd"]).astype("float32");head=ResidualReadout(seed,np.zeros((168,168),dtype="float32"));head.load_state_dict(torch.load(rec["path"],map_location="cpu",weights_only=True))
  pp=prediction(head,num[n:],z[n:],nf["p0"][n:],"cpu");diff=float(abs(pp-arrays[f"PlainFourier_s{seed}"]).max());assert diff<2e-5
  replay.append(dict(seed=seed,max_prediction_difference=diff))
 csvsave(out/"cpu_replays.csv",replay)
def analyze(c,r):
 p=np.load(r/"data.npz");n=int(p["n_train"]);pred=np.load(r/"predictions.npz");rows=list(__import__("csv").DictReader((r/"scores_by_window.csv").open()))
 start=rt.ix(c,c["validation"][0]);end=rt.ix(c,c["validation"][1]);days=len(np.arange(start,end-96+1,24));origins=pred["origin"]
 bc=c["bootstrap"];rng=np.random.default_rng(bc["seed"]);b=bc["block_days"];draw=(rng.integers(days,size=(bc["replicates"],int(np.ceil(days/b))))[:,:,None]+np.arange(b))%days;draw=draw.reshape(bc["replicates"],-1)[:,:days];cis=[]
 for other in ["FrozenBase","Fourier","StaticQwen","HistoryShuffleQwen"]:
  a=np.full(days,np.nan)
  for i,o in enumerate(origins):
   dif=[]
   for seed in c["seeds"]:
    q=pred[f"DynamicQwen_s{seed}"][i]-pred["truth"][i];v=pred[f"{other}_s{seed}"][i]-pred["truth"][i];dif.append(float(np.mean(q*q)-np.mean(v*v)))
   a[(int(o)-start)//24]=np.mean(dif)
  vals=a[draw];den=np.isfinite(vals).sum(1);boot=np.nansum(vals,1)/np.maximum(den,1);lo,hi=np.quantile(boot[den>0],[.025,.975])
  cis.append(dict(comparison="DynamicQwen-"+other,mse_difference=float(np.nanmean(a)),lo95=float(lo),hi95=float(hi),unique_origins=len(origins),block_days=b,replicates=bc["replicates"]))
 csvsave(r/"paired_timeblock_intervals.csv",cis)
 cases=[]
 for seed in c["seeds"]:
  d=pred[f"DynamicQwen_s{seed}"]-pred["truth"];f=pred[f"Fourier_s{seed}"]-pred["truth"]
  for i,o in enumerate(origins):cases.append(dict(seed=seed,origin=int(o),date=str(np.datetime64(c["base"],"h")+np.timedelta64(int(o),"h")),dynamic_mse=float(np.mean(d[i]**2)),fourier_mse=float(np.mean(f[i]**2)),difference=float(np.mean(d[i]**2)-np.mean(f[i]**2))))
 csvsave(r/"all_cases.csv",cases)
 # Deterministic CPU replay of all12readouts, without Qwen.
 ff=np.load(r/"features.npz");replays=[]
 for rec in read(r/"teacher_records.json"):
  seed=rec["seed"];nf=np.load(r/f"numeric_seed{seed}.npz");bp=nf["p0"]
  with torch.no_grad():num=numeric_summary(torch.tensor(p["x"]),torch.tensor(bp,dtype=torch.float32)).numpy()
  num=((num-nf["mu"])/nf["sd"]).astype("float32");head=ResidualReadout(seed,np.zeros((168,168),dtype="float32"));head.load_state_dict(torch.load(rec["path"],map_location="cpu",weights_only=True))
  out=prediction(head,num[n:],ff[rec["arm"]][n:],bp[n:],"cpu");ref=pred[f"{rec['arm']}_s{seed}"];diff=float(abs(out-ref).max());assert diff<2e-5
  replays.append(dict(arm=rec["arm"],seed=seed,max_prediction_difference=diff))
 csvsave(r/"cpu_replays.csv",replays);print("ANALYZED",json.dumps(cis),flush=True)
def diagnostics(c,r):
 import csv
 data=np.load(r/"data.npz");n=int(data["n_train"]);source=read(r/"source.json");extra=r/"plain_fourier_control"
 roots=[r]+([extra] if extra.exists() else [])
 combined=list(csv.DictReader((r/"scores_mean.csv").open()))
 if extra.exists():combined += [q for q in csv.DictReader((extra/"scores_mean.csv").open()) if q["arm"]!="FrozenBase"]
 csvsave(r/"comparison.csv",combined)
 fitrows=[];curves=[];allrecords=[]
 for where in roots:
  records=read(where/"teacher_records.json");allrecords+=records
  for rec in records:
   seed=rec["seed"];arm=rec["arm"];nf=np.load(where/f"numeric_seed{seed}.npz");pp=np.load(where/f"teacher_{arm}_s{seed}_train.npy")
   mse,lev,shape,mae=rt.components(pp,data["y"][:n]);bm,bl,bs,ba=rt.components(nf["p0"][:n],data["y"][:n])
   fitrows.append(dict(arm=arm,seed=seed,training_mse=float(mse.mean()),base_training_mse=float(bm.mean()),validation_mse=rec["mse"],training_shape=float(shape.mean()),validation_shape=rec["shape"],steps=rec["steps"]))
   curves+= [dict(arm=arm,seed=seed,**q) for q in csv.DictReader((where/f"teacher_{arm}_s{seed}_curve.csv").open())]
 csvsave(r/"training_vs_validation.csv",fitrows)
 curve_means=[]
 for arm in sorted(set(q["arm"] for q in curves)):
  for step in [100,200,300]:
   qq=[q for q in curves if q["arm"]==arm and int(q["step"])==step];z=dict(arm=arm,step=step)
   z.update({k:float(np.mean([float(q[k]) for q in qq])) for k in ["mse","mae","shape"]});curve_means.append(z)
 csvsave(r/"learning_curves_mean.csv",curve_means)
 stats,ss=data["statistics"],data["shuffled_statistics"];same=np.max(abs(stats-ss),axis=1)<1e-9
 ver=dict(new_teacher_models=len(allrecords),updates=len(allrecords)*c["steps"],all_model_sha_valid=all(rt.sha(q["path"])==q["sha256"] for q in allrecords),all_initial_predictions_exact=all(q["initial_max_error"]==0 for q in allrecords),max_mean_error=max(q["max_mean_error"] for q in allrecords),new_student_fits=0,
  train_unique_origin_days=int(len(np.unique(data["origin"][:n]//24))),validation_unique_origin_days=int(len(np.unique(data["origin"][n:]//24))),
  unchanged_summary_after_block_shuffle_train=int(same[:n].sum()),unchanged_summary_after_block_shuffle_validation=int(same[n:].sum()),
  timebase_source_unchanged=rt.sha("layers/ChronoGeomTimeBaseOfficial.py")==source["timebase_source_sha"],
  extraction_cache_unchanged=rt.sha(r/"dynamic_raw_layer24.npy")==read(r/"extraction.json")["raw_sha"],
  static_cache_unchanged=rt.sha(source["static_path"])==source["static_sha"])
 assert ver["all_model_sha_valid"] and ver["all_initial_predictions_exact"] and ver["max_mean_error"]<1e-12 and ver["timebase_source_unchanged"] and ver["extraction_cache_unchanged"] and ver["static_cache_unchanged"]
 save(r/"verification.json",ver)
 print("DIAGNOSTICS",json.dumps(dict(verification=ver,train_vs_val=fitrows,learning_curves=curve_means)),flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["prepare","extract","features","fit","analyze","plain","all"]);parser.add_argument("--device",default="cuda:0");args=parser.parse_args()
 c=read("docs/chronogeom/dynamic_teacher01_config.json");r=Path(c["run"]);t=time.monotonic()
 try:
  stages=["prepare","extract","features","fit","analyze"] if args.stage=="all" else [args.stage]
  for stage in stages:
   if time.monotonic()-t>c["workflow_cap_seconds"]:raise TimeoutError("workflow cap")
   if stage=="prepare":prep(c,r)
   elif stage=="extract":extract(c,r,args.device)
   elif stage=="features":makefeatures(c,r)
   elif stage=="fit":fit(c,r,args.device)
   elif stage=="plain":plain_fourier_control(c,r,args.device)
   else:analyze(c,r);diagnostics(c,r)
  if args.stage=="all":save(r/"completion.json",dict(seconds=time.monotonic()-t,git=rt.git(),new_llm_prompts=288,llm_training=False,new_teacher_fits=12,new_student_fits=0,gate=read(r/"teacher_gate.json")["passed"],all_evaluation_previously_viewed=True))
 except BaseException:
  if r.exists():save(r/f"failure_{args.stage}.json",dict(traceback=traceback.format_exc()))
  raise
if __name__=="__main__":main()
