"""Pilot02: frozen Qwen task contexts and unchanged TimeBase, two real domains."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,csv,json,hashlib,subprocess,time,traceback,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from exp.exp_chronogeom_downstream import CalendarTimeBase,predict
from exp.exp_chronogeom_qwen_capital import windows,tensorpack,setup
from scripts.chronogeom.task_context02_analysis import save,read,csvsave
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git():return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
def ix(c,domain,date):return int((np.datetime64(date,"h")-np.datetime64(c["domains"][domain]["base"],"h"))/np.timedelta64(1,"h"))
def extract(c,r,device):
 from transformers import AutoTokenizer,AutoModelForCausalLM
 out=r/"electricity_cache";out.mkdir(exist_ok=False)
 free,total=torch.cuda.mem_get_info(device);assert free>=22*2**30
 setup(device,22.);t=time.monotonic();tok=AutoTokenizer.from_pretrained(c["model_path"],local_files_only=True);tok.padding_side="left"
 prompts=[];days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
 for d in range(7):
  for h in range(24):
   text=c["context_templates"]["Electricity"].format(day=days[d],hour=h)
   chat=tok.apply_chat_template([{"role":"user","content":text}],tokenize=False,add_generation_prompt=True,enable_thinking=False);assert "<think>\n\n</think>" in chat
   prompts.append(dict(day=d,hour=h,text=text,chat=chat))
 save(out/"prompts.json",prompts)
 model=AutoModelForCausalLM.from_pretrained(c["model_path"],dtype=torch.bfloat16,device_map={"":device},local_files_only=True,attn_implementation="sdpa");model.eval()
 for p in model.parameters():p.requires_grad_(False)
 loaded=time.monotonic();buffer={};zs=[]
 def hook(mod,inp,result):
  z=result[0] if isinstance(result,tuple) else result;buffer["z"]=z[:,-1].detach().float().cpu().numpy()
 handle=model.model.layers[c["layer"]-1].register_forward_hook(hook)
 with torch.inference_mode():
  for j in range(0,168,4):
   if time.monotonic()-t>c["extract_cap_seconds"]:raise TimeoutError("Extraction budget")
   tokens=tok([p["chat"] for p in prompts[j:j+4]],padding=True,return_tensors="pt").to(device);model.model(**tokens,use_cache=False,return_dict=True);zs.append(buffer["z"])
 handle.remove();raw=np.concatenate(zs);assert raw.shape==(168,4096) and np.isfinite(raw).all();np.save(out/"layer24.npy",raw)
 save(out/"completion.json",dict(commit=git(),revision=c["revision"],layer=c["layer"],prompts=168,old_generic_bicycle_prompts_reused=336,nonthinking=True,llm_training=False,generated_tokens=0,load_seconds=loaded-t,total_seconds=time.monotonic()-t,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,raw_sha=sha(out/"layer24.npy")));print("EXTRACTED",read(out/"completion.json"),flush=True)
def build_tables(c,r):
 old=Path(c["old_run"]);projection=np.load(old/"projection.npz");basis=projection["basis"];mix=projection["mix"];raw=np.load(old/"raw_qwen.npz")["layer24"].astype("float64");e=np.load(r/"electricity_cache/layer24.npy").astype("float64")
 zs={};retained=[]
 for name,v in [("Generic32",raw[:168]),("Bicycle32",raw[168:336]),("Electricity32",e)]:
  v=v-v.mean(0);z=v@basis.T;zs[name]=z;retained.append(dict(context=name,raw_variance_retained=float(np.sum(z*z)/np.sum(v*v)),native_dimension=32))
 perm=projection["permutation"];zs["BicycleShuffle32"]=zs["Generic32"]+(zs["Bicycle32"]-zs["Generic32"])[perm];zs["ElectricityShuffle32"]=zs["Generic32"]+(zs["Electricity32"]-zs["Generic32"])[perm]
 previous=np.load(old/"tables.npz");aliases={"NoCalendar":"NoCalendar","Fourier32":"Fourier32","Calendar32":"Calendar32","Joint168":"Joint168","Generic32":"QwenGeneric32","Bicycle32":"QwenTask32","BicycleShuffle32":"QwenContextShuffle32"}
 tables={name:previous[key] for name,key in aliases.items()}
 for name in ["Electricity32","ElectricityShuffle32"]:
  z=zs[name]-zs[name].mean(0);z/=np.sqrt(np.mean(z*z));pad=np.zeros((168,168));pad[:,:32]=z;table=(pad@mix).astype("float32");table*=np.sqrt(168/32);tables[name]=table
 diag=[]
 for name,z in tables.items():
  sv=np.linalg.svd(z.astype(float)-z.mean(0),compute_uv=False);p=sv*sv;p=p[p>1e-15];p=p/p.sum() if len(p) else p
  diag.append(dict(arm=name,dimension=168,native_dimension=168 if name=="Joint168" else 1 if name=="NoCalendar" else 32,rank=int(np.sum(sv>max(float(sv[0]),1e-12)*1e-7)),entropy_rank=float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.,rms=float(np.sqrt(np.mean(z*z))),min_norm=float(np.linalg.norm(z,axis=1).min()),max_norm=float(np.linalg.norm(z,axis=1).max())))
  if name!="NoCalendar":assert abs(diag[-1]["rms"]-1)<1e-6
 np.savez_compressed(r/"tables.npz",**tables);np.savez_compressed(r/"native_contexts.npz",**zs);csvsave(r/"feature_diagnostics.csv",diag);csvsave(r/"projection_retention.csv",retained)
 save(r/"tables_source.json",dict(commit=git(),old_basis_sha=sha(old/"projection.npz"),old_raw_sha=sha(old/"raw_qwen.npz"),old_tables_sha=sha(old/"tables.npz"),electricity_raw_sha=sha(r/"electricity_cache/layer24.npy"),tables_sha=sha(r/"tables.npz"),capital_2024_eight_tables_byte_identical=True))
 print("PROJECTION_RETENTION",retained,flush=True)
def grid(c,domain,end):
 dc=c["domains"][domain]
 if domain=="capital":
  from exp.exp_chronogeom_qwen_capital import grid as oldgrid
  return oldgrid(dict(data_path=dc["path"]),end)
 dates=pd.date_range(dc["base"],end,freq="h",inclusive="left");sums=np.zeros(len(dates));counts=np.zeros(len(dates),dtype="int64");previous=None
 for chunk in pd.read_csv(dc["path"],sep=";",usecols=["Date","Time","Global_active_power"],dtype=str,keep_default_na=False,chunksize=200000):
  ts=pd.to_datetime(chunk["Date"]+" "+chunk["Time"],format="%d/%m/%Y %H:%M:%S")
  assert (ts.diff().dropna()==pd.Timedelta(minutes=1)).all()
  if previous is not None:assert ts.iloc[0]-previous==pd.Timedelta(minutes=1)
  previous=ts.iloc[-1];mask=ts<pd.Timestamp(end)
  if not mask.any():break
  # Convert target values only before the frozen boundary.
  vals=pd.to_numeric(chunk.loc[mask,"Global_active_power"],errors="coerce").to_numpy()
  hours=((ts[mask]-dates[0])/pd.Timedelta(hours=1)).to_numpy().astype("int64");good=np.isfinite(vals)
  sums+=np.bincount(hours[good],weights=vals[good],minlength=len(dates));counts+=np.bincount(hours[good],minlength=len(dates))
  if not mask.all():break
 values=np.full((len(dates),1),np.nan,dtype="float32");valid=counts==60;values[valid,0]=(sums[valid]/60).astype("float32")
 ids=(dates.dayofweek.to_numpy()*24+dates.hour.to_numpy()).astype("int64")
 return values,ids,dates
def prepare(c,r):
 build_tables(c,r)
 for domain,dc in c["domains"].items():
  out=r/domain;out.mkdir(exist_ok=False)
  if domain=="capital":shutil.copyfile(Path(c["old_run"])/"train_val.npz",out/"train_val.npz")
  else:
   values,ids,dates=grid(c,domain,dc["val_end"]);lo,hi=ix(c,domain,dc["train_start"]),ix(c,domain,dc["train_end"]);mu=np.nanmean(values[lo:hi],axis=0);sd=np.nanstd(values[lo:hi],axis=0);assert sd[0]>1e-6
   normalized=(values-mu)/sd;origins=np.arange(lo+96,hi-96+1);origins=origins[np.array([np.isfinite(normalized[o-96:o+96]).all() for o in origins])];assert len(origins)>100
   np.savez_compressed(out/"train_val.npz",values=normalized,ids=ids,mean=mu,sd=sd,train_origins=origins)
  b=np.load(out/"train_val.npz");p,counts=windows(b["values"],b["ids"],ix(c,domain,dc["train_end"]),ix(c,domain,dc["val_end"]))
  save(out/"prepared.json",dict(commit=git(),domain=domain,training_origins=len(b["train_origins"]),validation=counts,mean=b["mean"].tolist(),sd=b["sd"].tolist(),data_sha=sha(dc["path"]),prepared_sha=sha(out/"train_val.npz"),new_target_boundary=dc["val_end"],test_opened=False));print("PREPARED",domain,read(out/"prepared.json"),flush=True)
 save(r/"source.json",dict(commit=git(),main_source_sha=sha(__file__),common_timebase_wrapper_sha=sha("exp/exp_chronogeom_downstream.py"),timebase_sha=sha("layers/ChronoGeomTimeBaseOfficial.py"),config_sha=sha("docs/chronogeom/task_context02_config.json"),tables_sha=sha(r/"tables.npz")))
def fit(c,r,domain,seed,device):
 setup(device);dc=c["domains"][domain];dr=r/domain;out=dr/f"seed{seed}";out.mkdir(exist_ok=False);b=np.load(dr/"train_val.npz");tabs=np.load(r/"tables.npz")
 va,_=windows(b["values"],b["ids"],ix(c,domain,dc["train_end"]),ix(c,domain,dc["val_end"]));vp=tensorpack(va,device);values=torch.tensor(b["values"],device=device);idsall=torch.tensor(b["ids"],device=device);records=[];start=time.monotonic()
 aliases={"Original":"Original","NoCalendar":"NoCalendar","Fourier32":"Fourier32","Calendar32":"Calendar32","Joint168":"Joint168","Generic32":"QwenGeneric32","Bicycle32":"QwenTask32","BicycleShuffle32":"QwenContextShuffle32"}
 oldrecords=read(Path(c["old_run"])/"fit/records.json") if domain=="capital" and seed==2024 else []
 for arm in c["arms"]:
  if oldrecords and arm in aliases:
   oldarm=aliases[arm];rec=copy.deepcopy(next(q for q in oldrecords if q["arm"]==oldarm));rec.update(arm=arm,reused=True,new_seconds=0.);assert sha(rec["path"])==rec["sha256"]
   if arm!="Original":
    state=torch.load(rec["path"],map_location="cpu",weights_only=True);assert torch.equal(state["table"],torch.as_tensor(tabs[arm]))
   shutil.copyfile(Path(c["old_run"])/"fit"/(oldarm+"_val.npy"),out/(arm+"_val.npy"));records.append(rec);save(out/"records.json",records);continue
  m=CalendarTimeBase(c,seed,None if arm=="Original" else tabs[arm]).to(device);opt=torch.optim.Adam(m.parameters(),lr=c["lr"]);rng=np.random.default_rng(seed+271);curve=[];best=float("inf");tick=time.monotonic()
  for step in range(601):
   if time.monotonic()-start>c["fit_cap_seconds"]:raise TimeoutError("Per worker budget")
   if step%100==0:
    pred=predict(m,vp);vm=float(np.mean((pred-va["y"])**2));ma=float(np.mean(abs(pred-va["y"])));assert np.isfinite(vm);curve.append(dict(step=step,mse=vm,mae=ma))
    if vm<best:best=vm;beststep=step;state=copy.deepcopy(m.state_dict())
   if step==600:break
   o=torch.tensor(rng.choice(b["train_origins"],size=32),device=device);pa=o[:,None]+torch.arange(-96,0,device=device);fu=o[:,None]+torch.arange(96,device=device)
   m.train();loss=F.mse_loss(m(values[pa],idsall[fu]),values[fu]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
  m.load_state_dict(state);path=out/(arm+".pt");torch.save(state,path);pred=predict(m,vp);np.save(out/(arm+"_val.npy"),pred);csvsave(out/(arm+"_curve.csv"),curve)
  rec=dict(arm=arm,seed=seed,best_step=beststep,val_mse=best,val_mae=float(np.mean(abs(pred-va["y"]))),parameters=sum(p.numel() for p in m.parameters()),path=str(path),sha256=sha(path),step0_mse=curve[0]["mse"],new_seconds=time.monotonic()-tick,reused=False);records.append(rec);save(out/"records.json",records);print("FIT",domain,json.dumps(rec),flush=True);del m,opt
 assert len(records)==10
 save(out/"completion.json",dict(commit=git(),domain=domain,seed=seed,new_models=sum(not q["reused"] for q in records),reused_models=sum(q["reused"] for q in records),seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated(device)/2**30));print("FIT_COMPLETE",read(out/"completion.json"),flush=True)
def lock(c,r):
 lock=dict(commit=git(),tables_sha=sha(r/"tables.npz"),domains={});total=0
 for domain in c["domains"]:
  recs=[]
  for seed in c["seeds"]:
   recs+=read(r/domain/f"seed{seed}/records.json");total+=read(r/domain/f"seed{seed}/completion.json")["seconds"]
  assert len(recs)==30
  for m in recs:assert sha(m["path"])==m["sha256"]
  means={arm:float(np.mean([m["val_mse"] for m in recs if m["arm"]==arm])) for arm in c["arms"]};strong=min(c["strong_candidates"],key=means.get)
  lock["domains"][domain]=dict(models=recs,validation_means=means,strong=strong,confirmation_is_new=domain=="power")
 assert total<c["total_fit_cap_seconds"];lock["new_fit_worker_wall_seconds_sum"]=total;save(r/"lock.json",lock);print("LOCKED",json.dumps({k:{"strong":v["strong"],"validation":v["validation_means"]} for k,v in lock["domains"].items()}),flush=True)
def evaluate(c,r,device):
 setup(device);lock=read(r/"lock.json");tables=np.load(r/"tables.npz")
 for domain,dc in c["domains"].items():
  b=np.load(r/domain/"train_val.npz")
  for stage in ["validation","pilot","confirmation"]:
   out=r/domain/stage;out.mkdir(exist_ok=False)
   for rec in lock["domains"][domain]["models"]:assert sha(rec["path"])==rec["sha256"]
   save(out/"before_targets.json",dict(commit=git(),models_sha_locked=True,all_domains_locked=True,previously_consumed=domain=="capital",stage=stage))
   lo,hi=(ix(c,domain,dc["train_end"]),ix(c,domain,dc["val_end"])) if stage=="validation" else (ix(c,domain,dc["val_end"]),ix(c,domain,dc["pilot_end"])) if stage=="pilot" else (ix(c,domain,dc["pilot_end"]),ix(c,domain,dc["confirmation_end"]))
   if stage=="validation":values,ids=b["values"],b["ids"]
   else:
    raw,ids,dates=grid(c,domain,dc["pilot_end"] if stage=="pilot" else dc["confirmation_end"]);values=(raw-b["mean"])/b["sd"]
   p,counts=windows(values,ids,lo,hi);vp=tensorpack(p,device);preds={"Persistence":np.repeat(p["x"][:,-1:],96,axis=1),"Seasonal24":np.tile(p["x"][:,-24:],(1,4,1))};scores=[]
   for rec in lock["domains"][domain]["models"]:
    arm,seed=rec["arm"],rec["seed"]
    if stage=="validation":pred=np.load(r/domain/f"seed{seed}"/(arm+"_val.npy"))
    else:
     model=CalendarTimeBase(c,seed,None if arm=="Original" else tables[arm]).to(device);state=torch.load(rec["path"],map_location=device,weights_only=True)
     if arm!="Original":assert torch.equal(state["table"],torch.tensor(tables[arm],device=device))
     model.load_state_dict(state);pred=predict(model,vp);del model,state
    assert np.isfinite(pred).all();preds[f"{arm}_s{seed}"]=pred
   for name,pred in preds.items():
    e=pred-p["y"];mse=float(np.mean(e**2));mae=float(np.mean(abs(e)));scores.append(dict(model=name,mse=mse,mae=mae,raw_mse=mse*float(b["sd"][0])**2,raw_mae=mae*float(b["sd"][0]),n_origins=len(p["origin"])))
   csvsave(out/"scores.csv",scores);np.savez_compressed(out/"predictions.npz",**preds,truth=p["y"],origin=p["origin"])
   save(out/"completion.json",dict(commit=git(),domain=domain,stage=stage,counts=counts,lo=lo,hi=hi,new_confirmation=domain=="power" and stage=="confirmation",models_unchanged=all(sha(m["path"])==m["sha256"] for m in lock["domains"][domain]["models"])))
   print("EVALUATED",domain,stage,json.dumps(scores),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=["extract","prepare","fit","lock","evaluate"]);p.add_argument("--config",default="docs/chronogeom/task_context02_config.json");p.add_argument("--domain",choices=["capital","power"]);p.add_argument("--seed",type=int);p.add_argument("--device",default="cuda:0");args=p.parse_args();c=read(args.config);r=Path(c["run"]);r.mkdir(parents=True,exist_ok=True)
 try:
  if args.stage=="extract":extract(c,r,args.device)
  elif args.stage=="prepare":prepare(c,r)
  elif args.stage=="fit":fit(c,r,args.domain,args.seed,args.device)
  elif args.stage=="lock":lock(c,r)
  else:evaluate(c,r,args.device)
 except BaseException as e:
  save(r/f"failure_{args.stage}_{args.domain}_{args.seed}.json",dict(error=str(e),traceback=traceback.format_exc()));raise
if __name__=="__main__":main()
