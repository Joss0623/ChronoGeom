"""History-conditioned Qwen day-shape pilot65 in TSLib. Existing teacher/student readouts."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",
 TORCH_DEVICE_BACKEND_AUTOLOAD="0",CUBLAS_WORKSPACE_CONFIG=":4096:8")
import argparse,json,time,traceback
from pathlib import Path
import numpy as np,torch
from exp.exp_chronogeom_strong_teacher64 import PredictionTeacher,load_base,choose
from exp.exp_chronogeom_calendar_level_direct63 import to_pack,pred,center,statecpu,writecsv,interval
from exp.exp_chronogeom_teacher_content40 import calendar_arrays,check22
from exp.exp_chronogeom_event_knowledge19 import read,save,sha,git,scores
BIN_CENTERS=np.arange(48)*2+.5
ARMS=["UpdatedNoKD","HistoryFourier","HistoryQwen","ShuffledKnowledge"]
def interpolate(q):
 return np.interp(np.arange(96),BIN_CENTERS,np.asarray(q,float).reshape(48))
def profile_channels(q):
 q=np.asarray(q,float);q=q-q.mean(-1,keepdims=True)
 d=np.diff(q,axis=-1,prepend=q[...,:1]);dd=np.diff(d,axis=-1,prepend=d[...,:1])
 return np.stack([q,d,dd],-1)
def parse_answer(text):
 ob=json.loads(text[text.index("{"):text.rindex("}")+1]);q=np.asarray(ob["shape"],dtype=float)
 assert q.shape in [(48,),(4,12)] and np.isfinite(q).all()
 return q.reshape(48),ob
def condition_prompt(c,x,base,past_days,future_days,history_start,forecast_start):
 x=np.asarray(x,float).reshape(96);base=np.asarray(base,float).reshape(96)
 m=float(x.mean());s=max(float(x.std()),1e-3)
 inp=dict(history_start=str(history_start),forecast_start=str(forecast_start),
  observed_history=np.round((x-m)/s,2).tolist(),timebase_forecast=np.round((base-m)/s,2).tolist(),
  observed_day_calendar=past_days,known_future_day_calendar=future_days)
 return c["prompt"].replace("{INPUT}",json.dumps(inp,separators=(",",":"))),m,s
def history_fourier(x,base,past_calendar,future_calendar,ridge):
 x=np.asarray(x,float).reshape(96);base=np.asarray(base,float).reshape(96)
 m=float(x.mean());s=max(float(x.std()),1e-3)
 dp=np.c_[past_calendar,np.arange(-96,0)/96,np.ones(96)]
 df=np.c_[future_calendar,np.arange(96)/96,np.ones(96)]
 lam=ridge*96;eye=np.eye(dp.shape[1])*lam
 beta0=np.linalg.solve(df.T@df+eye,df.T@((base-m)/s))
 beta=np.linalg.solve(dp.T@dp+eye,dp.T@((x-m)/s)+lam*beta0)
 return (df@beta).reshape(48,2).mean(1)
def prepare(c,config):
 torch.set_num_threads(2)
 r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c)
 assets=check22(c);seen={v["path"] for v in assets}
 for key in ["source51","source63","source64"]:
  for p in Path(c[key]).iterdir():
   if p.is_file() and str(p) not in seen:assets.append(dict(path=str(p),sha=sha(p)));seen.add(str(p))
 for p in [Path(c["domains"]["Metro"]["source"])/"data.npz",Path(c["domains"]["Metro"]["source"])/"config.json",
           Path("exp/exp_chronogeom_calendar_level_direct63.py"),Path("exp/exp_chronogeom_strong_teacher64.py")]:
  if str(p) not in seen:assets.append(dict(path=str(p),sha=sha(p)));seen.add(str(p))
 save(r/"protected_assets.json",assets)
 src=Path(c["domains"]["Metro"]["source"]);a=np.load(src/"data.npz")
 grid,raw,common,kind,h=calendar_arrays(read(src/"config.json"),"Metro")
 old=np.array(read(Path(c["source63"])/"time_splits.json")["Metro"]["selection_indices"])
 fit=np.linspace(0,len(a["fit_x"])-1,12,dtype=int)
 si=np.linspace(0,12,6,dtype=int);sel=old[si]
 max_target=int(a["val_future"][sel].max())
 candidates=old[a["val_origin"][old]-96>max_target]
 assert len(candidates)>=3
 check=candidates[np.linspace(0,len(candidates)-1,3,dtype=int)]
 split=dict(fit=dict(source="fit",indices=fit.tolist()),selection=dict(source="val",indices=sel.tolist()),
            check=dict(source="val",indices=check.tolist()))
 assert a["val_origin"][check].min()-96>max_target
 for phase,v in split.items():
  v["origins"]=grid[a[v["source"]+"_origin"][v["indices"]]].astype(str).tolist()
  v["unique_target_hours"]=int(len(np.unique(a[v["source"]+"_future"][v["indices"]])))
 save(r/"split.json",split)
 calendar=np.concatenate([raw,common],-1)
 def days(ix):
  return [dict(date=str(grid[int(ix[j])].date()),weekday=int(grid[int(ix[j])].dayofweek),
   holiday=int(raw[int(ix[j]),0]),first_hour=int(grid[int(ix[j])].hour)) for j in range(0,96,24)]
 for seed in c["seeds"]:
  base=load_base(c,seed,"cpu");values={};prompts=[]
  for phase,v in split.items():
   indices=np.array(v["indices"]);pack=to_pack(a,v["source"],grid,raw,common,kind*24+h,"cpu",indices)
   forecast=pred(base,pack)
   with torch.no_grad():level=base.level_prediction(pack["x"],pack["calendar"]).numpy()
   prefix=phase+"_"
   for key in ["x","calendar","ids"]:
    values[prefix+key]=pack[key].numpy()
   values[prefix+"base"]=forecast;values[prefix+"level"]=level;values[prefix+"origin"]=pack["origin"]
   values[prefix+"future"]=pack["future"];values[prefix+"daykind"]=kind[pack["origin"]]
   means=[];scales=[];fourier=[];no=[]
   for j,o in enumerate(pack["origin"]):
    past=np.arange(int(o)-96,int(o));future=pack["future"][j]
    assert int(future.min())==int(o) and int(past.max())<int(o)
    prompt,m,s=condition_prompt(c,pack["x"][j].numpy(),forecast[j],days(past),days(future),grid[past[0]],grid[o])
    prompts.append(dict(seed=seed,phase=phase,index=j,origin=str(grid[o]),prompt=prompt))
    means.append(m);scales.append(s)
    fourier.append(history_fourier(pack["x"][j].numpy(),forecast[j],calendar[past],calendar[future],c["ridge"]))
    no.append(((forecast[j,:,0]-m)/s).reshape(48,2).mean(1))
   values[prefix+"mean"]=np.array(means);values[prefix+"sd"]=np.array(scales)
   values[prefix+"fourier48"]=np.array(fourier);values[prefix+"nokd48"]=np.array(no)
  np.savez_compressed(r/f"inputs_{seed}.npz",**values);save(r/f"prompts_{seed}.json",prompts)
 save(r/"protocol_lock.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),
  labels_accessed=False,outer_arrays_accessed=False,windows_per_seed=len(prompts)))
 print("PREPARED65",split,"protected",len(assets),flush=True)
def extract(c,seed):
 from transformers import AutoTokenizer,AutoModelForCausalLM
 torch.set_num_threads(2);r=Path(c["run"]);out=r/f"qwen_{seed}";out.mkdir(exist_ok=False)
 assert sha(__file__)==read(r/"protocol_lock.json")["source_sha"]
 inputs=np.load(r/f"inputs_{seed}.npz");prompts=read(r/f"prompts_{seed}.json")
 device="cuda:0";torch.cuda.set_device(device)
 assert torch.cuda.mem_get_info()[0]>=22*2**30
 torch.cuda.set_per_process_memory_fraction(c["llm_memory_gib"]*2**30/torch.cuda.get_device_properties(0).total_memory)
 began=time.monotonic();save(out/"progress.json",dict(stage="loading",completed=0))
 tok=AutoTokenizer.from_pretrained(c["model"],local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(c["model"],local_files_only=True,dtype=torch.bfloat16,
   device_map={"":device},attn_implementation="sdpa").eval()
 for p in model.parameters():p.requires_grad_(False)
 records=[];q={}
 save(out/"progress.json",dict(stage="generating",completed=0,load_seconds=time.monotonic()-began))
 for row in prompts:
  phase=row["phase"];i=row["index"];record=dict(row);valid=False
  fallback=inputs[phase+"_nokd48"][i];shape=fallback
  if time.monotonic()-began>=c["llm_seconds_per_gpu"]:
   record.update(valid=False,reason="activity_budget_exhausted",attempted=False)
  else:
   chat=tok.apply_chat_template([dict(role="user",content=row["prompt"])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   tokenized=tok(chat,return_tensors="pt").to(device)
   try:
    with torch.inference_mode():
     answer=model.generate(**tokenized,do_sample=False,max_new_tokens=c["max_new_tokens"],max_time=max(1.,c["llm_seconds_per_gpu"]-(time.monotonic()-began)),pad_token_id=tok.eos_token_id)
    generated=answer[0,tokenized["input_ids"].shape[1]:];text=tok.decode(generated,skip_special_tokens=True)
    record.update(answer=text,input_tokens=int(tokenized["input_ids"].shape[1]),output_tokens=int(len(generated)),attempted=True)
    shape,obj=parse_answer(text);valid=True;record["structured"]=obj
    record.update(valid=True,reason="valid")
   except Exception as exc:
    record.update(valid=False,reason=type(exc).__name__+":"+str(exc),attempted=True)
  q.setdefault(phase,[]).append(np.asarray(shape,float));record["fallback_used"]=not valid
  save(out/(phase+"_"+str(i)+".json"),record);records.append(record)
  save(out/"progress.json",dict(stage="generating",completed=len(records),valid=sum(int(v["valid"]) for v in records),
                              seconds=time.monotonic()-began))
  print("QWEN65",seed,phase,i,"valid",valid,flush=True)
 np.savez_compressed(out/"profiles.npz",**{phase:np.array(v) for phase,v in q.items()})
 save(out/"records.json",records)
 save(out/"completion.json",dict(complete=True,seed=seed,physical_gpu=c["gpu_by_seed"][str(seed)],GPU=str(torch.cuda.get_device_properties(0).uuid),
  seconds=time.monotonic()-began,valid=sum(int(v["valid"]) for v in records),total=len(records),
  attempted=sum(int(v["attempted"]) for v in records),peak_GiB=torch.cuda.max_memory_allocated()/2**30,
  frozen=True,thinking=False,new_model_downloads=0,revision=c["revision"]))
def read_targets(c,split,phases=None):
 a=np.load(Path(c["domains"]["Metro"]["source"])/"data.npz")
 return {phase:a[v["source"]+"_y"][v["indices"]] for phase,v in split.items() if phases is None or phase in phases}
def fit(c,config):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);r=Path(c["run"]);start=time.monotonic()
 assert sha(__file__)==read(r/"protocol_lock.json")["source_sha"] and sha(config)==read(r/"protocol_lock.json")["config_sha"]
 assets=read(r/"protected_assets.json")
 for v in assets:assert sha(v["path"])==v["sha"]
 for seed in c["seeds"]:assert read(r/f"qwen_{seed}/completion.json")["complete"]
 save(r/"extraction_lock.json",{str(seed):dict(profile_sha=sha(r/f"qwen_{seed}/profiles.npz"),records_sha=sha(r/f"qwen_{seed}/records.json")) for seed in c["seeds"]})
 split=read(r/"split.json");all_targets=read_targets(c,split,["fit","selection"]);device=c["device"]
 assert torch.cuda.mem_get_info(device)[0]>=c["gpu_memory_gib"]*2**30
 torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(c["gpu_memory_gib"]*2**30/torch.cuda.get_device_properties(device).total_memory,device)
 data={};states={};curves=[];diagnostics=[];shuffle_records=[];norms={};raw_predictions={};cached={};baselines={}
 blocks=np.array_split(np.arange(6),3);tables=read(Path(c["source63"])/"knowledge_tables.json")["Metro"]
 for seed in c["seeds"]:
  base=load_base(c,seed,device);inp=np.load(r/f"inputs_{seed}.npz");qq=np.load(r/f"qwen_{seed}/profiles.npz")
  d=dict(base=base,inputs=inp,features={},val_features={},all_features={},qprofiles={},packs={})
  raw_predictions[seed]={};norms[seed]={};cached[seed]={}
  originfit=inp["fit_origin"];rng=np.random.default_rng(c["shuffle_seed"])
  for phase in split:
   origins=inp[phase+"_origin"];kinds=inp[phase+"_daykind"];shuffled=[]
   for i,o in enumerate(origins):
    eligible=np.flatnonzero(originfit<o)
    same=eligible[inp["fit_daykind"][eligible]==kinds[i]]
    source_pool=same if len(same) else eligible
    j=int(rng.choice(source_pool)) if len(source_pool) else 0
    assert originfit[j]<o or (phase=="fit" and i==0 and j==0)
    shuffled.append(qq["fit"][j])
    shuffle_records.append(dict(seed=seed,phase=phase,index=i,source_fit_index=j,
      earlier_history=bool(originfit[j]<o),first_fit_self=bool(phase=="fit" and i==0)))
   qraw=dict(UpdatedNoKD=inp[phase+"_nokd48"],HistoryFourier=inp[phase+"_fourier48"],
             HistoryQwen=qq[phase],ShuffledKnowledge=np.array(shuffled))
   d["qprofiles"][phase]={arm:np.array([interpolate(v) for v in values]) for arm,values in qraw.items()}
   d["packs"][phase]={key:torch.tensor(inp[phase+"_"+key],device=device,dtype=torch.long if key=="ids" else torch.float32)
                       for key in ["x","calendar","ids"]}
   for arm in ARMS:
    q=d["qprofiles"][phase][arm]
    raw_predictions[seed][(phase,arm)]=inp[phase+"_level"]+inp[phase+"_sd"][:,None,None]*(q-q.mean(1,keepdims=True))[:,:,None]
  for arm in ARMS:
   kfit=profile_channels(d["qprofiles"]["fit"][arm]);mu=kfit.mean((0,1));sd=np.maximum(kfit.std((0,1)),1e-3)
   z=(kfit-mu)/sd;rms=max(float(np.sqrt(np.mean(z*z))),1e-6)
   norms[seed][arm]=dict(mean=mu.tolist(),sd=sd.tolist(),rms=rms)
   sv=np.linalg.svd(z.reshape(-1,3)-z.reshape(-1,3).mean(0),compute_uv=False)
   diagnostics.append(dict(seed=seed,arm=arm,input_dimensions=3,rank=int((sv>max(sv[0],1e-12)*1e-6).sum()),
                            fit_RMS=float(np.sqrt(np.mean((z/rms)**2)))))
   d["all_features"][arm]={};model=PredictionTeacher(base,torch.zeros(72,3,device=device)).to(device)
   for phase in split:
    pack=d["packs"][phase]
    with torch.no_grad():z0,off,level=model.features(pack["x"],pack["calendar"],pack["ids"])
    kn=(profile_channels(d["qprofiles"][phase][arm])-mu)/sd/rms
    z0=z0.detach();z0[:,:,39:]=torch.tensor(kn,device=device,dtype=torch.float32)
    d["all_features"][arm][phase]=(z0,off.detach(),level.detach())
   d["features"][arm]=d["all_features"][arm]["fit"];d["val_features"][arm]=d["all_features"][arm]["selection"]
  for phase in split:
   cached[seed][phase]={k:v.detach().cpu().numpy() for k,v in d["packs"][phase].items()}
   for arm in ARMS:
    for j,v in enumerate(d["all_features"][arm][phase]):cached[seed][f"{phase}_{arm}_{j}"]=v.detach().cpu().numpy()
  baselines[seed]=inp["selection_base"]
  data[seed]=d;states[seed]={}
  batches=np.random.default_rng(seed).integers(0,12,size=(c["steps"],c["batch"]))
  yf=torch.tensor(all_targets["fit"],device=device,dtype=torch.float32)
  for arm in ARMS:
   model=PredictionTeacher(base,torch.zeros(72,3,device=device)).to(device);opt=torch.optim.Adam(model.reader.parameters(),lr=c["lr"])
   states[seed][arm]={}
   for step in range(c["steps"]+1):
    assert time.monotonic()-start<c["fit_seconds_cap"],"pilot65fit cap"
    if step in c["checkpoints"]:
     states[seed][arm][step]=statecpu(model)
     with torch.no_grad():p=model.read_features(*d["val_features"][arm]).cpu().numpy()
     for bi,b in enumerate(blocks):curves.append(dict(seed=seed,arm=arm,step=step,block=bi,windows=len(b),**scores(p[b],all_targets["selection"][b])))
    if step==c["steps"]:break
    idx=torch.tensor(batches[step],device=device);p=model.read_features(*(v[idx] for v in d["features"][arm]))
    loss=(p-yf[idx]).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    assert torch.isfinite(loss)
   print("TEACHER65",seed,arm,"600steps",flush=True)
 choices={arm:choose(c,curves,baselines,all_targets["selection"],blocks,arm) for arm in ARMS}
 chosen={s:{arm:states[s][arm][choices[arm]["step"]] for arm in ARMS} for s in c["seeds"]}
 torch.save(states,r/"all_teacher_checkpoints.pt");torch.save(chosen,r/"teachers.pt")
 save(r/"teacher_selection.json",choices);save(r/"curves.json",curves);save(r/"feature_normalizers.json",norms)
 save(r/"knowledge_diagnostics.json",diagnostics);save(r/"shuffle_sources.json",shuffle_records)
 for seed,d in data.items():
  np.savez_compressed(r/f"features_{seed}.npz",**{k:v for k,v in cached[seed].items() if isinstance(v,np.ndarray)})
 save(r/"teacher_lock.json",dict(teachers_sha=sha(r/"teachers.pt"),selection_sha=sha(r/"teacher_selection.json"),
      before_later_check_scoring=True,before_any_outer_scoring=True))
 all_targets.update(read_targets(c,split,["check"]))
 rows=[];preds={};rawrows=[];fit_targets={}
 for phase in split:
  y=all_targets[phase];preds[phase+"_y"]=y
  for seed,d in data.items():
   inp=d["inputs"];preds[f"{seed}_{phase}_Frozen63NoKD"]=inp[phase+"_base"]
   rows.append(dict(seed=seed,phase=phase,arm="Frozen63NoKD",**scores(inp[phase+"_base"],y)))
   for arm in ARMS:
    rawrows.append(dict(seed=seed,phase=phase,arm=arm,**scores(raw_predictions[seed][(phase,arm)],y)))
    model=PredictionTeacher(d["base"],torch.zeros(72,3,device=device)).to(device);model.load_state_dict(chosen[seed][arm])
    with torch.no_grad():p=model.read_features(*d["all_features"][arm][phase]).cpu().numpy()
    preds[f"{seed}_{phase}_{arm}"]=p;rows.append(dict(seed=seed,phase=phase,arm=arm,**scores(p,y)))
    if phase=="fit":fit_targets[(seed,arm)]=torch.tensor(p,device=device,dtype=torch.float32)
 summary={phase:{arm:{key:float(np.mean([v[key] for v in rows if v["phase"]==phase and v["arm"]==arm])) for key in ["mse","mae","level","shape"]} for arm in ["Frozen63NoKD"]+ARMS} for phase in split}
 validity={str(s):read(r/f"qwen_{s}/completion.json") for s in c["seeds"]}
 checks={}
 for ref in ["Frozen63NoKD","UpdatedNoKD","HistoryFourier"]:
  q=summary["check"]["HistoryQwen"];b=summary["check"][ref]
  wins=sum(next(v for v in rows if v["seed"]==s and v["phase"]=="check" and v["arm"]=="HistoryQwen")["mse"]<
           next(v for v in rows if v["seed"]==s and v["phase"]=="check" and v["arm"]==ref)["mse"] for s in c["seeds"])
  checks[ref]=dict(all3metrics=all(q[k]<b[k] for k in ["mse","mae","shape"]),seed_wins=int(wins))
 passed=bool(choices["HistoryQwen"]["step"]>0 and all(v["all3metrics"] and v["seed_wins"]>=2 for v in checks.values())
             and all(v["valid"]==v["total"] for v in validity.values()))
 save(r/"teacher_gate.json",dict(pass_=passed,checks=checks,validity=validity,uses_outer=False,independent_confirmation=False))
 save(r/"summary.json",summary);save(r/"metrics.json",rows);save(r/"raw_profile_metrics.json",rawrows)
 np.savez_compressed(r/"predictions.npz",**preds);writecsv(r/"metrics.csv",rows)
 kdrows=[];student_fits=0
 if passed:
  kdstart=time.monotonic();kd_states={};kd_curves=[];kchoices={};kspec=dict(NoKD=None,QwenShapeKD="HistoryQwen",FourierShapeKD="HistoryFourier",ShuffleShapeKD="ShuffledKnowledge")
  for seed,d in data.items():
   kd_states[seed]={};batches=np.random.default_rng(seed).integers(0,12,size=(c["steps"],c["batch"]))
   yf=torch.tensor(all_targets["fit"],device=device,dtype=torch.float32)
   for arm,tgt in kspec.items():
    st=load_base(c,seed,device);opt=torch.optim.Adam([dict(params=[p for n,p in st.named_parameters() if n!="codes.weight"],lr=c["lr"]),dict(params=st.codes.parameters(),lr=c["map_lr"])])
    kd_states[seed][arm]={}
    for step in range(c["steps"]+1):
     assert time.monotonic()-start<c["fit_seconds_cap"]
     if step in c["checkpoints"]:
      kd_states[seed][arm][step]=statecpu(st);p=pred(st,d["packs"]["selection"])
      for bi,b in enumerate(blocks):kd_curves.append(dict(seed=seed,arm=arm,step=step,block=bi,**scores(p[b],all_targets["selection"][b])))
     if step==c["steps"]:break
     ix=torch.tensor(batches[step],device=device);pack=d["packs"]["fit"]
     ll,ss=st.components(pack["x"][ix],pack["calendar"][ix],pack["ids"][ix])
     loss=(ll-yf[ix].mean(1,keepdim=True)).square().mean()+(ss-center(yf[ix])).square().mean()
     if tgt:loss+=c["shape_weight"]*(ss-center(fit_targets[(seed,tgt)][ix])).square().mean()
     opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    student_fits+=1
  kchoices={arm:choose(c,kd_curves,baselines,all_targets["selection"],blocks,arm) for arm in kspec}
  students={s:{arm:kd_states[s][arm][kchoices[arm]["step"]] for arm in kspec} for s in c["seeds"]}
  torch.save(students,r/"students.pt");save(r/"student_selection.json",kchoices)
  for seed,d in data.items():
   for arm in kspec:
    st=load_base(c,seed,device,state=students[seed][arm]);p=pred(st,d["packs"]["check"])
    preds[f"{seed}_check_Student{arm}"]=p;kdrows.append(dict(seed=seed,arm=arm,**scores(p,all_targets["check"])))
  save(r/"student_metrics.json",kdrows);np.savez_compressed(r/"predictions.npz",**preds)
 save(r/"KD_status.json",dict(performed=passed,student_fits=student_fits))
 for v in assets:assert sha(v["path"])==v["sha"],v["path"]
 check22(c)
 save(r/"status.json",dict(complete=True,head_seconds=time.monotonic()-start,teacher_fits=12,teacher_updates=7200,
  student_fits=student_fits,student_updates=student_fits*c["steps"],LLM_calls=sum(v["attempted"] for v in validity.values()),
  LLM_valid=sum(v["valid"] for v in validity.values()),LLM_cases=sum(v["total"] for v in validity.values()),
  teacher_parameters=995,teacher_trainable=705,oldassets_unchanged=len(assets),outer_scoring=False,independent_confirmation=False))
 print("GATE65",read(r/"teacher_gate.json"),flush=True);print("SUMMARY65",summary,flush=True);print("STATUS65",read(r/"status.json"),flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--stage",choices=["prepare","extract","fit"],required=True);ap.add_argument("--seed",type=int)
 args=ap.parse_args();c=read(args.config)
 try:
  if args.stage=="prepare":prepare(c,args.config)
  elif args.stage=="extract":extract(c,args.seed)
  else:fit(c,args.config)
 except Exception:
  r=Path(c["run"])
  if r.exists():(r/(f"failure_{args.stage}_{args.seed}_"+str(time.time_ns())+".log")).write_text(traceback.format_exc())
  raise
