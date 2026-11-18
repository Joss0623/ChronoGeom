"""Single teacher-content comparison on immutable Capital/Metro22, no BDG2."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",CUBLAS_WORKSPACE_CONFIG=":4096:8",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,json,re,time,csv,calendar
from pathlib import Path
import numpy as np,pandas as pd,torch
from exp.exp_chronogeom_event_knowledge19 import Head,forward,scores,knowledge,read,save,sha,git
from exp.exp_chronogeom_residual03 import numeric_summary
def center(x):return x-x.mean(1,keepdim=True)
def check22(c):
 f=read(c["frozen22"]);root=Path(f["run"]);records=[dict(path=str(root/k),sha=v) for k,v in f["artifact_sha256"].items()]+f["base_assets"]
 for x in records:assert sha(x["path"])==x["sha"],x["path"]
 return records

def extract(c,config):
 from transformers import AutoTokenizer,AutoModelForCausalLM
 r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c);assets=check22(c);save(r/"frozen22_assets.json",assets);save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),framework_model_agnostic=True,current_instance=c["model"]))
 torch.set_num_threads(2);torch.manual_seed(2024);torch.cuda.set_per_process_memory_fraction(c["memory_gib"]*2**30/torch.cuda.get_device_properties(0).total_memory);assert torch.cuda.mem_get_info()[0]>=21*2**30
 tok=AutoTokenizer.from_pretrained(c["model"],local_files_only=True);m=AutoModelForCausalLM.from_pretrained(c["model"],local_files_only=True,dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa").eval()
 for p in m.parameters():p.requires_grad_(False)
 start=time.monotonic();profiles={};calls=0
 for name,spec in c["domains"].items():
  out=r/name;out.mkdir();old=read(Path(spec["source"])/"profiles.json");banks={k:v for k,v in old.items() if k in ["Task","Wrong"]}
  for label,domain in spec["missing"].items():
   prompt=spec["prompt"].replace("{DOMAIN}",domain);chat=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False);inp=tok(chat,return_tensors="pt").to("cuda:0")
   with torch.inference_mode():q=m.generate(**inp,do_sample=False,max_new_tokens=512,pad_token_id=tok.eos_token_id)
   ans=tok.decode(q[0,inp["input_ids"].shape[1]:],skip_special_tokens=True);save(out/(label+"_generation.json"),dict(prompt=prompt,chat=chat,answer=ans,tokens=q[0].tolist()));calls+=1
   match=re.search(r"\{.*\}",ans,re.S);assert match,ans
   ob=json.loads(match.group());a=np.array([ob[k] for k in ["workday","weekend","holiday"]],float);assert a.shape==(3,24) and np.isfinite(a).all() and (a>=0).all()
   maximum=float(a.max());banks[label]=(a*10/max(maximum,1e-8)).tolist();save(out/(label+"_scale.json"),dict(original_max=maximum,uniform_rescale=True,flat_rows=(a.std(1)==0).tolist(),no_query_repair=True))
   assert time.monotonic()-start<c["extract_seconds"];print("GENERATED",name,label,flush=True)
  assert set(banks)=={"Task","Generic","Wrong"};save(out/"profiles.json",banks);profiles[name]=banks
 save(r/"profiles.json",profiles);save(r/"extraction.json",dict(complete=True,new_generations=calls,weights_updated=0,model_revision=c["revision"],seconds_after_model_load=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,no_new_weights=True));check22(c)

def calendar_arrays(cfg,adapter):
 if adapter=="Capital":
  f=pd.read_csv(cfg["data"],usecols=["dteday","hr","weekday","holiday","workingday"]);grid=pd.date_range("2011-01-01","2013-01-01",freq="h",inclusive="left")
  holiday=f.groupby("dteday").holiday.first();hol=np.array([holiday.get(str(t.date()),np.nan) for t in grid]);years=366
 else:
  f=pd.read_csv(cfg["data"],keep_default_na=False,usecols=["date_time","holiday"]);ts=pd.to_datetime(f.date_time);grid=pd.date_range(ts.min().normalize(),ts.max().normalize()+pd.Timedelta(days=1),freq="h",inclusive="left")
  flags=pd.DataFrame(dict(day=ts.dt.normalize(),event=f.holiday!="None")).groupby("day").event.max();hol=flags.reindex(grid.normalize()).to_numpy(float);years=np.array([366 if calendar.isleap(y) else 365 for y in grid.year])
 h=grid.hour.to_numpy();wd=grid.dayofweek.to_numpy();week=(wd>=5).astype(int);kind=np.where(hol==1,2,week);phase=2*np.pi*(grid.dayofyear.to_numpy()-1)/years
 raw=np.stack([hol,week,(week==0)&(hol==0),np.sin(wd*2*np.pi/7),np.cos(wd*2*np.pi/7),np.sin(phase),np.cos(phase)],1).astype(float)
 common=np.concatenate([np.stack([fn(2*np.pi*k*h/24) for k in range(1,5) for fn in [np.sin,np.cos]],1)*(kind==d)[:,None] for d in range(3)],1)
 return grid,raw,common,kind,h

def fit(c):
 start=time.monotonic();r=Path(c["run"]);torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assets=check22(c);profiles=read(r/"profiles.json")
 allrows=[];curves=[];fits=[];preds={};selected={};weights={};data={};teacher_objects={};weak=[];sources=[];maxreplay=0.
 for domain,spec in c["domains"].items():
  src=Path(spec["source"]);out=r/domain;cfg=read(src/"config.json");grid,raw,common,kind,h=calendar_arrays(cfg,spec["adapter"]);packs=np.load(src/"data.npz");cached=np.load(src/"predictions.npz");norm=read(src/"normalizers.json");ksource=np.load(src/"knowledge.npz");oldstates=torch.load(src/spec["states"],map_location="cpu",weights_only=True);oldsel=read(src/spec["selection"])["chosen"];prefix=spec["teacher_prefix"]
  for file in ["data.npz","predictions.npz","knowledge.npz","normalizers.json",spec["states"],spec["selection"],"profiles.json"]:sources.append(dict(path=str(src/file),sha=sha(src/file)))
  KT=knowledge(np.array(profiles[domain]["Task"]),kind,h);assert np.array_equal(KT,ksource["QwenTask"])
  KG=knowledge(np.array(profiles[domain]["Generic"]),kind,h);KW=knowledge(np.array(profiles[domain]["Wrong"]),kind,h)
  KK=dict(FullTask=KT,Generic=KG,Delta=KT-KG,Fourier=ksource["Fourier"],HistoricalTemplate=ksource["HistoricalTemplate"],FullWrong=KW,FullShuffle=ksource["QwenShuffle"])
  oldmap=dict(FullTask="QwenTask",Fourier="Fourier",HistoricalTemplate="HistoricalTemplate",FullShuffle="QwenShuffle")
  if spec["wrong_cache"]:oldmap["FullWrong"]="QwenWrong"
  ds=dict(src=src,cfg=cfg,grid=grid,raw=raw,common=common,kind=kind,h=h,packs=packs,cached=cached,norm=norm,K=KK,z={},bp={},target={},bn={},oldstates=oldstates,oldsel=oldsel,oldmap=oldmap,spec=spec)
  data[domain]=ds;selected[domain]={};weights[domain]={};teacher_objects[domain]={}
  for seed in c["seeds"]:
   ds["z"][seed]={};ds["bp"][seed]={};ds["bn"][seed]={}
   for split in ["fit","val"]:
    y=packs[split+"_y"];x=torch.tensor(packs[split+"_x"],dtype=torch.float32);ix=packs[split+"_future"];bp=torch.tensor(cached[f"{seed}_{split}_FrozenTimeBase"],dtype=torch.float32);ds["bp"][seed][split]=bp;ds["target"][split]=torch.tensor(y,dtype=torch.float64);ds["bn"][seed][split]=scores(bp.numpy(),y)
    assert np.array_equal(y,cached[f"{seed}_{split}_y"]);assert np.array_equal(packs[split+"_origin"],cached[f"{seed}_{split}_origin"])
    ft=np.concatenate([numeric_summary(x,bp).numpy(),raw[ix],common[ix]],-1);pp=norm[str(seed)]["Fourier"]
    for arm,K in KK.items():
     kfit=K[packs["fit_future"]];km=kfit.mean((0,1));kr=max(float(np.sqrt(np.mean((kfit-km)**2))),1e-6)
     ds["z"][seed].setdefault(arm,{})[split]=torch.tensor(np.concatenate([(ft-np.array(pp["mean"]))/np.array(pp["sd"]),(K[ix]-km)/kr],-1),dtype=torch.float32)
   weights[domain][seed]={};teacher_objects[domain][seed]={}
  for arm in KK:
   new=arm not in oldmap
   for seed in c["seeds"]:
    head=Head(seed);state_by_step={}
    if not new:state_by_step=oldstates[seed][oldmap[arm]]
    else:
     optimizer=torch.optim.Adam(head.parameters(),lr=c["lr"]);batch=np.random.default_rng(seed).integers(0,len(ds["target"]["fit"]),size=(600,c["batch"]));bn=ds["bn"][seed]["fit"]
     for step in range(601):
      if step%100==0:state_by_step[step]=copy.deepcopy(head.state_dict())
      if step==600:break
      ix=batch[step];p=forward(head,ds["z"][seed][arm]["fit"][ix],ds["bp"][seed]["fit"][ix]);err=p-ds["target"]["fit"][ix]
      loss=.5*(err.square().mean()/bn["mse"]+err.abs().mean()/bn["mae"]);optimizer.zero_grad();loss.backward();optimizer.step();assert time.monotonic()-start<c["teacher_seconds"]
     fits.append(dict(domain=domain,seed=seed,arm=arm,updates=600,parameters=705))
    weights[domain][seed][arm]=state_by_step
    for step in range(0,601,100):
     head.load_state_dict(state_by_step[step]);head.eval()
     with torch.no_grad():pv=forward(head,ds["z"][seed][arm]["val"],ds["bp"][seed]["val"]).numpy()
     curves.append(dict(domain=domain,seed=seed,arm=arm,step=step,**scores(pv,ds["target"]["val"].numpy())))
    print("TEACHER",domain,seed,arm,"new" if new else "reused",flush=True)
   if not new:pick=oldsel[oldmap[arm]]
   else:
    def criterion(step):
     return float(np.mean([.5*(next(a for a in curves if a["domain"]==domain and a["seed"]==s and a["arm"]==arm and a["step"]==step)["mse"]/ds["bn"][s]["val"]["mse"]+next(a for a in curves if a["domain"]==domain and a["seed"]==s and a["arm"]==arm and a["step"]==step)["mae"]/ds["bn"][s]["val"]["mae"]) for s in c["seeds"]]))
    pick=min(range(0,601,100),key=lambda step:(criterion(step),step))
   selected[domain][arm]=pick
  save(out/"raw_prior_diagnostics.json",dict(Task_Generic_RMS=float(np.sqrt(np.mean((KT-KG)**2))),Task_RMS=float(np.sqrt(np.mean(KT**2))),Generic_RMS=float(np.sqrt(np.mean(KG**2))),Delta_is_normalized_knowledge_difference=True))
 # ALL teacher choices fixed using val before loading any outer labels.
 save(r/"teacher_selection.json",selected);torch.save(weights,r/"teacher_checkpoints.pt");save(r/"teacher_choice_lock.json",dict(selection_sha=sha(r/"teacher_selection.json"),state_sha=sha(r/"teacher_checkpoints.pt"),before_new_outer_metrics=True));save(r/"curves.json",curves);save(r/"fits.json",fits)
 for domain,ds in data.items():
  packs=ds["packs"];cached=ds["cached"];spec=ds["spec"];prefix=spec["teacher_prefix"]
  for seed in c["seeds"]:
   for split in ["fit","val",spec["eval_split"]]:
    label="eval" if split==spec["eval_split"] else split;y=packs[split+"_y"];bp=torch.tensor(cached[f"{seed}_{split}_FrozenTimeBase"],dtype=torch.float32);ix=packs[split+"_future"];x=torch.tensor(packs[split+"_x"],dtype=torch.float32);ft=np.concatenate([numeric_summary(x,bp).numpy(),ds["raw"][ix],ds["common"][ix]],-1);pp=ds["norm"][str(seed)]["Fourier"]
    preds[f"{domain}_{seed}_{label}_y"]=y;preds[f"{domain}_{seed}_{label}_origin"]=packs[split+"_origin"]
    for arm,K in ds["K"].items():
     kfit=K[packs["fit_future"]];km=kfit.mean((0,1));kr=max(float(np.sqrt(np.mean((kfit-km)**2))),1e-6);z=torch.tensor(np.concatenate([(ft-np.array(pp["mean"]))/np.array(pp["sd"]),(K[ix]-km)/kr],-1),dtype=torch.float32);m=Head(seed);m.load_state_dict(weights[domain][seed][arm][selected[domain][arm]]);m.eval()
     with torch.no_grad():p=forward(m,z,bp).numpy()
     if arm in ds["oldmap"]:
      oldp=cached[f"{seed}_{split}_{prefix}{ds['oldmap'][arm]}"];maxreplay=max(maxreplay,float(abs(oldp-p).max()));assert np.max(abs(oldp-p))<2e-6
      p=oldp # Original controls always use exact saved predictions.
     preds[f"{domain}_{seed}_{label}_{arm}"]=p;allrows.append(dict(domain=domain,seed=seed,split=label,arm=arm,training_budget=600,selected_step=selected[domain][arm],**scores(p,y)))
     if arm in ["FullWrong","FullShuffle"]:
      m.load_state_dict(weights[domain][seed][arm][200])
      with torch.no_grad():wp=forward(m,z,bp).numpy()
      preds[f"{domain}_{seed}_{label}_{arm}200"]=wp;weak.append(dict(domain=domain,seed=seed,split=label,arm=arm+"200",training_budget=200,selected_step=200,unequal_budget=True,**scores(wp,y)))
 save(r/"teacher_metrics.json",allrows);save(r/"unequal_budget200_metrics.json",weak);np.savez_compressed(r/"teacher_predictions.npz",**preds)
 summary={domain:{split:{arm:{k:float(np.mean([x[k] for x in allrows if x["domain"]==domain and x["split"]==split and x["arm"]==arm])) for k in ["mse","mae","level","shape"]} for arm in data[domain]["K"]} for split in ["val","eval"]} for domain in data}
 save(r/"teacher_summary.json",summary)
 def ratio(arm,ref,key,seed=None):
  return float(np.mean([a[key]/next(b for b in allrows if b["domain"]==a["domain"] and b["seed"]==a["seed"] and b["split"]=="val" and b["arm"]==ref)[key] for a in allrows if a["split"]=="val" and a["arm"]==arm and (seed is None or a["seed"]==seed)]))
 candidates={arm:.5*(ratio(arm,"Fourier","mse")+ratio(arm,"Fourier","mae")) for arm in ["FullTask","Generic","Delta"]};chosen=min(candidates,key=lambda a:(candidates[a],a));checks={ref:dict(mse=ratio(chosen,ref,"mse"),mae=ratio(chosen,ref,"mae")) for ref in ["Fourier","HistoricalTemplate","FullWrong","FullShuffle"]}
 gate=all(x["mse"]<1 and x["mae"]<=1 for x in checks.values()) and sum(ratio(chosen,"Fourier","mse",seed)<1 for seed in c["seeds"])>=2
 save(r/"KD_decision.json",dict(candidate=chosen,validation_candidates=candidates,matched_budget_checks=checks,pass_=gate,unequal_budget_panel_excluded=True))
 save(r/"frozen22_student_reference.json",read(c["frozen22"])["metrics"])
 for a in sources:assert sha(a["path"])==a["sha"]
 check22(c);save(r/"input_assets.json",sources);save(r/"teacher_status.json",dict(complete=True,new_fits=len(fits),new_updates=len(fits)*600,reused_teacher_fits=27,maximum_old_teacher_replay_difference=maxreplay,seconds=time.monotonic()-start,KD_pass=gate,selected_candidate=chosen,frozen22_unchanged=True,BDG2_used=False))
 print("TEACHER_COMPLETE",read(r/"teacher_status.json"),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--stage",choices=["extract","fit"],required=True);a=p.parse_args();c=read(a.config)
 if a.stage=="extract":extract(c,a.config)
 else:fit(c)
