"""Independent BDG2 confirmation of frozen-common57; separate train/test stages."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",CUBLAS_WORKSPACE_CONFIG=":4096:8",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,sys,time,subprocess,traceback
from pathlib import Path
import numpy as np,torch
from exp.exp_chronogeom_event_knowledge19 import Head,forward,knowledge,read,save,sha,git,scores
from exp.exp_chronogeom_teacher_content40 import check22
from exp.exp_chronogeom_downstream import CalendarTimeBase,predict
from exp.exp_chronogeom_residual03 import numeric_summary
from exp.exp_chronogeom_bdg2_confirm33 import dataset,extract_profiles
from exp.exp_chronogeom_metro_label_efficiency55 import masked_scores
from exp.exp_chronogeom_metro_anchored_qwen53 import csvsave,diagnose

def initialize(c,config):
 r=Path(c["root"]);r.mkdir(parents=True,exist_ok=False)
 reservation=read(c["reservation"]);selected=[]
 for cat in c["classes"]:selected += [x for x in reservation["selected"] if x["category"]==cat][:2]
 ids={x["building"] for x in selected};assert len(ids)==6
 previous=[]
 for p in Path("/home/home_new/qsmx/runs/ChronoGeom/downstream").rglob("selection_buildings.json"):
  v=read(p);rr=v.get("selected",[])
  if isinstance(rr,list):
   used={x.get("building","") if isinstance(x,dict) else str(x) for x in rr};assert not (ids&used),(str(p),ids&used)
  previous.append(dict(path=str(p),sha=sha(p)))
 save(r/"selection_buildings.json",dict(selected=selected,reservation_sha=sha(c["reservation"]),rule=c["building_rule"],prior_actual_selections=previous,targets_not_opened=True))
 save(r/"config.json",c)
 assets=check22(c)
 for key in ["source51","source57"]:
  assets += [dict(path=str(p),sha=sha(p)) for p in Path(c[key]).iterdir() if p.is_file() and p.suffix in [".json",".pt",".npz",".py"]]
 assets += [dict(path=str(Path(c["teacher_source"])/"profiles.json"),sha=sha(Path(c["teacher_source"])/"profiles.json"))]
 save(r/"old_assets.json",assets)
 for f in read(c["acquisition"])["files"]:assert sha(f["path"])==f["sha256"]
 tasks=read(Path(c["teacher_source"])/"profiles.json");save(r/"task_profiles.json",tasks)
 save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),data_acquisition=read(c["acquisition"]),Office_Healthcare_Task_identical=tasks["Office"]==tasks["Healthcare"],new_buildings=6,new_sites_not_claimed=True))
 workers=[]
 for i,meta in enumerate(selected):
  cc={**c,"target":meta["building"],"category":meta["category"],"timezone":meta["timezone"],"site":meta["site"],"run":str(r/meta["building"])}
  f=r/(meta["building"]+"_config.json");save(f,cc);workers.append(str(f))
 save(r/"workers.json",workers)
 print("SELECTED",[(x["building"],x["category"],x["site"]) for x in selected],flush=True)

def extract(c):
 import re,json
 from transformers import AutoTokenizer,AutoModelForCausalLM
 r=Path(c["root"]);g=r/"generic";g.mkdir(exist_ok=False)
 old=read(Path(c["teacher_source"])/"config.json");start=time.monotonic()
 free,total=torch.cuda.mem_get_info();assert free>=19*2**30,"Single-prompt BF16 extraction needs >=19GiB after CUDA initialization"
 cap=min(20*2**30,free-.5*2**30);torch.cuda.set_per_process_memory_fraction(cap/total)
 tok=AutoTokenizer.from_pretrained(c["model"],local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(c["model"],local_files_only=True,torch_dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa").eval()
 for p in model.parameters():p.requires_grad_(False)
 domain=c["generic_domain"];original=old["prompt"].replace("{DOMAIN}",domain);bank=None;calls=0
 for attempt in range(3):
  if attempt==0:prompt=original
  else:prompt=("Provide qualitative temporal knowledge for forecasting "+domain+" No observed electricity, labels, weather, or future observations are supplied. Format correction only: output a JSON object with EXACTLY 24 keys, the strings 0 through 23, one key per recorded local hour. For each key output an array of EXACTLY 3 integers in order [ordinary weekday, ordinary weekend, public holiday on a weekday], each from 0 to 10. Include EVERY hour 0 through 23; no extra keys, no explanation. Preserve qualitative electricity-use patterns for this building type. This is format attempt "+str(attempt)+".")
  text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  inp=tok(text,return_tensors="pt").to("cuda:0")
  with torch.inference_mode():q=model.generate(**inp,do_sample=False,max_new_tokens=512,pad_token_id=tok.eos_token_id)
  ans=tok.decode(q[0,inp["input_ids"].shape[1]:],skip_special_tokens=True);calls+=1
  save(g/f"generation{attempt}.json",dict(prompt=prompt,chat=text,answer=ans,tokens=q[0].tolist()))
  try:
   obj=json.loads(ans[ans.find("{"):ans.rfind("}")+1])
   a=np.array([obj[k] for k in ["workday","weekend","holiday"]],float) if attempt==0 else np.array([[obj[str(h)][j] for h in range(24)] for j in range(3)],float)
   if a.shape==(3,24) and np.isfinite(a).all() and (a>=0).all():bank=a*10/max(float(a.max()),1e-8);break
  except (ValueError,TypeError,KeyError,AttributeError):pass
  assert time.monotonic()-start<c["extract_seconds"]
 assert bank is not None,"Bounded schema repair failed; no metric-based retry"
 save(g/"profiles.json",dict(Generic=bank.tolist()))
 save(g/"extraction.json",dict(calls=calls,model=c["model"],revision=c["model_revision"],trainable_parameters=0,seconds=time.monotonic()-start,
  allocator_cap_GiB=cap/2**30,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,physical_GPU=os.environ.get("CUDA_VISIBLE_DEVICES"),resource_only_preflight_adjustment=True,git=git()))
 task=read(r/"task_profiles.json");generic=bank;kk=np.repeat(np.arange(3),24);hh=np.tile(np.arange(24),3)
 tables={cat:(knowledge(np.array(task[cat]),kk,hh)-knowledge(generic,kk,hh)).tolist() for cat in c["classes"]}
 save(r/"teacher_tables72.json",tables)
 save(r/"teacher_lock.json",dict(tables_sha=sha(r/"teacher_tables72.json"),tasks_sha=sha(r/"task_profiles.json"),generic_sha=sha(g/"profiles.json"),
  source=c["teacher_source"],before_building_fit=True,Office_Healthcare_identical=tables["Office"]==tables["Healthcare"]))
 print("GENERIC_EXTRACTED",read(g/"extraction.json"),flush=True)

def worker(c):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(.04);dev=torch.device("cuda:0");assert torch.cuda.mem_get_info()[0]>2*2**30
 r=Path(c["run"]);r.mkdir(exist_ok=False);start=time.monotonic();save(r/"config.json",c)
 packs,grid,raw,common,kind,h,wd,scale=dataset(c,r,False);assert "test" not in packs;save(r/"scale.json",scale)
 val=packs["val"];cal=np.arange(9);cut=int(val["future"][cal].max())+1+96
 sel=np.flatnonzero(val["origin"]>=cut)[:16];assert len(sel)==16
 future=val["future"][cal];u=np.unique(future);labels=[]
 for k in [0,1]:
  for hour in range(24):
   a=u[(kind[u]==k)&(h[u]==hour)];a=np.random.default_rng(c["label_seed"]+100*k+hour).permutation(a);n=2 if k==0 else 1
   assert len(a)>=n;labels += a[:n].tolist()
 mask=np.isin(future,labels)[:,:,None];assert len(set(labels))==72
 save(r/"time_split.json",dict(calibration_indices=cal.tolist(),check_indices=sel.tolist(),calibration_origins=grid[val["origin"][cal]].astype(str).tolist(),check_origins=grid[val["origin"][sel]].astype(str).tolist(),
  unique_label_hours=72,label_indices=sorted(labels),label_timestamps=grid[np.sort(labels)].astype(str).tolist(),positions=int(mask.sum()),all_data2016=True))
 profiles=np.array(read(Path(c["root"])/"teacher_tables72.json")[c["category"]])
 hh=np.tile(np.arange(24),3);tabraw=dict(QwenTaskGeneric=profiles,MatchedFourier=np.stack([np.sin(10*np.pi*hh/24),np.cos(10*np.pi*hh/24),np.cos(12*np.pi*hh/24)],1))
 for ss in c["shuffle_seeds"]:
  rng=np.random.default_rng(ss);perm=np.concatenate([24*k+rng.permutation(24) for k in range(3)]);tabraw["Shuffle"+str(ss)]=profiles[perm]
 ci=(kind*24+h)[future];si=(kind*24+h)[val["future"][sel]]
 tables={}
 for name,t in tabraw.items():
  z=t[ci];mu=z.mean((0,1));rms=max(float(np.sqrt(np.mean((z-mu)**2))),1e-6);tables[name]=(t-mu)/rms
 save(r/"tables.json",{k:v.tolist() for k,v in tables.items()})
 for p in packs.values():p["x"]=p["x"].to(dev);p["ids"]=p["ids"].to(dev)
 models={};curves=[];fits=[];internal=[];coefficient=[];initialerr=0.
 def guard():assert time.monotonic()-start<600,"Bounded building fit cap"
 for seed in c["seeds"]:
  m=CalendarTimeBase(c,seed,np.zeros((168,168),np.float32)).to(dev)
  assert sum(v.numel() for v in m.parameters())==6570
  opt=torch.optim.Adam(m.parameters(),lr=c["base_lr"],foreach=False);yy=torch.tensor(packs["base_fit"]["y"],dtype=torch.float32,device=dev)
  batches=np.random.default_rng(seed+271).integers(0,len(yy),size=(600,32));base_states={};base_rows=[]
  for step in range(601):
   if step%100==0:
    row=dict(seed=seed,stage="base",step=step,**scores(predict(m,packs["base_val"]),packs["base_val"]["y"]));base_rows.append(row);curves.append(row)
    base_states[step]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
   if step==600:break
   ix=torch.tensor(batches[step],device=dev);m.train();loss=(m(packs["base_fit"]["x"][ix],packs["base_fit"]["ids"][ix])-yy[ix]).square().mean()
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();guard()
  pick=min(base_rows,key=lambda v:(v["mse"],v["step"]))["step"];m.load_state_dict(base_states[pick]);m.eval()
  for v in m.parameters():v.requires_grad_(False)
  b={sp:torch.tensor(predict(m,packs[sp]),dtype=torch.float32,device=dev) for sp in ["fit","val"]}
  f={}
  for sp in ["fit","val"]:
   ix=packs[sp]["future"];f[sp]=np.concatenate([numeric_summary(packs[sp]["x"],b[sp]).cpu().numpy(),raw[ix],common[ix]],-1)
  mu=f["fit"].mean((0,1));sd=f["fit"].std((0,1)).clip(1e-6)
  kgrid=np.stack([np.sin(10*np.pi*h/24),np.cos(10*np.pi*h/24),np.cos(12*np.pi*h/24)],1)
  k=kgrid[packs["fit"]["future"]];km=k.mean((0,1));kr=max(float(np.sqrt(np.mean((k-km)**2))),1e-6)
  z={sp:torch.tensor(np.concatenate([(f[sp]-mu)/sd,(kgrid[packs[sp]["future"]]-km)/kr],-1),dtype=torch.float32,device=dev) for sp in ["fit","val"]}
  y=torch.tensor(packs["fit"]["y"],dtype=torch.float64,device=dev);bn=scores(b["fit"].cpu().numpy(),packs["fit"]["y"]);bb=np.random.default_rng(seed).integers(0,len(y),size=(600,32))
  def train_head(lr,teacher_shape=None):
   head=Head(seed).to(dev);op=torch.optim.Adam(head.parameters(),lr=lr,foreach=False)
   for step in range(600):
    ix=torch.tensor(bb[step],device=dev);pr=forward(head,z["fit"][ix],b["fit"][ix]);err=pr-y[ix]
    loss=.5*(err.square().mean()/bn["mse"]+err.abs().mean()/bn["mae"])
    if teacher_shape is not None:loss=loss+c["anchor_KD"]*((pr-pr.mean(1,keepdim=True))-teacher_shape[ix]).square().mean()/bn["mse"]
    op.zero_grad();loss.backward();op.step();guard()
   head.eval()
   for v in head.parameters():v.requires_grad_(False)
   return head
  teacher=train_head(c["teacher_lr"])
  with torch.no_grad():tp=forward(teacher,z["fit"],b["fit"]);shape=(tp-tp.mean(1,keepdim=True)).detach()
  anchor=train_head(c["anchor_lr"],shape)
  with torch.no_grad():av=forward(anchor,z["val"],b["val"]).cpu().numpy()
  xc=val["x"][cal].cpu();xs=val["x"][sel].cpu()
  fcal=np.concatenate([numeric_summary(xc,torch.tensor(av[cal],dtype=torch.float32)).numpy(),raw[future],common[future]],-1)
  fsel=np.concatenate([numeric_summary(xs,torch.tensor(av[sel],dtype=torch.float32)).numpy(),raw[val["future"][sel]],common[val["future"][sel]]],-1)
  cm=fcal.mean((0,1));cs=fcal.std((0,1)).clip(1e-3);zc=torch.tensor((fcal-cm)/cs,dtype=torch.float32);zs=torch.tensor((fsel-cm)/cs,dtype=torch.float32)
  yc=val["y"][cal];ys=val["y"][sel];known=np.zeros_like(yc);known[mask]=yc[mask]
  yt=torch.tensor(known,dtype=torch.float64);ac=torch.tensor(av[cal],dtype=torch.float64);mt=torch.tensor(mask,dtype=torch.float64)
  n=masked_scores(av[cal],known,mask);batch=np.random.default_rng(seed).integers(0,9,size=(200,32))
  def loss48(pr,ix):
   e=pr-yt[ix];w=mt[ix];return .5*((w*e.square()).sum()/w.sum()/n["mse"]+(w*e.abs()).sum()/w.sum()/n["mae"])
  head=Head(seed);op=torch.optim.Adam(head.parameters(),lr=c["lr"])
  zero=torch.zeros(9,96,3)
  for step in range(100):
   ix=batch[step];pr=ac[ix]+head(torch.cat([zc[ix],zero[ix]],-1));loss=loss48(pr,ix);op.zero_grad();loss.backward();op.step()
  with torch.no_grad():head.net[0].weight[:,39:].zero_()
  shared=copy.deepcopy(head.state_dict());heads={};checks=[]
  with torch.no_grad():common_pred=av[sel]+head(torch.cat([zs,torch.zeros(len(sel),96,3)],-1)).numpy()
  for name in c["arms"]:
   hh=Head(seed);hh.load_state_dict(shared)
   for v in hh.parameters():v.requires_grad_(False)
   w=hh.net[0].weight;w.requires_grad_(True);allowed=torch.zeros_like(w);allowed[:,39:]=1
   hook=w.register_hook(lambda g:g*allowed);oo=torch.optim.Adam([w],lr=c["lr"],weight_decay=0)
   tc=torch.tensor(tables[name][ci],dtype=torch.float32);ts=torch.tensor(tables[name][si],dtype=torch.float32)
   with torch.no_grad():init=av[sel]+hh(torch.cat([zs,ts],-1)).numpy()
   initialerr=max(initialerr,float(abs(init-common_pred).max()));assert np.array_equal(init,common_pred)
   for step in range(100):
    ix=batch[100+step];pr=ac[ix]+hh(torch.cat([zc[ix],tc[ix]],-1));loss=loss48(pr,ix)
    oo.zero_grad();loss.backward();assert torch.count_nonzero(w.grad[:,:39])==0;oo.step()
   state=copy.deepcopy(hh.state_dict())
   for key in state:
    assert torch.equal(state[key][:,:39],shared[key][:,:39]) if key=="net.0.weight" else torch.equal(state[key],shared[key])
   with torch.no_grad():pr=av[sel]+hh(torch.cat([zs,ts],-1)).numpy()
   internal.append(dict(building=c["target"],seed=seed,arm=name,**scores(pr,ys)))
   coefficient.append(dict(seed=seed,arm=name,changed=int(torch.count_nonzero(state["net.0.weight"][:,39:])),frozen_common_error=0))
   heads[name]=state
  models[seed]=dict(base=base_states[pick],base_step=pick,teacher={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()},
   anchor={k:v.detach().cpu().clone() for k,v in anchor.state_dict().items()},anchor_norm=dict(mean=mu.tolist(),sd=sd.tolist(),km=km.tolist(),kr=kr),
   common_norm=dict(mean=cm.tolist(),sd=cs.tolist()),common=shared,heads=heads)
  fits += [dict(seed=seed,stage=s,steps=600) for s in ["base","Fourier_teacher","Fourier_anchor"]]+[dict(seed=seed,stage="common",steps=100)]+[dict(seed=seed,stage=nm,steps=100) for nm in c["arms"]]
  print("FIT",c["target"],seed,"base_step",pick,flush=True);guard()
 torch.save(models,r/"models.pt");save(r/"fit_records.json",fits);save(r/"internal_metrics.json",internal);save(r/"base_curves.json",curves);save(r/"coefficient_checks.json",coefficient)
 save(r/"model_lock.json",dict(files={name:sha(r/name) for name in ["models.pt","scale.json","tables.json","time_split.json","config.json"]},
  teacher_lock_sha=sha(Path(c["root"])/"teacher_lock.json"),source_commit=git(),before_2017=True))
 save(r/"fit_status.json",dict(complete=True,seconds=time.monotonic()-start,fits=len(fits),updates=7200,initial_prediction_error=initialerr,test_opened=False))
 print("BUILDING_FIT_DONE",c["target"],flush=True)

def fit_all(c):
 r=Path(c["root"]);jobs=[];pending=list(read(r/"workers.json"));start=time.monotonic()
 while pending or jobs:
  while pending and len(jobs)<2:
   used={j[2] for j in jobs};gpu=next(g for g in c["worker_gpu_indices"] if g not in used);cf=pending.pop(0);cc=read(cf)
   log=(r/(cc["target"]+".log")).open("w");env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
   pp=subprocess.Popen([sys.executable,"-u","-m","exp.exp_chronogeom_bdg2_frozen_common58","--config",cf,"--stage","worker"],env=env,stdout=log,stderr=subprocess.STDOUT)
   jobs.append((pp,log,gpu,cc["target"]))
  for j in jobs[:]:
   code=j[0].poll()
   if code is not None:
    j[1].close();jobs.remove(j);assert code==0,("worker failed",j[3],code)
  assert time.monotonic()-start<1200,"Bounded fit total";time.sleep(1)
 locks={read(cf)["target"]:sha(Path(read(cf)["run"])/"model_lock.json") for cf in read(r/"workers.json")}
 save(r/"all_models_locked.json",dict(locks=locks,before_any_test_open=True,teacher_sha=sha(r/"teacher_lock.json"),fit_seconds=time.monotonic()-start))
 print("ALL6_LOCKED",locks,flush=True)

def evaluate(c):
 torch.set_num_threads(2);r=Path(c["root"]);locked=read(r/"all_models_locked.json");assert not (r/"evaluation_status.json").exists()
 torch.cuda.set_per_process_memory_fraction(.04);dev=torch.device("cuda:0");allrows=[]
 for cf in read(r/"workers.json"):
  cc=read(cf);br=Path(cc["run"]);assert sha(br/"model_lock.json")==locked["locks"][cc["target"]]
  lock=read(br/"model_lock.json")
  for name,ss in lock["files"].items():assert sha(br/name)==ss
  packs,grid,raw,common,kind,h,wd,scale=dataset(cc,br,True);assert scale==read(br/"scale.json")
  pack=packs["test"];models=torch.load(br/"models.pt",map_location="cpu",weights_only=True);tables={k:np.array(v) for k,v in read(br/"tables.json").items()}
  pack["x"]=pack["x"].to(dev);pack["ids"]=pack["ids"].to(dev);ix=pack["future"];sid=(kind*24+h)[ix]
  preds={};rows=[];replay=0.;meanerr=0.
  for seed in c["seeds"]:
   st=models[seed];base=CalendarTimeBase(cc,seed,np.zeros((168,168),np.float32)).to(dev);base.load_state_dict(st["base"]);base.eval()
   bp=torch.tensor(predict(base,pack),dtype=torch.float32,device=dev);n=st["anchor_norm"]
   ft=np.concatenate([numeric_summary(pack["x"],bp).cpu().numpy(),raw[ix],common[ix]],-1)
   ff=np.stack([np.sin(10*np.pi*h[ix]/24),np.cos(10*np.pi*h[ix]/24),np.cos(12*np.pi*h[ix]/24)],-1)
   zz=torch.tensor(np.concatenate([(ft-np.array(n["mean"]))/np.array(n["sd"]),(ff-np.array(n["km"]))/n["kr"]],-1),dtype=torch.float32,device=dev)
   ah=Head(seed).to(dev);ah.load_state_dict(st["anchor"]);ah.eval()
   with torch.no_grad():anchor=forward(ah,zz,bp).cpu().numpy()
   nm=st["common_norm"];z=np.concatenate([numeric_summary(pack["x"].cpu(),torch.tensor(anchor,dtype=torch.float32)).numpy(),raw[ix],common[ix]],-1)
   z39=torch.tensor((z-np.array(nm["mean"]))/np.array(nm["sd"]),dtype=torch.float32)
   shared=Head(seed);shared.load_state_dict(st["common"]);shared.eval()
   with torch.no_grad():g=anchor+shared(torch.cat([z39,torch.zeros(len(z39),96,3)],-1)).numpy()
   outs=dict(FourierAnchor=anchor,Common39Only=g)
   for name in c["arms"]:
    head=Head(seed);head.load_state_dict(st["heads"][name]);head.eval();feat=torch.cat([z39,torch.tensor(tables[name][sid],dtype=torch.float32)],-1)
    with torch.no_grad():pr=anchor+head(feat).numpy()
    clone=Head(seed);clone.load_state_dict(st["heads"][name]);clone.eval()
    with torch.no_grad():again=anchor+clone(feat).numpy()
    replay=max(replay,float(abs(pr-again).max()));meanerr=max(meanerr,float(abs((pr-g).mean(1)).max()));outs[name]=pr
   preds[f"{seed}_y"]=pack["y"];preds[f"{seed}_origin"]=pack["origin"]
   for name,pr in outs.items():
    preds[f"{seed}_{name}"]=pr;met=scores(pr,pack["y"])
    rows.append(dict(building=cc["target"],category=cc["category"],site=cc["site"],seed=seed,arm=name,windows=len(pr),
      **met,raw_mse=met["mse"]*scale["sd"]**2,raw_mae=met["mae"]*scale["sd"],raw_shape=met["shape"]*scale["sd"]**2))
  np.savez_compressed(br/"test_predictions.npz",**preds);save(br/"test_metrics.json",rows)
  save(br/"test_verification.json",dict(reload_error=replay,mean_preservation_error=meanerr,no_refit=True,model_lock_sha=sha(br/"model_lock.json")))
  allrows+=rows;print("EVALUATED",cc["target"],"windows",len(pack["y"]),flush=True)
 save(r/"test_metrics.json",allrows);csvsave(r/"test_metrics.csv",allrows)
 names=["FourierAnchor","Common39Only"]+c["arms"]
 summary={a:{k:float(np.mean([v[k] for v in allrows if v["arm"]==a])) for k in ["mse","mae","level","shape"]} for a in names}
 save(r/"summary.json",summary)
 save(r/"class_summary.json",{cat:{a:{k:float(np.mean([v[k] for v in allrows if v["category"]==cat and v["arm"]==a])) for k in ["mse","mae","shape"]} for a in names} for cat in c["classes"]})
 for aa in read(r/"old_assets.json"):assert sha(aa["path"])==aa["sha"]
 check22(c);save(r/"evaluation_status.json",dict(complete=True,rows=len(allrows),buildings=6,seeds=3,evaluation_passes=1,posttest_tuning=False,
  all_old_assets_unchanged=True,all6models_locked_before2017=True))
 print("SUMMARY",summary,flush=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--stage",required=True,choices=["init","extract","fit","worker","evaluate"])
 a=ap.parse_args();c=read(a.config)
 try:{"init":lambda:initialize(c,a.config),"extract":lambda:extract(c),"fit":lambda:fit_all(c),"worker":lambda:worker(c),"evaluate":lambda:evaluate(c)}[a.stage]()
 except Exception:
  r=Path(c.get("run",c["root"]))
  if r.exists():(r/("failure_"+a.stage+".log")).write_text(traceback.format_exc())
  raise
