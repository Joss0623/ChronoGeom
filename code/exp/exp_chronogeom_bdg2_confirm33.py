"""One blind Tetouan confirmation of the fixed energy switch in pinned TSLib."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",CUBLAS_WORKSPACE_CONFIG=":4096:8",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,calendar,copy,csv,sys,time,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
from exp.exp_chronogeom_event_knowledge19 import Head,forward,scores,knowledge,extract,read,save,sha,git
from exp.exp_chronogeom_downstream import CalendarTimeBase,predict
from exp.exp_chronogeom_residual03 import numeric_summary
from layers.ChronoGeomEnergySwitch import EnergySwitch,centered

def center(x):return x-x.mean(1,keepdim=True)

def dataset(c,r,full=False):
    f=pd.read_csv(c["data"],usecols=["timestamp",c["target"]],nrows=None if full else c["training_csv_nrows"])
    t=pd.to_datetime(f.timestamp);assert t.is_monotonic_increasing and not t.duplicated().any()
    if not full:assert len(f)==8472 and t.max()<pd.Timestamp("2016-12-19")
    grid=pd.date_range("2016-01-01","2018-01-01",freq="h",inclusive="left")
    v=pd.Series(f[c["target"]].to_numpy(float),index=t);v=v.where((v>=0)&np.isfinite(v))
    values=v.reindex(grid).to_numpy(float)
    a,b=map(pd.Timestamp,c["splits"]["base_fit"]);mask=(grid>=a)&(grid<b)&np.isfinite(values)
    mu=float(values[mask].mean());sd=float(values[mask].std());assert sd>1e-8
    values=(values-mu)/sd;h=grid.hour.to_numpy();wd=grid.dayofweek.to_numpy();week=(wd>=5).astype(int);kind=week
    phase=2*np.pi*(grid.dayofyear.to_numpy()-1)/np.where(grid.is_leap_year,366,365)
    raw=np.stack([np.zeros(len(grid)),week,1-week,np.sin(2*np.pi*wd/7),np.cos(2*np.pi*wd/7),np.sin(phase),np.cos(phase)],1)
    hh=np.stack([fn(2*np.pi*k*h/24) for k in range(1,5) for fn in [np.sin,np.cos]],1);common=np.concatenate([hh*(kind==k)[:,None] for k in range(3)],1)
    packs={};counts={}
    for split,(aa,bb) in c["splits"].items():
        if split=="test" and not full:continue
        lo=int((pd.Timestamp(aa)-grid[0]).total_seconds()/3600);hi=int((pd.Timestamp(bb)-grid[0]).total_seconds()/3600)
        origins=np.array([o for o in range(max(lo,96),hi-96+1,24) if np.isfinite(values[o-96:o+96]).all()])
        if split!="test" and split in c["min_counts"]:assert len(origins)>=c["min_counts"][split],(split,len(origins))
        assert len(origins)>0,(c["target"],split,"No complete windows; retain failure, do not replace building.")
        past=origins[:,None]+np.arange(-96,0);future=origins[:,None]+np.arange(96)
        packs[split]=dict(x=torch.tensor(values[past,None],dtype=torch.float32),y=values[future,None],ids=torch.tensor(wd[future]*24+h[future],dtype=torch.long),origin=origins,future=future)
        counts[split]=dict(windows=len(origins),unique_target_hours=len(np.unique(future)),start=str(grid[origins.min()]),last_origin=str(grid[origins.max()]))
    save(r/("full_data_metadata.json" if full else "train_data_metadata.json"),dict(building=c["target"],category=c["category"],timezone=c["timezone"],counts=counts,rows_read=len(f),last_raw_timestamp=str(t.max()),mean=mu,sd=sd,calendar="published recorded local timestamp; no fabricated DST shift",test_targets_loaded=full))
    return packs,grid,raw,common,kind,h,wd,dict(mean=mu,sd=sd)

def extract_profiles(c,r):
    import re
    from transformers import AutoTokenizer,AutoModelForCausalLM
    start=time.monotonic();device="cuda:0";assert torch.cuda.mem_get_info(device)[0]>=21*2**30
    torch.cuda.set_per_process_memory_fraction(22*2**30/torch.cuda.get_device_properties(device).total_memory)
    tok=AutoTokenizer.from_pretrained(c["model"],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(c["model"],local_files_only=True,torch_dtype=torch.bfloat16,device_map={"":device},attn_implementation="sdpa").eval()
    for p in model.parameters():p.requires_grad_(False)
    def generation(path,prompt):
        text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        if path.exists():
            rec=read(path);assert rec["prompt"]==prompt;return rec["answer"]
        inp=tok(text,return_tensors="pt").to(device)
        with torch.inference_mode():g=model.generate(**inp,do_sample=False,max_new_tokens=512,pad_token_id=tok.eos_token_id)
        ans=tok.decode(g[0,inp["input_ids"].shape[1]:],skip_special_tokens=True);save(path,dict(prompt=prompt,chat=text,answer=ans,tokens=g[0].tolist()))
        return ans
    out={};status=[]
    for name,domain in c["domains"].items():
        prompt=c["prompt"].replace("{DOMAIN}",domain);ans=generation(r/f"{name}_generation.json",prompt);mode="original"
        def parse(ans,rows=False):
            mt=re.search(r"\{.*\}",ans,re.S)
            if not mt:return None
            try:
                obj=json.loads(mt.group())
                bank=np.array([[obj[str(h)][j] for h in range(24)] for j in range(3)],float) if rows else np.array([obj[k] for k in ["workday","weekend","holiday"]],float)
                return bank if bank.shape==(3,24) and np.isfinite(bank).all() and (bank>=0).all() else None
            except (ValueError,KeyError,TypeError,IndexError):return None
        bank=parse(ans)
        for attempt in range(1,3):
            if bank is not None:break
            repair=("Provide qualitative temporal knowledge for forecasting "+domain+" No observed electricity, labels, weather, or future observations are supplied. Format correction only: output a JSON object with EXACTLY 24 keys, the strings 0 through 23, one key per recorded local hour. For each key output an array of EXACTLY 3 integers in order [ordinary weekday, ordinary weekend, public holiday on a weekday], each from 0 to 10. Include EVERY hour 0 through 23; no extra keys, no explanation. Preserve qualitative electricity-use patterns for this building type. This is format attempt "+str(attempt)+".")
            ans=generation(r/f"{name}_format_repair{attempt}.json",repair);bank=parse(ans,True);mode="hour-indexed repair"+str(attempt)
        assert bank is not None,(name,"Teacher schema invalid after bounded format-only repair; no target-guided prompt search")
        maximum=float(bank.max());bank=bank*10/max(maximum,1e-8)
        out[name]=bank.tolist();status.append(dict(domain=name,format=mode,original_max=maximum,uniform_scale=10/max(maximum,1e-8),constant_profiles=(abs(bank-bank.mean(1,keepdims=True)).sum(1)<=1e-12).tolist()))
        print("VALID_PROFILE",name,mode,out[name],flush=True);assert time.monotonic()-start<c["extract_seconds"]
    save(r/"profiles.json",out);save(r/"profile_validation.json",status)
    save(r/"extraction.json",dict(seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,model=c["model"],revision=c["model_revision"],trainable=0,git=git(),format_repairs_only=True))

def initialize(c,config):
    r=Path(c["root"]);r.mkdir(parents=True,exist_ok=False)
    aq=read(c["acquisition"])
    for a in aq["files"]:assert sha(a["path"])==a["sha256"]
    assert aq["revision"]==c["revision"]
    f=pd.read_csv(c["data"],nrows=c["training_csv_nrows"]);t=pd.to_datetime(f.timestamp)
    assert len(f)==8472 and str(t.iloc[0])=="2016-01-01 00:00:00" and str(t.iloc[-1])=="2016-12-18 23:00:00"
    assert (t.diff().dropna()==pd.Timedelta(hours=1)).all()
    meta=pd.read_csv(c["metadata"]);eligible={};allrows=[]
    for cat in c["classes"]:
        good=[]
        for _,m in meta.loc[meta.primaryspaceusage==cat].sort_values("building_id").iterrows():
            name=m.building_id
            rec=dict(building=name,category=cat,site=str(m.site_id),timezone=str(m.timezone),eligible=False,counts={})
            if name not in f or pd.isna(m.timezone):rec["reason"]="No electricity column or timezone";allrows.append(rec);continue
            v=f[name].to_numpy(float);valid=np.isfinite(v)&(v>=0);rec["training_finite_fraction"]=float(valid.mean())
            v=np.where(valid,v,np.nan)
            for split,(lo,hi) in c["splits"].items():
                if split=="test":continue
                a=int((pd.Timestamp(lo)-t.iloc[0]).total_seconds()/3600);b=int((pd.Timestamp(hi)-t.iloc[0]).total_seconds()/3600)
                rec["counts"][split]=int(sum(np.isfinite(v[o-96:o+96]).all() for o in range(max(a,96),b-96+1,24)))
            a,b=map(pd.Timestamp,c["splits"]["base_fit"]);vb=v[(t>=a)&(t<b)];rec["base_sd"]=float(np.nanstd(vb)) if np.isfinite(vb).any() else 0.
            rec["eligible"]=bool(valid.mean()>=c["min_training_finite"] and rec["base_sd"]>1e-8 and all(rec["counts"][k]>=z for k,z in c["min_counts"].items()))
            vlo,vhi=map(pd.Timestamp,c["splits"]["val"]);aa=int((vlo-t.iloc[0]).total_seconds()/3600);bb=int((vhi-t.iloc[0]).total_seconds()/3600)
            vo=[o for o in range(aa,bb-96+1,24) if np.isfinite(v[o-96:o+96]).all()];cut=int((pd.Timestamp(c["gate_cal_end"])-t.iloc[0]).total_seconds()/3600)
            rec["counts"]["gate_cal"]=int(sum(o+96<=cut for o in vo));rec["counts"]["gate_check"]=int(sum(o>=cut for o in vo))
            rec["eligible"]=bool(rec["eligible"] and rec["counts"]["gate_cal"]>=30 and rec["counts"]["gate_check"]>=6)
            rec["reason"]="training-only quality pass" if rec["eligible"] else "training completeness/variance/calibration threshold"
            allrows.append(rec)
            if rec["eligible"]:good.append(rec)
        assert len(good)>=2,(cat,len(good),"training-only eligibility; do not inspect test")
        first=good[0];second=next((a for a in good[1:] if a["site"]!=first["site"]),good[1]);eligible[cat]=[first,second]
    selected=[a for cat in c["classes"] for a in eligible[cat]]
    save(r/"selection_buildings.json",dict(rule=c["selection_rule"],selected=selected,all_training_eligibility=allrows,no_2017_targets_read=True))
    save(r/"config.json",c)
    assets=read("/home/home_new/qsmx/runs/ChronoGeom/downstream/energy_confirm31_tetouan_20261101_01/old_assets.json")
    for root,names in [("/home/home_new/qsmx/runs/ChronoGeom/downstream/energy_switch32_20261101_03",["models.json","predictions.npz","metrics.json"]),("/home/home_new/qsmx/runs/ChronoGeom/downstream/energy_confirm31_tetouan_20261101_01",["test_predictions.npz","metrics.json","QwenEnergy_switch.pt"])]:
        for name in names:assets.append(dict(path=str(Path(root)/name),sha=sha(Path(root)/name)))
    for a in assets:assert sha(a["path"])==a["sha"]
    save(r/"old_assets.json",assets);save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),acquisition=aq,selected_only_training=True))
    worker_configs=[]
    for a in selected:
        cc={**c,"target":a["building"],"category":a["category"],"timezone":a["timezone"],"site":a["site"],"run":str(r/a["building"]),"building_ids":[a["building"] for a in selected]}
        path=r/(a["building"]+"_config.json");save(path,cc);worker_configs.append(str(path))
    save(r/"workers.json",worker_configs)
    print("SELECTED_BUILDINGS",selected,flush=True)
    extract_profiles(c,r)

def apply_signed(gate,history,reference,candidate,pw,fw):
    positive,info=gate(history,reference,candidate,pw,fw)
    d=center(candidate-reference);B=d.square().mean((1,2),keepdim=True);aa=torch.where(B>1e-12,(info["cross_energy_estimate"].reshape(-1,1,1)/B.clamp_min(1e-12)).clamp(-1,1),torch.zeros_like(B))
    info["weight"]=aa
    return reference+aa*d,info

def orchestrate(c,config):
    import subprocess
    r=Path(c["root"]);start=time.monotonic();jobs=[]
    for j,conf in enumerate(read(r/"workers.json")):
        cc=read(conf);log=(r/(cc["target"]+".log")).open("w")
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=c["worker_gpu_uuids"][j]);proc=subprocess.Popen([sys.executable,"-u","-m","exp.exp_chronogeom_bdg2_confirm33","--config",conf,"--stage","worker"],stdout=log,stderr=subprocess.STDOUT,env=env)
        jobs.append((proc,log,cc["target"]))
    try:
        while True:
            state={name:p.poll() for p,_,name in jobs}
            if any(v is not None and v!=0 for v in state.values()):raise RuntimeError("Worker failure; preserve logs and checkpoints: "+str(state))
            if all(v==0 for v in state.values()):break
            if time.monotonic()-start>c["overall_seconds"]:raise TimeoutError("Bounded pilot wall budget reached")
            time.sleep(2)
    except BaseException:
        for p,_,_ in jobs:
            if p.poll() is None:p.terminate()
        raise
    finally:
        for _,log,_ in jobs:log.close()
    save(r/"status.json",dict(complete=True,seconds=time.monotonic()-start,workers=len(jobs),git=git(),test_evaluations_per_building=1))
    print("SIX_BUILDINGS_COMPLETE",flush=True)

def fit_and_evaluate(c):
    r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c);save(r/"old_assets.json",read(Path(c["root"])/"old_assets.json"));save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config=c,LLM_updates=0));save(r/"profiles.json",read(Path(c["root"])/"profiles.json"))
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.04);device=torch.device("cuda:0");start=time.monotonic()
    for a in read(r/"old_assets.json"):assert sha(a["path"])==a["sha"]
    packs,grid,raw,common,kind,h,wd,scale=dataset(c,r,False)
    assert "test" not in packs
    save(r/"scale.json",scale)
    for split,p in packs.items():
        p["x"]=p["x"].to(device);p["ids"]=p["ids"].to(device)
    np.savez_compressed(r/"fit_inputs.npz",**{f"{split}_{k}":v.cpu().numpy() if torch.is_tensor(v) else v for split,p in packs.items() for k,v in p.items()})
    fits=[];base_models={};basepred={};originalpred={};base_curves=[];base_choices={}
    def guard():assert time.monotonic()-start<c["fit_seconds"],"bounded pilot budget exceeded"
    for seed in c["seeds"]:
        base_models[seed]={};basepred[seed]={};originalpred[seed]={};base_choices[seed]={}
        for arm,fitname,valname,table in [("FrozenTimeBase","base_fit","base_val",np.zeros((168,168),np.float32))]:
            model=CalendarTimeBase(c,seed,table).to(device);opt=torch.optim.Adam(model.parameters(),lr=c["base_lr"],foreach=False)
            batch=np.random.default_rng(seed+271).integers(0,len(packs[fitname]["y"]),size=(600,32));target=torch.tensor(packs[fitname]["y"],dtype=torch.float32,device=device);states={};cv=[]
            for step in range(601):
                if step%100==0:
                    pp=predict(model,packs[valname]);q=dict(seed=seed,arm=arm,step=step,**scores(pp,packs[valname]["y"]));cv.append(q);base_curves.append(q);states[step]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if step==600:break
                ids=torch.as_tensor(batch[step],device=device);model.train();pr=model(packs[fitname]["x"][ids],packs[fitname]["ids"][ids]);loss=(pr-target[ids]).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();guard()
            chosen=min(cv,key=lambda x:(x["mse"],x["step"]))["step"];model.load_state_dict(states[chosen]);model.eval()
            for param in model.parameters():param.requires_grad_(False)
            base_models[seed][arm]=model;base_choices[seed][arm]=chosen;torch.save(model.state_dict(),r/f"{arm}_{seed}.pt")
            dest=basepred if arm=="FrozenTimeBase" else originalpred
            for split in ["fit","val"]:dest[seed][split]=torch.tensor(predict(model,packs[split]),dtype=torch.float32,device=device)
            fits.append(dict(seed=seed,arm=arm,steps=600,parameters=sum(p.numel() for p in model.parameters()),selected_step=chosen,fit=fitname))
            print("BASE",seed,arm,chosen,flush=True)
    save(r/"base_curves.json",base_curves);save(r/"base_selection.json",base_choices)
    for seed in c["seeds"]:
        model=base_models[seed]["FrozenTimeBase"]
        with torch.no_grad():assert torch.equal(model(packs["fit"]["x"][:2],packs["fit"]["ids"][:2]),model(packs["fit"]["x"][:2],(packs["fit"]["ids"][:2]+24)%168))
    profile=np.array(read(r/"profiles.json")[c["category"]]);wrong_profile=np.array(read(r/"profiles.json")[c["wrong_context"][c["category"]]])
    K=dict(Fourier=np.stack([np.sin(10*np.pi*h/24),np.cos(10*np.pi*h/24),np.cos(12*np.pi*h/24)],1),Qwen=knowledge(profile,kind,h),Wrong=knowledge(wrong_profile,kind,h))
    preps={};z={};norm={};nums={};batches={}
    for seed in c["seeds"]:
        preps[seed]={};z[seed]={};nums[seed]={}
        for split in ["fit","val"]:nums[seed][split]=numeric_summary(packs[split]["x"],basepred[seed][split]).cpu().numpy()
        ft=np.concatenate([nums[seed]["fit"],raw[packs["fit"]["future"]],common[packs["fit"]["future"]]],-1);mu=ft.mean((0,1));sd=np.maximum(ft.std((0,1)),1e-6)
        norm[seed]=scores(basepred[seed]["fit"].cpu().numpy(),packs["fit"]["y"]);batches[seed]=torch.as_tensor(np.random.default_rng(seed).integers(0,len(packs["fit"]["y"]),size=(600,32)),device=device)
        for arm in ["Qwen","Fourier","Wrong"]:
            kk=K[arm][packs["fit"]["future"]];km=kk.mean((0,1));ks=max(float(np.sqrt(np.mean((kk-km)**2))),1e-6);preps[seed][arm]=dict(mean=mu,sd=sd,km=km,ks=ks);z[seed][arm]={}
    def features(seed,split,arm,override=None):
        p=packs[split];ix=p["future"] if override is None else override;n=preps[seed][arm]
        ft=np.concatenate([nums[seed][split],raw[ix],common[ix]],-1)
        return torch.tensor(np.concatenate([(ft-n["mean"])/n["sd"],(K[arm][ix]-n["km"])/n["ks"]],-1),dtype=torch.float32,device=device)
    for seed in c["seeds"]:
        for arm in ["Qwen","Fourier","Wrong"]:
            for split in ["fit","val"]:z[seed][arm][split]=features(seed,split,arm)
    save(r/"normalizers.json",{seed:{arm:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in pp.items()} for arm,pp in arms.items()} for seed,arms in preps.items()})
    # Same28 context relation; dates selected fromfit calendar ONLY.
    lo,hi=map(pd.Timestamp,c["splits"]["fit"]);pairs=[]
    for day in pd.date_range(lo,hi-pd.Timedelta(days=9),freq="D"):
        if day.dayofweek==0:
            ix=grid.get_loc(day)+np.arange(96);jx=grid.get_loc(day+pd.Timedelta(days=5))+np.arange(96)
            assert grid[jx[-1]]<hi;pairs.append((ix,jx))
    ctx={split:{} for split in ["fit","val"]}
    for split in ctx:
        dates=grid[packs[split]["origin"]];mondays=np.array([grid[x[0][0]].value for x in pairs])
        assignment=np.abs(np.array([(d-pd.Timedelta(days=d.dayofweek)).value for d in dates])[:,None]-mondays[None,:]).argmin(1)
        for name,i in [("work",0),("weekend",1)]:ctx[split][name]=np.stack([pairs[j][i] for j in assignment])
    W=torch.zeros(2,96,dtype=torch.float64,device=device);W[0,[7,8,9]]=1/3;W[1,[16,17,18,19]]=1/4;W[:,[11,12,13,14]]=-1/4
    def relation(x):return torch.einsum("nhc,kh->nkc",x,W).squeeze(-1)
    testvec=torch.zeros(1,96,1,dtype=torch.float64,device=device);testvec[:,[7,8,9]]=1;assert torch.allclose(relation(testvec),torch.tensor([[1.,0.]],device=device,dtype=torch.float64),atol=1e-12)
    virtual={};teacher_virtual={};xstd={split:packs[split]["x"].double().std(1,unbiased=False).reshape(-1,1).clamp_min(1e-6) for split in ["fit","val"]}
    for seed in c["seeds"]:
        virtual[seed]={split:{name:features(seed,split,"Fourier",ix) for name,ix in ctx[split].items()} for split in ["fit","val"]}
        teacher_virtual[seed]={arm:{split:{name:features(seed,split,arm,ix) for name,ix in ctx[split].items()} for split in ["fit","val"]} for arm in ["Qwen","Fourier","Wrong"]}
    target=torch.tensor(packs["fit"]["y"],dtype=torch.float64,device=device);curves=[];headstates={};teacher_models={};soft={};prob={};ent={};student_models={}
    def fit_head(seed,name,ztrain,softtarget=None,relprob=None,teacher_arm=None,teacher_training=False,alternate=False):
        initialseed=seed+50000 if alternate else seed
        head=Head(initialseed).to(device);bp=basepred[seed]["fit"]
        assert sum(p.numel() for p in head.parameters())==705 and torch.equal(forward(head,ztrain,bp),bp.double())
        lr=c["teacher_lr"] if teacher_training else c["student_lr"];opt=torch.optim.Adam(head.parameters(),lr=lr,foreach=False)
        batch=torch.as_tensor(np.random.default_rng(seed+50000).integers(0,len(target),size=(600,32)),device=device) if alternate else batches[seed]
        out={};arm=teacher_arm if teacher_training else "Fourier"
        for step in range(601):
            if step%100==0:
                out[step]={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                with torch.no_grad():pv=forward(head,z[seed][arm]["val"],basepred[seed]["val"]).cpu().numpy()
                curves.append(dict(seed=seed,arm=name,step=step,**scores(pv,packs["val"]["y"])))
            if step==600:break
            ix=batch[step];pr=forward(head,ztrain[ix],bp[ix]);err=pr-target[ix]
            loss=.5*(err.square().mean()/norm[seed]["mse"]+err.abs().mean()/norm[seed]["mae"])
            if softtarget is not None:loss=loss+c["kd_weight"]*(center(pr)-softtarget[ix]).square().mean()/norm[seed]["mse"]
            if relprob is not None:
                rv=relation(forward(head,virtual[seed]["fit"]["weekend"][ix],bp[ix])-forward(head,virtual[seed]["fit"]["work"][ix],bp[ix]));lp=rv/xstd["fit"][ix]
                loss=loss+c["kd_weight"]*(F.binary_cross_entropy_with_logits(lp,relprob[ix])-F.binary_cross_entropy(relprob[ix],relprob[ix]))
            opt.zero_grad();loss.backward();opt.step();guard()
        step=c["teacher_step"] if teacher_training else c["student_step"];head.load_state_dict(out[step]);head.eval()
        for p in head.parameters():p.requires_grad_(False)
        fits.append(dict(seed=seed,arm=name,steps=600,selected_step=step,lr=lr,parameters=705,initialization_seed=initialseed))
        print("HEAD",seed,name,step,flush=True)
        return head,out
    for seed in c["seeds"]:
        teacher_models[seed]={};headstates[seed]={};soft[seed]={};prob[seed]={};ent[seed]={};student_models[seed]={}
        for arm in ["Qwen","Fourier","Wrong"]:
            model,st=fit_head(seed,"Teacher"+arm,z[seed][arm]["fit"],teacher_arm=arm,teacher_training=True)
            teacher_models[seed][arm]=model;headstates[seed]["Teacher"+arm]=st
            with torch.no_grad():
                soft[seed][arm]=center(forward(model,z[seed][arm]["fit"],basepred[seed]["fit"])).detach()
                tv=teacher_virtual[seed][arm]["fit"];rv=relation(forward(model,tv["weekend"],basepred[seed]["fit"])-forward(model,tv["work"],basepred[seed]["fit"]))
                prob[seed][arm]=torch.sigmoid(rv/xstd["fit"]).detach()
        jobs=[("NoKD",None,None,False),("NoKD2",None,None,True),("QwenKD23","Qwen",None,False),("QwenRelation",None,"Qwen",True),("FourierKD","Fourier",None,False),("FourierRelation",None,"Fourier",True),("WrongKD","Wrong",None,False),("WrongRelation",None,"Wrong",True)]
        for name,wave,rel,alternate in jobs:
            model,st=fit_head(seed,name,z[seed]["Fourier"]["fit"],None if wave is None else soft[seed][wave],None if rel is None else prob[seed][rel],alternate=alternate)
            student_models[seed][name]=model;headstates[seed][name]=st
    torch.save(headstates,r/"head_checkpoints.pt");save(r/"curves.json",curves);save(r/"fits.json",fits)
    # ONLY real internalval predictions calibrate fixed-form5parameter switches.
    valpred={};calpred={};gate_models={};gate_fits={};vp=packs["val"];end=vp["future"][:,-1]+1;cut=int((pd.Timestamp(c["gate_cal_end"])-grid[0]).total_seconds()/3600);checkend=int((pd.Timestamp(c["gate_check_end"])-grid[0]).total_seconds()/3600)
    trainids=np.flatnonzero(end<=cut);checkids=np.flatnonzero((vp["origin"]>=cut)&(end<=checkend));assert len(trainids)>=30 and len(checkids)>=6
    assert np.intersect1d(vp["future"][trainids].ravel(),vp["future"][checkids].ravel()).size==0
    past=vp["origin"][:,None]+np.arange(-96,0);pw=torch.tensor(wd[past]>=5,dtype=torch.float64,device=device);fw=torch.tensor(wd[vp["future"]]>=5,dtype=torch.float64,device=device)
    for seed in c["seeds"]:
        valpred[seed]={}
        with torch.inference_mode():
            for name,model in student_models[seed].items():valpred[seed][name]=forward(model,z[seed]["Fourier"]["val"],basepred[seed]["val"]).detach()
    families=dict(QwenEnergy=("QwenKD23","QwenRelation"),FourierEnergy=("FourierKD","FourierRelation"),NoKDEnergy=("NoKD","NoKD2"),WrongEnergy=("WrongKD","WrongRelation"))
    for family,(refname,candname) in families.items():
        gate=EnergySwitch().double().to(device);X=[];Y=[]
        for seed in c["seeds"]:
            b=valpred[seed][refname];cand=valpred[seed][candname];feat=gate.observable_features(vp["x"].double(),b,cand,pw,fw);X.append(feat[trainids])
            yy=torch.tensor(vp["y"][trainids],device=device);Y.append(((yy-b[trainids])*center(cand-b)[trainids]).mean((1,2)))
        XX=torch.cat(X);YY=torch.cat(Y);gate.set_training_normalizer(XX)
        design=torch.cat([(XX-gate.feature_mean)/gate.feature_scale,torch.ones(len(XX),1,device=device)],1);coef=torch.linalg.pinv(design)@YY
        with torch.no_grad():gate.alignment.weight.copy_(coef[:4].reshape(1,4));gate.alignment.bias.copy_(coef[4:])
        gate.eval()
        for p in gate.parameters():p.requires_grad_(False)
        gate_models[family]=gate;torch.save(gate.state_dict(),r/f"{family}_switch.pt")
        gate_fits[family]=dict(parameters=5,calibration_origins=len(trainids),check_origins=len(checkids),shared3seeds=True,coefficients=coef.cpu().tolist(),rank=int(torch.linalg.matrix_rank(design)))
    save(r/"gate_fits.json",gate_fits)
    # All weights, thresholds, choices and normalizers locked before firsttestdata load.
    save(r/"selection.json",dict(base_choices=base_choices,student_step=500,teacher_step=600,gate_cal_end=c["gate_cal_end"],no_hyperparameter_search=True,git=git()))
    files=[p for p in r.iterdir() if p.suffix==".pt"]+[r/"normalizers.json",r/"scale.json",r/"selection.json",r/"gate_fits.json",r/"profiles.json"]
    weightlock=[dict(path=str(p),sha=sha(p)) for p in files];save(r/"model_lock.json",dict(files=weightlock,before_first_test_data=True,fit_updates=len(fits)*600,git=git()))
    locksha=sha(r/"model_lock.json");save(r/"fit_status.json",dict(seconds=time.monotonic()-start,fits=len(fits),updates=len(fits)*600,test_opened=False))
    # ALL six model locks must exist before ANY worker opens its 2017 values.
    barrier_start=time.monotonic();locks=[Path(c["root"])/bid/"model_lock.json" for bid in c["building_ids"]]
    while not all(p.exists() for p in locks):
        assert time.monotonic()-barrier_start<c["barrier_seconds"],"All-building test firewall timeout"
        time.sleep(2)
    save(r/"all_building_locks.json",[dict(path=str(p),sha=sha(p)) for p in locks])
    # No training or model choices are allowed below this point.
    testpacks,tgrid,traw,tcommon,tkind,th,twd,tscale=dataset(c,r,True);assert tscale==scale
    tp=testpacks["test"];tp["x"]=tp["x"].to(device);tp["ids"]=tp["ids"].to(device);packs["test"]=tp
    assert np.intersect1d(tp["future"].ravel(),vp["future"].ravel()).size==0
    raw=traw;common=tcommon;nums_test={};testpred={};gateinfo={};replays=0;maxmean=0.
    testpast=tp["origin"][:,None]+np.arange(-96,0);tpw=torch.tensor(twd[testpast]>=5,dtype=torch.float64,device=device);tfw=torch.tensor(twd[tp["future"]]>=5,dtype=torch.float64,device=device)
    with torch.inference_mode():
        for seed in c["seeds"]:
            for arm,model in base_models[seed].items():testpred[f"{seed}_{arm}"]=predict(model,tp).astype(float)
            basepred[seed]["test"]=torch.tensor(testpred[f"{seed}_FrozenTimeBase"],dtype=torch.float32,device=device)
            nums[seed]["test"]=numeric_summary(tp["x"],basepred[seed]["test"]).cpu().numpy()
            zz=features(seed,"test","Fourier");local={}
            for name,model in student_models[seed].items():
                pr=forward(model,zz,basepred[seed]["test"]);local[name]=pr;testpred[f"{seed}_{name}"]=pr.cpu().numpy()
                replay=Head(seed).to(device);replay.load_state_dict(headstates[seed][name][500]);assert torch.equal(pr,forward(replay,zz,basepred[seed]["test"]));replays+=1
            for family,(refname,candname) in families.items():
                gate=gate_models[family];out,info=apply_signed(gate,tp["x"].double(),local[refname],local[candname],tpw,tfw)
                again,_=apply_signed(gate,tp["x"].double(),local[refname],local[candname],tpw,tfw);assert torch.equal(out,again);replays+=1
                if family=="QwenEnergy":
                    po,_=gate(tp["x"].double(),local[refname],local[candname],tpw,tfw);testpred[f"{seed}_QwenPositive"]=po.cpu().numpy()
                testpred[f"{seed}_{family}"]=out.cpu().numpy();gateinfo[f"{seed}_{family}_weight"]=info["weight"].cpu().numpy();gateinfo[f"{seed}_{family}_Ahat"]=info["cross_energy_estimate"].cpu().numpy()
                maxmean=max(maxmean,float(abs((out-local[refname]).mean(1)).max()))
            print("TEST_PREDICTION",seed,len(tp["origin"]),flush=True)
    np.savez_compressed(r/"test_predictions.npz",**testpred);np.savez_compressed(r/"gate_information.npz",**gateinfo)
    save(r/"prediction_lock.json",dict(model_lock_sha=locksha,predictions_sha=sha(r/"test_predictions.npz"),gateinfo_sha=sha(r/"gate_information.npz"),before_first_test_metric=True,git=git()))
    arms=["FrozenTimeBase","NoKD","NoKD2","FourierKD","FourierRelation","QwenKD23","QwenRelation","WrongKD","WrongRelation","NoKDEnergy","FourierEnergy","WrongEnergy","QwenPositive","QwenEnergy"]
    metrics=[];mis=[];perwindow=[];Y=tp["y"];maxidentity=0.;maxlevel=0.
    for seed in c["seeds"]:
        for arm in arms:
            pr=testpred[f"{seed}_{arm}"];q=scores(pr,Y);metrics.append(dict(seed=seed,arm=arm,**q))
            if arm not in ["OriginalTimeBase"]:maxlevel=max(maxlevel,abs(q["level"]-scores(testpred[f"{seed}_FrozenTimeBase"],Y)["level"]))
        b=testpred[f"{seed}_QwenKD23"];out=testpred[f"{seed}_QwenEnergy"];d=testpred[f"{seed}_QwenRelation"]-b;d-=d.mean(1,keepdims=True);A=((Y-b)*d).mean((1,2));B=(d*d).mean((1,2));alpha=gateinfo[f"{seed}_QwenEnergy_weight"].reshape(-1);eb=b-Y;eo=out-Y
        maxidentity=max(maxidentity,float(abs((eo*eo).mean((1,2))-(eb*eb).mean((1,2))+2*alpha*A-alpha*alpha*B).max()))
        for metric in ["mse","mae"]:
            fn=lambda e:(e*e if metric=="mse" else abs(e)).mean((1,2));gain=fn(eb)-fn(eo)
            mis.append(dict(seed=seed,metric=metric,windows=len(gain),helpful=int((gain>0).sum()),harmful=int((gain<0).sum()),unchanged=int((gain==0).sum()),positive_gain_mean=float(np.maximum(gain,0).mean()),miscorrection_loss_mean=float(np.maximum(-gain,0).mean()),net_gain_mean=float(gain.mean()),mean_gate=float(alpha.mean())))
        for i,o in enumerate(tp["origin"]):
            perwindow.append(dict(seed=seed,origin=str(tgrid[o]),gate=float(alpha[i]),mse23=float((eb[i]**2).mean()),mse_switch=float((eo[i]**2).mean()),mae23=float(abs(eb[i]).mean()),mae_switch=float(abs(eo[i]).mean()),mse_gain=float((eb[i]**2-eo[i]**2).mean()),mae_gain=float((abs(eb[i])-abs(eo[i])).mean())))
    save(r/"metrics.json",metrics);save(r/"miscorrections.json",mis);np.savez_compressed(r/"test_labels.npz",y=Y,origin=tp["origin"],future=tp["future"],x=tp["x"].cpu().numpy())
    with (r/"per_window.csv").open("w",newline="") as f:writer=csv.DictWriter(f,fieldnames=list(perwindow[0]));writer.writeheader();writer.writerows(perwindow)
    summary={a:{k:float(np.mean([q[k] for q in metrics if q["arm"]==a])) for k in ["mse","mae","level","shape"]} for a in arms}
    intervals=[];rng=np.random.default_rng(2024);clusters=(tp["origin"]-tp["origin"].min())//336;unique=np.unique(clusters)
    for ref in ["QwenKD23","NoKD","FourierKD","NoKDEnergy","FourierEnergy","WrongEnergy"]:
        for metric in ["mse","mae"]:
            delta=[]
            for seed in c["seeds"]:
                a=testpred[f"{seed}_QwenEnergy"]-Y;b=testpred[f"{seed}_{ref}"]-Y;delta.append(((a*a-b*b) if metric=="mse" else (abs(a)-abs(b))).mean((1,2)))
            dd=np.mean(delta,0);bs=[]
            for _ in range(2000):
                ix=np.concatenate([np.flatnonzero(clusters==j) for j in rng.choice(unique,len(unique),replace=True)]);bs.append(float(dd[ix].mean()))
            intervals.append(dict(reference=ref,metric=metric,delta=float(dd.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),blocks=len(unique),note="Paired14dayblocks,seedlossesaveragedfirst;overlappingwindowsnotIID."))
    assert maxmean<1e-12 and maxidentity<1e-12 and maxlevel<1e-10
    assert sha(r/"model_lock.json")==locksha
    for a in weightlock+read(r/"old_assets.json"):assert sha(a["path"])==a["sha"]
    save(r/"summary.json",dict(metrics=summary,paired_intervals=intervals))
    save(r/"verification.json",dict(fits=len(fits),updates=len(fits)*600,allhead600_and_primary500=True,shared3seed5param_gates=True,matched_energy_family_head_parameters=1410,gate_parameters=5,extraLabelsEqualAcrossGateFamilies=True,all_models_locked_before_firsttest=True,test_predictions_locked_before_metrics=True,prediction_replays=replays,maximum_mean_shift=maxmean,maximum_level_error_difference=maxlevel,maximum_energy_identity_error=maxidentity,old_Metro_Capital_SHA_unchanged=True,LLM_updates=0,one_test_evaluation=True))
    save(r/"status.json",dict(seconds=time.monotonic()-start,complete=True,git=git(),test_evaluations=1,new_dataset_independent_confirmation=True,no_post_test_changes=True))
    print("TEST_RESULT",summary,flush=True);print("MISCORRECTIONS",mis,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--stage",choices=["init","extract","run","worker"],required=True);a=ap.parse_args();c=read(a.config)
    if a.stage=="init":initialize(c,a.config)
    elif a.stage=="extract":extract_profiles(c,Path(c["root"]))
    elif a.stage=="run":orchestrate(c,a.config)
    else:fit_and_evaluate(c)
