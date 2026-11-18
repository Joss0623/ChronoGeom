"""Fixed no-gate ChronoGeom on six previously unused BDG2 buildings/sites."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",CUBLAS_WORKSPACE_CONFIG=":4096:8",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,csv,sys,time,subprocess
from pathlib import Path
import numpy as np,pandas as pd,torch
from exp.exp_chronogeom_event_knowledge19 import Head,forward,scores,read,save,sha,git
from exp.exp_chronogeom_downstream import CalendarTimeBase,predict
from exp.exp_chronogeom_residual03 import numeric_summary
from exp.exp_chronogeom_bdg2_confirm33 import dataset

def center(x):return x-x.mean(1,keepdim=True)

def initialize(c,config):
    r=Path(c["root"]);r.mkdir(parents=True,exist_ok=False)
    catalog=read(c["eligibility_source"]);used={a["building"] for a in catalog["selected"]};sites={a["site"] for a in catalog["selected"]}
    selected=[];counts={}
    for cat in c["classes"]:
        good=[a for a in catalog["all_training_eligibility"] if a["category"]==cat and a["eligible"] and a["building"] not in used]
        good.sort(key=lambda a:(a["site"] in sites,a["building"]));assert len(good)>=2
        first=good[0];second=next((a for a in good[1:] if a["site"]!=first["site"]),good[1]);selected += [first,second];counts[cat]=len(good)
    assert len(set(a["building"] for a in selected))==6 and not any(a["building"] in used for a in selected)
    selection=dict(selected=selected,used_buildings=sorted(used),used_sites=sorted(sites),selected_sites=sorted({a["site"] for a in selected}),all_selected_sites_unseen=not any(a["site"] in sites for a in selected),eligible_count=counts,rule=c["selection_rule"],source_catalog_sha=sha(c["eligibility_source"]),no_new_2017_targets_accessed=True)
    save(r/"selection_buildings.json",selection);save(r/"config.json",c)
    src=Path(c["teacher_source"]);assets=read(src/"old_assets.json")
    for name in ["profiles.json","responses.json","response_cache.npz","summary.json","selection_buildings.json"]:
        assets.append(dict(path=str(src/name),sha=sha(src/name)))
    assets += [dict(path=c["main_manifest"],sha=sha(c["main_manifest"])),dict(path=c["eligibility_source"],sha=sha(c["eligibility_source"])),dict(path="docs/chronogeom/MAIN_MODEL_NO_GATE.json",sha=sha("docs/chronogeom/MAIN_MODEL_NO_GATE.json"))]
    for root in [src,Path(read(c["main_manifest"])["source_run"])]:
        for meta in catalog["selected"]:
            old=root/meta["building"]
            for name in ["model_lock.json","head_checkpoints.pt","test_predictions.npz","metrics.json"]:assets.append(dict(path=str(old/name),sha=sha(old/name)))
    for item in read(c["acquisition"])["files"]:assert sha(item["path"])==item["sha256"]
    for a in assets:assert sha(a["path"])==a["sha"]
    save(r/"old_assets.json",assets);save(r/"profiles.json",read(src/"profiles.json"))
    save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),cache_sha=sha(src/"profiles.json"),LLM_calls=0,teacher_cache_reused=True,new_buildings=True,new_sites=selection["all_selected_sites_unseen"]))
    workers=[]
    for j,a in enumerate(selected):
        cc={**c,"target":a["building"],"category":a["category"],"site":a["site"],"timezone":a["timezone"],"run":str(r/a["building"]),"building_ids":[x["building"] for x in selected],"gpu_uuid":c["worker_gpu_uuids"][j]}
        path=r/(a["building"]+"_config.json");save(path,cc);workers.append(str(path))
    save(r/"workers.json",workers);print("NEW_BUILDINGS",selection,flush=True)

def orchestrate(c):
    r=Path(c["root"]);start=time.monotonic();jobs=[]
    for j,cf in enumerate(read(r/"workers.json")):
        cc=read(cf);log=(r/(cc["target"]+".log")).open("w");env=dict(os.environ,CUDA_VISIBLE_DEVICES=c["worker_gpu_uuids"][j])
        p=subprocess.Popen([sys.executable,"-u","-m","exp.exp_chronogeom_bdg2_nogate_confirm36","--config",cf,"--stage","worker"],stdout=log,stderr=subprocess.STDOUT,env=env);jobs.append((p,log,cc["target"]))
    try:
        while True:
            state={n:p.poll() for p,_,n in jobs}
            if any(v is not None and v!=0 for v in state.values()):raise RuntimeError("Worker failed; preserve all logs: "+str(state))
            if all(v==0 for v in state.values()):break
            assert time.monotonic()-start<c["overall_seconds"],"Bounded pilot wall limit";time.sleep(2)
    except BaseException:
        for p,_,_ in jobs:
            if p.poll() is None:p.terminate()
        raise
    finally:
        for _,f,_ in jobs:f.close()
    save(r/"status.json",dict(complete=True,seconds=time.monotonic()-start,workers=6,git=git(),no_LLM_calls=True,new_buildings_sites_confirmation=True));print("CONFIRMATION_COMPLETE",flush=True)

def worker(c):
    r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.04);device=torch.device("cuda:0");start=time.monotonic()
    assets=read(Path(c["root"])/"old_assets.json")
    for a in assets:assert sha(a["path"])==a["sha"]
    save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),LLM_calls=0,gate_parameters=0,auxiliary_heads=0))
    packs,grid,raw,common,kind,h,wd,scale=dataset(c,r,False);assert "test" not in packs;save(r/"scale.json",scale)
    for p in packs.values():p["x"]=p["x"].to(device);p["ids"]=p["ids"].to(device)
    np.savez_compressed(r/"fit_inputs.npz",**{f"{split}_{k}":v.cpu().numpy() if torch.is_tensor(v) else v for split,p in packs.items() for k,v in p.items()})
    def guard():assert time.monotonic()-start<c["fit_seconds"],"Pilot fit activity budget"
    fits=[];curves=[];bases={};basepred={};choices={};norms={};normalizers={};headstates={};students={};teachers={};z={}
    # New buildings use the SAME main35 baseline fitting/selection procedure, never old-building weights.
    for seed in c["seeds"]:
        model=CalendarTimeBase(c,seed,np.zeros((168,168),np.float32)).to(device);opt=torch.optim.Adam(model.parameters(),lr=c["base_lr"],foreach=False)
        batches=np.random.default_rng(seed+271).integers(0,len(packs["base_fit"]["y"]),size=(600,32));y=torch.tensor(packs["base_fit"]["y"],dtype=torch.float32,device=device);states={};records=[]
        for step in range(601):
            if step%100==0:
                q=dict(seed=seed,arm="FrozenTimeBase",step=step,**scores(predict(model,packs["base_val"]),packs["base_val"]["y"]));records.append(q);curves.append(q);states[step]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if step==600:break
            ix=torch.as_tensor(batches[step],device=device);model.train();loss=(model(packs["base_fit"]["x"][ix],packs["base_fit"]["ids"][ix])-y[ix]).square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();guard()
        pick=min(records,key=lambda x:(x["mse"],x["step"]))["step"];model.load_state_dict(states[pick]);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        bases[seed]=model;choices[seed]=pick;torch.save(model.state_dict(),r/f"FrozenTimeBase_{seed}.pt")
        basepred[seed]={s:torch.tensor(predict(model,packs[s]),dtype=torch.float32,device=device) for s in ["fit","val"]}
        fits.append(dict(seed=seed,arm="FrozenTimeBase",updates=600,selected_step=pick,parameters=6570));print("BASE",seed,pick,flush=True)
    profiles=read(Path(c["root"])/"profiles.json")
    K=dict(Fourier=np.stack([np.sin(10*np.pi*h/24),np.cos(10*np.pi*h/24),np.cos(12*np.pi*h/24)],1),Qwen=np.array(profiles[c["category"]])[kind,h],Wrong=np.array(profiles[c["wrong_context"][c["category"]]])[kind,h])
    nums={};prep={}
    for seed in c["seeds"]:
        nums[seed]={s:numeric_summary(packs[s]["x"],basepred[seed][s]).cpu().numpy() for s in ["fit","val"]};prep[seed]={};z[seed]={}
        f=np.concatenate([nums[seed]["fit"],raw[packs["fit"]["future"]],common[packs["fit"]["future"]]],-1);mu=f.mean((0,1));sd=f.std((0,1)).clip(1e-6)
        n=scores(basepred[seed]["fit"].cpu().numpy(),packs["fit"]["y"]);norms[seed]=dict(raw=n,mse=max(n["mse"],1e-12),mae=max(n["mae"],1e-12))
        for arm in ["Qwen","Fourier","Wrong"]:
            k=K[arm][packs["fit"]["future"]];km=k.mean((0,1));ks=max(float(np.sqrt(np.mean((k-km)**2))),1e-6);prep[seed][arm]=dict(mean=mu,sd=sd,km=km,ks=ks)
    def feature(seed,split,arm):
        ix=packs[split]["future"];n=prep[seed][arm];f=np.concatenate([nums[seed][split],raw[ix],common[ix]],-1)
        return torch.tensor(np.concatenate([(f-n["mean"])/n["sd"],(K[arm][ix]-n["km"])/n["ks"]],-1),dtype=torch.float32,device=device)
    for seed in c["seeds"]:
        for arm in ["Qwen","Fourier","Wrong"]:z[seed][arm]={s:feature(seed,s,arm) for s in ["fit","val"]}
    save(r/"normalizers.json",{s:{a:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in n.items()} for a,n in d.items()} for s,d in prep.items()});save(r/"loss_scales.json",norms)
    y=torch.tensor(packs["fit"]["y"],dtype=torch.float64,device=device)
    def fit_head(seed,name,ztrain,arm,soft=None,teacher=False):
        model=Head(seed).to(device);assert sum(p.numel() for p in model.parameters())==705
        assert torch.equal(forward(model,ztrain,basepred[seed]["fit"]),basepred[seed]["fit"].double())
        opt=torch.optim.Adam(model.parameters(),lr=c["teacher_lr"] if teacher else c["student_lr"],foreach=False);batches=torch.as_tensor(np.random.default_rng(seed).integers(0,len(y),size=(600,32)),device=device);states={}
        for step in range(601):
            if step%100==0:
                states[step]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                with torch.no_grad():p=forward(model,z[seed][arm]["val"],basepred[seed]["val"]).cpu().numpy()
                curves.append(dict(seed=seed,arm=name,step=step,**scores(p,packs["val"]["y"])))
            if step==600:break
            ix=batches[step];p=forward(model,ztrain[ix],basepred[seed]["fit"][ix]);e=p-y[ix];n=norms[seed]
            loss=.5*(e.square().mean()/n["mse"]+e.abs().mean()/n["mae"])
            if soft is not None:loss=loss+c["kd_weight"]*(center(p)-soft[ix]).square().mean()/n["mse"]
            opt.zero_grad();loss.backward();opt.step();guard()
        pick=c["teacher_step"] if teacher else c["student_step"];model.load_state_dict(states[pick]);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        fits.append(dict(seed=seed,arm=name,updates=600,selected_step=pick,parameters=705));print("HEAD",seed,name,pick,flush=True)
        return model,states
    for seed in c["seeds"]:
        headstates[seed]={};students[seed]={};teachers[seed]={};soft={}
        for arm in ["Qwen","Fourier","Wrong"]:
            m,st=fit_head(seed,"Teacher"+arm,z[seed][arm]["fit"],arm,teacher=True);teachers[seed][arm]=m;headstates[seed]["Teacher"+arm]=st
            with torch.no_grad():soft[arm]=center(forward(m,z[seed][arm]["fit"],basepred[seed]["fit"])).detach()
        for name,teacher_arm in [("NoKD",None),("FourierKD","Fourier"),("WrongKD","Wrong"),("QwenTaskKD","Qwen")]:
            m,st=fit_head(seed,name,z[seed]["Fourier"]["fit"],"Fourier",None if teacher_arm is None else soft[teacher_arm]);students[seed][name]=m;headstates[seed][name]=st
    assert len(fits)==24;torch.save(headstates,r/"head_checkpoints.pt");save(r/"curves.json",curves);save(r/"fits.json",fits)
    save(r/"selection.json",dict(base_steps=choices,teacher_step=600,student_step=500,no_hyperparameter_search=True,selection_only_2016=True))
    files=[p for p in r.iterdir() if p.suffix==".pt"]+[r/"normalizers.json",r/"scale.json",r/"selection.json",r/"loss_scales.json"]
    locks=[dict(path=str(p),sha=sha(p)) for p in files];save(r/"model_lock.tmp.json",dict(files=locks,git=git(),before_2017_access=True));(r/"model_lock.tmp.json").rename(r/"model_lock.json")
    save(r/"fit_status.json",dict(seconds=time.monotonic()-start,fits=24,updates=14400,test_opened=False))
    locksha=sha(r/"model_lock.json");alllocks=[Path(c["root"])/x/"model_lock.json" for x in c["building_ids"]];bs=time.monotonic()
    while not all(p.exists() for p in alllocks):assert time.monotonic()-bs<c["barrier_seconds"];time.sleep(2)
    save(r/"all_building_locks.json",[dict(path=str(p),sha=sha(p)) for p in alllocks])
    # ZERO fitting or settings changes below. All six new building models fixed before first test access.
    try:testpacks,tgrid,raw,common,_,_,_,testscale=dataset(c,r,True)
    except AssertionError as exc:
        if "No complete windows" not in str(exc):raise
        save(r/"status.json",dict(complete=True,evaluable=False,reason=str(exc),retained_without_replacement=True,git=git()));print("NON_EVALUABLE",str(exc),flush=True);return
    assert testscale==scale;tp=testpacks["test"];tp["x"]=tp["x"].to(device);tp["ids"]=tp["ids"].to(device);packs["test"]=tp
    assert np.intersect1d(tp["future"].ravel(),packs["val"]["future"].ravel()).size==0
    predictions={};replays=0;maxmean=0.
    with torch.inference_mode():
        for seed in c["seeds"]:
            b=torch.tensor(predict(bases[seed],tp),dtype=torch.float32,device=device);basepred[seed]["test"]=b;nums[seed]["test"]=numeric_summary(tp["x"],b).cpu().numpy();zz=feature(seed,"test","Fourier")
            predictions[f"{seed}_FrozenTimeBase"]=b.cpu().numpy().astype(float)
            for name,m in students[seed].items():
                p=forward(m,zz,b);predictions[f"{seed}_{name}"]=p.cpu().numpy();clone=Head(seed).to(device);clone.load_state_dict(headstates[seed][name][500]);clone.eval();assert torch.equal(p,forward(clone,zz,b));replays+=1;maxmean=max(maxmean,float(abs((p-b).mean(1)).max()))
    np.savez_compressed(r/"test_predictions.npz",**predictions);save(r/"prediction_lock.json",dict(models_sha=locksha,predictions_sha=sha(r/"test_predictions.npz"),before_test_metrics=True))
    metrics=[]
    for seed in c["seeds"]:
        for name in ["NoKD","FourierKD","WrongKD","QwenTaskKD","FrozenTimeBase"]:metrics.append(dict(seed=seed,arm=name,**scores(predictions[f"{seed}_{name}"],tp["y"])))
    save(r/"metrics.json",metrics);save(r/"summary.json",{a:{k:float(np.mean([x[k] for x in metrics if x["arm"]==a])) for k in ["mse","mae","level","shape"]} for a in ["NoKD","FourierKD","WrongKD","QwenTaskKD","FrozenTimeBase"]})
    np.savez_compressed(r/"test_labels.npz",y=tp["y"],x=tp["x"].cpu().numpy(),origin=tp["origin"],future=tp["future"])
    for a in assets+locks:assert sha(a["path"])==a["sha"]
    assert maxmean<1e-12 and sha(r/"model_lock.json")==locksha
    save(r/"verification.json",dict(fits=24,updates=14400,student_head_parameters=705,frozen_base_parameters=6570,total_parameters=7275,gate_parameters=0,auxiliary_heads=0,LLM_calls=0,max_window_mean_shift=maxmean,selected_forecast_replays=replays,all_old_SHA_unchanged=True))
    save(r/"status.json",dict(complete=True,evaluable=True,seconds=time.monotonic()-start,test_evaluations=1,new_building_confirmation=True,git=git()))
    print("RESULT",read(r/"summary.json"),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--stage",choices=["init","run","worker"],required=True);a=ap.parse_args();c=read(a.config)
    if a.stage=="init":initialize(c,a.config)
    elif a.stage=="run":orchestrate(c)
    else:worker(c)

