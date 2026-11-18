"""Fixed51 transfer to2017 Metro; evaluate untouched2018 only after all method locks."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,time,csv
from pathlib import Path
import numpy as np,pandas as pd,torch
from exp.exp_chronogeom_event_knowledge19 import Head,forward,scores,knowledge,read,save,sha,git
from exp.exp_chronogeom_teacher_content40 import calendar_arrays,center,check22
from exp.exp_chronogeom_capital_representation_transfer44 import CalendarMap
from exp.exp_chronogeom_capital_conditional_consensus48 import simplex_weights
from exp.exp_chronogeom_capital_consensus_modes49 import orthogonal_modes
from exp.exp_chronogeom_capital_conditional_coupled50 import simplex_fit,prototypes,partition,coupled_solve
from exp.exp_chronogeom_capital_component_horizon51 import horizon_coordinate
from exp.exp_chronogeom_downstream import CalendarTimeBase,predict as base_predict
from exp.exp_chronogeom_residual03 import numeric_summary

def ridge(D,target,alpha,tau=None):
    X=D.reshape(-1,D.shape[-1]);scale=np.sqrt(len(X));U,sv,Vt=np.linalg.svd(X/scale,full_matrices=False)
    tau=float(np.mean(sv**2)) if tau is None else tau
    return Vt.T@((sv/(sv*sv+alpha*tau))*(U.T@(target.reshape(-1)/scale)))
def C(x):return x-x.mean(1,keepdims=True)
def Papply(P,x):return np.einsum("ht,ntd->nhd",P,x)
def design(phi,q,hor=False):
    if not hor:return (C(phi.double().numpy())[:,:,None,:]*q.double().numpy()[:,None,:,None]).reshape(len(phi),96,48)
    t=horizon_coordinate(96);f=torch.stack([phi,phi*t[None,:,None]],2).double().numpy()
    d=(f[:,:,:,None,:]*q.double().numpy()[:,None,None,:,None]).reshape(len(phi),96,96);return C(d)
def corr(phi,q,theta,P):
    theta=np.asarray(theta,np.float32)
    if theta.ndim==3:
        theta=np.stack([theta,np.zeros_like(theta)],axis=1)
    vals=[];t=horizon_coordinate(96)
    with torch.no_grad():
        for v in theta:
            a=center(torch.einsum("bhi,ji,bj->bh",phi,torch.tensor(v[0]),q).double().unsqueeze(-1))
            b=center(torch.einsum("bhi,ji,bj->bh",phi*t[None,:,None],torch.tensor(v[1]),q).double().unsqueeze(-1))
            vals.append((a+b).numpy())
    return C(Papply(P,vals[0])+vals[1]-Papply(P,vals[1]))
def calctx(x,stats=None):
    a=x.double().numpy()[:,:,0];sd=a.std(1);v=np.c_[np.log(np.maximum(sd,1e-6)),(a[:,-24:].mean(1)-a[:,:24].mean(1))/np.maximum(sd,1e-6)]
    if stats is None:return v
    return torch.tensor(np.c_[np.ones(len(v)),np.tanh((v-np.array(stats["mean"]))/np.array(stats["sd"]))],dtype=torch.float32)
def load_common(c):
    src=Path(c["metro_source"]);sc=read(src/"config.json");p=np.load(src/"data.npz");cache=np.load(src/"predictions.npz");norm=read(src/"normalizers.json");grid,raw,common,kind,h=calendar_arrays(sc,"Metro")
    return src,sc,p,cache,norm,grid,raw,common,kind,h

def fit(c,config):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);start=time.monotonic();r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c)
    assets=check22(c)
    for root in [Path(c["metro_source"]),Path(c["teacher_source"]),Path(c["source51"])]:
        assets += [dict(path=str(p),sha=sha(p)) for p in root.iterdir() if p.is_file() and p.suffix in [".json",".npz",".pt"]]
    save(r/"inputs.json",assets);save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),fixed_Capital51_recipe=True))
    src,sc,p,cache,norm,grid,raw,common,kind,h=load_common(c);cid=kind*24+h
    cal=grid[p["val_future"][:,-1]]<pd.Timestamp(c["calibration_target_end"]);select=grid[p["val_origin"]]>=pd.Timestamp(c["selection_origin_start"])
    assert cal.sum()>=3 and select.sum()>=3 and not np.any(cal&select)
    assert grid[p["val_future"][cal,-1]].max()<grid[p["val_origin"][select].min()-96]
    save(r/"time_split.json",dict(calibration_indices=np.flatnonzero(cal).tolist(),selection_indices=np.flatnonzero(select).tolist(),calibration_origins=grid[p["val_origin"][cal]].astype(str).tolist(),selection_origins=grid[p["val_origin"][select]].astype(str).tolist(),confirmation=c["confirmation"],no_new_2018_predictions=True))
    tr=Path(c["teacher_source"]);profiles=read(tr/"Metro/profiles.json");teacher_states=torch.load(tr/"teacher_checkpoints.pt",map_location="cpu",weights_only=True)["Metro"];chosen=read(tr/"teacher_selection.json")["Metro"];tp=np.load(tr/"teacher_predictions.npz")
    kk=np.repeat(np.arange(3),24);hh=np.tile(np.arange(24),3);table=knowledge(np.array(profiles["Task"]),kk,hh)-knowledge(np.array(profiles["Generic"]),kk,hh)
    tables={s:{} for s in c["seeds"]};data={};students={};bases={};maprows=[];bridges=[]
    ctxstats=None
    for s in c["seeds"]:
        pp=norm[str(s)]["Fourier"];kf=table[cid[p["fit_future"]]];km=kf.mean((0,1));kr=max(float(np.sqrt(np.mean((kf-km)**2))),1e-6);target=torch.tensor((table-km)/kr,dtype=torch.float32)
        mapper=CalendarMap(s);optimizer=torch.optim.Adam(mapper.parameters(),lr=c["fixed_hyperparameters"]["map_lr"])
        for _ in range(c["fixed_hyperparameters"]["map_steps"]):
            loss=(mapper(torch.arange(72))-target).square().mean();optimizer.zero_grad();loss.backward();optimizer.step()
        mapper.eval();head=Head(s);head.load_state_dict(teacher_states[s]["Delta"][chosen["Delta"]]);head.eval();d={}
        for split in ["fit","val"]:
            x=torch.tensor(p[split+"_x"],dtype=torch.float32);bp=torch.tensor(cache[f"{s}_{split}_FrozenTimeBase"],dtype=torch.float32);ix=p[split+"_future"]
            ft=np.concatenate([numeric_summary(x,bp).numpy(),raw[ix],common[ix]],-1);z39=torch.tensor((ft-np.array(pp["mean"]))/np.array(pp["sd"]),dtype=torch.float32);ids=torch.tensor(cid[ix],dtype=torch.long)
            with torch.no_grad():
                exact=forward(head,torch.cat([z39,target[ids]],-1),bp).numpy();expected=tp[f"Metro_{s}_{split}_Delta"];err=float(abs(exact-expected).max());assert err<2e-6
                z=torch.cat([z39,mapper(ids)],-1);phi=head.net[1](head.net[0](z));initial=(bp.double()+center(head.net[2](phi).double())).numpy()
            bridges.append(dict(seed=s,split=split,max_error=err));d[split]=dict(phi=phi,bp=bp,x=x,initial=initial,z39=z39,ids=ids)
        # Frozen47 rule: fit16 final weights to other-two teacher consensus,alpha1.
        others=np.mean([tp[f"Metro_{z}_fit_Delta"] for z in c["seeds"] if z!=s],0);dv=ridge(C(d["fit"]["phi"].double().numpy()),C(others-d["fit"]["initial"]),1)
        oldw=head.net[2].weight.detach().numpy().reshape(-1);head.net[2].weight.data.copy_(torch.tensor((oldw+dv).astype(np.float32)).reshape(1,16))
        for split in ["fit","val"]:
            with torch.no_grad():d[split]["s47"]=(d[split]["bp"].double()+center(head.net[2](d[split]["phi"]).double())).numpy()
        if ctxstats is None:
            v=calctx(d["val"]["x"])[cal];ctxstats=dict(mean=v.mean(0).tolist(),sd=np.maximum(v.std(0),1e-6).tolist())
        for split in ["fit","val"]:d[split]["q"]=calctx(d[split]["x"],ctxstats);d[split]["D"]=design(d[split]["phi"],d[split]["q"])
        data[s]=d;students[s]=dict(mapper=copy.deepcopy(mapper.state_dict()),head=copy.deepcopy(head.state_dict()))
        bases[s]=torch.load(src/f"FrozenTimeBase_seed{s}.pt",map_location="cpu",weights_only=True)
        maprows.append(dict(seed=s,steps=200,parameters=216,map_mse=float((mapper.codes.weight-target).square().mean()),readout47_update_norm=float(np.linalg.norm(dv))))
    arr={f:np.stack([tp[f"Metro_{s}_val_{arm}"] for s in c["seeds"]]) for f,arm in dict(Qwen="Delta",Fourier="Fourier",Shuffle="FullShuffle").items()}
    for f in arr:arr[f]=arr[f]-arr[f].mean(2,keepdims=True)
    y=p["val_y"];X=arr["Qwen"][:,cal].reshape(3,-1).T;w0,wd=simplex_weights(X,C(y[cal]).reshape(-1));globalT=np.tensordot(w0,arr["Qwen"],axes=(0,0))
    for s in c["seeds"]:
        d=data[s]["val"];tau=float(np.mean(d["D"][cal]**2));dv=ridge(d["D"][cal],(globalT-C(d["s47"]))[cal],10,tau).reshape(3,16)
        students[s]["delta48"]=torch.tensor(dv.astype(np.float32))
        with torch.no_grad():d["s48"]=d["s47"]+center(torch.einsum("bhi,ji,bj->bh",d["phi"],students[s]["delta48"],d["q"]).double().unsqueeze(-1)).numpy()
    ref=np.mean([C(data[s]["val"]["s48"])[cal,:,0] for s in c["seeds"]],0);res={f:a[:,cal][:,:,:,0]-ref[None] for f,a in arr.items()};B,diag=orthogonal_modes(res,[4]);basis=B[4];assert basis.shape[1]==4;P=basis@basis.T
    first=data[c["seeds"][0]]["val"];mu,bw=prototypes(first["q"].numpy()[cal,1:]);pi=partition(first["q"].numpy()[cal,1:],mu,bw)
    wc=np.stack([simplex_fit(X,C(y[cal]).reshape(-1),np.repeat(pi[:,k],96),w0,max(wd["diversity_V"],1e-12)) for k in range(3)])
    weights=partition(first["q"].numpy()[:,1:],mu,bw)@wc;conditionalT=np.einsum("nm,mnhc->nhc",weights,arr["Qwen"])
    compArr=[np.einsum("ht,mntc->mnhc",P,arr["Qwen"][:,cal])];compArr.append(arr["Qwen"][:,cal]-compArr[0]);compY=[Papply(P,C(y[cal])),C(y[cal])-Papply(P,C(y[cal]))];weights51=[]
    for a,yy in zip(compArr,compY):
        V=max(float(np.mean((a-a.mean(0,keepdims=True))**2)),1e-12);xx=a.reshape(3,-1).T
        weights51.append(np.stack([simplex_fit(xx,yy.reshape(-1),np.repeat(pi[:,k],96),wc[k],V) for k in range(3)]))
    weights51=np.array(weights51);ww=partition(first["q"].numpy()[:,1:],mu,bw);tP=np.einsum("nm,mnhc->nhc",ww@weights51[0],arr["Qwen"]);tR=np.einsum("nm,mnhc->nhc",ww@weights51[1],arr["Qwen"]);teacher51=Papply(P,tP)+tR-Papply(P,tR)
    stages=[];preds={};rows=[]
    for s in c["seeds"]:
        d=data[s]["val"];D=d["D"];A=Papply(P,D);Z=np.concatenate([A,D-A],2);tau=float(np.mean(D[cal]**2))
        theta49=ridge(Z[cal],(globalT-C(d["s48"]))[cal],10,tau).reshape(2,3,16).astype(np.float32);s49=d["s48"]+corr(d["phi"],d["q"],theta49,P)
        delta50=coupled_solve(Z[cal].reshape(-1,96),(conditionalT-C(s49))[cal].reshape(-1),theta49,10,tau,0)
        theta50=(theta49.astype(np.float64)+delta50).astype(np.float32);s50=d["s48"]+corr(d["phi"],d["q"],theta50,P)
        theta51=np.zeros((2,2,3,16),np.float32);theta51[:,0]=theta50;Dh=design(d["phi"],d["q"],True);A=Papply(P,Dh);Zh=np.concatenate([A,Dh-A],2);delta51=ridge(Zh[cal],(teacher51-C(s50))[cal],1,tau).reshape(2,2,3,16);theta51=(theta51.astype(np.float64)+delta51).astype(np.float32)
        students[s]["mode_horizon"]=torch.tensor(theta51);students[s]["theta50"]=torch.tensor(theta50)
        valpred=d["s48"]+corr(d["phi"],d["q"],theta51,P)
        for label,mask in [("calibration",cal),("selection",select)]:
            rows.append(dict(seed=s,split=label,model="Fixed51",**scores(valpred[mask],y[mask])));preds[f"{s}_{label}_y"]=y[mask];preds[f"{s}_{label}_Fixed51"]=valpred[mask]
        stages.append(dict(seed=s,theta49_norm=float(np.linalg.norm(theta49)),theta50_norm=float(np.linalg.norm(theta50)),theta51_norm=float(np.linalg.norm(theta51))))
    torch.save(dict(base=bases,students=students,context_statistics=ctxstats,projection_basis=torch.tensor(basis),normalizers={s:norm[str(s)]["Fourier"] for s in c["seeds"]}),r/"inference_bundle.pt")
    save(r/"teacher_state.json",dict(global_weights=w0.tolist(),conditional_weights=wc.tolist(),component_weights=weights51.tolist(),prototypes=mu.tolist(),bandwidth=bw,projection_diagnostics=diag));save(r/"compile_diagnostics.json",dict(calendar_maps=maprows,teacher_bridge=bridges,stages=stages));save(r/"internal_metrics.json",rows);np.savez_compressed(r/"internal_predictions.npz",**preds)
    # Native baseline gets all historical training windows through the same calibration end;
    # no new confirmation values are parsed here.
    trainids=list(zip(p["full_train_origin"],p["full_train_x"],p["full_train_y"]))+list(zip(p["val_origin"][cal],p["val_x"][cal],p["val_y"][cal]))
    trainids=sorted({int(o):(x,yy) for o,x,yy in trainids}.items())
    def mark(orig):
        ix=orig[:,None]+np.arange(-96,0);return raw[ix].astype(np.float32)
    oo=np.array([o for o,xy in trainids]);xx=np.stack([xy[0] for o,xy in trainids]);yy=np.stack([xy[1] for o,xy in trainids]);vs=p["val_origin"][select]
    np.savez_compressed(r/"baseline_data.npz",train_x=xx,train_y=yy,train_mark=mark(oo),train_origin=oo,val_x=p["val_x"][select],val_y=p["val_y"][select],val_mark=mark(vs),val_origin=vs,**{f"val_bp_{seed}":data[seed]["val"]["bp"].numpy()[select] for seed in c["seeds"]})
    save(r/"model_lock.json",dict(model_sha=sha(r/"inference_bundle.pt"),teacher_sha=sha(r/"teacher_state.json"),config_sha=sha(r/"config.json"),before_confirmation=True,no_51_model_selection=True,source51_hyperparameters_frozen=True))
    for a in assets:assert sha(a["path"])==a["sha"],a["path"]
    check22(c);save(r/"fit_status.json",dict(complete=True,seconds=time.monotonic()-start,map_fits=3,map_steps=600,new_teacher_fits=0,NoKD_Fourier_Shuffle_fits=0,LLM_calls=0,Metro51_instances=3,new_confirmation_evaluated=False,parameters=7731,calibration_windows=int(cal.sum()),selection_windows=int(select.sum())))
    print("METRO51_FITTED",rows,flush=True)

def infer(bundle,sc,pack,raw,common,kind,h,seed,code_mode="original",fit_ids=None):
    m=CalendarTimeBase(sc,seed,np.zeros((168,168),np.float32));m.load_state_dict(bundle["base"][seed]);x=torch.tensor(pack["x"],dtype=torch.float32);bp=torch.tensor(base_predict(m,dict(x=x,ids=torch.tensor(pack["ids"],dtype=torch.long))),dtype=torch.float32)
    ix=pack["future"];pp=bundle["normalizers"][seed];ft=np.concatenate([numeric_summary(x,bp).numpy(),raw[ix],common[ix]],-1);z39=torch.tensor((ft-np.array(pp["mean"]))/np.array(pp["sd"]),dtype=torch.float32);st=bundle["students"][seed];codes=st["mapper"]["codes.weight"].clone();diag={}
    if code_mode!="original":
        prior=codes.numpy();ff=prior[fit_ids];mean=ff.mean((0,1));rms=max(float(np.sqrt(np.mean((ff-mean)**2))),1e-12)
        if code_mode=="constant":modified=np.tile(mean,(72,1))
        else:
            perm=np.concatenate([np.arange(24)*0+i*24+np.random.default_rng(2024+i).permutation(24) for i in range(3)]);modified=prior[perm];mm=modified[fit_ids].mean((0,1));rr=max(float(np.sqrt(np.mean((modified[fit_ids]-mm)**2))),1e-12);modified=(modified-mm)/rr*rms+mean
        codes=torch.tensor(modified,dtype=torch.float32);diag=dict(mode=code_mode,original_fit_rms=rms,new_fit_rms=float(np.sqrt(np.mean((modified[fit_ids]-modified[fit_ids].mean((0,1)))**2))))
    ids=torch.tensor((kind*24+h)[ix],dtype=torch.long);z=torch.cat([z39,codes[ids]],-1);head=Head(seed);head.load_state_dict(st["head"]);q=calctx(x,bundle["context_statistics"]);P=bundle["projection_basis"].numpy();P=P@P.T
    with torch.no_grad():phi=head.net[1](head.net[0](z));s47=(bp.double()+center(head.net[2](phi).double())).numpy();s48=s47+center(torch.einsum("bhi,ji,bj->bh",phi,st["delta48"],q).double().unsqueeze(-1)).numpy()
    result=s48+corr(phi,q,st["mode_horizon"].numpy(),P)
    return result,bp.numpy(),diag,z39,bp

def evaluate(c):
    r=Path(c["run"]);native=Path(c["native_baseline_run"]);assert (native/"model_lock.json").exists(),"Do not inspect confirmation until native baseline choices are locked."
    lock=read(r/"model_lock.json");assert sha(r/"inference_bundle.pt")==lock["model_sha"];assert sha(r/"teacher_state.json")==lock["teacher_sha"]
    src,sc,p,cache,norm,grid,raw,common,kind,h=load_common(c);bundle=torch.load(r/"inference_bundle.pt",map_location="cpu",weights_only=True)
    rawf=pd.read_csv(sc["data"],keep_default_na=False);rawf["ts"]=pd.to_datetime(rawf.date_time);vals=rawf.groupby("ts").traffic_volume.mean().reindex(grid).to_numpy(float);meta=read(src/"data_metadata.json");v=(vals-meta["mean"])/meta["sd"]
    a,b=c["confirmation"];lo=grid.get_loc(pd.Timestamp(a));hi=int(np.searchsorted(grid.values,np.datetime64(b)));orig=np.array([o for o in range(lo,hi-96+1,24) if np.isfinite(v[o-96:o+96]).all() and np.isfinite(raw[o:o+96]).all()])
    assert len(orig)>0;past=orig[:,None]+np.arange(-96,0);future=orig[:,None]+np.arange(96);newpack=dict(x=v[past,None].astype(np.float32),y=v[future,None],ids=grid.dayofweek.to_numpy()[future]*24+h[future],future=future,origin=orig)
    np.savez_compressed(r/"confirmation_data.npz",**newpack,mark=raw[past].astype(np.float32))
    fixed22=read(c["frozen22"]);base22=Path(fixed22["run"]);s22=torch.load(base22/"selected_students.pt",map_location="cpu",weights_only=True)["Metro"];p22=np.load(base22/"predictions.npz");ks=np.load(src/"knowledge.npz")
    allrows=[];preds={};drows=[];maxmean=0.;oldreplay=0.
    for split,pack in [("known2017",dict(x=p["confirm_x"],y=p["confirm_y"],ids=p["confirm_ids"],future=p["confirm_future"],origin=p["confirm_origin"])),("confirmation2018",newpack)]:
        ix=pack["future"];fitids=(kind*24+h)[p["fit_future"]]
        for s in c["seeds"]:
            prediction,bp,_,z39,bpt=infer(bundle,sc,pack,raw,common,kind,h,s);out={"Fixed51":prediction,"FrozenNumericalBase":bp};pp=norm[str(s)]["Fourier"];kfit=ks["Fourier"][p["fit_future"]];km=kfit.mean((0,1));kr=max(float(np.sqrt(np.mean((kfit-km)**2))),1e-6);zz=torch.cat([z39,torch.tensor((ks["Fourier"][ix]-km)/kr,dtype=torch.float32)],-1)
            for arm in ["NoKD","FourierKD","ShuffleKD"]:
                head=Head(s);head.load_state_dict(s22[s][arm])
                with torch.no_grad():pr=forward(head,zz,bpt).numpy()
                if split=="known2017":
                    saved=p22[f"Metro_{s}_eval_{arm}"];ee=float(abs(pr-saved).max());assert ee<3e-6;oldreplay=max(oldreplay,ee);pr=saved
                out["Fixed22_"+arm]=pr
            for mode in ["permutation","constant"]:
                pr,_,diag,_,_=infer(bundle,sc,pack,raw,common,kind,h,s,mode,fitids);out["InferenceCode_"+mode]=pr;drows.append(dict(seed=s,split=split,**diag))
            preds[f"{s}_{split}_y"]=pack["y"];preds[f"{s}_{split}_origin"]=pack["origin"]
            for name,pr in out.items():
                preds[f"{s}_{split}_{name}"]=pr;row=dict(seed=s,split=split,model=name,**scores(pr,pack["y"]));allrows.append(row)
                if name=="Fixed51":maxmean=max(maxmean,float(abs((pr-bp).mean(1)).max()))
    np.savez_compressed(r/"predictions.npz",**preds);save(r/"metrics.json",allrows);save(r/"intervention_diagnostics.json",drows)
    names=list(out);summary={sp:{name:{k:float(np.mean([v[k] for v in allrows if v["split"]==sp and v["model"]==name])) for k in ["mse","mae","level","shape"]} for name in names} for sp in ["known2017","confirmation2018"]};save(r/"summary.json",summary)
    groups=[]
    for i,(a,b) in enumerate(c["confirmation_quarters"]):
        mask=(grid[newpack["origin"]]>=a)&(grid[newpack["origin"]]<b)
        if not mask.any():continue
        for s in c["seeds"]:
            for name in names:groups.append(dict(quarter=i+1,seed=s,model=name,windows=int(mask.sum()),**scores(preds[f"{s}_confirmation2018_{name}"][mask],newpack["y"][mask])))
    save(r/"quarters.json",groups)
    save(r/"data_confirmation_metadata.json",dict(windows=len(orig),raw_file_sha=sha(sc["data"]),period=c["confirmation"],first_origin=str(grid[orig[0]]),last_origin=str(grid[orig[-1]]),normalization_mean=meta["mean"],normalization_sd=meta["sd"],unique_future_hours=len(np.unique(future)),strict_complete_windows=True))
    for a in read(r/"inputs.json"):assert sha(a["path"])==a["sha"],a["path"]
    check22(c);assert sha(r/"inference_bundle.pt")==lock["model_sha"];save(r/"evaluation_status.json",dict(complete=True,all_models_locked_before_eval=True,NoKD_Fourier_Shuffle_fits=0,old22_weights_and_results_unchanged=True,old22_replay_error=oldreplay,mean_preservation_error=maxmean,new_time_period=True,new_dataset=False,all_negative_cases_saved=True))
    print("METRO_CONFIRMATION",summary["confirmation2018"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--phase",choices=["fit","evaluate"],required=True);a=p.parse_args();c=read(a.config)
    fit(c,a.config) if a.phase=="fit" else evaluate(c)
