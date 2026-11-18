"""Bounded energy-switch-only optimization; frozen KD experts and controls."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,csv,json,time,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from exp.exp_chronogeom_event_knowledge19 import read,save,sha,git,Head,forward,scores
from exp.exp_chronogeom_downstream import CalendarTimeBase,predict as base_predict
from exp.exp_chronogeom_residual03 import numeric_summary
from layers.ChronoGeomEnergySwitch import EnergySwitch

def center(v):return v-v.mean(-1,keepdims=True)
def project(v,channels):
    if not channels:return center(v)[:,None,:]
    z=np.fft.rfft(v,axis=-1,norm="ortho");masks=np.zeros((3,49));masks[0,4]=1;masks[1,8]=1;masks[2,1:]=1;masks[2,[4,8]]=0
    return np.fft.irfft(z[:,None,:]*masks[None,:,:],n=96,axis=-1,norm="ortho")

def calendar(origin,future,t0):
    wd=pd.date_range(t0,periods=int(max(future.max()+1,origin.max()+96)),freq="h").dayofweek.to_numpy()
    return wd[origin[:,None]+np.arange(-96,0)]>=5,wd[future]>=5

def assemble(x,y,origin,future,t0,ref,cand,controls):
    pw,fw=calendar(origin,future,t0)
    return dict(x=np.asarray(x[...,0],dtype=np.float64),y=np.asarray(y[...,0],dtype=np.float64),origin=origin,future=future,t0=pd.Timestamp(t0),reference=np.stack(ref)[...,0],candidate=np.stack(cand)[...,0],week=fw.mean(1)-pw.mean(1),controls=controls)

def val_data(c,name):
    if name=="Metro":
        root=Path(c["metro"]);a=np.load(root/"Metro/data.npz");p=np.load(root/"predictions.npz");tc=read(Path(c["metro_teacher"])/"config.json")
        t0=pd.to_datetime(pd.read_csv(tc["data"],usecols=["date_time"]).date_time).min().normalize()
        return assemble(a["val_x"],a["val_y"],a["val_origin"],a["val_future"],t0,[p[f"Metro_{s}_val_QwenTaskKD23"] for s in c["seeds"]],[p[f"Metro_{s}_val_QwenRelationKD28"] for s in c["seeds"]],{})
    root=Path(c["tetouan"]);a=np.load(root/"fit_inputs.npz");cfg=read(root/"config.json");norm=read(root/"normalizers.json");states=torch.load(root/"head_checkpoints.pt",map_location="cpu",weights_only=True)
    grid=pd.date_range("2017-01-01","2018-01-01",freq="h",inclusive="left");h=grid.hour.to_numpy();wd=grid.dayofweek.to_numpy();week=(wd>=5).astype(int);ph=2*np.pi*(grid.dayofyear.to_numpy()-1)/365
    raw=np.stack([np.zeros(len(grid)),week,1-week,np.sin(2*np.pi*wd/7),np.cos(2*np.pi*wd/7),np.sin(ph),np.cos(ph)],1)
    hh=np.stack([fn(2*np.pi*k*h/24) for k in range(1,5) for fn in [np.sin,np.cos]],1);common=np.concatenate([hh*(week==k)[:,None] for k in range(3)],1)
    K=np.stack([np.sin(10*np.pi*h/24),np.cos(10*np.pi*h/24),np.cos(12*np.pi*h/24)],1);device=torch.device("cuda:0");p=dict(x=torch.tensor(a["val_x"],device=device),ids=torch.tensor(a["val_ids"],device=device))
    ref=[];cand=[]
    for seed in c["seeds"]:
        base=CalendarTimeBase(cfg,seed,np.zeros((168,168),np.float32)).to(device);base.load_state_dict(torch.load(root/f"FrozenTimeBase_{seed}.pt",map_location="cpu",weights_only=True));base.eval()
        b=torch.tensor(base_predict(base,p),device=device);num=numeric_summary(p["x"],b).cpu().numpy();ix=a["val_future"];n=norm[str(seed)]["Fourier"];ft=np.concatenate([num,raw[ix],common[ix]],-1)
        z=torch.tensor(np.concatenate([(ft-n["mean"])/n["sd"],(K[ix]-n["km"])/n["ks"]],-1),dtype=torch.float32)
        for arm,dest in [("QwenKD23",ref),("QwenRelation",cand)]:
            head=Head(seed).to(device);head.load_state_dict(states[seed][arm][500]);head.eval()
            with torch.no_grad():dest.append(forward(head,z.to(device),b).cpu().numpy())
    return assemble(a["val_x"],a["val_y"],a["val_origin"],a["val_future"],"2017-01-01",ref,cand,{})

def eval_data(c,name):
    if name=="Metro":
        root=Path(c["metro"]);a=np.load(root/"Metro/data.npz");p=np.load(root/"predictions.npz");old=np.load(Path(c["metro_oldouter"])/"predictions.npz");tc=read(Path(c["metro_teacher"])/"config.json")
        t0=pd.to_datetime(pd.read_csv(tc["data"],usecols=["date_time"]).date_time).min().normalize()
        controls={arm:np.stack([old[f"{s}_{arm}"][...,0] for s in c["seeds"]]) for arm in ["OriginalTimeBase","FrozenTimeBase","NoKD","FourierKD","HistoricalKD","ShuffleKD"]}
        return assemble(a["confirm_x"],a["confirm_y"],a["confirm_origin"],a["confirm_future"],t0,[p[f"Metro_{s}_eval_QwenTaskKD23"] for s in c["seeds"]],[p[f"Metro_{s}_eval_QwenRelationKD28"] for s in c["seeds"]],controls)
    root=Path(c["tetouan"]);a=np.load(root/"test_labels.npz");p=np.load(root/"test_predictions.npz")
    controls={arm:np.stack([p[f"{s}_{arm}"][...,0] for s in c["seeds"]]) for arm in ["OriginalTimeBase","FrozenTimeBase","NoKD","FourierKD","NoKDEnergy","FourierEnergy"]}
    return assemble(a["x"],a["y"],a["origin"],a["future"],"2017-01-01",[p[f"{s}_QwenKD23"] for s in c["seeds"]],[p[f"{s}_QwenRelation"] for s in c["seeds"]],controls)

def observations(data,scope):
    S,N,H=data["reference"].shape
    b=data["reference"].reshape(-1,H);cand=data["candidate"].reshape(-1,H);x=np.tile(data["x"],(S,1));wk=np.tile(data["week"],S)
    d=project(center(cand-b),scope=="unit_channels")
    repeated=np.tile(x[:,-24:],(1,4));proxy=project(center(repeated-b),scope=="unit_channels");hist=project(center(repeated),scope=="unit_channels")
    B=(d*d).mean(-1);proxycross=(proxy*d).mean(-1);E=(center(x)**2).mean(-1)[:,None].clip(1e-20)
    if scope=="raw_global":
        features=np.stack([B,proxycross,np.broadcast_to((center(x[:,-24:])**2).mean(-1)[:,None],B.shape),np.broadcast_to(wk[:,None],B.shape)],-1)
    else:
        proxyenergy=(proxy*proxy).mean(-1);cosine=proxycross/np.maximum(np.sqrt(B*proxyenergy),1e-8*E)
        features=np.stack([B/E,cosine,(hist*hist).mean(-1)/E,np.broadcast_to(wk[:,None],B.shape)],-1)
    assert np.isfinite(features).all()
    return dict(x=features,d=d,B=B,S=S,N=N,b=b)

def fit_gate(data,obs,mask,recipe):
    ix=np.tile(mask,obs["S"]);f=obs["x"][ix].reshape(-1,4);B=obs["B"][ix].reshape(-1)
    e=np.tile(data["y"],(obs["S"],1))[ix]-obs["b"][ix];A=(e[:,None,:]*obs["d"][ix]).mean(-1).reshape(-1)
    mu=f.mean(0);sd=f.std(0).clip(1e-8);X=np.c_[(f-mu)/sd,np.ones(len(f))]
    if recipe["fit"]=="A_OLS":coef=np.linalg.pinv(X)@A
    else:
        G=(X.T*B)@X/len(B);rhs=X.T@A/len(B);penalty=recipe["ridge"]*B.mean()*np.diag([1,1,1,1,0]);coef=np.linalg.pinv(G+penalty,rcond=1e-10)@rhs
    return dict(recipe=recipe,mean=mu.tolist(),sd=sd.tolist(),coefficients=coef.tolist(),calibration_origins=int(mask.sum()),rank=int(np.linalg.matrix_rank(X)),condition=float(np.linalg.cond(X)),parameters=5)

def infer(model,obs):
    r=model["recipe"];X=(obs["x"]-model["mean"])/model["sd"];raw=X@np.array(model["coefficients"][:4])+model["coefficients"][4]
    if r["fit"]=="A_OLS":raw=raw/obs["B"].clip(1e-12)
    a=np.clip(raw,-1 if r["range"]=="signed" else 0,1);a=np.where(obs["B"]>1e-12,a,0)
    q=obs["b"]+(a[:,:,None]*obs["d"]).sum(1)
    return q.reshape(obs["S"],obs["N"],96),a.reshape(obs["S"],obs["N"],-1)

def old_model(c,name):
    state=torch.load(Path(c["metro_oldgate"])/"switch.pt" if name=="Metro" else Path(c["tetouan"])/"QwenEnergy_switch.pt",map_location="cpu",weights_only=True)
    return dict(recipe=dict(id="Old5",scope="raw_global",fit="A_OLS",range="positive",ridge=0),mean=state["feature_mean"].tolist(),sd=state["feature_scale"].tolist(),coefficients=np.r_[state["alignment.weight"].numpy().ravel(),state["alignment.bias"].numpy().ravel()].tolist(),parameters=5)

def mask_dates(data,cut,end=None):
    dates=data["t0"]+pd.to_timedelta(data["origin"],unit="h")
    if end is None:return np.array(dates+pd.Timedelta(hours=96)<=pd.Timestamp(cut))
    return np.array((dates>=pd.Timestamp(cut))&(dates+pd.Timedelta(hours=96)<=pd.Timestamp(end)))

def run(c,config):
    start=time.monotonic();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.04);r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False)
    assets=read(Path(c["tetouan"])/"old_assets.json")+read(Path(c["metro_oldouter"])/"input_assets.json")
    for root,files in [(Path(c["tetouan"]),["test_predictions.npz","test_labels.npz","head_checkpoints.pt","metrics.json","model_lock.json","QwenEnergy_switch.pt"]),(Path(c["metro"]),["predictions.npz","selected_qwen_students.pt"])]:
        assets += [dict(path=str(root/f),sha=sha(root/f)) for f in files]
    for a in assets:assert sha(a["path"])==a["sha"]
    save(r/"config.json",c);save(r/"input_assets.json",assets);save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),numpy=np.__version__,torch=torch.__version__,LLM_calls=0,expert_updates=0,control_updates=0,device="CPU FP64 gate calibration"))
    recipes=[dict(id="Reference23",scope="raw_global",fit="risk",range="positive",ridge=0,zero=True)]
    for rg in c["ranges"]:recipes.append(dict(id="A_OLS_"+rg,scope="raw_global",fit="A_OLS",range=rg,ridge=0))
    for scope in c["scopes"]:
        for rg in c["ranges"]:
            for ridge in c["ridges"]:recipes.append(dict(id=f"{scope}_{rg}_{ridge}",scope=scope,fit="risk",range=rg,ridge=ridge))
    assert len(recipes)==21
    val={name:val_data(c,name) for name in ["Metro","Tetouan"]};obs={name:{scope:observations(d,scope) for scope in c["scopes"]} for name,d in val.items()}
    rows=[];foldmeta=[]
    for name,d in val.items():
        for fold,(cut,end) in enumerate(c["folds"]):
            tr=mask_dates(d,cut);va=mask_dates(d,cut,end);assert tr.sum()>=3 and va.sum()>=3
            assert not np.intersect1d(d["future"][tr].ravel(),d["future"][va].ravel()).size
            foldmeta.append(dict(domain=name,fold=fold,train=int(tr.sum()),validation=int(va.sum()),cut=cut,end=end))
            for rc in recipes:
                if rc.get("zero"):q=d["reference"]
                else:q,_=infer(fit_gate(d,obs[name][rc["scope"]],tr,rc),obs[name][rc["scope"]])
                ratios=[]
                for i,seed in enumerate(c["seeds"]):
                    met=scores(q[i,va,:,None],d["y"][va,:,None]);ref=scores(d["reference"][i,va,:,None],d["y"][va,:,None])
                    v=.5*(met["mse"]/ref["mse"]+met["mae"]/ref["mae"]);ratios.append(v)
                    rows.append(dict(domain=name,fold=fold,seed=seed,recipe=rc["id"],criterion=v,**met))
            print("INNER",name,fold,int(tr.sum()),int(va.sum()),flush=True)
    criteria={rc["id"]:float(np.mean([x["criterion"] for x in rows if x["recipe"]==rc["id"]])) for rc in recipes}
    chosen=min(recipes,key=lambda x:(criteria[x["id"]],recipes.index(x)));lam=chosen["ridge"]
    ablations=[rc for rc in recipes if not rc.get("zero") and (rc["fit"]=="A_OLS" or rc["ridge"]==lam)]
    save(r/"inner_rows.json",rows);save(r/"inner_criteria.json",criteria);save(r/"folds.json",foldmeta)
    save(r/"selection.json",dict(recipe=chosen,criterion=criteria[chosen["id"]],one_common_recipe_all_data_seeds=True,only_internal_labels=True,external_data_opened=False,ablations=[x["id"] for x in ablations]))
    models={};finalmask={}
    for name,d in val.items():
        tr=mask_dates(d,c["final_cal_end"]);finalmask[name]=int(tr.sum());assert tr.sum()==(23 if name=="Metro" else 39)
        models[name]={rc["id"]:fit_gate(d,obs[name][rc["scope"]],tr,rc) for rc in ablations}
        models[name]["Old5"]=old_model(c,name)
        refitted=fit_gate(d,obs[name]["raw_global"],tr,next(x for x in recipes if x["id"]=="A_OLS_positive"))
        assert max(abs(np.array(refitted["coefficients"])-models[name]["Old5"]["coefficients"]))<1e-10
        if chosen.get("zero"):models[name]["Selected"]=dict(recipe=chosen,coefficients=[0]*5,mean=[0]*4,sd=[1]*4,parameters=0)
        else:models[name]["Selected"]=models[name][chosen["id"]]
        assert models[name]["Old5"]["parameters"]==5
    save(r/"models.json",models);lock=sha(r/"models.json");save(r/"model_lock.json",dict(models_sha=lock,selection_sha=sha(r/"selection.json"),before_external_data_opened=True,final_calibration_origins=finalmask,additional_internal_selection_budget=True))
    print("SELECTED",chosen,criteria[chosen["id"]],flush=True)
    # All selection and final gate weights fixed above. No post-evaluation modifications.
    outer={name:eval_data(c,name) for name in val};allpred={};allgates={};oldreplay=0.;maxmean=0.;maxidentity=0.
    for name,d in outer.items():
        ob={scope:observations(d,scope) for scope in c["scopes"]};N=len(d["origin"])
        local={"QwenKD23":d["reference"],"RelationKD":d["candidate"],**d["controls"]}
        for arm,model in models[name].items():
            scope=model["recipe"]["scope"];q,a=infer(model,ob[scope]);local[arm]=q;allgates[f"{name}_{arm}"]=a
            maxmean=max(maxmean,float(abs((q-d["reference"]).mean(-1)).max()))
            o=ob[scope];actual=((q-d["y"])**2).mean(-1)-((d["reference"]-d["y"])**2).mean(-1)
            A=((np.tile(d["y"],(3,1))-o["b"])[:,None,:]*o["d"]).mean(-1).reshape(3,N,-1);B=o["B"].reshape(3,N,-1)
            theory=(a*a*B-2*a*A).sum(-1);maxidentity=max(maxidentity,float(abs(actual-theory).max()))
        source=np.load(Path(c["metro_oldouter"])/"predictions.npz" if name=="Metro" else Path(c["tetouan"])/"test_predictions.npz")
        key="EnergySwitch29_frozen" if name=="Metro" else "QwenEnergy"
        for i,seed in enumerate(c["seeds"]):oldreplay=max(oldreplay,float(abs(local["Old5"][i]-source[f"{seed}_{key}"][...,0]).max()))
        allpred.update({f"{name}_{s}_{arm}":q[i] for arm,q in local.items() for i,s in enumerate(c["seeds"])})
    np.savez_compressed(r/"predictions.npz",**allpred);np.savez_compressed(r/"gates.npz",**allgates)
    save(r/"prediction_lock.json",dict(model_sha=lock,prediction_sha=sha(r/"predictions.npz"),before_metrics=True))
    metrics=[];mis=[];window=[];summaries={};intervals=[];rng=np.random.default_rng(2024);labels={}
    for name,d in outer.items():
        N=len(d["origin"]);labels[name]=d["y"];labels[name+"_origin"]=d["origin"];labels[name+"_t0"]=np.array(str(d["t0"]))
        arms=[k.removeprefix(f"{name}_{c['seeds'][0]}_") for k in allpred if k.startswith(f"{name}_{c['seeds'][0]}_")]
        for seed in c["seeds"]:
            for arm in arms:metrics.append(dict(domain=name,seed=seed,arm=arm,**scores(allpred[f"{name}_{seed}_{arm}"][...,None],d["y"][...,None])))
            p=allpred[f"{name}_{seed}_Selected"];aa=allgates[f"{name}_Selected"][c["seeds"].index(seed)]
            for ref in ["QwenKD23","Old5"]:
                b=allpred[f"{name}_{seed}_{ref}"]
                for metric in ["mse","mae"]:
                    e=p-d["y"];eb=b-d["y"];fn=lambda v:(v*v if metric=="mse" else abs(v)).mean(-1);gain=fn(eb)-fn(e)
                    mis.append(dict(domain=name,seed=seed,reference=ref,metric=metric,windows=N,helpful=int((gain>1e-14).sum()),harmful=int((gain< -1e-14).sum()),unchanged=int((abs(gain)<=1e-14).sum()),positive_gain_mean=float(np.maximum(gain,0).mean()),miscorrection_loss_mean=float(np.maximum(-gain,0).mean()),net_gain_mean=float(gain.mean())))
            for j,o in enumerate(d["origin"]):
                b=allpred[f"{name}_{seed}_QwenKD23"][j];e=p[j]-d["y"][j];eb=b-d["y"][j]
                window.append(dict(domain=name,seed=seed,origin=str(d["t0"]+pd.Timedelta(hours=int(o))),weights=json.dumps(aa[j].tolist()),mse=float((e*e).mean()),mae=float(abs(e).mean()),mse_gain23=float((eb*eb-e*e).mean()),mae_gain23=float((abs(eb)-abs(e)).mean())))
        summaries[name]={arm:{k:float(np.mean([x[k] for x in metrics if x["domain"]==name and x["arm"]==arm])) for k in ["mse","mae","level","shape"]} for arm in arms}
        blocks=(d["origin"]-d["origin"].min())//336;uniq=np.unique(blocks)
        for ref in ["QwenKD23","Old5","NoKD","FourierKD"]:
            for metric in ["mse","mae"]:
                ee=np.stack([allpred[f"{name}_{s}_Selected"]-d["y"] for s in c["seeds"]]);eb=np.stack([allpred[f"{name}_{s}_{ref}"]-d["y"] for s in c["seeds"]])
                delta=((ee*ee-eb*eb) if metric=="mse" else (abs(ee)-abs(eb))).mean((0,2));bs=[]
                for _ in range(2000):
                    ix=np.concatenate([np.flatnonzero(blocks==j) for j in rng.choice(uniq,len(uniq),replace=True)]);bs.append(float(delta[ix].mean()))
                intervals.append(dict(domain=name,reference=ref,metric=metric,delta=float(delta.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),blocks=len(uniq),development_only=True))
    assert maxmean<1e-12 and maxidentity<1e-11 and oldreplay<1e-11 and sha(r/"models.json")==lock
    for a in assets:assert sha(a["path"])==a["sha"]
    save(r/"metrics.json",metrics);save(r/"miscorrections.json",mis);save(r/"summary.json",summaries);save(r/"intervals.json",intervals);np.savez_compressed(r/"labels.npz",**labels)
    with (r/"per_window.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(window[0]));w.writeheader();w.writerows(window)
    save(r/"verification.json",dict(maximum_mean_shift=maxmean,maximum_energy_identity_error=maxidentity,old_gate_replay_error=oldreplay,all_old_experts_controls_Capital_SHA_unchanged=True,parameters=5,shared_across_bands_and_seeds=True,selection_only_internal=True,outer_development_only=True,LLM_calls=0,expert_updates=0,control_updates=0))
    assert time.monotonic()-start<c["budget_seconds"]
    save(r/"status.json",dict(seconds=time.monotonic()-start,complete=True,selected=chosen,git=git(),external_evaluations=1,new_independent_confirmation=False))
    print("SUMMARY",summaries,flush=True);print("MISCORRECTIONS",mis,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args();run(read(a.config),a.config)

