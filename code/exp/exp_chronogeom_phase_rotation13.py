"""Phase-only distillation on a frozen TimeBase/calendar forecaster; no amplitude shortcut."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
import argparse,json,time,traceback,shutil
from pathlib import Path
import numpy as np
import torch
from exp import exp_chronogeom_temporal_shape_kd12 as old
s=old.s;u=s.u;rt=s.rt;csvsave=s.csvsave
TEACHER=dict(FourierKD="SeasonFourier32",QwenTaskKD="QwenTask",QwenWrongKD="QwenWrong",QwenShuffleKD="QwenTrainShuffle")
W=2*np.pi/24
C=np.cos(W*np.arange(96));S=np.sin(W*np.arange(96))

def base_phi(q):
    a=q["features"];state=q["ids"][:,0];annual=np.arctan2(a[:,20],a[:,21]);parts=[]
    for period in [24.,168.]:
        for k in range(1,5):parts.extend([np.sin(2*np.pi*k*state/period),np.cos(2*np.pi*k*state/period)])
    for k in range(1,9):parts.extend([np.sin(k*annual),np.cos(k*annual)])
    return np.c_[a,np.stack(parts,1)]

def base_fit(q,ii,c):
    _,energy,signal=s.components(q)
    return s.fit_strength(base_phi(q)[ii],energy[ii],signal[ii],c["base_regularizer"])

def base_predict(q,ii,m):
    alpha=s.strength(m,base_phi(q)[ii],[0,2])
    return q["base"][ii]+alpha[:,None]*(q["candidates"][ii,4]-q["base"][ii])

def crossfit_base(q,ii,c):
    out=np.zeros((len(ii),96));meta=[]
    for block,hold in enumerate(np.array_split(ii,c["base_crossfit_blocks"])):
        lo=int(q["origin"][hold].min());hi=int(q["origin"][hold].max()+96)
        fit=ii[(q["origin"][ii]+96<=lo)|(q["origin"][ii]>=hi)]
        assert len(fit)>=8 and not set(fit).intersection(hold)
        assert np.all((q["origin"][fit]+96<=lo)|(q["origin"][fit]>=hi))
        out[np.searchsorted(ii,hold)]=base_predict(q,hold,base_fit(q,fit,c))
        meta.append(dict(block=block,held=hold.tolist(),fit=fit.tolist(),target_interval=[lo,hi],target_overlap=False,
          offline_crossfit_uses_both_sides_within_fit_period=True))
    return out,meta

def pack(b,y):
    assert b.shape[1]==96
    mu=b.mean(1,keepdims=True);z=b-mu
    ac=2*np.mean(z*C,axis=1);bs=2*np.mean(z*S,axis=1)
    uu=ac[:,None]*C+bs[:,None]*S;vv=ac[:,None]*S-bs[:,None]*C
    energy=np.mean(uu*uu,1);amp=np.hypot(ac,bs)
    coords=np.c_[np.divide(ac,amp,out=np.zeros_like(ac),where=amp>1e-12),
                 np.divide(bs,amp,out=np.zeros_like(bs),where=amp>1e-12)]
    residue=z-uu
    target=y-mu-residue
    aa=np.mean(target*uu,1);bb=np.mean(target*vv,1)
    return dict(b=b,y=y,u=uu,v=vv,E=energy,A=aa,B=bb,coords=coords,base_mse=np.mean((b-y)**2,1),amp=amp,
                base_ac=ac,base_bs=bs)

def rotate(p,delta):
    return p["b"]+p["u"]*(np.cos(delta)[:,None]-1)+p["v"]*np.sin(delta)[:,None]

def mse_vector(p,delta):
    return p["base_mse"]+4*p["A"]*np.sin(delta/2)**2-2*p["B"]*np.sin(delta)

def predict(m,x,c):
    return c["delta_max"]*np.tanh(((x-m["mu"])/m["sd"])@m["coef"]+float(m["bias"]))

def fit_head(x,p,lam,c,teacher=None):
    mu=x.mean(0);sd=np.maximum(x.std(0),.001)
    xx=torch.tensor((x-mu)/sd,dtype=torch.float64)
    aa=torch.tensor(p["A"]);bb=torch.tensor(p["B"]);ee=torch.tensor(p["E"])
    ww=torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True)
    bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    target=None if teacher is None else torch.tensor(teacher,dtype=torch.float64)
    opt=torch.optim.Adam([ww,bias],lr=c["learning_rate"]);trace=[]
    for step in range(1,c["steps"]+1):
        opt.zero_grad();delta=c["delta_max"]*torch.tanh(xx@ww+bias)
        supervised=(4*aa*torch.sin(delta/2)**2-2*bb*torch.sin(delta)).mean()
        kd=torch.zeros((),dtype=torch.float64) if target is None else (4*ee*torch.sin((delta-target)/2)**2).mean()
        loss=supervised+lam*ww.square().sum()+c["kd_weight"]*kd
        loss.backward();opt.step()
        if step in [1,25,50,100]:trace.append(dict(step=step,supervised_without_constant=float(supervised.detach()),kd=float(kd.detach())))
    return dict(mu=mu,sd=sd,coef=ww.detach().numpy(),bias=bias.detach().numpy()),trace

def training_data(c,r):
    data={};hidden={};sources={}
    for seed in c["seeds"]:
        f=Path(c["reuse_train"])/f"training_seed{seed}.npz";b=np.load(f);sources[str(f)]=rt.sha(f)
        q={k:b[k] for k in b.files};q["features"]=old.annual_features(q["features"],q["origin"],c);data[seed,"train"]=q
        for arm in c["contexts"]:
            f=Path(c["reuse_phase_cache"])/"train_extraction"/f"hidden_{seed}_{arm}_train.npy"
            hidden[seed,arm,"train"]=np.load(f);sources[str(f)]=rt.sha(f)
        f=Path(c["reuse_phase_cache"])/f"teacher_s{seed}_SeasonFourier32.npz";sources[str(f)]=rt.sha(f)
    for f in [Path(__file__),Path(old.__file__),Path(s.__file__),Path(u.__file__),Path("docs/chronogeom/phase_rotation13_config.json")]:
        sources[str(f)]=rt.sha(f)
    rt.save(r/"source_train.json",dict(git=rt.git(),sources=sources,no_outer_files_opened=True))
    return data,hidden

def make_folds(c,data,h):
    folds={};metadata=[]
    for seed in c["seeds"]:
        q=data[seed,"train"];o=q["origin"]
        for begin,end in c["folds"]:
            lo=int((np.datetime64(begin,"h")-np.datetime64(c["clock_base"],"h"))/np.timedelta64(1,"h"))
            hi=int((np.datetime64(end,"h")-np.datetime64(c["clock_base"],"h"))/np.timedelta64(1,"h"))
            ii=np.where(o+96<=lo)[0];vi=np.where((o>=lo)&(o+96<=hi))[0]
            assert len(ii)>30 and len(vi)>=10 and (o[ii]+96).max()<=o[vi].min()
            train_b,cf=crossfit_base(q,ii,c);valid_b=base_predict(q,vi,base_fit(q,ii,c))
            dd=dict(data);dd[seed,"inner"]={k:v[vi] for k,v in q.items()};hh=dict(h)
            for arm in c["contexts"]:hh[seed,arm,"inner"]=h[seed,arm,"train"][vi]
            mat,_=old.matrices(c,dd,hh,seed,ii,dict(train=ii,inner=np.arange(len(vi))))
            pt=pack(train_b,q["y"][ii]);pv=pack(valid_b,q["y"][vi])
            for stage,p in [("train",pt),("inner",pv)]:
                for arm in mat[stage]:mat[stage][arm]=np.c_[mat[stage][arm],p["coords"]]
            folds[seed,begin]=dict(ii=ii,vi=vi,mat=mat,train=pt,valid=pv)
            metadata.append(dict(seed=seed,fold=begin,fit=len(ii),validation=len(vi),crossfit=cf))
    return folds,metadata

def select(c,r):
    r.mkdir(parents=True,exist_ok=False);rt.save(r/"config.json",c);shutil.copy2(__file__,r/Path(__file__).name)
    tick=time.monotonic();torch.set_num_threads(2)
    try:
        data,h=training_data(c,r);folds,meta=make_folds(c,data,h);rt.save(r/"crossfit_inner.json",meta)
        records=[]
        for (seed,name),f in folds.items():
            for arm in c["teacher_arms"]:
                for lam in c["lambdas"]:
                    m,_=fit_head(f["mat"]["train"][arm],f["train"],lam,c)
                    delta=predict(m,f["mat"]["inner"][arm],c)
                    score=float(mse_vector(f["valid"],delta).mean())
                    records.append(dict(seed=seed,fold=name,arm=arm,lam=lam,mse=score,ratio=score/float(f["valid"]["base_mse"].mean())))
        tc=old.choose(records);csvsave(r/"teacher_inner.csv",records);rt.save(r/"teacher_choice.json",tc)
        targets={}
        for key,f in folds.items():
            targets[key]={}
            for arm in c["teacher_arms"]:
                m,_=fit_head(f["mat"]["train"][arm],f["train"],tc[arm]["lam"],c)
                targets[key][arm]=predict(m,f["mat"]["train"][arm],c)
        records=[]
        for key,f in folds.items():
            seed,name=key
            for arm in c["student_arms"]:
                for lam in c["lambdas"]:
                    teacher=None if arm=="NoKD" else targets[key][TEACHER[arm]]
                    m,_=fit_head(f["mat"]["train"]["SeasonFourier32"],f["train"],lam,c,teacher)
                    delta=predict(m,f["mat"]["inner"]["SeasonFourier32"],c)
                    score=float(mse_vector(f["valid"],delta).mean())
                    records.append(dict(seed=seed,fold=name,arm=arm,lam=lam,mse=score,ratio=score/float(f["valid"]["base_mse"].mean())))
        sc=old.choose(records);csvsave(r/"student_inner.csv",records);rt.save(r/"student_choice.json",sc)
        fullmeta=[];curves=[];oracle=[]
        for seed in c["seeds"]:
            q=data[seed,"train"];ii=np.arange(len(q["origin"]));b,cf=crossfit_base(q,ii,c);p=pack(b,q["y"])
            mat,pca=old.matrices(c,data,h,seed,ii,dict(train=ii));mat=mat["train"]
            for arm in mat:mat[arm]=np.c_[mat[arm],p["coords"]]
            np.savez_compressed(r/f"phase_training_s{seed}.npz",b=b,y=q["y"],x=mat["SeasonFourier32"],origin=q["origin"])
            fullmeta.append(dict(seed=seed,crossfit=cf))
            baseline=base_fit(q,ii,c);ref=np.load(Path(c["reuse_phase_cache"])/f"teacher_s{seed}_SeasonFourier32.npz")
            assert max(float(abs(baseline[k]-ref[k]).max()) for k in baseline)<1e-10
            np.savez_compressed(r/f"frozen_base_s{seed}.npz",**baseline)
            for arm,(mu,w,info) in pca.items():np.savez_compressed(r/f"pca_s{seed}_{arm}.npz",mu=mu,w=w)
            teacher={}
            for arm in c["teacher_arms"]:
                m,trace=fit_head(mat[arm],p,tc[arm]["lam"],c);teacher[arm]=predict(m,mat[arm],c)
                np.savez_compressed(r/f"teacher_s{seed}_{arm}.npz",**m)
                curves.append(dict(seed=seed,arm=arm,role="teacher",trace=trace))
            for arm in c["student_arms"]:
                m,trace=fit_head(mat["SeasonFourier32"],p,sc[arm]["lam"],c,None if arm=="NoKD" else teacher[TEACHER[arm]])
                np.savez_compressed(r/f"student_s{seed}_{arm}.npz",**m);curves.append(dict(seed=seed,arm=arm,role="student",trace=trace))
                assert m["coef"].size+np.size(m["bias"])==c["trainable_parameters"]
            angle=np.arctan2(p["B"],p["A"]);valid=abs(angle)<=c["delta_max"]
            choices=np.stack([np.zeros(len(ii)),np.full(len(ii),-c["delta_max"]),np.full(len(ii),c["delta_max"]),np.where(valid,angle,0.)])
            best=np.min(np.stack([mse_vector(p,a) for a in choices]),axis=0)
            oracle.append(dict(seed=seed,n=len(ii),crossfit_base_mse=float(p["base_mse"].mean()),phase_only_oracle_mse=float(best.mean()),deployed=False,training_only=True))
        rt.save(r/"crossfit_full.json",fullmeta);rt.save(r/"curves.json",curves);rt.save(r/"training_oracle.json",oracle)
        src=rt.read(r/"source_train.json");assert all(rt.sha(p)==v for p,v in src["sources"].items())
        assert time.monotonic()-tick<c["budget"]["max_fit_seconds"]
        rt.save(r/"selection_lock.json",dict(seconds=time.monotonic()-tick,files={str(p):rt.sha(p) for p in r.iterdir() if p.is_file()},
            no_outer_files_opened=True,head_steps=c["steps"],parameters=c["trainable_parameters"],
            teacher_choice={a:v["lam"] for a,v in tc.items()},student_choice={a:v["lam"] for a,v in sc.items()}))
        print("SELECTED",json.dumps(dict(teacher={a:v["lam"] for a,v in tc.items()},student={a:v["lam"] for a,v in sc.items()})),flush=True)
    except BaseException:rt.save(r/"failure_select.json",dict(traceback=traceback.format_exc()));raise

def forecast_metrics(p,delta):
    pred=rotate(p,delta);err=pred-p["y"];level=err.mean(1)**2;shape=np.mean((err-err.mean(1,keepdims=True))**2,1)
    center=pred-pred.mean(1,keepdims=True);pc=2*np.mean(center*C,1);ps=2*np.mean(center*S,1)
    py=p["y"]-p["y"].mean(1,keepdims=True);yc=2*np.mean(py*C,1);ys=2*np.mean(py*S,1)
    phase=np.abs(np.angle(np.exp(1j*(np.arctan2(ps,pc)-np.arctan2(ys,yc)))))/W
    weight=np.sqrt(p["E"]*(yc*yc+ys*ys)/2)
    baseenergy=np.mean((p["b"]-p["b"].mean(1,keepdims=True))**2,1)
    residual_after=center-pc[:,None]*C-ps[:,None]*S
    residual_before=p["b"]-p["b"].mean(1,keepdims=True)-p["u"]
    checks=dict(mean=float(abs(pred.mean(1)-p["b"].mean(1)).max()),
       energy=float(abs(np.mean(center*center,1)-baseenergy).max()),
       amplitude=float(abs(np.hypot(pc,ps)-p["amp"]).max()),
       residue=float(abs(residual_after-residual_before).max()))
    assert max(checks.values())<1e-10
    return pred,dict(mse=np.mean(err*err,1),mae=np.mean(abs(err),1),level=level,shape=shape,phase=phase,weight=weight,
                    amplitude_error=abs(np.hypot(pc,ps)-np.hypot(yc,ys))),checks

def evaluate(c,r):
    lock=rt.read(r/"selection_lock.json");assert all(rt.sha(p)==v for p,v in lock["files"].items())
    tick=time.monotonic();previous=Path(c["reliability_run"]);pc=rt.read(previous/"config.json");cache=Path(pc["reuse_run"])
    sums={(v["seed"],v["stage"],v["row"]):v["summary"] for v in rt.read(previous/"shared_summaries.json")}
    sources={str(previous/"shared_summaries.json"):rt.sha(previous/"shared_summaries.json")};rows=[];cases=[];outputs={};maximum={k:0. for k in ["mean","energy","amplitude","residue"]}
    teacherrows=[]
    for seed in c["seeds"]:
        bm=np.load(r/f"frozen_base_s{seed}.npz");student={a:np.load(r/f"student_s{seed}_{a}.npz") for a in c["student_arms"]}
        for stage in c["stages"]:
            path=cache/f"seed{seed}_{stage}.npz";z=np.load(path);q={k:z[k] for k in z.files};sources[str(path)]=rt.sha(path)
            q["features"]=old.annual_features(np.stack([s.flatten(sums[seed,stage,j]) for j in range(len(q["origin"]))]),q["origin"],c)
            ii=np.arange(len(q["origin"]));b=base_predict(q,ii,bm);p=pack(b,q["y"]);x=np.c_[base_phi(q),p["coords"]]
            np.savez_compressed(r/f"deployment_s{seed}_{stage}.npz",x=x,b=b,u=p["u"],v=p["v"])
            angles=dict(FrozenBase=np.zeros(len(ii)),**{a:predict(m,x,c) for a,m in student.items()})
            for arm,delta in angles.items():
                pred,met,checks=forecast_metrics(p,delta)
                for k,v in checks.items():maximum[k]=max(maximum[k],v)
                outputs[f"s{seed}_{stage}_{arm}"]=pred;outputs[f"a{seed}_{stage}_{arm}"]=delta
                gain=p["base_mse"]-met["mse"]
                rows.append(dict(seed=seed,stage=stage,arm=arm,n=len(ii),**{k:float(met[k].mean()) for k in ["mse","mae","level","shape","amplitude_error"]},
                    phase_weighted_mae_hours=float(np.sum(met["weight"]*met["phase"])/max(float(met["weight"].sum()),1e-12)),
                    gross_gain=float(np.maximum(gain,0).mean()),gross_harm=float(np.maximum(-gain,0).mean()),
                    shift_mean_hours=float((delta/W).mean()),shift_sd_hours=float((delta/W).std())))
                for j in ii:cases.append(dict(seed=seed,stage=stage,arm=arm,origin=int(q["origin"][j]),delta=float(delta[j]),
                   shift_hours=float(delta[j]/W),mse=float(met["mse"][j]),mae=float(met["mae"][j]),level=float(met["level"][j]),shape=float(met["shape"][j]),
                   phase_error_hours=float(met["phase"][j]),phase_weight=float(met["weight"][j]),gain=float(gain[j])))
            rawerr=q["base"]-q["y"];rawlevel=rawerr.mean(1)**2;rawshape=np.mean((rawerr-rawerr.mean(1,keepdims=True))**2,1)
            _,rawmet,_=forecast_metrics(pack(q["base"],q["y"]),np.zeros(len(ii)))
            rows.append(dict(seed=seed,stage=stage,arm="OriginalTimeBase",n=len(ii),mse=float(np.mean(rawerr*rawerr)),mae=float(np.mean(abs(rawerr))),
               level=float(rawlevel.mean()),shape=float(rawshape.mean()),amplitude_error=float(rawmet["amplitude_error"].mean()),phase_weighted_mae_hours=float(np.sum(rawmet["weight"]*rawmet["phase"])/max(float(rawmet["weight"].sum()),1e-12)),gross_gain=0.,gross_harm=0.,shift_mean_hours=0.,shift_sd_hours=0.))
            hm={}
            for arm in c["contexts"]:
                path=Path(c["reuse_phase_cache"])/"evaluation_extraction"/f"hidden_{seed}_{arm}_{stage}.npy";h=np.load(path).astype(float);sources[str(path)]=rt.sha(path)
                basis=np.load(r/f"pca_s{seed}_{arm}.npz");hm[arm]=np.c_[q["features"],(h-basis["mu"])@basis["w"].T,p["coords"]]
            hm["QwenTrainShuffle"]=hm["QwenTask"];hm["SeasonFourier32"]=x
            for arm in c["teacher_arms"]:
                m=np.load(r/f"teacher_s{seed}_{arm}.npz");delta=predict(m,hm[arm],c);_,met,_=forecast_metrics(p,delta)
                teacherrows.append(dict(seed=seed,stage=stage,arm=arm,mse=float(met["mse"].mean()),mae=float(met["mae"].mean()),shift_mean_hours=float((delta/W).mean())))
    csvsave(r/"by_seed_period.csv",rows);csvsave(r/"cases.csv",cases);csvsave(r/"teacher_by_seed_period.csv",teacherrows);np.savez_compressed(r/"predictions.npz",**outputs)
    fields=["mse","mae","level","shape","amplitude_error","phase_weighted_mae_hours","gross_gain","gross_harm","shift_mean_hours","shift_sd_hours"]
    macro=[]
    for seed in c["seeds"]:
        for arm in ["OriginalTimeBase","FrozenBase"]+c["student_arms"]:
            a=[v for v in rows if v["seed"]==seed and v["arm"]==arm]
            macro.append(dict(seed=seed,arm=arm,**{k:float(np.mean([v[k] for v in a])) for k in fields}))
    csvsave(r/"macro_by_seed.csv",macro);summary=[]
    for arm in ["OriginalTimeBase","FrozenBase"]+c["student_arms"]:
        summary.append(dict(arm=arm,**{k:float(np.mean([v[k] for v in macro if v["arm"]==arm])) for k in fields}))
    csvsave(r/"summary.csv",summary)
    assert all(rt.sha(p)==v for p,v in lock["files"].items())
    src=rt.read(r/"source_train.json");assert all(rt.sha(p)==v for p,v in src["sources"].items())
    rt.save(r/"source_evaluation.json",dict(sources=sources,selection_checked_before_outer=True))
    assert time.monotonic()-tick<c["budget"]["max_eval_seconds"]
    rt.save(r/"completion.json",dict(selection_seconds=lock["seconds"],evaluation_seconds=time.monotonic()-tick,new_LLM_calls=0,
        LLM_updates=0,TimeBase_updates=0,head_parameters=c["trainable_parameters"],old_sources_and_choices_unchanged=True,
        conservation=maximum,evaluation_uses_saved_full_base=True))
    for v in summary:print("RESULT",json.dumps(v),flush=True)

def preflight(c):
    rng=np.random.default_rng(13);b=rng.normal(size=(8,96))+np.arange(8)[:,None];y=rng.normal(size=(8,96));p=pack(b,y)
    assert np.array_equal(rotate(p,np.zeros(8)),b)
    delta=rng.uniform(-c["delta_max"],c["delta_max"],8);pred,_,_=forecast_metrics(p,delta)
    assert abs(np.mean((pred-y)**2,1)-mse_vector(p,delta)).max()<1e-12
    other=rng.uniform(-c["delta_max"],c["delta_max"],8)
    exact=np.mean((pred-rotate(p,other))**2,1)
    assert abs(exact-4*p["E"]*np.sin((delta-other)/2)**2).max()<1e-12
    simple=pack(np.cos(W*np.arange(96))[None],np.cos(W*(np.arange(96)-1))[None])
    out=rotate(simple,np.array([W]));assert np.argmax(out[0,:24])==1
    bias=torch.zeros((),dtype=torch.float64,requires_grad=True);angle=c["delta_max"]*torch.tanh(bias)
    loss=4*float(simple["A"][0])*torch.sin(angle/2)**2-2*float(simple["B"][0])*torch.sin(angle);loss.backward()
    assert float(bias.grad)<0
    print("PREFLIGHT_PASS")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--preflight",action="store_true");ap.add_argument("--select",action="store_true");ap.add_argument("--evaluate",action="store_true");a=ap.parse_args()
    c=rt.read("docs/chronogeom/phase_rotation13_config.json");r=Path(c["run"])
    if a.preflight:preflight(c)
    elif a.select:select(c,r)
    elif a.evaluate:evaluate(c,r)
    else:raise ValueError("choose phase")
if __name__=="__main__":main()
