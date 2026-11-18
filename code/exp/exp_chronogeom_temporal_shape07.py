"""Continuous, level-preserving strength readout on frozen TimeBase/Qwen caches."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
import argparse,json,shutil,time,traceback
from pathlib import Path
import numpy as np
from exp import exp_chronogeom_temporal_utility05 as u
from exp.exp_chronogeom_temporal_reliability04 import flatten,csvsave
rt=u.rt

def components(q):
    d=q["candidates"][:,4]-q["base"]
    assert np.max(abs(d.mean(1)))<1e-12
    energy=np.mean(d*d,axis=1)
    signal=np.mean(d*(q["y"]-q["base"]),axis=1)
    return d,energy,signal

def fit_strength(x,q,s,lam):
    mu=x.mean(0);sd=np.maximum(x.std(0),.001)
    z=np.c_[np.ones(len(x)),(x-mu)/sd]
    a=z.T@(q[:,None]*z)
    a+=q.sum()*lam*np.diag(np.r_[0.,np.ones(x.shape[1])])
    coef=np.linalg.solve(a,z.T@s)
    assert np.all(np.isfinite(coef))
    return dict(mu=mu,sd=sd,coef=coef)

def strength(m,x,bounds):
    z=np.c_[np.ones(len(x)),(x-m["mu"])/m["sd"]]
    return np.clip(z@m["coef"],*bounds)

def load_data(c,r):
    p=Path(c["reuse_train"]);outer=Path(c["reuse_hidden"])
    pc=rt.read(Path(c["reliability_run"])/"config.json");cache=Path(pc["reuse_run"])
    summaries={(a["seed"],a["stage"],a["row"]):a["summary"] for a in rt.read(Path(c["reliability_run"])/"shared_summaries.json")}
    data={};hidden={};sources={}
    def track(path):
        sources[str(path)]=rt.sha(path)
        return path
    for seed in c["seeds"]:
        f=track(p/f"training_seed{seed}.npz");b=np.load(f);data[seed,"train"]={k:b[k] for k in b.files}
        for stage in c["stages"]:
            f=track(cache/f"seed{seed}_{stage}.npz");b=np.load(f);q={k:b[k] for k in b.files}
            q["features"]=np.stack([flatten(summaries[seed,stage,j]) for j in range(len(q["origin"]))])
            data[seed,stage]=q
        for arm in c["contexts"]:
            for stage in ["train"]+c["stages"]:
                f=track((p if stage=="train" else outer)/f"hidden_{seed}_{arm}_{stage}.npy")
                hidden[seed,arm,stage]=np.load(f)
    for f in [p/"config.json",p/"source.json",Path(c["reliability_run"])/"shared_summaries.json",Path(u.__file__),Path(__file__),Path("docs/chronogeom/temporal_shape07_config.json")]:track(f)
    rt.save(r/"source.json",dict(git=rt.git(),sources=sources,original_models_frozen=True,new_llm_calls=0,all_outer_periods_previously_viewed=True))
    return data,hidden

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--preflight",action="store_true");args=ap.parse_args()
    if args.preflight:
        rng=np.random.default_rng(4);x=rng.normal(size=(60,4));q=rng.uniform(.02,1,60);s=q*(.4+.2*x[:,0])
        m=fit_strength(x,q,s,.1);z=np.c_[np.ones(60),(x-m["mu"])/m["sd"]]
        grad=z.T@(q*(z@m["coef"])-s)+q.sum()*.1*np.r_[0.,m["coef"][1:]]
        assert abs(grad).max()<1e-10
        # Forecast objective and quadratic strength objective are the same up to base MSE.
        d=rng.normal(size=(60,96));d-=d.mean(1,keepdims=True);e=rng.normal(size=(60,96));a=rng.uniform(0,2,60)
        exact=((a[:,None]*d-e)**2).mean(1)
        quad=(e*e).mean(1)+a*a*(d*d).mean(1)-2*a*(d*e).mean(1)
        assert abs(exact-quad).max()<1e-12
        print("PREFLIGHT_PASS");return
    c=rt.read("docs/chronogeom/temporal_shape07_config.json");r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False)
    rt.save(r/"config.json",c);shutil.copy2(__file__,r/Path(__file__).name);tick=time.monotonic()
    try:
        data,hidden=load_data(c,r);pred={};rows=[];cases=[];selection=[];models={}
        for seed in c["seeds"]:
            q=data[seed,"train"];o=q["origin"];d,energy,signal=components(q)
            folds=[]
            for begin,end in c["folds"]:
                lo=int((np.datetime64(begin,"h")-np.datetime64("2006-12-16","h"))/np.timedelta64(1,"h"))
                hi=int((np.datetime64(end,"h")-np.datetime64("2006-12-16","h"))/np.timedelta64(1,"h"))
                ii=np.where(o+96<=lo)[0];vi=np.where((o>=lo)&(o+96<=hi))[0]
                assert len(ii)>=33 and len(vi)>=10 and (o[ii]+96).max()<=o[vi].min()
                dd=dict(data);dd[seed,"inner"]={k:v[vi] for k,v in q.items()}
                hh=dict(hidden)
                for arm in c["contexts"]:hh[seed,arm,"inner"]=hidden[seed,arm,"train"][vi]
                mat,_=u.matrices(c,dd,hh,seed,ii,dict(train=ii,inner=np.arange(len(vi))))
                folds.append((begin,ii,vi,mat))
            choice={}
            for arm in c["arms"]:
                grid=[]
                for lam in c["lambdas"]:
                    details=[]
                    for name,ii,vi,mat in folds:
                        m=fit_strength(mat["train"][arm],energy[ii],signal[ii],lam)
                        a=strength(m,mat["inner"][arm],c["alpha_bounds"])
                        e=((q["base"][vi]+a[:,None]*d[vi]-q["y"][vi])**2).mean()
                        details.append(dict(fold=name,fit=len(ii),selection=len(vi),mse=float(e),
                                            ratio=float(e/q["errors"][vi,0].mean()),alpha_mean=float(a.mean()),alpha_sd=float(a.std())))
                    grid.append(dict(lam=lam,loss=float(np.mean([v["ratio"] for v in details])),folds=details))
                best=min(grid,key=lambda v:(v["loss"],-v["lam"]));choice[arm]=best["lam"]
                selection.append(dict(seed=seed,arm=arm,lambda_value=best["lam"],inner_loss=best["loss"],grid=grid))
            ev={"train":np.arange(len(o))};ev.update({stage:np.arange(len(data[seed,stage]["origin"])) for stage in c["stages"]})
            mat,pca=u.matrices(c,data,hidden,seed,np.arange(len(o)),ev)
            for arm in c["arms"]:
                m=fit_strength(mat["train"][arm],energy,signal,choice[arm]);models[seed,arm]=m
                np.savez_compressed(r/f"strength_s{seed}_{arm}.npz",**m)
            for arm,(mu,w,info) in pca.items():np.savez_compressed(r/f"pca_s{seed}_{arm}.npz",mu=mu,w=w)
            rt.save(r/f"pca_s{seed}.json",{a:info for a,(_,_,info) in pca.items()})
            const=float(np.clip(signal.sum()/energy.sum(),*c["alpha_bounds"]))
            for stage in c["stages"]:
                v=data[seed,stage];vd,vq,vs=components(v);n=len(vd)
                alphas=dict(Base=np.zeros(n),FixedMix=np.ones(n),TrainConstant=np.full(n,const),
                      OracleBinary=(v["errors"][:,0]>v["errors"][:,4]).astype(float),
                      OracleContinuous=np.clip(np.divide(vs,vq,out=np.zeros(n),where=vq>1e-14),*c["alpha_bounds"]))
                alphas.update({arm:strength(models[seed,arm],mat[stage][arm],c["alpha_bounds"]) for arm in c["arms"]})
                for arm,a in alphas.items():
                    p=v["base"]+a[:,None]*vd
                    err=p-v["y"];mse=(err*err).mean(1);mae=abs(err).mean(1)
                    level=err.mean(1)**2;shape=((err-err.mean(1,keepdims=True))**2).mean(1)
                    assert abs(mse-level-shape).max()<1e-12
                    mean_error=float(abs(p.mean(1)-v["base"].mean(1)).max());assert mean_error<1e-12
                    gain=v["errors"][:,0]-mse
                    pred[f"s{seed}_{stage}_{arm}"]=p;pred[f"a{seed}_{stage}_{arm}"]=a
                    rows.append(dict(seed=seed,stage=stage,arm=arm,n=n,mse=float(mse.mean()),mae=float(mae.mean()),
                       level=float(level.mean()),shape=float(shape.mean()),gross_gain=float(np.maximum(gain,0).mean()),
                       gross_harm=float(np.maximum(-gain,0).mean()),alpha_mean=float(a.mean()),alpha_sd=float(a.std()),
                       beneficial=int((gain>1e-8).sum()),harmful=int((gain< -1e-8).sum()),max_mean_error=mean_error))
                    for j in range(n):cases.append(dict(seed=seed,stage=stage,arm=arm,origin=int(v["origin"][j]),alpha=float(a[j]),
                        mse=float(mse[j]),mae=float(mae[j]),level=float(level[j]),shape=float(shape[j]),gain=float(gain[j])))
            print("FITTED",seed,choice,"CONSTANT",const,flush=True)
        rt.save(r/"selection.json",selection);csvsave(r/"by_seed_period.csv",rows);csvsave(r/"cases.csv",cases)
        np.savez_compressed(r/"predictions.npz",**pred)
        arms=["Base","FixedMix","TrainConstant","OracleBinary","OracleContinuous"]+c["arms"]
        fields=["mse","mae","level","shape","gross_gain","gross_harm","alpha_mean","alpha_sd"]
        macro=[]
        for seed in c["seeds"]:
            for arm in arms:
                a=[v for v in rows if v["seed"]==seed and v["arm"]==arm]
                macro.append(dict(seed=seed,arm=arm,**{k:float(np.mean([v[k] for v in a])) for k in fields}))
        csvsave(r/"macro_by_seed.csv",macro);out=[]
        for arm in arms:out.append(dict(arm=arm,**{k:float(np.mean([v[k] for v in macro if v["arm"]==arm])) for k in fields}))
        csvsave(r/"summary.csv",out)
        get=lambda arm:next(v["mse"] for v in out if v["arm"]==arm)
        wins=sum(next(v["mse"] for v in macro if v["seed"]==s and v["arm"]=="QwenTask")<next(v["mse"] for v in macro if v["seed"]==s and v["arm"]=="TrainConstant") for s in c["seeds"])
        passed=all(get("QwenTask")<get(a) for a in ["FixedMix","TrainConstant","Fourier32","RFF32","Calendar168","QwenWrong","QwenTrainShuffle"]) and wins>=2
        rt.save(r/"teacher_value.json",dict(conditional_increment=bool(passed),seed_wins_vs_learnedconstant=int(wins),KD_executed=False,rule=c["followup"]))
        src=rt.read(r/"source.json");assert all(rt.sha(p)==v for p,v in src["sources"].items())
        assert time.monotonic()-tick<c["budget"]["max_workflow_seconds"]
        assert sum(p.stat().st_size for p in r.rglob("*") if p.is_file())<c["budget"]["max_cache_mib"]*2**20
        rt.save(r/"completion.json",dict(seconds=time.monotonic()-tick,all_sources_unchanged=True,LLM_updates=0,
                   TimeBase_updates=0,new_LLM_calls=0,KD=False,max_mean_difference=max(v["max_mean_error"] for v in rows)))
        for v in out:print("RESULT",json.dumps(v),flush=True)
        print("CONDITIONAL_INCREMENT",passed,flush=True)
    except BaseException:rt.save(r/"failure.json",dict(traceback=traceback.format_exc()));raise
if __name__=="__main__":main()
