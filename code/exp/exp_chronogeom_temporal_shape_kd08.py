"""Minimal response-shape KD into an80-coefficient LLM-free TimeBase correction."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
import argparse,json,time,traceback,shutil
from pathlib import Path
import numpy as np
from exp import exp_chronogeom_temporal_shape07 as s
u=s.u;rt=s.rt;csvsave=s.csvsave

def kd_fit(x,q,t,a,lam,eta):
    return s.fit_strength(x,(1+eta)*q,t+eta*q*a,lam)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--preflight",action="store_true");args=ap.parse_args()
    if args.preflight:
        rng=np.random.default_rng(3);x=rng.normal(size=(50,4));q=rng.uniform(.1,1,50);t=rng.normal(size=50)
        a=s.fit_strength(x,q,t,.1);b=kd_fit(x,q,t,t/q,.1,1)
        assert abs(a["coef"]-b["coef"]).max()<1e-12
        print("PREFLIGHT_PASS");return
    c=rt.read("docs/chronogeom/temporal_shape_kd08_config.json");r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False)
    rt.save(r/"config.json",c);shutil.copy2(__file__,r/Path(__file__).name);tick=time.monotonic()
    try:
        data,hidden=s.load_data(c,r);teacher_run=Path(c["teacher_run"])
        teacher_choice={(v["seed"],v["arm"]):v["lambda_value"] for v in rt.read(teacher_run/"selection.json")}
        src=rt.read(r/"source.json")
        for p in [Path(__file__),Path("docs/chronogeom/temporal_shape_kd08_config.json"),teacher_run/"selection.json"]:
            src["sources"][str(p)]=rt.sha(p)
        rt.save(r/"source.json",src)
        map_teacher=dict(FourierKD="Fourier32",QwenTaskKD="QwenTask",QwenWrongKD="QwenWrong",QwenShuffleKD="QwenTrainShuffle")
        selection=[];rows=[];cases=[];pred={};max_teacher_replay=0.
        for seed in c["seeds"]:
            tr=data[seed,"train"];d,q,t=s.components(tr);o=tr["origin"]
            def teacher_alpha(mat,ii,check=False):
                out=dict(FixedMixKD=np.ones(len(ii)),ConstantKD=np.full(len(ii),float(np.clip(t[ii].sum()/q[ii].sum(),*c["alpha_bounds"]))))
                for student,teacher in map_teacher.items():
                    m=s.fit_strength(mat["train"][teacher],q[ii],t[ii],teacher_choice[seed,teacher])
                    out[student]=s.strength(m,mat["train"][teacher],c["alpha_bounds"])
                    if check:
                        saved=np.load(teacher_run/f"strength_s{seed}_{teacher}.npz")
                        assert np.max(abs(m["coef"]-saved["coef"]))<1e-10
                out["NoKD"]=np.zeros(len(ii));return out
            folds=[]
            for begin,end in c["folds"]:
                lo=int((np.datetime64(begin,"h")-np.datetime64("2006-12-16","h"))/np.timedelta64(1,"h"))
                hi=int((np.datetime64(end,"h")-np.datetime64("2006-12-16","h"))/np.timedelta64(1,"h"))
                ii=np.where(o+96<=lo)[0];vi=np.where((o>=lo)&(o+96<=hi))[0]
                assert (o[ii]+96).max()<=o[vi].min()
                dd=dict(data);dd[seed,"inner"]={k:v[vi] for k,v in tr.items()}
                hh=dict(hidden)
                for arm in c["contexts"]:hh[seed,arm,"inner"]=hidden[seed,arm,"train"][vi]
                mat,_=u.matrices(c,dd,hh,seed,ii,dict(train=ii,inner=np.arange(len(vi))))
                teachers=teacher_alpha(mat,ii)
                folds.append((begin,ii,vi,mat,teachers))
            choice={}
            for arm in c["student_arms"]:
                grid=[]
                for lam in c["lambdas"]:
                    detail=[]
                    for name,ii,vi,mat,teachers in folds:
                        eta=0 if arm=="NoKD" else c["kd_weight"]
                        m=kd_fit(mat["train"]["Fourier32"],q[ii],t[ii],teachers[arm],lam,eta)
                        a=s.strength(m,mat["inner"]["Fourier32"],c["alpha_bounds"])
                        mse=float(((tr["base"][vi]+a[:,None]*d[vi]-tr["y"][vi])**2).mean())
                        detail.append(dict(fold=name,fit=len(ii),selection=len(vi),mse=mse,ratio=mse/float(tr["errors"][vi,0].mean())))
                    grid.append(dict(lam=lam,loss=float(np.mean([v["ratio"] for v in detail])),folds=detail))
                best=min(grid,key=lambda v:(v["loss"],-v["lam"]));choice[arm]=best["lam"]
                selection.append(dict(seed=seed,arm=arm,lambda_value=best["lam"],inner_loss=best["loss"],grid=grid))
            full=np.arange(len(o));mat,_=u.matrices(c,data,hidden,seed,full,dict(train=full))
            teachers=teacher_alpha(mat,full,True);np.savez_compressed(r/f"training_teacher_alpha_s{seed}.npz",**teachers)
            models={}
            for arm in c["student_arms"]:
                models[arm]=kd_fit(mat["train"]["Fourier32"],q,t,teachers[arm],choice[arm],0 if arm=="NoKD" else c["kd_weight"])
                np.savez_compressed(r/f"student_s{seed}_{arm}.npz",**models[arm])
            const=float(np.clip(t.sum()/q.sum(),*c["alpha_bounds"]))
            for stage in c["stages"]:
                v=data[seed,stage];vd,_,_=s.components(v)
                # Deployment path contains NO Qwen features/PCA; same47 summaries + analytical Fourier32.
                x=np.c_[v["features"],u.fourier(v["ids"][:,0,None])[:,0]]
                alpha=dict(Base=np.zeros(len(x)),FixedMix=np.ones(len(x)),TrainConstant=np.full(len(x),const))
                alpha.update({a:s.strength(m,x,c["alpha_bounds"]) for a,m in models.items()})
                for arm,a in alpha.items():
                    p=v["base"]+a[:,None]*vd;err=p-v["y"];e=(err*err).mean(1);mae=abs(err).mean(1)
                    level=err.mean(1)**2;shape=((err-err.mean(1,keepdims=True))**2).mean(1);gain=v["errors"][:,0]-e
                    me=float(abs(p.mean(1)-v["base"].mean(1)).max());assert me<1e-12 and abs(e-level-shape).max()<1e-12
                    pred[f"s{seed}_{stage}_{arm}"]=p;pred[f"a{seed}_{stage}_{arm}"]=a
                    rows.append(dict(seed=seed,stage=stage,arm=arm,n=len(a),mse=float(e.mean()),mae=float(mae.mean()),
                        level=float(level.mean()),shape=float(shape.mean()),gross_gain=float(np.maximum(gain,0).mean()),gross_harm=float(np.maximum(-gain,0).mean()),
                        alpha_mean=float(a.mean()),alpha_sd=float(a.std()),max_mean_error=me))
                    for j in range(len(a)):cases.append(dict(seed=seed,stage=stage,arm=arm,origin=int(v["origin"][j]),alpha=float(a[j]),
                        mse=float(e[j]),mae=float(mae[j]),level=float(level[j]),shape=float(shape[j]),gain=float(gain[j])))
            print("FITTED",seed,choice,flush=True)
        csvsave(r/"cases.csv",cases);csvsave(r/"by_seed_period.csv",rows);np.savez_compressed(r/"predictions.npz",**pred);rt.save(r/"selection.json",selection)
        arms=["Base","FixedMix","TrainConstant"]+c["student_arms"];fields=["mse","mae","level","shape","gross_gain","gross_harm","alpha_mean","alpha_sd"];macro=[]
        for seed in c["seeds"]:
            for arm in arms:
                a=[v for v in rows if v["seed"]==seed and v["arm"]==arm]
                macro.append(dict(seed=seed,arm=arm,**{k:float(np.mean([v[k] for v in a])) for k in fields}))
        csvsave(r/"macro_by_seed.csv",macro);out=[]
        for arm in arms:out.append(dict(arm=arm,**{k:float(np.mean([v[k] for v in macro if v["arm"]==arm])) for k in fields}))
        csvsave(r/"summary.csv",out)
        assert all(rt.sha(p)==v for p,v in src["sources"].items())
        assert time.monotonic()-tick<c["budget"]["max_workflow_seconds"]
        assert sum(p.stat().st_size for p in r.rglob("*") if p.is_file())<c["budget"]["max_cache_mib"]*2**20
        rt.save(r/"completion.json",dict(seconds=time.monotonic()-tick,all_sources_unchanged=True,LLM_updates=0,TimeBase_updates=0,
            new_LLM_calls=0,KD=True,student_coefficients=80,student_input="47 rounded summaries + Fourier32",eta=c["kd_weight"],
            teacher_inner_fit_only=True,teacher_hyperparameters_reuse07_selection=True,old_outer_development_only=True,
            max_mean_difference=max(v["max_mean_error"] for v in rows)))
        for v in out:print("RESULT",json.dumps(v),flush=True)
    except BaseException:rt.save(r/"failure.json",dict(traceback=traceback.format_exc()));raise
if __name__=="__main__":main()
