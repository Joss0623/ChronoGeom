"""Anchor frozen temporal teachers to the existing numeric response; ordinary shape KD."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
import argparse,json,time,traceback,shutil
from pathlib import Path
import numpy as np
from exp import exp_chronogeom_temporal_shape07 as s
u=s.u;rt=s.rt;csvsave=s.csvsave
TEACHER=dict(FourierKD="Fourier32",QwenTaskKD="QwenTask",QwenWrongKD="QwenWrong",QwenShuffleKD="QwenTrainShuffle")

def choose(rows):
    out={}
    for arm in sorted({v["arm"] for v in rows}):
        a=[v for v in rows if v["arm"]==arm]
        grid=[]
        for lam in sorted({v["lam"] for v in a}):
            b=[v for v in a if v["lam"]==lam]
            foldmean=[float(np.mean([v["ratio"] for v in b if v["fold"]==f])) for f in sorted({v["fold"] for v in b})]
            assert len(b)==9 and len(foldmean)==3
            grid.append(dict(lam=lam,loss=float(np.mean(foldmean)),fold_means=foldmean))
        best=min(grid,key=lambda v:(v["loss"],-v["lam"]))
        out[arm]=dict(lam=best["lam"],loss=best["loss"],grid=grid)
    return out

def load_train(c,r):
    data={};hidden={};sources={}
    for seed in c["seeds"]:
        f=Path(c["reuse_train"])/f"training_seed{seed}.npz";b=np.load(f)
        data[seed,"train"]={k:b[k] for k in b.files};sources[str(f)]=rt.sha(f)
        for arm in c["contexts"]:
            f=Path(c["reuse_train"])/f"hidden_{seed}_{arm}_train.npy"
            hidden[seed,arm,"train"]=np.load(f);sources[str(f)]=rt.sha(f)
    for f in [Path(__file__),Path(s.__file__),Path(u.__file__),Path("docs/chronogeom/temporal_shape_kd10_config.json"),Path("docs/chronogeom/current_temporal_kd_comparisons.json")]:
        sources[str(f)]=rt.sha(f)
    rt.save(r/"source_train.json",dict(git=rt.git(),sources=sources,no_outer_files_opened=True))
    return data,hidden

def fold_matrices(c,data,hidden):
    folds={}
    for seed in c["seeds"]:
        tr=data[seed,"train"];o=tr["origin"]
        for begin,end in c["folds"]:
            lo=int((np.datetime64(begin,"h")-np.datetime64("2006-12-16","h"))/np.timedelta64(1,"h"))
            hi=int((np.datetime64(end,"h")-np.datetime64("2006-12-16","h"))/np.timedelta64(1,"h"))
            ii=np.where(o+96<=lo)[0];vi=np.where((o>=lo)&(o+96<=hi))[0]
            assert len(ii)>=33 and len(vi)>=10 and (o[ii]+96).max()<=o[vi].min()
            dd=dict(data);dd[seed,"inner"]={k:v[vi] for k,v in tr.items()}
            hh=dict(hidden)
            for arm in c["contexts"]:hh[seed,arm,"inner"]=hidden[seed,arm,"train"][vi]
            mat,pca=u.matrices(c,dd,hh,seed,ii,dict(train=ii,inner=np.arange(len(vi))))
            folds[seed,begin]=(ii,vi,mat,pca)
    return folds

def fit_student(x,q,t,a,lam,eta):
    return s.fit_strength(x,(1+eta)*q,t+eta*q*a,lam)


def raw(m,x):
    return np.c_[np.ones(len(x)),(x-m["mu"])/m["sd"]]@m["coef"]

def context_diagnostics(c,r,data,h,folds):
    """Training-only explanation of09, before choosing10 settings; not new primary controls."""
    prior=Path(c["previous_run"])
    tc=rt.read(prior/"teacher_choice.json");sc=rt.read(prior/"student_choice.json")
    src=rt.read(r/"source_train.json")
    for f in [prior/"teacher_choice.json",prior/"student_choice.json"]:
        src["sources"][str(f)]=rt.sha(f)
    rt.save(r/"source_train.json",src)
    geometry=[];records=[]
    for seed in c["seeds"]:
        a=h[seed,"QwenTask","train"].astype(float);b=h[seed,"QwenWrong","train"].astype(float)
        da=a-b;ac=a-a.mean(0);bc=b-b.mean(0);ka=ac@ac.T;kb=bc@bc.T
        geometry.append(dict(seed=seed,n=len(a),
            raw_cosine=float(np.mean(np.sum(a*b,1)/np.linalg.norm(a,axis=1)/np.linalg.norm(b,axis=1))),
            centered_CKA=float(np.sum(ka*kb)/np.sqrt(np.sum(ka*ka)*np.sum(kb*kb))),
            context_difference_mean_energy_fraction=float(np.sum(da.mean(0)**2)/np.mean(np.sum(da*da,1)))))
    for (seed,name),(ii,vi,mat,pca) in folds.items():
        tr=data[seed,"train"];d,q,t=s.components(tr);m={};a={};train_a={}
        for arm in ["QwenTask","QwenWrong"]:
            m[arm]=s.fit_strength(mat["train"][arm],q[ii],t[ii],tc[arm]["lam"])
            a[arm]=s.strength(m[arm],mat["inner"][arm],c["alpha_bounds"])
            train_a[arm]=s.strength(m[arm],mat["train"][arm],c["alpha_bounds"])
        ma=s.fit_strength(mat["train"]["Fourier32"],q[ii],t[ii],c["anchor_regularizer"])
        no=s.strength(ma,mat["inner"]["Fourier32"],c["alpha_bounds"]);sa={}
        for arm in ["QwenTaskKD","QwenWrongKD"]:
            sm=fit_student(mat["train"]["Fourier32"],q[ii],t[ii],train_a[TEACHER[arm]],sc[arm]["lam"],1)
            sa[arm]=s.strength(sm,mat["inner"]["Fourier32"],c["alpha_bounds"])
        mu,w,_=pca["QwenTask"];wrong=h[seed,"QwenWrong","train"].astype(float)
        swap=s.strength(m["QwenTask"],np.c_[tr["features"][vi],(wrong[vi]-mu)@w.T],c["alpha_bounds"])
        aligned=s.strength(m["QwenTask"],np.c_[tr["features"][vi],(wrong[vi]-wrong[ii].mean(0))@w.T],c["alpha_bounds"])
        mse=lambda aa:float(((tr["base"][vi]+aa[:,None]*d[vi]-tr["y"][vi])**2).mean())
        change=sa["QwenTaskKD"]-no
        avg=float(np.sum(q[vi]*change)/q[vi].sum());variation=change-avg;grad=q[vi]*no-t[vi]
        constant_risk=float(np.mean(q[vi]*avg*avg+2*avg*grad))
        varying_risk=float(np.mean(q[vi]*variation*variation+2*variation*grad))
        assert abs(mse(sa["QwenTaskKD"])-mse(no)-constant_risk-varying_risk)<1e-12
        records.append(dict(seed=seed,fold=name,fit=len(ii),n=len(vi),NoKD=mse(no),TaskKD=mse(sa["QwenTaskKD"]),WrongKD=mse(sa["QwenWrongKD"]),
           constant_strength_change=avg,constant_change_risk=constant_risk,conditional_change_risk=varying_risk,
           native_Task_teacher_mse=mse(a["QwenTask"]),native_Wrong_teacher_mse=mse(a["QwenWrong"]),
           fixed_Task_readout_Wrong_input_mse=mse(swap),mean_aligned_Wrong_input_mse=mse(aligned),
           native_teacher_alpha_rmse=float(np.sqrt(np.mean((a["QwenTask"]-a["QwenWrong"])**2))),
           fixed_map_swap_alpha_rmse=float(np.sqrt(np.mean((a["QwenTask"]-swap)**2))),
           mean_aligned_swap_alpha_rmse=float(np.sqrt(np.mean((a["QwenTask"]-aligned)**2))),
           native_teacher_alpha_corr=float(np.corrcoef(a["QwenTask"],a["QwenWrong"])[0,1]),
           student_alpha_rmse=float(np.sqrt(np.mean((sa["QwenTaskKD"]-sa["QwenWrongKD"])**2)))))
    rt.save(r/"context_train_diagnostics.json",dict(train_only=True,no_outer_files_opened=True,geometry=geometry,folds=records,
        note="Swaps are fixed-map diagnostics, possibly shifted distributions; retain independently refit Wrong control. Context constant offsets removed by separate train centering are not proof of absent semantics."))

def select(c,r):
    r.mkdir(parents=True,exist_ok=False);rt.save(r/"config.json",c);shutil.copy2(__file__,r/Path(__file__).name);tick=time.monotonic()
    try:
        data,h=load_train(c,r);folds=fold_matrices(c,data,h);trecord=[]
        context_diagnostics(c,r,data,h,folds)
        anchors={}
        for (seed,name),(ii,vi,mat,pca) in folds.items():
            _,q,t=s.components(data[seed,"train"])
            am=s.fit_strength(mat["train"]["Fourier32"],q[ii],t[ii],c["anchor_regularizer"])
            anchors[seed,name]={stage:s.strength(am,mat[stage]["Fourier32"],c["alpha_bounds"]) for stage in ["train","inner"]}
        for (seed,name),(ii,vi,mat,pca) in folds.items():
            tr=data[seed,"train"];d,q,t=s.components(tr)
            for arm in c["teacher_arms"]:
                for lam in c["lambdas"]:
                    m=s.fit_strength(mat["train"][arm],q[ii],t[ii]-q[ii]*anchors[seed,name]["train"],lam)
                    a=np.clip(anchors[seed,name]["inner"]+raw(m,mat["inner"][arm]),*c["alpha_bounds"])
                    e=float(((tr["base"][vi]+a[:,None]*d[vi]-tr["y"][vi])**2).mean())
                    trecord.append(dict(seed=seed,fold=name,arm=arm,lam=lam,mse=e,ratio=e/float(tr["errors"][vi,0].mean())))
        tc=choose(trecord);csvsave(r/"teacher_inner.csv",trecord);rt.save(r/"teacher_choice.json",tc)
        teachers={}
        for (seed,name),(ii,vi,mat,pca) in folds.items():
            _,q,t=s.components(data[seed,"train"])
            teachers[seed,name]={}
            for arm in c["teacher_arms"]:
                m=s.fit_strength(mat["train"][arm],q[ii],t[ii]-q[ii]*anchors[seed,name]["train"],tc[arm]["lam"])
                teachers[seed,name][arm]=np.clip(anchors[seed,name]["train"]+raw(m,mat["train"][arm]),*c["alpha_bounds"])
        srecord=[]
        for (seed,name),(ii,vi,mat,pca) in folds.items():
            tr=data[seed,"train"];d,q,t=s.components(tr)
            for arm in c["student_arms"]:
                eta=0 if arm=="NoKD" else c["kd_weight"]
                ta=np.zeros(len(ii)) if arm=="NoKD" else teachers[seed,name][TEACHER[arm]]
                for lam in c["lambdas"]:
                    m=fit_student(mat["train"]["Fourier32"],q[ii],t[ii],ta,lam,eta)
                    a=s.strength(m,mat["inner"]["Fourier32"],c["alpha_bounds"])
                    e=float(((tr["base"][vi]+a[:,None]*d[vi]-tr["y"][vi])**2).mean())
                    srecord.append(dict(seed=seed,fold=name,arm=arm,lam=lam,mse=e,ratio=e/float(tr["errors"][vi,0].mean())))
        sc=choose(srecord);csvsave(r/"student_inner.csv",srecord);rt.save(r/"student_choice.json",sc)
        for seed in c["seeds"]:
            tr=data[seed,"train"];_,q,t=s.components(tr);ii=np.arange(len(q))
            mat,pca=u.matrices(c,data,h,seed,ii,dict(train=ii));ta={}
            am=s.fit_strength(mat["train"]["Fourier32"],q,t,c["anchor_regularizer"])
            np.savez_compressed(r/f"anchor_s{seed}.npz",**am)
            aa=s.strength(am,mat["train"]["Fourier32"],c["alpha_bounds"])
            for arm in c["teacher_arms"]:
                m=s.fit_strength(mat["train"][arm],q,t-q*aa,tc[arm]["lam"])
                np.savez_compressed(r/f"teacher_s{seed}_{arm}.npz",**m)
                ta[arm]=np.clip(aa+raw(m,mat["train"][arm]),*c["alpha_bounds"])
            np.savez_compressed(r/f"training_teacher_s{seed}.npz",**ta)
            for arm,(mu,w,info) in pca.items():np.savez_compressed(r/f"pca_s{seed}_{arm}.npz",mu=mu,w=w)
            for arm in c["student_arms"]:
                eta=0 if arm=="NoKD" else c["kd_weight"];a=np.zeros(len(q)) if arm=="NoKD" else ta[TEACHER[arm]]
                m=fit_student(mat["train"]["Fourier32"],q,t,a,sc[arm]["lam"],eta)
                np.savez_compressed(r/f"student_s{seed}_{arm}.npz",**m)
        src=rt.read(r/"source_train.json");assert all(rt.sha(p)==v for p,v in src["sources"].items())
        locks={str(p):rt.sha(p) for p in r.iterdir() if p.is_file()}
        assert time.monotonic()-tick<c["budget"]["max_workflow_seconds"]
        rt.save(r/"selection_lock.json",dict(seconds=time.monotonic()-tick,files=locks,training_only=True,no_outer_files_opened=True,
                 common_lambda_all_seeds=True,teacher_is_anchored_residual=True,anchor_regularizer=c["anchor_regularizer"],teacher_choice={a:v["lam"] for a,v in tc.items()},student_choice={a:v["lam"] for a,v in sc.items()}))
        print("SELECTED",json.dumps(dict(teachers={a:v["lam"] for a,v in tc.items()},students={a:v["lam"] for a,v in sc.items()})),flush=True)
    except BaseException:rt.save(r/"failure_select.json",dict(traceback=traceback.format_exc()));raise

def evaluate(c,r):
    lock=rt.read(r/"selection_lock.json");assert all(rt.sha(p)==v for p,v in lock["files"].items())
    tick=time.monotonic();prev=Path(c["reliability_run"]);pc=rt.read(prev/"config.json");cache=Path(pc["reuse_run"])
    summaries={(x["seed"],x["stage"],x["row"]):x["summary"] for x in rt.read(prev/"shared_summaries.json")}
    sources={str(prev/"shared_summaries.json"):rt.sha(prev/"shared_summaries.json")};pred={};rows=[];cases=[]
    for seed in c["seeds"]:
        models={arm:np.load(r/f"student_s{seed}_{arm}.npz") for arm in c["student_arms"]}
        for stage in c["stages"]:
            path=cache/f"seed{seed}_{stage}.npz";b=np.load(path);sources[str(path)]=rt.sha(path)
            v={k:b[k] for k in b.files};d,_,_=s.components(v)
            a=np.stack([s.flatten(summaries[seed,stage,j]) for j in range(len(v["origin"]))])
            x=np.c_[a,u.fourier(v["ids"][:,0,None])[:,0]]
            alphas=dict(Base=np.zeros(len(x)),**{arm:s.strength(m,x,c["alpha_bounds"]) for arm,m in models.items()})
            np.savez_compressed(r/f"deployment_input_s{seed}_{stage}.npz",x=x,base=v["base"],direction=d)
            for arm,alpha in alphas.items():
                p=v["base"]+alpha[:,None]*d;err=p-v["y"];mse=(err*err).mean(1);mae=abs(err).mean(1)
                level=err.mean(1)**2;shape=((err-err.mean(1,keepdims=True))**2).mean(1);gain=v["errors"][:,0]-mse
                me=float(abs(p.mean(1)-v["base"].mean(1)).max());assert me<1e-12 and abs(mse-level-shape).max()<1e-12
                pred[f"s{seed}_{stage}_{arm}"]=p;pred[f"a{seed}_{stage}_{arm}"]=alpha
                rows.append(dict(seed=seed,stage=stage,arm=arm,n=len(x),mse=float(mse.mean()),mae=float(mae.mean()),level=float(level.mean()),
                   shape=float(shape.mean()),gross_gain=float(np.maximum(gain,0).mean()),gross_harm=float(np.maximum(-gain,0).mean()),
                   alpha_mean=float(alpha.mean()),alpha_sd=float(alpha.std()),max_mean_error=me))
                for j in range(len(x)):cases.append(dict(seed=seed,stage=stage,arm=arm,origin=int(v["origin"][j]),alpha=float(alpha[j]),
                   mse=float(mse[j]),mae=float(mae[j]),level=float(level[j]),shape=float(shape[j]),gain=float(gain[j])))
    csvsave(r/"by_seed_period.csv",rows);csvsave(r/"cases.csv",cases);np.savez_compressed(r/"predictions.npz",**pred)
    fields=["mse","mae","level","shape","gross_gain","gross_harm","alpha_mean","alpha_sd"];macro=[]
    for seed in c["seeds"]:
        for arm in ["Base"]+c["student_arms"]:
            a=[v for v in rows if v["seed"]==seed and v["arm"]==arm]
            macro.append(dict(seed=seed,arm=arm,**{k:float(np.mean([v[k] for v in a])) for k in fields}))
    csvsave(r/"macro_by_seed.csv",macro);out=[]
    for arm in ["Base"]+c["student_arms"]:out.append(dict(arm=arm,**{k:float(np.mean([v[k] for v in macro if v["arm"]==arm])) for k in fields}))
    csvsave(r/"summary.csv",out)
    assert all(rt.sha(p)==v for p,v in sources.items()) and all(rt.sha(p)==v for p,v in lock["files"].items())
    rt.save(r/"source_evaluation.json",dict(sources=sources,selection_verified_before_loading_outer=True))
    assert time.monotonic()-tick+lock["seconds"]<c["budget"]["max_workflow_seconds"]
    assert sum(p.stat().st_size for p in r.rglob("*") if p.is_file())<c["budget"]["max_cache_mib"]*2**20
    rt.save(r/"completion.json",dict(selection_seconds=lock["seconds"],evaluation_seconds=time.monotonic()-tick,all_sources_unchanged=True,
       new_LLM_calls=0,LLM_updates=0,TimeBase_updates=0,KD=True,student_coefficients=80,student_input="47 past summaries + Fourier32",teacher_is_anchored_residual=True,
       mean_preservation=max(v["max_mean_error"] for v in rows),old_outer_development_only=True))
    for row in out:print("RESULT",json.dumps(row),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--select",action="store_true");ap.add_argument("--evaluate",action="store_true");ap.add_argument("--preflight",action="store_true");a=ap.parse_args()
    if a.preflight:
        test=[]
        for arm in ["a","b"]:
            for seed in [1,2,3]:
                for fold in [1,2,3]:
                    for lam in [.1,1]:test.append(dict(arm=arm,seed=seed,fold=fold,lam=lam,ratio=.9))
        assert all(v["lam"]==1 for v in choose(test).values())
        rng=np.random.default_rng(10);direction=rng.normal(size=(40,96));direction-=direction.mean(1,keepdims=True)
        err=rng.normal(size=(40,96));a0=rng.uniform(0,2,40);delta=rng.normal(size=40)
        q=(direction*direction).mean(1);t=(direction*err).mean(1)
        before=((a0[:,None]*direction-err)**2).mean(1)
        after=(((a0+delta)[:,None]*direction-err)**2).mean(1)
        assert np.max(abs(after-before-(q*delta**2-2*(t-q*a0)*delta)))<1e-12
        zero=s.fit_strength(rng.normal(size=(40,4)),q,np.zeros(40),1)
        assert np.max(abs(zero["coef"]))==0
        print("PREFLIGHT_PASS");return
    c=rt.read("docs/chronogeom/temporal_shape_kd10_config.json");r=Path(c["run"])
    if a.select:select(c,r)
    elif a.evaluate:evaluate(c,r)
    else:raise ValueError("choose separate --select or --evaluate phase")
if __name__=="__main__":main()
