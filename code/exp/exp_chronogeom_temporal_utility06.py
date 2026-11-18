"""Cross-season training coverage followup; fixed models, summaries and candidates."""
import argparse,json,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from exp import exp_chronogeom_temporal_utility05 as u
from exp.exp_chronogeom_temporal_reliability04 import summary,flatten,csvsave
from exp.exp_chronogeom_shape_opportunity03 import center
rt=u.rt

def prepare(c,r):
    data=u.prepare(c,r)
    cc=rt.read("docs/chronogeom/temporal_readout06_config.json")
    rr,oc,tr,tc,gc=rt.context(cc);b=np.load(rr/"power/known_data.npz")
    assert gc["domains"]["power"]["train_end"]==c["train"][0]
    p,counts=rt.windows(b["values"],b["ids"],*[rt.ix(cc,t) for t in c["train"]])
    week=np.load(Path(rt.read(Path(c["reliability_run"])/"config.json")["reuse_run"])/"weekly_template.npy")
    metadata=[]
    for seed in c["seeds"]:
        model,rec=rt.frozen_base(tc,tr,"power",seed,"cpu")
        p0=rt.old.predict(model,rt.tensorpack(p,"cpu"))[:,:,0].astype(float)
        assert rt.sha(rec["path"])==rec["sha256"]
        p1=.5*(center(p0)+center(week[p["ids"]]))
        p1=center(p1)+p0.mean(1,keepdims=True)
        # Keep the old compact references0/4 only; other columns are never evaluated.
        # All actual evaluated candidates remain the originalBase and MixWeek50.
        cand=np.repeat(p0[:,None],5,axis=1);cand[:,4]=p1
        x=p["x"][:,:,0].astype(float);y=p["y"][:,:,0].astype(float)
        ss=[summary(a,z,w,ids,week) for a,z,w,ids in zip(x,p0,p1,p["ids"])]
        q=dict(x=x,y=y,base=p0,ids=p["ids"],origin=p["origin"],candidates=cand,
               errors=((cand-y[:,None])**2).mean(2),features=np.stack([flatten(a) for a in ss]),summary=ss)
        data[seed,"train"]=q
        np.savez_compressed(r/f"training_seed{seed}.npz",**{k:v for k,v in q.items() if k!="summary"})
        metadata.append(dict(seed=seed,windows=len(x),base_checkpoint=rec))
        del model
    rt.save(r/"training_summary.json",{str(seed):data[seed,"train"]["summary"] for seed in c["seeds"]})
    rt.save(r/"training_meta.json",dict(range=c["train"],availability=counts,models=metadata,
       base_gradient_fit=["2009-04-27","2009-06-22"],base_checkpoint_selection_ended="2009-08-17",
       first_training_part_reuses_earlier_base_validation_not_independent_evaluation=True))
    src=rt.read(r/"source.json");src["code_sha"]=rt.sha(__file__);src["helper_sha"]=rt.sha(u.__file__)
    rt.save(r/"source.json",src)
    return data

def fit(c,r,data,hidden):
    records=[];selected={}
    for seed in c["seeds"]:
        q=data[seed,"train"];o=q["origin"];target=q["errors"][:,0]-q["errors"][:,4];folds=[]
        for begin,end in c["folds"]:
            lo=int((np.datetime64(begin,"h")-np.datetime64("2006-12-16","h"))/np.timedelta64(1,"h"))
            hi=int((np.datetime64(end,"h")-np.datetime64("2006-12-16","h"))/np.timedelta64(1,"h"))
            fitidx=np.where(o+96<=lo)[0];vi=np.where((o>=lo)&(o+96<=hi))[0]
            assert len(fitidx)>=33 and len(vi)>=10
            assert (o[fitidx]+96).max()<=o[vi].min()
            dd=dict(data);dd[seed,"inner"]={k:(v[vi] if isinstance(v,np.ndarray) else [v[j] for j in vi]) for k,v in q.items()}
            hh=dict(hidden)
            for arm in c["contexts"]:hh[seed,arm,"inner"]=hidden[seed,arm,"train"][vi]
            mat,pca=u.matrices(c,dd,hh,seed,fitidx,dict(train=fitidx,inner=np.arange(len(vi))))
            folds.append((begin,fitidx,vi,mat))
        choice={}
        for arm in c["arms"]:
            grid=[]
            for lam in c["lambdas"]:
                losses=[];detail=[]
                for name,ii,vi,mat in folds:
                    model=u.ridge_fit(mat["train"][arm],target[ii],lam)
                    g=u.ridge_predict(model,mat["inner"][arm]);pick=g>0
                    mse=float(np.where(pick,q["errors"][vi,4],q["errors"][vi,0]).mean())
                    ratio=mse/float(q["errors"][vi,0].mean());losses.append(ratio)
                    detail.append(dict(fold=name,fit=len(ii),selection=len(vi),mse=mse,normalized_mse=ratio,mixed=int(pick.sum())))
                grid.append(dict(lam=lam,loss=float(np.mean(losses)),folds=detail))
            best=min(grid,key=lambda v:(v["loss"],-v["lam"]));choice[arm]=best["lam"]
            records.append(dict(seed=seed,arm=arm,lambda_value=best["lam"],inner_loss=best["loss"],grid=grid))
        full=np.arange(len(o));ev={"train":full};ev.update({t:np.arange(len(data[seed,t]["origin"])) for t in c["stages"]})
        mat,pca=u.matrices(c,data,hidden,seed,full,ev);selected[seed]={t:{} for t in c["stages"]}
        for arm in c["arms"]:
            m=u.ridge_fit(mat["train"][arm],target,choice[arm]);np.savez_compressed(r/f"probe_s{seed}_{arm}.npz",**m)
            for t in c["stages"]:selected[seed][t][arm]=u.ridge_predict(m,mat[t][arm])
        for arm,(mu,w,info) in pca.items():np.savez_compressed(r/f"pca_s{seed}_{arm}.npz",mu=mu,w=w)
        rt.save(r/f"pca_s{seed}.json",{a:inf for a,(_,_,inf) in pca.items()})
        rt.save(r/"selection.json",records);print("FITTED",seed,choice,flush=True)
    return selected

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda:0");a=ap.parse_args()
    c=rt.read("docs/chronogeom/temporal_utility06_config.json")
    r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);rt.save(r/"config.json",c);shutil.copy2(__file__,r/Path(__file__).name)
    tick=time.monotonic();torch.set_num_threads(2)
    try:
        data=prepare(c,r)
        ec=dict(c);ec["stages"]=[]
        hidden=u.extract(ec,r,data,a.device)
        # Preserve the existing evaluation representations byte-for-byte; no prompt adaptation.
        old=Path(c["reuse_hidden"]);sha={}
        for seed in c["seeds"]:
            for arm in c["contexts"]:
                for stage in c["stages"]:
                    p=old/f"hidden_{seed}_{arm}_{stage}.npy"
                    hidden[seed,arm,stage]=np.load(p);sha[str(p)]=rt.sha(p)
        src=rt.read(r/"source.json");src["reused_hidden"]=sha;rt.save(r/"source.json",src)
        choice=fit(c,r,data,hidden);u.evaluate(c,r,data,choice)
        assert all(rt.sha(p)==v for p,v in src["sources"].items()) and all(rt.sha(p)==v for p,v in sha.items())
        assert time.monotonic()-tick<c["budget"]["max_workflow_seconds"]
        rt.save(r/"completion.json",dict(seconds=time.monotonic()-tick,old_sources_unchanged=True,
            original_evaluation_hidden_reused=True,LLM_updates=0,TimeBase_updates=0,KD=False))
        print("COMPLETE",flush=True)
    except BaseException:rt.save(r/"failure.json",dict(traceback=traceback.format_exc()));raise
if __name__=="__main__":main()
