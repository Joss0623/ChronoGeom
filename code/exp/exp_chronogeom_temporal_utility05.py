"""Train small utility readouts on frozen Qwen states; no backbone or LLM updates."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,hashlib,json,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from exp.exp_chronogeom_temporal_reliability04 import prompt,flatten,fingerprint,csvsave
from exp import exp_chronogeom_temporal_readout06 as rt
from exp.exp_chronogeom_dynamic_teacher01 import fourier

def prepare(c,r):
    prev=Path(c["reliability_run"]);pc=rt.read(prev/"config.json");cache=Path(pc["reuse_run"])
    summary_path=prev/"shared_summaries.json"
    summaries={(a["seed"],a["stage"],a["row"]):a["summary"] for a in rt.read(summary_path)}
    sources=dict(rt.read(prev/"source.json")["sources"]);sources[str(summary_path)]=rt.sha(summary_path)
    sources[str(prev/"config.json")]=rt.sha(prev/"config.json")
    sources["exp/exp_chronogeom_temporal_reliability04.py"]=rt.sha("exp/exp_chronogeom_temporal_reliability04.py")
    assert all(rt.sha(p)==v for p,v in sources.items())
    data={}
    for seed in c["seeds"]:
        for stage in ["train"]+c["stages"]:
            q=np.load(cache/f"seed{seed}_{stage}.npz")
            q={k:q[k] for k in q.files}
            q["summary"]=[summaries[seed,stage,j] for j in range(len(q["x"]))]
            q["features"]=np.stack([flatten(a) for a in q["summary"]])
            data[seed,stage]=q
    rt.save(r/"source.json",dict(git=rt.git(),code_sha=rt.sha(__file__),sources=sources,
          frozen_cached_forecasts=True,no_new_forecast_inputs=True,all_outer_periods_previously_viewed=True))
    return data

def extract(c,r,data,device):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(c["model_path"],local_files_only=True)
    requests=[];tokens=[]
    for seed in c["seeds"]:
        for arm,context in c["contexts"].items():
            for stage in ["train"]+c["stages"]:
                for j,s in enumerate(data[seed,stage]["summary"]):
                    text=prompt(s,context,False)
                    chat=tok.apply_chat_template([dict(role="user",content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    tt=tok.encode(chat,add_special_tokens=False)
                    requests.append(dict(seed=seed,arm=arm,stage=stage,row=j,text=text))
                    tokens.append(tt)
    assert len(requests)<=c["budget"]["max_requests"] and max(map(len,tokens))<2304
    rt.save(r/"requests.json",requests)
    rt.setup(device,c["budget"]["gpu_cap_gib"])
    assert torch.cuda.mem_get_info(device)[0]>=c["budget"]["gpu_cap_gib"]*2**30
    tick=time.monotonic()
    llm=AutoModelForCausalLM.from_pretrained(c["model_path"],local_files_only=True,dtype=torch.bfloat16,
            device_map={"":device},attn_implementation="sdpa")
    llm.eval()
    for p in llm.parameters():p.requires_grad_(False)
    before=fingerprint(llm);features={}
    with torch.inference_mode():
        for j,(req,tt) in enumerate(zip(requests,tokens)):
            if time.monotonic()-tick>c["budget"]["max_llm_seconds"]:raise TimeoutError("frozen hidden extraction cap")
            batch=dict(input_ids=torch.tensor([tt],device=device))
            batch["attention_mask"]=torch.ones_like(batch["input_ids"])
            h=llm.model(**batch,use_cache=False,return_dict=True).last_hidden_state[0,-1].float().cpu().numpy()
            key=(req["seed"],req["arm"],req["stage"]);features.setdefault(key,[]).append(h)
            if (j+1)%512==0:print("HIDDEN",j+1,len(requests),round(time.monotonic()-tick,2),flush=True)
    for key,v in features.items():
        a=np.stack(v);features[key]=a
        np.save(r/("hidden_%s_%s_%s.npy"%key),a)
    assert before==fingerprint(llm)
    assert all(not p.requires_grad and p.grad is None for p in llm.parameters())
    rt.save(r/"extraction.json",dict(seconds=time.monotonic()-tick,requests=len(requests),max_tokens=max(map(len,tokens)),
        peak_gib=torch.cuda.max_memory_allocated(device)/2**30,fingerprint_before=before,fingerprint_after=fingerprint(llm),
        last_token_after_final_norm=True,llm_updates=0,TimeBase_updates=0))
    del llm;torch.cuda.empty_cache()
    return features

def pca_fit(x,d):
    # Exact economy PCA from the sample Gram matrix; avoid a full 4096D SVD.
    mu=x.mean(0);z=x-mu
    ev,u=np.linalg.eigh(z@z.T)
    order=np.argsort(ev)[::-1];ev=ev[order];u=u[:,order]
    rank=min(d,int(np.sum(ev>max(float(ev[0])*1e-10,1e-12))))
    w=np.zeros((d,x.shape[1]))
    w[:rank]=u[:,:rank].T@z/np.sqrt(ev[:rank,None])
    assert np.max(abs(w[:rank]@w[:rank].T-np.eye(rank)))<1e-5
    return mu,w,dict(retained_rank=rank,explained_variance=float(ev[:rank].sum()/max(ev.sum(),1e-12)))

def ridge_fit(x,y,lam):
    mu=x.mean(0);sd=np.maximum(x.std(0),.001);z=(x-mu)/sd
    ym=float(y.mean());coef=np.linalg.solve(z.T@z+len(z)*lam*np.eye(z.shape[1]),z.T@(y-ym))
    return dict(mu=mu,sd=sd,coef=coef,intercept=ym)

def ridge_predict(m,x):return ((x-m["mu"])/m["sd"])@m["coef"]+m["intercept"]

def matrices(c,data,hidden,seed,fitidx,evals):
    train=data[seed,"train"];tx=train["features"];mu=tx[fitidx].mean(0);sd=np.maximum(tx[fitidx].std(0),.001)
    rng=np.random.default_rng(seed+4405);W=rng.normal(size=(tx.shape[1],32))/np.sqrt(tx.shape[1]);phase=rng.uniform(0,2*np.pi,size=32)
    models={}
    for arm in c["contexts"]:
        models[arm]=pca_fit(hidden[seed,arm,"train"][fitidx].astype(float),c["latent_dim"])
    out={}
    for stage,ixs in evals.items():
        q=data[seed,stage];a=q["features"][ixs];state=q["ids"][ixs,0]
        out[stage]={}
        out[stage]["SummaryOnly"]=a
        out[stage]["Fourier32"]=np.c_[a,fourier(state[:,None])[:,0]]
        out[stage]["RFF32"]=np.c_[a,np.sqrt(2/32)*np.cos(((a-mu)/sd)@W+phase)]
        out[stage]["Calendar168"]=np.c_[a,np.eye(168)[state]]
        for arm,(hm,w,info) in models.items():
            h=(hidden[seed,arm,stage][ixs].astype(float)-hm)@w.T
            out[stage][arm]=np.c_[a,h]
        out[stage]["QwenTrainShuffle"]=out[stage]["QwenTask"].copy()
    # Break only train identity correspondence, preserving correct numerical summaries.
    perm=np.random.default_rng(seed+5505).permutation(len(fitidx))
    out["train"]["QwenTrainShuffle"][:,47:]=out["train"]["QwenTask"][perm,47:]
    return out,models

def fit(c,r,data,hidden):
    records=[];selected={};cut=int((np.datetime64(c["internal_cut"],"h")-np.datetime64("2006-12-16","h"))/np.timedelta64(1,"h"))
    for seed in c["seeds"]:
        q=data[seed,"train"];fitidx=np.where(q["origin"]+96<=cut)[0];valididx=np.where(q["origin"]>=cut)[0]
        assert len(fitidx)>50 and len(valididx)>20
        assert (q["origin"][fitidx]+96).max()<=q["origin"][valididx].min()
        # Validation array is copied into a temporary named set, keeping train labels separate.
        inner_data=dict(data);inner_data[seed,"inner"]={k:(v[valididx] if isinstance(v,np.ndarray) else [v[j] for j in valididx]) for k,v in q.items()}
        inner_h=dict(hidden)
        for arm in c["contexts"]:inner_h[seed,arm,"inner"]=hidden[seed,arm,"train"][valididx]
        mat,pca=matrices(c,inner_data,inner_h,seed,fitidx,dict(train=fitidx,inner=np.arange(len(valididx))))
        target=q["errors"][:,0]-q["errors"][:,4]
        choice={}
        for arm in c["arms"]:
            losses=[]
            for lam in c["lambdas"]:
                m=ridge_fit(mat["train"][arm],target[fitidx],lam);g=ridge_predict(m,mat["inner"][arm]);pick=g>0
                e=np.where(pick,q["errors"][valididx,4],q["errors"][valididx,0])
                losses.append(dict(lam=lam,mse=float(e.mean()),mix_count=int(pick.sum())))
            # Fixed tie-break prefers stronger regularization.
            best=min(losses,key=lambda v:(v["mse"],-v["lam"]));choice[arm]=best["lam"]
            records.append(dict(seed=seed,arm=arm,fit=len(fitidx),selection=len(valididx),purged=len(q["origin"])-len(fitidx)-len(valididx),
                  lambda_value=choice[arm],inner_mse=best["mse"],grid=losses))
        full=np.arange(len(q["origin"]));evals={"train":full}
        evals.update({t:np.arange(len(data[seed,t]["origin"])) for t in c["stages"]})
        mat,pca=matrices(c,data,hidden,seed,full,evals);selected[seed]={t:{} for t in c["stages"]}
        for arm in c["arms"]:
            m=ridge_fit(mat["train"][arm],target,choice[arm]);np.savez_compressed(r/f"probe_s{seed}_{arm}.npz",**m)
            for t in c["stages"]:selected[seed][t][arm]=ridge_predict(m,mat[t][arm])
        for arm,(mu,w,info) in pca.items():np.savez_compressed(r/f"pca_s{seed}_{arm}.npz",mu=mu,w=w)
        rt.save(r/f"pca_s{seed}.json",{arm:info for arm,(_,_,info) in pca.items()})
        rt.save(r/"selection.json",records)
        print("FITTED",seed,choice,flush=True)
    return selected

def evaluate(c,r,data,selected):
    rows=[];cases=[];predictions={}
    for seed in c["seeds"]:
        for stage in c["stages"]:
            q=data[seed,stage];n=len(q["y"])
            scores=dict(Base=np.full(n,-1.),FixedMix=np.ones(n),Oracle=q["errors"][:,0]-q["errors"][:,4],**selected[seed][stage])
            for arm,g in scores.items():
                pick=g>0;p=np.where(pick[:,None],q["candidates"][:,4],q["base"]);e=np.mean((p-q["y"])**2,1);gain=q["errors"][:,0]-e
                assert np.max(abs(p.mean(1)-q["base"].mean(1)))<1e-12
                predictions[f"s{seed}_{stage}_{arm}"]=p
                rows.append(dict(seed=seed,stage=stage,arm=arm,n=n,mse=float(e.mean()),mae=float(abs(p-q["y"]).mean()),
                    gross_gain=float(np.maximum(gain,0).mean()),gross_harm=float(np.maximum(-gain,0).mean()),
                    mixed=int(pick.sum()),beneficial=int((gain>1e-8).sum()),harmful=int((gain< -1e-8).sum())))
                for j in range(n):
                    cases.append(dict(seed=seed,stage=stage,arm=arm,origin=int(q["origin"][j]),predicted_gain=float(g[j]),mix=int(pick[j]),
                         mse=float(e[j]),mae=float(abs(p[j]-q["y"][j]).mean()),gain=float(gain[j])))
    csvsave(r/"by_seed_period.csv",rows);csvsave(r/"cases.csv",cases);np.savez_compressed(r/"predictions.npz",**predictions)
    macro=[]
    for seed in c["seeds"]:
        for arm in ["Base","FixedMix","Oracle"]+c["arms"]:
            a=[v for v in rows if v["seed"]==seed and v["arm"]==arm]
            macro.append(dict(seed=seed,arm=arm,**{k:float(np.mean([v[k] for v in a])) for k in ["mse","mae","gross_gain","gross_harm"]}))
    csvsave(r/"macro_by_seed.csv",macro)
    summary=[]
    for arm in ["Base","FixedMix","Oracle"]+c["arms"]:
        a=[v for v in macro if v["arm"]==arm]
        summary.append(dict(arm=arm,**{k:float(np.mean([v[k] for v in a])) for k in ["mse","mae","gross_gain","gross_harm"]}))
    csvsave(r/"summary.csv",summary)
    get=lambda arm:next(a["mse"] for a in summary if a["arm"]==arm)
    seedwins=sum(next(a["mse"] for a in macro if a["seed"]==s and a["arm"]=="QwenTask")<next(a["mse"] for a in macro if a["seed"]==s and a["arm"]=="FixedMix") for s in c["seeds"])
    passed=all(get("QwenTask")<get(arm) for arm in ["FixedMix","Fourier32","RFF32","Calendar168","QwenTrainShuffle"]) and seedwins>=2
    rt.save(r/"gate.json",dict(passed=bool(passed),rule=c["followup"],seed_wins_vs_fixed=int(seedwins),KD_executed=False))
    for v in summary:print("RESULT",json.dumps(v),flush=True)
    print("GATE",passed,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda:0");ap.add_argument("--preflight",action="store_true");a=ap.parse_args()
    c=rt.read("docs/chronogeom/temporal_utility05_config.json")
    if a.preflight:
        x=np.random.default_rng(1).normal(size=(60,100))
        mu,w,info=pca_fit(x,32);m=ridge_fit(x[:,:47],x[:,0],.1)
        assert np.isfinite(ridge_predict(m,x[:,:47])).all()
        assert fourier(np.array([[0],[25]])).shape==(2,1,32)
        print("PREFLIGHT PCA",info,"ridge finite, shared Fourier32 confirmed");return
    r=Path(c["run"]);r.mkdir(exist_ok=False,parents=True);rt.save(r/"config.json",c);shutil.copy2(__file__,r/Path(__file__).name)
    tick=time.monotonic();torch.set_num_threads(2)
    try:
        data=prepare(c,r);hidden=extract(c,r,data,a.device);fitdata=fit(c,r,data,hidden);evaluate(c,r,data,fitdata)
        sources=rt.read(r/"source.json")["sources"];assert all(rt.sha(p)==v for p,v in sources.items())
        assert time.monotonic()-tick<c["budget"]["max_workflow_seconds"]
        rt.save(r/"completion.json",dict(seconds=time.monotonic()-tick,old_sources_unchanged=True,
             LLM_updates=0,TimeBase_updates=0,only_small_CPU_probes_fit=True))
        print("COMPLETE",flush=True)
    except BaseException:rt.save(r/"failure.json",dict(traceback=traceback.format_exc()));raise
if __name__=="__main__":main()
