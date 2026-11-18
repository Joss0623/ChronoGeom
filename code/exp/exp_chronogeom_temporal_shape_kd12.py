"""Frozen Qwen receives the same annual position as matched80-coefficient students."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
import argparse,json,time,traceback,shutil,subprocess
from pathlib import Path
import numpy as np
from exp import exp_chronogeom_temporal_shape07 as s
u=s.u;rt=s.rt;csvsave=s.csvsave
TEACHER=dict(SeasonFourierKD="SeasonFourier32",FourierKD="Fourier32",QwenTaskKD="QwenTask",QwenWrongKD="QwenWrong",QwenShuffleKD="QwenTrainShuffle")

def annual_features(x,origin,c):
    x=np.array(x,float,copy=True)
    # These two hourly phase columns are constant because all current windows start at00.
    assert np.max(abs(x[:,20]))<1e-12 and np.max(abs(x[:,21]-1))<1e-12
    dt=np.datetime64(c["clock_base"],"h")+np.asarray(origin).astype("timedelta64[h]")
    years=dt.astype("datetime64[Y]")
    begin=years.astype("datetime64[h]");end=(years+1).astype("datetime64[h]")
    phase=2*np.pi*((dt-begin)/np.timedelta64(1,"h"))/((end-begin)/np.timedelta64(1,"h"))
    x[:,20]=np.sin(phase);x[:,21]=np.cos(phase)
    assert np.all(np.isfinite(x)) and x.shape[1]==47
    return x

def matrices(c,data,hidden,seed,fitidx,evals):
    out,pca=u.matrices(c,data,hidden,seed,fitidx,evals)
    for stage,ii in evals.items():
        v=data[seed,stage];state=v["ids"][ii,0];a=v["features"][ii]
        annual=np.arctan2(a[:,20],a[:,21])
        parts=[]
        for period in [24.,168.]:
            for k in range(1,5):
                phase=2*np.pi*k*state/period;parts.extend([np.sin(phase),np.cos(phase)])
        for k in range(1,9):parts.extend([np.sin(k*annual),np.cos(k*annual)])
        f=np.stack(parts,axis=1);assert f.shape==(len(a),32)
        out[stage]["SeasonFourier32"]=np.c_[a,f]
    return out,pca

def seasonal_summary(old,features):
    out={g:dict(v) for g,v in old.items()}
    out["calendar"]={("annual_sin" if k=="hour_sin" else "annual_cos" if k=="hour_cos" else k):
                      (float(features[20]) if k=="hour_sin" else float(features[21]) if k=="hour_cos" else v)
                     for k,v in old["calendar"].items()}
    assert out["calendar"]["annual_sin"]==features[20] and out["calendar"]["annual_cos"]==features[21]
    return out

def seasonal_prompt(summary,context):
    from exp.exp_chronogeom_temporal_reliability04 import prompt
    text=prompt(summary,context,False)
    definition=("Known annual calendar position: annual_sin and annual_cos equal sin and cos of2*pi times "
       "the fraction of the civil calendar year elapsed at forecast issuance, accounting for leap years. "
       "Near January1 the pair is(0,1),near April1 (1,0),near July1 (0,-1),near October1 (-1,0). "
       "Use this calendar position together with the task context and recent observed history. "
       "It does not provide future weather,occupancy,events,or observed demand.\n")
    return text.replace("Shared summary:\n",definition+"Shared summary:\n")

def frozen_extract(c,out,data,stages,max_seconds):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    import torch
    out.mkdir(exist_ok=False);tick=time.monotonic();device=c["device"]
    tok=AutoTokenizer.from_pretrained(c["model_path"],local_files_only=True)
    req=[];tokens=[]
    for seed in c["seeds"]:
        for arm,context in c["contexts"].items():
            for stage in stages:
                for row,summary in enumerate(data[seed,stage]["summary"]):
                    text=seasonal_prompt(summary,context)
                    chat=tok.apply_chat_template([dict(role="user",content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    tt=tok.encode(chat,add_special_tokens=False)
                    req.append(dict(seed=seed,arm=arm,stage=stage,row=row,text=text));tokens.append(tt)
    assert len(req)<=c["budget"]["max_requests"] and max(map(len,tokens))<=c["budget"]["max_prompt_tokens"]
    rt.save(out/"requests.json",req)
    rt.setup(device,c["budget"]["gpu_cap_gib"])
    free_before=torch.cuda.mem_get_info(device)[0]/2**30
    assert free_before>=c["budget"]["min_free_gib"],free_before
    llm=AutoModelForCausalLM.from_pretrained(c["model_path"],local_files_only=True,dtype=torch.bfloat16,
              device_map={"":device},attn_implementation="sdpa")
    llm.eval()
    for p in llm.parameters():p.requires_grad_(False)
    before=u.fingerprint(llm);hidden={}
    with torch.inference_mode():
        for j,(r,tt) in enumerate(zip(req,tokens)):
            if time.monotonic()-tick>max_seconds:raise TimeoutError("global frozen extraction budget")
            x=torch.tensor([tt],device=device);mask=torch.ones_like(x)
            h=llm.model(input_ids=x,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state[0,-1].float().cpu().numpy()
            hidden.setdefault((r["seed"],r["arm"],r["stage"]),[]).append(h)
            if (j+1)%256==0:print("HIDDEN",out.name,j+1,len(req),round(time.monotonic()-tick,2),flush=True)
    assert u.fingerprint(llm)==before and all(not p.requires_grad and p.grad is None for p in llm.parameters())
    for key,values in hidden.items():
        hidden[key]=np.stack(values);np.save(out/("hidden_%s_%s_%s.npy"%key),hidden[key])
    rt.save(out/"extraction.json",dict(seconds=time.monotonic()-tick,requests=len(req),max_tokens=max(map(len,tokens)),
        device=device,free_gib_before_model=free_before,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,
        fingerprint_before=before,fingerprint_after=u.fingerprint(llm),llm_updates=0,nonthinking=True,
        hook="model final-normalized last token",prompt_fixed=True))
    del llm;torch.cuda.empty_cache()
    return hidden

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
    summary_path=Path(c["reuse_train"])/"training_summary.json"
    summaries=rt.read(summary_path);sources[str(summary_path)]=rt.sha(summary_path)
    for seed in c["seeds"]:
        f=Path(c["reuse_train"])/f"training_seed{seed}.npz";b=np.load(f)
        data[seed,"train"]={k:b[k] for k in b.files};sources[str(f)]=rt.sha(f)
        data[seed,"train"]["features"]=annual_features(data[seed,"train"]["features"],data[seed,"train"]["origin"],c)
        data[seed,"train"]["summary"]=[seasonal_summary(v,data[seed,"train"]["features"][j]) for j,v in enumerate(summaries[str(seed)])]
    for f in [Path(__file__),Path(s.__file__),Path(u.__file__),Path("docs/chronogeom/temporal_shape_kd12_config.json"),Path("docs/chronogeom/current_temporal_kd_comparisons.json"),Path("docs/chronogeom/temporal_readout06_config.json")]:
        sources[str(f)]=rt.sha(f)
    rt.save(r/"source_train.json",dict(git=rt.git(),sources=sources,no_outer_files_opened=True))
    hidden=frozen_extract(c,r/"train_extraction",data,["train"],c["budget"]["max_llm_seconds"])
    src=rt.read(r/"source_train.json")
    for p in (r/"train_extraction").iterdir():src["sources"][str(p)]=rt.sha(p)
    rt.save(r/"source_train.json",src)
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
            dd=dict(data);dd[seed,"inner"]={k:(v[vi] if isinstance(v,np.ndarray) else [v[j] for j in vi]) for k,v in tr.items()}
            hh=dict(hidden)
            for arm in c["contexts"]:hh[seed,arm,"inner"]=hidden[seed,arm,"train"][vi]
            mat,_=matrices(c,dd,hh,seed,ii,dict(train=ii,inner=np.arange(len(vi))))
            folds[seed,begin]=(ii,vi,mat)
    return folds

def fit_student(x,q,t,a,lam,eta):
    return s.fit_strength(x,(1+eta)*q,t+eta*q*a,lam)

def select(c,r):
    r.mkdir(parents=True,exist_ok=False);rt.save(r/"config.json",c);shutil.copy2(__file__,r/Path(__file__).name);tick=time.monotonic()
    try:
        data,h=load_train(c,r);folds=fold_matrices(c,data,h);trecord=[]
        for (seed,name),(ii,vi,mat) in folds.items():
            tr=data[seed,"train"];d,q,t=s.components(tr)
            for arm in c["teacher_arms"]:
                for lam in c["lambdas"]:
                    m=s.fit_strength(mat["train"][arm],q[ii],t[ii],lam)
                    a=s.strength(m,mat["inner"][arm],c["alpha_bounds"])
                    e=float(((tr["base"][vi]+a[:,None]*d[vi]-tr["y"][vi])**2).mean())
                    trecord.append(dict(seed=seed,fold=name,arm=arm,lam=lam,mse=e,ratio=e/float(tr["errors"][vi,0].mean())))
        tc=choose(trecord);csvsave(r/"teacher_inner.csv",trecord);rt.save(r/"teacher_choice.json",tc)
        teachers={}
        for (seed,name),(ii,vi,mat) in folds.items():
            _,q,t=s.components(data[seed,"train"])
            teachers[seed,name]={}
            for arm in c["teacher_arms"]:
                m=s.fit_strength(mat["train"][arm],q[ii],t[ii],tc[arm]["lam"])
                teachers[seed,name][arm]=s.strength(m,mat["train"][arm],c["alpha_bounds"])
        srecord=[]
        for (seed,name),(ii,vi,mat) in folds.items():
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
            mat,pca=matrices(c,data,h,seed,ii,dict(train=ii));ta={}
            for arm in c["teacher_arms"]:
                m=s.fit_strength(mat["train"][arm],q,t,tc[arm]["lam"])
                np.savez_compressed(r/f"teacher_s{seed}_{arm}.npz",**m)
                ta[arm]=s.strength(m,mat["train"][arm],c["alpha_bounds"])
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
                 common_lambda_all_seeds=True,annual_position_in_all_groups=True,teacher_prompt_includes_same_annual_fields=True,teacher_choice={a:v["lam"] for a,v in tc.items()},student_choice={a:v["lam"] for a,v in sc.items()}))
        print("SELECTED",json.dumps(dict(teachers={a:v["lam"] for a,v in tc.items()},students={a:v["lam"] for a,v in sc.items()})),flush=True)
    except BaseException:rt.save(r/"failure_select.json",dict(traceback=traceback.format_exc()));raise

def evaluate(c,r):
    lock=rt.read(r/"selection_lock.json");assert all(rt.sha(p)==v for p,v in lock["files"].items())
    tick=time.monotonic();prev=Path(c["reliability_run"]);pc=rt.read(prev/"config.json");cache=Path(pc["reuse_run"])
    summaries={(x["seed"],x["stage"],x["row"]):x["summary"] for x in rt.read(prev/"shared_summaries.json")}
    sources={str(prev/"shared_summaries.json"):rt.sha(prev/"shared_summaries.json")};pred={};rows=[];cases=[]
    outer={}
    for seed in c["seeds"]:
        for stage in c["stages"]:
            path=cache/f"seed{seed}_{stage}.npz";b=np.load(path);sources[str(path)]=rt.sha(path)
            v={k:b[k] for k in b.files}
            v["features"]=annual_features(np.stack([s.flatten(summaries[seed,stage,j]) for j in range(len(v["origin"]))]),v["origin"],c)
            v["summary"]=[seasonal_summary(summaries[seed,stage,j],v["features"][j]) for j in range(len(v["origin"]))]
            outer[seed,stage]=v
    used=rt.read(r/"train_extraction"/"extraction.json")["seconds"]
    h=frozen_extract(c,r/"evaluation_extraction",outer,c["stages"],c["budget"]["max_llm_seconds"]-used)
    for p in (r/"evaluation_extraction").iterdir():sources[str(p)]=rt.sha(p)
    teacher_cases=[]
    for seed in c["seeds"]:
        for stage in c["stages"]:
            v=outer[seed,stage];d,_,_=s.components(v)
            for arm in c["contexts"]:
                basis=np.load(r/f"pca_s{seed}_{arm}.npz");m=np.load(r/f"teacher_s{seed}_{arm}.npz")
                x=np.c_[v["features"],(h[seed,arm,stage].astype(float)-basis["mu"])@basis["w"].T]
                alpha=s.strength(m,x,c["alpha_bounds"]);pp=v["base"]+alpha[:,None]*d
                for j in range(len(alpha)):teacher_cases.append(dict(seed=seed,stage=stage,arm=arm,origin=int(v["origin"][j]),alpha=float(alpha[j]),
                    mse=float(((pp[j]-v["y"][j])**2).mean()),mae=float(abs(pp[j]-v["y"][j]).mean())))
    csvsave(r/"teacher_cases.csv",teacher_cases)
    rt.save(r/"extraction_total.json",dict(requests=rt.read(r/"train_extraction"/"extraction.json")["requests"]+rt.read(r/"evaluation_extraction"/"extraction.json")["requests"],
         seconds=used+rt.read(r/"evaluation_extraction"/"extraction.json")["seconds"],new_LLM_calls=True,LLM_updates=0))
    for seed in c["seeds"]:
        models={arm:np.load(r/f"student_s{seed}_{arm}.npz") for arm in c["student_arms"]}
        for stage in c["stages"]:
            path=cache/f"seed{seed}_{stage}.npz";b=np.load(path);sources[str(path)]=rt.sha(path)
            v={k:b[k] for k in b.files};d,_,_=s.components(v)
            a=np.stack([s.flatten(summaries[seed,stage,j]) for j in range(len(v["origin"]))])
            a=annual_features(a,v["origin"],c)
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
       new_LLM_calls=rt.read(r/"extraction_total.json")["requests"],LLM_updates=0,TimeBase_updates=0,KD=True,student_coefficients=80,student_input="45 prior summaries + annual sin/cos + Fourier32",annual_position_in_all_groups=True,
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
        xx=np.zeros((3,47));xx[:,21]=1
        yy=annual_features(xx,np.array([0,365*24,366*24]),dict(clock_base="2009-01-01"))
        assert abs(yy[0,20])<1e-12 and abs(yy[0,21]-1)<1e-12 and np.array_equal(yy[0],yy[1])
        features=np.zeros(47);features[20]=.8;features[21]=.6
        ss=seasonal_summary(dict(calendar=dict(start_weekday=1,start_hour=0,hour_sin=0,hour_cos=1)),features)
        assert ss["calendar"]["annual_sin"]==.8 and "hour_sin" not in ss["calendar"]
        text=seasonal_prompt(ss,"household electricity")
        assert '"annual_sin":0.8' in text and "civil calendar year" in text
        print("PREFLIGHT_PASS");return
    c=rt.read("docs/chronogeom/temporal_shape_kd12_config.json");r=Path(c["run"])
    if a.select:select(c,r)
    elif a.evaluate:evaluate(c,r)
    else:raise ValueError("choose separate --select or --evaluate phase")
if __name__=="__main__":main()
