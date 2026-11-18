"""Two-candidate temporal-prior reliability, using frozen TSLib forecasts and Qwen."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,csv,hashlib,inspect,json,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from exp.exp_chronogeom_shape_opportunity03 import center
from exp import exp_chronogeom_temporal_readout06 as rt
from scripts.chronogeom.teacher_pairwise02 import outputs

def csvsave(path, rows):
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader();w.writerows(rows)

def corr(a,b):
    a=center(np.asarray(a,float));b=center(np.asarray(b,float))
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den>1e-10 else 0.

def spectral(v,start_hour):
    v=center(np.asarray(v,float));f=np.fft.rfft(v);e=abs(f)**2
    e[1:-1]*=2
    total=max(float(e.sum()),1e-12);k=len(v)//24
    phase=np.exp(2j*np.pi*start_hour/24)*np.conj(f[k])/abs(f[k]) if abs(f[k])>1e-10 else 0j
    return dict(rms=float(np.sqrt(np.mean(v*v))),range90=float(np.quantile(v,.95)-np.quantile(v,.05)),
                daily_harmonic_share=float(e[k]/total),peak_clock_cos=float(phase.real),peak_clock_sin=float(phase.imag),
                faster_than6h_energy=float(e[np.fft.rfftfreq(len(v))>1/6].sum()/total))

def describe(v,hour,day,week):
    z=center(v);s=spectral(z,hour);dd=z.reshape(4,24)
    s.update(corr_last_day=corr(z,day),corr_week_template=corr(z,week),
             daily_nonrepeat_fraction=float(np.mean((dd-dd.mean(0))**2)/(np.mean(z*z)+1e-10)),
             last_minus_first_day_mean=float(dd[-1].mean()-dd[0].mean()))
    return s

def summary(x,p0,p1,ids,week):
    """Only past observations, model forecasts, calendar and a prefit historical table."""
    x=np.asarray(x,float);p0=np.asarray(p0,float);p1=np.asarray(p1,float);ids=np.asarray(ids,int)
    state=int(ids[0]);hour=state%24;days=x.reshape(4,24)
    past_ids=(state+np.arange(-96,0))%168
    past_week=week[past_ids];future_week=week[ids];day=center(np.tile(x[-24:],4))
    firstday_profile=center(np.tile(days.mean(0),4))
    f=np.fft.rfft(center(days),axis=-1)[:,1]
    hist=dict(mean=float(x.mean()),std=float(x.std()),last_day_mean=float(days[-1].mean()),
              last_day_std=float(days[-1].std()),daily_mean_change=float(days[-1].mean()-days[0].mean()),
              day_repeat_mse_scaled=float(np.mean((days[1:]-days[:-1])**2)/(x.var()+1e-10)),
              adjacent_day_corr=float(np.mean([corr(days[j],days[j+1]) for j in range(3)])),
              daily_amplitude_cv=float(days.std(1).std()/(days.std(1).mean()+1e-10)),
              daily_phase_consistency=float(abs(f.sum())/(abs(f).sum()+1e-10)),
              corr_historical_week=corr(x,past_week),
              historical_week_shape_mse=float(np.mean((center(x)-center(past_week))**2)),
              last_day_week_corr=corr(x[-24:],past_week[-24:]))
    hist.update({"history_"+k:v for k,v in spectral(x,hour).items() if k not in ["rms","range90"]})
    cal=dict(start_weekday=state//24,start_hour=hour,
             weekday_sin=float(np.sin(2*np.pi*(state//24)/7)),weekday_cos=float(np.cos(2*np.pi*(state//24)/7)),
             hour_sin=float(np.sin(2*np.pi*hour/24)),hour_cos=float(np.cos(2*np.pi*hour/24)),
             future_weekend_fraction=float(np.mean(ids//24>=5)))
    d0=center(p0);d1=center(p1)
    comparison=dict(correction_rms=float(np.sqrt(np.mean((d1-d0)**2))),
                    last_day_proxy_gain=float(np.mean((d0-day)**2)-np.mean((d1-day)**2)),
                    four_day_profile_proxy_gain=float(np.mean((d0-firstday_profile)**2)-np.mean((d1-firstday_profile)**2)),
                    week_template_proxy_gain=float(np.mean((d0-center(future_week))**2)-np.mean((d1-center(future_week))**2)))
    out=dict(history=hist,calendar=cal,original=describe(p0,hour,day,future_week),
             mixed=describe(p1,hour,day,future_week),comparison=comparison)
    # Both models receive these exact same rounded values; no unrounded classifier inputs.
    out={g:{k:round(float(v),3) for k,v in a.items()} for g,a in out.items()}
    assert all(np.isfinite(v) for a in out.values() for v in a.values())
    return out

def flatten(q):
    return np.array([v for group in q.values() for v in group.values()],float)

def prompt(q,context,swap):
    choices=("50% original plus 50% historical weekly shape","original TimeBase shape") if swap else ("original TimeBase shape","50% original plus 50% historical weekly shape")
    return (
      "Task: "+context+"\n"
      "Choose between two forecasts of the next96 hourly values. Their96-hour means are identical. "
      "The weekly template is the historical mean by hour-of-week, fitted before all current inputs. "
      "It is a calendar routine, not future observed demand. Both current occupancy and future weather/events are unknown. "
      "Use task context and the SAME past-only numerical summary below. "
      "All magnitudes use fixed training-standardized demand units. "
      "start_weekday is0=Monday to6=Sunday; start_hour is0..23. "
      "Correlation ranges -1..1. Daily phase consistency and harmonic shares range0..1. "
      "peak_clock_cos/sin describe the24h harmonic's peak clock; highfrequency share is variation faster than6hours. "
      "Proxy gains measure fit to PAST shapes and are not future errors. A positive proxy is not a guaranteed future improvement. "
      "Use the weekly mixture when the calendar pattern is likely to improve the actual future shape; otherwise retain the original. "
      "Avoid a default preference for smoother or smaller-amplitude forecasts.\n"
      "Shared summary:\n"+json.dumps(q,separators=(",",":"))+"\n"
      "Option A: "+choices[0]+".\nOption B: "+choices[1]+".\n"
      "Which option is expected to have lower actual next96-hour MSE? Reply with only A or B.")

def prepare(c,r):
    old=Path(c["reuse_run"]);oldc=rt.read(old/"config.json");assert oldc["pool"][4]=="MixWeek50"
    sources=dict(rt.read(old/"source.json")["assets"])
    for p,h in sources.items():assert rt.sha(p)==h
    week=np.load(old/"weekly_template.npy");sources[str(old/"weekly_template.npy")]=rt.sha(old/"weekly_template.npy")
    data={};schema=None;all_summary=[]
    for seed in c["seeds"]:
        for stage in ["train"]+c["stages"]:
            path=old/f"seed{seed}_{stage}.npz";sources[str(path)]=rt.sha(path);q=np.load(path)
            q={k:q[k] for k in q.files}
            q["p1"]=q["candidates"][:,4].copy()
            assert np.max(abs(q["p1"].mean(1)-q["base"].mean(1)))<1e-12
            summaries=[summary(x,a,b,ids,week) for x,a,b,ids in zip(q["x"],q["base"],q["p1"],q["ids"])]
            names=[g+"."+k for g,a in summaries[0].items() for k in a]
            if schema is None:schema=names
            assert names==schema
            q["features"]=np.stack([flatten(s) for s in summaries]);q["summary"]=summaries
            data[seed,stage]=q
            for j,s in enumerate(summaries):
                all_summary.append(dict(seed=seed,stage=stage,row=j,origin=int(q["origin"][j]),summary=s))
    rt.save(r/"shared_summaries.json",all_summary)
    rt.save(r/"source.json",dict(git=rt.git(),code_sha=rt.sha(__file__),sources=sources,
             features=schema,summary_has_no_target_argument="y" not in inspect.signature(summary).parameters,
             periods_previously_viewed=True,original_forecasts_reused=True))
    return data

def fit_classifier(c,r,data):
    out={}
    for seed in c["seeds"]:
        q=data[seed,"train"];x=q["features"]
        mu=x.mean(0);sd=np.maximum(x.std(0),.001);xx=np.c_[(x-mu)/sd,np.ones(len(x))]
        gain=q["errors"][:,0]-q["errors"][:,4];labels=(gain>0).astype(float)
        weight=abs(gain);weight/=max(weight.mean(),1e-12)
        reg=c["classifier"]["regularizer"]
        def obj(w):
            z=xx@w;p=w.copy();p[-1]=0
            return float(np.mean(weight*(np.logaddexp(0,z)-labels*z))+.5*reg*np.dot(p,p)), xx.T@(weight*(expit(z)-labels))/len(xx)+reg*p
        fit=minimize(obj,np.zeros(xx.shape[1]),jac=True,method="L-BFGS-B",
                     options=dict(maxiter=c["classifier"]["maxiter"],ftol=1e-10))
        assert fit.success,fit.message
        np.savez_compressed(r/f"classifier_seed{seed}.npz",mu=mu,sd=sd,coef=fit.x)
        out[seed]={}
        for stage in ["train"]+c["stages"]:
            z=data[seed,stage]["features"];score=np.c_[(z-mu)/sd,np.ones(len(z))]@fit.x
            out[seed][stage]=(score>0).astype(int)
        rt.save(r/f"classifier_seed{seed}.json",dict(parameters=len(fit.x),iterations=int(fit.nit),converged=bool(fit.success),
                 training_windows=len(x),beneficial_train=int(labels.sum()),train_only=True,
                 reg=reg,cost_weights_train_only=True,mean=mu.tolist(),sd=sd.tolist(),coef=fit.x.tolist()))
    return out

def fingerprint(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());v=p.detach().reshape(-1)
        h.update(torch.cat([v[:8],v[-8:]]).float().cpu().numpy().tobytes())
    return h.hexdigest()

def judge(c,r,data,device):
    req=[]
    for seed in c["seeds"]:
        for stage in c["stages"]:
            q=data[seed,stage]
            for j,s in enumerate(q["summary"]):
                for context,desc in c["contexts"].items():
                    for swap in [False,True]:
                        req.append(dict(seed=seed,stage=stage,row=j,context=context,swap=swap,text=prompt(s,desc,swap)))
    assert len(req)<=c["budget"]["max_requests"]
    texts=list(dict.fromkeys(v["text"] for v in req));lookup={s:j for j,s in enumerate(texts)}
    for p in req:p["unique"]=lookup[p["text"]]
    rt.save(r/"requests.json",req)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(c["model_path"],local_files_only=True)
    tokens=[tok.encode(tok.apply_chat_template([dict(role="user",content=s)],tokenize=False,
             add_generation_prompt=True,enable_thinking=False),add_special_tokens=False) for s in texts]
    assert max(map(len,tokens))<=c["budget"]["max_prompt_tokens"]
    rt.save(r/"tokenization.json",dict(requests=len(req),unique=len(tokens),max_tokens=max(map(len,tokens)),
             same_summaries=True,thinking=False))
    rt.setup(device,c["budget"]["gpu_cap_gib"])
    assert torch.cuda.mem_get_info(device)[0]>=c["budget"]["gpu_cap_gib"]*2**30
    tick=time.monotonic()
    llm=AutoModelForCausalLM.from_pretrained(c["model_path"],local_files_only=True,dtype=torch.bfloat16,
             device_map={"":device},attn_implementation="sdpa")
    llm.eval()
    for p in llm.parameters():p.requires_grad_(False)
    initial=fingerprint(llm);ids=[tok.encode(z,add_special_tokens=False)[0] for z in "AB"]
    assert len(set(ids))==2
    margins=[]
    with torch.inference_mode():
        for j,tt in enumerate(tokens):
            if time.monotonic()-tick>c["budget"]["max_llm_seconds"]:raise TimeoutError("frozen LLM time cap")
            batch=dict(input_ids=torch.tensor([tt],device=device))
            batch["attention_mask"]=torch.ones_like(batch["input_ids"])
            z=outputs(llm,batch,ids)[0]
            margins.append(float(z[1]-z[0]))
            if (j+1)%256==0:
                rt.save(r/"partial_margins.json",margins)
                print("JUDGE",j+1,len(tokens),round(time.monotonic()-tick,2),flush=True)
    rt.save(r/"margins.json",margins)
    assert initial==fingerprint(llm)
    assert all(not p.requires_grad and p.grad is None for p in llm.parameters())
    selections={};orders=[]
    for seed in c["seeds"]:
        selections[seed]={}
        for stage in c["stages"]:
            selections[seed][stage]={}
            for arm in c["contexts"]:
                qq=[p for p in req if p["seed"]==seed and p["stage"]==stage and p["context"]==arm]
                picks=[];bad=0
                for j in range(len(data[seed,stage]["x"])):
                    pair=[p for p in qq if p["row"]==j];assert len(pair)==2 and not pair[0]["swap"] and pair[1]["swap"]
                    normal=margins[pair[0]["unique"]];swapped=margins[pair[1]["unique"]]
                    picks.append(int((normal-swapped)/2>0));bad+=int((normal>0)!=(swapped<0))
                selections[seed][stage][arm]=np.array(picks)
                orders.append(dict(seed=seed,stage=stage,arm=arm,n=len(picks),inconsistent=bad,fraction=bad/len(picks)))
    csvsave(r/"order_diagnostics.csv",orders)
    rt.save(r/"teacher_completion.json",dict(seconds=time.monotonic()-tick,requests=len(req),unique=len(tokens),
        peak_gib=torch.cuda.max_memory_allocated(device)/2**30,model_fingerprint_before=initial,
        model_fingerprint_after=fingerprint(llm),llm_updates=0,TimeBase_updates=0,old_LoRA_loaded=False))
    del llm;torch.cuda.empty_cache()
    return selections

def evaluate(c,r,data,stat,llm):
    cases=[];rows=[];byseed=[]
    for seed in c["seeds"]:
        for stage in c["stages"]:
            q=data[seed,stage];n=len(q["x"])
            choices=dict(Base=np.zeros(n,int),FixedMix=np.ones(n,int),HistClassifier=stat[seed][stage],
                         Oracle=(q["errors"][:,4]<q["errors"][:,0]).astype(int),**llm[seed][stage])
            for arm,pick in choices.items():
                pred=np.where(pick[:,None]>0,q["p1"],q["base"]);err=((pred-q["y"])**2).mean(1)
                base=q["errors"][:,0];gain=base-err
                for j in range(n):
                    cases.append(dict(seed=seed,stage=stage,arm=arm,origin=int(q["origin"][j]),choice=int(pick[j]),
                        mse=float(err[j]),mae=float(abs(pred[j]-q["y"][j]).mean()),gain=float(gain[j]),
                        shape=float(np.mean((center(pred[j])-center(q["y"][j]))**2))))
                rows.append(dict(seed=seed,stage=stage,arm=arm,n=n,mse=float(err.mean()),
                    mae=float(abs(pred-q["y"]).mean()),relative_gain=float(gain.mean()/base.mean()),
                    gross_gain=float(np.maximum(gain,0).mean()),gross_harm=float(np.maximum(-gain,0).mean()),
                    corrected=int(pick.sum()),beneficial=int((gain>1e-8).sum()),harmful=int((gain< -1e-8).sum())))
    csvsave(r/"cases.csv",cases);csvsave(r/"by_seed_period.csv",rows)
    for seed in c["seeds"]:
        for arm in dict.fromkeys(x["arm"] for x in rows):
            a=[x for x in rows if x["seed"]==seed and x["arm"]==arm]
            byseed.append(dict(seed=seed,arm=arm,**{k:float(np.mean([x[k] for x in a])) for k in ["mse","mae","gross_gain","gross_harm"]}))
    csvsave(r/"macro_by_seed.csv",byseed)
    overall=[]
    for stage in c["stages"]+["macro"]:
        for arm in dict.fromkeys(x["arm"] for x in rows):
            a=[x for x in rows if x["arm"]==arm and (stage=="macro" or x["stage"]==stage)]
            overall.append(dict(stage=stage,arm=arm,n_per_seed=sum(x["n"] for x in a)//3,
                **{k:float(np.mean([x[k] for x in a])) for k in ["mse","mae","gross_gain","gross_harm"]},
                corrected_seed_total=sum(x["corrected"] for x in a),beneficial_seed_total=sum(x["beneficial"] for x in a),
                harmful_seed_total=sum(x["harmful"] for x in a)))
    csvsave(r/"summary.csv",overall)
    def get(stage,arm):return next(x for x in overall if x["stage"]==stage and x["arm"]==arm)
    wins=lambda s:next(x["mse"] for x in byseed if x["seed"]==s and x["arm"]=="QwenTask")<next(x["mse"] for x in byseed if x["seed"]==s and x["arm"]=="FixedMix")
    passed=all(get("macro","QwenTask")["mse"]<get("macro",a)["mse"] for a in ["Base","FixedMix","HistClassifier","QwenWrong"]) and all(wins(s) for s in c["seeds"]) and sum(get(t,"QwenTask")["mse"]<get(t,"FixedMix")["mse"] for t in c["stages"])>=3
    rt.save(r/"gate.json",dict(passed=bool(passed),rule=c["gate"],independent_confirmation_executed=False,KD_executed=False))
    for a in overall:
        if a["stage"]=="macro":print("RESULT",json.dumps(a),flush=True)
    print("GATE",passed,flush=True)

def preflight(c):
    old=Path(c["reuse_run"]);q=np.load(old/"seed2024_train.npz");week=np.load(old/"weekly_template.npy")
    s=summary(q["x"][0],q["base"][0],q["candidates"][0,4],q["ids"][0],week)
    assert np.isfinite(flatten(s)).all()
    assert "y" not in inspect.signature(summary).parameters
    assert "truth" not in json.dumps(s)
    a=json.loads(json.dumps(s))
    assert np.array_equal(flatten(a),flatten(s))
    p=prompt(s,c["contexts"]["QwenTask"],False);n=prompt(s,c["contexts"]["QwenTask"],True)
    assert p!=n and json.dumps(s,separators=(",",":")) in p and json.dumps(s,separators=(",",":")) in n
    print("PREFLIGHT",len(flatten(s)),"features; exact shared quantization; no target argument; swap identity checked")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda:0");ap.add_argument("--preflight",action="store_true")
    a=ap.parse_args();c=rt.read("docs/chronogeom/temporal_reliability04_config.json")
    if a.preflight:preflight(c);return
    r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);rt.save(r/"config.json",c)
    shutil.copy2(__file__,r/Path(__file__).name)
    tick=time.monotonic();torch.set_num_threads(2)
    try:
        data=prepare(c,r);stat=fit_classifier(c,r,data)
        print("PREPARED",[(seed,stage,len(q["x"])) for (seed,stage),q in data.items()],flush=True)
        llm=judge(c,r,data,a.device);evaluate(c,r,data,stat,llm)
        sources=rt.read(r/"source.json")["sources"];assert all(rt.sha(p)==h for p,h in sources.items())
        rt.save(r/"completion.json",dict(seconds=time.monotonic()-tick,old_assets_unchanged=True,
            no_LLM_or_TimeBase_training=True,new_encoder=False,KD=False))
        assert time.monotonic()-tick<c["budget"]["max_workflow_seconds"]
        print("COMPLETE",flush=True)
    except BaseException:
        rt.save(r/"failure.json",dict(traceback=traceback.format_exc()));raise
if __name__=="__main__":main()
