"""Qwen-only Metro tuning; published22 comparators are immutable."""
import os
os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse, copy, itertools, sys, time
from pathlib import Path
import numpy as np
import torch
from exp.exp_chronogeom_event_knowledge19 import Head, forward, scores, read, save, sha, git
from exp.exp_chronogeom_event_confirm21 import dataset
from exp.exp_chronogeom_residual03 import numeric_summary

def center(x):
    return x-x.mean(1,keepdim=True)

def run(c, config):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    start=time.monotonic()
    r=Path(c["run"]); r.mkdir(parents=True,exist_ok=False)
    ref=Path(c["reference"]); src=Path(c["teacher_run"])
    freeze=read("docs/chronogeom/FROZEN_COMPARATORS_22.json")
    assert str(ref)==freeze["run"] and c["seeds"]==freeze["seeds"]
    frozen_files=[dict(path=str(ref/f),sha=v) for f,v in freeze["artifact_sha256"].items()]
    assets=frozen_files+freeze["base_assets"]
    for f in ["config.json","normalizers.json","knowledge.npz","data.npz","predictions.npz","student_checkpoints.pt"]:
        assets.append(dict(path=str(src/f),sha=sha(src/f)))
    assets.append(dict(path="docs/chronogeom/FROZEN_COMPARATORS_22.json",sha=sha("docs/chronogeom/FROZEN_COMPARATORS_22.json")))
    for a in assets: assert sha(a["path"])==a["sha"]
    save(r/"config.json",c)
    save(r/"input_assets.json",assets)
    save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),python=sys.version,torch=torch.__version__,numpy=np.__version__))
    cfg=read(src/"config.json")
    folder=r/"Metro";folder.mkdir()
    packs,grid,raw,common,kind,h,hist=dataset(cfg,folder)
    cached=np.load(src/"predictions.npz")
    normalizers=read(src/"normalizers.json")
    K=np.load(src/"knowledge.npz")["Fourier"]
    old22=torch.load(ref/"all_trial_checkpoints.pt",map_location="cpu",weights_only=True)["Metro"]["QwenTaskKD"]
    old21=torch.load(src/"student_checkpoints.pt",map_location="cpu",weights_only=True)
    z={};base={};teacher={};norm={};valnorm={};batches={}
    target=torch.tensor(packs["fit"]["y"],dtype=torch.float64)
    def features(seed,split):
        p=packs[split]
        assert np.array_equal(p["origin"],cached[f"{seed}_{split}_origin"])
        assert np.array_equal(p["y"],cached[f"{seed}_{split}_y"])
        bp=torch.tensor(cached[f"{seed}_{split}_FrozenTimeBase"],dtype=torch.float32)
        pp=normalizers[str(seed)]["Fourier"];ix=p["future"]
        ft=np.concatenate([numeric_summary(p["x"],bp).numpy(),raw[ix],common[ix]],-1)
        zz=torch.tensor(np.concatenate([(ft-np.array(pp["mean"]))/np.array(pp["sd"]),(K[ix]-np.array(pp["knowledge_mean"]))/pp["knowledge_rms"]],-1),dtype=torch.float32)
        return zz,bp
    for seed in c["seeds"]:
        z[seed]={};base[seed]={}
        for split in ["fit","val"]:
            z[seed][split],base[seed][split]=features(seed,split)
        teacher[seed]=center(torch.tensor(cached[f"{seed}_fit_Teacher_QwenTask"],dtype=torch.float64)).detach()
        norm[seed]=scores(base[seed]["fit"].numpy(),packs["fit"]["y"])
        valnorm[seed]=scores(base[seed]["val"].numpy(),packs["val"]["y"])
        batches[seed]=np.random.default_rng(seed).integers(0,len(target),size=(c["steps"],c["batch"]))
    states={};curves=[];options={};fits=[];anchor22=0;anchor21=0
    for lr,weight,kdstart in itertools.product(c["lrs"],c["weights"],c["starts"]):
        key=f"lr{lr:g}_kd{weight:g}_start{kdstart}"
        states[key]={};trialrows=[]
        for seed in c["seeds"]:
            head=Head(seed);assert sum(p.numel() for p in head.parameters())==705
            opt=torch.optim.Adam(head.parameters(),lr=lr)
            initial=forward(head,z[seed]["fit"],base[seed]["fit"])
            assert torch.equal(initial,base[seed]["fit"].double())
            (initial-target).square().mean().backward()
            gradient=float(head.net[-1].weight.grad.abs().sum());assert gradient>0
            head.zero_grad();checkpoints={}
            for step in range(c["steps"]+1):
                if step%c["check_every"]==0:
                    with torch.no_grad(): pv=forward(head,z[seed]["val"],base[seed]["val"]).numpy()
                    met=scores(pv,packs["val"]["y"])
                    row=dict(recipe=key,seed=seed,step=step,**met);curves.append(row);trialrows.append(row)
                    checkpoints[step]=copy.deepcopy(head.state_dict())
                    if lr==.002 and weight in [.03,.1,.3]:
                        old=old22[f"kd{weight:g}_start{kdstart}"][seed][step]
                        assert all(torch.equal(v,old[k]) for k,v in head.state_dict().items()),("old22",key,seed,step)
                        anchor22+=1
                    if lr==.001 and weight==.1 and kdstart==0:
                        old=old21[seed]["QwenTaskKD"][step]
                        assert all(torch.equal(v,old[k]) for k,v in head.state_dict().items()),("old21",key,seed,step)
                        anchor21+=1
                if step==c["steps"]:break
                ids=batches[seed][step]
                pred=forward(head,z[seed]["fit"][ids],base[seed]["fit"][ids])
                err=pred-target[ids]
                loss=.5*(err.square().mean()/norm[seed]["mse"]+err.abs().mean()/norm[seed]["mae"])
                if step>=kdstart:
                    aux=(center(pred)-teacher[seed][ids]).square().mean()
                    loss=loss+weight*aux/norm[seed]["mse"]
                opt.zero_grad();loss.backward();opt.step()
                assert time.monotonic()-start<c["fit_seconds"],"CPU budget exceeded"
            states[key][seed]=checkpoints
            fits.append(dict(recipe=key,seed=seed,steps=c["steps"],parameters=705,initial_gradient=gradient))
        choices=[]
        for step in range(0,c["steps"]+1,c["check_every"]):
            vals=[.5*(q["mse"]/valnorm[q["seed"]]["mse"]+q["mae"]/valnorm[q["seed"]]["mae"]) for q in trialrows if q["step"]==step]
            assert len(vals)==3
            choices.append((float(np.mean(vals)),step))
        value,step=min(choices)
        options[key]=dict(score=value,step=step,lr=lr,weight=weight,start=kdstart,actual_KD_updates=max(0,step-kdstart))
        print("INNER",key,options[key],flush=True)
    key=min(options,key=lambda k:(options[k]["score"],options[k]["weight"],options[k]["lr"],options[k]["start"],options[k]["step"]))
    chosen=dict(recipe=key,**options[key])
    save(r/"selection.json",dict(chosen=chosen,candidates=options,outer_candidates_evaluated=False,controls_retrained=0,selection="Metro train-internal validation; common three-seed recipe and step"))
    lock=sha(r/"selection.json")
    save(r/"evaluation_lock.json",dict(selection_sha=lock,git=git(),elapsed_before_evaluation=time.monotonic()-start))
    save(r/"curves.json",curves);save(r/"fits.json",fits)
    torch.save(states,r/"all_qwen_trial_checkpoints.pt")
    # Only the single inner-selected Qwen recipe is evaluated on viewed outer data.
    reference=np.load(ref/"predictions.npz")
    selected={};predictions={};metrics=[];maxmean=0.;maxlevel=0.;replays=0
    controls=freeze["controls"]
    oldmetrics=read(ref/"metrics.json")
    for seed in c["seeds"]:
        state=states[key][seed][chosen["step"]];selected[seed]=state
        for split,label in [("fit","fit"),("val","val"),("confirm","eval")]:
            zz,bp=features(seed,split)
            head=Head(seed);head.load_state_dict(state)
            with torch.no_grad():pred=forward(head,zz,bp).numpy()
            replay=Head(seed);replay.load_state_dict(copy.deepcopy(state))
            with torch.no_grad():assert np.array_equal(pred,forward(replay,zz,bp).numpy());replays+=1
            root=f"Metro_{seed}_{label}"
            y=packs[split]["y"];assert np.array_equal(y,reference[root+"_y"])
            predictions[root+"_y"]=y;predictions[root+"_origin"]=packs[split]["origin"]
            outputs={a:reference[root+"_"+a] for a in controls}
            outputs["QwenTaskKD22"]=reference[root+"_QwenTaskKD"]
            outputs["QwenTaskKD23"]=pred
            maxmean=max(maxmean,float(abs((pred-bp.numpy()).mean(1)).max()))
            baselevel=scores(bp.numpy(),y)["level"]
            for arm,p in outputs.items():
                met=scores(p,y);metrics.append(dict(domain="Metro",seed=seed,split=label,arm=arm,**met))
                predictions[root+"_"+arm]=p
                assert abs(met["mse"]-met["level"]-met["shape"])<1e-12
                if arm in controls:
                    previous=next(q for q in oldmetrics if q["domain"]=="Metro" and q["seed"]==seed and q["split"]==label and q["arm"]==arm)
                    assert all(abs(met[k]-previous[k])<1e-15 for k in met),("frozen metric changed",arm,seed,split)
                if arm=="QwenTaskKD23":maxlevel=max(maxlevel,abs(met["level"]-baselevel))
    torch.save(selected,r/"selected_qwen_students.pt")
    np.savez_compressed(r/"predictions.npz",**predictions);save(r/"metrics.json",metrics)
    arms=controls+["QwenTaskKD22","QwenTaskKD23"]
    summary={s:{a:{k:float(np.mean([q[k] for q in metrics if q["split"]==s and q["arm"]==a])) for k in ["mse","mae","level","shape"]} for a in arms} for s in ["fit","val","eval"]}
    intervals=[];rng=np.random.default_rng(2024)
    origins=predictions["Metro_2024_eval_origin"];clusters=(origins-origins.min())//336;unique=np.unique(clusters)
    for arm in ["NoKD","FourierKD","HistoricalKD","ShuffleKD","QwenTaskKD22"]:
        for metric in ["mse","mae"]:
            delta=[]
            for seed in c["seeds"]:
                root=f"Metro_{seed}_eval";y=predictions[root+"_y"]
                a=predictions[root+"_QwenTaskKD23"]-y;b=predictions[root+"_"+arm]-y
                delta.append(((a*a-b*b) if metric=="mse" else (abs(a)-abs(b))).mean((1,2)))
            dd=np.mean(delta,0);boot=[]
            for _ in range(2000):
                ids=np.concatenate([np.flatnonzero(clusters==k) for k in rng.choice(unique,len(unique),replace=True)])
                boot.append(float(dd[ids].mean()))
            intervals.append(dict(reference=arm,metric=metric,delta=float(dd.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),blocks=len(unique),note="Viewed development; seeds averaged then paired14day block resampling, not independent confirmation."))
    assert anchor22==126 and anchor21==21 and maxmean<1e-10 and maxlevel<1e-10
    assert sha(r/"selection.json")==lock
    for a in assets:assert sha(a["path"])==a["sha"]
    save(r/"summary.json",dict(metrics=summary,paired_intervals=intervals,selection=chosen,Capital_Qwen_unchanged22=True))
    save(r/"verification.json",dict(fits=len(fits),total_updates=len(fits)*c["steps"],parameters=705,anchor22=anchor22,anchor21=anchor21,selected_replays=replays,mean_error_max=maxmean,level_difference_max=maxlevel,frozen_control_metric_equality=True,all_source_SHA_unchanged=True,selection_lock_unchanged=True,controls_trained=0,LLM_calls=0,TimeBase_updates=0))
    save(r/"status.json",dict(seconds=time.monotonic()-start,git=git(),fits=len(fits),complete=True))
    print("CHOSEN",chosen,flush=True)
    print("EVAL",summary["eval"],flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args()
    run(read(a.config),a.config)
