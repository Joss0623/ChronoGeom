"""Rolling multi-time-block inner selection for unchanged Qwen Shape KD in TSLib."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0",CUBLAS_WORKSPACE_CONFIG=":4096:8")
import argparse,copy,itertools,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from exp.exp_chronogeom_event_knowledge19 import Head,forward,scores,read,save,sha,git
from exp.exp_chronogeom_event_confirm21 import dataset
from exp.exp_chronogeom_residual03 import numeric_summary

def center(x):return x-x.mean(1,keepdim=True)

def run(c,config):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.04)
    device=torch.device("cuda:0");torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False)
    ref=Path(c["reference"]);src=Path(c["teacher_run"]);previous=Path(c["previous"])
    freeze=read("docs/chronogeom/FROZEN_COMPARATORS_22.json")
    assert str(ref)==freeze["run"] and c["seeds"]==freeze["seeds"]
    assets=[dict(path=str(ref/f),sha=v) for f,v in freeze["artifact_sha256"].items()]+freeze["base_assets"]
    for f in ["config.json","normalizers.json","knowledge.npz","data.npz","predictions.npz","teacher_checkpoints.pt","teacher_selection.json"]:
        assets.append(dict(path=str(src/f),sha=sha(src/f)))
    for f in ["selected_qwen_students.pt","predictions.npz","metrics.json","selection.json"]:
        assets.append(dict(path=str(previous/f),sha=sha(previous/f)))
    assets.append(dict(path="docs/chronogeom/FROZEN_COMPARATORS_22.json",sha=sha("docs/chronogeom/FROZEN_COMPARATORS_22.json")))
    for a in assets:assert sha(a["path"])==a["sha"]
    save(r/"config.json",c);save(r/"input_assets.json",assets)
    save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),gpu_uuid=c["gpu_uuid"],tf32=False))
    cfg=read(src/"config.json");folder=r/"Metro";folder.mkdir()
    packs,grid,raw,common,kind,h,hist=dataset(cfg,folder)
    cached=np.load(src/"predictions.npz");normalizers=read(src/"normalizers.json");K=np.load(src/"knowledge.npz")["Fourier"]
    nfit=len(packs["fit"]["y"]);assert nfit==73
    allfuture=np.concatenate([packs["fit"]["future"],packs["val"]["future"]])
    allorigin=np.concatenate([packs["fit"]["origin"],packs["val"]["origin"]])
    target=torch.tensor(packs["fit"]["y"],dtype=torch.float64,device=device)
    targetall=torch.cat([target,torch.tensor(packs["val"]["y"],dtype=torch.float64,device=device)])
    folds=[];foldmeta=[]
    for i,((a,b),(nt,nv)) in enumerate(zip(c["blocks"],c["expected_counts"])):
        lo=grid.get_loc(pd.Timestamp(a));hi=grid.get_loc(pd.Timestamp(b))
        tr=np.flatnonzero(allfuture[:,-1]<lo)
        va=np.flatnonzero((allorigin>=lo)&(allfuture[:,-1]<hi))
        assert len(tr)==nt and len(va)==nv
        assert np.intersect1d(allfuture[tr].ravel(),allfuture[va].ravel()).size==0
        assert allfuture[tr].max()<allfuture[va].min()
        for old in folds:assert np.intersect1d(allfuture[old["val_cpu"]].ravel(),allfuture[va].ravel()).size==0
        folds.append(dict(index=i,train=torch.as_tensor(tr,device=device),val=torch.as_tensor(va,device=device),val_cpu=va))
        foldmeta.append(dict(fold=i,start=a,end=b,train_windows=len(tr),validation_windows=len(va),train_origins=allorigin[tr].tolist(),validation_origins=allorigin[va].tolist(),maximum_fit_target=str(grid[allfuture[tr].max()]),minimum_validation_target=str(grid[allfuture[va].min()]),maximum_validation_target=str(grid[allfuture[va].max()]),validation_unique_target_hours=len(np.unique(allfuture[va])),target_overlap=0))
    assert torch.equal(folds[0]["train"],torch.arange(nfit,device=device))
    save(r/"folds.json",foldmeta)
    z={};base={};teacher={};validationnorm={}
    def features(seed,split):
        p=packs[split]
        assert np.array_equal(p["origin"],cached[f"{seed}_{split}_origin"]) and np.array_equal(p["y"],cached[f"{seed}_{split}_y"])
        bp=torch.tensor(cached[f"{seed}_{split}_FrozenTimeBase"],dtype=torch.float32)
        pp=normalizers[str(seed)]["Fourier"];ix=p["future"]
        ft=np.concatenate([numeric_summary(p["x"],bp).numpy(),raw[ix],common[ix]],-1)
        zz=torch.tensor(np.concatenate([(ft-np.array(pp["mean"]))/np.array(pp["sd"]),(K[ix]-np.array(pp["knowledge_mean"]))/pp["knowledge_rms"]],-1),dtype=torch.float32)
        return zz.to(device),bp.to(device)
    for seed in c["seeds"]:
        z[seed]={};base[seed]={}
        for split in ["fit","val"]:z[seed][split],base[seed][split]=features(seed,split)
        z[seed]["all"]=torch.cat([z[seed]["fit"],z[seed]["val"]])
        base[seed]["all"]=torch.cat([base[seed]["fit"],base[seed]["val"]])
        teacher[seed]=center(torch.tensor(np.concatenate([cached[f"{seed}_fit_Teacher_QwenTask"],cached[f"{seed}_val_Teacher_QwenTask"]]),dtype=torch.float64,device=device)).detach()
        for fold in folds:
            ids=fold["val"];validationnorm[fold["index"],seed]=scores(base[seed]["all"][ids].cpu().numpy(),targetall[ids].cpu().numpy())
    states={};curves=[];fits=[];options={}
    def train(seed,lr,weight,trainids,validids=None,recipe="",fold="final",record=True):
        head=Head(seed).to(device);assert sum(p.numel() for p in head.parameters())==705 and all(p.is_cuda for p in head.parameters())
        zz=z[seed]["all"][trainids];bp=base[seed]["all"][trainids];tt=targetall[trainids];soft=teacher[seed][trainids]
        norm=scores(bp.cpu().numpy(),tt.cpu().numpy())
        initial=forward(head,zz,bp);assert torch.equal(initial,bp.double())
        (initial-tt).square().mean().backward();grad=float(head.net[-1].weight.grad.abs().sum());assert grad>0;head.zero_grad()
        batches=torch.as_tensor(np.random.default_rng(seed).integers(0,len(trainids),size=(c["steps"],c["batch"])),device=device)
        opt=torch.optim.Adam(head.parameters(),lr=lr,foreach=False);out={};rows=[]
        for step in range(c["steps"]+1):
            if step%c["check_every"]==0:
                out[step]={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                if validids is not None:
                    with torch.no_grad():pv=forward(head,z[seed]["all"][validids],base[seed]["all"][validids]).cpu().numpy()
                    met=scores(pv,targetall[validids].cpu().numpy())
                    rows.append(dict(recipe=recipe,fold=fold,seed=seed,step=step,**met))
            if step==c["steps"]:break
            ids=batches[step];pred=forward(head,zz[ids],bp[ids]);err=pred-tt[ids]
            loss=.5*(err.square().mean()/norm["mse"]+err.abs().mean()/norm["mae"])
            aux=(center(pred)-soft[ids]).square().mean()
            loss=loss+weight*aux/norm["mse"]
            opt.zero_grad();loss.backward();opt.step()
            assert time.monotonic()-start<c["fit_seconds"],"bounded CUDA runtime exceeded"
        if record:fits.append(dict(recipe=recipe,fold=fold,seed=seed,steps=c["steps"],train_windows=len(trainids),parameters=705,device=str(device),initial_gradient=grad))
        return out,rows
    def aggregate(rows):
        cell={}
        for q in rows:
            bn=validationnorm[q["fold"],q["seed"]]
            cell[q["fold"],q["seed"]]=.5*(q["mse"]/bn["mse"]+q["mae"]/bn["mae"])
        assert len(cell)==9
        block=[float(np.mean([cell[f,s] for s in c["seeds"]])) for f in range(3)]
        seedscore=[float(np.mean([cell[f,s] for f in range(3)])) for s in c["seeds"]]
        return dict(score=float(np.mean(block)),worst_block=max(block),worst_seed=max(seedscore),block_scores=block,seed_scores=seedscore)
    for lr,weight in itertools.product(c["lrs"],c["weights"]):
        key=f"lr{lr:g}_kd{weight:g}";states[key]={};trialrows=[]
        for fold in folds:
            f=fold["index"];states[key][f]={}
            for seed in c["seeds"]:
                out,rows=train(seed,lr,weight,fold["train"],fold["val"],key,f)
                states[key][f][seed]=out;curves.extend(rows);trialrows.extend(rows)
        candidates=[dict(step=step,**aggregate([q for q in trialrows if q["step"]==step])) for step in range(0,c["steps"]+1,c["check_every"])]
        best=min(candidates,key=lambda q:(q["score"],q["worst_block"],q["worst_seed"],q["step"]))
        options[key]=dict(lr=lr,weight=weight,batch=32,start=0,actual_KD_updates=best["step"],**best)
        print("CV",key,options[key],flush=True)
    key=min(options,key=lambda k:(options[k]["score"],options[k]["worst_block"],options[k]["worst_seed"],options[k]["weight"],options[k]["lr"],options[k]["step"]))
    chosen=dict(recipe=key,**options[key])
    save(r/"cv_selection.json",dict(chosen=chosen,candidates=options,folds=foldmeta,outer_scores_seen=False,final_weights_trained=False,selection="Equal mean3chronological blocks x3seeds normalized joint risk; ONE shared recipe AND step"))
    cvlock=sha(r/"cv_selection.json");save(r/"curves.json",curves)
    torch.save(states,r/"fold_checkpoints.pt")
    final={};deterministic_repeats=0;fold0_repeats=0
    trainids=torch.arange(nfit,device=device)
    for seed in c["seeds"]:
        out,_=train(seed,chosen["lr"],chosen["weight"],trainids,recipe=key)
        final[seed]=out
        for step,sd in out.items():
            assert all(torch.equal(v,states[key][0][seed][step][k]) for k,v in sd.items()),("final must reset to original73 only",seed,step)
            fold0_repeats+=1
    for seed in c["seeds"]:
        out,_=train(seed,chosen["lr"],chosen["weight"],trainids,recipe=key,record=False)
        for step,sd in out.items():
            assert all(torch.equal(v,final[seed][step][k]) for k,v in sd.items()),("GPU final deterministic replay",seed,step)
            deterministic_repeats+=1
    torch.save(final,r/"final_checkpoints.pt")
    save(r/"fits.json",fits)
    assert sha(r/"cv_selection.json")==cvlock
    save(r/"selection.json",dict(chosen=chosen,cv_selection_sha=cvlock,final_states_sha=sha(r/"final_checkpoints.pt"),final_gradient_windows=nfit,controls_retrained=0,outer_candidates_evaluated=False))
    lock=sha(r/"selection.json")
    save(r/"evaluation_lock.json",dict(selection_sha=lock,cv_selection_sha=cvlock,git=git(),elapsed_before_outer=time.monotonic()-start))
    states={key:final}
    # Only the single inner-selected Qwen recipe is evaluated on viewed outer data.
    reference=np.load(ref/"predictions.npz")
    selected={};predictions={};metrics=[];maxmean=0.;maxlevel=0.;replays=0
    controls=freeze["controls"]
    oldmetrics=read(ref/"metrics.json")
    oldpred=np.load(previous/"predictions.npz")
    for seed in c["seeds"]:
        state=states[key][seed][chosen["step"]];selected[seed]=state
        for split,label in [("fit","fit"),("val","val"),("confirm","eval")]:
            zz,bp=features(seed,split)
            head=Head(seed).to(device);head.load_state_dict(state)
            with torch.no_grad():pred=forward(head,zz,bp).cpu().numpy()
            replay=Head(seed).to(device);replay.load_state_dict(copy.deepcopy(state))
            with torch.no_grad():assert np.array_equal(pred,forward(replay,zz,bp).cpu().numpy());replays+=1
            root=f"Metro_{seed}_{label}"
            y=packs[split]["y"];assert np.array_equal(y,reference[root+"_y"])
            predictions[root+"_y"]=y;predictions[root+"_origin"]=packs[split]["origin"]
            outputs={a:reference[root+"_"+a] for a in controls}
            outputs["QwenTaskKD22"]=reference[root+"_QwenTaskKD"]
            outputs["QwenTaskKD23"]=oldpred[root+"_QwenTaskKD23"]
            outputs["QwenTaskKD25"]=pred
            maxmean=max(maxmean,float(abs((pred-bp.cpu().numpy()).mean(1)).max()))
            baselevel=scores(bp.cpu().numpy(),y)["level"]
            for arm,p in outputs.items():
                met=scores(p,y);metrics.append(dict(domain="Metro",seed=seed,split=label,arm=arm,**met))
                predictions[root+"_"+arm]=p
                assert abs(met["mse"]-met["level"]-met["shape"])<1e-12
                if arm in controls:
                    previous=next(q for q in oldmetrics if q["domain"]=="Metro" and q["seed"]==seed and q["split"]==label and q["arm"]==arm)
                    assert all(abs(met[k]-previous[k])<1e-15 for k in met),("frozen metric changed",arm,seed,split)
                if arm=="QwenTaskKD25":maxlevel=max(maxlevel,abs(met["level"]-baselevel))
    torch.save(selected,r/"selected_qwen_students.pt")
    np.savez_compressed(r/"predictions.npz",**predictions);save(r/"metrics.json",metrics)
    arms=controls+["QwenTaskKD22","QwenTaskKD23","QwenTaskKD25"]
    summary={s:{a:{k:float(np.mean([q[k] for q in metrics if q["split"]==s and q["arm"]==a])) for k in ["mse","mae","level","shape"]} for a in arms} for s in ["fit","val","eval"]}
    intervals=[];rng=np.random.default_rng(2024)
    origins=predictions["Metro_2024_eval_origin"];clusters=(origins-origins.min())//336;unique=np.unique(clusters)
    for arm in ["NoKD","FourierKD","HistoricalKD","ShuffleKD","QwenTaskKD22","QwenTaskKD23"]:
        for metric in ["mse","mae"]:
            delta=[]
            for seed in c["seeds"]:
                root=f"Metro_{seed}_eval";y=predictions[root+"_y"]
                a=predictions[root+"_QwenTaskKD25"]-y;b=predictions[root+"_"+arm]-y
                delta.append(((a*a-b*b) if metric=="mse" else (abs(a)-abs(b))).mean((1,2)))
            dd=np.mean(delta,0);boot=[]
            for _ in range(2000):
                ids=np.concatenate([np.flatnonzero(clusters==k) for k in rng.choice(unique,len(unique),replace=True)])
                boot.append(float(dd[ids].mean()))
            intervals.append(dict(reference=arm,metric=metric,delta=float(dd.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),blocks=len(unique),note="Viewed development; seeds averaged then paired14day block resampling, not independent confirmation."))
    assert maxmean<1e-10 and maxlevel<1e-10
    assert sha(r/"selection.json")==lock and sha(r/"cv_selection.json")==cvlock
    for a in assets:assert sha(a["path"])==a["sha"]
    save(r/"summary.json",dict(metrics=summary,paired_intervals=intervals,selection=chosen,Capital_Qwen_unchanged22=True))
    save(r/"verification.json",dict(fits=len(fits),total_updates=len(fits)*c["steps"],parameters=705,deterministic_checkpoint_repeats=deterministic_repeats,final_equals_fold0_checkpoints=fold0_repeats,final_fit_windows=73,fold_target_disjointness=True,training_device="cuda",peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,selected_replays=replays,mean_error_max=maxmean,level_difference_max=maxlevel,frozen_control_metric_equality=True,all_source_SHA_unchanged=True,selection_lock_unchanged=True,controls_trained=0,LLM_calls=0,TimeBase_updates=0))
    save(r/"status.json",dict(seconds=time.monotonic()-start,git=git(),fits=len(fits),repeat_fits=3,fold_fits=54,final_fits=3,device=torch.cuda.get_device_name(),peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,complete=True))
    print("CHOSEN",chosen,flush=True)
    print("EVAL",summary["eval"],flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args()
    run(read(a.config),a.config)
