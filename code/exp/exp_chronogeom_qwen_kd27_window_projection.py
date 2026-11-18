"""Fixed23 recipe with per-window forecast-shape gradient compatibility."""
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
    for f in ["config.json","fold_checkpoints.pt","curves.json"]:
        assets.append(dict(path=str(Path(c["cv_reference"])/f),sha=sha(Path(c["cv_reference"])/f)))
    for f in ["raw_anchor_checkpoints.pt","predictions.npz","config.json","gradient_log.json"]:
        assets.append(dict(path=str(Path(c["anchor_run"])/f),sha=sha(Path(c["anchor_run"])/f)))
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

    # Fixed method experiment: preserve23 settings rather than search.
    def project_teacher(gs,gd):
        s64=gs.double();d64=gd.double()
        dot=(s64*d64).sum(-1,keepdim=True);n2=(s64*s64).sum(-1,keepdim=True)
        coef=torch.where((dot<0)&(n2>0),dot/n2.clamp_min(torch.finfo(torch.float64).tiny),torch.zeros_like(dot))
        out=(d64-coef*s64).to(gd.dtype)
        return out,dot.squeeze(-1),(s64*out.double()).sum(-1)
    checks=[]
    for name,aa,bb,expected in [
        ("aligned",[1.,0.],[2.,3.],[2.,3.]),
        ("orthogonal",[1.,0.],[0.,2.],[0.,2.]),
        ("opposite",[1.,2.],[-2.,-4.],[0.,0.]),
        ("mixed",[1.,0.],[-2.,3.],[0.,3.])]:
        aa=torch.tensor(aa,device=device);bb=torch.tensor(bb,device=device)
        out,before,after=project_teacher(aa,bb)
        assert torch.equal(out,torch.tensor(expected,device=device)) and float(after)>=0
        checks.append(dict(case=name,before=float(before),after=float(after)))
    aa=torch.tensor([[1.,0.],[1.,0.]],device=device);bb=torch.tensor([[-2.,3.],[2.,-3.]],device=device)
    out,bef,aft=project_teacher(aa,bb)
    assert torch.equal(out,torch.tensor([[0.,3.],[2.,-3.]],device=device)) and bool((aft>=0).all())
    checks.append(dict(case="per_window_matrix",before=bef.cpu().tolist(),after=aft.cpu().tolist()))
    save(r/"projection_math_checks.json",checks)
    oldpred=np.load(previous/"predictions.npz")
    diagnosis=[]
    for seed in c["seeds"]:
        yy=cached[f"{seed}_fit_y"];bp=cached[f"{seed}_fit_FrozenTimeBase"];tt=cached[f"{seed}_fit_Teacher_QwenTask"];ss=oldpred[f"Metro_{seed}_fit_QwenTaskKD23"]
        def C(x):return x-x.mean(1,keepdims=True)
        d=C(tt-bp);rr=C(yy-bp);dot=(d*rr).sum((1,2))
        diagnosis.append(dict(seed=seed,train_windows=len(yy),teacher_vs_base_wrong_direction_fraction=float((dot<=0).mean()),teacher_vs_student_forecast_gradient_conflict_fraction=float(((C(ss-yy)*C(ss-tt)).sum((1,2))<0).mean()),teacher_fit_shape_mse=float((C(tt-yy)**2).mean()),student23_fit_shape_mse=float((C(ss-yy)**2).mean()),scope="Training only; output-space gradients, not parameter-gradient proof."))
    save(r/"train_teacher_diagnosis.json",diagnosis)
    cvref=Path(c["cv_reference"])
    rawold=torch.load(cvref/"fold_checkpoints.pt",map_location="cpu",weights_only=True)["lr0.004_kd1"]
    rawcurves=[q for q in read(cvref/"curves.json") if q["recipe"]=="lr0.004_kd1"]
    states={};curves=[];fits=[];gradient_log=[]
    def train(seed,trainids,validids=None,arm="WindowProjectedQwenKD",fold=0,record=True):
        head=Head(seed).to(device);params=list(head.parameters())
        assert sum(p.numel() for p in params)==705 and all(p.is_cuda for p in params)
        zz=z[seed]["all"][trainids];bp=base[seed]["all"][trainids];tt=targetall[trainids];soft=teacher[seed][trainids]
        norm=scores(bp.cpu().numpy(),tt.cpu().numpy())
        initial=forward(head,zz,bp);assert torch.equal(initial,bp.double())
        (initial-tt).square().mean().backward();grad=float(head.net[-1].weight.grad.abs().sum());assert grad>0;head.zero_grad()
        batches=torch.as_tensor(np.random.default_rng(seed).integers(0,len(trainids),size=(c["steps"],c["batch"])),device=device)
        opt=torch.optim.Adam(params,lr=c["lr"],foreach=False)
        out={};rows=[]
        for step in range(c["steps"]+1):
            if step%c["check_every"]==0:
                out[step]={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                if validids is not None:
                    with torch.no_grad():pv=forward(head,z[seed]["all"][validids],base[seed]["all"][validids]).cpu().numpy()
                    met=scores(pv,targetall[validids].cpu().numpy())
                    rows.append(dict(arm=arm,fold=fold,seed=seed,step=step,**met))
            if step==c["steps"]:break
            ids=batches[step];pred=forward(head,zz[ids],bp[ids]);err=pred-tt[ids]
            sup=.5*(err.square().mean()/norm["mse"]+err.abs().mean()/norm["mae"])
            kd=(center(pred)-soft[ids]).square().mean()/norm["mse"]
            gsparam=torch.cat([x.detach().reshape(-1) for x in torch.autograd.grad(sup,params,retain_graph=True)])
            H=pred[0].numel();N=len(pred)
            gs=center(err.detach()/norm["mse"]+.5*err.detach().sign()/norm["mae"])/H
            gd=2*(center(pred.detach())-soft[ids])/(H*norm["mse"])
            if step==0:
                actualsup=torch.autograd.grad(sup,pred,retain_graph=True)[0].detach()
                actualkd=torch.autograd.grad(kd,pred,retain_graph=True)[0].detach()
                assert torch.allclose(center(actualsup)*N,gs,atol=1e-12,rtol=1e-12)
                assert torch.allclose(actualkd*N,gd,atol=1e-12,rtol=1e-12)
            projected,beforevec,aftervec=project_teacher(gs.flatten(1),gd.flatten(1))
            normsS=gs.flatten(1).norm(dim=1);normsP=projected.norm(dim=1)
            assert bool((aftervec>=-(1e-6*normsS*normsP+1e-12)).all())
            projected=projected.reshape_as(pred)
            before=beforevec.sum();after=aftervec.sum()
            sn=float(gs.norm());dn=float(gd.norm());pn=float(projected.norm())
            old=torch.cat([p.detach().reshape(-1) for p in params]).double()
            opt.zero_grad()
            pred.backward((gs+c["kd_weight"]*projected)/N)
            opt.step()
            updated=torch.cat([p.detach().reshape(-1) for p in params]).double()
            if record:
                gradient_log.append(dict(seed=seed,fold=fold,arm=arm,update=step+1,sup=float(sup.detach()),kd=float(kd.detach()),dot_before=float(before),dot_after=float(after),cos_before=float(before)/(sn*dn) if sn*dn>0 else 0.,KD_norm=dn,KD_kept_norm=pn if arm!="RawQwenKDReplay" else dn,KD_retained_fraction=pn/dn if arm!="RawQwenKDReplay" and dn>0 else 1.,Adam_SUP_dot_step=float(torch.dot(gsparam.double(),updated-old)),window_conflict_fraction=float((beforevec<0).double().mean()),projection_scope="per_window_output_shape"))
            assert time.monotonic()-start<c["fit_seconds"],"bounded CUDA runtime exceeded"
        if record:fits.append(dict(seed=seed,fold=fold,arm=arm,steps=c["steps"],train_windows=len(trainids),parameters=705,initial_gradient=grad,device=str(device)))
        return out,rows
    chosen=dict(recipe="supervised_compatible_KD",lr=c["lr"],weight=c["kd_weight"],batch=c["batch"],step=c["primary_step"],start=0,actual_KD_updates=c["primary_step"],selection="Fixed23 settings; no parameter/method/checkpoint selection")
    save(r/"fixed_method_protocol.json",dict(chosen=chosen,method=c["method"],git=git(),no_search=True))
    methodlock=sha(r/"fixed_method_protocol.json")
    proposed={}
    for fold in folds:
        f=fold["index"];proposed[f]={}
        for seed in c["seeds"]:
            out,rows=train(seed,fold["train"],fold["val"],fold=f)
            proposed[f][seed]=out;curves.extend(rows)
        print("PROJECTED_FOLD_DONE",f,flush=True)
    rawstates=torch.load(Path(c["anchor_run"])/"raw_anchor_checkpoints.pt",map_location="cpu",weights_only=True)
    raw_exact=0;raw_max_nonbias_diff=0.
    for seed in c["seeds"]:
        for step,sd in rawstates[seed].items():
            assert all(torch.equal(v,rawold[0][seed][step][k]) for k,v in sd.items())
            raw_exact+=1
    deterministic_repeats=0
    for seed in c["seeds"]:
        out,_=train(seed,folds[0]["train"],folds[0]["val"],fold=0,record=False)
        for step,sd in out.items():
            assert all(torch.equal(v,proposed[0][seed][step][k]) for k,v in sd.items()),("GPU method repeat",seed,step)
            deterministic_repeats+=1
    fixedstep=c["primary_step"];cvcomparison=[]
    for arm,rows in [("WindowProjectedQwenKD",curves),("RawQwenKD25_cached",rawcurves)]:
        if arm=="WindowProjectedQwenKD":rows=[q for q in rows if q["arm"]==arm]
        for q in rows:
            if q["step"]!=fixedstep:continue
            bn=validationnorm[q["fold"],q["seed"]]
            cvcomparison.append(dict(arm=arm,fold=q["fold"],seed=q["seed"],score=.5*(q["mse"]/bn["mse"]+q["mae"]/bn["mae"]),mse=q["mse"],mae=q["mae"]))
    assert len(cvcomparison)==18
    save(r/"cv_comparison.json",cvcomparison)
    save(r/"cv_selection.json",dict(chosen=chosen,note="Fixed23 recipe and step. CV diagnostics only, no checkpoint or method selection.",diagnostics=cvcomparison))
    cvlock=sha(r/"cv_selection.json")
    save(r/"curves.json",curves);save(r/"gradient_log.json",gradient_log);save(r/"fits.json",fits)
    torch.save(proposed,r/"projected_fold_checkpoints.pt");torch.save(rawstates,r/"raw_anchor_checkpoints.pt")
    final=proposed[0];torch.save(final,r/"final_checkpoints.pt")
    assert sha(r/"fixed_method_protocol.json")==methodlock
    save(r/"selection.json",dict(chosen=chosen,cv_selection_sha=cvlock,method_protocol_sha=methodlock,final_states_sha=sha(r/"final_checkpoints.pt"),raw_states_sha=sha(r/"raw_anchor_checkpoints.pt"),final_gradient_windows=nfit,controls_retrained=0,outer_candidates_evaluated=False))
    lock=sha(r/"selection.json")
    save(r/"evaluation_lock.json",dict(selection_sha=lock,cv_selection_sha=cvlock,method_protocol_sha=methodlock,git=git(),elapsed_before_outer=time.monotonic()-start))
    states={chosen["recipe"]:final};key=chosen["recipe"]
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
            outputs["QwenTaskKD27"]=pred
            rawhead=Head(seed).to(device);rawhead.load_state_dict(rawstates[seed][chosen["step"]])
            with torch.no_grad():outputs["RawQwenKDReplay"]=forward(rawhead,zz,bp).cpu().numpy()
            maxmean=max(maxmean,float(abs((pred-bp.cpu().numpy()).mean(1)).max()))
            baselevel=scores(bp.cpu().numpy(),y)["level"]
            for arm,p in outputs.items():
                met=scores(p,y);metrics.append(dict(domain="Metro",seed=seed,split=label,arm=arm,**met))
                predictions[root+"_"+arm]=p
                assert abs(met["mse"]-met["level"]-met["shape"])<1e-12
                if arm in controls:
                    previous=next(q for q in oldmetrics if q["domain"]=="Metro" and q["seed"]==seed and q["split"]==label and q["arm"]==arm)
                    assert all(abs(met[k]-previous[k])<1e-15 for k in met),("frozen metric changed",arm,seed,split)
                if arm=="QwenTaskKD27":maxlevel=max(maxlevel,abs(met["level"]-baselevel))
    torch.save(selected,r/"selected_qwen_students.pt")
    np.savez_compressed(r/"predictions.npz",**predictions);save(r/"metrics.json",metrics)
    arms=controls+["QwenTaskKD22","QwenTaskKD23","RawQwenKDReplay","QwenTaskKD27"]
    summary={s:{a:{k:float(np.mean([q[k] for q in metrics if q["split"]==s and q["arm"]==a])) for k in ["mse","mae","level","shape"]} for a in arms} for s in ["fit","val","eval"]}
    intervals=[];rng=np.random.default_rng(2024)
    origins=predictions["Metro_2024_eval_origin"];clusters=(origins-origins.min())//336;unique=np.unique(clusters)
    for arm in ["NoKD","FourierKD","HistoricalKD","ShuffleKD","QwenTaskKD22","QwenTaskKD23","RawQwenKDReplay"]:
        for metric in ["mse","mae"]:
            delta=[]
            for seed in c["seeds"]:
                root=f"Metro_{seed}_eval";y=predictions[root+"_y"]
                a=predictions[root+"_QwenTaskKD27"]-y;b=predictions[root+"_"+arm]-y
                delta.append(((a*a-b*b) if metric=="mse" else (abs(a)-abs(b))).mean((1,2)))
            dd=np.mean(delta,0);boot=[]
            for _ in range(2000):
                ids=np.concatenate([np.flatnonzero(clusters==k) for k in rng.choice(unique,len(unique),replace=True)])
                boot.append(float(dd[ids].mean()))
            intervals.append(dict(reference=arm,metric=metric,delta=float(dd.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),blocks=len(unique),note="Viewed development; seeds averaged then paired14day block resampling, not independent confirmation."))
    assert maxmean<1e-10 and maxlevel<1e-10
    assert sha(r/"selection.json")==lock and sha(r/"cv_selection.json")==cvlock and sha(r/"fixed_method_protocol.json")==methodlock
    for a in assets:assert sha(a["path"])==a["sha"]
    save(r/"summary.json",dict(metrics=summary,paired_intervals=intervals,selection=chosen,Capital_Qwen_unchanged22=True))
    save(r/"verification.json",dict(fits=len(fits),total_updates=len(fits)*c["steps"],parameters=705,deterministic_checkpoint_repeats=deterministic_repeats,raw_anchor_exact_checkpoints=raw_exact,raw_anchor_nonbias_parameter_diff=raw_max_nonbias_diff,final_fit_windows=73,fold_target_disjointness=True,training_device="cuda",peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,selected_replays=replays,mean_error_max=maxmean,level_difference_max=maxlevel,frozen_control_metric_equality=True,all_source_SHA_unchanged=True,selection_lock_unchanged=True,controls_trained=0,LLM_calls=0,TimeBase_updates=0))
    save(r/"status.json",dict(seconds=time.monotonic()-start,git=git(),fits=len(fits),repeat_fits=3,projected_fold_fits=9,raw_anchor_fits=0,cached_raw_anchors_from26=True,final_weights_from_original73_fold0=True,device=torch.cuda.get_device_name(),peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,complete=True))
    print("CHOSEN",chosen,flush=True)
    print("EVAL",summary["eval"],flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args()
    run(read(a.config),a.config)
