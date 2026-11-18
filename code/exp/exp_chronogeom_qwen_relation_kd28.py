"""Task-conditioned two-relation KD, fixed TimeBase/Qwen teacher and23 recipe."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0",CUBLAS_WORKSPACE_CONFIG=":4096:8")
import argparse,copy,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from exp.exp_chronogeom_event_knowledge19 import Head,forward,scores,read,save,sha,git
from exp.exp_chronogeom_event_confirm21 import dataset
from exp.exp_chronogeom_residual03 import numeric_summary

def run(c,config):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.04);device=torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False)
    src=Path(c["teacher_run"]);ref=Path(c["reference"]);previous=Path(c["previous"])
    freeze=read("docs/chronogeom/FROZEN_COMPARATORS_22.json")
    assert str(ref)==freeze["run"] and c["seeds"]==freeze["seeds"]
    assets=[dict(path=str(ref/f),sha=v) for f,v in freeze["artifact_sha256"].items()]+freeze["base_assets"]
    for f in ["config.json","normalizers.json","knowledge.npz","data.npz","predictions.npz","profiles.json","teacher_checkpoints.pt","teacher_selection.json"]:
        assets.append(dict(path=str(src/f),sha=sha(src/f)))
    for f in ["selected_qwen_students.pt","predictions.npz","metrics.json","selection.json"]:
        assets.append(dict(path=str(previous/f),sha=sha(previous/f)))
    assets.append(dict(path="docs/chronogeom/FROZEN_COMPARATORS_22.json",sha=sha("docs/chronogeom/FROZEN_COMPARATORS_22.json")))
    for a in assets:assert sha(a["path"])==a["sha"]
    save(r/"config.json",c);save(r/"input_assets.json",assets)
    save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),gpu_uuid=c["gpu_uuid"],tf32=False))
    cfg=read(src/"config.json");folder=r/"Metro";folder.mkdir()
    packs,grid,raw,common,kind,h,hist=dataset(cfg,folder)
    cache=np.load(src/"predictions.npz");preps=read(src/"normalizers.json")
    knowledge=np.load(src/"knowledge.npz");kt={a:knowledge[a] for a in ["Fourier","QwenTask"]}
    nfit=len(packs["fit"]["y"]);assert nfit==73
    W=torch.zeros(2,96,dtype=torch.float64,device=device)
    for i,name in enumerate(["morning","evening"]):
        hours=c["relations"][name];noon=c["relations"]["midday"]
        W[i,hours]=1/len(hours);W[i,noon]=-1/len(noon)
    assert float(W.sum(1).abs().max())<1e-14
    def relation(diff):return torch.einsum("nhc,kh->nkc",diff,W).squeeze(-1)
    zero=torch.zeros(1,96,1,dtype=torch.float64,device=device)
    example=zero.clone();example[:,c["relations"]["morning"]]=1.
    assert torch.allclose(relation(example),torch.tensor([[1.,0.]],device=device,dtype=torch.float64),atol=1e-12)
    example2=zero.clone();example2[:,c["relations"]["evening"]]=-2.
    assert torch.allclose(relation(example2),torch.tensor([[0.,-2.]],device=device,dtype=torch.float64),atol=1e-12)
    assert torch.allclose(relation(example+7),relation(example),atol=1e-12)
    shared=torch.arange(96,dtype=torch.float64,device=device).reshape(1,96,1)/13
    assert torch.allclose(relation((example+shared)-(zero+shared)),relation(example-zero),atol=1e-12)
    save(r/"relation_math_checks.json",dict(morning_peak=True,evening_peak=True,offset_invariance=True,shared_numerical_base_cancellation=True,W=W.cpu().tolist()))
    lo,hi=map(pd.Timestamp,cfg["splits"]["fit"]);pairs=[]
    for day in pd.date_range(lo,hi-pd.Timedelta(days=9),freq="D"):
        if day.dayofweek!=0:continue
        sat=day+pd.Timedelta(days=5)
        workix=grid.get_loc(day)+np.arange(96);weekix=grid.get_loc(sat)+np.arange(96)
        if grid[weekix[-1]]>=hi:continue
        if not (np.isfinite(raw[workix]).all() and np.isfinite(raw[weekix]).all()):continue
        if kind[workix[0]]!=0 or kind[weekix[0]]!=1:continue
        assert grid[workix[0]]>=lo and grid[weekix[-1]]<hi
        assert np.array_equal(h[workix],np.tile(np.arange(24),4)) and np.array_equal(h[weekix],np.tile(np.arange(24),4))
        pairs.append(dict(workday=str(day.date()),weekend=str(sat.date()),work_ix=workix,weekend_ix=weekix))
    assert len(pairs)>0
    context={};contextmeta={}
    starts=np.array([pd.Timestamp(p["workday"]).value for p in pairs],dtype=np.int64)
    for split in ["fit","val"]:
        dates=grid[packs[split]["origin"]];weekstart=np.array([(d-pd.Timedelta(days=d.dayofweek)).value for d in dates],dtype=np.int64)
        chosenpair=np.abs(weekstart[:,None]-starts[None,:]).argmin(1)
        context[split]={name:np.stack([pairs[i][name+"_ix"] for i in chosenpair]) for name in ["work","weekend"]}
        contextmeta[split]=[dict(origin=str(dates[j]),workday=pairs[i]["workday"],weekend=pairs[i]["weekend"]) for j,i in enumerate(chosenpair)]
    save(r/"calendar_contexts.json",dict(candidate_pairs=[{k:v for k,v in p.items() if not k.endswith("_ix")} for p in pairs],assignments=contextmeta,source="Calendar metadata only, ORIGINALFIT period, full96h coherent calendars, no observed demand used to choose pairs."))
    def features(seed,split,arm="Fourier",override=None):
        p=packs[split];bp=torch.tensor(cache[f"{seed}_{split}_FrozenTimeBase"],dtype=torch.float32)
        assert np.array_equal(p["origin"],cache[f"{seed}_{split}_origin"])
        ix=p["future"] if override is None else override
        ft=np.concatenate([numeric_summary(p["x"],bp).numpy(),raw[ix],common[ix]],-1)
        pp=preps[str(seed)][arm]
        z=torch.tensor(np.concatenate([(ft-np.array(pp["mean"]))/np.array(pp["sd"]),(kt[arm][ix]-np.array(pp["knowledge_mean"]))/pp["knowledge_rms"]],-1),dtype=torch.float32)
        return z.to(device),bp.to(device)
    z={};base={};virtual={};relT={};probT={};entropyT={};scale={};norm={}
    target=torch.tensor(packs["fit"]["y"],dtype=torch.float64,device=device)
    teacher_states=torch.load(src/"teacher_checkpoints.pt",map_location="cpu",weights_only=True)
    teacher_choice=read(src/"teacher_selection.json")["chosen"]["QwenTask"]
    teacher_replays=[];relation_arrays={}
    for split in ["fit","val"]:
        xx=packs[split]["x"].double().to(device)
        scale[split]=xx.std(dim=1,unbiased=False).reshape(-1,1).clamp_min(1e-6).detach()
    for seed in c["seeds"]:
        z[seed]={};base[seed]={};virtual[seed]={};relT[seed]={};probT[seed]={};entropyT[seed]={}
        t=Head(seed).to(device);t.load_state_dict(teacher_states[seed]["QwenTask"][teacher_choice]);t.eval()
        for param in t.parameters():param.requires_grad_(False)
        assert sum(p.numel() for p in t.parameters() if p.requires_grad)==0
        for split in ["fit","val"]:
            z[seed][split],base[seed][split]=features(seed,split)
            virtual[seed][split]={name:features(seed,split,"Fourier",ix)[0] for name,ix in context[split].items()}
            zreal,_=features(seed,split,"QwenTask")
            zw,_=features(seed,split,"QwenTask",context[split]["work"])
            ze,_=features(seed,split,"QwenTask",context[split]["weekend"])
            with torch.no_grad():
                natural=forward(t,zreal,base[seed][split]).cpu().numpy()
                RT=relation(forward(t,ze,base[seed][split])-forward(t,zw,base[seed][split])).detach()
                pt=torch.sigmoid(RT/scale[split]).detach()
            diff=float(abs(natural-cache[f"{seed}_{split}_Teacher_QwenTask"]).max());assert diff<2e-6
            teacher_replays.append(dict(seed=seed,split=split,maximum_error=diff))
            relT[seed][split]=RT;probT[seed][split]=pt;entropyT[seed][split]=F.binary_cross_entropy(pt,pt,reduction="none").detach()
            relation_arrays[f"{seed}_{split}_teacher_R"]=RT.cpu().numpy();relation_arrays[f"{seed}_{split}_teacher_p"]=pt.cpu().numpy()
        norm[seed]=scores(base[seed]["fit"].cpu().numpy(),packs["fit"]["y"])
    for split in ["fit","val"]:relation_arrays[f"{split}_history_std"]=scale[split].cpu().numpy()
    save(r/"teacher_replay.json",teacher_replays)
    np.savez_compressed(r/"teacher_relations.npz",**relation_arrays)
    def relation_logits(head,seed,split,ids=None):
        bp=base[seed][split];v=virtual[seed][split]
        if ids is not None:bp=bp[ids];v={k:x[ids] for k,x in v.items()}
        R=relation(forward(head,v["weekend"],bp)-forward(head,v["work"],bp))
        s=scale[split] if ids is None else scale[split][ids]
        return R/s,R
    chosen=dict(recipe="task_conditioned_two_relation_KD",lr=c["lr"],weight=c["kd_weight"],batch=c["batch"],step=c["primary_step"],start=0,actual_KD_updates=c["primary_step"])
    save(r/"fixed_protocol.json",dict(chosen=chosen,relations=c["relations"],calendar_contexts_sha=sha(r/"calendar_contexts.json"),no_search=True,git=git()))
    methodlock=sha(r/"fixed_protocol.json")
    allstates={};curves=[];fits=[];losslog=[];relation_predictions={}
    def train(seed,record=True):
        head=Head(seed).to(device);assert sum(p.numel() for p in head.parameters())==705
        init=forward(head,z[seed]["fit"],base[seed]["fit"]);assert torch.equal(init,base[seed]["fit"].double())
        lg,rs=relation_logits(head,seed,"fit");assert torch.equal(rs,torch.zeros_like(rs))
        initkd=F.binary_cross_entropy_with_logits(lg,probT[seed]["fit"])-entropyT[seed]["fit"].mean()
        initkd.backward();initialgrad=float(head.net[-1].weight.grad.abs().sum());assert initialgrad>1e-12
        head.zero_grad();opt=torch.optim.Adam(head.parameters(),lr=c["lr"],foreach=False)
        batches=torch.as_tensor(np.random.default_rng(seed).integers(0,nfit,size=(c["steps"],c["batch"])),device=device)
        out={}
        for step in range(c["steps"]+1):
            if step%c["check_every"]==0:
                out[step]={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                if record:
                    with torch.no_grad():
                        pv=forward(head,z[seed]["val"],base[seed]["val"]).cpu().numpy()
                        lv,rv=relation_logits(head,seed,"val")
                        rkl=F.binary_cross_entropy_with_logits(lv,probT[seed]["val"])-entropyT[seed]["val"].mean()
                    curves.append(dict(seed=seed,step=step,relation_KL=float(rkl),relation_RMSE=float(((rv-relT[seed]["val"])**2).mean().sqrt()),**scores(pv,packs["val"]["y"])))
            if step==c["steps"]:break
            ids=batches[step];pred=forward(head,z[seed]["fit"][ids],base[seed]["fit"][ids]);err=pred-target[ids]
            sup=.5*(err.square().mean()/norm[seed]["mse"]+err.abs().mean()/norm[seed]["mae"])
            logits,R=relation_logits(head,seed,"fit",ids)
            kd=F.binary_cross_entropy_with_logits(logits,probT[seed]["fit"][ids])-entropyT[seed]["fit"][ids].mean()
            loss=sup+c["kd_weight"]*kd
            opt.zero_grad();loss.backward();opt.step()
            if record:losslog.append(dict(seed=seed,update=step+1,SUP=float(sup.detach()),relation_KL=float(kd.detach()),total=float(loss.detach())))
            assert time.monotonic()-start<c["fit_seconds"],"bounded GPU budget exceeded"
        if record:fits.append(dict(seed=seed,steps=c["steps"],primary=c["primary_step"],parameters=705,train_windows=73,initial_relation_gradient=initialgrad))
        return out
    for seed in c["seeds"]:
        allstates[seed]=train(seed);print("RELATION_STUDENT_DONE",seed,flush=True)
    repeats=0
    for seed in c["seeds"]:
        again=train(seed,False)
        for step,sd in allstates[seed].items():
            assert all(torch.equal(v,again[step][k]) for k,v in sd.items()),("deterministic relation replay",seed,step)
            repeats+=1
    torch.save(allstates,r/"all_checkpoints.pt");save(r/"curves.json",curves);save(r/"fits.json",fits);save(r/"training_losses.json",losslog)
    assert sha(r/"fixed_protocol.json")==methodlock
    save(r/"selection.json",dict(chosen=chosen,primary_fixed_before_training=True,method_protocol_sha=methodlock,states_sha=sha(r/"all_checkpoints.pt"),controls_retrained=0,outer_candidates_evaluated=False))
    lock=sha(r/"selection.json");save(r/"evaluation_lock.json",dict(selection_sha=lock,method_protocol_sha=methodlock,git=git(),elapsed_before_outer=time.monotonic()-start))
    states={chosen["recipe"]:allstates};key=chosen["recipe"]
    # Only the single inner-selected Qwen recipe is evaluated on viewed outer data.
    reference=np.load(ref/"predictions.npz")
    selected={};predictions={};metrics=[];maxmean=0.;maxlevel=0.;replays=0
    controls=freeze["controls"]
    oldpred=np.load(previous/"predictions.npz")
    oldmetrics=read(ref/"metrics.json")
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
            outputs["QwenRelationKD28"]=pred
            maxmean=max(maxmean,float(abs((pred-bp.cpu().numpy()).mean(1)).max()))
            baselevel=scores(bp.cpu().numpy(),y)["level"]
            for arm,p in outputs.items():
                met=scores(p,y);metrics.append(dict(domain="Metro",seed=seed,split=label,arm=arm,**met))
                predictions[root+"_"+arm]=p
                assert abs(met["mse"]-met["level"]-met["shape"])<1e-12
                if arm in controls:
                    previous=next(q for q in oldmetrics if q["domain"]=="Metro" and q["seed"]==seed and q["split"]==label and q["arm"]==arm)
                    assert all(abs(met[k]-previous[k])<1e-15 for k in met),("frozen metric changed",arm,seed,split)
                if arm=="QwenRelationKD28":maxlevel=max(maxlevel,abs(met["level"]-baselevel))
    torch.save(selected,r/"selected_qwen_students.pt")
    np.savez_compressed(r/"predictions.npz",**predictions);save(r/"metrics.json",metrics)
    arms=controls+["QwenTaskKD22","QwenTaskKD23","QwenRelationKD28"]
    summary={s:{a:{k:float(np.mean([q[k] for q in metrics if q["split"]==s and q["arm"]==a])) for k in ["mse","mae","level","shape"]} for a in arms} for s in ["fit","val","eval"]}
    intervals=[];rng=np.random.default_rng(2024)
    origins=predictions["Metro_2024_eval_origin"];clusters=(origins-origins.min())//336;unique=np.unique(clusters)
    for arm in ["NoKD","FourierKD","HistoricalKD","ShuffleKD","QwenTaskKD22","QwenTaskKD23"]:
        for metric in ["mse","mae"]:
            delta=[]
            for seed in c["seeds"]:
                root=f"Metro_{seed}_eval";y=predictions[root+"_y"]
                a=predictions[root+"_QwenRelationKD28"]-y;b=predictions[root+"_"+arm]-y
                delta.append(((a*a-b*b) if metric=="mse" else (abs(a)-abs(b))).mean((1,2)))
            dd=np.mean(delta,0);boot=[]
            for _ in range(2000):
                ids=np.concatenate([np.flatnonzero(clusters==k) for k in rng.choice(unique,len(unique),replace=True)])
                boot.append(float(dd[ids].mean()))
            intervals.append(dict(reference=arm,metric=metric,delta=float(dd.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),blocks=len(unique),note="Viewed development; seeds averaged then paired14day block resampling, not independent confirmation."))
    assert maxmean<1e-10 and maxlevel<1e-10
    assert sha(r/"selection.json")==lock and sha(r/"fixed_protocol.json")==methodlock
    for a in assets:assert sha(a["path"])==a["sha"]
    save(r/"summary.json",dict(metrics=summary,paired_intervals=intervals,selection=chosen,Capital_Qwen_unchanged22=True))
    save(r/"verification.json",dict(fits=len(fits),total_updates=len(fits)*c["steps"],parameters=705,deterministic_checkpoint_repeats=repeats,teacher_cache_replay_max=max(q["maximum_error"] for q in teacher_replays),teacher_trainable=0,selected_replays=replays,mean_error_max=maxmean,level_difference_max=maxlevel,frozen_control_metric_equality=True,all_source_SHA_unchanged=True,selection_lock_unchanged=True,controls_trained=0,LLM_calls=0,TimeBase_updates=0))
    save(r/"status.json",dict(seconds=time.monotonic()-start,git=git(),fits=len(fits),repeat_fits=3,device=torch.cuda.get_device_name(),peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,complete=True))
    print("CHOSEN",chosen,flush=True)
    print("EVAL",summary["eval"],flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args()
    run(read(a.config),a.config)
