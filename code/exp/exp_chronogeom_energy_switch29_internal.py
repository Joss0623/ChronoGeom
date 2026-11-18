"""Five-parameter energy-switch INTERNAL pilot, with all expert weights frozen."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",CUBLAS_WORKSPACE_CONFIG=":4096:8",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from layers.ChronoGeomEnergySwitch import EnergySwitch,centered,mix_by_cross_energy
from exp.exp_chronogeom_event_knowledge19 import read,save,sha,git

def metric(p,y):
    e=p-y
    return dict(mse=float(e.square().mean()),mae=float(e.abs().mean()))

def run(c,config):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda:0");torch.cuda.set_per_process_memory_fraction(.04)
    start=time.monotonic();r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False)
    src=Path(c["experts_run"]);teacher=Path(c["source_teacher_run"])
    paths=[src/"predictions.npz",src/"Metro/data.npz",src/"config.json",src/"selected_qwen_students.pt",teacher/"config.json",Path("layers/ChronoGeomEnergySwitch.py")]
    assets=[dict(path=str(p),sha=sha(p)) for p in paths]
    save(r/"config.json",c);save(r/"input_assets.json",assets)
    save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),python=sys.version,torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name(),gpu_uuid=c["gpu_uuid"]))
    data=np.load(src/"Metro/data.npz");pred=np.load(src/"predictions.npz");tc=read(teacher/"config.json")
    t0=pd.to_datetime(pd.read_csv(tc["data"],usecols=["date_time"]).date_time).min().normalize()
    origins=data["val_origin"];future=data["val_future"];times=t0+pd.to_timedelta(origins,unit="h");ends=times+pd.Timedelta(hours=96)
    cutoff=pd.Timestamp(c["calibration_end"]);last=pd.Timestamp(c["check_end"])
    tr=np.flatnonzero(ends<=cutoff);check=np.flatnonzero((times>=cutoff)&(ends<=last));purged=np.setdiff1d(np.arange(len(origins)),np.concatenate([tr,check]))
    assert len(tr)==23 and len(check)==6 and len(purged)==3
    assert np.intersect1d(future[tr].ravel(),future[check].ravel()).size==0
    assert future[tr].max()<future[check].min()
    past=origins[:,None]+np.arange(-96,0);pastweek=((t0.dayofweek+past//24)%7>=5).astype(float);fweek=((t0.dayofweek+future//24)%7>=5).astype(float)
    history=torch.tensor(data["val_x"],dtype=torch.float64,device=device);pastweek=torch.tensor(pastweek,device=device);fweek=torch.tensor(fweek,device=device)
    model=EnergySwitch().double().to(device)
    assert sum(p.numel() for p in model.parameters())==5
    bundles={};features=[];labels=[]
    for seed in c["seeds"]:
        root=f"Metro_{seed}_val"
        reference=torch.tensor(pred[root+"_QwenTaskKD23"],device=device);candidate=torch.tensor(pred[root+"_QwenRelationKD28"],device=device)
        obs=model.observable_features(history,reference,candidate,pastweek,fweek)
        with torch.no_grad():initial,_=model(history,reference,candidate,pastweek,fweek)
        assert torch.equal(initial,reference)
        # Only calibration labels are used before the fit lock.
        ytrain=torch.tensor(pred[root+"_y"][tr],device=device)
        delta=centered(candidate-reference)
        Atrain=((ytrain-reference[tr])*delta[tr]).mean((1,2))
        features.append(obs[tr]);labels.append(Atrain)
        bundles[seed]=dict(reference=reference,candidate=candidate,delta=delta,features=obs)
    fitX=torch.cat(features);fitA=torch.cat(labels)
    model.set_training_normalizer(fitX)
    design=torch.cat([(fitX-model.feature_mean)/model.feature_scale,torch.ones(len(fitX),1,device=device)],dim=1)
    coeff=torch.linalg.pinv(design)@fitA
    with torch.no_grad():
        model.alignment.weight.copy_(coeff[:4].reshape(1,4))
        model.alignment.bias.copy_(coeff[4:])
    rank=int(torch.linalg.matrix_rank(design))
    torch.save(model.state_dict(),r/"switch.pt");save(r/"fit.json",dict(parameters=5,shared_across_seeds=True,rows=len(fitA),unique_origins=len(tr),rank=rank,coefficients=coeff.cpu().tolist(),feature_mean=model.feature_mean.cpu().tolist(),feature_scale=model.feature_scale.cpu().tolist(),calibration_origins=origins[tr].tolist(),check_origins=origins[check].tolist(),purged_origins=origins[purged].tolist(),no_external_arrays_accessed=True))
    weightsha=sha(r/"switch.pt");save(r/"check_lock.json",dict(weight_sha=weightsha,fit_sha=sha(r/"fit.json"),git=git(),check_labels_used_for_fit=False))
    rows=[];outputs={};maxlevel=0.;maxidentity=0.
    with torch.no_grad():
        for seed,d in bundles.items():
            mixed,info=model(history,d["reference"],d["candidate"],pastweek,fweek)
            for split,idx in [("calibration",tr),("internal_check",check)]:
                yy=torch.tensor(pred[f"Metro_{seed}_val_y"][idx],device=device)
                b=d["reference"][idx];cand=d["candidate"][idx];sw=mixed[idx];delta=d["delta"][idx];e=yy-b;A=(e*delta).mean((1,2));B=delta.square().mean((1,2));oracleweight=(A/B.clamp_min(1e-12)).clamp(0,1);oracle=b+oracleweight[:,None,None]*delta
                for arm,pr in [("Qwen23",b),("Relation28",cand),("EnergySwitch29",sw),("Oracle_diagnostic_only",oracle)]:
                    rows.append(dict(seed=seed,split=split,arm=arm,windows=len(idx),**metric(pr,yy)))
                    outputs[f"{seed}_{split}_{arm}"]=pr.cpu().numpy()
                outputs[f"{seed}_{split}_y"]=yy.cpu().numpy();outputs[f"{seed}_{split}_weight"]=info["weight"][idx].cpu().numpy();outputs[f"{seed}_{split}_Ahat"]=info["cross_energy_estimate"][idx].cpu().numpy();outputs[f"{seed}_{split}_A"]=A.cpu().numpy();outputs[f"{seed}_{split}_B"]=B.cpu().numpy()
                maxlevel=max(maxlevel,float(abs((sw-b).mean(1)).max()))
                aa=info["weight"][idx].reshape(-1)
                maxidentity=max(maxidentity,float(abs((sw-yy).square().mean((1,2))-e.square().mean((1,2))+2*aa*A-aa.square()*B).max()))
                print("CHECK",seed,split,metric(sw,yy),"mean_weight",float(aa.mean()),flush=True)
    assert maxlevel<1e-12 and maxidentity<1e-12 and sha(r/"switch.pt")==weightsha
    for a in assets:assert sha(a["path"])==a["sha"]
    assert time.monotonic()-start<60
    save(r/"metrics.json",rows);np.savez_compressed(r/"predictions.npz",**outputs)
    summary={split:{arm:{k:float(np.mean([q[k] for q in rows if q["split"]==split and q["arm"]==arm])) for k in ["mse","mae"]} for arm in ["Qwen23","Relation28","EnergySwitch29","Oracle_diagnostic_only"]} for split in ["calibration","internal_check"]}
    save(r/"summary.json",summary);save(r/"verification.json",dict(parameters=5,shared3seed_gate=True,train_origins=23,check_origins=6,purged_origins=3,target_hours_disjoint=True,check_after_weight_lock=True,weight_and_sources_SHA_unchanged=True,maximum_level_difference=maxlevel,maximum_energy_identity_error=maxidentity,external_test_arrays_accessed=False,LLM_calls=0,TimeBase_updates=0,experts_updated=0,control_updates=0))
    save(r/"status.json",dict(seconds=time.monotonic()-start,complete=True,gate_fit_only=True,new_independent_confirmation=False,git=git()))
    print("SUMMARY",summary,flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args();run(read(a.config),a.config)
