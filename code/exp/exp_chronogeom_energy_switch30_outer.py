"""Frozen energy-switch29 evaluation on Metro outer data; no fit operations."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",CUBLAS_WORKSPACE_CONFIG=":4096:8",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,csv,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from layers.ChronoGeomEnergySwitch import EnergySwitch,centered
from exp.exp_chronogeom_event_knowledge19 import read,save,sha,git,scores

def run(c,config):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device("cuda:0");torch.cuda.set_per_process_memory_fraction(.04);torch.cuda.reset_peak_memory_stats()
    start=time.monotonic();r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False)
    switch=Path(c["switch_run"]);expert=Path(c["expert_run"]);ref=Path(c["reference22"]);ref23=Path(c["reference23"]);teacher=Path(c["teacher_run"])
    original_lock=read(switch/"check_lock.json")
    assert sha(switch/"switch.pt")==original_lock["weight_sha"]==c["switch_sha"]
    assert sha(switch/"fit.json")==original_lock["fit_sha"]
    freeze=read("docs/chronogeom/FROZEN_COMPARATORS_22.json")
    assert str(ref)==freeze["run"] and c["seeds"]==freeze["seeds"]
    assets=[dict(path=str(ref/f),sha=v) for f,v in freeze["artifact_sha256"].items()]+freeze["base_assets"]
    for root,names in [(switch,["switch.pt","fit.json","config.json","check_lock.json"]),(expert,["predictions.npz","Metro/data.npz","config.json","selected_qwen_students.pt"]),(ref23,["selected_qwen_students.pt","predictions.npz","metrics.json"]),(teacher,["config.json"])]:
        for f in names:assets.append(dict(path=str(root/f),sha=sha(root/f)))
    for f in ["layers/ChronoGeomEnergySwitch.py","docs/chronogeom/FROZEN_COMPARATORS_22.json"]:
        assets.append(dict(path=f,sha=sha(f)))
    for a in assets:assert sha(a["path"])==a["sha"]
    for a in read(switch/"input_assets.json"):assert sha(a["path"])==a["sha"]
    save(r/"config.json",c);save(r/"input_assets.json",assets)
    save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),gpu_uuid=c["gpu_uuid"]))
    model=EnergySwitch().double().to(device);model.load_state_dict(torch.load(switch/"switch.pt",map_location=device,weights_only=True));model.eval()
    for p in model.parameters():p.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters())==5 and not any(p.requires_grad for p in model.parameters())
    state_before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    save(r/"weights_lock.json",dict(switch_sha=sha(switch/"switch.pt"),fit_sha=sha(switch/"fit.json"),frozen_state={k:v.tolist() for k,v in state_before.items()},fitting=False,normalizer_updates=False))
    data=np.load(expert/"Metro/data.npz");cache=np.load(expert/"predictions.npz");fixed=np.load(ref/"predictions.npz");old23=np.load(ref23/"predictions.npz")
    tc=read(teacher/"config.json");t0=pd.to_datetime(pd.read_csv(tc["data"],usecols=["date_time"]).date_time).min().normalize()
    origin=data["confirm_origin"];future=data["confirm_future"];assert len(origin)==85 and future.shape==(85,96)
    fit=read(switch/"fit.json");oldmeta=np.isin(data["val_origin"],fit["calibration_origins"]+fit["check_origins"])
    assert np.intersect1d(future.ravel(),data["val_future"][oldmeta].ravel()).size==0
    assert (t0+pd.to_timedelta(future.ravel().min(),unit="h"))>=pd.Timestamp("2017-08-21")
    assert (t0+pd.to_timedelta(future.ravel().max(),unit="h"))<pd.Timestamp("2018-01-01")
    history=torch.tensor(data["confirm_x"],dtype=torch.float64,device=device)
    past=origin[:,None]+np.arange(-96,0);pw=torch.tensor(((t0.dayofweek+past//24)%7>=5).astype(float),device=device);fw=torch.tensor(((t0.dayofweek+future//24)%7>=5).astype(float),device=device)
    controls=freeze["controls"];arms=controls+["QwenTaskKD23","QwenRelationKD28","EnergySwitch29_frozen"]
    predictions={};info_arrays={};replays=0;maxmean=0.
    # Freeze predictions and gates before reading external y for scoring.
    with torch.inference_mode():
        for seed in c["seeds"]:
            key=f"Metro_{seed}_eval"
            assert np.array_equal(origin,cache[key+"_origin"]) and np.array_equal(origin,fixed[key+"_origin"])
            assert np.array_equal(cache[key+"_QwenTaskKD23"],old23[key+"_QwenTaskKD23"])
            base=torch.tensor(cache[key+"_QwenTaskKD23"],device=device);candidate=torch.tensor(cache[key+"_QwenRelationKD28"],device=device)
            out,info=model(history,base,candidate,pw,fw);repeat,ri=model(history,base,candidate,pw,fw)
            assert torch.equal(out,repeat) and torch.equal(info["weight"],ri["weight"]);replays+=1
            for arm in controls:predictions[f"{seed}_{arm}"]=fixed[key+"_"+arm]
            predictions[f"{seed}_QwenTaskKD23"]=base.cpu().numpy();predictions[f"{seed}_QwenRelationKD28"]=candidate.cpu().numpy();predictions[f"{seed}_EnergySwitch29_frozen"]=out.cpu().numpy()
            info_arrays[f"{seed}_weight"]=info["weight"].cpu().numpy();info_arrays[f"{seed}_Ahat"]=info["cross_energy_estimate"].cpu().numpy();info_arrays[f"{seed}_B"]=info["correction_energy"].cpu().numpy()
            info_arrays[f"{seed}_features"]=model.observable_features(history,base,candidate,pw,fw).cpu().numpy()
            maxmean=max(maxmean,float(abs((out-base).mean(1)).max()))
            print("FROZEN_PREDICTION_DONE",seed,len(out),flush=True)
    for k,v in model.state_dict().items():assert torch.equal(v.detach().cpu(),state_before[k])
    np.savez_compressed(r/"predictions.npz",**predictions);np.savez_compressed(r/"gate_information.npz",**info_arrays)
    save(r/"origins.json",dict(origin=origin.tolist(),timestamp=(t0+pd.to_timedelta(origin,unit="h")).astype(str).tolist(),windows=85,unique_target_hours=len(np.unique(future))))
    prediction_sha=sha(r/"predictions.npz");gate_sha=sha(r/"gate_information.npz")
    save(r/"evaluation_lock.json",dict(prediction_sha=prediction_sha,gate_sha=gate_sha,switch_sha=c["switch_sha"],source_sha=sha(__file__),no_fit=True,before_external_label_scoring=True,git=git()))
    metrics=[];miscorrections=[];windows=[];yarrays={};maxidentity=0.;maxlevel=0.
    fixedmetrics=read(ref/"metrics.json")
    for seed in c["seeds"]:
        key=f"Metro_{seed}_eval";y=cache[key+"_y"]
        assert np.array_equal(y,data["confirm_y"]) and np.array_equal(y,fixed[key+"_y"]);yarrays[str(seed)]=y
        for arm in arms:
            met=scores(predictions[f"{seed}_{arm}"],y);metrics.append(dict(seed=seed,arm=arm,**met))
            if arm in controls:
                q=next(x for x in fixedmetrics if x["domain"]=="Metro" and x["seed"]==seed and x["split"]=="eval" and x["arm"]==arm)
                assert all(abs(met[k]-q[k])<1e-15 for k in met)
        base=predictions[f"{seed}_QwenTaskKD23"];mixed=predictions[f"{seed}_EnergySwitch29_frozen"];candidate=predictions[f"{seed}_QwenRelationKD28"]
        eb=base-y;em=mixed-y;d=candidate-base;d=d-d.mean(1,keepdims=True);A=((y-base)*d).mean((1,2));B=(d*d).mean((1,2));aa=info_arrays[f"{seed}_weight"].reshape(-1)
        maxidentity=max(maxidentity,float(abs((em*em).mean((1,2))-(eb*eb).mean((1,2))+2*aa*A-aa*aa*B).max()))
        maxlevel=max(maxlevel,abs(scores(base,y)["level"]-scores(mixed,y)["level"]))
        for name in ["mse","mae"]:
            before=(eb*eb if name=="mse" else abs(eb)).mean((1,2));after=(em*em if name=="mse" else abs(em)).mean((1,2));gain=before-after
            bad=np.flatnonzero(gain<0);worst=int(gain.argmin())
            miscorrections.append(dict(seed=seed,metric=name,windows=len(gain),helpful=int((gain>0).sum()),harmful=int((gain<0).sum()),unchanged=int((gain==0).sum()),positive_gain_mean=float(np.maximum(gain,0).mean()),miscorrection_loss_mean=float(np.maximum(-gain,0).mean()),net_gain_mean=float(gain.mean()),conditional_harm_mean=float(-gain[bad].mean()) if len(bad) else 0.,worst_harm=float(max(0.,-gain[worst])),worst_harm_origin=str(t0+pd.Timedelta(hours=int(origin[worst]))),mean_gate=float(aa.mean()),zero_gate_fraction=float((aa==0).mean()),full_gate_fraction=float((aa==1).mean())))
            assert abs(np.maximum(gain,0).mean()-np.maximum(-gain,0).mean()-gain.mean())<1e-12
        for i,o in enumerate(origin):
            windows.append(dict(seed=seed,origin=str(t0+pd.Timedelta(hours=int(o))),gate=float(aa[i]),Ahat=float(info_arrays[f"{seed}_Ahat"][i,0]),true_A_diagnostic=float(A[i]),B=float(B[i]),mse23=float((eb[i]**2).mean()),mse_switch=float((em[i]**2).mean()),mae23=float(abs(eb[i]).mean()),mae_switch=float(abs(em[i]).mean()),mse_gain=float((eb[i]**2-em[i]**2).mean()),mae_gain=float((abs(eb[i])-abs(em[i])).mean())))
    save(r/"metrics.json",metrics);save(r/"miscorrections.json",miscorrections);np.savez_compressed(r/"labels.npz",**yarrays)
    with (r/"per_window.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(windows[0]));writer.writeheader();writer.writerows(windows)
    summary={arm:{k:float(np.mean([q[k] for q in metrics if q["arm"]==arm])) for k in ["mse","mae","level","shape"]} for arm in arms}
    uncertainty=[];rng=np.random.default_rng(2024);block=(origin-origin.min())//336;unique=np.unique(block)
    for reference in ["QwenTaskKD23","NoKD","FourierKD","HistoricalKD","ShuffleKD"]:
        for metric in ["mse","mae"]:
            deltas=[]
            for seed in c["seeds"]:
                y=yarrays[str(seed)];a=predictions[f"{seed}_EnergySwitch29_frozen"]-y;b=predictions[f"{seed}_{reference}"]-y
                deltas.append(((a*a-b*b) if metric=="mse" else (abs(a)-abs(b))).mean((1,2)))
            delta=np.mean(deltas,0);bootstrap=[]
            for _ in range(2000):
                ids=np.concatenate([np.flatnonzero(block==b) for b in rng.choice(unique,len(unique),replace=True)]);bootstrap.append(float(delta[ids].mean()))
            uncertainty.append(dict(reference=reference,metric=metric,switch_minus_reference=float(delta.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),blocks=len(unique),note="Descriptive paired14day blocks,seedlosses averaged first,vieweddevelopment/notindependentconfirmation."))
    assert maxmean<1e-12 and maxidentity<1e-12 and maxlevel<1e-12
    assert sha(r/"predictions.npz")==prediction_sha and sha(r/"gate_information.npz")==gate_sha
    for a in assets:assert sha(a["path"])==a["sha"]
    for k,v in model.state_dict().items():assert torch.equal(v.detach().cpu(),state_before[k])
    assert time.monotonic()-start<60
    save(r/"summary.json",dict(metrics=summary,paired_intervals=uncertainty,mean_miscorrection={metric:{key:float(np.mean([q[key] for q in miscorrections if q["metric"]==metric])) for key in ["positive_gain_mean","miscorrection_loss_mean","net_gain_mean"]} for metric in ["mse","mae"]}))
    save(r/"verification.json",dict(parameters=5,fitted_shared29_weights_exact=True,prediction_replays=replays,source_and_Capital_assets_unchanged=True,fixed22_metrics_exact=True,state_dict_unchanged=True,prediction_gate_lock_unchanged=True,calibration_outer_targets_disjoint=True,maximum_level_difference=maxlevel,maximum_mean_shift=maxmean,maximum_energy_identity_error=maxidentity,training_updates=0,normalization_updates=0,LLM_calls=0,expert_updates=0,control_updates=0))
    save(r/"status.json",dict(seconds=time.monotonic()-start,complete=True,git=git(),device=torch.cuda.get_device_name(),peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,training=False,new_independent_confirmation=False))
    print("OUTER_METRICS",summary,flush=True);print("MISCORRECTIONS",miscorrections,flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args();run(read(a.config),a.config)
