"""Export the user-selected no-gate main student; no training or LLM calls."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",CUBLAS_WORKSPACE_CONFIG=":4096:8",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
from pathlib import Path
import argparse,csv,time
import numpy as np,pandas as pd,torch
from exp.exp_chronogeom_event_knowledge19 import Head,forward,read,save,sha,git
from exp.exp_chronogeom_downstream import CalendarTimeBase,predict
from exp.exp_chronogeom_residual03 import numeric_summary

def run(source,output):
    start=time.monotonic();src=Path(source);r=Path(output);r.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.04);device=torch.device("cuda:0")
    cfg=read(src/"config.json");sel=read(src/"selection_buildings.json")["selected"]
    assets=read(src/"old_assets.json")
    for meta in sel:
        case=src/meta["building"]
        assets += read(case/"model_lock.json")["files"]+[dict(path=str(case/"test_predictions.npz"),sha=sha(case/"test_predictions.npz")),dict(path=str(case/"metrics.json"),sha=sha(case/"metrics.json"))]
    for a in assets:assert sha(a["path"])==a["sha"]
    save(r/"input_assets.json",assets);save(r/"selection.json",dict(user_requested_main="Qwen temporal relation response teacher + Shape output KD, single head, no switch",source_run=str(src),source_arm="QwenKD23",step=500,development_results_previously_viewed=True,no_new_independent_result=True,LLM_calls=0,training_updates=0,git=git()))
    grid=pd.date_range("2016-01-01","2018-01-01",freq="h",inclusive="left");h=grid.hour.to_numpy();wd=grid.dayofweek.to_numpy();week=(wd>=5).astype(int)
    phase=2*np.pi*(grid.dayofyear.to_numpy()-1)/np.where(grid.is_leap_year,366,365)
    cal=np.stack([np.zeros(len(grid)),week,1-week,np.sin(2*np.pi*wd/7),np.cos(2*np.pi*wd/7),np.sin(phase),np.cos(phase)],1)
    hh=np.stack([fn(2*np.pi*k*h/24) for k in range(1,5) for fn in [np.sin,np.cos]],1);common=np.concatenate([hh*(week==k)[:,None] for k in range(3)],1)
    K=np.stack([np.sin(10*np.pi*h/24),np.cos(10*np.pi*h/24),np.cos(12*np.pi*h/24)],1)
    exported=[];metrics=[];maxerror=0.;maxmean=0.
    for meta in sel:
        name=meta["building"];case=src/name;c=read(case/"config.json");norm=read(case/"normalizers.json");scale=read(case/"scale.json")
        states=torch.load(case/"head_checkpoints.pt",map_location="cpu",weights_only=True);p=np.load(case/"test_labels.npz");old=np.load(case/"test_predictions.npz");future=p["future"]
        x=torch.tensor(p["x"],device=device);ids=torch.tensor(wd[future]*24+h[future],device=device)
        for seed in cfg["seeds"]:
            model=CalendarTimeBase(c,seed,np.zeros((168,168),np.float32)).to(device)
            model.load_state_dict(torch.load(case/f"FrozenTimeBase_{seed}.pt",map_location=device,weights_only=True));model.eval()
            head=Head(seed).to(device);head.load_state_dict(states[seed]["QwenKD23"][500]);head.eval()
            for module in [model,head]:
                for param in module.parameters():param.requires_grad_(False)
            backbone_parameters=sum(a.numel() for a in model.parameters());head_parameters=sum(a.numel() for a in head.parameters());assert (backbone_parameters,head_parameters)==(6570,705)
            normalizer=norm[str(seed)]["Fourier"]
            bundle=dict(backbone={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},normalizer=normalizer,data_scale=scale,building=name,seed=seed,seq_len=96,pred_len=96)
            assert set(bundle)=={"backbone","head","normalizer","data_scale","building","seed","seq_len","pred_len"}
            path=r/f"{name}_seed{seed}.pt";torch.save(bundle,path)
            # Reload ONLY exported body/head, then reproduce the original main-head forecasts.
            loaded=torch.load(path,map_location="cpu",weights_only=True);model.load_state_dict(loaded["backbone"]);head.load_state_dict(loaded["head"])
            with torch.inference_mode():
                b=torch.tensor(predict(model,dict(x=x,ids=ids)),device=device)
                f=np.concatenate([numeric_summary(x,b).cpu().numpy(),cal[future],common[future]],-1);n=loaded["normalizer"]
                z=torch.tensor(np.concatenate([(f-n["mean"])/n["sd"],(K[future]-n["km"])/n["ks"]],-1),dtype=torch.float32,device=device)
                q=forward(head,z,b).cpu().numpy();base=b.cpu().numpy()
            err=float(abs(q-old[f"{seed}_QwenKD23"]).max());maxerror=max(maxerror,err);maxmean=max(maxmean,float(abs((q-base).mean(1)).max()))
            assert err<1e-10
            exported.append(dict(building=name,category=meta["category"],seed=seed,checkpoint=str(path),sha256=sha(path),head_parameters=705,frozen_backbone_parameters=6570,total_parameters=7275,source_checkpoint_sha=sha(case/"head_checkpoints.pt"),replay_max_error=err))
        for row in read(case/"metrics.json"):
            if row["arm"] in ["NoKD","FourierKD","WrongKD","QwenKD23"]:metrics.append(dict(building=name,category=meta["category"],**row))
        print("EXPORTED",name,flush=True)
    summary={}
    for arm in ["NoKD","FourierKD","WrongKD","QwenKD23"]:
        b=[]
        for meta in sel:
            rows=[a for a in metrics if a["building"]==meta["building"] and a["arm"]==arm]
            b.append({k:float(np.mean([a[k] for a in rows])) for k in ["mse","mae","level","shape"]})
        summary[arm]={k:float(np.mean([a[k] for a in b])) for k in ["mse","mae","level","shape"]}
    save(r/"manifest.json",dict(title="ChronoGeom: Distilling Temporal Transformation Laws into Minimal Forecasters",main_variant="Qwen task-conditioned temporal relation response teacher + Shape output KD; single student head",models=exported,inference_requires_LLM=False,inference_requires_teacher_cache=False,inference_requires_switch=False,inference_requires_auxiliary_relation_head=False,parameters_per_student=7275,trainable_head_parameters=705,frozen_backbone_parameters=6570,removed_parameters=710,previous_dual_total_corrected=7985,feature_recipe="Same42analytic/history/calendar student features from34; teacher response features usedONLY duringtraining.",source_run=str(src),git=git()))
    save(r/"summary.json",summary)
    with (r/"by_seed.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    for a in assets:assert sha(a["path"])==a["sha"]
    assert maxmean<1e-12 and time.monotonic()-start<90
    save(r/"verification.json",dict(exported_models=len(exported),end_to_end_replays=len(exported),maximum_prediction_error=maxerror,maximum_window_mean_change=maxmean,only_backbone_and_primary_head_exported=True,all_old_weights_results_SHA_unchanged=True,training_updates=0,LLM_calls=0,gate_calls=0))
    save(r/"status.json",dict(complete=True,seconds=time.monotonic()-start,git=git(),selection_after_development=True))
    print("MAIN_SINGLE_HEAD",summary,flush=True);print("VERIFIED",maxerror,maxmean)
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--source",required=True);a.add_argument("--output",required=True);v=a.parse_args();run(v.source,v.output)

