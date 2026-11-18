"""Matched innovation-head ablations; does not refit any immutable22 comparator."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,time,traceback
from pathlib import Path
import numpy as np,torch
from exp.exp_chronogeom_event_knowledge19 import Head,read,save,sha,git,scores
from exp.exp_chronogeom_teacher_content40 import calendar_arrays,check22
from exp.exp_chronogeom_metro_anchored_qwen53 import diagnose,csvsave

def rankdiag(z):
    a=z.reshape(-1,z.shape[-1]).astype(float);a-=a.mean(0)
    sv=np.linalg.svd(a,compute_uv=False)
    return dict(dimensions=z.shape[-1],joint_RMS=float(np.sqrt(np.mean(a*a))),
                effective_rank=int(np.sum(sv>max(sv[0],1e-12)*1e-6)),singular_values=sv.tolist())
def run(c,config):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);start=time.monotonic()
    r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c)
    root=Path(c["source53"]);assets=check22(c)
    for src in [root,Path(c["source51"])]:
        assets += [dict(path=str(p),sha=sha(p)) for p in src.iterdir() if p.is_file() and p.suffix in [".json",".npz",".pt",".py"]]
    source=Path(c["source"])
    assets += [dict(path=str(source/k),sha=sha(source/k)) for k in ["config.json","data.npz","knowledge.npz"]]
    save(r/"inputs.json",assets);save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config)))
    old=np.load(root/"predictions.npz");old_bundle=torch.load(root/"teacher_bundle.pt",map_location="cpu",weights_only=True)
    split=read(root/"time_split.json");cal=np.array(split["calibration_indices"]);sel=np.array(split["selection_indices"])
    p=np.load(source/"data.npz");ks=np.load(source/"knowledge.npz");grid,_,_,kind,h=calendar_arrays(read(source/"config.json"),"Metro")
    cid=kind*24+h;ci=cid[p["val_future"][cal]];si=cid[p["val_future"][sel]]
    assert np.array_equal(old["2024_cal_y"],p["val_y"][cal]) and np.array_equal(old["2024_selection_y"],p["val_y"][sel])
    qraw=old_bundle["table72"].numpy();rng=np.random.default_rng(c["shuffle_seed"])
    perm=np.concatenate([24*k+rng.permutation(24) for k in range(3)]);shraw=qraw[perm]
    hh=np.tile(np.arange(24),3);fraw=np.stack([np.sin(10*np.pi*hh/24),np.cos(10*np.pi*hh/24),np.cos(12*np.pi*hh/24)],1)
    assert np.array_equal(fraw[cid],ks["Fourier"])
    tables=dict(MatchedFourierInnovation=fraw,MatchedShuffledQwenInnovation=shraw)
    save(r/"input_tables.json",dict(Fourier=fraw.tolist(),Qwen=qraw.tolist(),shuffled=shraw.tolist(),permutation=perm.tolist()))
    save(r/"protocol_lock.json",dict(deploy_step=c["deploy_step"],Qwen_model_sha=sha(root/"teacher_bundle.pt"),
        source53_step=read(root/"selection.json")["chosen_step"],no_reselection=True,config_sha=sha(config),before_new_fitting=True))
    assert read(root/"selection.json")["chosen_step"]==c["deploy_step"]
    data={};norms={};featurediag=[];maxfeatureerr=0.
    for s in c["seeds"]:
        d={label:dict(anchor=old[f"{s}_{label}_anchor"],y=old[f"{s}_{label}_y"],common=torch.tensor(old[f"{s}_{label}_z"][:,:,:39]))
           for label in ["cal","selection"]}
        data[s]=d;norms[s]={}
        # Prove that the source53 reader sees exactly the corresponding cached Qwen features.
        nm=old_bundle["normalizations"][s]
        recon=(qraw[ci]-np.array(nm["kmean"]))/nm["krms"]
        assert np.array_equal(recon.astype(np.float32),old[f"{s}_cal_z"][:,:,39:])
        featurediag.append(dict(seed=s,arm="Qwen53",**rankdiag(old[f"{s}_cal_z"][:,:,39:])))
        for arm,table in tables.items():
            k=table[ci];mean=k.mean((0,1));rms=max(float(np.sqrt(np.mean((k-mean)**2))),1e-6)
            norms[s][arm]=dict(mean=mean.tolist(),rms=rms)
            for label,ids in [("cal",ci),("selection",si)]:
                z=torch.cat([d[label]["common"],torch.tensor((table[ids]-mean)/rms,dtype=torch.float32)],-1)
                assert np.array_equal(z[:,:,:39].numpy(),old[f"{s}_{label}_z"][:,:,:39])
                d[label][arm]=z
            featurediag.append(dict(seed=s,arm=arm,**rankdiag(d["cal"][arm].numpy()[:,:,39:])))
    states={};preds={};curves=[];initialerr=0.;lastgrad=[]
    for s,d in data.items():
        states[s]={};batches=np.random.default_rng(s).integers(0,len(cal),size=(c["steps"],c["batch"]))
        base=scores(d["cal"]["anchor"],d["cal"]["y"]);a=torch.tensor(d["cal"]["anchor"],dtype=torch.float64);y=torch.tensor(d["cal"]["y"],dtype=torch.float64)
        for arm in c["new_arms"]:
            model=Head(s);opt=torch.optim.Adam(model.parameters(),lr=c["lr"]);states[s][arm]={}
            assert sum(v.numel() for v in model.parameters())==705
            assert all(torch.equal(v,states53) for v,states53 in zip(model.state_dict().values(),torch.load(root/"all_teacher_checkpoints.pt",map_location="cpu",weights_only=True)[s][0].values()))
            for step in range(c["steps"]+1):
                if time.monotonic()-start>c["seconds_cap"]:raise TimeoutError("Bounded ablation cap")
                if step%100==0:
                    states[s][arm][step]=copy.deepcopy(model.state_dict());model.eval()
                    for label in ["cal","selection"]:
                        with torch.no_grad():pr=d[label]["anchor"]+model(d[label][arm]).numpy()
                        preds[f"{s}_{arm}_{step}_{label}"]=pr
                        if step==0:
                            e=float(abs(pr-d[label]["anchor"]).max());initialerr=max(initialerr,e);assert e==0
                        curves.append(dict(seed=s,arm=arm,step=step,split=label,**scores(pr,d[label]["y"])))
                if step==c["steps"]:break
                model.train();ix=batches[step];err=a[ix]+model(d["cal"][arm][ix])-y[ix]
                loss=.5*(err.square().mean()/base["mse"]+err.abs().mean()/base["mae"])
                opt.zero_grad();loss.backward()
                assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
                if step==0:lastgrad.append(dict(seed=s,arm=arm,norm=float(model.net[2].weight.grad.norm())))
                opt.step()
            print("FITTED",s,arm,"600updates/fixed100",flush=True)
    rows=[];diag=[];blockrows=[];windowrows=[];reloaderr=0.;step=c["deploy_step"]
    names=["Frozen22FourierAnchor","Qwen53"]+c["new_arms"]
    for s,d in data.items():
        for label in ["cal","selection"]:
            outs=dict(Frozen22FourierAnchor=d[label]["anchor"],Qwen53=old[f"{s}_{step}_{label}"])
            outs.update({arm:preds[f"{s}_{arm}_{step}_{label}"] for arm in c["new_arms"]})
            for arm,pr in outs.items():
                rows.append(dict(seed=s,split=label,arm=arm,**scores(pr,d[label]["y"])))
                diag.append(dict(seed=s,split=label,arm=arm,**diagnose(pr,d[label]["anchor"],d[label]["y"])))
                preds[f"{s}_{arm}_{label}"]=pr
            if label=="selection":
                for bi,b in enumerate(np.array_split(np.arange(len(sel)),3)):
                    for arm,pr in outs.items():blockrows.append(dict(seed=s,block=bi,windows=len(b),arm=arm,**scores(pr[b],d[label]["y"][b])))
                for j in range(len(sel)):
                    for arm,pr in outs.items():
                        windowrows.append(dict(seed=s,origin=str(grid[p["val_origin"][sel[j]]]),arm=arm,
                            mse=float(np.mean((pr[j]-d[label]["y"][j])**2)),mae=float(np.mean(abs(pr[j]-d[label]["y"][j])))))
            preds[f"{s}_{label}_y"]=d[label]["y"]
        for arm in c["new_arms"]:
            head=Head(s);head.load_state_dict(states[s][arm][step]);head.eval()
            with torch.no_grad():rr=d["selection"]["anchor"]+head(d["selection"][arm]).numpy()
            reloaderr=max(reloaderr,float(abs(rr-preds[f"{s}_{arm}_selection"]).max()))
            torch.save(dict(head=head.state_dict(),normalization=norms[s][arm],table=torch.tensor(tables[arm]),
                common_source53=str(root),frozen22_manifest=c["frozen22"]),r/f"{s}_{arm}.pt")
    summary={label:{arm:{k:float(np.mean([v[k] for v in rows if v["split"]==label and v["arm"]==arm])) for k in ["mse","mae","level","shape"]}
                    for arm in names} for label in ["cal","selection"]}
    comparisons=[]
    for arm in c["new_arms"]:
        a=summary["selection"]["Qwen53"];b=summary["selection"][arm]
        comparisons.append(dict(comparator=arm,Qwen_minus_MSE=a["mse"]-b["mse"],Qwen_minus_MAE=a["mae"]-b["mae"],
            Qwen_MSE_better_seeds=sum(next(v["mse"] for v in rows if v["seed"]==s and v["split"]=="selection" and v["arm"]=="Qwen53")<
                next(v["mse"] for v in rows if v["seed"]==s and v["split"]=="selection" and v["arm"]==arm) for s in c["seeds"])))
    torch.save(states,r/"all_checkpoints.pt");np.savez_compressed(r/"predictions.npz",**preds)
    save(r/"metrics.json",rows);save(r/"summary.json",summary);save(r/"comparison.json",comparisons);save(r/"curves.json",curves)
    save(r/"feature_diagnostics.json",featurediag);save(r/"normalizations.json",norms);save(r/"correction_diagnostics.json",diag);save(r/"block_metrics.json",blockrows)
    csvsave(r/"metrics.csv",rows);csvsave(r/"all_window_effects.csv",windowrows)
    for a in assets:assert sha(a["path"])==a["sha"],a["path"]
    check22(c);assert reloaderr==0
    save(r/"verification.json",dict(protected_SHA_checks=len(assets),fixed22_unchanged=True,versions51_53_unchanged=True,
        initial_anchor_error=initialerr,head_reload_error=reloaderr,common39_byte_equal=True,initial_heads_equal=True,
        gradient_checks=lastgrad,mean_preservation_error=max(v["mean_preservation_error"] for v in diag),
        no_checkpoint_reselection=True,outer_arrays_scored=False))
    save(r/"status.json",dict(complete=True,seconds=time.monotonic()-start,new_innovation_fits=6,new_updates=3600,
        immutable22_control_fits=0,Qwen53_refits=0,deploy_step=100,LLM_calls=0,GPU_loaded=False,
        independent_confirmation=False,new_KD=0))
    print("SUMMARY",summary,flush=True);print("COMPARISONS",comparisons,flush=True);print("STATUS",read(r/"status.json"),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args();c=read(a.config)
    try:run(c,a.config)
    except Exception:
        r=Path(c["run"])
        if r.exists():(r/"failure.log").write_text(traceback.format_exc())
        raise
