"""Internal-only Qwen innovation teacher on immutable version22 Fourier predictions."""
import os
os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,csv,time,traceback
from pathlib import Path
import numpy as np
import torch
from exp.exp_chronogeom_event_knowledge19 import Head,forward,knowledge,read,save,sha,git,scores
from exp.exp_chronogeom_teacher_content40 import calendar_arrays,check22
from exp.exp_chronogeom_residual03 import numeric_summary
from exp.exp_chronogeom_capital_representation_transfer44 import CalendarMap

def C(v):
    return v-v.mean(1,keepdims=True)
def csvsave(path,rows):
    if rows:
        with Path(path).open("w") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def feature_transform(x,anchor,raw,common,ix,table,normalization=None):
    ft=np.concatenate([numeric_summary(x,torch.tensor(anchor,dtype=torch.float32)).numpy(),raw[ix],common[ix]],-1)
    kt=table[ix]
    if normalization is None:
        normalization=dict(mean=ft.mean((0,1)).tolist(),sd=np.maximum(ft.std((0,1)),1e-3).tolist(),
                           kmean=kt.mean((0,1)).tolist(),krms=max(float(np.sqrt(np.mean((kt-kt.mean((0,1)))**2))),1e-6))
    nm=normalization
    z39=(ft-np.array(nm["mean"]))/np.array(nm["sd"])
    z3=(kt-np.array(nm["kmean"]))/nm["krms"]
    return torch.tensor(np.concatenate([z39,z3],-1),dtype=torch.float32),normalization
def diagnose(pred,anchor,y):
    e=anchor-y;d=pred-anchor
    old=(e**2).mean((1,2));new=((pred-y)**2).mean((1,2));diff=new-old
    return dict(mse_delta=float(diff.mean()),cross_term=float(2*np.mean(e*d)),energy=float(np.mean(d*d)),
        mse_identity_error=float(abs(diff.mean()-(2*np.mean(e*d)+np.mean(d*d)))),
        helped_windows=int((diff<0).sum()),harmed_windows=int((diff>0).sum()),unchanged_windows=int((diff==0).sum()),
        benefit_on_helped=float(np.maximum(-diff,0).mean()),harm_on_harmed=float(np.maximum(diff,0).mean()),
        correction_rms=float(np.sqrt(np.mean(d*d))),mean_preservation_error=float(abs(d.mean(1)).max()),
        error_correlation=float(np.corrcoef(e.reshape(-1),(pred-y).reshape(-1))[0,1]))
def execute(c,config):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c)
    started=time.monotonic()
    assets=check22(c)
    for root in [Path(c["source51"]),Path(c["teacher40"])/"Metro"]:
        assets += [dict(path=str(p),sha=sha(p)) for p in root.iterdir() if p.is_file() and p.suffix in [".json",".pt",".npz"]]
    assets += [dict(path=str(Path(c["source"])/k),sha=sha(Path(c["source"])/k)) for k in ["data.npz","predictions.npz","normalizers.json","knowledge.npz","config.json"]]
    save(r/"input_assets.json",assets)
    save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),torch=torch.__version__,numpy=np.__version__,CPU_threads=2))
    # Synthetic invariants; only final layer is zero, so learning can start.
    syn=Head(2024);zz=torch.randn(2,96,42);target=torch.randn(2,96,1,dtype=torch.float64)
    z0=syn(zz);assert torch.count_nonzero(z0)==0
    (z0-target).square().mean().backward()
    assert float(syn.net[0].weight.abs().sum())>0 and float(syn.net[2].weight.grad.abs().sum())>0
    save(r/"preflight.json",dict(parameters=sum(p.numel() for p in syn.parameters()),initial_zero=True,
        last_weight_gradient_norm=float(syn.net[2].weight.grad.norm()),first_weights_nonzero=True))
    src=Path(c["source"]);p=np.load(src/"data.npz");cache=np.load(src/"predictions.npz")
    f22=read(c["frozen22"]);fr=Path(f22["run"]);a22=np.load(fr/"predictions.npz")
    states22=torch.load(fr/"selected_students.pt",map_location="cpu",weights_only=True)["Metro"]
    grid,raw,common,kind,h=calendar_arrays(read(src/"config.json"),"Metro")
    split=read(c["split52"]);cal=np.array(split["calibration_indices"]);sel=np.array(split["selection_indices"])
    assert len(cal)==9 and len(sel)==16 and not set(cal)&set(sel)
    assert grid[p["fit_future"].max()]<grid[p["val_future"][cal].min()]
    assert grid[p["val_future"][cal].max()]<grid[p["val_origin"][sel].min()-96]
    origins=p["val_origin"];future=p["val_future"]
    blocks=np.array_split(np.arange(len(sel)),c["selection_blocks"])
    save(r/"time_split.json",dict(calibration_indices=cal.tolist(),selection_indices=sel.tolist(),
        calibration_origins=grid[origins[cal]].astype(str).tolist(),selection_origins=grid[origins[sel]].astype(str).tolist(),
        selection_blocks=[b.tolist() for b in blocks],anchor_fit_target_end=str(grid[p["fit_future"].max()]),
        calibration_after_anchor_forecasthead_fit=True,development_only=True,
        anchor_hyperparameters_previously_used_old_validation=True,outer_arrays_accessed=False))
    profiles=read(Path(c["teacher40"])/"Metro/profiles.json")
    table=knowledge(np.array(profiles["Task"]),kind,h)-knowledge(np.array(profiles["Generic"]),kind,h)
    canonical_kind=np.repeat(np.arange(3),24);canonical_hour=np.tile(np.arange(24),3)
    t72=knowledge(np.array(profiles["Task"]),canonical_kind,canonical_hour)-knowledge(np.array(profiles["Generic"]),canonical_kind,canonical_hour)
    ks=np.load(src/"knowledge.npz");norms=read(src/"normalizers.json");data={};anchor_errors=[]
    for s in c["seeds"]:
        x=torch.tensor(p["val_x"],dtype=torch.float32);bp=torch.tensor(cache[f"{s}_val_FrozenTimeBase"],dtype=torch.float32)
        pp=norms[str(s)]["Fourier"];ix=future
        ft=np.concatenate([numeric_summary(x,bp).numpy(),raw[ix],common[ix]],-1)
        kfit=ks["Fourier"][p["fit_future"]];km=kfit.mean((0,1));kr=max(float(np.sqrt(np.mean((kfit-km)**2))),1e-6)
        zold=torch.tensor(np.concatenate([(ft-np.array(pp["mean"]))/np.array(pp["sd"]),(ks["Fourier"][ix]-km)/kr],-1),dtype=torch.float32)
        anchorhead=Head(s);anchorhead.load_state_dict(states22[s]["FourierKD"]);anchorhead.eval()
        for v in anchorhead.parameters():v.requires_grad_(False)
        with torch.no_grad():replay=forward(anchorhead,zold,bp).numpy()
        anchor=a22[f"Metro_{s}_val_FourierKD"]
        err=float(abs(replay-anchor).max());assert err<3e-6;anchor_errors.append(dict(seed=s,max_error=err))
        assert np.array_equal(p["val_y"],a22[f"Metro_{s}_val_y"])
        zc,nm=feature_transform(x[cal],anchor[cal],raw,common,ix[cal],table)
        zs,_=feature_transform(x[sel],anchor[sel],raw,common,ix[sel],table,nm)
        data[s]=dict(cal_z=zc,selection_z=zs,cal_anchor=anchor[cal],selection_anchor=anchor[sel],
            cal_y=p["val_y"][cal],selection_y=p["val_y"][sel],normalization=nm)
    save(r/"anchor_replay.json",anchor_errors)
    states={};curves=[];preds={};selected_data={};norm_records={};initialerr=0.
    for s,d in data.items():
        model=Head(s);assert sum(v.numel() for v in model.parameters())==705
        opt=torch.optim.Adam(model.parameters(),lr=c["lr"]);rng=np.random.default_rng(s)
        batches=rng.integers(0,len(cal),size=(c["steps"],c["batch"]))
        states[s]={};norm_records[s]=d["normalization"];anchor_score=scores(d["cal_anchor"],d["cal_y"])
        ac=torch.tensor(d["cal_anchor"],dtype=torch.float64);yc=torch.tensor(d["cal_y"],dtype=torch.float64)
        for step in range(c["steps"]+1):
            if time.monotonic()-started>c["seconds_cap"]:raise TimeoutError("Bounded teacher pilot cap")
            if step in c["checkpoints"]:
                states[s][step]=copy.deepcopy(model.state_dict());model.eval()
                for label in ["cal","selection"]:
                    with torch.no_grad():pr=d[label+"_anchor"]+model(d[label+"_z"]).numpy()
                    preds[f"{s}_{step}_{label}"]=pr
                    if step==0:
                        e=float(abs(pr-d[label+"_anchor"]).max());initialerr=max(initialerr,e);assert e==0
                    masks=[np.arange(len(pr))] if label=="cal" else blocks
                    for bi,mask in enumerate(masks):
                        curves.append(dict(seed=s,step=step,split=label,block=bi,windows=len(mask),**scores(pr[mask],d[label+"_y"][mask])))
            if step==c["steps"]:break
            model.train();ixb=batches[step];prediction=ac[ixb]+model(d["cal_z"][ixb]);err=prediction-yc[ixb]
            loss=.5*(err.square().mean()/anchor_score["mse"]+err.abs().mean()/anchor_score["mae"])
            opt.zero_grad();loss.backward()
            assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
            opt.step()
        print("TEACHER_FITTED",s,"updates",c["steps"],flush=True)
    options=[]
    for step in c["checkpoints"]:
        risks=[]
        for s,d in data.items():
            rs=[]
            for bi,b in enumerate(blocks):
                base=scores(d["selection_anchor"][b],d["selection_y"][b])
                q=next(v for v in curves if v["seed"]==s and v["step"]==step and v["split"]=="selection" and v["block"]==bi)
                rs.append(.5*(q["mse"]/base["mse"]+q["mae"]/base["mae"]))
            risks.append(rs)
        risks=np.array(risks);options.append(dict(step=step,score=float((risks.mean()+risks.mean(1).max()+risks.mean(0).max())/3),
            mean_risk=float(risks.mean()),worst_seed=float(risks.mean(1).max()),worst_block=float(risks.mean(0).max())))
    chosen=min(options,key=lambda v:(v["score"],v["step"]))["step"]
    save(r/"selection.json",dict(chosen_step=chosen,shared_seeds=c["seeds"],options=options,outer_data_used=False))
    torch.save(states,r/"all_teacher_checkpoints.pt")
    teacher_bundle=dict(normalizations=norm_records,table72=torch.tensor(t72),heads={s:states[s][chosen] for s in c["seeds"]},
                        frozen22_anchor_heads={s:states22[s]["FourierKD"] for s in c["seeds"]},
                        anchor_manifest=c["frozen22"],anchor_source=c["source"],teacher_source=c["teacher40"])
    torch.save(teacher_bundle,r/"teacher_bundle.pt")
    save(r/"teacher_lock.json",dict(model_sha=sha(r/"teacher_bundle.pt"),selection_sha=sha(r/"selection.json"),
        config_sha=sha(config),source_commit=git(),before_any_outer_scoring=True))
    rows=[];diagnostics=[];blockrows=[];windowrows=[];reloaderr=0.;helped=0
    for s,d in data.items():
        for label in ["cal","selection"]:
            y=d[label+"_y"];anchor=d[label+"_anchor"];pred=preds[f"{s}_{chosen}_{label}"]
            for name,a in [("Frozen22FourierKD",anchor),("QwenInnovationTeacher",pred)]:
                rows.append(dict(seed=s,split=label,model=name,**scores(a,y)))
            diagnostics.append(dict(seed=s,split=label,**diagnose(pred,anchor,y)))
            ids=cal if label=="cal" else sel
            for j in range(len(y)):
                windowrows.append(dict(seed=s,split=label,origin=str(grid[origins[ids[j]]]),
                    anchor_mse=float(np.mean((anchor[j]-y[j])**2)),teacher_mse=float(np.mean((pred[j]-y[j])**2)),
                    anchor_mae=float(np.mean(abs(anchor[j]-y[j]))),teacher_mae=float(np.mean(abs(pred[j]-y[j])))))
        # Recomputed fixed controls on exact same selection subset, never refitted.
        for arm in ["NoKD","ShuffleKD"]:
            pr=a22[f"Metro_{s}_val_{arm}"][sel]
            rows.append(dict(seed=s,split="selection",model="Fixed22"+arm,**scores(pr,d["selection_y"])))
        for bi,b in enumerate(blocks):
            for name,pr in [("Frozen22FourierKD",d["selection_anchor"]),("QwenInnovationTeacher",preds[f"{s}_{chosen}_selection"])]:
                blockrows.append(dict(seed=s,block=bi,windows=len(b),model=name,**scores(pr[b],d["selection_y"][b])))
        reload=Head(s);reload.load_state_dict(torch.load(r/"teacher_bundle.pt",map_location="cpu",weights_only=True)["heads"][s]);reload.eval()
        with torch.no_grad():rp=d["selection_anchor"]+reload(d["selection_z"]).numpy()
        reloaderr=max(reloaderr,float(abs(rp-preds[f"{s}_{chosen}_selection"]).max()))
    summary={}
    for label in ["cal","selection"]:
        summary[label]={name:{k:float(np.mean([v[k] for v in rows if v["split"]==label and v["model"]==name])) for k in ["mse","mae","level","shape"]}
                        for name in dict.fromkeys(v["model"] for v in rows if v["split"]==label)}
    a=summary["selection"]["Frozen22FourierKD"];b=summary["selection"]["QwenInnovationTeacher"]
    seedwins=sum(scores(preds[f"{s}_{chosen}_selection"],d["selection_y"])["mse"]<scores(d["selection_anchor"],d["selection_y"])["mse"] for s,d in data.items())
    blockwins=0
    for bi in range(len(blocks)):
        aa=np.mean([v["mse"] for v in blockrows if v["block"]==bi and v["model"]=="Frozen22FourierKD"])
        bb=np.mean([v["mse"] for v in blockrows if v["block"]==bi and v["model"]=="QwenInnovationTeacher"])
        blockwins+=int(bb<aa)
    passed=bool(chosen>0 and b["mse"]<a["mse"] and b["mae"]<a["mae"] and seedwins>=2 and blockwins>=2)
    gate=dict(pass_=passed,chosen_step=chosen,selection_anchor=a,selection_teacher=b,MSE_improved_seeds=seedwins,
              MSE_improved_blocks=blockwins,independent_confirmation=False,LLM_specificity_established=False)
    save(r/"teacher_gate.json",gate)
    kdrows=[];kdinfo=[]
    if passed:
        for s,d in data.items():
            mapper=CalendarMap(s);opt=torch.optim.Adam(mapper.parameters(),lr=.05)
            target=torch.tensor((t72-np.array(d["normalization"]["kmean"]))/d["normalization"]["krms"],dtype=torch.float32)
            for _ in range(200):
                loss=(mapper(torch.arange(72))-target).square().mean();opt.zero_grad();loss.backward();opt.step()
            head=Head(s);head.load_state_dict(states[s][chosen]);head.eval();mapper.eval()
            ids=torch.tensor((kind*24+h)[future[sel]],dtype=torch.long)
            with torch.no_grad():
                z=torch.cat([d["selection_z"][:,:,:39],mapper(ids)],-1)
                pp=d["selection_anchor"]+head(z).numpy()
                me=float((mapper(torch.arange(72))-target).square().mean())
            teacher=preds[f"{s}_{chosen}_selection"]
            preds[f"{s}_compiled_selection"]=pp
            kdrows.append(dict(seed=s,split="selection",model="CalendarCompiledStudent",**scores(pp,d["selection_y"])))
            kdinfo.append(dict(seed=s,map_steps=200,map_parameters=216,reader_inherited=True,forecast_label_updates=0,
                map_mse=me,max_prediction_deviation=float(abs(pp-teacher).max())))
            torch.save(dict(mapper=mapper.state_dict(),head=head.state_dict(),normalization=d["normalization"],anchor_seed=s,
                 anchor=states22[s]["FourierKD"],anchor_source=c["source"],original_frozen22_manifest=c["frozen22"]),r/f"compiled_student_{s}.pt")
    for s,d in data.items():
        for label in ["cal","selection"]:
            preds[f"{s}_{label}_y"]=d[label+"_y"];preds[f"{s}_{label}_anchor"]=d[label+"_anchor"]
            preds[f"{s}_{label}_z"]=d[label+"_z"].numpy()
    np.savez_compressed(r/"predictions.npz",**preds)
    save(r/"teacher_metrics.json",rows);save(r/"teacher_summary.json",summary);save(r/"correction_diagnostics.json",diagnostics)
    save(r/"block_metrics.json",blockrows);save(r/"curves.json",curves);save(r/"compilation.json",dict(performed=passed,metrics=kdrows,diagnostics=kdinfo))
    csvsave(r/"teacher_metrics.csv",rows);csvsave(r/"all_window_effects.csv",windowrows);csvsave(r/"block_metrics.csv",blockrows)
    for v in assets:assert sha(v["path"])==v["sha"],v["path"]
    check22(c)
    assert reloaderr==0 and initialerr==0
    assert max(v["mean_preservation_error"] for v in diagnostics)<1e-10
    save(r/"verification.json",dict(protected_SHA_checks=len(assets),fixed22_unchanged=True,source51_unchanged=True,
        initial_anchor_error=initialerr,reload_error=reloaderr,mean_preservation_error=max(v["mean_preservation_error"] for v in diagnostics),
        mse_identity_error=max(v["mse_identity_error"] for v in diagnostics),no_outer_scoring=True))
    save(r/"status.json",dict(complete=True,seconds=time.monotonic()-started,teacher_fits=3,teacher_updates=1800,
        head_parameters=705,shared_checkpoint=chosen,KD_performed=passed,KD_map_updates=600 if passed else 0,
        KD_forecasting_updates=0,NoKD_Fourier_Shuffle_fits=0,LLM_calls=0,GPU_loaded=False,independent_confirmation=False,
        outer_predictions_or_targets_accessed=False))
    print("TEACHER_GATE",gate,flush=True);print("SUMMARY",summary,flush=True)
    print("STATUS",read(r/"status.json"),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);args=ap.parse_args()
    cc=read(args.config)
    try: execute(cc,args.config)
    except Exception:
        rr=Path(cc["run"])
        if rr.exists(): (rr/"failure.log").write_text(traceback.format_exc())
        raise
