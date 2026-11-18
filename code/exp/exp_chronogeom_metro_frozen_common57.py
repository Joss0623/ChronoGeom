"""Frozen common39 reader; train only48 existing input-time coefficients."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,time,traceback
from pathlib import Path
import numpy as np,torch
from exp.exp_chronogeom_event_knowledge19 import Head,read,save,sha,git,scores
from exp.exp_chronogeom_teacher_content40 import calendar_arrays,check22
from exp.exp_chronogeom_metro_anchored_qwen53 import csvsave,diagnose
from exp.exp_chronogeom_metro_label_efficiency55 import masked_scores,feature_diag

def pred(model,z39,code):
    return model(torch.cat([z39,code],-1))
def run(c,config):
    torch.set_num_threads(c["CPU_threads"]);torch.use_deterministic_algorithms(True)
    r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);start=time.monotonic();save(r/"config.json",c)
    assets=check22(c)
    for key in ["source51","source53","source54","source55"]:
        assets += [dict(path=str(p),sha=sha(p)) for p in Path(c[key]).iterdir() if p.is_file() and p.suffix in [".json",".pt",".npz",".py"]]
    assets += [dict(path=str(Path(c["source"])/k),sha=sha(Path(c["source"])/k)) for k in ["data.npz","config.json"]]
    save(r/"inputs.json",assets);save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),torch=torch.__version__,numpy=np.__version__))
    root=Path(c["source55"]);source=np.load(Path(c["source"])/"data.npz");p=np.load(root/"predictions.npz")
    old=np.load(Path(c["source53"])/"predictions.npz");split=read(root/"time_split.json");cal=np.array(split["calibration_indices"]);sel=np.array(split["selection_indices"])
    grid,raw,common,kind,h=calendar_arrays(read(Path(c["source"])/"config.json"),"Metro")
    ids=kind*24+h;ci=ids[source["val_future"][cal]];si=ids[source["val_future"][sel]]
    labels=read(root/"label_masks.json")["labels72"];mask=np.isin(source["val_future"][cal],labels["unique_indices"])[:,:,None]
    assert len(labels["unique_indices"])==72 and int(mask.sum())==172
    rawtables=read(Path(c["source54"])/"input_tables.json");qraw=np.array(rawtables["Qwen"])
    qf=read(root/"fixed_feature_tables.json");tables=dict(QwenTaskGeneric=np.array(qf["QwenTaskGeneric"]),MatchedFourier=np.array(qf["MatchedFourier"]))
    permutations={}
    for ss in c["shuffle_seeds"]:
        rng=np.random.default_rng(ss);perm=np.concatenate([24*k+rng.permutation(24) for k in range(3)])
        if ss==54001:assert np.array_equal(perm,rawtables["permutation"])
        a=qraw[perm];fit=a[ci];mu=fit.mean((0,1));rms=max(float(np.sqrt(np.mean((fit-mu)**2))),1e-6)
        tables["Shuffle"+str(ss)]=(a-mu)/rms;permutations[str(ss)]=perm.tolist()
    save(r/"tables.json",{k:v.tolist() for k,v in tables.items()});save(r/"permutations.json",permutations)
    save(r/"label_mask.json",labels);save(r/"time_split.json",split)
    save(r/"protocol_lock.json",dict(config_sha=sha(config),permutation_sha=sha(r/"permutations.json"),table_sha=sha(r/"tables.json"),
        common_steps=100,time_steps=100,all5arms=c["arms"],before_new_training=True,no_selection=True))
    codecal={k:torch.tensor(v[ci],dtype=torch.float32) for k,v in tables.items()}
    codesel={k:torch.tensor(v[si],dtype=torch.float32) for k,v in tables.items()}
    save(r/"feature_diagnostics.json",{k:feature_diag(v,ci) for k,v in tables.items()})
    predictions={};models={};common_states={};records=[];curves=[];changes=[];blockrows=[];effects=[];diags=[]
    initialerr=0.;frozenerr=0.;reloaderr=0.;gradoutside=0.
    for s in c["seeds"]:
        xc=torch.tensor(old[f"{s}_cal_z"][:,:,:39]);xs=torch.tensor(old[f"{s}_selection_z"][:,:,:39])
        y=p[f"{s}_cal_y"];vy=p[f"{s}_selection_y"];a=p[f"{s}_cal_anchor"];av=p[f"{s}_selection_anchor"]
        known=np.zeros_like(y);known[mask]=y[mask]
        yt=torch.tensor(known,dtype=torch.float64);at=torch.tensor(a,dtype=torch.float64);mt=torch.tensor(mask,dtype=torch.float64)
        baseline=masked_scores(a,known,mask)
        zero_c=torch.zeros(len(xc),96,3);zero_s=torch.zeros(len(xs),96,3)
        batches=np.random.default_rng(s).integers(0,len(xc),size=(c["common_steps"]+c["time_steps"],c["batch"]))
        def loss_for(pr,ix):
            e=pr-yt[ix];w=mt[ix]
            return .5*((w*e.square()).sum()/w.sum()/baseline["mse"]+(w*e.abs()).sum()/w.sum()/baseline["mae"])
        common_head=Head(s);opt=torch.optim.Adam(common_head.parameters(),lr=c["lr"])
        for step in range(c["common_steps"]):
            ix=batches[step];pr=at[ix]+pred(common_head,xc[ix],zero_c[ix]);loss=loss_for(pr,ix)
            opt.zero_grad();loss.backward();opt.step()
            curves.append(dict(seed=s,stage="common",arm="Common39Only",step=step+1,loss=float(loss.detach())))
        common_head.eval()
        with torch.no_grad():
            before=av+pred(common_head,xs,zero_s).numpy()
            common_head.net[0].weight[:,39:].zero_()
            base=av+pred(common_head,xs,zero_s).numpy()
        assert np.array_equal(before,base)
        common_states[s]=copy.deepcopy(common_head.state_dict());models[s]={}
        predictions[f"{s}_y"]=vy;predictions[f"{s}_anchor"]=av;predictions[f"{s}_Common39Only"]=base
        records.append(dict(seed=s,arm="Common39Only",time_steps=0,**scores(base,vy)))
        print("COMMON",s,scores(base,vy),flush=True)
        for arm in c["arms"]:
            model=Head(s);model.load_state_dict(common_states[s])
            for pp in model.parameters():pp.requires_grad_(False)
            weight=model.net[0].weight;weight.requires_grad_(True)
            allowed=torch.zeros_like(weight);allowed[:,39:]=1
            handle=weight.register_hook(lambda g:g*allowed)
            optimizer=torch.optim.Adam([weight],lr=c["lr"],weight_decay=0)
            with torch.no_grad():init=av+pred(model,xs,codesel[arm]).numpy()
            initialerr=max(initialerr,float(abs(init-base).max()));assert np.array_equal(init,base)
            states0=copy.deepcopy(model.state_dict());firstgrad=0.
            for step in range(c["time_steps"]):
                if time.monotonic()-start>c["seconds_cap"]:raise TimeoutError("Bounded frozen-common pilot")
                ix=batches[c["common_steps"]+step];pr=at[ix]+pred(model,xc[ix],codecal[arm][ix])
                loss=loss_for(pr,ix);optimizer.zero_grad();loss.backward()
                assert torch.isfinite(weight.grad).all()
                outside=float(abs(weight.grad[:,:39]).max());gradoutside=max(gradoutside,outside);assert outside==0
                if step==0:firstgrad=float(weight.grad[:,39:].norm());assert firstgrad>0
                optimizer.step()
                curves.append(dict(seed=s,stage="time48",arm=arm,step=step+1,loss=float(loss.detach())))
            model.eval()
            with torch.no_grad():pr=av+pred(model,xs,codesel[arm]).numpy()
            state=copy.deepcopy(model.state_dict());models[s][arm]=dict(head=state,table=torch.tensor(tables[arm],dtype=torch.float32))
            for key,v in state.items():
                if key=="net.0.weight":assert torch.equal(v[:,:39],common_states[s][key][:,:39])
                else:assert torch.equal(v,common_states[s][key]),key
            assert torch.count_nonzero(optimizer.state[weight]["exp_avg"][:,:39])==0
            changed=int(torch.count_nonzero(state["net.0.weight"][:,39:]-states0["net.0.weight"][:,39:]))
            changes.append(dict(seed=s,arm=arm,allowed_time_coefficients=48,actually_changed_coefficients=changed,
                outside_changed=0,first_time_gradient_norm=firstgrad,optimizer_parameter_tensor_entries=672,
                note="Gradient mask restricts updates to48; common624entries and other33readercoefficients bitwisefrozen."))
            predictions[f"{s}_{arm}"]=pr;records.append(dict(seed=s,arm=arm,time_steps=100,**scores(pr,vy)))
            diags.append(dict(seed=s,arm=arm,**diagnose(pr,base,vy)))
            for bi,ix in enumerate(np.array_split(np.arange(len(vy)),3)):
                blockrows.append(dict(seed=s,arm=arm,block=bi,windows=len(ix),**scores(pr[ix],vy[ix])))
            for j in range(len(vy)):
                effects.append(dict(seed=s,arm=arm,origin=str(grid[source["val_origin"][sel[j]]]),
                    common_mse=float(np.mean((base[j]-vy[j])**2)),mse=float(np.mean((pr[j]-vy[j])**2)),mae=float(np.mean(abs(pr[j]-vy[j])))))
            m2=Head(s);m2.load_state_dict(state);m2.eval()
            with torch.no_grad():again=av+pred(m2,xs,codesel[arm]).numpy()
            er=float(abs(again-pr).max());reloaderr=max(reloaderr,er);assert er==0
            print("TIME48",s,arm,scores(pr,vy),"changed",changed,flush=True)
    names=["Common39Only"]+c["arms"]
    summary={arm:{k:float(np.mean([v[k] for v in records if v["arm"]==arm])) for k in ["mse","mae","level","shape"]} for arm in names}
    summary["MeanOf3Shuffles"]={k:float(np.mean([summary["Shuffle"+str(ss)][k] for ss in c["shuffle_seeds"]])) for k in ["mse","mae","level","shape"]}
    save(r/"summary.json",summary);save(r/"metrics.json",records);save(r/"curves.json",curves);save(r/"coefficient_checks.json",changes)
    save(r/"block_metrics.json",blockrows);save(r/"correction_diagnostics.json",diags)
    csvsave(r/"metrics.csv",records);csvsave(r/"all_window_effects.csv",effects)
    torch.save(common_states,r/"common_heads.pt");torch.save(dict(models=models,common=common_states,
        source53_normalizations=torch.load(Path(c["source53"])/"teacher_bundle.pt",map_location="cpu",weights_only=True)["normalizations"],
        anchor_manifest=c["frozen22"],source=c["source"]),r/"models.pt")
    np.savez_compressed(r/"predictions.npz",**predictions)
    for aa in assets:assert sha(aa["path"])==aa["sha"],aa["path"]
    check22(c)
    save(r/"verification.json",dict(initial_common_prediction_error=initialerr,frozen_common_weight_error=frozenerr,
        gradient_outside48_error=gradoutside,all15_reload_error=reloaderr,protected_SHA_checks=len(assets),
        all22_51_53_54_55_unchanged=True,only72unique_targets_used=True,outer_data_scored=False,no_selection=True))
    save(r/"status.json",dict(complete=True,seconds=time.monotonic()-start,common_fits=3,time48_fits=15,
        common_updates=300,time48_updates=1500,unique_labels=72,common_steps=100,time_steps=100,
        immutable22_control_fits=0,LLM_calls=0,GPU_loaded=False,student_KD_runs=0,independent_confirmation=False))
    print("SUMMARY",summary,flush=True);print("STATUS",read(r/"status.json"),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);args=ap.parse_args();cc=read(args.config)
    try:run(cc,args.config)
    except Exception:
        rr=Path(cc["run"])
        if rr.exists():(rr/"failure.log").write_text(traceback.format_exc())
        raise
