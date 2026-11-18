"""Three supervision budgets for fixed53 conditional shape heads; internal-only."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,copy,time,traceback
from pathlib import Path
import numpy as np,torch
from exp.exp_chronogeom_event_knowledge19 import Head,read,save,sha,git,scores
from exp.exp_chronogeom_teacher_content40 import calendar_arrays,check22
from exp.exp_chronogeom_capital_representation_transfer44 import CalendarMap
from exp.exp_chronogeom_metro_anchored_qwen53 import csvsave,diagnose

def code_values(mapper,trainable,frequency):
    raw=mapper.codes.weight
    if not trainable:return raw
    mean=(frequency[:,None]*raw).sum(0)
    z=raw-mean
    rms=torch.sqrt((frequency[:,None]*z.square()).sum()/3).clamp_min(1e-6)
    return z/rms
def output(head,mapper,z39,ids,trainable,frequency):
    z3=code_values(mapper,trainable,frequency)[ids]
    return head(torch.cat([z39,z3],-1))
def masked_scores(pred,y,mask):
    m=np.asarray(mask,bool);e=(pred-y)[m]
    return dict(mse=float(np.mean(e*e)),mae=float(np.mean(abs(e))))
def feature_diag(table,ids):
    a=table[ids].reshape(-1,3).astype(float);mu=a.mean(0);sv=np.linalg.svd(a-mu,compute_uv=False)
    return dict(joint_RMS=float(np.sqrt(np.mean((a-mu)**2))),mean=mu.tolist(),
        effective_rank=int(np.sum(sv>max(float(sv[0]),1e-12)*1e-6)),singular_values=sv.tolist())
def run(c,config):
    torch.set_num_threads(c["CPU_threads"]);torch.use_deterministic_algorithms(True)
    r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);start=time.monotonic();save(r/"config.json",c)
    assets=check22(c)
    for root in [Path(c["source51"]),Path(c["source53"]),Path(c["source54"])]:
        assets += [dict(path=str(p),sha=sha(p)) for p in root.iterdir() if p.is_file() and p.suffix in [".json",".npz",".pt",".py"]]
    src=Path(c["source"])
    assets += [dict(path=str(src/k),sha=sha(src/k)) for k in ["data.npz","config.json"]]
    save(r/"inputs.json",assets);save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),torch=torch.__version__,numpy=np.__version__))
    root=Path(c["source53"]);old=np.load(root/"predictions.npz");p=np.load(src/"data.npz")
    split=read(root/"time_split.json");cal=np.array(split["calibration_indices"]);sel=np.array(split["selection_indices"])
    grid,raw,common,kind,h=calendar_arrays(read(src/"config.json"),"Metro")
    ci=(kind*24+h)[p["val_future"][cal]];si=(kind*24+h)[p["val_future"][sel]]
    targetids=p["val_future"][cal];unique=np.unique(targetids);assert len(unique)==360 and len(cal)==9 and len(sel)==16
    frequency=torch.tensor(np.bincount(ci.reshape(-1),minlength=72)/ci.size,dtype=torch.float32)
    groups=[]
    for k in sorted(np.unique(kind[unique])):
        for hh in range(24):
            ids=unique[(kind[unique]==k)&(h[unique]==hh)]
            ids=np.random.default_rng(c["label_seed"]+int(k)*100+hh).permutation(ids)
            assert len(ids)==(10 if k==0 else 5)
            groups.append((int(k),hh,ids))
    masks={};mask_records={};strata=[];prev=set()
    for b in c["budgets"]:
        label=b["name"];chosen=np.concatenate([ids[:int(round(len(ids)*b["fraction"]))] for k,hh,ids in groups])
        selected=set(map(int,chosen));assert prev.issubset(selected);prev=selected
        assert len(selected)==b["unique_target_hours"]
        mask=np.isin(targetids,np.sort(chosen))[:,:,None];masks[label]=mask
        counts={int(u):int(np.sum(targetids==u)) for u in chosen}
        mask_records[label]=dict(unique_target_hours=len(selected),window_label_positions=int(mask.sum()),
            unique_indices=sorted(selected),unique_timestamps=grid[np.sort(chosen)].astype(str).tolist(),
            overlapping_label_multiplicity={str(k):v for k,v in counts.items()},observed_histories=9,
            complete_target_windows=int(mask.all(1).sum()),hidden_target_positions=int((~mask).sum()))
        for k,hh,ids in groups:
            strata.append(dict(budget=label,daytype=k,hour=hh,available=len(ids),labeled=sum(int(u) in selected for u in ids)))
    save(r/"label_masks.json",mask_records);save(r/"label_strata.json",strata)
    save(r/"time_split.json",split)
    save(r/"protocol_lock.json",dict(config_sha=sha(config),mask_sha=sha(r/"label_masks.json"),deploy_step=100,
        arms=c["arms"],no_new_selection=True,before_new_fitting=True))
    tables=read(Path(c["source54"])/"input_tables.json")
    rawtables=dict(QwenTaskGeneric=np.array(tables["Qwen"]),MatchedFourier=np.array(tables["Fourier"]),ShuffledQwen=np.array(tables["shuffled"]))
    fixed={};norms={}
    for arm,table in rawtables.items():
        k=table[ci];mu=k.mean((0,1));rms=max(float(np.sqrt(np.mean((k-mu)**2))),1e-6)
        fixed[arm]=(table-mu)/rms;norms[arm]=dict(mean=mu.tolist(),rms=rms)
    save(r/"fixed_feature_tables.json",{k:v.tolist() for k,v in fixed.items()});save(r/"feature_normalizations.json",norms)
    predictions={};allstates={};curves=[];rows=[];blocks=[];windowrows=[];diags=[];featurediags=[];checks=[]
    full_replay=0.;initialerr=0.;maxreload=0.;label_gradient_error=0.
    for s in c["seeds"]:
        common_cal=torch.tensor(old[f"{s}_cal_z"][:,:,:39]);common_sel=torch.tensor(old[f"{s}_selection_z"][:,:,:39])
        y=old[f"{s}_cal_y"];vy=old[f"{s}_selection_y"];anchor=old[f"{s}_cal_anchor"];va=old[f"{s}_selection_anchor"]
        assert np.array_equal(fixed["QwenTaskGeneric"][ci].astype(np.float32),old[f"{s}_cal_z"][:,:,39:])
        assert np.array_equal(y,p["val_y"][cal]) and np.array_equal(vy,p["val_y"][sel])
        batch=np.random.default_rng(s).integers(0,9,size=(c["steps"],c["batch"]))
        ac=torch.tensor(anchor,dtype=torch.float64);ct=torch.tensor(ci,dtype=torch.long);st=torch.tensor(si,dtype=torch.long)
        predictions[f"{s}_selection_y"]=vy;predictions[f"{s}_selection_anchor"]=va
        predictions[f"{s}_cal_y"]=y;predictions[f"{s}_cal_anchor"]=anchor
        allstates[s]={}
        for b in c["budgets"]:
            label=b["name"];mask=masks[label];mt=torch.tensor(mask,dtype=torch.float64)
            # Hidden training targets are replaced BEFORE forming any training tensor/normalizer.
            known=np.zeros_like(y);known[mask]=y[mask];yt=torch.tensor(known,dtype=torch.float64)
            anchor_loss=masked_scores(anchor,known,mask)
            assert anchor_loss["mse"]>0 and anchor_loss["mae"]>0
            allstates[s][label]={}
            # Check withheld-target perturbations cannot affect the supervised gradient.
            probe=torch.tensor(anchor,dtype=torch.float64,requires_grad=True)
            loss=(mt*(probe-yt).square()).sum()/mt.sum()
            grad=torch.autograd.grad(loss,probe)[0]
            if (~mask).any():label_gradient_error=max(label_gradient_error,float(abs(grad[~mask]).max()))
            for arm in c["arms"]:
                trainable=arm=="LearnableCalendar";mapper=CalendarMap(s);head=Head(s)
                if not trainable:
                    with torch.no_grad():mapper.codes.weight.copy_(torch.tensor(fixed[arm],dtype=torch.float32))
                for v in mapper.parameters():v.requires_grad_(trainable)
                params=list(head.parameters())+(list(mapper.parameters()) if trainable else [])
                assert sum(v.numel() for v in head.parameters())==705
                assert sum(v.numel() for v in mapper.parameters())==216
                opt=torch.optim.Adam(params,lr=c["lr"]);checkpoints={}
                for step in range(c["steps"]+1):
                    if time.monotonic()-start>c["seconds_cap"]:raise TimeoutError("Bounded36fit pilot")
                    if step in c["checkpoints"]:
                        head.eval();mapper.eval()
                        with torch.no_grad():
                            pc=anchor+output(head,mapper,common_cal,ct,trainable,frequency).numpy()
                            ps=va+output(head,mapper,common_sel,st,trainable,frequency).numpy()
                        checkpoints[step]=dict(head=copy.deepcopy(head.state_dict()),mapper=copy.deepcopy(mapper.state_dict()))
                        predictions[f"{s}_{label}_{arm}_{step}_selection"]=ps
                        trainmet=masked_scores(pc,known,mask)
                        curves.append(dict(seed=s,budget=label,arm=arm,step=step,split="masked_cal",**trainmet))
                        curves.append(dict(seed=s,budget=label,arm=arm,step=step,split="selection",**scores(ps,vy)))
                        if step==0:
                            initialerr=max(initialerr,float(abs(pc-anchor).max()),float(abs(ps-va).max()));assert initialerr==0
                        if label=="labels360" and arm=="QwenTaskGeneric":
                            full_replay=max(full_replay,float(abs(ps-old[f"{s}_{step}_selection"]).max()))
                        if step==c["deploy_step"]:
                            selected=ps.copy();table_selected=code_values(mapper,trainable,frequency).detach().numpy()
                    if step==c["steps"]:break
                    head.train();mapper.train();ix=batch[step];weights=mt[ix]
                    pred=ac[ix]+output(head,mapper,common_cal[ix],ct[ix],trainable,frequency);err=pred-yt[ix]
                    loss=.5*((weights*err.square()).sum()/weights.sum()/anchor_loss["mse"]+
                             (weights*err.abs()).sum()/weights.sum()/anchor_loss["mae"])
                    opt.zero_grad();loss.backward()
                    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in params)
                    opt.step()
                allstates[s][label][arm]=checkpoints
                record=dict(seed=s,budget=label,unique_labels=b["unique_target_hours"],arm=arm,
                    step=c["deploy_step"],trainable_parameters=921 if trainable else 705,total_table_reader_parameters=921,
                    **scores(selected,vy))
                rows.append(record);diags.append(dict(seed=s,budget=label,arm=arm,**diagnose(selected,va,vy)))
                featurediags.append(dict(seed=s,budget=label,arm=arm,**feature_diag(table_selected,ci)))
                for bi,ix in enumerate(np.array_split(np.arange(16),3)):
                    blocks.append(dict(seed=s,budget=label,arm=arm,block=bi,windows=len(ix),**scores(selected[ix],vy[ix])))
                for j in range(16):
                    windowrows.append(dict(seed=s,budget=label,arm=arm,origin=str(grid[p["val_origin"][sel[j]]]),
                        mse=float(np.mean((selected[j]-vy[j])**2)),mae=float(np.mean(abs(selected[j]-vy[j])))))
                chosen=checkpoints[c["deploy_step"]]
                h2=Head(s);h2.load_state_dict(chosen["head"]);m2=CalendarMap(s);m2.load_state_dict(chosen["mapper"]);h2.eval();m2.eval()
                with torch.no_grad():rp=va+output(h2,m2,common_sel,st,trainable,frequency).numpy()
                re=float(abs(rp-selected).max());maxreload=max(maxreload,re);assert re==0
                checks.append(dict(seed=s,budget=label,arm=arm,reload_error=re,updates=c["steps"]))
                print("FIT",s,label,arm,"fixed100",record["mse"],record["mae"],flush=True)
    summary={b["name"]:{arm:{k:float(np.mean([v[k] for v in rows if v["budget"]==b["name"] and v["arm"]==arm])) for k in ["mse","mae","level","shape"]}
                       for arm in c["arms"]} for b in c["budgets"]}
    anchors={k:float(np.mean([scores(old[f"{s}_selection_anchor"],old[f"{s}_selection_y"])[k] for s in c["seeds"]])) for k in ["mse","mae","level","shape"]}
    save(r/"summary.json",dict(anchor=anchors,budgets=summary))
    save(r/"metrics.json",rows);save(r/"curves.json",curves);save(r/"block_metrics.json",blocks);save(r/"correction_diagnostics.json",diags)
    save(r/"feature_diagnostics.json",featurediags);save(r/"reload_checks.json",checks)
    csvsave(r/"metrics.csv",rows);csvsave(r/"all_window_effects.csv",windowrows)
    torch.save(allstates,r/"all_checkpoints.pt");np.savez_compressed(r/"predictions.npz",**predictions)
    for a in assets:assert sha(a["path"])==a["sha"],a["path"]
    check22(c);assert full_replay<3e-6 and label_gradient_error==0
    save(r/"verification.json",dict(protected_SHA_checks=len(assets),fixed22_unchanged=True,versions51_53_54_unchanged=True,
        full_budget_Qwen53_max_replay_error=full_replay,unlabeled_target_gradient=label_gradient_error,
        all36_reloads_max_error=maxreload,initial_anchor_error=initialerr,nested_hour_daytype_balanced_budgets=True,
        no_outer_scoring=True,no_new_selection=True))
    save(r/"status.json",dict(complete=True,seconds=time.monotonic()-start,fits=len(rows),updates=len(rows)*c["steps"],
        old22_control_refits=0,LLM_calls=0,GPU_loaded=False,student_KD_runs=0,deploy_step=100,
        independent_confirmation=False,end_to_end_data_efficiency=False))
    print("SUMMARY",read(r/"summary.json"),flush=True);print("STATUS",read(r/"status.json"),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args();c=read(a.config)
    try:run(c,a.config)
    except Exception:
        r=Path(c["run"])
        if r.exists():(r/"failure.log").write_text(traceback.format_exc())
        raise
