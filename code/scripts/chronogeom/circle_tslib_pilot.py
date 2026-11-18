"""Bounded training and preregistered MIT Circle-to-TSLib scientific pilot."""
import argparse,copy,csv,hashlib,json,os,platform,subprocess,time,traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from models.Transformer import Model
from data_provider.chronogeom_circle_pilot import make_pack,args_for,nuisance,H
from scripts.chronogeom.tslib_circle_transfer import fit_circle,evaluate_circle,capture_site,forward_with_patch

def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def csv_rows(path,rows):
    with path.open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def numpy_pred(model,pack,device):
    with torch.inference_mode():return model(*args_for(pack,device)).cpu().numpy()
def batch_forward(model,pack,device,batch=96):
    return numpy_pred(model,pack,device)
def setup():
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
def config_model(cfg):return SimpleNamespace(**cfg)

def train(cfg,root,device):
    root.mkdir(exist_ok=True);(root/"states").mkdir();(root/"data").mkdir()
    packs={s:make_pack(s,cfg["n_"+s]) for s in ["train","val","fit","test"]}
    for s,p in packs.items():np.savez_compressed(root/"data"/(s+".npz"),**p)
    for shift in cfg["shifts"]:np.savez_compressed(root/"data"/f"test_shift_{shift}.npz",**make_pack("test",cfg["n_test"],shift))
    save_json(root/"config.json",cfg)
    repo=Path.cwd()
    env=dict(source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=device,
        cuda=torch.version.cuda,cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),gpu=torch.cuda.get_device_name(0) if device=="cuda" else None,
        files={str(p):sha(p) for p in [repo/"models/Transformer.py",repo/"layers/Embed.py",
            repo/"data_provider/chronogeom_circle_pilot.py",Path(__file__)]},
        existing_checkpoint_audit={"standard_checkpoints":0,"phase1_states":540,"phase1b_states":384,
            "reason_not_used":"fixed engineered time features or incompatible readouts, not a learned temporal Transformer"},
        split_ids={s:{"min":int(p["ids"].min()),"max":int(p["ids"].max()),"n":len(p["ids"])} for s,p in packs.items()})
    save_json(root/"provenance.json",env)
    tr=packs["train"];val=packs["val"];xt=args_for(tr,device);yt=torch.as_tensor(tr["y"],device=device)
    start=time.monotonic();curves=[];metrics=[]
    for seed in cfg["seeds"]:
        torch.manual_seed(seed);model=Model(config_model(cfg)).to(device)
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.save(initial,root/"states"/f"init_{seed}.pt")
        opt=torch.optim.Adam(model.parameters(),lr=cfg["lr"])
        rng=np.random.default_rng(seed);best=np.inf;best_state=None;best_step=None
        for step in range(1,cfg["steps"]+1):
            if time.monotonic()-start>cfg["train_cap_s"]:
                torch.save({"state":model.state_dict(),"step":step,"optimizer":opt.state_dict()},root/"states"/f"timeout_{seed}.pt")
                raise TimeoutError("Fixed train budget exceeded; partial checkpoint retained")
            model.train();ix=rng.integers(len(tr["x"]),size=cfg["batch"]);opt.zero_grad()
            pred=model(*(x[ix] for x in xt));loss=((pred-yt[ix])**2).mean()
            if not torch.isfinite(loss):raise ValueError("Nonfinite training loss")
            loss.backward();opt.step()
            if step%100==0:
                model.eval();vp=numpy_pred(model,val,device);vm=float(np.mean((vp-val["y"])**2))
                row=dict(seed=seed,step=step,minibatch_loss=float(loss.detach()),validation_mse=vm,elapsed_s=time.monotonic()-start)
                curves.append(row);print(json.dumps(row),flush=True)
                if vm<best:
                    best=vm;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.save({"state":best_state,"selected_step":best_step,"validation_mse":best},root/"states"/f"best_{seed}.pt")
        torch.save({"state":{k:v.detach().cpu() for k,v in model.state_dict().items()},"optimizer":opt.state_dict(),"steps":cfg["steps"]},root/"states"/f"last_{seed}.pt")
        model.load_state_dict(best_state);model.eval()
        for s in ["train","val","test"]:
            p=numpy_pred(model,packs[s],device)
            np.savez_compressed(root/f"base_{seed}_{s}.npz",prediction=p,target=packs[s]["y"],mean=packs[s]["mean"])
            metrics.append(dict(seed=seed,split=s,mse=float(np.mean((p-packs[s]["y"])**2)),
                clean_mse=float(np.mean((p-packs[s]["mean"])**2)),parameters=sum(p.numel() for p in model.parameters()),selected_step=best_step))
        csv_rows(root/"learning_curves.csv",curves);csv_rows(root/"forecast_metrics.csv",metrics)
    save_json(root/"train_completion.json",dict(wall_s=time.monotonic()-start,seeds=cfg["seeds"],fit_count=len(cfg["seeds"]),completed=True))

def eval_geometry(fit,acts,phase,nuis,mask):
    return evaluate_circle(fit,acts[mask],phase[mask],nuis[mask])

def diagnostics(x,changed,fit,covinv):
    delta=changed-x
    relative=np.linalg.norm(delta,axis=(1,2))/np.maximum(np.linalg.norm(x,axis=(1,2)),1e-12)
    mahal=np.einsum("nhd,de,nhe->nh",changed-fit.mean,covinv,changed-fit.mean).mean(1)
    return dict(delta_norm=np.linalg.norm(delta,axis=(1,2)),activation_norm=np.linalg.norm(changed,axis=(1,2)),
        relative_delta=relative,mahal=mahal)

def output_metrics(pred,clean_pred,corrupt_pred,clean,corrupt):
    pp=pred[...,0];cp=clean_pred[...,0];bp=corrupt_pred[...,0]
    j=np.arange(1,H+1);z=np.column_stack([np.ones(H),j,np.sin(2*np.pi*j/24),np.cos(2*np.pi*j/24)])
    co=pp@np.linalg.pinv(z).T;co_clean=cp@np.linalg.pinv(z).T
    co_corrupt=bp@np.linalg.pinv(z).T
    angle=lambda a:np.arctan2(a[:,3],a[:,2])
    wrap=lambda a:np.arctan2(np.sin(a),np.cos(a))
    amp=np.hypot(co[:,2],co[:,3])
    return dict(clean_truth_mse=np.mean((pred-clean["mean"])**2,axis=(1,2)),
        corrupt_truth_mse=np.mean((pred-corrupt["mean"])**2,axis=(1,2)),
        clean_pred_mse=np.mean((pred-clean_pred)**2,axis=(1,2)),
        base_clean_pred_mse=np.mean((corrupt_pred-clean_pred)**2,axis=(1,2)),
        phase_error=np.abs(wrap(angle(co)-angle(co_clean))),
        phase_change=wrap(angle(co)-angle(co_corrupt)),amplitude=amp,
        trend=co[:,1],offset=co[:,0],phase_defined=amp>=1e-4)

def analyze(cfg,root,device):
    out=root/"analysis";out.mkdir(exist_ok=False);(out/"bases").mkdir();(out/"predictions").mkdir()
    start=time.monotonic();georows=[];records=[];controls=[];siteres=[]
    fitpack=make_pack("fit",cfg["n_fit"]);clean=make_pack("test",cfg["n_test"])
    phasefit=(fitpack["t"]%24).ravel();phaseeval=(clean["t"]%24).ravel()
    fitmask=~np.isin(phasefit,cfg["geometry_excluded_hours"]);heldout=np.isin(phaseeval,cfg["geometry_excluded_hours"])
    nufit=nuisance(fitpack);nueval=nuisance(clean)
    permutation=np.random.default_rng(20261024).permutation(24)
    for seed in cfg["seeds"]:
        model=Model(config_model(cfg)).to(device)
        state=torch.load(root/"states"/f"best_{seed}.pt",map_location="cpu",weights_only=True)
        for stage in ["init","trained"]:
            model.load_state_dict(torch.load(root/"states"/f"init_{seed}.pt",map_location="cpu",weights_only=True) if stage=="init" else state["state"]);model.eval()
            for si,site in enumerate(cfg["sites"]):
                if time.monotonic()-start>cfg["analysis_cap_s"]:raise TimeoutError("Analysis hard cap")
                clean_pred,full_eval=capture_site(model,site,args_for(clean,device))
                _,full_fit=capture_site(model,site,args_for(fitpack,device))
                af=full_fit[:,-H:].cpu().numpy().reshape(-1,cfg["d_model"])
                ae=full_eval[:,-H:].cpu().numpy().reshape(-1,cfg["d_model"])
                fit=fit_circle(af[fitmask],phasefit[fitmask],24,nuisance=nufit[fitmask])
                sf=fit_circle(af[fitmask],permutation[phasefit[fitmask]],24,nuisance=nufit[fitmask])
                np.savez_compressed(out/"bases"/f"{stage}_{seed}_{si}.npz",
                    **{k:v for k,v in asdict(fit).items() if isinstance(v,np.ndarray)},shuffled_basis=sf.circle_basis)
                for cohort,mask in [("all",np.ones(len(ae),dtype=bool)),("geometry_heldout_hours",heldout)]:
                    g=eval_geometry(fit,ae,phaseeval,nueval,mask)
                    sg=eval_geometry(sf,ae,permutation[phaseeval],nueval,mask)
                    train_b=np.column_stack([np.ones(len(nufit)),nufit])@fit.base_coef
                    eval_b=np.column_stack([np.ones(len(nueval)),nueval])@fit.base_coef
                    proj=(af-train_b)@fit.circle_basis
                    target=np.column_stack([np.cos(2*np.pi*phasefit/24),np.sin(2*np.pi*phasefit/24)])
                    pb=np.linalg.lstsq(np.column_stack([proj[fitmask],np.ones(sum(fitmask))]),target[fitmask],rcond=1e-10)[0]
                    yy=np.column_stack([(ae-eval_b)@fit.circle_basis,np.ones(len(ae))])@pb
                    tt=np.column_stack([np.cos(2*np.pi*phaseeval/24),np.sin(2*np.pi*phaseeval/24)])
                    pr2=1-np.sum((yy[mask]-tt[mask])**2)/np.sum((tt[mask]-tt[mask].mean(0))**2)
                    georows.append(dict(seed=seed,stage=stage,site=site,cohort=cohort,n_trajectories=len(clean["x"]),n_activation_points=int(mask.sum()),
                        **g,shuffled_incremental_r2=sg["incremental_r2"],phase_decode_r2=float(pr2),
                        pca_overlap=float(np.sum((fit.circle_basis.T@fit.pca_basis)**2)/max(fit.circle_basis.shape[1],1))))
                plotdata=out/f"geometry_{stage}_{seed}_{si}.npz"
                np.savez_compressed(plotdata,projection=ae@fit.circle_basis,hour=phaseeval,ids=np.repeat(clean["ids"],H))
                if stage=="init":continue
                rank=fit.circle_basis.shape[1]
                rngs=[np.random.default_rng(r) for r in cfg["random_seeds"]]
                bases={"circle":fit.circle_basis,"pca":fit.pca_basis,
                    **{f"random{k}":np.linalg.qr(r.normal(size=(cfg["d_model"],rank)))[0] for k,r in enumerate(rngs)},
                    "shuffled":sf.circle_basis}
                covariance=np.cov(af[fitmask]-np.column_stack([np.ones(sum(fitmask)),nufit[fitmask]])@fit.base_coef,rowvar=False)
                covinv=np.linalg.pinv(covariance+np.eye(cfg["d_model"])*max(np.trace(covariance)/cfg["d_model"]*.01,1e-6))
                for shift in cfg["shifts"]:
                    corrupt=make_pack("test",cfg["n_test"],shift)
                    bpred,bacts=capture_site(model,site,args_for(corrupt,device))
                    base=bacts[:,-H:].cpu().numpy();donor=full_eval[:,-H:].cpu().numpy()
                    cleanp=clean_pred.cpu().numpy();basep=bpred.cpu().numpy()
                    changes={"noop":base.copy(),"full_donor":donor.copy()}
                    for name,q in bases.items():changes[name]=base+((donor-base)@q)@q.T
                    reference=np.linalg.norm(changes["circle"]-base,axis=(1,2))
                    scales={}
                    for name in ["pca","random0","shuffled"]:
                        d=changes[name]-base;norm=np.linalg.norm(d,axis=(1,2));scale=np.divide(reference,norm,out=np.ones_like(norm),where=norm>1e-10)
                        scales[name+"_norm"]=scale
                        changes[name+"_norm"]=base+d*scale[:,None,None]
                    for name,q in [(k,bases[k]) for k in ["circle","pca","random0"]]:
                        changes[name+"_ablate"]=base+((fit.mean-base)@q)@q.T
                    d=changes["random0_ablate"]-base;norm=np.linalg.norm(d,axis=(1,2))
                    ref=np.linalg.norm(changes["circle_ablate"]-base,axis=(1,2));scale=np.divide(ref,norm,out=np.ones_like(norm),where=norm>1e-10)
                    scales["random0_ablate_norm"]=scale;changes["random0_ablate_norm"]=base+d*scale[:,None,None]
                    changes["full_ablate"]=np.broadcast_to(fit.mean,base.shape).copy()
                    for name,changed in changes.items():
                        changed=changed.astype(base.dtype)
                        if time.monotonic()-start>cfg["analysis_cap_s"]:raise TimeoutError("Analysis hard cap")
                        d=bacts.clone();d[:,-H:]=torch.as_tensor(changed,device=device,dtype=d.dtype)
                        pred=forward_with_patch(model,site,args_for(corrupt,device),d,mode="full").cpu().numpy()
                        metr=output_metrics(pred,cleanp,basep,clean,corrupt);diag=diagnostics(base,changed,fit,covinv)
                        normmatch=np.abs(diag["delta_norm"]-(ref if name=="random0_ablate_norm" else reference))/np.maximum(ref if name=="random0_ablate_norm" else reference,1e-10)
                        controls.append(dict(seed=seed,site=site,shift=shift,condition=name,max_noop_error=float(np.max(np.abs(pred-basep))) if name=="noop" else None,
                            max_full_clean_error=float(np.max(np.abs(pred-cleanp))) if name=="full_donor" and site=="decoder.norm" else None))
                        for i in range(len(pred)):
                            records.append(dict(seed=seed,site=site,shift=shift,condition=name,trajectory=int(clean["ids"][i]),rank=(cfg["d_model"] if name in ["full_donor","full_ablate"] else rank),
                                **{k:bool(v[i]) if k=="phase_defined" else float(v[i]) for k,v in metr.items()},
                                **{k:float(v[i]) for k,v in diag.items()},
                                scale=float(scales[name][i]) if name in scales else 1.,
                                norm_match_relative_error=float(normmatch[i]) if name in scales else None))
                        np.savez_compressed(out/"predictions"/f"{seed}_{si}_{shift}_{name}.npz",
                            prediction=pred,clean_prediction=cleanp,corrupt_prediction=basep,clean_truth=clean["mean"],corrupt_truth=corrupt["mean"],ids=clean["ids"])
                    print(json.dumps(dict(analysis_seed=seed,site=site,shift=shift,conditions=len(changes),elapsed_s=time.monotonic()-start)),flush=True)
        csv_rows(out/"geometry.csv",georows);csv_rows(out/"interventions.csv",records)
        save_json(out/"control_checks.json",controls)
    errors=[r["max_noop_error"] for r in controls if r["max_noop_error"] is not None]
    full=[r["max_full_clean_error"] for r in controls if r["max_full_clean_error"] is not None]
    save_json(out/"completion.json",dict(wall_s=time.monotonic()-start,intervention_rows=len(records),geometry_rows=len(georows),noop_max=max(errors),full_final_site_max=max(full),source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),complete=True))
    assert max(errors)<1e-5 and max(full)<1e-5,"Core control failure; all results retained"

def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["train","analyze"]);p.add_argument("--config",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda")
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());root=Path(a.output);setup()
    try:
        (train if a.stage=="train" else analyze)(cfg,root,a.device)
    except Exception:
        root.mkdir(parents=True,exist_ok=True)
        (root/(a.stage+"_failure.txt")).write_text(traceback.format_exc());raise
if __name__=="__main__":main()
