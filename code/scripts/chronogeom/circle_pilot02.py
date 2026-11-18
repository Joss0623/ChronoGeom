"""Pilot02: fixed-checkpoint collection, native audit and matched interventions."""
import argparse,csv,json,time,hashlib,subprocess,traceback
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import numpy as np
import torch
from scipy.spatial import cKDTree
from models.Transformer import Model
from data_provider.chronogeom_circle_pilot02 import make_pack,test_pack,args_for,nuisance
from scripts.chronogeom.circle_pilot02_math import (native_audit,calibrate,apply_residual,decode_metrics,phase_decoder,centroids)
from scripts.chronogeom.tslib_circle_transfer import forward_with_patch
H=24

def js(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def table(p,rows):
    with p.open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def phase_y(t):
    return np.stack([np.cos(2*np.pi*t/24),np.sin(2*np.pi*t/24)],axis=-1)

@contextmanager
def input_mode(model,mode):
    handles=[];modules=dict(model.named_modules())
    names=[]
    if mode in ["calendar_zero","both_zero"]:names+=["enc_embedding.temporal_embedding","dec_embedding.temporal_embedding"]
    if mode in ["position_zero","both_zero"]:names+=["enc_embedding.position_embedding","dec_embedding.position_embedding"]
    for name in names:handles.append(modules[name].register_forward_hook(lambda m,a,o:torch.zeros_like(o)))
    try:yield
    finally:
        for handle in handles:handle.remove()

def capture_all(model,pack,sites,mode="baseline"):
    cache={};mods=dict(model.named_modules());handles=[]
    for site in sites:
        def hook(m,a,o,key=site):cache.setdefault(key,[]).append(o.detach().cpu().numpy().copy())
        handles.append(mods[site].register_forward_hook(hook))
    try:
        with input_mode(model,mode),torch.inference_mode():pred=model(*args_for(pack,"cpu")).numpy()
        assert all(len(cache[s])==1 for s in sites)
        return pred,{s:cache[s][0] for s in sites}
    finally:
        for h in handles:h.remove()

def get_model(root,seed,stage):
    cfg=json.loads((Path(root)/"config.json").read_text());m=Model(SimpleNamespace(**cfg)).eval()
    f=Path(root)/"states"/f"{'init' if stage=='init' else 'best'}_{seed}.pt"
    state=torch.load(f,map_location="cpu",weights_only=True);m.load_state_dict(state if stage=="init" else state["state"])
    return m

def packs(cfg):
    return {"fit":make_pack("fit",cfg["n_fit"]),"val":make_pack("val",cfg["n_val"]),"test":test_pack(cfg["n_test_a"])}

def collect(cfg,root):
    out=root/"cache";out.mkdir();(root/"data").mkdir();start=time.monotonic();data=packs(cfg)
    for k,v in data.items():np.savez_compressed(root/"data"/f"{k}.npz",**v)
    for shift in cfg["shifts"]:
        np.savez_compressed(root/"data"/f"test_{shift}.npz",**test_pack(cfg["n_test_a"],shift))
    provenance=dict(base_commit=cfg["base_commit"],source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        torch=torch.__version__,numpy=np.__version__,device="cpu",new_forecast_training=0,
        checkpoint_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(cfg["old_run"])/"states").glob("*_*.pt") if p.name.startswith(("best","init"))})
    js(root/"provenance.json",provenance);forecast=[]
    for seed in cfg["seeds"]:
        for stage in ["init","trained"]:
            model=get_model(cfg["old_run"],seed,stage)
            modes=["baseline"] if stage=="init" else ["baseline","calendar_zero","position_zero","both_zero"]
            for mode in modes:
                for split,pack in data.items():
                    if time.monotonic()-start>cfg["stage_cap_s"]:raise TimeoutError("Collection cap")
                    pred,acts=capture_all(model,pack,cfg["sites"],mode)
                    np.savez_compressed(out/f"{seed}_{stage}_{mode}_{split}.npz",prediction=pred,
                        **{f"site{i}":acts[s][:,-H:] for i,s in enumerate(cfg["sites"])})
                    if split=="test":
                        forecast.append(dict(seed=seed,stage=stage,mode=mode,mse=float(np.mean((pred-pack["y"])**2)),
                            clean_mse=float(np.mean((pred-pack["mean"])**2))))
            # Conflict inputs: correct history values, all clock fields shifted +6 hours.
            if stage=="trained":
                shifted=test_pack(cfg["n_test_a"],6);conflict=dict(data["test"])
                conflict["xm"]=shifted["xm"];conflict["dm"]=shifted["dm"]
                pred,_=capture_all(model,conflict,cfg["sites"])
                np.savez_compressed(out/f"{seed}_calendar_conflict.npz",prediction=pred)
                forecast.append(dict(seed=seed,stage=stage,mode="calendar_conflict",mse=float(np.mean((pred-data["test"]["y"])**2)),
                    clean_mse=float(np.mean((pred-data["test"]["mean"])**2))))
            print("collected",seed,stage,round(time.monotonic()-start,2),flush=True)
    table(root/"input_forecast.csv",forecast)
    js(root/"collect_completion.json",dict(wall_s=time.monotonic()-start,complete=True))

def geom(cfg,root):
    out=root/"native";out.mkdir();start=time.monotonic();data=packs(cfg);rows=[];transports=[]
    h=data["fit"]["t"].ravel()%24;th=data["test"]["t"].ravel()%24
    nf=nuisance(data["fit"]);nt=nuisance(data["test"])
    for seed in cfg["seeds"]:
        for stage in ["init","trained"]:
            modes=["baseline"] if stage=="init" else ["baseline","calendar_zero","position_zero","both_zero"]
            for mode in modes:
                fit=np.load(root/"cache"/f"{seed}_{stage}_{mode}_fit.npz")
                test=np.load(root/"cache"/f"{seed}_{stage}_{mode}_test.npz")
                for i,site in enumerate(cfg["sites"]):
                    if time.monotonic()-start>cfg["stage_cap_s"]:raise TimeoutError("Native stage cap")
                    x=fit[f"site{i}"].reshape(-1,16).astype(float);tx=test[f"site{i}"].reshape(-1,16).astype(float)
                    metric,arrays=native_audit(x,h,tx,th,cfg["n_fit"],
                        cfg["permutations"] if mode=="baseline" else 0,
                        cfg["pca_bootstrap"] if mode=="baseline" else 0,1729+seed)
                    rows.append(dict(seed=seed,stage=stage,mode=mode,site=site,n_test_trajectories=len(data["test"]["x"]),**metric))
                    np.savez_compressed(out/f"{seed}_{stage}_{mode}_{i}.npz",**arrays)
                    # Analysis-only phase readability on the same fit/test interface.
                    cf,_,q,dec,diag,_=calibrate_simple(x,h,nf,tx,th,nt)
                    rows[-1]["supervised_phase_r2"]=diag
                    if stage=="trained" and mode=="baseline" and i>=1:
                        xf=fit["site1"].reshape(-1,16);xt=test["site1"].reshape(-1,16)
                        z=np.column_stack([xf,np.ones(len(xf))]);b=np.linalg.lstsq(z,x,rcond=1e-10)[0]
                        p=np.column_stack([xt,np.ones(len(xt))])@b
                        r2=1-np.sum((p-tx)**2)/np.sum((tx-tx.mean(0))**2)
                        transports.append(dict(seed=seed,target_site=site,input_site="dec_embedding",test_linear_transport_r2=float(r2)))
                print("native",seed,stage,mode,round(time.monotonic()-start,2),flush=True)
            table(root/"native_geometry.csv",rows)
    table(root/"linear_transport.csv",transports)
    js(root/"native_completion.json",dict(wall_s=time.monotonic()-start,records=len(rows),complete=True))

def calibrate_simple(x,h,nuis,tx,th,tnuis):
    from scripts.chronogeom.circle_pilot02_math import residual_fit
    from scripts.chronogeom.tslib_circle_transfer import fit_circle
    r,b=residual_fit(x,nuis);tr=apply_residual(tx,tnuis,b)
    fit=fit_circle(x,h,24,nuisance=nuis);y=phase_y(h);ty=phase_y(th)
    coef=phase_decoder(r,y,fit.circle_basis);r2,_,_=decode_metrics(tr,ty,fit.circle_basis,coef)
    return fit,b,fit.circle_basis,coef,r2,None

def metrics(pred,cp,bp,clean,bad):
    z=np.column_stack([np.ones(H),np.arange(1,H+1),np.sin(2*np.pi*np.arange(1,H+1)/24),np.cos(2*np.pi*np.arange(1,H+1)/24)])
    pinv=np.linalg.pinv(z);co=pred[...,0]@pinv.T;cc=cp[...,0]@pinv.T;cb=bp[...,0]@pinv.T
    angle=lambda a:np.arctan2(a[:,3],a[:,2]);wrap=lambda a:np.angle(np.exp(1j*a))
    amp=np.hypot(co[:,2],co[:,3]);res=pred[...,0]-co@z.T
    return dict(clean_pred_mse=np.mean((pred-cp)**2,axis=(1,2)),base_clean_pred_mse=np.mean((bp-cp)**2,axis=(1,2)),
        clean_truth_mse=np.mean((pred-clean["mean"])**2,axis=(1,2)),corrupt_truth_mse=np.mean((pred-bad["mean"])**2,axis=(1,2)),
        phase_error=np.abs(wrap(angle(co)-angle(cc))),phase_error_true=np.abs(wrap(angle(co)-2*np.pi*clean["t0"]/24)),
        phase_change=wrap(angle(co)-angle(cb)),amplitude=amp,amplitude_error=np.abs(amp-1),
        offset_error=np.abs(co[:,0]-clean["c"]),trend_error=np.abs(co[:,1]-clean["b"]),
        offset_vs_clean=np.abs(co[:,0]-cc[:,0]),trend_vs_clean=np.abs(co[:,1]-cc[:,1]),
        nonharmonic_mse=np.mean(res**2,axis=1),phase_defined=amp>=1e-4)

def analyze(cfg,root):
    out=root/"interventions";out.mkdir();(out/"bases").mkdir();(out/"predictions").mkdir()
    data=packs(cfg);clean=data["test"];n=len(clean["x"]);start=time.monotonic()
    h=data["fit"]["t"].ravel()%24;vh=data["val"]["t"].ravel()%24;th=clean["t"].ravel()%24
    nf=nuisance(data["fit"]);nv=nuisance(data["val"]);nt=nuisance(clean)
    rows=[];calrows=[];testscores=[];control=[];perhour=[];complements=[]
    for seed in cfg["seeds"]:
        model=get_model(cfg["old_run"],seed,"trained")
        fcache=np.load(root/"cache"/f"{seed}_trained_baseline_fit.npz");vcache=np.load(root/"cache"/f"{seed}_trained_baseline_val.npz")
        tcache=np.load(root/"cache"/f"{seed}_trained_baseline_test.npz");cp=tcache["prediction"]
        badcache={};altpred={};altcache={}
        for shift in cfg["shifts"]:
            bad=test_pack(cfg["n_test_a"],shift)
            bp,acts=capture_all(model,bad,cfg["sites"]);badcache[shift]=(bad,bp,acts)
        alt=test_pack(cfg["n_test_a"],0,True);ap,aa=capture_all(model,alt,cfg["sites"])
        for si,site in enumerate(cfg["sites"]):
            if time.monotonic()-start>cfg["stage_cap_s"]:raise TimeoutError("Intervention cap")
            x=fcache[f"site{si}"].reshape(-1,16).astype(float);vx=vcache[f"site{si}"].reshape(-1,16).astype(float)
            tx=tcache[f"site{si}"].reshape(-1,16).astype(float)
            fit,b,bases,dec,diag,pool=calibrate(x,h,nf,vx,vh,nv,cfg["n_fit"],cfg,seed)
            np.savez_compressed(out/"bases"/f"{seed}_{si}.npz",nuisance_coef=b,circle_phase_coef=fit.full_coef[-2:],mean=fit.mean,
                **bases,**{f"decode_{k}":v for k,v in dec.items()})
            np.savez_compressed(out/"bases"/f"candidate_pool_{seed}_{si}.npz",**pool)
            rtest=apply_residual(tx,nt,b);rfit=apply_residual(x,nf,b);cov=np.cov(rfit,rowvar=False)
            comp=np.eye(16)-fit.circle_basis@fit.circle_basis.T
            compcoef=np.linalg.lstsq(np.column_stack([rfit@comp,np.ones(len(rfit))]),phase_y(h),rcond=1e-8)[0]
            comppred=np.column_stack([rtest@comp,np.ones(len(rtest))])@compcoef
            compr2=1-np.mean(np.sum((comppred-phase_y(th))**2,axis=1))
            complements.append(dict(seed=seed,site=site,complement_rank=14,test_phase_r2=float(compr2)))
            for d in diag:calrows.append(dict(seed=seed,site=site,**d))
            target_r2=None;captures={}
            for name,q in bases.items():
                r2,err,pred=decode_metrics(rtest,phase_y(th),q,dec[name])
                cap=float(np.sum((rtest@q)**2)/max(np.sum(rtest**2),1e-12));captures[name]=cap
                testscores.append(dict(seed=seed,site=site,space=name,test_phase_r2=r2,test_capture=cap))
                for hour in range(24):perhour.append(dict(seed=seed,site=site,space=name,hour=hour,phase_mse=float(err[th==hour].mean())))
            legacy=np.load(Path(cfg["old_run"])/"analysis/bases"/f"trained_{seed}_{si}.npz")
            # Native support diagnostics: 512 fixed raw fit states, not fabricated means.
            ids=np.linspace(0,len(x)-1,512).astype(int);tree=cKDTree(x[ids])
            hc=centroids(x,h);within=x-hc[h];cw=within.T@within/len(within)
            between=np.var(hc,axis=0).sum();ridge=max(between/16*.01,np.trace(cw)/16*.01,1e-6)
            ci=np.linalg.pinv(cw+np.eye(16)*ridge);scale=max(np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=1))),1e-6)
            for shift,(bad,bp,acts) in badcache.items():
                full=acts[site];base=full[:,-H:].astype(float);donor=tcache[f"site{si}"].astype(float)
                changes={"noop":base.copy(),"full_donor":donor.copy()}
                for name,q in bases.items():changes[name]=base+((donor-base)@q)@q.T
                reference=np.linalg.norm(changes["circle"]-base,axis=(1,2));matched={}
                for name in list(bases):
                    if name=="circle":continue
                    d=changes[name]-base;norm=np.linalg.norm(d,axis=(1,2))
                    factor=np.divide(reference,norm,out=np.ones_like(norm),where=norm>1e-10)
                    changes[name+"_norm"]=base+d*factor[:,None,None];matched[name+"_norm"]=(factor,reference)
                changes["legacy_circle"]=base+((donor-base)@legacy["circle_basis"])@legacy["circle_basis"].T
                changes["legacy_pca"]=base+((donor-base)@legacy["pca_basis"])@legacy["pca_basis"].T
                d=changes["random0"]-base;dn=np.linalg.norm(d,axis=(1,2));ref=np.linalg.norm(changes["legacy_circle"]-base,axis=(1,2))
                fac=np.divide(ref,dn,out=np.ones_like(dn),where=dn>1e-10);changes["legacy_random0_norm"]=base+d*fac[:,None,None];matched["legacy_random0_norm"]=(fac,ref)
                natural=aa[site][:,-H:].astype(float)
                changes["conditional_full"]=natural
                for name in ["circle","info_match"]:changes["conditional_"+name]=base+((natural-base)@bases[name])@bases[name].T
                hd=(phase_y(clean["t"]) - phase_y(bad["t"]))@fit.full_coef[-2:]
                changes["harmonic_mean_delta"]=base+hd
                norm=np.linalg.norm(hd,axis=(1,2));fac=np.divide(reference,norm,out=np.ones_like(norm),where=norm>1e-10)
                changes["harmonic_mean_delta_norm"]=base+hd*fac[:,None,None];matched["harmonic_mean_delta_norm"]=(fac,reference)
                for name in ["circle","residual_pca","info_match"]:
                    q=bases[name];changes[name+"_ablate"]=base+((fit.mean-base)@q)@q.T
                q=bases["random0"];d=((fit.mean-base)@q)@q.T;norm=np.linalg.norm(d,axis=(1,2));ref=np.linalg.norm(changes["circle_ablate"]-base,axis=(1,2))
                fac=np.divide(ref,norm,out=np.ones_like(norm),where=norm>1e-10);changes["random0_ablate_norm"]=base+d*fac[:,None,None];matched["random0_ablate_norm"]=(fac,ref)
                changes["full_ablate"]=np.broadcast_to(fit.mean,base.shape).copy()
                for condition,changed in changes.items():
                    if time.monotonic()-start>cfg["stage_cap_s"]:raise TimeoutError("Intervention cap")
                    changed=changed.astype(np.float32);d=full.copy();d[:,-H:]=changed
                    pred=forward_with_patch(model,site,args_for(bad,"cpu"),torch.from_numpy(d),mode="full").numpy()
                    np.savez_compressed(out/"predictions"/f"{seed}_{si}_{shift}_{condition}.npz",prediction=pred)
                    m=metrics(pred,cp,bp,clean,bad);delta=changed-base
                    dn=np.linalg.norm(delta,axis=(1,2));flat=changed.reshape(-1,16).astype(float)
                    nn=tree.query(flat,k=1,workers=2)[0].reshape(n,H)/scale
                    targetdiff=flat-hc[th];mahal=np.einsum("nd,de,ne->n",targetdiff,ci,targetdiff).reshape(n,H).mean(1)
                    targetdist=np.linalg.norm(targetdiff,axis=1).reshape(n,H).mean(1)/scale
                    # Exact finite-lookup support at hour layer; centroid distance elsewhere is diagnostic only.
                    nd=np.sqrt(np.maximum(np.sum(flat*flat,1)[:,None]+np.sum(hc*hc,1)[None,:]-2*flat@hc.T,0)).min(1).reshape(n,H)/scale
                    outside=(nd>1e-6).mean(1)
                    matcherr=np.abs(dn-matched[condition][1])/np.maximum(matched[condition][1],1e-10) if condition in matched else np.full(n,np.nan)
                    err0=float(np.max(abs(pred-bp))) if condition=="noop" else None
                    errfull=float(np.max(abs(pred-cp))) if condition=="full_donor" and si==3 else None
                    control.append(dict(seed=seed,site=site,shift=shift,condition=condition,noop_max=err0,full_max=errfull))
                    for i in range(n):
                        rows.append(dict(seed=seed,site=site,shift=shift,condition=condition,trajectory=int(clean["ids"][i]),
                            cohort=int(clean["cohort"][i]),**{k:bool(v[i]) if k=="phase_defined" else float(v[i]) for k,v in m.items()},
                            delta_norm=float(dn[i]),relative_delta=float(dn[i]/max(np.linalg.norm(base[i]),1e-10)),
                            nn_state_distance=float(nn[i].mean()),target_centroid_distance=float(targetdist[i]),
                            mahal=float(mahal[i]),outside_centroid_support=float(outside[i]),
                            norm_match_error=float(matcherr[i]),scale=float(matched[condition][0][i]) if condition in matched else 1.))
                print("intervened",seed,site,shift,len(changes),round(time.monotonic()-start,2),flush=True)
            table(root/"phase_complement.csv",complements);table(root/"calibration.csv",calrows);table(root/"test_space_scores.csv",testscores);table(root/"phase_error_by_hour.csv",perhour)
            table(root/"interventions.csv",rows)
    js(root/"control_checks.json",control)
    assert max(v["noop_max"] for v in control if v["noop_max"] is not None)<=1e-5
    assert max(v["full_max"] for v in control if v["full_max"] is not None)<=1e-5
    js(root/"analysis_completion.json",dict(wall_s=time.monotonic()-start,records=len(rows),conditions=len(changes),complete=True))

def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["collect","geometry","analyze"]);p.add_argument("--config",required=True);p.add_argument("--run",required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());root=Path(a.run)
    torch.set_num_threads(cfg["threads"]);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    try:{"collect":collect,"geometry":geom,"analyze":analyze}[a.stage](cfg,root)
    except Exception:
        (root/(a.stage+"_failure.txt")).write_text(traceback.format_exc());raise
if __name__=="__main__":main()
