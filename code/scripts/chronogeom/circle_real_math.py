"""Pilot02 matching algorithm, now with purged temporal-block phase CV for real data."""
import numpy as np
from scripts.chronogeom.circle_pilot02_math import basis,residual_fit,apply_residual,stats,score_many,phase_decoder,decode_metrics
from scripts.chronogeom.tslib_circle_transfer import fit_circle
from data_provider.chronogeom_circle_real import purged_folds

def blocked_cv_basis(x,h,nuis,origins,grid):
    y=np.column_stack([np.cos(2*np.pi*h/24),np.sin(2*np.pi*h/24)])
    groups=np.arange(len(x)).reshape(len(origins),24);folds=purged_folds(origins);results=[]
    for lam in grid:
        losses=[]
        for train,test in folds:
            tr=groups[train].ravel();te=groups[test].ravel()
            r,b=residual_fit(x[tr],nuis[tr]);rt=apply_residual(x[te],nuis[te],b)
            c=r.T@r/len(r);g=r.T@y[tr]/len(r)
            w=np.linalg.solve(c+np.eye(x.shape[1])*max(np.trace(c)/x.shape[1],1e-9)*lam,g)
            q=basis(w.T);coef=phase_decoder(r,y[tr],q)
            _,err,_=decode_metrics(rt,y[te],q,coef);losses.append(float(err.mean()))
        results.append(dict(lam=lam,cv_mse=float(np.mean(losses))))
    selected=min(results,key=lambda v:v['cv_mse'])['lam']
    r,b=residual_fit(x,nuis);c=r.T@r/len(r);g=r.T@y/len(r)
    w=np.linalg.solve(c+np.eye(x.shape[1])*max(np.trace(c)/x.shape[1],1e-9)*selected,g)
    return basis(w.T),results

def calibrate_real(x,h,nuis,vx,vh,vn,ntraj,cfg,seed,origins):
    fit=fit_circle(x,h,24,nuisance=nuis);r,b=residual_fit(x,nuis);vr=apply_residual(vx,vn,b)
    y=np.column_stack([np.cos(2*np.pi*h/24),np.sin(2*np.pi*h/24)])
    vy=np.column_stack([np.cos(2*np.pi*vh/24),np.sin(2*np.pi*vh/24)])
    c,g,_=stats(r,y);cv,gv,syv=stats(vr,vy)
    qcv,cvs=blocked_cv_basis(x,h,nuis,origins,cfg["ridge_grid"])
    shuffle=np.random.default_rng(20281024).permutation(24)
    sf=fit_circle(x,shuffle[h],24,nuisance=nuis)
    named={"circle":fit.circle_basis,"raw_pca":fit.pca_basis,"residual_pca":basis(r),
        **{f"random{k}":np.linalg.qr(np.random.default_rng(1729+k).normal(size=(x.shape[1],2)))[0] for k in range(3)},
        "shuffled":sf.circle_basis,"cv_phase":qcv}
    if any(q.shape[1]!=2 for q in named.values()):raise ValueError("Baseline rank not 2; retain failure")
    rng=np.random.default_rng(1729+seed)
    pool=[np.linalg.qr(rng.normal(size=(cfg["candidate_random"],x.shape[1],2)))[0]]
    for anchor in [named["circle"],qcv]:
        normal=rng.normal(size=(cfg["candidate_complements"],x.shape[1],2))
        normal=normal-np.einsum("di,ei,mej->mdj",anchor,anchor,normal)
        comp=np.linalg.qr(normal)[0]
        for theta in cfg["mix_angles"]:pool.append(np.cos(theta)*anchor[None]+np.sin(theta)*comp)
    pool=np.concatenate(pool)
    overlap=np.sum(np.linalg.svd(np.einsum("di,mdj->mij",named["circle"],pool),compute_uv=False)**2,axis=1)/2
    angles=np.degrees(np.arccos(np.clip(np.linalg.svd(np.einsum("di,mdj->mij",named["circle"],pool),compute_uv=False),0,1)))
    eligible=(overlap<=cfg["match_overlap_max"])&(angles.max(1)>=cfg["match_max_angle_min_deg"])
    scores,recon,vrec=score_many(pool,c,g,cv,gv,syv)
    target,rec0,vrec0=score_many(named["circle"][None],c,g,cv,gv,syv);target,rec0,vrec0=target[0],rec0[0],vrec0[0]
    rc=np.abs(recon-rec0)/max(rec0,1e-12);vc=np.abs(vrec-vrec0)/max(vrec0,1e-12);pc=np.abs(scores-target)
    ixrec=int(np.argmin(np.where(eligible,rc,np.inf)))
    # Phase calibration first, reconstruction as a deterministic tie breaker only.
    ixinfo=int(np.lexsort((rc,np.where(eligible,pc,np.inf)))[0])
    valid=eligible&(pc<=cfg["phase_tolerance"])&(rc<=cfg["reconstruction_tolerance"])&(vc<=cfg["reconstruction_tolerance"])
    if valid.any():ixjoint=int(np.argmin(np.where(valid,overlap,np.inf)))
    else:ixjoint=int(np.argmin(np.where(eligible,pc/cfg["phase_tolerance"]+(rc+vc)/cfg["reconstruction_tolerance"],np.inf)))
    for name,ix in [("recon_match",ixrec),("info_match",ixinfo),("joint_match",ixjoint)]:named[name]=pool[ix]
    diag=[]
    for name,q in named.items():
        sc,cap,capv=score_many(q[None],c,g,cv,gv,syv)
        singular=np.linalg.svd(named["circle"].T@q,compute_uv=False)
        diag.append(dict(space=name,rank=q.shape[1],validation_phase_r2=float(sc[0]),target_phase_r2=float(target),
            phase_match=bool(abs(sc[0]-target)<=cfg["phase_tolerance"]),fit_capture=float(cap[0]),val_capture=float(capv[0]),
            target_fit_capture=float(rec0),target_val_capture=float(vrec0),
            reconstruction_match=bool(abs(cap[0]-rec0)/max(rec0,1e-12)<=cfg["reconstruction_tolerance"] and abs(capv[0]-vrec0)/max(vrec0,1e-12)<=cfg["reconstruction_tolerance"]),
            principal_angle_min=float(np.degrees(np.arccos(np.clip(singular[0],0,1)))),
            principal_angle_max=float(np.degrees(np.arccos(np.clip(singular[-1],0,1)))),
            overlap=float(np.sum(singular**2)/2),joint_candidate_found=bool(valid.any())))
    decoders={name:phase_decoder(r,y,q) for name,q in named.items()}
    return fit,b,named,decoders,diag,dict(pool=pool,validation_phase_scores=scores,fit_capture=recon,val_capture=vrec,overlap=overlap,eligible=eligible,
        selected=np.array([ixrec,ixinfo,ixjoint]),cv_grid=np.array([[a["lam"],a["cv_mse"]] for a in cvs]))
