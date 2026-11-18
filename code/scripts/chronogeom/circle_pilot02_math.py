"""Pilot02 geometry and calibrated subspaces. No forecasting model training."""
import numpy as np
from scipy.stats import rankdata
from scripts.chronogeom.tslib_circle_transfer import fit_circle

def basis(a,rank=2):
    _,s,v=np.linalg.svd(a,full_matrices=False)
    r=int(np.sum(s>max(1e-12,s[0]*1e-9 if len(s) else 0)))
    return v[:min(rank,r)].T

def centroids(x,h):
    return np.stack([x[np.asarray(h)==i].mean(0) for i in range(24)])

def distance(x):
    return np.sqrt(np.maximum(np.sum(x*x,1)[:,None]+np.sum(x*x,1)[None,:]-2*x@x.T,0))

def order_metrics(c,q,center):
    d=distance(c);ii,jj=np.triu_indices(24,1)
    geo=np.minimum(abs(ii-jj),24-abs(ii-jj))
    a=rankdata(d[ii,jj]);b=rankdata(geo)
    rho=float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 else np.nan
    ds=d.copy();np.fill_diagonal(ds,np.inf)
    near=np.argsort(ds,axis=1)[:,:2]
    hits=np.mean([sum(k in near[i] for k in [(i-1)%24,(i+1)%24])/2 for i in range(24)])
    edges=d[np.arange(24),(np.arange(24)+1)%24]
    y=(c-center)@q
    if y.shape[1]>=2 and np.var(y[:,1])>1e-16:
        angle=np.arctan2(y[:,1],y[:,0]);step=np.angle(np.exp(1j*(np.roll(angle,-1)-angle)))
        consistent=max(np.mean(step>0),np.mean(step<0));winding=abs(step.sum()/(2*np.pi))
        rad=np.linalg.norm(y,axis=1);rcv=rad.std()/max(rad.mean(),1e-12)
    else:consistent=winding=rcv=np.nan
    four=np.fft.rfft(c-c.mean(0),axis=0)
    energy=np.sum(abs(four)**2,axis=1);energy[1:-1]*=2;energy[0]=0
    harmonic=energy[1]/max(energy.sum(),1e-12)
    return dict(cyclic_distance_rho=rho,adjacent_ratio=float(edges.mean()/max(d[ii,jj].mean(),1e-12)),
        nearest_two_hit=float(hits),wrap_ratio=float(edges[-1]/max(edges[:-1].mean(),1e-12)),
        angular_consistency=float(consistent),winding=float(winding),radius_cv=float(rcv),
        harmonic1_fraction=float(harmonic)),d,energy,y

def native_audit(fitx,fith,testx,testh,n_traj,permutations=1000,boot=32,seed=1729):
    c=centroids(fitx,fith);tc=centroids(testx,testh);center=c.mean(0)
    _,sv,vt=np.linalg.svd(c-center,full_matrices=False);q=vt[:2].T
    rank=int(np.sum(sv>max(sv[0]*1e-8,1e-12)))
    qe=q if rank>=2 else q[:,:rank]
    metric,d,e,y=order_metrics(tc,qe,center)
    metric.update(centroid_rank=rank,pca2_fraction=float(np.sum(sv[:2]**2)/max(np.sum(sv**2),1e-12)),
        pca_axis_ratio=float(sv[1]/max(sv[0],1e-12)),eigen_gap23=float((sv[1]**2-sv[2]**2)/max(sv[1]**2,1e-12)))
    rng=np.random.default_rng(seed)
    null={k:[] for k in ["cyclic_distance_rho","adjacent_ratio","nearest_two_hit","angular_consistency","harmonic1_fraction"]}
    for _ in range(permutations):
        m,_,_,_=order_metrics(tc[rng.permutation(24)],qe,center)
        for k in null:null[k].append(m[k])
    for k,z in null.items():
        a=np.asarray(z);v=metric[k]
        metric[k+"_permutation_p"]=float((1+np.sum(a<=v if k=="adjacent_ratio" else a>=v))/(len(a)+1)) if np.isfinite(v) and len(a) else np.nan
    angles=[]
    fitblocks=fitx.reshape(n_traj,24,-1);hblocks=np.asarray(fith).reshape(n_traj,24)
    if rank>=2:
        for _ in range(boot):
            ids=rng.integers(n_traj,size=n_traj)
            cc=centroids(fitblocks[ids].reshape(-1,fitx.shape[1]),hblocks[ids].ravel())
            qb=basis(cc-cc.mean(0))
            if qb.shape[1]==2:
                angles.append(np.degrees(np.arccos(np.clip(np.linalg.svd(q.T@qb,compute_uv=False),0,1))))
    metric["stability_max_angle_deg"]=float(np.max(angles)) if angles else np.nan
    passfields=[metric["pca2_fraction"]>=.8,metric["cyclic_distance_rho"]>=.8,metric["adjacent_ratio"]<=.4,
        metric["nearest_two_hit"]>=.75,metric["angular_consistency"]>=.9,
        abs(metric["winding"]-1)<1e-5,metric["radius_cv"]<=.25,metric["harmonic1_fraction"]>=.8]
    metric["native_circle_gate"]=bool(all(passfields))
    return metric,dict(fit_centroids=c,test_centroids=tc,pca_basis=q,center=center,distance_matrix=d,harmonic_energy=e,
        raw_projection=y,stability_angles=np.asarray(angles),**{"null_"+k:np.array(v) for k,v in null.items()})

def residual_fit(x,nuis):
    z=np.column_stack([np.ones(len(x)),nuis]);b=np.linalg.lstsq(z,x,rcond=1e-10)[0]
    return x-z@b,b

def apply_residual(x,nuis,b):
    return x-np.column_stack([np.ones(len(x)),nuis])@b

def stats(r,y):
    return r.T@r/len(r),r.T@y/len(r),float(np.mean(np.sum(y*y,axis=1)))

def score_many(q,c,g,cv,gv,syv):
    qq=np.asarray(q)
    c2=np.einsum("mdi,de,mej->mij",qq,c,qq);g2=np.einsum("mdi,de->mie",qq,g)
    coef=np.linalg.pinv(c2,rcond=1e-10)@g2
    w=qq@coef
    mse=np.einsum("mdi,de,mei->m",w,cv,w)-2*np.einsum("mdi,di->m",w,gv)+syv
    scores=1-mse/max(syv,1e-12)
    rec=np.einsum("mdi,de,mei->m",qq,c,qq)/max(np.trace(c),1e-12)
    valrec=np.einsum("mdi,de,mei->m",qq,cv,qq)/max(np.trace(cv),1e-12)
    return scores,rec,valrec

def phase_decoder(r,y,q):
    return np.linalg.lstsq(np.column_stack([r@q,np.ones(len(r))]),y,rcond=1e-10)[0]

def decode_metrics(r,y,q,coef):
    p=np.column_stack([r@q,np.ones(len(r))])@coef
    err=np.sum((p-y)**2,axis=1)
    return float(1-err.mean()/max(np.mean(np.sum((y-y.mean(0))**2,axis=1)),1e-12)),err,p

def cv_basis(x,h,nuis,ntraj,grid):
    y=np.column_stack([np.cos(2*np.pi*h/24),np.sin(2*np.pi*h/24)])
    groups=np.arange(len(x)).reshape(ntraj,24)
    folds=np.array_split(np.random.default_rng(1729).permutation(ntraj),3);results=[]
    for lam in grid:
        losses=[]
        for fold in folds:
            te=groups[fold].ravel();tr=np.setdiff1d(np.arange(len(x)),te)
            r,b=residual_fit(x[tr],nuis[tr]);rt=apply_residual(x[te],nuis[te],b)
            c=r.T@r/len(r);g=r.T@y[tr]/len(r)
            w=np.linalg.solve(c+np.eye(x.shape[1])*max(np.trace(c)/x.shape[1],1e-9)*lam,g)
            q=basis(w.T);coef=phase_decoder(r,y[tr],q)
            _,err,_=decode_metrics(rt,y[te],q,coef);losses.append(float(err.mean()))
        results.append(dict(lam=lam,cv_mse=float(np.mean(losses))))
    selected=min(results,key=lambda v:v["cv_mse"])["lam"]
    r,b=residual_fit(x,nuis);c=r.T@r/len(r);g=r.T@y/len(r)
    w=np.linalg.solve(c+np.eye(x.shape[1])*max(np.trace(c)/x.shape[1],1e-9)*selected,g)
    return basis(w.T),results

def calibrate(x,h,nuis,vx,vh,vn,ntraj,cfg,seed):
    fit=fit_circle(x,h,24,nuisance=nuis);r,b=residual_fit(x,nuis);vr=apply_residual(vx,vn,b)
    y=np.column_stack([np.cos(2*np.pi*h/24),np.sin(2*np.pi*h/24)])
    vy=np.column_stack([np.cos(2*np.pi*vh/24),np.sin(2*np.pi*vh/24)])
    c,g,_=stats(r,y);cv,gv,syv=stats(vr,vy)
    qcv,cvs=cv_basis(x,h,nuis,ntraj,cfg["ridge_grid"])
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
