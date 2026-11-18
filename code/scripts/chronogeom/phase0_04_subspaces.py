"""FP64 subspace audit. No dense 4096x4096 projectors or model access."""
import numpy as np
from scripts.chronogeom.phase0_03_geometry import fit_spaces

def orth(rows,tol=1e-6):
    rows=np.asarray(rows,dtype=np.float64)
    if not len(rows):return np.empty((0,rows.shape[-1]))
    _,s,v=np.linalg.svd(rows,full_matrices=False)
    return v[s>max(tol*s[0],1e-12)]

def relation(a,b):
    a,b=orth(a),orth(b);s=np.clip(np.linalg.svd(a@b.T,compute_uv=False),0,1);over=float(np.sum(s*s));ra,rb=len(a),len(b)
    return dict(rank_a=ra,rank_b=rb,cosines=s.tolist(),angles_deg=np.degrees(np.arccos(s)).tolist(),overlap_trace=over,overlap_fraction_a=over/ra if ra else None,overlap_fraction_b=over/rb if rb else None,projector_frobenius=float(np.sqrt(max(0,ra+rb-2*over))),chordal=float(np.sqrt(max(0,(ra+rb)/2-over))))

def decompose(h,f,tol=1e-6):
    h,f=orth(h),orth(f)
    # Stable near-intersection: singular values of residual give sin(theta),
    # avoiding sqrt(1-cos²) cancellation at nearly zero angles.
    residual=h-(h@f.T)@f;left,s,_=np.linalg.svd(residual,full_matrices=False)
    shared=orth(left[:,s<=tol].T@h)
    uh=orth(h-(h@shared.T)@shared,tol);uf=orth(f-(f@shared.T)@shared,tol)
    return shared,uh,uf,s

def spaces_for(values,hidden,seed=42):
    sp,info=fit_spaces(values,hidden,seed);h8=orth(sp['helix']);h=orth(sp['helix'][:7]);f=orth(sp['fourier'][:7]);p=orth(sp['pca'][:7]);shared,uh,uf,s=decompose(h,f)
    linear=orth(h8-(h8@f.T)@f)
    result=dict(helix7=h,fourier7=f,pca7=p,shared=shared,unique_h=uh,unique_f=uf,linear_increment=linear,helix8=h8,pca1=orth(sp['pca'][:1]),pca6=orth(sp['pca'][:6]))
    rng=np.random.default_rng(seed);r=np.linalg.qr(rng.normal(size=(hidden.shape[1],7)))[0].T
    result.update(random1=r[:1],random6=r[:6],random7=r)
    diagnostic=dict(h_f=relation(h,f),h_pca=relation(h,p),f_pca=relation(f,p),h8_f=relation(h8,f),shared_rank=len(shared),unique_h_rank=len(uh),unique_f_rank=len(uf),linear_increment_rank=len(linear),residual_singular_values=s.tolist(),intersection_rank_sensitivity={str(t):len(decompose(h,f,t)[0]) for t in [1e-7,1e-6,1e-5,1e-4]},linear_h_unique=relation(linear,uh),linear_f_unique=relation(linear,uf),orthogonality_max={k:float(np.max(np.abs(u@u.T-np.eye(len(u))))) if len(u) else 0 for k,u in result.items()},decomposition_error_h=float(np.linalg.norm(h-(h@np.concatenate([shared,uh]).T)@np.concatenate([shared,uh]))),decomposition_error_f=float(np.linalg.norm(f-(f@np.concatenate([shared,uf]).T)@np.concatenate([shared,uf]))))
    return result,diagnostic,info['helix']['mean'].astype(float)
