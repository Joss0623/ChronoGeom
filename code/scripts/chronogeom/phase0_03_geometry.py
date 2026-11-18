"""Train-only rank-controlled subspaces and explicit intervention operators."""
import numpy as np
from scipy.linalg import eigh
from scripts.chronogeom.representation_probe import basis

METHODS=('helix','pca','fourier','linear','polynomial','random','shuffled')

def principal_rows(y, rank):
    """PCA via a small row Gram matrix, equivalent to SVD, no test observations."""
    y=np.asarray(y,dtype=np.float64)
    gram=y@y.T
    vals,vec=eigh(gram,subset_by_index=(max(0,len(y)-rank),len(y)-1))
    order=np.argsort(vals)[::-1];vals=vals[order];vec=vec[:,order]
    keep=vals>max(1e-14,vals[0]*1e-12)
    return (vec[:,keep].T@y)/np.sqrt(vals[keep,None])

def fit_spaces(values, hidden, seed=42):
    a=np.asarray(values);y=np.asarray(hidden,dtype=np.float64);mean=y.mean(0);spaces={};info={}
    pca=principal_rows(y-mean,8)
    for method in METHODS:
        coef=None
        if method=='pca':
            u=pca;rank=len(u);columns=None;design_rank=None
        elif method=='random':
            u=np.linalg.qr(np.random.default_rng(seed).normal(size=(y.shape[1],8)))[0].T
            rank=8;columns=None;design_rank=None
        else:
            kind={'shuffled':'helix'}.get(method,method)
            design=basis(a,kind,degree=8)
            if method=='shuffled':
                ids=np.unique(a);permuted=np.random.default_rng(seed).permutation(ids)
                mapping=dict(zip(ids,permuted));design=basis([mapping[v] for v in a],'helix')
            coef,_,design_rank,_=np.linalg.lstsq(design,y,rcond=1e-10)
            # Orthonormalize centered design, then SVD the fitted variation map.
            # This is reduced-rank regression ordered by train fitted-state variance;
            # it avoids polynomial coefficient magnitude/cancellation defining rank.
            xc=design[:,:-1]-design[:,:-1].mean(0)
            q,sx,_=np.linalg.svd(xc,full_matrices=False)
            feature_rank=int(np.sum(sx>max(sx[0]*1e-10,1e-12)))
            fitted_map=q[:,:feature_rank].T@(y-mean)
            _,s,v=np.linalg.svd(fitted_map,full_matrices=False)
            rank=int(np.sum(s>max(s[0]*1e-8,1e-12))) if len(s) else 0
            u=v[:rank]
            columns=design.shape[1]
        spaces[method]=u.astype(np.float32)
        info[method]={'native_rank':len(u),'design_columns':columns,'design_rank':None if design_rank is None else int(design_rank),'coef':coef,'mean':mean.astype(np.float32)}
    return spaces,info

def project(x,u):return (x@u.T)@u

def make_change(base, donor, u, operation, mean=None, target_norm=None):
    """Return ideal FP32 replacement and diagnostics; matched norms are NOT projections."""
    base=np.asarray(base,dtype=np.float32)
    if operation=='noop':return base.copy(),np.zeros(len(base),dtype=bool)
    if operation=='swap':delta=project(np.asarray(donor,dtype=np.float32)-base,u)
    elif operation=='ablate_mean':delta=-project(base-np.asarray(mean,dtype=np.float32),u)
    elif operation=='full_swap':return np.asarray(donor,dtype=np.float32).copy(),np.zeros(len(base),dtype=bool)
    elif operation=='full_mean':return np.broadcast_to(mean,base.shape).copy(),np.zeros(len(base),dtype=bool)
    elif operation=='full_zero':return np.zeros_like(base),np.zeros(len(base),dtype=bool)
    else:raise ValueError(operation)
    norms=np.linalg.norm(delta,axis=1);zero=norms<=1e-12
    if target_norm is not None:
        factor=np.divide(target_norm,norms,out=np.zeros_like(norms),where=~zero)
        delta=delta*factor[:,None]
    return base+delta,zero

def subspace_error(y,mean,u):
    y=np.asarray(y,dtype=np.float64);pred=mean+project(y-mean,u.astype(np.float64))
    return float(np.mean((y-pred)**2))

def identity_weights(pairs,draws=2000,seed=1729,mode='shared_nodes'):
    """Node-resampling sensitivity, not an IID pair bootstrap or guaranteed finite-sample CI.

    Shared numerical identities receive the same weight across operand roles. Each
    distinct identity in an edge contributes once, including self-equal a/b cases.
    """
    rng=np.random.default_rng(seed)
    if mode=='shared_nodes':
        groups=sorted({v for p in pairs for v in [p['clean_a'],p['corrupt_a'],p['b']]})
        counts=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=draws)
        lookup={g:i for i,g in enumerate(groups)}
        return np.column_stack([np.prod(counts[:,[lookup[v] for v in sorted({p['clean_a'],p['corrupt_a'],p['b']})]],axis=1) for p in pairs])
    key={'clean':'clean_a','corrupt':'corrupt_a','b':'b'}[mode];groups=sorted({p[key] for p in pairs});lookup={g:i for i,g in enumerate(groups)}
    counts=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=draws)
    return counts[:,[lookup[p[key]] for p in pairs]]

def interval(values,weights):
    values=np.asarray(values,dtype=float);den=weights.sum(1);valid=den>0
    estimates=(weights[valid]@values)/den[valid]
    return {'point':float(values.mean()),'low':float(np.quantile(estimates,.025)),'high':float(np.quantile(estimates,.975)),'valid_draws':int(valid.sum()),'zero_weight_draws':int((~valid).sum())}
