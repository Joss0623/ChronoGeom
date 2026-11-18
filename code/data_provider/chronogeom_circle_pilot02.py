"""New independent trajectories with unchanged Pilot01 physical data law."""
import numpy as np
from data_provider.chronogeom_circle_pilot import L,H,LABEL,P,args_for,nuisance
SPLITS=["fit","val","test_a","test_b"]

def make_pack(split,n,shift=0,independent_noise=False):
    idx=SPLITS.index(split);rng=np.random.default_rng(20281024+1000*idx)
    hour=np.tile(np.arange(24),int(np.ceil(n/24)))[:n].copy();rng.shuffle(hour)
    t0=rng.integers(0,28,n)*24+hour+shift
    c=rng.uniform(-.4,.4,n);b=rng.uniform(-.004,.004,n)
    j=np.arange(-L+1,H+1);t=t0[:,None]+j
    mean=c[:,None]+b[:,None]*j+np.sin(2*np.pi*t/P)
    nrng=np.random.default_rng(20291024+idx*1000) if independent_noise else rng
    y=mean+nrng.normal(size=mean.shape)*np.where(j<=0,.15,.03)
    d=np.floor_divide(t,24)
    mark=np.stack([np.ones_like(t),d%28+1,d%7,t%24],axis=-1).astype(np.float32)
    return dict(x=y[:,:L,None].astype(np.float32),xm=mark[:,:L],
        xd=np.concatenate([y[:,L-LABEL:L],np.zeros((n,H))],axis=1)[...,None].astype(np.float32),
        dm=mark[:,L-LABEL:],y=y[:,L:,None].astype(np.float32),mean=mean[:,L:,None].astype(np.float32),
        t=t[:,L:],t0=t0,c=c,b=b,ids=np.arange(n)+(idx+1)*1000000,
        cohort=np.full(n,idx))

def test_pack(n=96,shift=0,independent_noise=False):
    a=make_pack("test_a",n,shift,independent_noise);b=make_pack("test_b",n,shift,independent_noise)
    return {k:np.concatenate([a[k],b[k]]) for k in a}
