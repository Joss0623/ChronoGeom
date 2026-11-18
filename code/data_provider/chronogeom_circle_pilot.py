"""Controlled calendar-driven forecasting dataset; all future values withheld."""
import numpy as np
import torch
L=24
H=24
LABEL=12
P=24
SPLITS=["train","val","fit","test"]

def make_pack(split,n,shift=0):
    rng=np.random.default_rng(20261024+1000*SPLITS.index(split))
    hour=np.tile(np.arange(24),int(np.ceil(n/24)))[:n].copy();rng.shuffle(hour)
    t0=rng.integers(0,28,n)*24+hour+shift
    c=rng.uniform(-.4,.4,n);b=rng.uniform(-.004,.004,n)
    j=np.arange(-L+1,H+1);t=t0[:,None]+j
    mean=c[:,None]+b[:,None]*j+np.sin(2*np.pi*t/P)
    noise=rng.normal(size=mean.shape)*np.where(j<=0,.15,.03)
    y=mean+noise
    day=np.floor_divide(t,24)
    marks=np.stack([np.ones_like(t),day%28+1,day%7,t%24],axis=-1).astype(np.float32)
    dec=np.concatenate([y[:,L-LABEL:L],np.zeros((n,H))],axis=1)[...,None]
    return dict(x=y[:,:L,None].astype(np.float32),xm=marks[:,:L],xd=dec.astype(np.float32),
                dm=marks[:,L-LABEL:],y=y[:,L:,None].astype(np.float32),
                mean=mean[:,L:,None].astype(np.float32),t=t[:,L:],
                c=c,b=b,ids=np.arange(n)+100000*SPLITS.index(split),t0=t0)

def args_for(pack,device):
    return tuple(torch.as_tensor(pack[k],device=device) for k in ["x","xm","xd","dm"])

def nuisance(pack):
    n=len(pack["x"]);h=np.broadcast_to(np.arange(H)[None,:]/H,(n,H))
    d=np.floor_divide(pack["t"],24)
    columns=[np.broadcast_to(pack["c"][:,None],(n,H)),np.broadcast_to(pack["b"][:,None],(n,H)),h]
    columns.extend([(d%28==i).astype(float) for i in range(1,28)])
    columns.extend([(d%7==i).astype(float) for i in range(1,7)])
    return np.stack(columns,axis=-1).reshape(n*H,-1)
