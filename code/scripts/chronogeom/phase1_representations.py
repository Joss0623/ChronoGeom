"""Fixed numerical/time representations and train-only interface adapters."""
import numpy as np
M_ENCODINGS=['scalar','sincos','fourier','helix','ttw_phase','wrong_sincos']
T_ENCODINGS=['none','conventional','sincos','fourier','helix','ttw_phase','wrong_sincos','wrong_fourier','shuffled_fourier']
SEEDS=[0,1,2]
def split_ids():
 x=np.random.default_rng(20261024).permutation(80);return dict(train=np.sort(x[:64]),val=np.sort(x[64:72]),interpolation=np.sort(x[72:]),extrapolation=np.arange(80,160))
def periods(kind,track='M',period=16,unit=1):
 if kind in ['fourier','helix','shuffled_fourier']:return np.array([2,5,10,100] if track=='M' else [period,period/2])*unit
 if kind=='wrong_fourier':return np.array([1.5*period,.75*period])*unit
 return np.array([12 if track=='M' and kind=='wrong_sincos' else (10 if track=='M' else period*(1.5 if kind=='wrong_sincos' else 1))])*unit

def encode(x,kind,track='M',period=16,unit=1):
 x=np.asarray(x,dtype=float);scale=(79 if track=='M' else 96)*unit;p=periods(kind,track,period,unit)
 if kind=='none':return np.zeros(x.shape+(1,))
 if kind=='scalar':return x[...,None]/scale
 if kind=='ttw_phase':return ((x%p[0])/p[0])[...,None]
 if kind=='conventional':return np.stack([x/scale,(x%(24*unit))/(24*unit),((np.floor(x/(24*unit)))%7)/7],axis=-1)
 a=2*np.pi*x[...,None]/p;out=np.stack([np.cos(a),np.sin(a)],axis=-1).reshape(x.shape+(-1,))
 return np.concatenate([x[...,None]/scale,out],axis=-1) if kind=='helix' else out

def fit_interface(z,dimension):
 z=np.asarray(z);flat=z.reshape(-1,z.shape[-1]);mean=flat.mean(0);std=flat.std(0);std=np.where(std<1e-8,1,std)
 # A dense fixed isometry, not a learned encoder or information-increasing map.
 lift=np.linalg.qr(np.random.default_rng(20261024).normal(size=(dimension,z.shape[-1])))[0].T
 return dict(mean=mean,std=std,lift=lift)
def transform(z,stats):return (((z-stats['mean'])/stats['std'])@stats['lift']).astype(np.float32)
def rank(z):
 s=np.linalg.svd(z.reshape(-1,z.shape[-1])-z.reshape(-1,z.shape[-1]).mean(0),compute_uv=False)
 return int(np.sum(s>max(s[0]*1e-8,1e-12))) if len(s) else 0
