"""TSLib controlled information regimes; no future observations in the model interface."""
import numpy as np
from torch.utils.data import Dataset
TASKS=['A','B','C','D'];ENCODINGS=['none','raw_timestamp','sincos','fourier','helix','ttw_phase'];COHORTS=['iid','period','noise','trend','phase','clock_range','units','combo','new_system'];L=16;H=32

def generate(task,split,n,cohort='iid'):
 actual='iid' if cohort=='units' else cohort;seed=20261024+TASKS.index(task)*100000+['train','val','test'].index(split)*10000+COHORTS.index(actual)*1000;rng=np.random.default_rng(seed);p=24 if cohort in ['period','combo'] else 16;sigma=.2 if cohort=='noise' else (.15 if cohort=='combo' else .05);slope=.06 if cohort in ['trend','combo'] else .03;u=1000. if cohort=='units' else 1.;h0=np.arange(-15,1);h=np.arange(1,H+1);rows=[]
 for i in range(n):
  c=rng.normal(0,.5);b=rng.uniform(-slope,slope);t0=int(rng.integers(4*p));psi=2*np.pi*t0/p;clock=float(t0)
  if cohort in ['phase','combo']:
   if task=='D':clock+=p/4
   elif task in ['B','C']:psi+=np.pi/3;clock+=p/6
  if cohort in ['clock_range','combo']:clock+=8*p
  past=c+b*h0;mu=c+b*h
  if task in ['C','D']:past=past+np.sin(psi+2*np.pi*h0/p)
  if task in ['B','C','D']:mu=mu+np.sin(psi+2*np.pi*h/p)
  if cohort=='new_system':
   z=rng.uniform(.2,.8);path=[]
   for j in range(L+H):path.append(z);z=3.2*z*(1-z)
   past=np.array(path[:L]);mu=np.array(path[L:]);hist=[];z=past[-1]
   for j in range(H):z=3.2*z*(1-z);hist.append(z)
   hist=np.array(hist)
  elif task in ['A','B']:hist=past[-1]+(past[-1]-past[-2])*h
  else:
   design=lambda hh:np.column_stack([np.ones(len(hh)),hh,np.sin(2*np.pi*hh/p),np.cos(2*np.pi*hh/p)])
   hist=design(h)@np.linalg.lstsq(design(h0),past,rcond=1e-12)[0]
  full=mu.copy();noise=rng.normal(0,sigma,H);y=mu+noise;rows.append((past*u,y*u,(clock+h)*u,hist*u,full*u,p*u))
 a=list(zip(*rows));return dict(x=np.array(a[0]),y=np.array(a[1]),future_time=np.array(a[2]),history_oracle=np.array(a[3]),full_oracle=np.array(a[4]),period=np.array(a[5]),unit_time=u,unit_value=u,sigma=sigma,information_gap=.5 if task=='B' and cohort!='new_system' else 0.,seed=seed,task=task,cohort=cohort,split=split)

def encoding(pack,kind):
 t=pack['future_time']/pack['unit_time'];p=pack['period'][:,None]/pack['unit_time'];a=2*np.pi*t/p
 if kind=='none':return np.zeros(t.shape+(0,))
 if kind=='raw_timestamp':return (t/64)[...,None]
 if kind=='ttw_phase':return ((t%p)/p)[...,None]
 harmonics=[1] if kind=='sincos' else [1,2,3];z=np.stack([f(k*a) for k in harmonics for f in [np.cos,np.sin]],axis=-1)
 return np.concatenate([(t/64)[...,None],z],axis=-1) if kind=='helix' else z

def interface(train,kind):
 z=encoding(train,kind);d=z.shape[-1];mean=z.reshape(-1,d).mean(0) if d else np.zeros(0);q=np.linalg.qr(np.random.default_rng(20261024).normal(size=(8,d)))[0].T;rms=float(np.sqrt(np.mean(np.sum((z-mean)**2,axis=-1))));s=np.linalg.svd((z-mean).reshape(-1,d),compute_uv=False) if d else np.array([]);rank=int(np.sum(s>max(s[0]*1e-8,1e-12))) if len(s) else 0
 return dict(mean=mean,lift=q,rms=rms,rank=rank,native_dimension=d,value_mean=float(np.mean(train['x']/train['unit_value'])),value_std=float(np.std(train['x']/train['unit_value'])))
def design(pack,kind,stats,energy,nonlinear=False):
 e=(encoding(pack,kind)-stats['mean'])@stats['lift']
 if energy=='rms' and stats['rms']>0:e=e/stats['rms']
 x=(pack['x']/pack['unit_value']-stats['value_mean'])/stats['value_std'];p=(pack['period']/pack['unit_time']-16)/16;parts=[np.broadcast_to(x[:,None,:],(len(x),H,L)),np.broadcast_to(p[:,None,None],(len(x),H,1)),e]
 if nonlinear:parts.insert(2,np.broadcast_to((np.arange(1,H+1)/H)[None,:,None],(len(x),H,1)))
 f=np.concatenate(parts,axis=-1).astype(np.float32);y=((pack['y']/pack['unit_value']-stats['value_mean'])/stats['value_std']).astype(np.float32)
 return f,y,e
class Dataset_ChronoGeomIdentifiability(Dataset):
 def __init__(self,pack):self.pack=pack
 def __len__(self):return len(self.pack['x'])
 def __getitem__(self,i):return self.pack['x'][i,:,None],self.pack['y'][i,:,None],np.zeros((L,0)),self.pack['future_time'][i,:,None]
