"""Independent known-mechanism trajectories for the isolated TSLib Phase1 pilot."""
import numpy as np
import torch
from torch.utils.data import Dataset
from scripts.chronogeom.phase1_representations import encode,fit_interface,transform
FAMILIES=['trend','periodic','mixed','regime']
def task_spec(stage,family):
 return dict(stage=stage,family=family,period=16 if stage=='development' else 24,switch=64 if stage=='development' else 80,phase=0 if stage=='development' else .4,amplitude_range=[.8,1.2] if stage=='development' else [.6,1.6],slope_range=[.2,.6] if stage=='development' else [.4,.9],noise_sd=.02,seq_len=16,pred_len=8)
def generate(spec,split):
 n={'train':128,'val':32,'test':64}[split];stage=spec['stage'];family=spec['family'];seed=20261024+FAMILIES.index(family)*100+(['train','val','test'].index(split))*10+(10000 if stage!='development' else 0);rng=np.random.default_rng(seed);times=[];values=[];metadata=[]
 for i in range(n):
  t0=int(rng.integers(96,192)) if stage!='development' and split=='test' else int(rng.integers(0,96));t=t0+np.arange(24);amp=rng.uniform(*spec['amplitude_range']);slope=rng.uniform(*spec['slope_range']);offset=rng.uniform(-.5,.5);p=spec['period'];switch=spec['switch']
  trend=slope*t/32;phase=spec['phase']+2*np.pi*t/p;periodic=amp*np.sin(phase)
  if family=='trend':signal=offset+trend
  elif family=='periodic':signal=offset+periodic
  elif family=='mixed':signal=offset+trend+periodic
  else:
   phase=spec['phase']+2*np.pi*(np.minimum(t,switch)/p+np.maximum(t-switch,0)/(p/2));signal=offset+trend+.3*slope*np.maximum(t-switch,0)/32+amp*np.where(t>=switch,1.5,1)*np.sin(phase)+.5*(t>=switch)
  y=signal+rng.normal(0,spec['noise_sd'],len(t));times.append(t);values.append(y);metadata.append(dict(instance_id=f'{stage}_{family}_{split}_{i}',t0=t0,amplitude=float(amp),slope=float(slope),offset=float(offset),data_seed=seed))
 return np.array(times,dtype=float),np.array(values,dtype=float),metadata

class ChronoGeomSynthetic(Dataset):
 """One independent trajectory -> one TSLib (x,y,x_mark,y_mark) sample."""
 def __init__(self,times,values,marks,value_mean,value_std):
  self.x=torch.tensor(((values[:,:16]-value_mean)/value_std)[...,None],dtype=torch.float32);self.y=torch.tensor(((values[:,16:]-value_mean)/value_std)[...,None],dtype=torch.float32);self.x_mark=torch.tensor(marks[:,:16],dtype=torch.float32);self.y_mark=torch.tensor(marks[:,16:],dtype=torch.float32);self.times=times
 def __len__(self):return len(self.x)
 def __getitem__(self,index):return self.x[index],self.y[index],self.x_mark[index],self.y_mark[index]

def datasets(spec,encoding,evaluation_split):
 # evaluation trajectories are generated only when the caller requests the split.
 tr=generate(spec,'train');ev=generate(spec,evaluation_split);raw=encode(tr[0],encoding,track='T',period=spec['period']);stats=fit_interface(raw[:,:16],8);mean=float(tr[1][:,:16].mean());std=float(tr[1][:,:16].std());out=[]
 for j,pack in enumerate([tr,ev]):
  z=transform(encode(pack[0],encoding,track='T',period=spec['period']),stats)
  if encoding=='shuffled_fourier':
   z=z.copy()
   for i in range(len(z)):
    rng=np.random.default_rng(72000+j*1000+i);z[i,:16]=z[i,rng.permutation(16)];z[i,16:]=z[i,16+rng.permutation(8)]
  out.append(ChronoGeomSynthetic(pack[0],pack[1],z,mean,std))
 return out,tr,ev,mean,std
