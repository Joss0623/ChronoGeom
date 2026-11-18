"""ETTh1 adapter inside pinned TSLib: official borders, train-only scaler, real clocks."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

class RealData:
    def __init__(self,path,L=96,label=48,H=24):
        self.df=pd.read_csv(path,usecols=['date','OT']);self.L=L;self.label=label;self.H=H
        dates=pd.DatetimeIndex(pd.to_datetime(self.df.date))
        if not (np.diff(dates.asi8)==3600*10**9).all():raise ValueError('Not strictly hourly')
        if self.df.OT.isna().any():raise ValueError('Missing OT')
        self.epoch=dates.to_numpy().astype('datetime64[h]').astype(np.int64)
        # Same population SD/mean and 12/4/4 x 30-day borders as official Dataset_ETT_hour.
        self.mu=float(self.df.OT.iloc[:8640].mean());self.sd=float(self.df.OT.iloc[:8640].std(ddof=0))
        self.values=((self.df.OT.to_numpy()-self.mu)/self.sd).astype(np.float32)
        self.train_origins=np.arange(L,8640-H+1)
        self.origins={
            'model_val':self.schedule(8640,9216),
            'fit':self.schedule(9216,10368),
            'val':self.schedule(10368,11520),
            'test':self.schedule(11520+6,14400-6,stride=2)}

    def schedule(self,start,end,stride=1):
        # Rotate start hour independently of observations; every H24 block includes all hours.
        first=start+int((-self.epoch[start])%24)
        days=np.arange(0,max(0,(end-first)//24),stride)
        starts=first+24*days+(np.arange(len(days))*5)%24
        return starts[(starts>=start)&(starts+self.H<=end)]

    @staticmethod
    def marks(t):
        shape=t.shape;dt=pd.DatetimeIndex(pd.to_datetime(t.ravel(),unit='h'))
        return np.column_stack([dt.month,dt.day,dt.dayofweek,dt.hour]).reshape(*shape,4).astype(np.float32)

    def pack(self,origins,clock_shift=0):
        origins=np.asarray(origins,dtype=int);past=origins[:,None]+np.arange(-self.L,0)
        fut=origins[:,None]+np.arange(self.H);dec=origins[:,None]+np.arange(-self.label,self.H)
        x=self.values[past];mean=x.mean(1);j=np.arange(-self.L,0,dtype=float);j-=j.mean()
        trend=(x@j)/np.sum(j*j)
        xd=np.concatenate([x[:,-self.label:],np.zeros((len(x),self.H),np.float32)],axis=1)
        return dict(x=x[...,None],xm=self.marks(self.epoch[past]+clock_shift),xd=xd[...,None],
            dm=self.marks(self.epoch[dec]+clock_shift),y=self.values[fut][...,None],t=self.epoch[fut]+clock_shift,
            actual_t=self.epoch[fut],origins=origins,ids=origins,hist_mean=mean,hist_trend=trend,hist_std=x.std(1),
            cohort=(origins>=12960).astype(int),block=(origins-11520)//336)

    def split(self,name):return self.pack(self.origins[name])

def args_for(pack,device):
    return tuple(torch.as_tensor(pack[k],dtype=torch.float32,device=device) for k in ['x','xm','xd','dm'])

def nuisance(pack):
    n,H=pack['t'].shape
    repeat=lambda a:np.repeat(a,H)
    rel=np.tile(np.arange(1,H+1)/H,n)
    day=pack['dm'][:,-H:]
    return np.column_stack([repeat(pack['hist_mean']),repeat(pack['hist_trend'])*96,repeat(pack['hist_std']),rel,
        np.eye(12)[day[:,:,0].astype(int).ravel()-1][:,1:],np.eye(7)[day[:,:,2].astype(int).ravel()][:,1:]])

def purged_folds(origins,L=96,H=24):
    origins=np.asarray(origins);weeks=(origins-origins.min())//168
    folds=[]
    for keys in np.array_split(np.unique(weeks),3):
        te=np.where(np.isin(weeks,keys))[0]
        lo=(origins[te]-L).min();hi=(origins[te]+H).max()
        tr=np.where((origins+H<=lo)|(origins-L>=hi))[0]
        if len(tr)<3 or len(te)<3:raise ValueError('Too few purged CV windows')
        folds.append((tr,te))
    return folds
