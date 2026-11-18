"""Strict access-level ETTh1 adapter: old test quarantined, final reserve never parsed."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_provider.chronogeom_circle_real import RealData

class RecoveryData(RealData):
    def __init__(self,path,run=None,access='develop'):
        if access not in ['develop','qualification','blind']:raise ValueError(access)
        if access!='develop':
            if run is None or not (Path(run)/'training_lock.json').exists():raise PermissionError('Final training lock required')
        if access=='blind':
            gate=json.loads((Path(run)/'qualification.json').read_text())
            if not (gate['clock_qualified'] or gate['linear_qualified']):raise PermissionError('Qualification failed; blind data remains sealed')
        self.access=access;self.read_limit={'develop':10368,'qualification':11520,'blind':15912}[access]
        self.df=pd.read_csv(path,usecols=['date','OT'],nrows=self.read_limit);self.L=96;self.label=48;self.H=24
        assert len(self.df)==self.read_limit
        dates=pd.DatetimeIndex(pd.to_datetime(self.df.date));assert (np.diff(dates.asi8)==3600*10**9).all()
        self.epoch=dates.to_numpy().astype('datetime64[h]').astype(np.int64)
        self.mu=float(self.df.OT.iloc[:8640].mean());self.sd=float(self.df.OT.iloc[:8640].std(ddof=0))
        self.values=((self.df.OT.to_numpy()-self.mu)/self.sd).astype(np.float32)
        self.train_origins=np.arange(96,8640-24+1)
        self.origins={'select':np.arange(8640,10368-24+1,25),'fit':np.arange(7488,8640-24+1,25)}
        if access in ['qualification','blind']:self.origins['calibration']=np.arange(10368,11520-24+1,25)
        if access=='blind':self.origins['test']=np.arange(14400,15912-24+1,25)
    def pack(self,origins,clock_shift=0):
        origins=np.asarray(origins,dtype=int)
        if np.any((origins<14400)&(origins+24>11520)):raise PermissionError('Pilot01 test targets quarantined')
        if np.any(origins<96) or np.any(origins+24>self.read_limit):raise PermissionError('Data access level exceeded')
        result=super().pack(origins,clock_shift)
        result['cohort']=(origins>=15156).astype(int);result['block']=(origins-14400)//168
        return result
