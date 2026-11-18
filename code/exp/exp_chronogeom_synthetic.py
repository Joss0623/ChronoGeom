"""Isolated TSLib experiment using unchanged official backbones and one common covariate interface."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from models import DLinear,TSMixer
from data_provider.chronogeom_synthetic import datasets
from scripts.chronogeom.phase1_representations import encode,rank

class CovariateReadout(nn.Module):
 def __init__(self,backbone):
  super().__init__();cfg=SimpleNamespace(task_name='long_term_forecast',seq_len=16,pred_len=8,moving_avg=5,enc_in=9,d_model=16,e_layers=1,dropout=0.)
  self.backbone={'DLinear':DLinear,'TSMixer':TSMixer}[backbone].Model(cfg);self.channel=nn.Linear(9,1);self.future=nn.Linear(8,1,bias=False)
  with torch.no_grad():self.channel.weight.zero_();self.channel.weight[0,0]=1;self.channel.bias.zero_();self.future.weight.zero_()
 def forward(self,x,past,future):
  # Known timestamps only; no future observation or target goes into this method.
  z=self.backbone(torch.cat([x,past],dim=-1),past,None,future)
  return self.channel(z)+self.future(future)

class Exp_ChronoGeomSynthetic:
 def __init__(self,spec,encoding,backbone,seed,evaluation_split):
  self.spec=spec;self.encoding=encoding;self.backbone=backbone;self.seed=seed;self.evaluation_split=evaluation_split
 def run(self):
  ds,tr,ev,mean,std=datasets(self.spec,self.encoding,self.evaluation_split);train,test=ds;torch.manual_seed(self.seed);model=CovariateReadout(self.backbone);opt=torch.optim.Adam(model.parameters(),lr=.003);rng=np.random.default_rng(self.seed);curve=[]
  for step in range(80):
   ix=rng.integers(len(train),size=32);opt.zero_grad();p=model(train.x[ix],train.x_mark[ix],train.y_mark[ix]);loss=((p-train.y[ix])**2).mean();loss.backward();opt.step()
   if step%10==0 or step==79:curve.append(dict(step=step,loss=float(loss.detach())))
  model.eval()
  with torch.inference_mode():
   prediction=model(test.x,test.x_mark,test.y_mark).numpy();train_mse=float(((model(train.x,train.x_mark,train.y_mark)-train.y)**2).mean())
  target=test.y.numpy();per=np.mean((prediction-target)**2,axis=(1,2));mae=np.mean(np.abs(prediction-target),axis=(1,2))*std
  # Train-only task-alignment diagnostic. No validation/test signal is used.
  e=encode(tr[0],self.encoding,track='T',period=self.spec['period'])
  if self.encoding=='shuffled_fourier':
   e=e.copy()
   for i in range(len(e)):
    r=np.random.default_rng(72000+i);e[i,:16]=e[i,r.permutation(16)];e[i,16:]=e[i,16+r.permutation(8)]
  design=np.column_stack([e.reshape(-1,e.shape[-1]),np.ones(e.shape[0]*e.shape[1])]);target_train=tr[1].ravel();coef=np.linalg.lstsq(design,target_train,rcond=1e-8)[0];alignment=float(np.mean((design@coef-target_train)**2)/np.var(target_train))
  return dict(prediction=prediction*std+mean,target=target*std+mean,loss=per,mae=mae,train_mse=train_mse,parameters=sum(p.numel() for p in model.parameters()),state=model.state_dict(),curve=curve,diagnostic=[alignment,float(rank(e[:,:16])),float(e.shape[-1])],instance_metadata=ev[2],spec=self.spec,value_mean=mean,value_std=std)
