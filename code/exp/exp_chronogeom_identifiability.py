"""Standard ridge and tiny MLP forecasting readouts within the pinned TSLib tree."""
import numpy as np
import torch
from data_provider.chronogeom_identifiability import design,interface,H

def fit_readout(train,kind,energy,decoder,seed):
 stats=interface(train,kind);x,y,e=design(train,kind,stats,energy,decoder=='mlp');curve=[]
 if decoder=='linear':
  coefs=[]
  for h in range(H):
   z=np.column_stack([x[:,h],np.ones(len(x))]).astype(float);pen=np.eye(z.shape[1])*1e-6;pen[-1,-1]=0;coefs.append(np.linalg.solve(z.T@z+pen,z.T@y[:,h]))
  state=np.array(coefs);parameters=state.size
 else:
  torch.manual_seed(seed);model=torch.nn.Sequential(torch.nn.Linear(x.shape[-1],32),torch.nn.Tanh(),torch.nn.Linear(32,1))
  # P is constant in training: zero initial weight avoids random extrapolation
  # from an unidentifiable basic metadata coordinate. Same for all conditions.
  with torch.no_grad():model[0].weight[:,16]=0
  opt=torch.optim.Adam(model.parameters(),lr=.01);xt=torch.tensor(x);yt=torch.tensor(y);rng=np.random.default_rng(seed)
  for step in range(120):
   ix=rng.integers(len(x),size=32);opt.zero_grad();pred=model(xt[ix]).squeeze(-1);loss=((pred-yt[ix])**2).mean();loss.backward();opt.step()
   if step%20==0 or step==119:curve.append(dict(step=step,loss=float(loss.detach())))
  state=model.state_dict();parameters=sum(p.numel() for p in model.parameters())
 return dict(stats=stats,state=state,decoder=decoder,kind=kind,energy=energy,parameters=parameters,curve=curve)

def predict_readout(fit,pack):
 stats=fit['stats'];x,y,e=design(pack,fit['kind'],stats,fit['energy'],fit['decoder']=='mlp')
 if fit['decoder']=='linear':
  z=np.concatenate([x,np.ones(x.shape[:2]+(1,))],axis=-1);p=np.einsum('nhd,hd->nh',z,fit['state'])
 else:
  model=torch.nn.Sequential(torch.nn.Linear(x.shape[-1],32),torch.nn.Tanh(),torch.nn.Linear(32,1));model.load_state_dict(fit['state']);model.eval()
  with torch.inference_mode():p=model(torch.tensor(x)).squeeze(-1).numpy()
 p=p*stats['value_std']+stats['value_mean'];truth=pack['y']/pack['unit_value'];mu=pack['full_oracle']/pack['unit_value'];loss=np.mean((p-truth)**2,axis=1);cleanloss=np.mean((p-mu)**2,axis=1)
 return dict(prediction=p,loss=loss,clean_loss=cleanloss,mae=np.mean(np.abs(p-truth),axis=1),encoding_rms=float(np.sqrt(np.mean(np.sum(e**2,axis=-1)))),design_rank=int(np.linalg.matrix_rank(np.column_stack([x[:,0].astype(float),np.ones(len(x))]),tol=1e-6)),x_shape=list(x.shape))
