import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,sys,json,time,hashlib,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from scripts.chronogeom.phase1_representations import *
TASKS=['magnitude','order','addition','subtraction','mod10','phase_shift']
def task_data(ids,task):
 if task in ['order','addition','subtraction']:
  a,b=np.meshgrid(ids,ids,indexing='ij');a,b=a.ravel(),b.ravel()
 else:a=ids;b=ids
 if task=='magnitude':y=a[:,None]/79
 elif task=='order':y=np.eye(2)[(a>b).astype(int)]
 elif task=='addition':y=(a+b)[:,None]/158
 elif task=='subtraction':y=(a-b)[:,None]/79
 elif task=='mod10':y=np.eye(10)[a%10]
 else:y=np.stack([np.cos(2*np.pi*(a+3)/10),np.sin(2*np.pi*(a+3)/10)],axis=1)
 return a,b,y.astype(np.float32)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output;out.mkdir(exist_ok=False);torch.set_num_threads(2);start=time.monotonic();splits=split_ids();rows=[];curves=[];preds=out/'predictions';preds.mkdir();states=out/'states';states.mkdir()
 for task in TASKS:
  data={k:task_data(v,task) for k,v in splits.items()}
  for kind in M_ENCODINGS:
   st=fit_interface(encode(splits['train'],kind),9);features={}
   for split,(a,b,y) in data.items():
    x=transform(encode(a,kind),st);features[split]=np.concatenate([x,transform(encode(b,kind),st)],axis=1) if task in ['order','addition','subtraction'] else x
   x=features['train'];y=data['train'][2];design=np.column_stack([x,np.ones(len(x))]).astype(float);reg=np.eye(design.shape[1])*1e-6;reg[-1,-1]=0;coef=np.linalg.solve(design.T@design+reg,design.T@y)
   for readout,seed in [('linear',-1)]+[('mlp',i) for i in SEEDS]:
    if time.monotonic()-start>285:raise TimeoutError('Track M300s cap; completed results retained')
    if readout=='mlp':
     torch.manual_seed(seed);model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.Tanh(),torch.nn.Linear(32,y.shape[1]));opt=torch.optim.Adam(model.parameters(),lr=.01);rng=np.random.default_rng(seed);xt=torch.tensor(x);yt=torch.tensor(y)
     for step in range(200):
      ix=rng.integers(len(x),size=128);opt.zero_grad();loss=torch.mean((model(xt[ix])-yt[ix])**2);loss.backward();opt.step()
      if step%20==0 or step==199:curves.append(dict(task=task,encoding=kind,readout=readout,seed=seed,step=step,loss=float(loss.detach())))
     torch.save(model.state_dict(),states/f'{task}_{kind}_{seed}.pt');parameters=sum(p.numel() for p in model.parameters())
     predict=lambda z:model(torch.tensor(z)).detach().numpy()
    else:np.save(states/f'{task}_{kind}_linear.npy',coef);parameters=coef.size;predict=lambda z:np.column_stack([z,np.ones(len(z))])@coef
    for split,(a,b,target) in data.items():
     p=predict(features[split]);per=np.mean((p-target)**2,axis=1);accuracy=None;mae=None;angle=None
     if task in ['order','mod10']:accuracy=float(np.mean(p.argmax(1)==target.argmax(1)))
     elif task=='phase_shift':angle=float(np.mean(np.abs(np.angle(np.exp(1j*(np.arctan2(p[:,1],p[:,0])-np.arctan2(target[:,1],target[:,0])))))))
     else:
      scale=158 if task=='addition' else 79;mae=float(np.mean(np.abs(p-target))*scale);accuracy=float(np.mean(np.rint(p[:,0]*scale)==np.rint(target[:,0]*scale)))
     rows.append(dict(task=task,encoding=kind,readout=readout,seed=seed,split=split,n=len(a),parameters=parameters,mse=float(per.mean()),accuracy=accuracy,mae_original_units=mae,phase_mae_radians=angle,native_dimension=encode(np.array([0]),kind).shape[-1]))
     np.savez_compressed(preds/f'{task}_{kind}_{readout}_{seed}_{split}.npz',a=a,b=b,target=target,prediction=p,loss=per)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
 print('COMPLETE M',len(rows),'rows',time.monotonic()-start,flush=True);pd.DataFrame(curves).to_csv(out/'learning_curves.csv',index=False);(out/'metadata.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seeds=SEEDS,threads=2,torch=torch.__version__,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__':main()
