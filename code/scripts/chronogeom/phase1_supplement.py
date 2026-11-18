"""Post-primary analytical capacity/time calibration only; no model or rule changes."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scripts.chronogeom.phase1_representations import *
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output;out.mkdir(exist_ok=False);rows=[];train=split_ids()['train'];x=np.arange(200)
for ps in [[10],[10,100],[2,5,10,100]]:
 for linear in [False,True]:
  def e(v):
   a=2*np.pi*v[:,None]/np.array(ps);z=np.stack([np.cos(a),np.sin(a)],axis=-1).reshape(len(v),-1);return np.column_stack([v/79,z]) if linear else z
  st=fit_interface(e(train),9);z=transform(e(x),st);a=transform(e(np.sort(train)),st);s=np.median(np.linalg.norm(np.diff(a,axis=0),axis=1));d=pdist(z.astype(float))/s;rows.append(dict(periods=str(ps),linear=linear,native_dimension=z.shape[1] if False else e(x).shape[-1],effective_rank=rank(e(train)),near_collision_pairs=int(np.sum(d<1e-6))))
pd.DataFrame(rows).to_csv(out/'dimension_sweep.csv',index=False);rows=[]
for p in [16,24]:
 for k in T_ENCODINGS:
  if k=='shuffled_fourier':continue
  trainx=np.arange(96);grid=np.arange(216);z=encode(grid,k,'T',p);st=fit_interface(encode(trainx,k,'T',p),8);dist=pdist(transform(z,st).astype(float));rows.append(dict(period_hours=p,encoding=k,native_dimension=z.shape[-1],effective_train_rank=rank(encode(trainx,k,'T',p)),near_collision_pairs=int(np.sum(dist<1e-6)),full_vector_period_return=float(np.mean(np.linalg.norm(encode(trainx+p,k,'T',p)-encode(trainx,k,'T',p),axis=1)))))
pd.DataFrame(rows).to_csv(out/'time_diagnostics.csv',index=False)
