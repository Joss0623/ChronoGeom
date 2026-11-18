import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scripts.chronogeom.phase1_representations import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False);train=split_ids()['train'];grid=np.arange(200.);rows=[];precision=[];cache={}
 for kind in M_ENCODINGS:
  raw=encode(grid,kind);z=encode(train,kind);st=fit_interface(z,9);norm=transform(raw,st).astype(float);nn=np.linalg.norm(np.diff(transform(encode(np.sort(train),kind),st),axis=0),axis=1);scale=float(np.median(nn[nn>1e-8]));dist=pdist(norm)/scale;rawdist=pdist(raw);cache[kind]=raw
  x=np.arange(80.);delta=3.;base=encode(x,kind);expected=base.copy();p=periods(kind)[0]
  if kind=='scalar':expected+=delta/79
  elif kind=='ttw_phase':expected=(expected+delta/p)%1
  else:
   offset=1 if kind=='helix' else 0
   if offset:expected[:,0]+=delta/79
   for j,period in enumerate(periods(kind)):
    c,s=base[:,offset+2*j],base[:,offset+2*j+1];angle=2*np.pi*delta/period;expected[:,offset+2*j]=c*np.cos(angle)-s*np.sin(angle);expected[:,offset+2*j+1]=s*np.cos(angle)+c*np.sin(angle)
  physical_return=np.mean(np.linalg.norm(transform(encode(x+10,kind),st)-transform(base,st),axis=1))/scale
  rows.append(dict(encoding=kind,native_dimension=raw.shape[1],effective_train_rank=rank(z),n_grid=200,n_pairs=len(dist),exact_bitwise_collision_pairs=int(np.sum(rawdist==0)),near_collision_pairs=int(np.sum(dist<1e-6)),near_collision_fraction=float(np.mean(dist<1e-6)),min_scaled_distance=float(dist.min()),train_distance_scale=scale,intrinsic_shift_max_error=float(np.max(np.abs(encode(x+delta,kind)-expected))),period10_return_scaled=physical_return,unit_consistent_max_error=float(np.max(np.abs(encode(x*1000,kind,unit=1000)-base))),unit_blind_mean_error=float(np.mean((encode(x*1000,kind)-base)**2))))
  for dtype in ['float64','float32','bfloat16']:
   cast=lambda v: np.asarray(v,np.float64) if dtype=='float64' else (np.asarray(v,np.float32).astype(float) if dtype=='float32' else torch.as_tensor(np.asarray(v),dtype=torch.float64).bfloat16().float().numpy().astype(float))
   for magnitude in [0.,1.,100.,1e6]:
    for eps in [1.,.1,.01,.001,.0001,.000001]:
     v=np.array([magnitude,magnitude+eps]);xi=cast(v);e=cast(encode(xi,kind));precision.append(dict(encoding=kind,dtype=dtype,magnitude=magnitude,epsilon=eps,input_collided=bool(xi[0]==xi[1]),output_exact_collision=bool(np.array_equal(e[0],e[1])),native_l2=float(np.linalg.norm(e[1]-e[0])),input_actual_step=float(xi[1]-xi[0])))
 pd.DataFrame(rows).to_csv(out/'diagnostics.csv',index=False);pd.DataFrame(precision).to_csv(out/'precision.csv',index=False);np.savez(out/'grid.npz',values=grid,**cache);(out/'splits.json').write_text(json.dumps({k:v.tolist() for k,v in split_ids().items()},indent=2)+'\n');(out/'fone_status.json').write_text(json.dumps(dict(formal_comparison=False,reason='Official parser semantics not cleared in Phase0; fixed decimal_fourier_number is not full FoNE',new_downloads=False),indent=2)+'\n');print(pd.DataFrame(rows)[['encoding','native_dimension','effective_train_rank','near_collision_pairs','intrinsic_shift_max_error']].to_string(index=False))
if __name__=='__main__':main()
