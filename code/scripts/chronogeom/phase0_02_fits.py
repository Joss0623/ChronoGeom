"""All-layer partial replication; no model loading, no test-driven layer selection."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(k,'2')
from pathlib import Path
import sys,json,csv,time,hashlib,subprocess,argparse
import numpy as np
from scipy.linalg import eigh
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.chronogeom.representation_probe import basis,metrics,split_values

def write_csv(p,rows):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def global_pca(y,k=100):
 mean=y.mean(0);c=y-mean
 vals,vecs=eigh(c@c.T,subset_by_index=(len(y)-k,len(y)-1))
 order=np.argsort(vals)[::-1];vals=vals[order];vecs=vecs[:,order]
 keep=vals>max(vals[0]*1e-12,1e-16)
 comp=(vecs[:,keep].T@c)/np.sqrt(vals[keep,None])
 return mean,comp

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 cfg=json.loads(args.config.read_text());args.output.mkdir(parents=True,exist_ok=False)
 import torch
 torch.set_num_threads(2);start=time.monotonic()
 d=torch.load(cfg['cache'],weights_only=True,map_location='cpu');a=d['a'].numpy().astype(float);b=d['b'].numpy();h=d['hidden_states']
 designs={'linear':basis(a,'linear'),'circle100':basis(a,'circle',periods=[100]),'fourier':basis(a,'fourier'),'helix':basis(a,'helix'),'poly8_scaled':basis(a,'polynomial',degree=8),'poly9_scaled':basis(a,'polynomial',degree=9),'poly9_raw':np.column_stack([a**i for i in range(1,10)]+[np.ones(len(a))]),'shuffled_helix':basis(a,'helix')}
 splits={};rows=[];spectrum=[];dtype_rows=[];pca_rows=[]
 for seed in cfg['seeds']:
  tr,te=split_values(a,seed,.8);splits[f'observed80_direct_s{seed}']=(tr,te)
  ids=np.random.RandomState(seed).choice(np.arange(100),75,replace=False);tr=np.isin(a,ids);splits[f'official75_globalPCA100_s{seed}']=(tr,~tr)
 splits['range80_direct_s0']=(a<80,a>=80)
 (args.output/'splits.json').write_text(json.dumps({k:{'train_rows':np.where(tr)[0].tolist(),'test_rows':np.where(te)[0].tolist(),'train_values':np.unique(a[tr]).tolist(),'test_values':np.unique(a[te]).tolist()} for k,(tr,te) in splits.items()},indent=2)+'\n')
 for l in range(h.shape[1]):
  y=h[:,l].float().numpy().astype(np.float64);mean,comp=global_pca(y);z=(y-mean)@comp.T
  for key,(tr,te) in splits.items():
   protocol,seedtext=key.rsplit('_s',1);seed=int(seedtext);target=z if 'globalPCA' in protocol else y
   for name,des in designs.items():
    x=des[tr].copy()
    if name=='shuffled_helix':x=x[np.random.default_rng(seed+1000).permutation(len(x))]
    coef,_,rank,sv=np.linalg.lstsq(x,target[tr],rcond=cfg['primary_rcond']);pred=des@coef
    if 'globalPCA' in protocol:pred=pred@comp+mean
    base=dict(layer=l,protocol=protocol,seed=seed,method=name,columns=des.shape[1],effective_rank=int(rank),condition=float(sv[0]/sv[-1]) if sv[-1]>0 else float('inf'))
    for group,mask in [('train',tr),('heldout',te),('test_a_0_49',te&(a<50)),('test_a_50_99',te&(a>=50))]+[(f'test_b_{i*25}_{i*25+24}',te&(b>=i*25)&(b<(i+1)*25)) for i in range(4)]:
     if mask.sum()<2: continue
     row=dict(base,group=group,n_rows=int(mask.sum()),n_values=int(len(np.unique(a[mask]))));row.update(metrics(y[mask],pred[mask]));ids=np.unique(a[mask]);yt=np.stack([y[mask&(a==i)].mean(0) for i in ids]);yp=np.stack([pred[mask&(a==i)].mean(0) for i in ids]);row['identity_r2']=metrics(yt,yp)['r2'];rows.append(row)
    if protocol=='observed80_direct' and seed==0 and name=='helix':
     cf=np.linalg.lstsq(x.astype(np.float32),target[tr].astype(np.float32),rcond=cfg['primary_rcond'])[0];pf=des.astype(np.float32)@cf
     dtype_rows.append(dict(layer=l,float64_r2=metrics(y[te],pred[te])['r2'],float32_r2=metrics(y[te],pf[te])['r2'],relative_prediction_l2=float(np.linalg.norm(pred-pf)/np.linalg.norm(pred))))
   # PCA sees heldout activation: separate diagnostic, no numerical-prediction ranking.
   if protocol=='official75_globalPCA100':
    for rank in [2,7,8,9]:
     pr=(y-mean)@comp[:rank].T@comp[:rank]+mean;pca_rows.append(dict(layer=l,seed=seed,rank=rank,**metrics(y[te],pr[te])))
  ids=np.unique(a);ym=np.stack([y[a==i].mean(0) for i in ids]);linear=basis(ids,'linear');res=ym-linear@np.linalg.lstsq(linear,ym,rcond=1e-10)[0]
  # Irregular-grid partial regression; missing45 is neither filled nor FFT-interpolated.
  for k in range(1,51):
   freq=k/100;xx=basis(ids,'fourier',periods=[1/freq]);xx-=linear@np.linalg.lstsq(linear,xx,rcond=1e-10)[0];fit=xx@np.linalg.lstsq(xx,res,rcond=1e-10)[0]
   spectrum.append(dict(layer=l,frequency=freq,period=1/freq,partial_r2=float(np.sum(fit**2)/np.sum(res**2)),effective_rank=int(np.linalg.matrix_rank(xx,tol=1e-10))))
  print('completed layer',l,flush=True)
 write_csv(args.output/'fits.csv',rows);write_csv(args.output/'fourier_partial.csv',spectrum);write_csv(args.output/'dtype_sensitivity.csv',dtype_rows);write_csv(args.output/'pca_diagnostics.csv',pca_rows)
 meta={'config':cfg,'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cache_sha256':hashlib.sha256(Path(cfg['cache']).read_bytes()).hexdigest(),'seconds':time.monotonic()-start,'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'global_pca':'all500 rows; Gram eigenvectors; threshold relative1e-12; n<=100','fit_rows':len(rows),'no_context_variants':'b strata only, not alternate prompts'}
 (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
