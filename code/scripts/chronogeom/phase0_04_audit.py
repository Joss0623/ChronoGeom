import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,argparse,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.chronogeom.phase0_04_subspaces import spaces_for
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
cfg=json.loads((args.source/'config.json').read_text());screen=json.loads((args.source/'screening.json').read_text());pairs=[p for p in screen['cohorts']['fixed'] if p['selected']];fit=json.loads((Path(cfg['source_run'])/'fit_tokenization.json').read_text());d=np.load(args.source/'fit_activations.npz');paired=np.load(args.source/'fixed_paired_hidden.npz');arrays={};diags=[];donors=[]
for li,l in enumerate(cfg['layers']):
 y=d['hidden'][:,li,0].astype(float);spaces,diag,mean=spaces_for(d['a'],y,42+l*101);diag['layer']=l;diags.append(diag)
 for k,u in spaces.items():arrays[f'{l}_{k}']=u
 arrays[f'{l}_mean']=mean
 # Covariance diagnostic PCA16 only; no full precision matrix is estimated.
 _,s,v=np.linalg.svd(y-mean,full_matrices=False);u=v[:16];var=(s[:16]**2)/(len(y)-1);ridge=max(var[0]*1e-4,1e-12);arrays[f'{l}_diag_axes']=u;arrays[f'{l}_diag_var']=var+ridge
 refs=[]
 for i,p in enumerate(pairs):
  candidates=np.array([j for j,r in enumerate(fit) if r['b']==p['b']],dtype=int)
  if not len(candidates):refs.append(-1);donors.append(dict(layer=l,pair_id=p['id'],available=False,b=p['b'],reason='no same-b training activation'));continue
  base=paired['clean'][i,li,0].astype(float);delta=y[candidates]-base;h8=spaces['helix8'];bg=delta-(delta@h8.T)@h8;dist=np.linalg.norm(bg,axis=1);j=int(candidates[np.argmin(dist)]);refs.append(j);donors.append(dict(layer=l,pair_id=p['id'],available=True,fit_row=j,reference_a=fit[j]['a'],b=p['b'],n_candidates=len(candidates),background_l2=float(dist.min()),rule='same b, minimum complement-of-H8 L2; ties first original fit row; no answers/logits'))
 arrays[f'{l}_reference_indices']=np.array(refs)
np.savez(out/'subspaces.npz',**arrays);(out/'geometry.json').write_text(json.dumps(diags,indent=2)+'\n');(out/'donor_manifest.json').write_text(json.dumps(donors,indent=2)+'\n');(out/'input_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.source/'fit_activations.npz',args.source/'fixed_paired_hidden.npz',args.source/'screening.json',Path(cfg['source_run'])/'fit_tokenization.json']},indent=2)+'\n');print([(r['layer'],r['shared_rank'],r['unique_h_rank'],r['unique_f_rank'],r['linear_increment_rank']) for r in diags])
