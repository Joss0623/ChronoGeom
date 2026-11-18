"""Read-only Phase1 audit. OLS references are not Bayes-risk proofs."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,argparse,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scripts.chronogeom.phase1_representations import T_ENCODINGS,encode,fit_interface,transform,rank
from data_provider.chronogeom_synthetic import task_spec
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output;out.mkdir(exist_ok=False);r=args.source;repo=Path(__file__).resolve().parents[2];review={};rows=[];refs=[]
for p in list((repo/'docs/chronogeom').glob('PHASE1_*.md'))+[r/'rule_lock/locked_rule.json',r/'rule_lock/heldout_predictions_before_eval.csv',r/'track_m/metrics.csv',r/'track_t_development/metrics.csv',r/'track_t_heldout/metrics.csv']:
 b=p.read_bytes();review[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
for stage in ['development','heldout']:
 for family in ['trend','periodic','mixed','regime']:
  tr=np.load(r/'canonical_inputs'/f'{stage}_{family}_train.npz');spec=task_spec(stage,family)
  for e in T_ENCODINGS:
   stats=fit_interface(encode(tr['times'],e,'T',spec['period'])[:,:16],8);z=transform(encode(tr['times'],e,'T',spec['period']),stats);rows.append(dict(stage=stage,family=family,encoding=e,nominal_dimension=8,native_dimension=stats['mean'].size,effective_rank=rank(z[:,:16]),train_past_vector_rms=float(np.sqrt(np.mean(np.sum(z[:,:16]**2,axis=-1)))),train_future_vector_rms=float(np.sqrt(np.mean(np.sum(z[:,16:]**2,axis=-1))))))
for family in ['trend','periodic','mixed','regime']:
 d=np.load(r/'canonical_inputs'/f'heldout_{family}_test.npz');spec=task_spec('heldout',family);p=spec['period']/2 if family=='regime' else spec['period'];h=np.arange(24);phase=2*np.pi*h/p
 base=np.column_stack([np.ones(24),h]) if family=='trend' else (np.column_stack([np.ones(24),np.sin(phase),np.cos(phase)]) if family=='periodic' else np.column_stack([np.ones(24),h,np.sin(phase),np.cos(phase)]))
 for info in ['history_only_law_OLS','clock_informed_law_OLS']:
  losses=[]
  for time,y in zip(d['times'],d['values']):
   design=base
   if info.startswith('clock') and family!='trend':
    ph=spec['phase']+2*np.pi*time/spec['period'] if family!='regime' else spec['phase']+2*np.pi*(np.minimum(time,spec['switch'])/spec['period']+np.maximum(time-spec['switch'],0)/(spec['period']/2))
    design=np.column_stack([np.ones(24),np.sin(ph)]) if family=='periodic' else np.column_stack([np.ones(24),h,np.sin(ph)])
   coef=np.linalg.lstsq(design[:16],y[:16],rcond=1e-10)[0];losses.append(np.mean((design[16:]@coef-y[16:])**2))
  refs.append(dict(family=family,reference=info,n=64,raw_mse=float(np.mean(losses)),label='law-informed local OLS; not exact Bayes'))
pd.DataFrame(rows).to_csv(out/'phase1_actual_input_energy.csv',index=False);pd.DataFrame(refs).to_csv(out/'phase1_law_ols_references.csv',index=False);(out/'read_manifest.json').write_text(json.dumps(review,indent=2)+'\n');print(pd.DataFrame(refs).to_string(index=False))
