import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,argparse,json,time,hashlib,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from data_provider.chronogeom_identifiability import *
from exp.exp_chronogeom_identifiability import fit_readout,predict_readout

def save_pack(p,pack):np.savez_compressed(p,**{k:v for k,v in pack.items() if isinstance(v,np.ndarray)})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['train','test'],required=True);ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root;out=root/args.stage;out.mkdir(exist_ok=False);torch.set_num_threads(2);start=time.monotonic();rows=[];curves=[];(out/'predictions').mkdir();(out/'data').mkdir()
 if args.stage=='train':(out/'states').mkdir()
 else:
  lock=root/'lock/locked_rules.json';assert lock.exists();assert lock.read_bytes()==subprocess.check_output(['git','show','HEAD:docs/chronogeom/phase1b/locked_rules.json']);locksha=hashlib.sha256(lock.read_bytes()).hexdigest()
 for task in TASKS:
  trainmax=generate(task,'train',256);val=generate(task,'val',64)
  if args.stage=='train':save_pack(out/'data'/f'{task}_train256.npz',trainmax);save_pack(out/'data'/f'{task}_val.npz',val)
  testsets={c:generate(task,'test',64,c) for c in COHORTS} if args.stage=='test' else {'train':trainmax,'val':val}
  if args.stage=='test':
   for c,p in testsets.items():save_pack(out/'data'/f'{task}_{c}.npz',p)
  for n in [64,256]:
   tr={k:(v[:n] if isinstance(v,np.ndarray) else v) for k,v in trainmax.items()}
   for kind in ENCODINGS:
    for energy in ['native','rms']:
     for decoder,seed in [('linear',-1)]+[('mlp',s) for s in [0,1,2]]:
      if time.monotonic()-start>285:raise TimeoutError('300s stage cap; partial outputs retained')
      name=f'{task}_{n}_{kind}_{energy}_{decoder}_{seed}';path=root/'train/states'/(name+'.pt')
      if args.stage=='train':
       fit=fit_readout(tr,kind,energy,decoder,seed);torch.save(fit,path)
       for point in fit['curve']:curves.append(dict(task=task,n_train=n,encoding=kind,energy=energy,decoder=decoder,seed=seed,**point))
       packs={'train':tr,'val':val}
      else:fit=torch.load(path,weights_only=False);packs=testsets
      for cohort,pack in packs.items():
       result=predict_readout(fit,pack);hist=pack['history_oracle']/pack['unit_value'];full=pack['full_oracle']/pack['unit_value'];truth=pack['y']/pack['unit_value'];rh=float(np.mean((truth-hist)**2));rf=float(np.mean((truth-full)**2));rows.append(dict(task=task,n_train=n,encoding=kind,energy=energy,decoder=decoder,seed=seed,cohort=cohort,n=len(truth),mse=float(result['loss'].mean()),mae=float(result['mae'].mean()),clean_mse=float(result['clean_loss'].mean()),raw_unit_mse=float(result['loss'].mean()*pack['unit_value']**2),history_oracle_realized=rh,full_oracle_realized=rf,theoretical_history_risk=pack['sigma']**2+pack['information_gap'],theoretical_full_risk=pack['sigma']**2,parameters=fit['parameters'],native_dimension=fit['stats']['native_dimension'],effective_encoding_rank=fit['stats']['rank'],encoding_train_rms=1. if energy=='rms' and fit['stats']['rms']>0 else fit['stats']['rms'],encoding_eval_rms=result['encoding_rms'],design_rank=result['design_rank']))
       np.savez_compressed(out/'predictions'/f'{name}_{cohort}.npz',prediction=result['prediction'],loss=result['loss'],clean_loss=result['clean_loss'])
      pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
      print(args.stage,name,round(rows[-1]['mse'],5),flush=True)
 pd.DataFrame(curves).to_csv(out/'learning_curves.csv',index=False);meta=dict(seconds=time.monotonic()-start,rows=len(rows),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),device='cpu',threads=2,stage=args.stage,seeds=[0,1,2],code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('data_provider/chronogeom_identifiability.py'),Path('exp/exp_chronogeom_identifiability.py')]})
 if args.stage=='test':meta['locked_rules_sha256']=locksha
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print('COMPLETE',args.stage,len(rows),time.monotonic()-start,flush=True)
if __name__=='__main__':main()
