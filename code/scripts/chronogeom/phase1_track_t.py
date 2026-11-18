import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,sys,json,time,subprocess,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from data_provider.chronogeom_synthetic import FAMILIES,task_spec
from exp.exp_chronogeom_synthetic import Exp_ChronoGeomSynthetic
from scripts.chronogeom.phase1_representations import T_ENCODINGS,SEEDS

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['development','heldout'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--locked-rule',type=Path);args=ap.parse_args();out=args.output;out.mkdir(exist_ok=False);torch.set_num_threads(2);start=time.monotonic();rows=[];curves=[];preds=out/'predictions';preds.mkdir();states=out/'states';states.mkdir()
 if args.stage=='heldout':assert args.locked_rule and args.locked_rule.exists(),'Freeze validation rule before heldout';locked_sha=hashlib.sha256(args.locked_rule.read_bytes()).hexdigest()
 else:locked_sha=None
 for family in FAMILIES:
  spec=task_spec(args.stage,family)
  for backbone in ['DLinear','TSMixer']:
   for encoding in T_ENCODINGS:
    for seed in SEEDS:
     if time.monotonic()-start>285:raise TimeoutError('TSLib stage300s cap; completed records retained')
     fit=Exp_ChronoGeomSynthetic(spec,encoding,backbone,seed,'val' if args.stage=='development' else 'test').run();name=f'{family}_{backbone}_{encoding}_{seed}';np.savez_compressed(preds/(name+'.npz'),prediction=fit['prediction'],target=fit['target'],loss=fit['loss'],mae=fit['mae']);torch.save(fit['state'],states/(name+'.pt'))
     rows.append(dict(stage=args.stage,family=family,backbone=backbone,encoding=encoding,seed=seed,n=len(fit['loss']),mse=float(fit['loss'].mean()),mae_original_units=float(fit['mae'].mean()),train_mse=fit['train_mse'],parameters=fit['parameters'],alignment_nmse=fit['diagnostic'][0],effective_rank=fit['diagnostic'][1],native_dimension=fit['diagnostic'][2],value_std=fit['value_std']))
     for c in fit['curve']:curves.append(dict(family=family,backbone=backbone,encoding=encoding,seed=seed,**c))
     pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);(out/f'{family}_instances.json').write_text(json.dumps(fit['instance_metadata'],indent=2)+'\n');print(args.stage,name,round(rows[-1]['mse'],5),flush=True)
 pd.DataFrame(curves).to_csv(out/'learning_curves.csv',index=False);(out/'metadata.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),stage=args.stage,seeds=SEEDS,threads=2,locked_rule_sha256=locked_sha,backbones_unchanged=True,task_specs=[task_spec(args.stage,f) for f in FAMILIES],torch=torch.__version__),indent=2)+'\n');print('COMPLETE',args.stage,len(rows),time.monotonic()-start,flush=True)
if __name__=='__main__':main()
