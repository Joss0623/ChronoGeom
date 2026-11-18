"""Deterministic input archive + checkpoint replay; no training/test selection."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from data_provider.chronogeom_synthetic import *
from exp.exp_chronogeom_synthetic import CovariateReadout
from scripts.chronogeom.phase1_representations import *
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();r=args.root;out=r/'canonical_inputs';out.mkdir(exist_ok=False);torch.set_num_threads(2);checks=[]
for stage in ['development','heldout']:
 for family in FAMILIES:
  spec=task_spec(stage,family)
  for split in ['train','val' if stage=='development' else 'test']:
   times,values,meta=generate(spec,split);np.savez_compressed(out/f'{stage}_{family}_{split}.npz',times=times,values=values)
   if split!='train':assert meta==json.loads((r/f'track_t_{stage}'/f'{family}_instances.json').read_text())
  tr=generate(spec,'train')
  for kind in T_ENCODINGS:
   stats=fit_interface(encode(tr[0],kind,track='T',period=spec['period'])[:,:16],8);np.savez(out/f'{stage}_{family}_{kind}_interface.npz',**stats)
for kind in M_ENCODINGS:np.savez(out/f'm_{kind}_interface.npz',**fit_interface(encode(split_ids()['train'],kind),9))
for backbone in ['DLinear','TSMixer']:
 for kind in ['none','sincos']:
  spec=task_spec('heldout','periodic');ds,*_,mean,std=datasets(spec,kind,'test');model=CovariateReadout(backbone);model.load_state_dict(torch.load(r/'track_t_heldout/states'/f'periodic_{backbone}_{kind}_0.pt',weights_only=True));model.eval()
  with torch.inference_mode():p=model(ds[1].x,ds[1].x_mark,ds[1].y_mark).numpy()*std+mean
  saved=np.load(r/'track_t_heldout/predictions'/f'periodic_{backbone}_{kind}_0.npz')['prediction'];error=float(np.max(np.abs(p-saved)));assert error==0;checks.append(dict(backbone=backbone,encoding=kind,max_abs_error=error))
(r/'checkpoint_replay.json').write_text(json.dumps(dict(training_rerun=False,test_selection_changed=False,checks=checks,archive='Regenerated from immutable generator, seeds and task specs; exact saved-checkpoint predictions verified'),indent=2)+'\n');print('canonical inputs archived; exact checkpoint replay',len(checks))
