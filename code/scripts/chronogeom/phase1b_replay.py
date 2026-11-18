"""Saved-checkpoint/data replay, never retrains or changes selection."""
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from data_provider.chronogeom_identifiability import generate
from exp.exp_chronogeom_identifiability import predict_readout
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root;torch.set_num_threads(2);rows=[]
for task,n,kind,dec,seed,cohort in [('A',256,'none','linear',-1,'iid'),('B',256,'sincos','linear',-1,'period'),('B',64,'raw_timestamp','mlp',0,'iid'),('C',256,'sincos','mlp',1,'iid'),('D',256,'sincos','linear',-1,'phase')]:
 name=f'{task}_{n}_{kind}_rms_{dec}_{seed}';fit=torch.load(r/'train/states'/(name+'.pt'),weights_only=False);p=generate(task,'test',64,cohort);saved=np.load(r/'test/data'/f'{task}_{cohort}.npz')
 for k in saved.files:np.testing.assert_array_equal(p[k],saved[k])
 pred=predict_readout(fit,p)['prediction'];target=np.load(r/'test/predictions'/f'{name}_{cohort}.npz')['prediction'];err=float(np.max(np.abs(pred-target)));assert err==0;rows.append(dict(checkpoint=name,cohort=cohort,max_abs_error=err))
(r/'checkpoint_replay.json').write_text(json.dumps(dict(training_rerun=False,checks=rows),indent=2)+'\n');print('Exact replay',len(rows))
