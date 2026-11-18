"""Post-cast input RMS audit only; no prediction, fit or selection changes."""
from pathlib import Path
import argparse,sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from data_provider.chronogeom_identifiability import *
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();r=args.root;rows=[]
for task in TASKS:
 for n in [64,256]:
  p=generate(task,'train',n)
  for kind in ENCODINGS:
   st=interface(p,kind)
   for energy in ['native','rms']:
    f,_,e=design(p,kind,st,energy);actual=float(np.sqrt(np.mean(np.sum(f[...,-8:].astype(float)**2,axis=-1))));rows.append(dict(task=task,n_train=n,encoding=kind,energy=energy,precast_rms=float(np.sqrt(np.mean(np.sum(e**2,axis=-1)))),actual_float32_input_rms=actual))
d=pd.DataFrame(rows);d.to_csv(r/'analysis/actual_input_energy.csv',index=False);sub=d[(d.energy=='rms')&(d.encoding!='none')];err=float(np.max(np.abs(sub.actual_float32_input_rms-1)));assert err<1e-6;(r/'analysis/input_precision_note.json').write_text(json.dumps(dict(actual_float32_rms_error=err,original_metrics_rms_scope='FP64 interface before final FP32 feature cast',changes_to_training_predictions_or_selection=False),indent=2)+'\n');print(err)
