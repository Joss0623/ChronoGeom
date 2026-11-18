"""Freeze validation-only and explicitly privileged Oracle-gated selections."""
from pathlib import Path
import argparse,json,hashlib
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();r=args.root;out=r/'lock';out.mkdir(exist_ok=False);d=pd.read_csv(r/'train/metrics.csv');q=d[(d.cohort=='val')&(d.energy=='rms')].groupby(['task','n_train','decoder','encoding'],as_index=False).mse.mean();rows=[]
for (task,n,decoder),g in q.groupby(['task','n_train','decoder']):
 best=g.loc[g.mse.idxmin()];time=g[g.encoding!='none'];bt=time.loc[time.mse.idxmin()];rows.append(dict(task=task,n_train=int(n),decoder=decoder,validation_only=best.encoding,oracle_gated=bt.encoding if task=='B' else 'none',training_oracle_gap=.5 if task=='B' else 0.,best_validation_mse=float(best.mse)))
q.to_csv(out/'validation_candidates.csv',index=False);lock=dict(choices=rows,primary_energy='rms',test_outcomes_used=False,oracle_gated_is_privileged=True,validation_sha256=hashlib.sha256((r/'train/metrics.csv').read_bytes()).hexdigest(),rule='minimum seed-mean validation MSE; Oracle gate knows training-law risk gap; no test adaptation');(out/'locked_rules.json').write_text(json.dumps(lock,indent=2)+'\n');print(pd.DataFrame(rows).to_string(index=False))
