"""Final CPU analysis: dependent M identities, independent T trajectories, frozen rule."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scripts.chronogeom.phase0_03_geometry import identity_weights,interval

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();r=args.root;out=args.output;out.mkdir(exist_ok=False)
 m=pd.read_csv(r/'track_m/metrics.csv');t=pd.read_csv(r/'track_t_heldout/metrics.csv');d=pd.read_csv(r/'track_t_development/metrics.csv');locked=json.loads((r/'rule_lock/locked_rule.json').read_text());pre=pd.read_csv(r/'rule_lock/heldout_predictions_before_eval.csv');assert hashlib.sha256((r/'rule_lock/heldout_predictions_before_eval.csv').read_bytes()).hexdigest()==locked['predictions_sha256'];assert hashlib.sha256((r/'rule_lock/locked_rule.json').read_bytes()).hexdigest()==json.loads((r/'track_t_heldout/metadata.json').read_text())['locked_rule_sha256']
 summary_cols=['mse','accuracy','mae_original_units','phase_mae_radians'];m.groupby(['task','encoding','readout','split'])[summary_cols].agg(['mean','std']).to_csv(out/'m_summary.csv');t.groupby(['family','backbone','encoding'])[['mse','mae_original_units','train_mse']].agg(['mean','std']).to_csv(out/'t_summary.csv')
 mcis=[];wcache={}
 for (task,readout,split),group in m[m.split.isin(['interpolation','extrapolation'])].groupby(['task','readout','split']):
  seeds=sorted(group.seed.unique());basefiles=[np.load(r/'track_m/predictions'/f'{task}_scalar_{readout}_{seed}_{split}.npz') for seed in seeds];base=np.mean([v['loss'] for v in basefiles],axis=0);a,b=basefiles[0]['a'],basefiles[0]['b'];key=(task in ['order','addition','subtraction'],split)
  if key not in wcache:wcache[key]=identity_weights([dict(clean_a=int(aa),corrupt_a=int(bb),b=int(bb)) for aa,bb in zip(a,b)],1000,1729)
  for kind in group.encoding.unique():
   if kind=='scalar':continue
   files=[np.load(r/'track_m/predictions'/f'{task}_{kind}_{readout}_{seed}_{split}.npz') for seed in seeds];delta=np.mean([v['loss'] for v in files],axis=0)-base;ci=interval(delta,wcache[key]);mcis.append(dict(task=task,readout=readout,split=split,comparison=kind+'_minus_scalar',n_cases=len(a),n_identities=len(set(a)|set(b)),metric='target-vector MSE difference',**ci))
 pd.DataFrame(mcis).to_csv(out/'m_identity_intervals.csv',index=False)
 tcis=[];rule_eval=[]
 for (family,backbone),group in t.groupby(['family','backbone']):
  seeds=sorted(group.seed.unique());none=np.mean([np.load(r/'track_t_heldout/predictions'/f'{family}_{backbone}_none_{s}.npz')['loss'] for s in seeds],axis=0);indices=np.random.default_rng(1729).integers(len(none),size=(1000,len(none)))
  for e in group.encoding.unique():
   if e=='none':continue
   loss=np.mean([np.load(r/'track_t_heldout/predictions'/f'{family}_{backbone}_{e}_{s}.npz')['loss'] for s in seeds],axis=0);delta=loss-none;sample=delta[indices].mean(1);tcis.append(dict(family=family,backbone=backbone,encoding=e,n_trajectories=len(none),n_model_seeds=len(seeds),mean_mse_difference=float(delta.mean()),low95=float(np.quantile(sample,.025)),high95=float(np.quantile(sample,.975))))
  truth=group.groupby('encoding').mse.mean();pg=pre[(pre.family==family)&(pre.backbone==backbone)].set_index('encoding').reindex(truth.index);chosen=pg.index[pg.chosen][0];best=truth.idxmin();devmeans=d[d.backbone==backbone].groupby(['family','encoding'],as_index=False).mse.mean().groupby('encoding').mse.apply(lambda v:np.mean(np.log(v)));constant=devmeans.idxmin();pred=pg.predicted_log_mse.to_numpy();actual=np.log(truth.to_numpy());pairs=[np.sign(pred[i]-pred[j])*np.sign(actual[i]-actual[j]) for i in range(len(pred)) for j in range(i)];pairacc=float(np.mean([(1 if x>0 else (.5 if x==0 else 0)) for x in pairs]));rule_eval.append(dict(family=family,backbone=backbone,locked_rule=locked['rules'][backbone]['type'],chosen=chosen,constant_validation_choice=constant,test_best=best,spearman=float(spearmanr(pred,actual).statistic),pairwise_ranking_accuracy=pairacc,chosen_mse=float(truth[chosen]),none_mse=float(truth['none']),constant_mse=float(truth[constant]),oracle_mse=float(truth[best]),regret_over_oracle=float(truth[chosen]/truth[best]-1),relative_to_none=float(truth[chosen]/truth['none']-1)))
 pd.DataFrame(tcis).to_csv(out/'t_trajectory_intervals.csv',index=False);pd.DataFrame(rule_eval).to_csv(out/'heldout_rule_evaluation.csv',index=False)
 # All selected-run shapes and parameter counts are checked, not only winners.
 param_m=m.groupby(['task','readout']).parameters.nunique();param_t=t.groupby('backbone').parameters.nunique();quality=dict(m_metric_rows=len(m),m_expected_rows=576,t_development_rows=len(d),t_heldout_rows=len(t),t_expected_rows_per_stage=216,m_parameters_equal=bool((param_m==1).all()),t_parameters_equal=bool((param_t==1).all()),all_metrics_finite=bool(np.isfinite(m.mse).all() and np.isfinite(t.mse).all() and np.isfinite(d.mse).all()),frozen_rule_sha_verified=True,m_intervals=len(mcis),t_intervals=len(tcis),uncertainty='M shared-operand graph percentile bootstrap exploratory; T resamples64independent trajectories after averaging the same3model seeds. No IID horizons or independent-replicate claims from repeated seeds.')
 quality['gate']=quality['m_metric_rows']==576 and len(t)==len(d)==216 and quality['m_parameters_equal'] and quality['t_parameters_equal'] and quality['all_metrics_finite'];(out/'quality.json').write_text(json.dumps(quality,indent=2)+'\n');print(json.dumps(quality),flush=True);print(pd.DataFrame(rule_eval).round(4).to_string(index=False))
if __name__=='__main__':main()
