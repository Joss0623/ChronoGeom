"""Dependent-model, independent-trajectory analysis; never chooses test encodings."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,argparse,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from data_provider.chronogeom_identifiability import *

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();r=args.root;out=r/'analysis';out.mkdir(exist_ok=False);d=pd.read_csv(r/'test/metrics.csv');tr=pd.read_csv(r/'train/metrics.csv');lock=json.loads((r/'lock/locked_rules.json').read_text());assert json.loads((r/'test/metadata.json').read_text())['locked_rules_sha256']==hashlib.sha256((r/'lock/locked_rules.json').read_bytes()).hexdigest()
keys=['task','n_train','decoder','encoding','energy','cohort'];summary=d.groupby(keys,as_index=False).agg(mse=('mse','mean'),seed_sd=('mse','std'),mae=('mae','mean'),clean_mse=('clean_mse','mean'),parameters=('parameters','first'),rank=('effective_encoding_rank','first'),native_dimension=('native_dimension','first'),train_rms=('encoding_train_rms','first'),test_rms=('encoding_eval_rms','mean'),history_oracle_mse=('history_oracle_realized','first'),full_oracle_mse=('full_oracle_realized','first'),theory_history=('theoretical_history_risk','first'),theory_full=('theoretical_full_risk','first'));summary.to_csv(out/'summary.csv',index=False)
cache={};unit_error=0.
for k,g in d.groupby(keys):
 task,n,decoder,kind,energy,cohort=k;losses=[]
 for s in sorted(g.seed.unique()):
  path=r/'test/predictions'/f'{task}_{n}_{kind}_{energy}_{decoder}_{s}_{cohort}.npz';p=np.load(path);losses.append(p['loss'])
  if cohort=='units':
   iid=np.load(r/'test/predictions'/f'{task}_{n}_{kind}_{energy}_{decoder}_{s}_iid.npz');unit_error=max(unit_error,float(np.max(np.abs(p['prediction']-iid['prediction']))))
 cache[k]=np.mean(losses,axis=0)
ix=np.random.default_rng(1729).integers(64,size=(1000,64));cirows=[]
def add(k1,k2,contrast):
 a=cache[k1];b=cache[k2];delta=a-b;sample=delta[ix].mean(1);task,n,dec,enc,energy,cohort=k1;cirows.append(dict(task=task,n_train=n,decoder=dec,cohort=cohort,energy=energy,contrast=contrast,encoding=enc,n_trajectories=64,point=float(delta.mean()),low95=float(np.quantile(sample,.025)),high95=float(np.quantile(sample,.975))))
for task in TASKS:
 for n in [64,256]:
  for dec in ['linear','mlp']:
   for cohort in COHORTS:
    for energy in ['native','rms']:
     for kind in ENCODINGS[1:]:add((task,n,dec,kind,energy,cohort),(task,n,dec,'none',energy,cohort),'encoding_minus_none')
     add((task,n,dec,'sincos',energy,cohort),(task,n,dec,'raw_timestamp',energy,cohort),'sincos_minus_raw')
    for kind in ENCODINGS:add((task,n,dec,kind,'rms',cohort),(task,n,dec,kind,'native',cohort),'rms_minus_native')
pd.DataFrame(cirows).to_csv(out/'paired_trajectory_intervals.csv',index=False)
rule_rows=[]
for rule in lock['choices']:
 for cohort in COHORTS:
  q=summary[(summary.task==rule['task'])&(summary.n_train==rule['n_train'])&(summary.decoder==rule['decoder'])&(summary.cohort==cohort)&(summary.energy=='rms')].set_index('encoding');none=float(q.loc['none','mse'])
  for name in ['validation_only','oracle_gated']:
   enc=rule[name];mse=float(q.loc[enc,'mse']);rule_rows.append(dict(task=rule['task'],n_train=rule['n_train'],decoder=rule['decoder'],cohort=cohort,rule=name,chosen=enc,mse=mse,none_mse=none,difference_vs_none=mse-none,test_best_encoding=q.mse.idxmin(),test_best_encoding_mse=float(q.mse.min()),theory_information_gain=float(q.theory_history.iloc[0]-q.theory_full.iloc[0])))
pd.DataFrame(rule_rows).to_csv(out/'frozen_rule_evaluation.csv',index=False)
oracles=summary.groupby(['task','cohort'],as_index=False)[['history_oracle_mse','full_oracle_mse','theory_history','theory_full']].first();oracles.to_csv(out/'oracle_risks.csv',index=False)
# Numerically stable full design ranks (not the original absolute-threshold,
# first-horizon diagnostic retained in raw metrics). No forecast changes.
ranks=[]
for task in TASKS:
 for n in [64,256]:
  p=generate(task,'train',n)
  for kind in ENCODINGS:
   st=interface(p,kind)
   for energy in ['native','rms']:
    for dec in ['linear','mlp']:
     x,_,_=design(p,kind,st,energy,dec=='mlp');matrices=[x[:,h] for h in range(H)] if dec=='linear' else [x.reshape(-1,x.shape[-1])];rs=[];conds=[]
     for m in matrices:
      z=np.column_stack([m.astype(float),np.ones(len(m))]);s=np.linalg.svd(z,compute_uv=False);rank=int(np.sum(s>s[0]*1e-6));rs.append(rank);conds.append(float(s[0]/s[rank-1]))
     ranks.append(dict(task=task,n_train=n,encoding=kind,energy=energy,decoder=dec,input_dimension=x.shape[-1],rank_min=min(rs),rank_max=max(rs),positive_spectrum_condition_max=max(conds),relative_rank_threshold=1e-6))
pd.DataFrame(ranks).to_csv(out/'stable_input_design_ranks.csv',index=False)
qtrain=tr[(tr.cohort=='train')&(tr.energy=='rms')&(tr.encoding!='none')];error=float(np.max(np.abs(qtrain.encoding_eval_rms-1)));q=dict(train_metric_rows=len(tr),expected_train_rows=768,test_metric_rows=len(d),expected_test_rows=3456,all_finite=bool(np.isfinite(d[['mse','clean_mse','encoding_eval_rms']].to_numpy()).all()),matched_nominal_parameters=bool((d.groupby('decoder').parameters.nunique()==1).all()),actual_training_rms_error=error,unit_pair_max_prediction_error=unit_error,locked_rules_verified=True,paired_intervals=len(cirows),parent_phase1_rule_unchanged=hashlib.sha256(Path('docs/chronogeom/phase1/locked_rule.json').read_bytes()).hexdigest()==json.loads((r/'preflight.json').read_text())['parent_phase1_rule_sha256'],warning='Only64independent trajectories per cohort;32horizons and3fixed model seeds are not independent sample counts. Units are paired with iid. New-system is one new functional family, not four. No multiplicity correction or broad significance claims.')
q['gate']=len(tr)==768 and len(d)==3456 and q['all_finite'] and q['matched_nominal_parameters'] and error<1e-12 and unit_error<1e-5 and q['parent_phase1_rule_unchanged'];(out/'quality.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q),flush=True)
