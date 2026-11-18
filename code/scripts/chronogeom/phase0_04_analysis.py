"""CPU-only dependent-pair uncertainty and full diagnostics; no model calls."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,argparse,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scripts.chronogeom.phase0_03_geometry import identity_weights,interval

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();run=args.run;out=args.output;out.mkdir(exist_ok=False);cfg=json.loads((run/'config.json').read_text());source=Path(cfg['source']);screen=json.loads((source/'screening.json').read_text());pairs=[p for p in screen['cohorts']['fixed'] if p['selected']];raw=pd.read_json(run/'interventions.jsonl',lines=True);df=raw[raw.control_seed==42];keys=['layer','operation','method','rank','mode','control_seed'];summ=[]
 for key,g in raw.groupby(keys,sort=True):
  row=dict(zip(keys,key));row.update(n=len(g),effect_mean=g.effect.mean(),effect_sd=g.effect.std(ddof=1),accuracy=g.target_correct.mean(),baseline_accuracy=g.baseline_clean_correct.mean(),ld_before=g.baseline_ld.mean(),ld_after=g.patched_ld.mean(),prediction_changed=g.prediction_changed.mean(),l2_mean=g.patch_l2.mean(),l2_median=g.patch_l2.median(),rounding_mean=g.rounding_l2.mean(),h7_recon_mse=g.helix7_reconstruction_mse.mean(),f7_recon_mse=g.fourier7_reconstruction_mse.mean(),p7_recon_mse=g.pca7_reconstruction_mse.mean(),score_m2=g.train_pca16_regularized_score_m2.mean(),pca16_off_mse=g.train_pca16_complement_mse.mean(),nn_same_b_l2=g.nearest_same_b_train_l2.mean(),nn_ratio_median=g.nearest_distance_ratio.median(),max_norm_error=g.relative_norm_error.max());summ.append(row)
 summary=pd.DataFrame(summ);summary.to_csv(out/'condition_summary.csv',index=False)
 wcache={};cirows=[];effectrows=[];groupcounts=[]
 comparisons={('swap','raw'):[('helix7','fourier7'),('helix7','pca7'),('shared','helix7'),('shared','fourier7'),('unique_h','unique_f'),('linear_increment','unique_f'),('linear_increment','pca1'),('linear_increment','random1')],('swap','equal_linear_increment'):[('linear_increment',m) for m in ['unique_h','unique_f','pca1','random1']],('conditional','raw'):[('shared','pca6'),('shared','random6'),('unique_h','unique_f'),('linear_increment','unique_f'),('linear_increment','pca1'),('linear_increment','random1'),('helix7','fourier7'),('helix7','pca7')],('rescue','raw'):[('helix7','fourier7'),('helix7','pca7'),('helix7','random7'),('shared','helix7'),('shared','fourier7'),('unique_h','unique_f'),('linear_increment','unique_f')]}
 for (l,op,mode),part in df.groupby(['layer','operation','mode']):
  ids=sorted(part.pair_id.unique());pp=[p for p in pairs if p['id'] in ids];pp=sorted(pp,key=lambda p:p['id']);assert [p['id'] for p in pp]==ids
  for role in ['all','carry_yes','carry_no']:
   sub=pp if role=='all' else [p for p in pp if ((p['clean_a']%10+p['b']%10>=10)==(role=='carry_yes'))];groupcounts.append(dict(layer=l,operation=op,mode=mode,group=role,n=len(sub),clean_ids=len({p['clean_a'] for p in sub}),corrupt_ids=len({p['corrupt_a'] for p in sub}),b_ids=len({p['b'] for p in sub}),union_ids=len({v for p in sub for v in [p['clean_a'],p['corrupt_a'],p['b']]})))
  for bootmode in ['shared_nodes','clean','corrupt','b']:
   for seed in ([1729,1730,1731] if bootmode=='shared_nodes' else [1729]):
    cache=(tuple(ids),bootmode,seed)
    if cache not in wcache:wcache[cache]=identity_weights(pp,2000,seed,bootmode)
    weights=wcache[cache]
    for ma,mb in comparisons.get((op,mode),[]):
     ga=part[part.method==ma].set_index('pair_id').reindex(ids);gb=part[part.method==mb].set_index('pair_id').reindex(ids)
     if ga.effect.isna().any() or gb.effect.isna().any():continue
     effect=interval((ga.effect-gb.effect).to_numpy(),weights);acc=interval((ga.target_correct.astype(float)-gb.target_correct.astype(float)).to_numpy(),weights)
     cirows.append(dict(layer=l,operation=op,mode=mode,comparison=ma+'_minus_'+mb,rank_a=int(ga['rank'].iloc[0]),rank_b=int(gb['rank'].iloc[0]),n=len(ids),bootstrap=bootmode,seed=seed,accuracy_difference=acc['point'],accuracy_low=acc['low'],accuracy_high=acc['high'],**effect))
    if bootmode=='shared_nodes' and seed==1729:
     for method,g in part.groupby('method'):
      g=g.set_index('pair_id').reindex(ids);effect=interval(g.effect.to_numpy(),weights);acc=interval((g.target_correct.astype(float)-g.baseline_clean_correct.astype(float)).to_numpy(),weights);effectrows.append(dict(layer=l,operation=op,mode=mode,method=method,n=len(ids),accuracy_change=acc['point'],accuracy_low=acc['low'],accuracy_high=acc['high'],**effect))
 pd.DataFrame(cirows).to_csv(out/'paired_intervals.csv',index=False);pd.DataFrame(effectrows).to_csv(out/'method_intervals.csv',index=False);pd.DataFrame(groupcounts).to_csv(out/'group_counts.csv',index=False)
 mainnorm=raw[(raw['mode']=='equal_linear_increment')&(raw.requested_l2>0)];noop=raw[raw.method=='noop'];full0=raw[(raw.operation=='swap')&(raw.method=='full')&(raw.layer==0)];fullres=raw[(raw.operation=='rescue')&(raw.method=='full')];meta=json.loads((run/'metadata.json').read_text());expected=6*(17*85+26*82)+4*(85+82+82)
 quality=dict(rows=len(raw),expected_rows=expected,rows_complete=len(raw)==expected,all_finite=bool(np.isfinite(raw[['baseline_ld','patched_ld','effect','patch_l2','train_pca16_regularized_score_m2','helix7_reconstruction_mse']].to_numpy()).all()),noop_max_logit_change=float(noop.max_logit_change.max()),full0_swap_accuracy=float(full0.target_correct.mean()),full_rescue_accuracy=float(fullres.target_correct.mean()),cache_max_error=meta['cache_max_error'],subset_batchshape_hidden_max_difference=meta['subset_batchshape_hidden_max_difference'],positive_norm_targets=len(mainnorm),max_norm_relative_error=float(mainnorm.relative_norm_error.max()),p95_norm_relative_error=float(mainnorm.relative_norm_error.quantile(.95)),zero_direction_positive_target=int(mainnorm.zero_direction.sum()),intervals=len(cirows),interpretation='Exploratory dependent-identity percentile intervals conditional on one model, fixed fit and selected prompt graph. No finite-sample coverage guarantee; no multiplicity adjustment.')
 quality['gate']=bool(quality['rows_complete'] and quality['all_finite'] and quality['noop_max_logit_change']==0 and quality['full0_swap_accuracy']==1 and quality['full_rescue_accuracy']==1 and meta['baseline_drift']==0)
 (out/'quality.json').write_text(json.dumps(quality,indent=2)+'\n');(out/'provenance.json').write_text(json.dumps(dict(analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),raw_sha256=hashlib.sha256((run/'interventions.jsonl').read_bytes()).hexdigest()),indent=2)+'\n');print(json.dumps(quality),flush=True)
if __name__=='__main__':main()
