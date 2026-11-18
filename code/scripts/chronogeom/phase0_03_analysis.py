"""Paired dependence-aware exploratory intervals; no IID row-based significance."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.chronogeom.phase0_03_geometry import identity_weights,interval

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((args.run/'config.json').read_text());screen=json.loads((args.run/'screening.json').read_text());df=pd.read_json(args.run/'interventions.jsonl',lines=True)
    keys=['cohort','layer','site','method','rank','operation','mode','control_seed','fit_profile']
    summary=df.groupby(keys,dropna=False).agg(n=('effect','size'),effect_mean=('effect','mean'),effect_sd=('effect','std'),accuracy=('target_correct','mean'),baseline_accuracy=('baseline_correct','mean'),ld_before=('baseline_ld','mean'),ld_after=('patched_ld','mean'),prediction_changed=('prediction_changed','mean'),norm_mean=('patch_l2','mean'),rounding_mean=('rounding_l2','mean'),max_norm_error=('relative_norm_error','max'),reconstruction_mse=('test_reconstruction_mse','first'),number_prediction_mse=('number_to_hidden_mse','first')).reset_index();summary.to_csv(out/'condition_summary.csv',index=False,lineterminator='\n')
    counts=[];subgroups=[]
    for cohort,pairs in screen['cohorts'].items():
        groups={'all':lambda p:True,'carry_clean_no':lambda p:p['clean_a']%10+p['b']%10<10,'carry_clean_yes':lambda p:p['clean_a']%10+p['b']%10>=10}
        for k in range(4):
            groups[f'clean_bin_{25*k}_{25*k+24}']=lambda p,k=k:25*k<=p['clean_a']<25*(k+1)
            groups[f'corrupt_bin_{25*k}_{25*k+24}']=lambda p,k=k:25*k<=p['corrupt_a']<25*(k+1)
        for c in [False,True]:
            for r in [False,True]:groups[f'carry_transition_{int(c)}{int(r)}']=lambda p,c=c,r=r:((p['clean_a']%10+p['b']%10>=10)==c and (p['corrupt_a']%10+p['b']%10>=10)==r)
        for trained in [False,True]:groups['clean_identity_'+('trained' if trained else 'heldout')]=lambda p,t=trained:(p['clean_a'] in cfg['fit_values'])==t
        for high in [False,True]:groups['clean_magnitude_'+('80_99' if high else '0_79')]=lambda p,h=high:(p['clean_a']>=80)==h
        for name,predicate in groups.items():
            candidates=[p for p in pairs if predicate(p)];selected=[p for p in candidates if p['selected']];ids={p['id'] for p in selected}
            counts.append(dict(cohort=cohort,group=name,candidates=len(candidates),selected=len(selected),current_joint_correct=sum(p['current_joint_correct'] for p in candidates),clean_identities=len({p['clean_a'] for p in selected}),corrupt_identities=len({p['corrupt_a'] for p in selected}),b_identities=len({p['b'] for p in selected}),operand_union=len({v for p in selected for v in [p['clean_a'],p['corrupt_a'],p['b']]})))
            if not ids:continue
            part=df[(df.cohort==cohort)&df.pair_id.isin(ids)&(df.control_seed==42)&((df['rank']==7)|df.method.isin(['full','noop','linear']))]
            g=part.groupby(['layer','site','method','rank','operation','mode'],dropna=False).agg(n=('effect','size'),effect_mean=('effect','mean'),accuracy=('target_correct','mean'),norm_mean=('patch_l2','mean')).reset_index();g.insert(0,'group',name);g.insert(0,'cohort',cohort);subgroups.append(g)
    pd.DataFrame(counts).to_csv(out/'group_counts.csv',index=False,lineterminator='\n');pd.concat(subgroups,ignore_index=True).to_csv(out/'group_effects.csv',index=False,lineterminator='\n')
    caches={};ci=[]
    def select(cohort,l,method,op,mode):
        return df[(df.cohort==cohort)&(df.layer==l)&(df.site=='a')&(df.method==method)&(df['rank']==7)&(df.operation==op)&(df['mode']==mode)&(df.control_seed==42)]
    for cohort,pairs_all in screen['cohorts'].items():
        pairs=[p for p in pairs_all if p['selected']];n=len(pairs)
        if not n:continue
        pair_ids=[p['id'] for p in pairs];minclusters=min(len({p['clean_a'] for p in pairs}),len({p['corrupt_a'] for p in pairs}));tested_layers=cfg['layers'] if cohort=='fixed' else cfg['rank_sensitivity_layers']
        for l in tested_layers:
            for op in ['swap','ablate_mean']:
                for mode in ['raw','equal_helix']:
                    refmode=mode if op=='swap' else 'raw';h=select(cohort,l,'helix',op,refmode)
                    if len(h)!=n:continue
                    h=h.set_index('pair_id').loc[pair_ids]
                    for method in ['pca','fourier','polynomial','random','shuffled']:
                        b=select(cohort,l,method,op,mode)
                        if len(b)!=n:continue
                        b=b.set_index('pair_id').loc[pair_ids];delta=(h.effect-b.effect).to_numpy();acc=(h.target_correct.astype(float)-b.target_correct.astype(float)).to_numpy()
                        if op=='ablate_mean':acc=-acc
                        modes=['shared_nodes','clean','corrupt','b']
                        for bootmode in modes:
                            seeds=cfg['bootstrap_seeds'] if bootmode=='shared_nodes' and cohort=='fixed' else cfg['bootstrap_seeds'][:1]
                            for seed in seeds:
                                key=(cohort,bootmode,seed)
                                if key not in caches:caches[key]=identity_weights(pairs,cfg['bootstrap_draws'],seed,bootmode)
                                effect=interval(delta,caches[key]);accuracy=interval(acc,caches[key])
                                ci.append(dict(cohort=cohort,layer=l,site='a',operation=op,mode=mode,comparison='helix_minus_'+method,bootstrap=bootmode,bootstrap_seed=seed,n_pairs=n,min_a_clusters=minclusters,insufficient=n<10 or minclusters<5,effect_point=effect['point'],low95=effect['low'],high95=effect['high'],accuracy_difference=accuracy['point'],accuracy_low95=accuracy['low'],accuracy_high95=accuracy['high'],valid_draws=effect['valid_draws'],zero_weight_draws=effect['zero_weight_draws']))
    pd.DataFrame(ci).to_csv(out/'paired_intervals.csv',index=False,lineterminator='\n')
    selected=screen['cohorts']['fixed'];selected=[p for p in selected if p['selected']];all_noop=df[df.method=='noop'];full0=df[(df.cohort=='fixed')&(df.layer==0)&(df.site=='a')&(df.method=='full')&(df.operation=='swap')]
    norm=df[(df['mode']=='equal_helix')&(df.requested_l2>1e-8)]
    quality={'rows':len(df),'fixed_n':len(selected),'fixed_selection_unchanged':screen['fixed_selection_unchanged'],'fixed_baseline_drift':screen['fixed_baseline_drift'],'noop_max':float(all_noop.max_logit_change.max()),'full0_restoration':float(full0.target_correct.mean()),'all_finite':bool(np.isfinite(df.ld_change).all()),'norm_rows_positive_target':len(norm),'norm_relative_error_median':float(norm.relative_norm_error.median()),'norm_relative_error_p95':float(norm.relative_norm_error.quantile(.95)),'norm_relative_error_max':float(norm.relative_norm_error.max()),'norm_error_over1pct_fraction':float((norm.relative_norm_error>.01).mean()),'unmatchable_nonzero_count':int(((norm.zero_direction)&(norm.requested_l2>0)).sum()),'bootstrap_intervals':len(ci),'interval_warning':'Exploratory percentile resampling conditional on fixed model/selected prompt graph; shared-node extension does not have guaranteed finite-sample95%coverage; no multiple-testing correction or significance claims.'}
    quality['execution_gate']=quality['noop_max']<=1e-5 and quality['full0_restoration']==1 and quality['fixed_baseline_drift']==0 and quality['all_finite'];(out/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    summary[summary.control_seed.isin([42,43,44])&summary.method.isin(['random','shuffled'])].to_csv(out/'control_seed_sensitivity.csv',index=False,lineterminator='\n')
    with plt.rc_context({'font.size':9,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained');methods=['helix','pca','fourier','polynomial','random','shuffled'];colors=['#D55E00','#0072B2','#009E73','#000000','#777777','#CC79A7'];markers=['D','s','^','o','x','v']
        for method,color,marker in zip(methods,colors,markers):
            for ax,op,mode,title in zip(axes,['swap','swap','ablate_mean'],['raw','equal_helix','raw'],['Rank7 swaps','Rank7 swaps, Helix-matched L2','Rank7 centered-coordinate ablation']):
                part=summary[(summary.cohort=='fixed')&(summary.site=='a')&(summary.method==method)&(summary['rank']==7)&(summary.operation==op)&(summary['mode']==mode)&(summary.control_seed==42)].sort_values('layer')
                ax.plot(part.layer,part.effect_mean,marker=marker,color=color,label=method,linestyle='--',markersize=4);ax.set(title=title,xlabel='Block-input index',ylabel='Mean LD gain' if op=='swap' else 'Mean LD loss');ax.set_xticks(cfg['layers']);ax.axhline(0,color='.6',linewidth=.5);ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);fig.suptitle('Same frozen85pair cohort; all six preset indices; no independent-model uncertainty implied');fig.savefig(out/'specificity_ablation.png',dpi=180);fig.savefig(out/'specificity_ablation.pdf');plt.close(fig)
        # Paired confidence intervals: one shared weight matrix across layers and contrasts.
        intervals=pd.DataFrame(ci);fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,op in zip(axes,['swap','ablate_mean']):
            for method,color,offset in [('pca','#0072B2',-.18),('polynomial','#000000',0),('fourier','#009E73',.18)]:
                part=intervals[(intervals.cohort=='fixed')&(intervals.operation==op)&(intervals['mode']=='raw')&(intervals.comparison=='helix_minus_'+method)&(intervals.bootstrap=='shared_nodes')&(intervals.bootstrap_seed==1729)].sort_values('layer')
                # Bounds may not contain the point with a skewed percentile bootstrap; draw explicit segments.
                for row in part.itertuples():ax.plot([row.low95,row.high95],[row.layer+offset]*2,color=color,linewidth=1)
                ax.scatter(part.effect_point,part.layer+offset,color=color,label='vs '+method,s=18)
            ax.axvline(0,color='.5',linewidth=.7);ax.set(xlabel='Helix minus baseline (paired mean effect)',ylabel='Block-input index',title=op);ax.set_yticks(cfg['layers']);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle('Exploratory shared-operand node percentile95% intervals; one checkpoint; no multiplicity correction');fig.savefig(out/'paired_intervals.png',dpi=180);fig.savefig(out/'paired_intervals.pdf');plt.close(fig)
    (out/'figure_description.md').write_text('Primary six-site layer indices all retained. Point/line means are conditional on frozen85pairs; dashed connectors are guides, not unmeasured layers. Shared-identity resampling intervals are exploratory and reused across conditions/layers; no IID-record pooling, no independent-model uncertainty, no significance stars. Raw and equal-energy controls differ; scaled controls need not lie on a natural clean representation manifold. All group counts, empty cells and unexecuted rank conditions are recorded.\n')
    print(json.dumps(quality),flush=True)
if __name__=='__main__':main()
