"""All-cell, calendar-block summaries for the frozen real-data Circle pilot."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(root):
    out=root/'summary';out.mkdir();figs=out/'figures';figs.mkdir()
    cfg=json.loads((root/'config.json').read_text());d=pd.read_csv(root/'interventions.csv');ids=np.sort(d.trajectory.unique())
    meta=d.drop_duplicates('trajectory').set_index('trajectory').reindex(ids);blocks=meta.block.to_numpy();keys=np.unique(blocks)
    rng=np.random.default_rng(1729);draw=rng.choice(keys,(cfg['bootstrap'],len(keys)),replace=True)
    # Variable cluster sizes: repeat whole 14-day blocks; retain all model/shift repetitions within source.
    boot=[np.concatenate([np.where(blocks==b)[0] for b in row]) for row in draw]
    np.savez_compressed(out/'bootstrap.npz',block_draws=draw,trajectory_ids=ids,blocks=blocks)
    rows=[];coh=[];seedrows=[];by={}
    for (arm,site,cond),v in d.groupby(['arm','site','condition'],sort=False):
        t=v.groupby('trajectory').mean(numeric_only=True).reindex(ids);by[(arm,site,cond)]=t
        e=t.target_pred_mse.to_numpy();den=t.base_target_mse.to_numpy();diff=t.recipient_truth_mse.to_numpy()-t.base_truth_mse.to_numpy()
        bs=np.array([1-e[z].mean()/max(den[z].mean(),1e-12) for z in boot]);bd=np.array([diff[z].mean() for z in boot])
        row=dict(arm=arm,site=site,condition=cond,n_trajectories=len(ids),blocks=len(keys),records=len(v),recovery=1-e.mean()/max(den.mean(),1e-12),
            low=np.quantile(bs,.025),high=np.quantile(bs,.975),base_target_mse=den.mean(),target_pred_mse=e.mean(),truth_mse_change=diff.mean(),truth_change_low=np.quantile(bd,.025),truth_change_high=np.quantile(bd,.975),
            phase_error_target_deg=np.degrees(v.loc[v.phase_defined,'phase_error_target'].mean()),phase_error_observed_deg=np.degrees(v.loc[v.observed_phase_defined,'phase_error_observed'].mean()),
            phase_change_deg=np.degrees(v.loc[v.phase_defined,'phase_change'].mean()),target_phase_change_deg=np.degrees(v.loc[v.phase_defined,'target_phase_change'].mean()),phase_undefined=int((~v.phase_defined).sum()),
            norm_mismatch_rows=int((v.norm_match_error>.01).sum()))
        for k in ['recipient_truth_mse','base_truth_mse','donor_truth_mse','prediction_change_mse','amplitude','amplitude_change','amplitude_error_observed','offset_change','trend_change','offset_error_observed','trend_error_observed','nonharmonic_mse','delta_norm','relative_delta','nn_state_distance','target_centroid_distance','mahal']:row[k]=v[k].mean()
        rows.append(row)
        for c,part in v.groupby('cohort'):
            coh.append(dict(arm=arm,site=site,condition=cond,cohort=int(c),n=part.trajectory.nunique(),recovery=1-part.target_pred_mse.mean()/max(part.base_target_mse.mean(),1e-12),recipient_truth_mse=part.recipient_truth_mse.mean()))
        for seed,part in v.groupby('seed'):
            seedrows.append(dict(arm=arm,site=site,condition=cond,seed=int(seed),n=part.trajectory.nunique(),recovery=1-part.target_pred_mse.mean()/max(part.base_target_mse.mean(),1e-12),recipient_truth_mse=part.recipient_truth_mse.mean()))
    s=pd.DataFrame(rows);s.to_csv(out/'intervention_summary.csv',index=False);pd.DataFrame(coh).to_csv(out/'cohort_repeats.csv',index=False);pd.DataFrame(seedrows).to_csv(out/'seed_repeats.csv',index=False)
    contrasts=[]
    for arm in ['clock','natural']:
        for site in cfg['sites']:
            c=by[(arm,site,'circle')];den=c.base_target_mse.to_numpy()
            for other in ['cv_phase_norm','info_match_norm','joint_match_norm','residual_pca_norm','random0_norm','harmonic_mean_delta_norm']:
                o=by[(arm,site,other)];diff=o.target_pred_mse.to_numpy()-c.target_pred_mse.to_numpy();bs=np.array([diff[z].mean()/max(den[z].mean(),1e-12) for z in boot])
                contrasts.append(dict(arm=arm,site=site,contrast='circle_minus_'+other,difference=diff.mean()/max(den.mean(),1e-12),low=np.quantile(bs,.025),high=np.quantile(bs,.975)))
    pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv',index=False)
    byshift=d.groupby(['arm','site','shift','condition'])[['target_pred_mse','base_target_mse','recipient_truth_mse','amplitude_change','nonharmonic_mse']].mean().reset_index()
    byshift['recovery']=1-byshift.target_pred_mse/byshift.base_target_mse
    byshift['recipient_time_information']=byshift.apply(lambda a:'offline only: +6h donor history includes recipient future' if a.arm=='natural' and a['shift']>0 else 'no recipient-future values in model inputs',axis=1)
    byshift.to_csv(out/'shift_availability_audit.csv',index=False)
    cal=pd.read_csv(root/'calibration.csv');test=pd.read_csv(root/'test_space_scores.csv');ref=test[test.space.eq('circle')][['seed','site','test_phase_r2','test_capture']].rename(columns={'test_phase_r2':'circle_test_phase_r2','test_capture':'circle_test_capture'})
    m=cal.merge(test,on=['seed','site','space']).merge(ref,on=['seed','site']);m['test_phase_match']=abs(m.test_phase_r2-m.circle_test_phase_r2)<=cfg['phase_tolerance']
    m['test_reconstruction_relative_error']=abs(m.test_capture-m.circle_test_capture)/m.circle_test_capture;m['test_reconstruction_match']=m.test_reconstruction_relative_error<=cfg['reconstruction_tolerance']
    m['strict_joint_match']=m.phase_match&m.reconstruction_match&m.test_phase_match&m.test_reconstruction_match;m.to_csv(out/'matching_diagnostics.csv',index=False)
    ng=pd.read_csv(root/'native_geometry.csv');ng.groupby(['stage','site']).mean(numeric_only=True).to_csv(out/'native_summary.csv')
    # Every failure retained; direction is judged relative to donor model's phase, not a physical sinusoid Oracle.
    fail=d[d.condition.eq('circle')].copy();fail['negative_recovery']=fail.target_pred_mse>fail.base_target_mse
    fail['phase_away_from_donor']=fail.phase_defined&(fail.phase_change*fail.target_phase_change<0)
    fail.to_csv(out/'all_circle_cases.csv',index=False);fail.groupby(['arm','site'])[['negative_recovery','phase_away_from_donor']].sum().to_csv(out/'failure_counts.csv')
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    short=lambda a:a.replace('dec_embedding.temporal_embedding.','')
    fig,ax=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for si,site in enumerate(cfg['sites']):
        allxy=[]
        for row,stage in enumerate(['init','best']):
            a=np.load(root/'native'/f'0_{stage}_{si}.npz');xy=a['raw_projection'];allxy.append(xy)
            axis=ax[row,si];axis.plot(np.r_[xy[:,0],xy[0,0]],np.r_[xy[:,1],xy[0,1]],color='gray',lw=.8);axis.scatter(xy[:,0],xy[:,1],c=np.arange(24),cmap='twilight',s=20)
            for hour in [0,6,12,18]:axis.annotate(str(hour),xy[hour],fontsize=7)
            axis.set(title=stage+' / '+short(site),xlabel='Unsupervised PC1',ylabel='PC2');axis.set_aspect('equal',adjustable='box')
        lim=np.max(abs(np.concatenate(allxy)))*1.1
        for row in range(2):ax[row,si].set(xlim=(-lim,lim),ylim=(-lim,lim))
    fig.savefig(figs/'native_geometry.png',dpi=180);fig.savefig(figs/'native_geometry.pdf');plt.close(fig)
    conditions=['circle','cv_phase_norm','info_match_norm','residual_pca_norm','random0_norm','joint_match_norm','harmonic_mean_delta_norm','full_donor']
    fig,ax=plt.subplots(2,4,figsize=(12,7),sharex=True,sharey=True,layout='constrained')
    for row,arm in enumerate(['clock','natural']):
        for si,site in enumerate(cfg['sites']):
            v=s[(s.arm==arm)&(s.site==site)].set_index('condition').loc[conditions]
            axis=ax[row,si];axis.hlines(np.arange(len(v)),v.low,v.high,color='#0072B2',lw=1);axis.scatter(v.recovery,np.arange(len(v)),color='#0072B2',s=18)
            axis.set_yticks(np.arange(len(v)),conditions);axis.axvline(0,color='gray',lw=.8);axis.set(title=arm+' / '+short(site),xlabel='Recovery toward donor prediction')
    ax[0,0].invert_yaxis();fig.savefig(figs/'matched_interventions.png',dpi=180);fig.savefig(figs/'matched_interventions.pdf');plt.close(fig)
    # Fixed first test origin, seed0, natural +6h; never choose a visually favorable case.
    pack=np.load(root/'data/test.npz');date=str(np.datetime64(int(pack['actual_t'][0,0]),'h'))
    fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    for axis,si in zip(ax,[0,3]):
        for cond,color,style in [('noop','black','-'),('full_donor','#999999','--'),('circle','#0072B2','-'),('cv_phase_norm','#D55E00','--'),('info_match_norm','#009E73',':'),('harmonic_mean_delta_norm','#CC79A7','-.')]:
            y=np.load(root/'interventions/predictions'/f'0_{si}_natural_6_{cond}.npz')['prediction'][0,:,0]
            axis.plot(np.arange(1,25),y,label=cond,color=color,ls=style,lw=1.3)
        axis.plot(np.arange(1,25),pack['y'][0,:,0],'o',ms=3,mfc='none',color='black',label='Recipient observations')
        axis.set(title='Fixed first case '+date+' / '+short(cfg['sites'][si]),ylabel='Train-standardized OT')
    ax[0].legend(ncol=3,fontsize=7);ax[-1].set_xlabel('Forecast horizon (hours)');fig.savefig(figs/'fixed_case.png',dpi=180);fig.savefig(figs/'fixed_case.pdf');plt.close(fig)
    js=dict(source_commit=json.loads((root/'provenance.json').read_text())['source_commit'],
        selection='native figure seed0 fixed; waveform first test origin,seed0,natural+6 predetermined; all cases and seeds preserved',
        uncertainty='1000 resampled 14-day blocks,9 blocks,59 source windows,model/shift averaged; conditional on fixed trained models and fitted directions',
        projection='fit hour-centroid PCA2,orthogonal only,no whitening or phase-supervised rotation; labels only aggregation/evaluation',
        units='Train-only standardized OT; joint offset/trend/sin/cos decomposition is descriptive,not a physical phase Oracle',
        matching='phase readout R2 not full information; see all match flags; joint-match includes failures')
    (out/'figure_provenance.json').write_text(json.dumps(js,indent=2));print(s[s.condition.isin(['circle','cv_phase_norm','info_match_norm','harmonic_mean_delta_norm'])][['arm','site','condition','recovery','recipient_truth_mse','amplitude_change','nonharmonic_mse']].round(5).to_string(index=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);main(Path(ap.parse_args().run))
