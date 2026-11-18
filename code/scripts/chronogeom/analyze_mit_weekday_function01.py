"""Fixed descriptive analysis; no model selection or test-dependent refitting."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def write(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def plot(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((run/'summary.csv').open()));geom=read(run/'geometry.json')
    fig,ax=plt.subplots(2,3,figsize=(12,6.4),constrained_layout=True)
    days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];methods=['full','phase2','pca2_norm','random2_norm','learned_transition','analytic_rotation']
    labels=['Full','Phase2','PCA2*','Random2*','Learned','Analytic'];colors=['#777777','#0072B2','#009E73','#999999','#D55E00','#CC79A7']
    for j,layer in enumerate([6,14,20]):
        sp=np.load(run/f'space_layer{layer}.npz');cent=sp['centroids'];_,sv,v=np.linalg.svd(cent,full_matrices=False);q=cent@v[:2].T
        a=ax[0,j];a.plot(np.r_[q[:,0],q[0,0]],np.r_[q[:,1],q[0,1]],color='#999999',lw=1)
        a.scatter(q[:,0],q[:,1],c=np.arange(7),cmap='viridis',s=45,zorder=3)
        for i,d in enumerate(days):a.annotate(d,q[i],xytext=(4,4),textcoords='offset points',fontsize=8)
        a.set_aspect('equal',adjustable='datalim');a.margins(.25);a.set_title(f'Layer {layer}: native centroid PCA2\n{100*geom[j]["native_pca2_variance_share"]:.1f}% centroid energy',fontsize=10);a.set_xlabel('PC1 (unscaled)');a.set_ylabel('PC2 (unscaled)')
        a=ax[1,j];values=[float(next(r['logit_difference_change'] for r in rows if r['context']=='-1' and r['subset']=='all' and r['layer']==str(layer) and r['method']==m)) for m in methods]
        a.bar(np.arange(6),values,color=colors);a.axhline(0,c='black',lw=.7);a.set_xticks(np.arange(6),labels,rotation=40,ha='right');a.set_ylabel('Mean target logit-difference change');a.set_title(f'All 84 pairs, layer {layer}',fontsize=10)
    fig.suptitle('GPT-J weekday intervention: phase information has local effects; learned and analytic rotations are close',fontsize=11)
    fig.savefig(run/'weekday_function01.png',dpi=180);fig.savefig(run/'weekday_function01.pdf');plt.close(fig)

def analyze(run):
    end=read(run/'completion.json');records=read(run/'prompts.json');base=read(run/'baseline.json');pairs=read(run/'pairs.json');cfg=read(run/'config.json')
    rows=list(csv.DictReader((run/'interventions.csv').open()))
    integer=['pair_id','layer','context','day','offset','delta','candidate_prediction','global_token']
    boolean=['joint_candidate_correct','joint_global_correct','norm_match_valid','candidate_target_correct','global_target_correct','corrupt_candidate_target_correct','corrupt_global_target_correct']
    text=['method','global_text']
    for r in rows:
        for k,v in r.items():
            if k in integer:r[k]=int(v)
            elif k in boolean:r[k]=v=='True'
            elif k not in text:r[k]=float(v)
    summaries=[]
    methods=list(dict.fromkeys(r['method'] for r in rows))
    for context in [-1,0,1]:
        for layer in cfg['layers']:
            for eligible in ['all','joint_candidate','joint_global']:
                for method in methods:
                    a=[r for r in rows if r['layer']==layer and r['method']==method and (context==-1 or r['context']==context) and (eligible=='all' or r[eligible+'_correct'])]
                    rec=dict(context=context,layer=layer,subset=eligible,method=method,n=len(a))
                    for key in ['logit_difference_change','logit_difference','candidate_target_correct','global_target_correct','patch_norm','relative_patch_norm','phase_donor_distance','activation_donor_distance']:
                        rec[key]=float(np.mean([r[key] for r in a])) if a else None
                    summaries.append(rec)
    write(run/'summary.csv',summaries)
    # Conditional descriptive resampling by the seven clean weekday categories only.
    # Shared corrupted identities and fitted model remain dependencies: not population significance.
    contrasts=[('phase2','pca2_norm'),('phase2','random2_norm'),('learned_transition','analytic_rotation'),('learned_transition','phase2')]
    ci=[];rng=np.random.default_rng(1730);resamples=rng.integers(0,7,(2000,7))
    index={(r['layer'],r['pair_id'],r['method']):r for r in rows}
    for context in [-1,0,1]:
        for layer in [-1]+cfg['layers']:
            for a,b in contrasts:
                byday=[]
                for day in range(7):
                    matches=[r for r in rows if r['day']==day and r['method']==a and (context==-1 or r['context']==context) and (layer==-1 or r['layer']==layer)]
                    byday.append(np.mean([r['logit_difference_change']-index[(r['layer'],r['pair_id'],b)]['logit_difference_change'] for r in matches]))
                byday=np.array(byday);samples=byday[resamples].mean(1);lo,hi=np.quantile(samples,[.025,.975])
                ci.append(dict(context=context,layer=layer,contrast=a+' minus '+b,mean=float(byday.mean()),conditional_low=float(lo),conditional_high=float(hi),clean_weekday_groups=7))
    write(run/'paired_descriptive_intervals.csv',ci)
    baseline=[]
    for ctx in [0,1]:
        for off in [1,2,3]:
            b=[r for r in base if r['context']==ctx and r['offset']==off]
            baseline.append(dict(context=ctx,offset=off,n=len(b),candidate_correct=sum(x['candidate_correct'] for x in b),global_correct=sum(x['global_correct'] for x in b)))
    write(run/'baseline_by_context_offset.csv',baseline)
    raw=np.load(run/'raw_activations.npz')['hidden'].astype(float);fit=np.array([r['split']=='fit' for r in records]);test=~fit;phase=np.array([[np.cos(r['day']*2*np.pi/7),np.sin(r['day']*2*np.pi/7)] for r in records]);decode=[]
    for li,layer in enumerate(cfg['layers']):
        s=np.load(run/f'space_layer{layer}.npz')
        for name in ['phase2','pca2','random2']:
            z=(raw[:,li]-s['mu'])@s[name];x=np.c_[z,np.ones(len(z))];coef=np.linalg.lstsq(x[fit],phase[fit],rcond=1e-10)[0];pred=x@coef
            for ctx in [0,1]:
                ix=test&np.array([r['context']==ctx for r in records]);err=np.sum((pred[ix]-phase[ix])**2);den=np.sum((phase[ix]-phase[ix].mean(0))**2)
                angle=np.angle((pred[ix,0]+1j*pred[ix,1])/(phase[ix,0]+1j*phase[ix,1]));decode.append(dict(layer=layer,space=name,context=ctx,n=int(ix.sum()),phase_r2=float(1-err/den),phase_absolute_error_degrees=float(np.rad2deg(abs(angle)).mean())))
    write(run/'heldout_offset_phase_readout.csv',decode)
    mismatch=[]
    for layer in cfg['layers']:
        for name in ['pca2_norm','random2_norm']:
            ratios=[]
            for r in rows:
                if r['layer']!=layer or r['method']!=name:continue
                target=index[(layer,r['pair_id'],'phase2')]['patch_norm'];ratios.append(abs(r['patch_norm']/target-1) if target else 0.)
            mismatch.append(dict(layer=layer,method=name,max_actual_relative_norm_error=float(max(ratios)),mean_actual_relative_norm_error=float(np.mean(ratios))))
    write(run/'actual_norm_match.csv',mismatch)
    examples=[]
    for p in pairs:
        for layer in cfg['layers']:
            a=index[(layer,p['id'],'phase2')];b=index[(layer,p['id'],'analytic_rotation')];f=index[(layer,p['id'],'full')]
            examples.append(dict(layer=layer,pair_id=p['id'],context=p['context'],prompt=records[p['corrupt_id']]['text'],desired_answer=records[p['clean_id']]['answer'],corrupt_global=base[p['corrupt_id']]['global_text'],clean_global=base[p['clean_id']]['global_text'],phase_global=a['global_text'],analytic_global=b['global_text'],full_global=f['global_text'],phase_target_correct=a['global_target_correct'],phase_ld_change=a['logit_difference_change'],analytic_ld_change=b['logit_difference_change']))
    write(run/'all_cases.csv',examples)
    # Post-hoc transparency only: three test queries duplicate frozen demonstrations.
    # Preserve all primary rows; do not refit, select layers, or call this confirmation.
    demo={(0,1),(3,2),(5,3)}
    retained={p['id'] for p in pairs if p['context']==1 and all((records[p[k]]['day'],p['offset']) not in demo for k in ['clean_id','corrupt_id'])}
    sensitivity=[]
    for layer in cfg['layers']:
        for method in methods:
            a=[r for r in rows if r['layer']==layer and r['method']==method and r['pair_id'] in retained]
            sensitivity.append(dict(layer=layer,method=method,n=len(a),ld_change=float(np.mean([r['logit_difference_change'] for r in a])),global_target_count=sum(r['global_target_correct'] for r in a),joint_global_pairs=sum(r['joint_global_correct'] for r in a)))
    write(run/'posthoc_demo_overlap_sensitivity.csv',sensitivity)
    # This is a report convenience, not selecting a new model, layer or parameter.
    print(json.dumps(end,indent=2));print('Global all-pair summaries:')
    for r in summaries:
        if r['context']==-1 and r['subset']=='all':print(r)
    plot(run)
    dump(run/'analysis_completion.json',dict(n_rows=len(rows),n_case_rows=len(examples),n_summaries=len(summaries),bootstrap_replicates=2000,bootstrap_groups=7,limitations='Descriptive conditional interval; shared corrupted weekdays and fitted basis dependencies remain, one fixed model/random seed.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);analyze(p.parse_args().run)
