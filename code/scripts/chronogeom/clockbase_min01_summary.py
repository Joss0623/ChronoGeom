"""Prediction-focused summary of the fixed ClockBase pilot, no fitting or selection."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def csvsave(p,rows):
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def circ_shift(x,d):
 p=x.shape[-1];omega=2*np.pi*np.fft.rfftfreq(p);omega[-1]=0
 return x+np.fft.irfft(np.fft.rfft(x,axis=-1)*(np.exp(-1j*d[...,None]*omega)-1),n=p,axis=-1)
def effective_rank(x):
 s=np.linalg.svd(x-x.mean(-1,keepdims=True),compute_uv=False)**2;q=s/np.maximum(s.sum(-1,keepdims=True),1e-30)
 return np.exp(-np.sum(q*np.log(np.maximum(q,1e-30)),axis=-1))
def summarize(run,stage):
 c=json.loads((run/'config.json').read_text());d=run/stage;a=np.load(d/'predictions.npz');truth=a['truth'];arms=c['arms']+(['OracleClock'] if stage=='synthetic' else []);seeds=c['seeds'];groups=c['synthetic_regimes'] if stage=='synthetic' else ['Traffic16']
 errors={arm:np.stack([((a[f'{arm}_s{s}'].astype(float)-truth)**2).mean(1) for s in seeds]) for arm in arms};rows=[];contrasts=[];rng=np.random.default_rng(c['bootstrap_seed'])
 for gi,group in enumerate(groups):
  ix=a['regime']==gi if stage=='synthetic' else np.ones(len(truth),dtype=bool)
  for arm in arms:
   scores=errors[arm][:,ix].mean(1);rows.append(dict(stage=stage,group=group,arm=arm,n_instances=int(ix.sum()),mse_mean=float(scores.mean()),mse_seed0=float(scores[0]),mse_seed1=float(scores[1]),mse_seed2=float(scores[2]),relative_gain_vs_TimeBase=float(1-scores.mean()/errors['TimeBase'][:,ix].mean())))
  for baseline in ['TimeBase','SinCos',('PhaseMix' if 'PhaseMix' in arms else 'MatchedMix')]:
   difference=(errors['ClockBase']-errors[baseline])[:,ix].mean(0)
   if stage=='synthetic':
    boot=np.mean(difference[rng.integers(len(difference),size=(c['bootstrap_repeats'],len(difference)))],axis=1);units=len(difference)
   else:
    origin=a['origin'][ix];u=np.unique(origin);daydiff=np.array([difference[origin==o].mean() for o in u]);block=np.arange(len(u))//c.get('bootstrap_days',7);ids=np.unique(block);totals=np.array([daydiff[block==b].sum() for b in ids]);counts=np.array([(block==b).sum() for b in ids]);choice=rng.integers(len(ids),size=(c['bootstrap_repeats'],len(ids)));boot=totals[choice].sum(1)/counts[choice].sum(1);units=len(ids)
   lo,hi=np.quantile(boot,[.025,.975]);contrasts.append(dict(stage=stage,group=group,contrast='ClockBase-'+baseline,difference=float(difference.mean()),ci95_low=float(lo),ci95_high=float(hi),resampling_units=units,units_kind='independent_trajectory' if stage=='synthetic' else str(c.get('bootstrap_days',7))+'day_origin_block',seed_differences=json.dumps((errors['ClockBase'][:,ix].mean(1)-errors[baseline][:,ix].mean(1)).tolist())))
 if stage=='real':
  import pandas as pd
  scaler=np.load(d/'train_val.npz');v=(pd.read_csv(c['real_path'],usecols=c['real_columns'])[c['real_columns']].to_numpy(dtype='float32')-scaler['mean'])/scaler['sd']
  origins=a['origin'];channels=np.tile(np.arange(len(c['real_columns'])),len(np.unique(origins)));naives=[]
  persistence=np.repeat(v[origins-1,channels,None],c['pred_len'],axis=1)
  seasonal=v[origins[:,None]-c['period']+np.arange(c['pred_len'])%c['period'],channels[:,None]]
  for name,pred in [('Persistence',persistence),('Seasonal24',seasonal)]:naives.append(dict(method=name,mse=float(np.mean((pred-truth)**2)),mae=float(np.mean(abs(pred-truth)))))
  csvsave(d/'naive_scores.csv',naives)
 csvsave(d/'aggregate.csv',rows);csvsave(d/'contrasts.csv',contrasts)
 # Every condition retained; fixed baseline worst and ClockBase harm cases available.
 cases=[]
 for gi,group in enumerate(groups):
  ix=np.flatnonzero(a['regime']==gi) if stage=='synthetic' else np.arange(len(truth))
  for reason,values in [('largest_TimeBase_error',errors['TimeBase'].mean(0)),('largest_ClockBase_harm',(errors['ClockBase']-errors['TimeBase']).mean(0))]:
   chosen=ix[np.argsort(values[ix])[-8:][::-1]]
   for index in chosen:
    cases.append(dict(group=group,selection=reason,index=int(index),origin=int(a['origin'][index]),**{arm:float(errors[arm][:,index].mean()) for arm in arms}))
 csvsave(d/'failure_cases.csv',cases)
 if stage=='synthetic':
  test=np.load(d/'test.npz');x=test['clean'][:,:,0].reshape(len(truth),-1,c['period']);past=np.arange(1-x.shape[1],1);base_rank=effective_rank(x);true_rank=effective_rank(circ_shift(x,-test['speed'][:,None]*past));rankrows=[]
  for seed in seeds:
   v=np.load(d/f'clock_speed_s{seed}.npz')['speed'];estimated=effective_rank(circ_shift(x,-v[:,None]*past))
   for gi,group in enumerate(groups):
    ix=test['regime']==gi;rankrows.append(dict(seed=seed,group=group,raw_effective_rank=float(base_rank[ix].mean()),true_aligned_rank=float(true_rank[ix].mean()),estimated_aligned_rank=float(estimated[ix].mean())))
  csvsave(d/'rank_effect.csv',rankrows)
 print('AGGREGATE',stage)
 for r in rows:print(r)
 print('PAIRED DIFFERENCES')
 for r in contrasts:print(r)
 return rows

def plot(run):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 c=json.loads((run/'config.json').read_text());fig,axes=plt.subplots(1,3,figsize=(14,3.6),constrained_layout=True);examples=[('synthetic','shift_drift'),('synthetic','continuous_drift'),('real','Traffic16')]
 for ax,(stage,group) in zip(axes,examples):
  a=np.load(run/stage/'predictions.npz');idx=int(np.flatnonzero(a['regime']==c['synthetic_regimes'].index(group))[0]) if stage=='synthetic' else 0
  ax.plot(a['truth'][idx],color='black',label='Truth',lw=1.7)
  for arm,col in [('TimeBase','#999999'),('MatchedMix','#009E73'),('ClockBase','#D55E00')]:ax.plot(a[arm+'_s0'][idx],color=col,label=arm,lw=1.2)
  ax.set_title(group+' / fixed first example');ax.set_xlabel('Forecast hour');ax.set_ylabel('Signal units' if stage=='synthetic' else 'Training SD')
 axes[0].legend(fontsize=8);fig.savefig(run/'clockbase_min01_examples.png',dpi=170);fig.savefig(run/'clockbase_min01_examples.pdf');plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--stage',choices=['synthetic','real']);p.add_argument('--plot',action='store_true');a=p.parse_args()
 if a.plot:plot(a.run)
 else:summarize(a.run,a.stage)
