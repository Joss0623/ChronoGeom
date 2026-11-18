"""Fixed five-horizon dispatch reusing the existing TSLib pilot; no new model framework."""
import argparse,copy,json,os,shutil,time,traceback
from pathlib import Path
import numpy as np
from exp import exp_chronogeom_clockbase as e
from scripts.chronogeom.clockbase_min01_summary import summarize,csvsave

def config(c,h):
 a=copy.deepcopy(c);a.update(run=str(Path(c['root_run'])/f'H{h}'),pred_len=h,bootstrap_days=max(7,(h+23)//24));return a

def prepare(c):
 root=Path(c['root_run']);root.mkdir(parents=True,exist_ok=False);e.dump(root/'config.json',c);e.dump(root/'source.json',dict(commit=e.commit(),experiment_sha256=e.sha(Path(e.__file__)),runner_sha256=e.sha(__file__),data_sha256=e.sha(c['prepared_data_source']),timebase_sha256=e.sha(e.ROOT/'layers/ChronoGeomTimeBaseOfficial.py')))
 old=Path(c['prepared_data_source']).parent
 for h in c['horizons']:
  a=config(c,h);r=Path(a['run']);(r/'real').mkdir(parents=True);e.dump(r/'config.json',a);os.link(c['prepared_data_source'],r/'real/train_val.npz')
  m=json.loads((old/'data_manifest.json').read_text());m.update(seq_len=96,pred_len=h,test_status=c['test_status']);e.dump(r/'real/data_manifest.json',m)
  e.selfcheck(a)
 print('All five configurations prepared; enhanced arms match in parameter count',flush=True)

def train(c,seed,device):
 for h in c['horizons']:
  a=config(c,h);r=Path(a['run']);print('START',h,seed,device,flush=True)
  try:
   e.fit(a,r,'real',seed,device)
   a['lr']=.0003;e.fit(a,r,'real',seed,device,True)
  except BaseException as ex:
   e.dump(Path(c['root_run'])/f'failure_H{h}_s{seed}.json',dict(error=str(ex),traceback=traceback.format_exc()));raise
  print('FINISHED',h,seed,flush=True)
 e.dump(Path(c['root_run'])/f'worker_s{seed}_completion.json',dict(seed=seed,device=device,commit=e.commit(),horizons=c['horizons']))

def evaluate(c,device):
 root=Path(c['root_run']);locks=[]
 # Lock every selected model across the complete grid before opening any grid test.
 for h in c['horizons']:
  r=Path(config(c,h)['run'])/'real'
  for seed in c['seeds']:
   records=[]
   for folder in [r/f'seed{seed}',r/f'seed{seed}_lr0003']:records+=json.loads((folder/'completion.json').read_text())['records']
   for arm in c['arms']:
    a=min([x for x in records if x['arm']==arm],key=lambda x:x['val_mse']);locks.append(dict(horizon=h,seed=seed,arm=arm,path=a['path'],sha256=e.sha(a['path']),val_mse=a['val_mse']))
 e.dump(root/'all_grid_test_lock.json',dict(commit=e.commit(),models=locks,test_status=c['test_status']))
 matrix=[];allfits=[];resources=[];contrasts=[]
 for h in c['horizons']:
  a=config(c,h);r=Path(a['run']);e.evaluate(a,r,'real',device);rows=summarize(r,'real');scores=list(__import__('csv').DictReader((r/'real/scores.csv').open()))
  for row in rows:
   select=[x for x in scores if x['arm']==row['arm']];mse=np.array([float(x['mse']) for x in select]);mix=next(x['mse_mean'] for x in rows if x['arm']=='PhaseMix')
   matrix.append(dict(horizon=h,arm=row['arm'],mse=row['mse_mean'],mae=float(np.mean([float(x['mae']) for x in select])),seed_sd=float(mse.std(ddof=1)),seed0=row['mse_seed0'],seed1=row['mse_seed1'],seed2=row['mse_seed2'],phase_mix_relative_gain=float(1-mix/row['mse_mean']),n_instances=row['n_instances'],bootstrap_days=a['bootstrap_days']))
  preds=np.load(r/'real/predictions.npz');origin=preds['origin'];days=np.unique(origin);block=np.arange(len(days))//a['bootstrap_days'];blocks=np.unique(block);rng=np.random.default_rng(c['bootstrap_seed']+h)
  errors={arm:np.mean([((preds[f'{arm}_s{seed}'].astype(float)-preds['truth'])**2).mean(1) for seed in c['seeds']],axis=0) for arm in c['arms']}
  for baseline in ['TimeBase','SinCos','ClockBase']:
   diff=errors['PhaseMix']-errors[baseline];daily=np.array([diff[origin==day].mean() for day in days]);sums=np.array([daily[block==b].sum() for b in blocks]);counts=np.array([(block==b).sum() for b in blocks]);choice=rng.integers(len(blocks),size=(c['bootstrap_repeats'],len(blocks)));draw=sums[choice].sum(1)/counts[choice].sum(1);lo,hi=np.quantile(draw,[.025,.975])
   contrasts.append(dict(horizon=h,contrast='PhaseMix-'+baseline,difference=float(daily.mean()),lo95=float(lo),hi95=float(hi),block_days=a['bootstrap_days'],n_blocks=len(blocks),n_origins=len(days)))
  csvsave(root/'phase_mix_contrasts.csv',contrasts)
  for seed in c['seeds']:
   for folder in [r/'real'/f'seed{seed}',r/'real'/f'seed{seed}_lr0003']:
    d=json.loads((folder/'completion.json').read_text());allfits.extend(dict(horizon=h,**x) for x in d['records'])
   resources.append(dict(horizon=h,seed=seed,cumulative_seconds=d['seconds'],peak_gib=d['peak_gib']))
  csvsave(root/'grid_scores.csv',matrix);csvsave(root/'all_fits.csv',allfits);csvsave(root/'resources.csv',resources)
  print('HORIZON EVALUATED',h,flush=True)
 for item in locks:assert e.sha(item['path'])==item['sha256']
 e.dump(root/'evaluation_completion.json',dict(commit=e.commit(),horizons=c['horizons'],selected_models=len(locks),fits=len(allfits),record_fit_seconds=sum(x['seconds'] for x in allfits),all_locked_states_unchanged=True,test_status=c['test_status']))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','evaluate']);p.add_argument('--config',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda:2');a=p.parse_args();c=json.loads(Path(a.config).read_text())
 if a.action=='prepare':prepare(c)
 elif a.action=='train':train(c,a.seed,a.device)
 else:evaluate(c,a.device)
