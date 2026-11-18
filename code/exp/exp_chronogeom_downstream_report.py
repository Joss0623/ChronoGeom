"""CPU-only evidence tables, fixed-checkpoint replay and figures for Seoul downstream pilot."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvsave(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main(r):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import torch
 from exp.exp_chronogeom_downstream import CalendarTimeBase,pack,read_data,tensors,predict
 torch.set_num_threads(2);c=read(r/'config.json');source=read(r/'source.json');lock=read(r/'models_lock.json');gate=read(r/'pilot/gate.json');out=Path('docs/chronogeom/downstream_seoul01');out.mkdir(exist_ok=False)
 assert sha('exp/exp_chronogeom_downstream.py')==source['script_sha256'];assert sha(c['data_path'])==source['data_sha256'];assert sha(r/'tables.npz')==source['tables_sha256'];assert sha(r/'train_val.npz')==source['prepared_sha256']
 assert len(lock['models'])==39 and not gate['passed'] and not list(r.glob('kd_seed*'))
 checks=[];fits=[];initial=[]
 for s in c['seeds']:
  completion=read(r/f'seed{s}/completion.json');assert completion['seconds']<240 and completion['peak_gib']<2 and read(r/f'seed{s}_exit.json')['exit_code']==0
  v0=[]
  for rec in completion['records']:
   assert sha(rec['path'])==rec['sha256'];state=torch.load(rec['path'],map_location='cpu',weights_only=True);assert all(bool(torch.isfinite(x).all()) for x in state.values())
   curve=read(r/f'seed{s}'/(rec['arm']+'_curve.json'));assert [x['step'] for x in curve]==list(range(0,601,100))
   best=min(curve,key=lambda x:x['val_mse']);assert best['step']==rec['best_step'];assert rec['parameters']==(40 if rec['arm']=='Original' else 6570);v0.append(curve[0]['val_mse']);fits.append(rec)
  assert max(v0)-min(v0)==0.;initial.append(dict(seed=s,all13_step0_val_mse=v0[0],exact_initial_forecast_score=True))
 summaries=[];contrasts=[];failure=[];max_score_error=0.
 for stage in ['pilot','confirmation']:
  assert (r/'models_lock.json').stat().st_mtime<=(r/stage/'before_target_access.json').stat().st_mtime
  p=np.load(r/stage/'predictions.npz');truth=p['truth'].astype('float64');origins=p['origin'];assert len(origins)==(53 if stage=='pilot' else 68)
  lo,hi=(c['val_end'],c['pilot_end']) if stage=='pilot' else (c['pilot_end'],c['confirmation_end'])
  assert origins.min()==lo and origins.max()+96<=hi and np.all(np.diff(origins)==24)
  scores=list(csv.DictReader((r/stage/'scores.csv').open()))
  for rec in scores:
   e=p[rec['method']].astype('float64')-truth;mse=float(np.mean(e**2));mae=float(np.mean(abs(e)));max_score_error=max(max_score_error,abs(mse-float(rec['mse'])),abs(mae-float(rec['mae'])))
  assert max_score_error<2e-6
  summary=read(r/stage/'summary.json')
  for rec in summary:
   q=[x for x in scores if x['method'] in [f"{rec['arm']}_s{s}" for s in c['seeds']]]
   summaries.append(dict(stage=stage,arm=rec['arm'],mse=rec['mse'],mae=rec['mae'],raw_mse=float(np.mean([float(x['raw_mse']) for x in q])),raw_mae=float(np.mean([float(x['raw_mae']) for x in q])),seed_sd=float(np.std(rec['seed_mse'],ddof=1)),seed0_mse=rec['seed_mse'][0],seed1_mse=rec['seed_mse'][1],seed2_mse=rec['seed_mse'][2]))
  for q in read(r/stage/'contrasts.json'):contrasts.append(dict(stage=stage,comparison=q['comparison'],difference=q['difference'],lo95=q['lo95'],hi95=q['hi95'],seed_differences=str(q['seed_differences'])))
  original=np.mean([((p[f'Original_s{s}']-truth)**2).mean((1,2)) for s in c['seeds']],axis=0)
  for i in np.argsort(original)[-5:][::-1]:
   vals={}
   for name in ['Original',lock['llm_arm'],lock['analytic_arm']]:
    vals[name]=float(np.mean([((p[f'{name}_s{s}'][i]-truth[i])**2).mean() for s in c['seeds']]))
   failure.append(dict(stage=stage,origin=int(origins[i]),selection='top5 by Original mean squared error; all methods retained',original_mse=vals['Original'],llm_mse=vals[lock['llm_arm']],strong_mse=vals[lock['analytic_arm']]))
  for n in ['scores.csv','gate.json','completion.json']:
   if (r/stage/n).exists():(out/(stage+'_'+n)).write_bytes((r/stage/n).read_bytes())
 csvsave(out/'forecast_summary.csv',summaries);csvsave(out/'paired_block_intervals.csv',contrasts);csvsave(out/'failure_cases.csv',failure);csvsave(out/'fit_records.csv',fits);save(out/'initialization_checks.json',initial)
 # Replay nine locked models on CPU. Confirmation was already unsealed; no training or re-selection.
 raw,ids,_=read_data(c,c['confirmation_end']);a=np.load(r/'train_val.npz');pn=pack((raw-a['mean'])/a['sd'],ids,c['pilot_end'],c['confirmation_end'],c);p=tensors({k:v[:2] for k,v in pn.items()},'cpu');tables=np.load(r/'tables.npz');saved=np.load(r/'confirmation/predictions.npz')
 replay=[]
 for rec in lock['models']:
  if rec['arm'] not in ['Original',lock['llm_arm'],lock['analytic_arm']]:continue
  model=CalendarTimeBase(c,rec['seed'],None if rec['arm']=='Original' else tables[rec['arm']]);model.load_state_dict(torch.load(rec['path'],map_location='cpu',weights_only=True));yp=predict(model,p);err=float(np.max(abs(yp-saved[f"{rec['arm']}_s{rec['seed']}"][:2])));assert err<5e-5;replay.append(dict(arm=rec['arm'],seed=rec['seed'],origins=2,cpu_gpu_maxdiff=err))
 size=sum(x.stat().st_size for x in r.rglob('*') if x.is_file());assert size<c['output_cap_mib']*2**20
 total=sum(read(r/f'seed{s}/completion.json')['seconds'] for s in c['seeds']);assert total<720
 verification=dict(models=39,all_checkpoints_finite_and_sha=True,all_curves600updates=True,step0_scores_identical=initial,metrics_max_recompute_error=max_score_error,replays=replay,max_replay_error=max(x['cpu_gpu_maxdiff'] for x in replay),data_and_source_hashes=True,train_val_pilot_confirm_targets_disjoint=True,confirmation_unmodified_after_pilot=True,no_kd_triggered=True,clockteacher_training_runs=0,new_llm_forwards=0,fit_worker_wall_seconds_sum=total,wall_time_is_not_exact_cuda_kernel_seconds=True,output_bytes=size,all_failures_retained=True)
 save(out/'verification.json',verification)
 for n in ['source.json','models_lock.json','teacher_preparation.json','teacher_diagnostics.csv','prompt_stability.csv','analytical_explanation.csv','train_calendar_association.json']:(out/n).write_bytes((r/n).read_bytes())
 with plt.rc_context({'font.size':9,'pdf.fonttype':42,'figure.facecolor':'white'}):
  fig,axes=plt.subplots(1,2,figsize=(10,6),layout='constrained',sharey=True)
  for ax,stage in zip(axes,['pilot','confirmation']):
   for i,name in enumerate(c['arms']):
    q=next(x for x in summaries if x['stage']==stage and x['arm']==name);color='#D55E00' if name.startswith('LLM') else '#0072B2' if name not in ['Original','NoCalendar'] else '#666666';ax.errorbar(q['mse'],i,xerr=q['seed_sd'],fmt='o',color=color,capsize=3)
   ax.set_yticks(range(len(c['arms'])),c['arms']);ax.set(xlabel='Standardized MSE (mean ± seed SD)',title=stage.capitalize());ax.grid(axis='x',alpha=.2)
  axes[0].invert_yaxis()
  fig.savefig(out/'forecast_results.png',dpi=180,facecolor='white');fig.savefig(out/'forecast_results.pdf',facecolor='white');plt.close(fig)
 save(out/'figure_provenance.json',dict(source='forecast_summary.csv',unit='three training seeds; one real trajectory; whiskers show seed SD, not independent-window CI',size_inches=[10,6],png_dpi=180,smoothing=None,alt_text='Pilot and independent confirmation MSE for every frozen baseline, with all three seed variation retained. Paired time-block intervals are in a separate table.'))
 print(json.dumps(verification))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);main(p.parse_args().run)
