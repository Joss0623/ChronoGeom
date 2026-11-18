"""CPU-only verification and report artifacts for ClockTeacher03."""
from pathlib import Path
import argparse,csv,datetime as dt,json,re
from collections import Counter
from scripts.chronogeom.clock_teacher_lora01 import bootstrap,sha
from scripts.chronogeom.clock_teacher_circle03 import DAYS,COHORTS,targets
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def csvsave(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(r):
 import numpy as np,matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 bootstrap()
 import torch
 assert read(r/'process_exit.json')['exit_code']==0;done=read(r/'completion.json');c=read(r/'config.json');src=read(r/'source.json');rows=read(r/'examples.json');byid={x['id']:x for x in rows};test=[x for x in rows if x['split'] in COHORTS]
 out=Path('docs/chronogeom/clock_teacher_circle03');out.mkdir(exist_ok=False)
 assert sha(r/'examples.json')==src['examples_sha256'] and sha(r/'fixed_frame.npz')==src['frame_sha256']
 assert sha(Path('scripts/chronogeom/clock_teacher_circle03.py'))==src['script_sha256']
 old={tuple(x['identity']) for p in c['old_runs'] for x in read(Path(p)/'examples.json')}
 assert len(old)==1056 and len(rows)==816 and len({tuple(x['identity']) for x in rows})==816
 assert not ({tuple(x['identity']) for x in rows}&old)
 for x in rows:
  d=dt.datetime(2024,1,1)+dt.timedelta(days=x['day'],hours=x['hour']+x['delta']);assert (d.weekday(),d.hour)==(x['answer_day'],x['answer_hour'])
  assert (x['hour']+x['delta'])//24==x['answer_carry']
 pred=read(r/'test_predictions.json');assert len(pred)==480
 for x in pred:
  y=byid[x['id']];m=re.fullmatch(r'\s*([+-]\d+)\s+(\d{1,2})\s+('+'|'.join(DAYS)+r')\s*',x['text']);v=bool(m and int(m.group(2))<24)
  qok=bool(v and int(m.group(1))==y['answer_carry']);hok=bool(v and int(m.group(2))==y['answer_hour']);dok=bool(v and DAYS.index(m.group(3))==y['answer_day'])
  for k,z in dict(valid_format=v,carry_correct=qok,hour_correct=hok,day_correct=dok,clock_correct=hok and dok,correct=qok and hok and dok).items():assert x[k]==z
 curves=read(r/'validation_curve.json');locked=read(r/'test_lock.json');adapter=[]
 for arm in ['SFT','Circle']:
  selected=min([x for x in curves if x['arm']==arm],key=lambda x:(-x['correct'],x['ce'],x['step']));assert selected==locked['selected'][arm]
  for x in [x for x in curves if x['arm']==arm]:
   assert sha(Path(x['path']))==x['sha256'];t=torch.load(x['path'],map_location='cpu',weights_only=True);assert sum(v.numel() for v in t.values())==262144
   assert all(bool(torch.isfinite(v).all()) for v in t.values());adapter.append(dict(arm=arm,step=x['step'],sha256=x['sha256']))
 assert sha(c['warm_adapter'])==c['warm_adapter_sha256']
 cases=[]
 for p in pred:
  x=byid[p['id']];cases.append(dict(arm=p['arm'],id=p['id'],split=p['split'],source_id=p['source_id'],carry=x['answer_carry'],prompt=x['text'],expected=x['answer'].strip(),generated=p['text'],correct=p['correct'],clock_correct=p['clock_correct'],valid_format=p['valid_format'],carry_correct=p['carry_correct'],hour_correct=p['hour_correct'],day_correct=p['day_correct']))
 csvsave(out/'all_test_cases.csv',cases)
 hourstats=[]
 for arm in ['SFT','Circle']:
  for name,condition in [('hour_changes',lambda x:x['delta']%24!=0),('zero_shift',lambda x:x['delta']==0),('whole_day_nonzero',lambda x:x['delta']!=0 and x['delta']%24==0)]:
   rr=[p for p in pred if p['arm']==arm and condition(byid[p['id']])];hourstats.append(dict(arm=arm,group=name,n=len(rr),correct=sum(x['correct'] for x in rr),status='posthoc descriptive, no reselection'))
 save(out/'hour_change_diagnostic.json',hourstats)
 rng=np.random.default_rng(27102705);intervals=[]
 sf={p['id']:p for p in pred if p['arm']=='SFT'};ci={p['id']:p for p in pred if p['arm']=='Circle'}
 for group in ['all']+COHORTS+['carry','noncarry']:
  ids=[x['id'] for x in test if group=='all' or group==x['split'] or group=='carry' and x['carry'] or group=='noncarry' and not x['carry']]
  sources=sorted({byid[i]['source_id'] for i in ids});nn=np.array([sum(byid[i]['source_id']==s for i in ids) for s in sources]);dd=np.array([sum(int(ci[i]['correct'])-int(sf[i]['correct']) for i in ids if byid[i]['source_id']==s) for s in sources]);ix=rng.integers(len(sources),size=(2000,len(sources)));values=dd[ix].sum(1)/nn[ix].sum(1);lo,hi=np.quantile(values,[.025,.975]);intervals.append(dict(group=group,n=len(ids),sources=len(sources),circle_minus_sft=float(dd.sum()/nn.sum()),ci95_low=float(lo),ci95_high=float(hi)))
 csvsave(out/'source_cluster_bootstrap.csv',intervals)
 patch=read(r/'patch_predictions.json');assert len(patch)==240;ps=[]
 for arm in ['SFT','Circle']:
  for condition in ['Noop','FullNatural','CircleNatural','CircleRotate','RandomMatchedNorm']:
   rr=[x for x in patch if x['arm']==arm and x['condition']==condition];assert len(rr)==24
   if condition=='Noop':assert all(x['text']==(sf if arm=='SFT' else ci)[x['id']]['text'] and x['perturbation_norm']==0 for x in rr)
   ps.append(dict(arm=arm,condition=condition,n=24,sources=len({x['source_id'] for x in rr}),original_correct=sum(x['original_correct'] for x in rr),counterfactual_correct=sum(x['counterfactual_correct'] for x in rr),mean_cf_ld=float(np.mean([x['cf_logit_difference'] for x in rr])),mean_perturbation=float(np.mean([x['perturbation_norm'] for x in rr])),mean_activation_norm=float(np.mean([x['activation_norm'] for x in rr])),mean_nearest_canonical_distance=float(np.mean([x['nearest_canonical_distance'] for x in rr]))))
  a=[x for x in patch if x['arm']==arm and x['condition']=='CircleNatural'];b=[x for x in patch if x['arm']==arm and x['condition']=='RandomMatchedNorm'];assert all(abs(x['perturbation_norm']-y['perturbation_norm'])<1e-4 for x,y in zip(a,b))
 csvsave(out/'patch_summary.csv',ps);csvsave(out/'all_patch_cases.csv',patch)
 energies=[];frame=np.load(r/'fixed_frame.npz');query=np.load(r/'query_calibration.npz')
 for arm in ['SFT','Circle']:
  h=np.load(r/f'hidden_{arm}.npz')
  for slot,mu,scale in [('source',frame['source_mean'],frame['source_scale']),('query',query['mean'],query['scale'])]:
   z=((h[slot].astype('float32')-mu)@frame['q'])/scale.repeat(2)/np.sqrt(2)
   energies.append(dict(arm=arm,slot=slot,mean_daily_squared_norm=float(np.mean(np.sum(z[:,:2]**2,axis=1))),mean_weekly_squared_norm=float(np.mean(np.sum(z[:,2:]**2,axis=1))),effective_rank=int(np.linalg.matrix_rank(z-z.mean(0))),dimension=4))
 csvsave(out/'representation_energy.csv',energies)
 train=read(r/'train64_predictions.json');trainstats=[dict(arm=a,n=64,correct=sum(x['correct'] for x in train if x['arm']==a),clock_correct=sum(x['clock_correct'] for x in train if x['arm']==a),valid_format=sum(x['valid_format'] for x in train if x['arm']==a)) for a in ['SFT','Circle']]
 save(out/'train64.json',trainstats)
 for n in ['dataset_counts.csv','metrics.csv','geometry.csv','train_curve.csv']: (out/n).write_bytes((r/n).read_bytes())
 save(out/'validation_curve.json',curves)
 for n in ['preflight.json','completion.json','source.json','test_lock.json']:(out/n).write_bytes((r/n).read_bytes())
 assert done['active_seconds']<600 and done['load_seconds']<300 and done['peak_gib']<13 and done['disk_bytes']<50*2**20
 assert len(read(r/'analytical_sincos.json'))==240 and all(x['correct'] for x in read(r/'analytical_sincos.json'))
 verified=dict(dataset_datetime_all816=True,all1056_previous_identities_excluded=True,predictions_reparsed=True,all_adapters_finite=adapter,all48_noop_texts_exact=True,random_norms_matched=True,analytical_sincos_correct=240,train64=trainstats,bootstrap_seed=27102705,bootstrap_draws=2000,bootstrap_unit='source_id; conditional on one training seed, fixed generator',script_sha256=src['script_sha256'])
 save(out/'verification.json',verified)
 with plt.rc_context({'font.size':10,'pdf.fonttype':42,'figure.facecolor':'white'}):
  fig,ax=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
  for arm,color,marker in [('SFT','#0072B2','o'),('Circle','#D55E00','s')]:
   vv=[x for x in curves if x['arm']==arm];ax[0].plot([x['step'] for x in vv],[100*x['correct']/64 for x in vv],marker=marker,color=color,label=arm)
   ax[1].plot([x['step'] for x in vv],[x['ce'] for x in vv],marker=marker,color=color,label=arm)
  ax[0].set(xlabel='Optimizer updates',ylabel='Validation complete answer accuracy (%)',ylim=(0,100),title='64 fixed validation questions')
  ax[1].set(xlabel='Optimizer updates',ylabel='Validation answer-token CE',ylim=(0,max(x['ce'] for x in curves)*1.15),title='Secondary selection criterion')
  for a in ax:a.set_xticks(c['validation_steps']);a.legend();a.grid(alpha=.2)
  fig.savefig(out/'validation_curve.png',dpi=180,facecolor='white');fig.savefig(out/'validation_curve.pdf',facecolor='white');plt.close(fig)
 save(out/'figure_provenance.json',dict(source=str(r/'validation_curve.json'),transform='100*correct/64',smoothing=None,uncertainty='none: fixed validation set and one training seed',dimensions_inches=[9,3.5],png_dpi=180,matplotlib=matplotlib.__version__,alt_text='Both complete validation checkpoints for SFT and Circle are shown; accuracy and CE panels; no interpolation beyond connecting observed checkpoints.'))
 print(json.dumps(dict(metrics=list(csv.DictReader((r/'metrics.csv').open())),train64=trainstats,bootstrap=intervals,patch=ps,geometry=list(csv.DictReader((r/'geometry.csv').open())),completion=done),ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);main(p.parse_args().run)
