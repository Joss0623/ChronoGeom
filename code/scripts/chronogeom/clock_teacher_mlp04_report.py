"""CPU-only paired analysis for MLP04. No training or inference."""
from pathlib import Path
import argparse,csv,datetime as dt,json,re
from collections import Counter
from scripts.chronogeom.clock_teacher_lora01 import bootstrap,sha
from scripts.chronogeom.clock_teacher_circle03 import DAYS,COHORTS
ARMS=['QV4_cached','QV14','QV_MLP4']
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvsave(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def groups(x,g):
 return (g=='all' or x['split']==g or g=='hour_change' and x['delta']%24!=0 or
 g=='carry' and x['carry'] or g=='noncarry' and not x['carry'] or
 g=='hour_change_carry' and x['delta']%24!=0 and x['carry'] or
 g=='hour_change_noncarry' and x['delta']%24!=0 and not x['carry'])
GROUPS=['all','hour_change','carry','hour_change_carry','hour_change_noncarry']+COHORTS+['noncarry']
def main(r):
 import numpy as np,matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 bootstrap()
 import torch
 c=read(r/'config.json');src=read(r/'source.json');old=Path(c['previous_run']);done=read(r/'completion.json');assert read(r/'process_exit.json')['exit_code']==0
 out=Path('docs/chronogeom/clock_teacher_mlp04');out.mkdir(exist_ok=False)
 for name in ['examples.json','fixed_frame.npz','query_calibration.npz','dataset_counts.csv','analytical_sincos.json']:assert sha(r/name)==sha(old/name)
 assert sha('scripts/chronogeom/clock_teacher_mlp04.py')==src['script_sha256'];assert sha('scripts/chronogeom/clock_teacher_circle03.py')==src['helper_script_sha256']
 assert c['fixed_aux_weights']==read(old/'preflight.json')['fixed_aux_weights']==read(r/'preflight.json')['fixed_aux_weights']
 rows=read(r/'examples.json');byid={x['id']:x for x in rows};test=[x for x in rows if x['split'] in COHORTS];assert len(test)==240
 for x in rows:
  t=dt.datetime(2024,1,1)+dt.timedelta(days=x['day'],hours=x['hour']+x['delta']);assert (t.weekday(),t.hour)==(x['answer_day'],x['answer_hour']);assert (x['hour']+x['delta'])//24==x['answer_carry']
 pred=[dict(x,arm='QV4_cached') for x in read(old/'test_predictions.json') if x['arm']=='Circle']+read(r/'test_predictions.json');assert len(pred)==720
 for p in pred:
  x=byid[p['id']];m=re.fullmatch(r'\s*([+-]\d+)\s+(\d{1,2})\s+('+'|'.join(DAYS)+r')\s*',p['text']);v=bool(m and int(m.group(2))<24);qok=bool(v and int(m.group(1))==x['answer_carry']);hok=bool(v and int(m.group(2))==x['answer_hour']);dok=bool(v and DAYS.index(m.group(3))==x['answer_day'])
  for k,z in dict(valid_format=v,carry_correct=qok,hour_correct=hok,day_correct=dok,correct=qok and hok and dok,clock_correct=hok and dok).items():assert p[k]==z
  p['hour_carry_correct']=qok and hok;p['day_carry_correct']=qok and dok
 csvsave(out/'all_test_cases.csv',[dict(arm=p['arm'],id=p['id'],split=p['split'],source_id=p['source_id'],delta=byid[p['id']]['delta'],carry=byid[p['id']]['answer_carry'],prompt=byid[p['id']]['text'],expected=p['expected'],generated=p['text'],**{k:p[k] for k in ['correct','clock_correct','valid_format','carry_correct','hour_correct','day_correct','hour_carry_correct','day_carry_correct']}) for p in pred])
 success=[]
 for p in pred:
  x=byid[p['id']]
  if p['arm']=='QV_MLP4' and p['correct'] and x['delta']%24!=0:success.append(dict(id=p['id'],prompt=x['text'],expected=p['expected'],generated=p['text'],delta=x['delta'],carry=x['answer_carry']))
 save(out/'all_nontrivial_successes.json',success)
 summary=[]
 for arm in ARMS+['SinCos']:
  for g in GROUPS:
   rr=[x for x in test if groups(x,g)];pp=[p for p in pred if p['arm']==arm and groups(byid[p['id']],g)]
   summary.append(dict(arm=arm,group=g,n=len(rr),sources=len({x['source_id'] for x in rr}),**{k:len(rr) if arm=='SinCos' else sum(p[k] for p in pp) for k in ['correct','clock_correct','valid_format','carry_correct','hour_correct','day_correct','hour_carry_correct','day_carry_correct']}))
 csvsave(out/'summary.csv',summary)
 curve=read(r/'validation_curve.json');lock=read(r/'test_lock.json');adapter_checks=[]
 for arm in ARMS[1:]:
  chosen=min([x for x in curve if x['arm']==arm],key=lambda x:(-x['correct'],x['ce'],x['step']));assert chosen==lock['selected'][arm]
  initial=torch.load(r/f'initial_{arm}.pt',map_location='cpu',weights_only=True)
  for x in [x for x in curve if x['arm']==arm]:
   t=torch.load(x['path'],map_location='cpu',weights_only=True);assert sha(x['path'])==x['sha256'];assert sum(v.numel() for v in t.values())==917504;assert all(bool(torch.isfinite(v).all()) for v in t.values())
  t=torch.load(chosen['path'],map_location='cpu',weights_only=True);norms={}
  for group in ['qv','mlp']:
   vals=[float(((t[k]-v).float()**2).sum()) for k,v in initial.items() if ('fc_' in k)==(group=='mlp')];norms[group]=float(np.sqrt(sum(vals))) if vals else 0.
  assert norms['qv']>0
  if arm=='QV_MLP4':assert norms['mlp']>0
  adapter_checks.append(dict(arm=arm,selected_sha256=chosen['sha256'],trainable_parameters=917504,tensors=len(t),update_l2=norms))
 pre=read(r/'preflight.json');assert pre['weights_reused_exact'] and pre['query_calibration_reused_exact']
 gmlp=next(x for x in pre['checks'] if x['arm']=='QV_MLP4')['group_gradient_norms'];assert all(gmlp[k]>0 for k in ['ce_mlp','state_mlp','rotation_mlp'])
 rng=np.random.default_rng(27102706);boot=[];paired=[]
 for baseline in ['QV14','QV4_cached']:
  aa={p['id']:p for p in pred if p['arm']==baseline};bb={p['id']:p for p in pred if p['arm']=='QV_MLP4'}
  pair=Counter((aa[i]['correct'],bb[i]['correct']) for i in aa);paired.append(dict(baseline=baseline,both_correct=pair[(True,True)],baseline_only=pair[(True,False)],mlp_only=pair[(False,True)],both_wrong=pair[(False,False)]))
  for group in ['all','hour_change','carry','hour_change_carry']:
   ids=[x['id'] for x in test if groups(x,group)];sources=sorted({byid[i]['source_id'] for i in ids});den=np.array([sum(byid[i]['source_id']==s for i in ids) for s in sources])
   ix=rng.integers(len(sources),size=(2000,len(sources)))
   for metric in ['correct','hour_correct','carry_correct','hour_carry_correct']:
    num=np.array([sum(int(bb[i][metric])-int(aa[i][metric]) for i in ids if byid[i]['source_id']==s) for s in sources]);z=num[ix].sum(1)/den[ix].sum(1);lo,hi=np.quantile(z,[.025,.975]);boot.append(dict(baseline=baseline,group=group,metric=metric,n=len(ids),sources=len(sources),mlp_minus_baseline=float(num.sum()/den.sum()),ci95_low=float(lo),ci95_high=float(hi)))
 csvsave(out/'source_cluster_bootstrap.csv',boot);save(out/'paired_correctness.json',paired)
 patch=read(r/'patch_predictions.json');assert len(patch)==240;ps=[]
 for arm in ARMS[1:]:
  oldtext={p['id']:p['text'] for p in pred if p['arm']==arm};npred={p['id']:p for p in patch if p['arm']==arm and p['condition']=='Noop'}
  assert len(npred)==24 and all(p['text']==oldtext[p['id']] and p['perturbation_norm']==0 for p in npred.values())
  for cond in ['Noop','FullNatural','CircleNatural','CircleRotate','RandomMatchedNorm']:
   rr=[p for p in patch if p['arm']==arm and p['condition']==cond];assert len(rr)==24
   ps.append(dict(arm=arm,condition=cond,n=24,original_correct=sum(p['original_correct'] for p in rr),cf_correct=sum(p['counterfactual_correct'] for p in rr),output_changes=sum(p['text']!=npred[p['id']]['text'] for p in rr),mean_cf_ld=float(np.mean([p['cf_logit_difference'] for p in rr])),mean_perturbation=float(np.mean([p['perturbation_norm'] for p in rr]))))
  rr=[p for p in patch if p['arm']==arm and p['condition']=='CircleNatural'];zz=[p for p in patch if p['arm']==arm and p['condition']=='RandomMatchedNorm'];assert all(abs(a['perturbation_norm']-b['perturbation_norm'])<1e-4 for a,b in zip(rr,zz))
 csvsave(out/'patch_summary.csv',ps);csvsave(out/'all_patch_cases.csv',patch)
 tr=[dict(x,arm='QV4_cached') for x in read(old/'train64_predictions.json') if x['arm']=='Circle']+read(r/'train64_predictions.json')
 trainstats=[]
 for arm in ARMS:
  pp=[p for p in tr if p['arm']==arm];assert len(pp)==64;trainstats.append(dict(arm=arm,n=64,correct=sum(p['correct'] for p in pp),hour_correct=sum(p['hour_correct'] for p in pp),carry_correct=sum(p['carry_correct'] for p in pp),valid_format=sum(p['valid_format'] for p in pp)))
 save(out/'train64.json',trainstats)
 allgeo=[]
 for root,isold in [(old,True),(r,False)]:
  for x in csv.DictReader((root/'geometry.csv').open()):
   if isold and x['arm']!='Circle':continue
   if isold:x['arm']='QV4_cached'
   allgeo.append(x)
 csvsave(out/'geometry.csv',allgeo)
 allcurve=[dict(x,arm='QV4_cached') for x in read(old/'validation_curve.json') if x['arm']=='Circle']+curve;save(out/'validation_curve.json',allcurve)
 for name in ['dataset_counts.csv','train_curve.csv','preflight.json','completion.json','source.json','test_lock.json','packages.json','initialization.json']:(out/name).write_bytes((r/name).read_bytes())
 assert done['active_seconds']<600 and done['load_seconds']<300 and done['peak_gib']<13 and done['disk_bytes']<50*2**20
 oracle=read(r/'analytical_sincos.json');assert len(oracle)==240 and all(x['correct'] for x in oracle)
 save(out/'verification.json',dict(copied_data_frame_query_exact=True,fixed_loss_weights_exact=True,all816_datetime_labels_verified=True,all720_outputs_reparsed=True,adapter_checks=adapter_checks,all48_noop_outputs_exact=True,random_norms_matched=True,analytical_correct=240,cached_qv4_predictions_sha256=sha(old/'test_predictions.json'),bootstrap_seed=27102706,bootstrap_draws=2000,bootstrap_unit='source_id; descriptive conditional on single seed and already-viewed pilot',script_sha256=src['script_sha256']))
 with plt.rc_context({'font.size':10,'pdf.fonttype':42,'figure.facecolor':'white'}):
  fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
  for arm,color,marker in [('QV4_cached','#666666','^'),('QV14','#0072B2','o'),('QV_MLP4','#D55E00','s')]:
   vv=[x for x in allcurve if x['arm']==arm];axes[0].plot([x['step'] for x in vv],[100*x['correct']/64 for x in vv],color=color,marker=marker,label=arm);axes[1].plot([x['step'] for x in vv],[x['ce'] for x in vv],color=color,marker=marker,label=arm)
  axes[0].set(xlabel='Updates',ylabel='Full-answer validation accuracy (%)',ylim=(0,100),title='Fixed 64 validation questions')
  axes[1].set(xlabel='Updates',ylabel='Answer-token CE',ylim=(0,1.15*max(x['ce'] for x in allcurve)),title='Same Circle loss and training budget')
  for ax in axes:ax.set_xticks([256,384]);ax.grid(alpha=.2);ax.legend(fontsize=8)
  fig.savefig(out/'validation_curve.png',dpi=180,facecolor='white');fig.savefig(out/'validation_curve.pdf',facecolor='white');plt.close(fig)
 save(out/'figure_provenance.json',dict(source=[str(old/'validation_curve.json'),str(r/'validation_curve.json')],normalization='100*correct/64',smoothing=None,uncertainty='none; one seed and fixed validation examples',size_inches=[9,3.5],png_dpi=180,alt_text='Actual validation checkpoints for cached QV4 and newly trained equal-parameter QV14 and QV-plus-MLP4.'))
 print(json.dumps(dict(summary=summary,train64=trainstats,paired=paired,bootstrap=boot,patch=ps,adapter_checks=adapter_checks,completion=done),ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);main(p.parse_args().run)
