"""Same-budget Shape output KD; reuse event19 frozen teacher predictions/TSLib head."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
import argparse,copy,time,sys
from pathlib import Path
import numpy as np
import torch
from exp.exp_chronogeom_event_knowledge19 import data,Head,forward,scores,read,save,sha,git
from exp.exp_chronogeom_residual03 import numeric_summary

def center(x):return x-x.mean(1,keepdim=True)
def run(c,config):
 start=time.monotonic();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 r=Path(c['run']);r.mkdir(parents=True,exist_ok=False);old=Path(c['teacher_run']);tc=read(old/'config.json')
 save(r/'config.json',c);save(r/'provenance.json',dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),python=sys.version,torch=torch.__version__,numpy=np.__version__))
 files=['predictions.npz','normalizers.json','all_checkpoints.pt','selection.json','knowledge.npz','config.json','frozen_assets.json']
 inputs={n:sha(old/n) for n in files};assets=read(old/'frozen_assets.json');save(r/'input_assets.json',dict(teacher_files=inputs,base_checkpoints=assets))
 for a in assets:assert sha(a['path'])==a['sha256']
 packs,grid,raw,common,kind,h,hol,hist,sd=data(tc,r)
 cached=np.load(old/'predictions.npz');norms=read(old/'normalizers.json');K=np.load(old/'knowledge.npz')['Fourier'];oldstates=torch.load(old/'all_checkpoints.pt',map_location='cpu',weights_only=True)
 base={};z={};teachers={};states={};curves=[];checks=[];batches={}
 for seed in c['seeds']:
  base[seed]={};z[seed]={};teachers[seed]={};states[seed]={}
  prep=norms[str(seed)]['Fourier'];mu=np.array(prep['mean']);sigma=np.array(prep['sd']);km=np.array(prep['knowledge_mean']);ks=prep['knowledge_rms']
  for split,p in packs.items():
   assert np.array_equal(p['origin'],cached[f'{seed}_{split}_origin']) and np.array_equal(p['y'],cached[f'{seed}_{split}_y'])
   bp=cached[f'{seed}_{split}_FrozenTimeBase'];base[seed][split]=torch.tensor(bp,dtype=torch.float32)
   ix=p['future'];ns=numeric_summary(p['x'],base[seed][split]).numpy();ft=np.concatenate([ns,raw[ix],common[ix]],-1)
   z[seed][split]=torch.tensor(np.concatenate([(ft-mu)/sigma,(K[ix]-km)/ks],-1),dtype=torch.float32)
  save(r/f'student_preprocessing_seed{seed}.json',prep)
  target=torch.tensor(packs['fit']['y']);bp=base[seed]['fit'];norm=scores(bp.numpy(),packs['fit']['y'])
  batches[seed]=np.random.default_rng(seed).integers(0,len(target),size=(c['steps'],c['batch']))
  for arm,teacher in c['arms'].items():
   head=Head(seed);optimizer=torch.optim.Adam(head.parameters(),lr=c['lr']);states[seed][arm]={}
   initial=forward(head,z[seed]['fit'],bp);assert torch.equal(initial,bp.double())
   loss=(initial-target).square().mean();loss.backward();grad=float(head.net[-1].weight.grad.abs().sum());assert grad>0;head.zero_grad()
   teach=None if teacher is None else center(torch.tensor(cached[f'{seed}_fit_{teacher}'],dtype=torch.float64)).detach()
   if teach is not None:assert not teach.requires_grad
   teachers[seed][arm]=teach
   for step in range(c['steps']+1):
    if step%100==0:
     with torch.no_grad():pv=forward(head,z[seed]['val'],base[seed]['val']).numpy()
     curves.append(dict(seed=seed,arm=arm,step=step,**scores(pv,packs['val']['y'])));states[seed][arm][step]=copy.deepcopy(head.state_dict())
     if arm=='NoKD':
      for key,value in head.state_dict().items():assert torch.equal(value,oldstates[seed]['Fourier'][step][key]),('NoKD must exactly reproduce19',seed,step,key)
    if step==c['steps']:break
    ids=batches[seed][step];pred=forward(head,z[seed]['fit'][ids],bp[ids]);e=pred-target[ids]
    sup=.5*(e.square().mean()/norm['mse']+e.abs().mean()/norm['mae'])
    kd=pred.new_zeros(()) if teach is None else (center(pred)-teach[ids]).square().mean()/norm['mse']
    loss=sup+c['kd_weight']*kd;optimizer.zero_grad();loss.backward();optimizer.step()
    assert time.monotonic()-start<c['fit_seconds']
   checks.append(dict(seed=seed,arm=arm,parameters=sum(p.numel() for p in head.parameters()),initial_prediction_error=0.,initial_gradient_l1=grad,updates=c['steps'],same_student_inputs=True,teacher_requires_grad=False))
   print('TRAINED',seed,arm,flush=True)
 # No evaluation metrics enter this seven-checkpoint selection.
 chosen={};choices=[]
 for arm in c['arms']:
  opts=[]
  for step in range(0,c['steps']+1,100):
   vals=[]
   for seed in c['seeds']:
    q=next(q for q in curves if q['seed']==seed and q['arm']==arm and q['step']==step);bm=scores(base[seed]['val'].numpy(),packs['val']['y']);vals.append(.5*(q['mse']/bm['mse']+q['mae']/bm['mae']))
   opts.append((float(np.mean(vals)),step))
  chosen[arm]=min(opts)[1];choices.append(dict(arm=arm,selection_candidates=[dict(step=step,joint_score=score) for score,step in opts]))
 save(r/'selection.json',dict(chosen=chosen,criteria=choices,git=git(),note='Saved before outer metric evaluation; Capital tail is previously viewed development. No teacher/hyperparameter selection from outer outcomes.'))
 save(r/'curves.json',curves);torch.save(states,r/'all_checkpoints.pt')
 predictions={};metrics=[];subgroups=[];replays=0;maxmean=0.;maxlevel=0.;maxmetric=0.
 for seed in c['seeds']:
  for split,p in packs.items():
   predictions[f'{seed}_{split}_y']=p['y'];predictions[f'{seed}_{split}_origin']=p['origin'];out={}
   for name in ['OriginalTimeBase','FrozenTimeBase']:out[name]=cached[f'{seed}_{split}_{name}']
   for arm in c['arms']:
    head=Head(seed);head.load_state_dict(states[seed][arm][chosen[arm]])
    with torch.no_grad():pred=forward(head,z[seed][split],base[seed][split]).numpy()
    out[arm]=pred;meanerr=float(abs((pred-base[seed][split].numpy()).mean(1)).max());maxmean=max(maxmean,meanerr);assert meanerr<1e-10
    replay=Head(seed);replay.load_state_dict(states[seed][arm][chosen[arm]])
    with torch.no_grad():assert np.array_equal(pred,forward(replay,z[seed][split],base[seed][split]).numpy());replays+=1
    if arm=='NoKD':assert chosen[arm]==read(old/'selection.json')['chosen']['Fourier'] and np.array_equal(pred,cached[f'{seed}_{split}_Fourier'])
   ix=p['future'];hh=h[ix];kk=kind[ix];groups=dict(all=np.ones(ix.shape,bool),workday=kk==0,weekend=kk==1,holiday=kk==2,morning7_9=(hh>=7)&(hh<=9),evening16_19=(hh>=16)&(hh<=19))
   for arm,pred in out.items():
    predictions[f'{seed}_{split}_{arm}']=pred;met=scores(pred,p['y']);metrics.append(dict(seed=seed,split=split,arm=arm,step=chosen.get(arm),**met))
    e=pred-p['y'];independent=float(e.ravel()@e.ravel()/e.size);maxmetric=max(maxmetric,abs(independent-met['mse']));assert abs(met['mse']-met['level']-met['shape'])<1e-12
    if arm in c['arms']:maxlevel=max(maxlevel,abs(met['level']-scores(base[seed][split].numpy(),p['y'])['level']))
    for group,mask in groups.items():
     err=e[...,0][mask]
     if len(err):subgroups.append(dict(seed=seed,split=split,arm=arm,group=group,unique_hours=int(len(np.unique(ix[mask]))),windows=int(mask.any(1).sum()),mse=float((err**2).mean()),mae=float(abs(err).mean())))
 np.savez_compressed(r/'predictions.npz',**predictions);save(r/'metrics.json',metrics);save(r/'subgroups.json',subgroups)
 for n,hashvalue in inputs.items():assert sha(old/n)==hashvalue
 for a in assets:assert sha(a['path'])==a['sha256']
 allarms=['OriginalTimeBase','FrozenTimeBase']+list(c['arms'])
 summary={split:{arm:{k:float(np.mean([q[k] for q in metrics if q['split']==split and q['arm']==arm])) for k in ['mse','mae','level','shape']} for arm in allarms} for split in packs}
 # Average seeds before calendar-block bootstrap; never pool dependent windows/seeds as IID.
 uncertainty=[];quality=[];rng=np.random.default_rng(2024)
 for split in ['val','test']:
  origins=packs[split]['origin'];clusters=(origins-origins.min())//(14*24);unique=np.unique(clusters)
  for ref in ['OriginalTimeBase','FrozenTimeBase','NoKD','FourierKD','HistoricalKD','ShuffleKD']:
   for metric in ['mse','mae']:
    dd=[]
    for seed in c['seeds']:
     y=predictions[f'{seed}_{split}_y'];a=predictions[f'{seed}_{split}_QwenTaskKD']-y;b=predictions[f'{seed}_{split}_{ref}']-y
     dd.append(((a*a-b*b) if metric=='mse' else (abs(a)-abs(b))).mean((1,2)))
    diff=np.mean(dd,0);boot=[]
    for _ in range(2000):
     idx=np.concatenate([np.flatnonzero(clusters==k) for k in rng.choice(unique,len(unique),replace=True)]);boot.append(float(diff[idx].mean()))
    uncertainty.append(dict(split=split,reference=ref,metric=metric,task_minus_reference=float(diff.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),calendar_clusters=len(unique),interpretation='Exploratory only: few clusters, repeated windows; seeds averaged first; no independent confirmation'))
  for seed in c['seeds']:
   y=predictions[f'{seed}_{split}_y'];p=predictions[f'{seed}_{split}_QwenTaskKD'];b=predictions[f'{seed}_{split}_FrozenTimeBase'];delta=((p-y)**2-(b-y)**2).mean((1,2))
   quality.append(dict(seed=seed,split=split,improved=int((delta<0).sum()),worsened=int((delta>0).sum()),mean_savings=float(np.maximum(-delta,0).mean()),mean_damage=float(np.maximum(delta,0).mean())))
 save(r/'summary.json',dict(metrics=summary,paired_intervals=uncertainty,task_window_corrections=quality))
 save(r/'verification.json',dict(per_fit=checks,noKD_all21_checkpoints_exact_to19=True,noKD_all9_forecasts_exact_to19=True,prediction_replay_cases=replays,mean_error_max=maxmean,level_error_difference_max=maxlevel,independent_metric_error_max=maxmetric,source_teachers_and_6_base_assets_unchanged=True,teacher_train_predictions='Standard in-sample teacher soft targets; not OOF teacher estimates; validation/test targets not distilled',student_inference='No Qwen/teacher table; shared analytical features and saved head only'))
 save(r/'status.json',dict(seconds=time.monotonic()-start,heads=15,steps_each=c['steps'],llm_forwards=0,llm_updates=0,timebase_updates=0,git=git()))
 print('CHOSEN',chosen,flush=True);print('METRICS',summary,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args();run(read(a.config),a.config)
