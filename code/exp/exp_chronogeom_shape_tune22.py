"""Bounded inner-validation tuning of the unchanged Shape KD20/21 forecaster."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
import argparse,copy,time,itertools,sys
from pathlib import Path
import numpy as np
import torch
from exp.exp_chronogeom_event_knowledge19 import Head,forward,scores,read,save,sha,git,data as capital_data
from exp.exp_chronogeom_event_confirm21 import dataset as metro_data
from exp.exp_chronogeom_residual03 import numeric_summary

def center(x):return x-x.mean(1,keepdim=True)
def run(c,config):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);start=time.monotonic();r=Path(c['run']);r.mkdir(parents=True,exist_ok=False)
 save(r/'config.json',c);save(r/'provenance.json',dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),python=sys.version,torch=torch.__version__,numpy=np.__version__))
 ds={};inputs=[];assets=[];curves=[];index={};states={};inits=[];reproduction_counts=dict(SUP=0,KD=0)
 for name,spec in c['domains'].items():
  src=Path(spec['teacher_run']);ref=Path(spec['reference_run']);folder=r/name;folder.mkdir();cfg=read(src/'config.json')
  for p in [src/'config.json',src/'normalizers.json',src/'knowledge.npz',src/'data.npz',src/'predictions.npz',ref/spec['reference_states'],ref/'predictions.npz',ref/'selection.json']:inputs.append(dict(path=str(p),sha=sha(p)))
  if spec['adapter']=='Capital':packs,grid,raw,common,kind,h,hol,hist,sd=capital_data(cfg,folder);source_assets=read(src/'frozen_assets.json');assets.extend(dict(path=a['path'],sha=a['sha256']) for a in source_assets)
  else:packs,grid,raw,common,kind,h,hist=metro_data(cfg,folder);assets.extend(dict(path=a['path'],sha=a['sha']) for a in read(src/'base_assets.json'))
  cached=np.load(src/'predictions.npz');refpred=np.load(ref/'predictions.npz');normalizers=read(src/'normalizers.json');K=np.load(src/'knowledge.npz')['Fourier']
  d=dict(spec=spec,p=packs,raw=raw,common=common,kind=kind,h=h,grid=grid,K=K,prep=normalizers,cache=cached,refs=refpred,z={},base={},target=torch.tensor(packs['fit']['y']),norm={},valnorm={},batches={},teacher={},oldstates=torch.load(ref/spec['reference_states'],map_location='cpu',weights_only=True))
  for seed in c['seeds']:
   d['z'][seed]={};d['base'][seed]={};pp=normalizers[str(seed)]['Fourier'];mu=np.array(pp['mean']);scale=np.array(pp['sd']);km=np.array(pp['knowledge_mean']);ks=pp['knowledge_rms']
   for split in ['fit','val']:
    p=packs[split];assert np.array_equal(p['origin'],cached[f'{seed}_{split}_origin']) and np.array_equal(p['y'],cached[f'{seed}_{split}_y'])
    bp=torch.tensor(cached[f'{seed}_{split}_FrozenTimeBase'],dtype=torch.float32);d['base'][seed][split]=bp
    ix=p['future'];ft=np.concatenate([numeric_summary(p['x'],bp).numpy(),raw[ix],common[ix]],-1);d['z'][seed][split]=torch.tensor(np.concatenate([(ft-mu)/scale,(K[ix]-km)/ks],-1),dtype=torch.float32)
   d['norm'][seed]=scores(d['base'][seed]['fit'].numpy(),packs['fit']['y']);d['valnorm'][seed]=scores(d['base'][seed]['val'].numpy(),packs['val']['y']);d['batches'][seed]=np.random.default_rng(seed).integers(0,len(packs['fit']['y']),size=(600,c['batch']))
   d['teacher'][seed]={a:center(torch.tensor(cached[f"{seed}_fit_{spec['teacher_prefix']}{t}"],dtype=torch.float64)).detach() for a,t in c['arms'].items()}
  ds[name]=d;states[name]={}
 for p in assets:assert sha(p['path'])==p['sha']
 save(r/'input_assets.json',dict(files=inputs,bases=assets))
 def train(name,arm,key,seed,lr,weight=0.,kdstart=0):
  d=ds[name];head=Head(seed);opt=torch.optim.Adam(head.parameters(),lr=lr);z=d['z'][seed];bp=d['base'][seed]['fit'];norm=d['norm'][seed];target=d['target'];teacher=None if arm=='NoKD' else d['teacher'][seed][arm]
  init=forward(head,z['fit'],bp);assert torch.equal(init,bp.double());loss=(init-target).square().mean();loss.backward();grad=float(head.net[-1].weight.grad.abs().sum());assert grad>0;head.zero_grad();out={}
  match=lr==.001 and (arm=='NoKD' or (weight==.1 and kdstart==0))
  for step in range(601):
   if step%100==0:
    with torch.no_grad():pv=forward(head,z['val'],d['base'][seed]['val']).numpy()
    met=scores(pv,d['p']['val']['y']);q=dict(domain=name,arm=arm,recipe=key,seed=seed,step=step,**met);curves.append(q);index[name,arm,key,seed,step]=q;out[step]=copy.deepcopy(head.state_dict())
    if match:
     for k,v in head.state_dict().items():assert torch.equal(v,d['oldstates'][seed][arm][step][k]),('reproduction',name,arm,seed,step,k)
     reproduction_counts['SUP' if arm=='NoKD' else 'KD']+=1
   if step==600:break
   ids=d['batches'][seed][step];pred=forward(head,z['fit'][ids],bp[ids]);err=pred-target[ids];loss=.5*(err.square().mean()/norm['mse']+err.abs().mean()/norm['mae'])
   if teacher is not None and step>=kdstart:
    aux=(center(pred)-teacher[ids]).square().mean()
    # Preserve source20/21 floating-point association for exact anchor replay.
    loss=loss+weight*(aux/norm['mse']) if d['spec']['adapter']=='Capital' else loss+weight*aux/norm['mse']
   opt.zero_grad();loss.backward();opt.step();assert time.monotonic()-start<c['fit_seconds'],'bounded CPU budget exceeded'
  inits.append(dict(domain=name,arm=arm,recipe=key,seed=seed,lr=lr,weight=weight,start=kdstart,steps=600,params=sum(p.numel() for p in head.parameters()),initial_gradient=grad,initial_prediction_error=0.))
  return out
 def choose(arm,key):
  result={}
  for name,d in ds.items():
   options=[]
   for step in range(0,601,100):
    v=[]
    for seed in c['seeds']:
     q=index[name,arm,key,seed,step];bn=d['valnorm'][seed];v.append(.5*(q['mse']/bn['mse']+q['mae']/bn['mae']))
    options.append((float(np.mean(v)),step))
   value,step=min(options);result[name]=dict(score=value,step=step)
  return dict(global_score=float(np.mean([v['score'] for v in result.values()])),domains=result)
 supoptions={}
 for lr in c['lrs']:
  key=f'lr{lr:g}'
  for name in ds:
   states[name].setdefault('NoKD',{})[key]={seed:train(name,'NoKD',key,seed,lr) for seed in c['seeds']}
  supoptions[key]=dict(lr=lr,**choose('NoKD',key));print('SUP',key,supoptions[key],flush=True)
 supkey=min(supoptions,key=lambda k:(supoptions[k]['global_score'],supoptions[k]['lr']));lr=supoptions[supkey]['lr'];save(r/'learning_rate_selection.json',dict(chosen=supkey,lr=lr,candidates=supoptions,outer_scores_seen=False))
 kdoptions={a:{} for a in c['arms']}
 for weight,kdstart in itertools.product(c['kd_weights'],c['kd_starts']):
  key=f'kd{weight:g}_start{kdstart}'
  for arm in c['arms']:
   for name in ds:states[name].setdefault(arm,{})[key]={seed:train(name,arm,key,seed,lr,weight,kdstart) for seed in c['seeds']}
   kdoptions[arm][key]=dict(weight=weight,start=kdstart,lr=lr,**choose(arm,key));print('KD',arm,key,kdoptions[arm][key],flush=True)
 chosen={'NoKD':dict(recipe=supkey,lr=lr,weight=0.,start=0,**choose('NoKD',supkey))}
 for arm in c['arms']:
  key=min(kdoptions[arm],key=lambda k:(kdoptions[arm][k]['global_score'],kdoptions[arm][k]['weight'],kdoptions[arm][k]['start']));chosen[arm]=dict(recipe=key,**kdoptions[arm][key])
 save(r/'selection.json',dict(chosen=chosen,lr_candidates=supoptions,kd_candidates=kdoptions,git=git(),outer_scores_seen=False,note='Both old evaluation periods are development only. All recipes use INNER validation only, one global recipe per arm across both domains.'))
 lock=sha(r/'selection.json');save(r/'evaluation_lock.json',dict(selection_sha=lock,source_sha=sha(__file__)));save(r/'curves.json',curves);torch.save(states,r/'all_trial_checkpoints.pt');save(r/'fits.json',inits)
 # Evaluate only preselected recipes after the lock; no candidate-grid outer scoring.
 metrics=[];panels=[];predictions={};selected={};maxmean=0.;maxlevel=0.;maxmetric=0.;replays=0
 for name,d in ds.items():
  selected[name]={};spec=d['spec'];prep=d['prep']
  for seed in c['seeds']:
   selected[name][seed]={}
   for split in ['fit','val',spec['eval_split']]:
    p=d['p'][split];bp=torch.tensor(d['cache'][f'{seed}_{split}_FrozenTimeBase'],dtype=torch.float32);pp=prep[str(seed)]['Fourier'];ix=p['future'];ft=np.concatenate([numeric_summary(p['x'],bp).numpy(),d['raw'][ix],d['common'][ix]],-1)
    z=torch.tensor(np.concatenate([(ft-np.array(pp['mean']))/np.array(pp['sd']),(d['K'][ix]-np.array(pp['knowledge_mean']))/pp['knowledge_rms']],-1),dtype=torch.float32);label='eval' if split==spec['eval_split'] else split
    predictions[f'{name}_{seed}_{label}_y']=p['y'];predictions[f'{name}_{seed}_{label}_origin']=p['origin']
    refs=dict(OriginalTimeBase='OriginalTimeBase',FrozenTimeBase='FrozenTimeBase',PreviousNoKD='NoKD',PreviousQwenTaskKD='QwenTaskKD')
    outputs={a:d['refs'][f'{seed}_{split}_{old}'] for a,old in refs.items()}
    for arm,recipe in chosen.items():
     key=recipe['recipe'];step=recipe['domains'][name]['step'];state=states[name][arm][key][seed][step];head=Head(seed);head.load_state_dict(state)
     with torch.no_grad():pred=forward(head,z,bp).numpy()
     selected[name][seed][arm]=copy.deepcopy(state);outputs[arm]=pred;maxmean=max(maxmean,float(abs((pred-bp.numpy()).mean(1)).max()));rep=Head(seed);rep.load_state_dict(state)
     with torch.no_grad():assert np.array_equal(pred,forward(rep,z,bp).numpy());replays+=1
    baseline=scores(bp.numpy(),p['y'])
    for arm,pred in outputs.items():
     met=scores(pred,p['y']);metrics.append(dict(domain=name,seed=seed,split=label,arm=arm,**met));predictions[f'{name}_{seed}_{label}_{arm}']=pred;e=pred-p['y'];maxmetric=max(maxmetric,abs(float(e.ravel()@e.ravel()/e.size)-met['mse']));assert abs(met['mse']-met['level']-met['shape'])<1e-12
     if arm in chosen:maxlevel=max(maxlevel,abs(met['level']-baseline['level']))
    # Prespecified common Qwen-recipe panel for controls, using already trained trials only.
    commonkey=chosen['QwenTaskKD']['recipe']
    for arm in c['arms']:
     recipe=kdoptions[arm][commonkey];step=recipe['domains'][name]['step'];head=Head(seed);head.load_state_dict(states[name][arm][commonkey][seed][step])
     with torch.no_grad():pred=forward(head,z,bp).numpy()
     panels.append(dict(domain=name,seed=seed,split=label,arm=arm,recipe=commonkey,step=step,**scores(pred,p['y'])))
 torch.save(selected,r/'selected_students.pt');np.savez_compressed(r/'predictions.npz',**predictions);save(r/'metrics.json',metrics);save(r/'common_recipe_metrics.json',panels)
 allarms=list(refs)+list(chosen);summary={name:{split:{a:{k:float(np.mean([q[k] for q in metrics if q['domain']==name and q['split']==split and q['arm']==a])) for k in ['mse','mae','level','shape']} for a in allarms} for split in ['fit','val','eval']} for name in ds}
 intervals=[];rng=np.random.default_rng(2024)
 for name in ds:
  origins=predictions[f'{name}_2024_eval_origin'];clusters=(origins-origins.min())//336;unique=np.unique(clusters)
  for ref in ['PreviousQwenTaskKD','NoKD','FourierKD','HistoricalKD','ShuffleKD']:
   for metric in ['mse','mae']:
    dd=[]
    for seed in c['seeds']:
     y=predictions[f'{name}_{seed}_eval_y'];a=predictions[f'{name}_{seed}_eval_QwenTaskKD']-y;b=predictions[f'{name}_{seed}_eval_{ref}']-y;dd.append(((a*a-b*b) if metric=='mse' else (abs(a)-abs(b))).mean((1,2)))
    delta=np.mean(dd,0);boots=[]
    for _ in range(2000):
     ids=np.concatenate([np.flatnonzero(clusters==k) for k in rng.choice(unique,len(unique),replace=True)]);boots.append(float(delta[ids].mean()))
    intervals.append(dict(domain=name,reference=ref,metric=metric,task_minus_reference=float(delta.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),blocks=len(unique),note='Descriptive old-development block bootstrap, seeds averaged before resampling; not confirmatory.'))
 actual={a:{name:dict(step=q['domains'][name]['step'],actual_KD_updates=max(0,q['domains'][name]['step']-q['start']) if a!='NoKD' else 0) for name in ds} for a,q in chosen.items()}
 assert maxmean<1e-10 and maxlevel<1e-10 and sha(r/'selection.json')==lock
 for p in inputs+assets:assert sha(p['path'])==p['sha']
 save(r/'summary.json',dict(metrics=summary,paired_intervals=intervals,selected_KD_updates=actual));save(r/'verification.json',dict(fits=len(inits),updates_each=600,all_head_parameters_705=all(x['params']==705 for x in inits),anchor_checkpoint_reproductions=reproduction_counts,replayed_selected_predictions=replays,mean_error_max=maxmean,level_error_difference_max=maxlevel,independent_metric_error_max=maxmetric,source_files_and_bases_unchanged=True,selection_unchanged=True,actual_KD_updates=actual))
 save(r/'status.json',dict(seconds=time.monotonic()-start,fits=len(inits),git=git(),LLM_calls=0,LLM_updates=0,TimeBase_updates=0,new_encoder=False,new_confirmation=False));print('CHOSEN',chosen,flush=True)
 for name in ds:print('EVAL',name,summary[name]['eval'],flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args();run(read(a.config),a.config)
