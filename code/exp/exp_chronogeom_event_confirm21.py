"""Single independent Metro confirmation of fixed event Shape KD20 in TSLib."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
import argparse,copy,time,hashlib,shutil,sys,calendar
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from exp.exp_chronogeom_event_knowledge19 import Head,forward,scores,knowledge,extract,read,save,sha,git
from exp.exp_chronogeom_downstream import CalendarTimeBase,predict
from exp.exp_chronogeom_residual03 import numeric_summary

def center(x):return x-x.mean(1,keepdim=True)
def dataset(c,r):
 f=pd.read_csv(c['data'],keep_default_na=False);f['ts']=pd.to_datetime(f['date_time'])
 g=f.groupby('ts');disagreement=int((g.traffic_volume.nunique()>1).sum());v=g.traffic_volume.mean()
 grid=pd.date_range(f.ts.min().normalize(),f.ts.max().normalize()+pd.Timedelta(days=1),freq='h',inclusive='left');v=v.reindex(grid).to_numpy(float)
 flags=f.assign(day=f.ts.dt.normalize(),event=f.holiday!='None').groupby('day').event.max();hol=flags.reindex(grid.normalize()).to_numpy(float)
 wd=grid.dayofweek.to_numpy();h=grid.hour.to_numpy();week=(wd>=5).astype(int);kind=np.where(hol==1,2,week)
 years=np.array([366 if calendar.isleap(y) else 365 for y in grid.year]);a=2*np.pi*(grid.dayofyear.to_numpy()-1)/years
 raw=np.stack([hol,week,((week==0)&(hol==0)),np.sin(2*np.pi*wd/7),np.cos(2*np.pi*wd/7),np.sin(a),np.cos(a)],1).astype(float)
 harmonics=np.stack([fn(2*np.pi*k*h/24) for k in range(1,5) for fn in [np.sin,np.cos]],1);common=np.concatenate([harmonics*(kind==k)[:,None] for k in range(3)],1)
 a,b=c['splits']['base_fit'];mask=(grid>=a)&(grid<b)&np.isfinite(v);mu=float(v[mask].mean());sd=float(v[mask].std());assert sd>0;v=(v-mu)/sd
 packs={};counts={}
 for split,(a,b) in c['splits'].items():
  lo=grid.get_loc(pd.Timestamp(a));hi=grid.get_loc(pd.Timestamp(b));origins=np.array([o for o in range(lo,hi-96+1,24) if np.isfinite(v[o-96:o+96]).all() and np.isfinite(raw[o:o+96]).all()])
  assert len(origins)>0;past=origins[:,None]+np.arange(-96,0);future=origins[:,None]+np.arange(96)
  packs[split]=dict(x=torch.tensor(v[past,None],dtype=torch.float32),y=v[future,None],ids=torch.tensor(wd[future]*24+h[future],dtype=torch.long),origin=origins,future=future)
  counts[split]=dict(windows=len(origins),unique_target_hours=len(np.unique(future)),holiday_dates=sorted(set(grid[np.unique(future[hol[future]==1])].strftime('%Y-%m-%d'))))
 hist=np.zeros((3,24));n=np.zeros((3,24),int);a,b=c['splits']['fit'];mask=(grid>=a)&(grid<b)&np.isfinite(v)
 for k in range(3):
  for hour in range(24):
   ix=mask&(kind==k)&(h==hour);n[k,hour]=ix.sum();pool=mask&(h==hour);hist[k,hour]=v[ix].mean() if ix.any() else v[pool].mean()
 save(r/'data_metadata.json',dict(source=c['data_source'],sha=sha(c['data']),rows=len(f),unique_timestamps=len(g),duplicate_rows=len(f)-len(g),duplicate_disagreement_hours=disagreement,counts=counts,mean=mu,sd=sd,historical_counts=n.tolist()))
 np.savez_compressed(r/'data.npz',**{f'{split}_{key}':val.numpy() if torch.is_tensor(val) else val for split,p in packs.items() for key,val in p.items()})
 return packs,grid,raw,common,kind,h,hist

def choose(curves,arms,seeds,base,packs):
 chosen={}
 for arm in arms:
  choices=[]
  for step in range(0,601,100):
   vals=[]
   for seed in seeds:
    b=scores(base[seed]['val'].numpy(),packs['val']['y']);q=next(x for x in curves if x['arm']==arm and x['seed']==seed and x['step']==step);vals.append(.5*(q['mse']/b['mse']+q['mae']/b['mae']))
   choices.append((float(np.mean(vals)),step))
  chosen[arm]=min(choices)[1]
 return chosen

def fit(c,r):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);start=time.monotonic()
 packs,grid,raw,common,kind,h,hist=dataset(c,r);seeds=c['seeds'];scalars=[];base_models={};base_curves=[];base_choices={};base={};original={}
 # Original40 gets full training period; residual numeric base is fitted on the earlier disjoint part.
 for seed in seeds:
  base_models[seed]={};base_choices[seed]={};base[seed]={};original[seed]={}
  for arm,fitname,valname,table in [('OriginalTimeBase','full_train','val',None),('FrozenTimeBase','base_fit','base_val',np.zeros((168,168),np.float32))]:
   m=CalendarTimeBase(c,seed,table);opt=torch.optim.Adam(m.parameters(),lr=c['lr']);rng=np.random.default_rng(seed+271);p=packs[fitname];y=torch.tensor(p['y'],dtype=torch.float32);states={};cv=[]
   for step in range(601):
    if step%100==0:
     pp=predict(m,packs[valname]);q=dict(seed=seed,arm=arm,step=step,**scores(pp,packs[valname]['y']));base_curves.append(q);cv.append(q);states[step]=copy.deepcopy(m.state_dict())
    if step==600:break
    ids=rng.integers(0,len(y),size=c['batch']);m.train();yp=m(p['x'][ids],p['ids'][ids]);loss=(yp-y[ids]).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();assert time.monotonic()-start<c['fit_seconds']
   chosen=min(cv,key=lambda q:q['mse'])['step'];m.load_state_dict(states[chosen]);m.eval()
   for param in m.parameters():param.requires_grad_(False)
   path=r/f'{arm}_seed{seed}.pt';torch.save(m.state_dict(),path);base_models[seed][arm]=m;base_choices[seed][arm]=chosen
   scalars.append(dict(seed=seed,arm=arm,path=str(path),sha=sha(path),parameters=sum(p.numel() for p in m.parameters()),fit=fitname,selection=valname,step=chosen))
   for split in ['fit','val']:
    pp=torch.tensor(predict(m,packs[split]));(base if arm=='FrozenTimeBase' else original)[seed][split]=pp
   print('BASE',seed,arm,chosen,flush=True)
 save(r/'base_selection.json',base_choices);save(r/'base_curves.json',base_curves);save(r/'base_assets.json',scalars)
 task=np.array(read(r/'profiles.json')['Task']);perm=np.random.default_rng(2024).permutation(24);shuffled=task.copy();shuffled[1:]=task[0]+(task[1:]-task[0])[:,perm]
 K=dict(Fourier=np.stack([np.sin(10*np.pi*h/24),np.cos(10*np.pi*h/24),np.cos(12*np.pi*h/24)],1),HistoricalTemplate=knowledge(hist,kind,h),QwenTask=knowledge(task,kind,h),QwenShuffle=knowledge(shuffled,kind,h))
 np.savez_compressed(r/'knowledge.npz',**K,historical_profiles=hist,shuffle_permutation=perm)
 zs={};preps={};batches={};norms={};diagnostics=[]
 def features(seed,split,arm):
  p=packs[split];ix=p['future'];nums=numeric_summary(p['x'],base[seed][split]).numpy();ft=np.concatenate([nums,raw[ix],common[ix]],-1);n=preps[seed][arm]
  return torch.tensor(np.concatenate([(ft-n['mean'])/n['sd'],(K[arm][ix]-n['knowledge_mean'])/n['knowledge_rms']],-1),dtype=torch.float32)
 for seed in seeds:
  p=packs['fit'];ix=p['future'];nums=numeric_summary(p['x'],base[seed]['fit']).numpy();ft=np.concatenate([nums,raw[ix],common[ix]],-1);mu=ft.mean((0,1));sd=np.maximum(ft.std((0,1)),1e-6);zs[seed]={};preps[seed]={}
  norms[seed]=scores(base[seed]['fit'].numpy(),p['y']);batches[seed]=np.random.default_rng(seed).integers(0,len(p['y']),size=(600,c['batch']))
  for arm in c['teachers']:
   k=K[arm][ix];km=k.mean((0,1));ks=max(float(np.sqrt(np.mean((k-km)**2))),1e-6);preps[seed][arm]=dict(mean=mu,sd=sd,knowledge_mean=km,knowledge_rms=ks)
   zs[seed][arm]={split:features(seed,split,arm) for split in ['fit','val']}
   mat=zs[seed][arm]['fit'].numpy().reshape(-1,42);sv=np.linalg.svd(mat-mat.mean(0),compute_uv=False);w=sv**2/(sv**2).sum();diagnostics.append(dict(seed=seed,arm=arm,rank=int(np.linalg.matrix_rank(mat-mat.mean(0))),effective_rank=float(np.exp(-np.sum(w[w>0]*np.log(w[w>0])))),knowledge_rms=float(np.sqrt(np.mean(mat[:,-3:]**2)))))
 save(r/'normalizers.json',{seed:{arm:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in d.items()} for arm,d in arms.items()} for seed,arms in preps.items()});save(r/'feature_diagnostics.json',diagnostics)
 def train(seed,z,teacher=None,expected=None):
  head=Head(seed);opt=torch.optim.Adam(head.parameters(),lr=c['lr']);bp=base[seed]['fit'];target=torch.tensor(packs['fit']['y']);norm=norms[seed];states={};curves=[]
  init=forward(head,z['fit'],bp);assert torch.equal(init,bp.double());err=(init-target).square().mean();err.backward();assert head.net[-1].weight.grad.abs().sum()>0;head.zero_grad()
  teacher=None if teacher is None else center(teacher).detach()
  for step in range(601):
   if step%100==0:
    with torch.no_grad():pv=forward(head,z['val'],base[seed]['val']).numpy()
    curves.append(dict(step=step,**scores(pv,packs['val']['y'])));states[step]=copy.deepcopy(head.state_dict())
    if expected is not None:
     for key,value in head.state_dict().items():assert torch.equal(value,expected[step][key]),('NoKD must exactly equal Fourier SUP',seed,step,key)
   if step==600:break
   ids=batches[seed][step];pred=forward(head,z['fit'][ids],bp[ids]);e=pred-target[ids];loss=.5*(e.square().mean()/norm['mse']+e.abs().mean()/norm['mae'])
   if teacher is not None:loss=loss+c['kd_weight']*(center(pred)-teacher[ids]).square().mean()/norm['mse']
   opt.zero_grad();loss.backward();opt.step();assert time.monotonic()-start<c['fit_seconds']
  return states,curves
 teacher_states={};teacher_curves=[]
 for seed in seeds:
  teacher_states[seed]={}
  for arm in c['teachers']:
   states,cv=train(seed,zs[seed][arm]);teacher_states[seed][arm]=states;teacher_curves.extend(dict(seed=seed,arm=arm,**q) for q in cv);print('TEACHER',seed,arm,flush=True)
 tc=choose(teacher_curves,c['teachers'],seeds,base,packs);save(r/'teacher_selection.json',dict(chosen=tc,git=git(),confirm_metrics_seen=False));torch.save(teacher_states,r/'teacher_checkpoints.pt');save(r/'teacher_curves.json',teacher_curves)
 soft={};teacher_predictions={}
 for seed in seeds:
  soft[seed]={}
  for arm in c['teachers']:
   m=Head(seed);m.load_state_dict(teacher_states[seed][arm][tc[arm]]);m.eval()
   for p in m.parameters():p.requires_grad_(False)
   with torch.no_grad():soft[seed][arm]=forward(m,zs[seed][arm]['fit'],base[seed]['fit']).detach()
   teacher_predictions[f'{seed}_fit_{arm}']=soft[seed][arm].numpy()
 student_states={};student_curves=[]
 for seed in seeds:
  student_states[seed]={}
  for arm,teacher in c['arms'].items():
   states,cv=train(seed,zs[seed]['Fourier'],None if teacher is None else soft[seed][teacher],teacher_states[seed]['Fourier'] if teacher is None else None);student_states[seed][arm]=states;student_curves.extend(dict(seed=seed,arm=arm,**q) for q in cv);print('STUDENT',seed,arm,flush=True)
 sc=choose(student_curves,c['arms'],seeds,base,packs);save(r/'selection.json',dict(chosen=sc,teachers=tc,bases=base_choices,confirm_metrics_seen=False,git=git(),note='All settings/checkpoints locked before the FIRST confirm prediction or metric.'))
 torch.save(student_states,r/'student_checkpoints.pt');save(r/'student_curves.json',student_curves)
 locksha=sha(r/'selection.json');save(r/'evaluation_lock.json',dict(selection_sha=locksha,teacher_selection_sha=sha(r/'teacher_selection.json'),config_sha=sha(r/'config.json'),source_sha=sha(__file__),first_confirm_evaluation=True))
 # First access to confirmation model predictions/errors happens only below this lock.
 preds={};metrics=[];subgroups=[];maxmean=0.;maxmetric=0.;maxlevel=0.;replays=0
 for seed in seeds:
  for split in ['fit','val','confirm']:
   p=packs[split];out={}
   for name,m in base_models[seed].items():out[name]=predict(m,p).astype(float)
   if split=='confirm':base[seed][split]=torch.tensor(out['FrozenTimeBase'],dtype=torch.float32)
   for family,names,allstates,chosen in [('Teacher',c['teachers'],teacher_states,tc),('Student',c['arms'],student_states,sc)]:
    for arm in names:
     inputarm=arm if family=='Teacher' else 'Fourier';z=features(seed,split,inputarm);m=Head(seed);m.load_state_dict(allstates[seed][arm][chosen[arm]])
     with torch.no_grad():pr=forward(m,z,base[seed][split]).numpy()
     label='Teacher_'+arm if family=='Teacher' else arm;out[label]=pr;maxmean=max(maxmean,float(abs((pr-out['FrozenTimeBase']).mean(1)).max()))
     replay=Head(seed);replay.load_state_dict(allstates[seed][arm][chosen[arm]])
     with torch.no_grad():assert np.array_equal(pr,forward(replay,z,base[seed][split]).numpy());replays+=1
   preds[f'{seed}_{split}_y']=p['y'];preds[f'{seed}_{split}_origin']=p['origin'];ix=p['future'];hh=h[ix];kk=kind[ix];groups=dict(all=np.ones(ix.shape,bool),workday=kk==0,weekend=kk==1,holiday=kk==2,morning7_9=(hh>=7)&(hh<=9),evening16_19=(hh>=16)&(hh<=19))
   for arm,pr in out.items():
    preds[f'{seed}_{split}_{arm}']=pr;q=scores(pr,p['y']);metrics.append(dict(seed=seed,split=split,arm=arm,**q));e=pr-p['y'];maxmetric=max(maxmetric,abs(float(e.ravel()@e.ravel()/e.size)-q['mse']));assert abs(q['mse']-q['level']-q['shape'])<1e-12
    if arm not in ['OriginalTimeBase','FrozenTimeBase']:maxlevel=max(maxlevel,abs(q['level']-scores(out['FrozenTimeBase'],p['y'])['level']))
    for group,mask in groups.items():
     ee=e[...,0][mask]
     if len(ee):subgroups.append(dict(seed=seed,split=split,arm=arm,group=group,unique_hours=int(len(np.unique(ix[mask]))),windows=int(mask.any(1).sum()),mse=float((ee**2).mean()),mae=float(abs(ee).mean())))
   assert np.array_equal(out['NoKD'],out['Teacher_Fourier'])
 np.savez_compressed(r/'predictions.npz',**preds);np.savez_compressed(r/'teacher_fit_targets.npz',**teacher_predictions);save(r/'metrics.json',metrics);save(r/'subgroups.json',subgroups)
 allarms=['OriginalTimeBase','FrozenTimeBase']+list(c['arms'])+['Teacher_'+a for a in c['teachers']];summary={split:{a:{k:float(np.mean([x[k] for x in metrics if x['split']==split and x['arm']==a])) for k in ['mse','mae','level','shape']} for a in allarms} for split in ['fit','val','confirm']}
 uncertainty=[];rng=np.random.default_rng(2024);origins=packs['confirm']['origin'];clusters=(origins-origins.min())//336;unique=np.unique(clusters)
 for reference in ['OriginalTimeBase','FrozenTimeBase','NoKD','FourierKD','HistoricalKD','ShuffleKD']:
  for metric in ['mse','mae']:
   dd=[]
   for seed in seeds:
    y=preds[f'{seed}_confirm_y'];a=preds[f'{seed}_confirm_QwenTaskKD']-y;b=preds[f'{seed}_confirm_{reference}']-y;dd.append(((a*a-b*b) if metric=='mse' else (abs(a)-abs(b))).mean((1,2)))
   delta=np.mean(dd,0);boots=[]
   for _ in range(2000):
    ix=np.concatenate([np.flatnonzero(clusters==k) for k in rng.choice(unique,len(unique),replace=True)]);boots.append(float(delta[ix].mean()))
   uncertainty.append(dict(reference=reference,metric=metric,task_minus_reference=float(delta.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),calendar14day_clusters=len(unique),note='Exploratory temporal block bootstrap; average seeds before resampling, not IID windows/seeds.'))
 assert sha(r/'selection.json')==locksha and maxmean<1e-10 and maxlevel<1e-10
 for rec in scalars:assert sha(rec['path'])==rec['sha']
 save(r/'summary.json',dict(metrics=summary,paired_intervals=uncertainty));save(r/'verification.json',dict(teacher_and_student_fits=27,base_fits=6,updates_each=600,head_parameters=705,noKD_21states_exact_to_supervised_Fourier=True,noKD_all9_predictions_exact=True,prediction_replays=replays,mean_error_max=maxmean,level_error_difference_max=maxlevel,independent_metric_error_max=maxmetric,base_assets_unchanged=True,selection_unchanged_after_confirm=True,confirmation_first_open_after_all_checkpoints_locked=True,LLM_trainable=0,student_inference_requires_LLM=False))
 save(r/'status.json',dict(fit_seconds=time.monotonic()-start,git=git(),LLM_updates=0,new_encoder=False,confirm_evaluations=1));print('RESULT',summary['confirm'],flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--stage',choices=['extract','fit'],required=True);ap.add_argument('--gpu',type=int,default=6);a=ap.parse_args();c=read(a.config);r=Path(c['run'])
 if a.stage=='extract':
  r.mkdir(parents=True,exist_ok=False);save(r/'config.json',c);save(r/'provenance.json',dict(git=git(),source_sha=sha(__file__),config_sha=sha(a.config),data_sha=sha(c['data']),python=sys.version,torch=torch.__version__,numpy=np.__version__));extract(c,r,a.gpu)
 else:
  assert read(r/'config.json')==c;fit(c,r)
if __name__=='__main__':main()
