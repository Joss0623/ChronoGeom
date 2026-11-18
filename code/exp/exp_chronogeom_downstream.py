"""Bounded real-demand TimeBase teacher-value pilot inside the pinned TSLib checkout."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
import argparse,copy,csv,hashlib,json,math,subprocess,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from layers.ChronoGeomTimeBaseOfficial import Model as TimeBase
from models.ChronoGeomTimeBase import timestamp_feature_loss
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def csvsave(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read_data(c,end):
 import pandas as pd
 df=pd.read_csv(c['data_path'],encoding='latin1',nrows=end,usecols=['Date','Hour','Rented Bike Count'])
 assert len(df)==end
 dates=pd.to_datetime(df['Date'],format='%d/%m/%Y')+pd.to_timedelta(df['Hour'],unit='h')
 assert (dates.diff().dropna()==pd.Timedelta(hours=1)).all()
 values=df['Rented Bike Count'].to_numpy(dtype='float32')[:,None];assert np.isfinite(values).all()
 ids=(dates.dt.dayofweek.to_numpy()*24+dates.dt.hour.to_numpy()).astype('int64')
 return values,ids,dates
def normalized(z):
 z=z-z.mean(0,keepdims=True);return z/max(np.sqrt(np.mean(np.sum(z*z,axis=1))),1e-12)
def build_tables(c,r):
 start=time.monotonic();cache=Path(c['llm_cache']);assert int(np.load(cache/'lookup.npz')['selected_layer'])==6;records=json.loads((cache/'prompts.json').read_text());raw=np.load(cache/'raw_spans.npz')['hidden'][:,0].astype('float64');full=[]
 for template in [0,1]:
  ix=[i for i,v in enumerate(records) if v['purpose']=='lookup' and v['template']==template]
  assert [(records[i]['day'],records[i]['hour']) for i in ix]==[(d,h) for d in range(7) for h in range(24)]
  full.append(raw[ix])
 center=full[0].mean(0);u,s,vt=np.linalg.svd(full[0]-center,full_matrices=False);basis=vt[:32]
 z0=(full[0]-center)@basis.T;z1=(full[1]-center)@basis.T
 oldproj=np.load(cache/'geometry_layer6.npz');p0=(full[0]-oldproj['center'])@oldproj['projection'].T/oldproj['scale'];p1=(full[1]-oldproj['center'])@oldproj['projection'].T/oldproj['scale']
 d=np.repeat(np.arange(7),24);h=np.tile(np.arange(24),7)
 f4=np.c_[np.cos(2*np.pi*h/24),np.sin(2*np.pi*h/24),np.cos(2*np.pi*d/7),np.sin(2*np.pi*d/7)]
 cols=[]
 for x,p,ks in [(h,24,range(1,12)),(d,7,range(1,4))]:
  for k in ks:cols.extend([np.cos(2*np.pi*k*x/p),np.sin(2*np.pi*k*x/p)])
 cols.extend(f4[:,i]*f4[:,j] for i in [0,1] for j in [2,3])
 native={'NoCalendar':np.zeros((168,1)),'RawCalendar':np.c_[d/3.-1,h/11.5-1],'SinCos4':f4,'Fourier32':np.column_stack(cols),'Calendar32':np.c_[np.eye(24)[h],np.eye(7)[d],(d>=5).astype(float)],'Joint168':np.eye(168),'LLM4_cached':np.load(cache/'lookup.npz')['llm4'],'LLM32_t0':z0,'LLM32_t1':z1,'LLM4_prompt0':p0,'LLM4_prompt1':p1}
 perm=np.random.default_rng(c['shuffle_seed']).permutation(168)
 native.update(LLM32_shuffle=z0[perm],LLM4_shuffle=native['LLM4_cached'][perm],LLM32_t1_shuffle=z1[perm])
 mix=np.linalg.qr(np.random.default_rng(c['feature_seed']).normal(size=(168,168)))[0];tables={};diagnostics=[]
 for name,z in native.items():
  zn=normalized(z.astype(float));padded=np.zeros((168,168));padded[:,:zn.shape[1]]=zn;table=(padded@mix*np.sqrt(168)).astype('float32');tables[name]=table
  sv=np.linalg.svd(zn,compute_uv=False);rank=int(np.sum(sv>max(sv[0],1e-12)*1e-7));power=sv**2;power=power[power>1e-20];prob=power/power.sum() if len(power) else np.array([])
  diagnostics.append(dict(name=name,native_dimension=z.shape[1],input_width=168,rank=rank,entropy_rank=float(np.exp(-np.sum(prob*np.log(prob)))) if len(prob) else 0.,rms=float(np.sqrt(np.mean(table**2))),min_norm=float(np.linalg.norm(table,axis=1).min()),max_norm=float(np.linalg.norm(table,axis=1).max()),distinct_states=len(np.unique(np.round(table,5),axis=0))))
 np.savez_compressed(r/'tables.npz',**tables);np.savez_compressed(r/'pca_assets.npz',basis=basis,center=center,mix=mix,permutation=perm)
 stable=[]
 for name,a,b in [('PCA32',z0,z1),('Legacy4',p0,p1)]:
  za,zb=normalized(a),normalized(b);ga,gb=za@za.T,zb@zb.T
  stable.append(dict(space=name,centered_kernel_cosine=float(np.sum(ga*gb)/np.linalg.norm(ga)/np.linalg.norm(gb)),raw_coordinate_nrmse=float(np.linalg.norm(a-b)/np.linalg.norm(a-a.mean(0))),centered_coordinate_nrmse=float(np.linalg.norm((a-a.mean(0))-(b-b.mean(0)))/np.linalg.norm(a-a.mean(0)))))
 explained=[]
 for target in ['LLM4_cached','LLM32_t0','LLM32_t1']:
  y=normalized(native[target])
  for name in ['SinCos4','Fourier32','Calendar32','Joint168']:
   x=normalized(native[name]);pred=x@np.linalg.lstsq(x,y,rcond=1e-10)[0];explained.append(dict(target=target,basis=name,relative_residual_energy=float(np.sum((y-pred)**2)/np.sum(y*y))))
 csvsave(r/'teacher_diagnostics.csv',diagnostics);csvsave(r/'prompt_stability.csv',stable);csvsave(r/'analytical_explanation.csv',explained)
 dump(r/'teacher_preparation.json',dict(seconds=time.monotonic()-start,gpu_seconds=0,new_llm_forwards=0,source_completion=json.loads((cache/'completion.json').read_text()),raw_sha256=sha(cache/'raw_spans.npz'),prompts_sha256=sha(cache/'prompts.json'),lookup_sha256=sha(cache/'lookup.npz'),selected_layer=6,pca32_fit='template0 known training calendar states, no forecasting labels'))
 return tables
class CalendarTimeBase(nn.Module):
 def __init__(self,c,seed,table=None):
  super().__init__();torch.manual_seed(seed)
  cfg=SimpleNamespace(seq_len=c['seq_len'],pred_len=c['pred_len'],enc_in=1,period_len=24,basis_num=4,individual=False,use_period_norm=True,use_orthogonal=False)
  self.body=TimeBase(cfg);self.use_calendar=table is not None
  if self.use_calendar:
   self.register_buffer('table',torch.as_tensor(table,dtype=torch.float32));torch.manual_seed(seed+10000)
   self.time_mlp=nn.Sequential(nn.Linear(168,32),nn.GELU(),nn.Linear(32,32));self.readout=nn.Linear(32,2);nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)
 def codes(self,override=None):return self.time_mlp(self.table if override is None else override)
 def forward(self,x,ids,override=None):
  p=self.body(x)
  if self.use_calendar:
   effects=self.readout(self.codes(override));q=effects[ids];p=p*(1+torch.tanh(q[...,:1]))+q[...,1:2]
  return p
def pack(values,ids,lo,hi,c):
 L,H=c['seq_len'],c['pred_len'];origins=np.arange(lo,hi-H+1,c['evaluation_stride']);assert len(origins)>0 and origins.min()>=L
 return dict(x=np.stack([values[o-L:o] for o in origins]),y=np.stack([values[o:o+H] for o in origins]),ids=np.stack([ids[o:o+H] for o in origins]),origin=origins)
def tensors(p,device):return {k:torch.as_tensor(v,device=device) for k,v in p.items()}
def prepare(c,r):
 r.mkdir(parents=True,exist_ok=False);dump(r/'config.json',c)
 assert sha('layers/ChronoGeomTimeBaseOfficial.py')=='f273f20df34be28e50854f1e55791750d26bdf56996135db6822832b0fd3e9e6'
 values,ids,dates=read_data(c,c['val_end']);tr=values[c['train_start']:c['train_end']];mu=tr.mean(0);sd=tr.std(0);assert sd[0]>1e-6 and len(np.unique(ids[c['train_start']:c['train_end']]))==168
 np.savez_compressed(r/'train_val.npz',values=(values-mu)/sd,ids=ids,mean=mu,sd=sd)
 tables=build_tables(c,r);calendar_means=np.array([tr[ids[c['train_start']:c['train_end']]==i].mean() for i in range(168)]);fitted=calendar_means[ids[c['train_start']:c['train_end']]]
 loo=[];truth=[]
 for week in range(8):
  a=tr[:,0].copy();j=np.arange(len(a));mask=j//168!=week;sid=ids[c['train_start']:c['train_end']];means=np.array([a[mask&(sid==i)].mean() for i in range(168)]);loo.extend(means[sid[~mask]].tolist());truth.extend(a[~mask].tolist())
 dump(r/'train_calendar_association.json',dict(status='training-only descriptive association, not forecast improvement conditional on history',n=1344,in_sample_calendar_r2=float(1-np.mean((fitted-tr[:,0])**2)/np.var(tr)),leave_week_out_calendar_r2=float(1-np.mean((np.array(loo)-truth)**2)/np.var(tr)),mean=float(mu[0]),sd=float(sd[0])))
 m=CalendarTimeBase(c,0,tables['RawCalendar']);x=torch.randn(2,96,1);ii=torch.arange(96).repeat(2,1)%168;assert torch.equal(m(x,ii),m.body(x));assert sum(p.numel() for p in m.parameters())==6570
 a=CalendarTimeBase(c,0);assert sum(p.numel() for p in a.parameters())==40
 manifest=dict(commit=commit(),script_sha256=sha(__file__),tslib_upstream='4e938a1767106324dd753b2a44832bf870a0252e',timebase_revision='369b330f3d77371fcc7e8c75c808d01330c40899',timebase_sha256=sha('layers/ChronoGeomTimeBaseOfficial.py'),data_sha256=sha(c['data_path']),tables_sha256=sha(r/'tables.npz'),prepared_sha256=sha(r/'train_val.npz'),labels_parsed_until=c['val_end'],pilot_and_confirmation_targets_opened=False,initial_adapter_identity_exact=True,calendar_parameters=6570,original_parameters=40)
 dump(r/'source.json',manifest);print('PREPARED',json.dumps(manifest),flush=True)
def predict(m,p,override=None):
 out=[];m.eval()
 with torch.no_grad():
  for i in range(0,len(p['x']),128):out.append(m(p['x'][i:i+128],p['ids'][i:i+128],override).cpu().numpy())
 return np.concatenate(out)
def ncodes(z):return (z-z.mean(0,keepdim=True))/(z-z.mean(0,keepdim=True)).square().mean().sqrt().clamp_min(1e-6)
def kd_loss(mode,student,teacher,x,ids,yp,yt,shift):
 if mode=='Output':return F.mse_loss(yp,yt)
 if mode in ['Feature','Relation']:
  zs=ncodes(student.codes());zt=ncodes(teacher.codes()).detach()
  return timestamp_feature_loss(zs,zt) if mode=='Feature' else F.mse_loss(zs@zs.T/32,zt@zt.T/32)
 with torch.no_grad():target=teacher(x,(ids+shift)%168)-yt
 actual=student(x,(ids+shift)%168)-yp
 return F.mse_loss(actual,target)
def fit(c,r,seed,device,kd=False):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(c['gpu_cap_gib']*2**30/torch.cuda.get_device_properties(device).total_memory,device)
 output=r/(f'kd_seed{seed}' if kd else f'seed{seed}');output.mkdir(exist_ok=False);tables=np.load(r/'tables.npz');a=np.load(r/'train_val.npz');values=torch.tensor(a['values'],device=device);allids=torch.tensor(a['ids'],device=device);va=tensors(pack(a['values'],a['ids'],c['train_end'],c['val_end'],c),device)
 begin=time.monotonic();cap=c['seed_kd_cap_seconds'] if kd else c['seed_fit_cap_seconds'];records=[]
 if kd:
  assert json.loads((r/'pilot/gate.json').read_text())['passed'];lockfile=json.loads((r/'models_lock.json').read_text());jobs=[(kind,mode) for kind in ['LLM','Analytic'] for mode in c['kd_modes']]
 else:jobs=[(arm,None) for arm in c['arms']]
 for arm,mode in jobs:
  if time.monotonic()-begin>cap:raise TimeoutError('per-seed fixed fitting budget exceeded')
  key=arm if not kd else f'{arm}_{mode}';table=None if arm=='Original' and not kd else tables['RawCalendar' if kd else arm]
  m=CalendarTimeBase(c,seed,table).to(device);teacher=None;lam=0.
  if kd:
   teacher_arm=lockfile['llm_arm' if arm=='LLM' else 'analytic_arm'];trec=next(z for z in lockfile['models'] if z['seed']==seed and z['arm']==teacher_arm);assert sha(trec['path'])==trec['sha256'];teacher=CalendarTimeBase(c,seed,tables[teacher_arm]).to(device);teacher.load_state_dict(torch.load(trec['path'],map_location=device,weights_only=True));teacher.eval()
   for p in teacher.parameters():p.requires_grad_(False)
  opt=torch.optim.Adam(m.parameters(),lr=c['lr']);rng=np.random.default_rng(seed+271);curve=[];best=float('inf');beststate=None;start=time.monotonic();initial_gradient=None
  for step in range(c['steps']+1):
   if time.monotonic()-begin>cap:raise TimeoutError('per-seed fixed fitting budget exceeded')
   if step%c['validation_every']==0:
    pred=predict(m,va);v=float(np.mean((pred-va['y'].cpu().numpy())**2));assert np.isfinite(v)
    curve.append(dict(step=step,val_mse=v))
    if v<best:best=v;beststep=step;beststate=copy.deepcopy(m.state_dict())
   if step==c['steps']:break
   origins=torch.tensor(rng.integers(c['train_start']+c['seq_len'],c['train_end']-c['pred_len']+1,size=c['batch_size']),device=device);past=origins[:,None]+torch.arange(-c['seq_len'],0,device=device);future=origins[:,None]+torch.arange(c['pred_len'],device=device)
   x,y,ids=values[past],values[future],allids[future];m.train();yp=m(x,ids);loss=F.mse_loss(yp,y)
   if kd:
    with torch.no_grad():yt=teacher(x,ids)
    aux=kd_loss(mode,m,teacher,x,ids,yp,yt,c['response_shifts'][step%len(c['response_shifts'])])
    if step==0:
     params=list(m.parameters());g1=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);g2=torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True);n1=float(torch.sqrt(sum(g.square().sum() for g in g1 if g is not None)));n2=float(torch.sqrt(sum(g.square().sum() for g in g2 if g is not None)));assert n1>0 and n2>1e-12;lam=c['kd_initial_gradient_ratio']*n1/n2;initial_gradient=dict(forecast=n1,aux=n2,lambda_value=lam);del g1,g2
    loss=loss+lam*aux
   assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
  m.load_state_dict(beststate);path=output/f'{key}.pt';torch.save(beststate,path);dump(output/f'{key}_curve.json',curve)
  # Same batch1 shape for all inference-cost measurements; no GPT-J call.
  samplex,sampleids=va['x'][:1],va['ids'][:1]
  with torch.no_grad():
   for _ in range(10):m(samplex,sampleids)
   torch.cuda.synchronize(device);tick=time.monotonic()
   for _ in range(100):m(samplex,sampleids)
   torch.cuda.synchronize(device);latency=(time.monotonic()-tick)*10
  rec=dict(arm=key,seed=seed,best_step=beststep,val_mse=best,parameters=sum(p.numel() for p in m.parameters()),seconds=time.monotonic()-start,path=str(path),sha256=sha(path),batch1_inference_ms=latency,table_bytes=0 if not m.use_calendar else m.table.numel()*4,device=device,initial_gradient=initial_gradient)
  records.append(rec);dump(output/'records.json',records);print('FIT',json.dumps(rec),flush=True);del opt,m,teacher
 dump(output/'completion.json',dict(commit=commit(),seed=seed,records=records,seconds=time.monotonic()-begin,peak_gib=torch.cuda.max_memory_allocated(device)/2**30));print('COMPLETE_SEED',seed,kd,flush=True)
def lock(c,r,kd=False):
 models=[]
 for seed in c['seeds']:models+=json.loads((r/(f'kd_seed{seed}' if kd else f'seed{seed}')/'completion.json').read_text())['records']
 for m in models:assert sha(m['path'])==m['sha256']
 if kd:dump(r/'kd_lock.json',dict(commit=commit(),models=models));return
 means={arm:float(np.mean([x['val_mse'] for x in models if x['arm']==arm])) for arm in c['arms']}
 llm=min(c['llm_candidates'],key=lambda x:means[x]);analytic=min(c['strong_candidates'],key=lambda x:means[x])
 dump(r/'models_lock.json',dict(commit=commit(),models=models,validation_means=means,llm_arm=llm,analytic_arm=analytic,shuffled_arm=c['shuffle_map'][llm],test_opened=False))
 print('LOCKED',llm,analytic,flush=True)
def mbb(d,c):
 rng=np.random.default_rng(c['bootstrap_seed']);n=len(d);length=c['bootstrap_block_days'];starts=rng.integers(n,size=(c['bootstrap_replicates'],int(np.ceil(n/length))));ix=(starts[:,:,None]+np.arange(length))%n;ix=ix.reshape(c['bootstrap_replicates'],-1)[:,:n];b=d[ix].mean(1)
 return float(np.quantile(b,.025)),float(np.quantile(b,.975))
def evaluate(c,r,stage,device):
 torch.set_num_threads(2);lockfile=json.loads((r/'models_lock.json').read_text());models=list(lockfile['models']);out=r/stage;out.mkdir(exist_ok=False)
 if stage=='confirmation':
  gate=json.loads((r/'pilot/gate.json').read_text())
  if gate['passed']:
   kdlock=json.loads((r/'kd_lock.json').read_text());models+=kdlock['models']
 for m in models:assert sha(m['path'])==m['sha256']
 dump(out/'before_target_access.json',dict(commit=commit(),stage=stage,models_sha_locked=True,llm_arm=lockfile['llm_arm'],analytic_arm=lockfile['analytic_arm'],metadata_only_read_previously=True))
 lo,hi=(c['val_end'],c['pilot_end']) if stage=='pilot' else (c['pilot_end'],c['confirmation_end'])
 raw,ids,dates=read_data(c,hi);a=np.load(r/'train_val.npz');values=(raw-a['mean'])/a['sd'];pn=pack(values,ids,lo,hi,c);p=tensors(pn,device);tables=np.load(r/'tables.npz');preds={};rows=[];errs={};eff=[]
 naive={'Persistence':np.repeat(pn['x'][:,-1:],c['pred_len'],axis=1),'Seasonal24':np.tile(pn['x'][:,-24:],(1,c['pred_len']//24,1))}
 for name,pp in naive.items():preds[name]=pp
 for rec in models:
  arm,seed=rec['arm'],rec['seed'];is_kd=arm.startswith('LLM_') or arm.startswith('Analytic_')
  model=CalendarTimeBase(c,seed,None if arm=='Original' else tables['RawCalendar' if is_kd else arm]).to(device);model.load_state_dict(torch.load(rec['path'],map_location=device,weights_only=True));pp=predict(model,p);preds[f'{arm}_s{seed}']=pp
  if arm==lockfile['llm_arm']:
   for j,name in enumerate(c['prompt_map'][arm]):preds[f'Prompt{j}_s{seed}']=predict(model,p,torch.tensor(tables[name],device=device))
  del model
 for key,pp in preds.items():
  error=pp-pn['y'];mse=float(np.mean(error**2));mae=float(np.mean(abs(error)));rows.append(dict(method=key,mse=mse,mae=mae,raw_mse=mse*float(a['sd'][0])**2,raw_mae=mae*float(a['sd'][0]),n_origins=len(pn['origin'])))
 csvsave(out/'scores.csv',rows);np.savez_compressed(out/'predictions.npz',**preds,truth=pn['y'],origin=pn['origin'])
 arms=list(c['arms'])+['Prompt0','Prompt1']+sorted({x['arm'] for x in models if x['arm'] not in c['arms']})
 summary=[]
 for arm in arms:
  e=np.stack([((preds[f'{arm}_s{s}']-pn['y'])**2).mean((1,2)) for s in c['seeds']]);errs[arm]=e.mean(0);ss=[next(x for x in rows if x['method']==f'{arm}_s{s}') for s in c['seeds']]
  summary.append(dict(arm=arm,mse=float(np.mean([x['mse'] for x in ss])),mae=float(np.mean([x['mae'] for x in ss])),seed_mse=[x['mse'] for x in ss],seed_mae=[x['mae'] for x in ss]))
 dump(out/'summary.json',summary);llm=lockfile['llm_arm'];analytic=lockfile['analytic_arm'];comparisons=[]
 for other in [analytic,'RawCalendar','NoCalendar','Original',lockfile['shuffled_arm'],'SinCos4']:
  d=errs[llm]-errs[other];low,high=mbb(d,c);ss1=next(x for x in summary if x['arm']==llm)['seed_mse'];ss2=next(x for x in summary if x['arm']==other)['seed_mse'];comparisons.append(dict(comparison=llm+'-'+other,difference=float(d.mean()),lo95=low,hi95=high,seed_differences=[x-y for x,y in zip(ss1,ss2)]))
 dump(out/'contrasts.json',comparisons)
 if stage=='pilot':
  main=next(x for x in comparisons if x['comparison']==llm+'-'+analytic);smap={x['arm']:x for x in summary};checks=dict(three_seeds_better_strong=all(x<0 for x in main['seed_differences']),block_interval_better_strong=main['hi95']<0,mean_better_controls=all(smap[llm]['mse']<smap[name]['mse'] for name in ['Original','NoCalendar','RawCalendar',lockfile['shuffled_arm']]),prompt0_better_strong=smap['Prompt0']['mse']<smap[analytic]['mse'],prompt1_better_strong=smap['Prompt1']['mse']<smap[analytic]['mse'])
  dump(out/'gate.json',dict(passed=all(checks.values()),checks=checks,llm_arm=llm,analytic_arm=analytic,decision='Run predeclared KD' if all(checks.values()) else 'Skip all KD; confirm unchanged direct models'))
 dump(out/'completion.json',dict(commit=commit(),stage=stage,target_interval=[lo,hi],n_origins=len(pn['origin']),channels=1,pred_len=96,bootstrap_block_days=14,bootstrap_replicates=c['bootstrap_replicates'],data_sha256=sha(c['data_path']),models_after_sha_valid=all(sha(m['path'])==m['sha256'] for m in models)))
 print('EVALUATED',stage,json.dumps(summary),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit','lock','pilot','confirmation','fit_kd','lock_kd']);p.add_argument('--config',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda:3');args=p.parse_args();c=json.loads(Path(args.config).read_text());r=Path(c['run'])
 try:
  if args.stage=='prepare':prepare(c,r)
  elif args.stage in ['fit','fit_kd']:fit(c,r,args.seed,args.device,args.stage=='fit_kd')
  elif args.stage in ['lock','lock_kd']:lock(c,r,args.stage=='lock_kd')
  else:evaluate(c,r,args.stage,args.device)
 except BaseException as e:
  if r.exists():dump(r/f'failure_{args.stage}_seed{args.seed}.json',dict(error=str(e),traceback=traceback.format_exc()))
  raise
if __name__=='__main__':main()

def initialize_calendar_from_retained(student, retained):
 """Preserve a retained constant-calendar predictor exactly at student initialization."""
 with torch.no_grad():
  effects=retained.readout(retained.codes())
  if not torch.equal(effects,effects[:1].expand_as(effects)):
   raise ValueError("Retained calendar effects must be constant across identities")
  student.body.load_state_dict(retained.body.state_dict())
  student.readout.weight.zero_()
  student.readout.bias.copy_(effects[0])
 return student
