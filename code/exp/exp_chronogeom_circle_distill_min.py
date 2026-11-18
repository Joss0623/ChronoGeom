"""Bounded timestamp-relation distillation pilot inside TSLib; no public trainer edits."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import argparse,copy,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from exp.exp_chronogeom_clockbase import make_base,dump,sha,commit,csvsave,pack_real,select,tensors

def unit_codes(z):
 return F.normalize(z-z.mean(0,keepdim=True),dim=1,eps=1e-6)

class CircleStudent(nn.Module):
 def __init__(self,c,seed):
  super().__init__();self.body=make_base(c)
  torch.manual_seed(seed+10000)
  self.time_mlp=nn.Sequential(nn.Linear(2,16),nn.GELU(),nn.Linear(16,4))
  self.gate=nn.Parameter(torch.zeros(()));self.temperature=c['temperature']
  ids=torch.arange(168);self.register_buffer('calendar',torch.stack(((ids//24)/3.-1.,(ids%24)/11.5-1.),1))
 def codes(self):return unit_codes(self.time_mlp(self.calendar))
 def forward(self,x,ids):
  e=self.codes();z=e[ids];k=F.softmax(torch.bmm(z,z.transpose(1,2))/self.temperature,dim=-1)
  mixed=x+torch.tanh(self.gate)*(torch.bmm(k,x)-x)
  return self.body(mixed),e

class DirectedCircleStudent(CircleStudent):
 def __init__(self,c,seed):
  super().__init__(c,seed);self.directional_beta=nn.Parameter(torch.zeros(2))
 def forward(self,x,ids):
  e=self.codes();z=e[ids];pairs=z.reshape(z.shape[0],z.shape[1],2,2);real=pairs[...,0];imag=pairs[...,1]
  score=torch.bmm(z,z.transpose(1,2))+torch.bmm(real*self.directional_beta,imag.transpose(1,2))-torch.bmm(imag*self.directional_beta,real.transpose(1,2))
  k=F.softmax(score/self.temperature,dim=-1);mixed=x+torch.tanh(self.gate)*(torch.bmm(k,x)-x)
  return self.body(mixed),e

def make_student(c,seed):return DirectedCircleStudent(c,seed) if c.get('directional') else CircleStudent(c,seed)

def phase_response(e,shifts,eps):
 # Real/imaginary parts of conjugate(q_t)*q_(t+delta), one plane per factor.
 q=F.normalize(e.reshape(168,2,2),dim=-1,eps=eps);index=(torch.arange(168,device=e.device)[None,:]+torch.tensor(shifts,device=e.device)[:,None])%168
 later=q[index];current=q[None,:,:,:]
 return torch.stack(((current*later).sum(-1),current[...,0]*later[...,1]-current[...,1]*later[...,0]),-1)

def targets(c,tables,device):
 result={}
 for name in tables.files:
  if name=='permutation':continue
  e=torch.tensor(tables[name],device=device)
  result[name]=phase_response(e,c['phase_shifts'],c['phase_eps']) if c.get('directional') and c.get('objective','phase')!='gram' and name!='Gram' else e@e.T
 return result

def auxiliary(c,e,arm,target):
 if arm=='None':arm=c.get('primary','LLM')
 actual=phase_response(e,c['phase_shifts'],c['phase_eps']) if c.get('directional') and c.get('objective','phase')!='gram' and arm!='Gram' else e@e.T
 return F.mse_loss(actual,target[arm])

def pack(values,ids,lo,hi,c):
 a=pack_real(values,ids%24,lo,hi,c,24);a['ids']=np.repeat(np.stack([ids[o-c['seq_len']:o] for o in np.unique(a['origin'])]),values.shape[1],axis=0);return a

def predict(m,p):
 out=[];m.eval()
 with torch.no_grad():
  for i in range(0,len(p['x']),128):out.append(m(p['x'][i:i+128],p['ids'][i:i+128])[0].cpu())
 return torch.cat(out).numpy()

def balanced_teacher(c,r):
 # Factor means from existing full hidden vectors; no forecast or sinusoidal labels.
 cache=Path(c['raw_cache']);records=json.loads((cache/'prompts.json').read_text());raw=np.load(cache/'raw_spans.npz')['hidden'];grids=[]
 for template in [0,1]:
  ix=[i for i,v in enumerate(records) if v['purpose']=='lookup' and v['template']==template]
  assert [(records[i]['day'],records[i]['hour']) for i in ix]==[(d,h) for d in range(7) for h in range(24)]
  z=raw[ix,0].astype('float64').reshape(7,24,-1);grids.append(z-z.mean((0,1),keepdims=True))
 z=np.mean(grids,axis=0);hour=z.mean(0);week=z.mean(1)
 assets={};scores={};diagnostics={}
 for name,v in [('hour',hour),('weekday',week)]:
  v=v-v.mean(0);u,s,vt=np.linalg.svd(v,full_matrices=False);assert s[1]>s[0]*1e-8
  score=u[:,:2]*s[:2];rms=np.sqrt(np.mean(np.sum(score*score,axis=1)));scores[name]=score/rms
  assets[name+'_basis']=vt[:2];assets[name+'_scores']=score;assets[name+'_rms']=np.array(rms)
  diagnostics[name]=dict(pca2_explained_energy=float(np.sum(s[:2]**2)/np.sum(s*s)),min_score_norm=float(np.linalg.norm(scores[name],axis=1).min()))
 h=np.tile(scores['hour'],(7,1));w=np.repeat(scores['weekday'],24,axis=0)
 # Compensate global scalar after unchanged joint row normalization, without changing angles.
 def normalized(log_scale):
  a=np.c_[h*np.exp(log_scale),w];a-=a.mean(0);return a/np.linalg.norm(a,axis=1,keepdims=True)
 lo,hi=-20.,20.
 assert np.mean(normalized(lo)[:,:2]**2)*2<.5<np.mean(normalized(hi)[:,:2]**2)*2
 for _ in range(100):
  mid=(lo+hi)/2
  if np.mean(np.sum(normalized(mid)[:,:2]**2,axis=1))<.5:lo=mid
  else:hi=mid
 log_scale=(lo+hi)/2;raw_balanced=np.c_[h*np.exp(log_scale),w];e=unit_codes(torch.tensor(raw_balanced,dtype=torch.float32)).numpy()
 energy=np.sum(e*e,axis=0).reshape(2,2).sum(1)/168;assert np.max(abs(energy-.5))<1e-6
 diagnostics.update(layer=6,templates=[0,1],hour_scale_after_initial_rms=float(np.exp(log_scale)),final_factor_energy=dict(hour=float(energy[0]),weekday=float(energy[1])),method='PCA2 of factor means in full layer6 hidden, global factor RMS then scalar balance under unchanged row-unit policy; no sinusoidal coordinate fitting')
 assets['raw_balanced']=raw_balanced;assets['final_codes']=e;assets['hour_scale']=np.array(np.exp(log_scale));np.savez_compressed(r/'balanced_extraction.npz',**assets);dump(r/'balanced_extraction.json',diagnostics)
 return e

def prepare(c,r):
 import pandas as pd
 r.mkdir(parents=True,exist_ok=False);dump(r/'config.json',c)
 a=np.load(c['prepared_data']);dates=pd.to_datetime(pd.read_csv(c['data_path'],usecols=['date'],nrows=c['val_end'])['date'])
 assert (dates.diff().dropna()==pd.Timedelta(hours=1)).all()
 ids=(dates.dt.dayofweek.to_numpy()*24+dates.dt.hour.to_numpy()).astype('int64')
 assert np.array_equal(ids%24,a['hour'])
 np.savez_compressed(r/'train_val.npz',**{k:a[k] for k in ['values','mean','sd']},ids=ids)
 z=np.load(c['lookup']);assert int(z['selected_layer'])==6;llm=z['llm4'];d=np.repeat(np.arange(7),24);h=np.tile(np.arange(24),7)
 f=np.c_[np.cos(2*np.pi*h/24),np.sin(2*np.pi*h/24),np.cos(2*np.pi*d/7),np.sin(2*np.pi*d/7)]
 tables={k:unit_codes(torch.tensor(v,dtype=torch.float32)).numpy() for k,v in [('LLM',llm),('SinCos',f)]}
 perm=np.random.default_rng(c['shuffle_seed']).permutation(168);tables['Shuffle']=tables['LLM'][perm]
 if c.get('balanced_teacher'):
  old=np.load(Path(c['reuse_run'])/'teachers.npz')
  for name in ['LLM','SinCos','Shuffle']:assert np.array_equal(tables[name],old[name])
  tables['Balanced']=balanced_teacher(c,r);tables['BalancedShuffle']=tables['Balanced'][perm]
 if c.get('directional'):
  frozen=np.load(Path(c['frozen_teacher_run'])/'teachers.npz')
  tables={'LLMDir':frozen['Balanced'],'Analytic':frozen['SinCos'],'ShuffleDir':frozen['BalancedShuffle'],'Gram':frozen['Balanced']}
  if c.get('objective')=='gram':tables={'Gram':frozen['Balanced'],'SinCosGram':frozen['SinCos'],'ShuffleGram':frozen['BalancedShuffle']}
  assert np.array_equal(perm,frozen['permutation'])
 np.savez_compressed(r/'teachers.npz',**tables,permutation=perm)
 diagnostics=[]
 for name,z in tables.items():
  g=z@z.T;assert np.isfinite(g).all() and np.max(abs(np.diag(g)-1))<1e-5
  diagnostics.append(dict(teacher=name,dimension=4,rank=int(np.linalg.matrix_rank(z)),code_rms=float(np.sqrt(np.mean(np.sum(z*z,axis=1)))),kernel_rms=float(np.sqrt(np.mean(g*g)))))
 csvsave(r/'teacher_diagnostics.csv',diagnostics)
 warm=[]
 for seed in c['seeds']:
  p=Path(c['warm_root'])/f'seed{seed}'/'Pretrain.pt';warm.append(dict(seed=seed,path=str(p),sha256=sha(p)))
 # Local mechanistic fixtures before fitting: exact identity, live finite gradients, actual capacity.
 torch.manual_seed(0);m=make_student(c,0);x=torch.randn(4,96,1);ix=torch.arange(96).repeat(4,1)
 p,e=m(x,ix);assert torch.equal(p,m.body(x));assert sum(p.numel() for p in m.parameters())==(159 if c.get('directional') else 157)
 with torch.no_grad():m.gate.fill_(.1)
 loss=m(x,ix)[0].square().mean();loss.backward()
 assert m.time_mlp[0].weight.grad is not None and torch.isfinite(m.time_mlp[0].weight.grad).all() and m.time_mlp[0].weight.grad.abs().sum()>0
 assert torch.isfinite(m.gate.grad)
 if c.get('directional'):
  assert m.directional_beta.grad is not None and torch.isfinite(m.directional_beta.grad).all() and torch.all(m.directional_beta.grad.abs()>0)
  ideal=torch.tensor(tables['SinCosGram'] if c.get('objective')=='gram' else tables['Analytic']);u=phase_response(ideal,c['phase_shifts'],c['phase_eps']);ids0=torch.arange(168)
  for i,delta in enumerate(c['phase_shifts']):
   daystep=((ids0+delta)%168)//24-ids0//24
   angle=torch.stack((torch.full((168,),2*torch.pi*delta/24),2*torch.pi*daystep/7),-1)
   exact=torch.stack((angle.cos(),angle.sin()),-1);assert float((u[i]-exact).abs().max())<2e-6
  plus=phase_response(ideal,[1],c['phase_eps'])[0];minus=phase_response(ideal,[-1],c['phase_eps'])[0];assert float((plus[:,0,1]+minus[:,0,1]).abs().max())<2e-6
  assert float(plus[:,0,1].abs().mean())>.2
 dump(r/'source.json',dict(commit=commit(),script_sha256=sha(__file__),teacher_lookup_sha256=sha(c['lookup']),prepared_data_sha256=sha(c['prepared_data']),data_sha256=sha(c['data_path']),timebase_sha256=sha('layers/ChronoGeomTimeBaseOfficial.py'),warmstarts=warm,fixture_identity_exact=True,fixture_finite_live_gradients=True,parameters=sum(p.numel() for p in m.parameters())))
 if c.get('balanced_teacher'):
  source=json.loads((r/'source.json').read_text());source.update(raw_hidden_sha256=sha(Path(c['raw_cache'])/'raw_spans.npz'),prompts_sha256=sha(Path(c['raw_cache'])/'prompts.json'),reused_source_sha256=sha(Path(c['reuse_run'])/'source.json'),old_teacher_arrays_exact=True,student_ast_sha256=c['student_ast_sha256']);dump(r/'source.json',source)
 if c.get('directional'):
  source=json.loads((r/'source.json').read_text());source.update(frozen_teacher_sha256=sha(Path(c['frozen_teacher_run'])/'teachers.npz'),phase_shifts=c['phase_shifts'],directional_gradient_and_analytic_rotation_checks=True,parent_student_ast_sha256=c['parent_student_ast_sha256']);dump(r/'source.json',source)
 print('PREPARED',json.dumps(diagnostics),flush=True)

def train(c,r,seed,device):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.1,device)
 out=r/f'seed{seed}';out.mkdir(exist_ok=False);a=np.load(r/'train_val.npz');values=torch.tensor(a['values'],device=device);ids=torch.tensor(a['ids'],device=device)
 va=tensors(pack(a['values'],a['ids'],c['train_end'],c['val_end'],c),device)
 tables=np.load(r/'teachers.npz');target=targets(c,tables,device)
 warmrec=next(x for x in json.loads((r/'source.json').read_text())['warmstarts'] if x['seed']==seed);assert sha(warmrec['path'])==warmrec['sha256']
 state=torch.load(warmrec['path'],map_location=device,weights_only=True);state={k.removeprefix('body.'):v for k,v in state.items()}
 records=[];start=time.monotonic()
 for arm in c['arms']:
  if arm in c.get('reuse_arms',[]):
   old=Path(c['reuse_run'])/f'seed{seed}';oldrecords=json.loads((old/'completion.json').read_text())['records']
   for rec in oldrecords:
    if rec['arm']==arm:
     assert sha(rec['path'])==rec['sha256'];rec=dict(rec,reused_from=str(old),new_fit_seconds=0.)
     import shutil
     shutil.copy2(old/f"{arm}_lr{rec['lr']}_curve.json",out/f"{arm}_lr{rec['lr']}_curve.json")
     records.append(rec);print('REUSED',arm,seed,rec['lr'],flush=True)
   continue
  for lr in c['learning_rates']:
   torch.manual_seed(seed);m=make_student(c,seed).to(device);m.body.load_state_dict(state)
   opt=torch.optim.Adam(m.parameters(),lr=lr);rng=np.random.default_rng(seed+271);best=float('inf');beststate=None;curve=[];t0=time.monotonic()
   with torch.no_grad():assert torch.equal(m(va['x'][:4],va['ids'][:4])[0],m.body(va['x'][:4]))
   for step in range(c['steps']+1):
    if time.monotonic()-start>c['seconds_per_seed']:raise TimeoutError('bounded total per-seed fitting budget exceeded')
    if step%c['validation_every']==0:
     y=predict(m,va);val=float(np.mean((y-va['y'].cpu().numpy())**2));assert np.isfinite(val)
     with torch.no_grad():e=m.codes();aux=float(auxiliary(c,e,arm,target))
     curve.append(dict(step=step,val_mse=val,teacher_relation_mse=aux))
     if val<best:best=val;beststate=copy.deepcopy(m.state_dict());beststep=step
    if step==c['steps']:break
    origins=torch.tensor(rng.integers(96,c['train_end']-96+1,size=c['batch']),device=device);ch=torch.tensor(rng.integers(16,size=c['batch']),device=device)
    past=origins[:,None]+torch.arange(-96,0,device=device);future=origins[:,None]+torch.arange(96,device=device)
    x=values[past,ch[:,None]][:,:,None];truth=values[future,ch[:,None]][:,:,None]
    m.train();y,e=m(x,ids[past]);forecast=F.mse_loss(y,truth);aux=e.new_zeros(()) if arm=='None' else auxiliary(c,e,arm,target);loss=forecast+c['lambda']*aux
    if not torch.isfinite(loss):raise FloatingPointError(f'{arm} lr{lr} step{step} nonfinite loss')
    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
   m.load_state_dict(beststate);path=out/f'{arm}_lr{lr}.pt';torch.save(beststate,path);dump(out/f'{arm}_lr{lr}_curve.json',curve)
   with torch.no_grad():e=m.codes();aux=float(auxiliary(c,e,arm,target))
   rec=dict(seed=seed,arm=arm,lr=lr,best_step=beststep,val_mse=best,relation_mse=aux,gate=float(torch.tanh(m.gate)),parameters=sum(p.numel() for p in m.parameters()),seconds=time.monotonic()-t0,path=str(path),sha256=sha(path));
   if c.get('directional'):rec['directional_beta']=m.directional_beta.detach().cpu().tolist()
   records.append(rec);print(json.dumps(rec),flush=True)
 dump(out/'completion.json',dict(commit=commit(),seed=seed,device=device,records=records,seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated(device)/2**30))
 print('COMPLETE',seed,flush=True)

def evaluate(c,r,device):
 import pandas as pd
 torch.set_num_threads(2);assert not (r/'test_lock.json').exists(),'do not reopen/reselect pilot test'
 completions=[json.loads((r/f'seed{s}'/'completion.json').read_text()) for s in c['seeds']];selected=[]
 for d in completions:
  for arm in c['arms']:selected.append(min([v for v in d['records'] if v['arm']==arm],key=lambda x:x['val_mse']))
 assert len(selected)==len(c['arms'])*len(c['seeds'])
 for x in selected:assert sha(x['path'])==x['sha256']
 dump(r/'test_lock.json',dict(commit=commit(),models=selected,test_status=c['test_status']))
 a=np.load(r/'train_val.npz');df=pd.read_csv(c['data_path'],usecols=['date']+c['columns']);dates=pd.to_datetime(df['date']);assert len(df)==17544
 ids=(dates.dt.dayofweek.to_numpy()*24+dates.dt.hour.to_numpy()).astype('int64');values=(df[c['columns']].to_numpy(dtype='float32')-a['mean'])/a['sd']
 test=pack(values,ids,c['val_end'],len(df),c);t=tensors(test,device);truth=test['y'][:,:,0]
 saved=dict(truth=truth,origin=test['origin'],channel=test['channel']);scores=[];modelnotes=[]
 for rec in selected:
  m=make_student(c,rec['seed']).to(device);m.load_state_dict(torch.load(rec['path'],map_location=device,weights_only=True));y=predict(m,t)[:,:,0];saved[f"{rec['arm']}_s{rec['seed']}"]=y
  scores.append(dict(seed=rec['seed'],arm=rec['arm'],mse=float(np.mean((y-truth)**2)),mae=float(np.mean(abs(y-truth))),n=len(y)))
  with torch.no_grad():e=m.codes();np.savez_compressed(r/f"codes_{rec['arm']}_s{rec['seed']}.npz",codes=e.cpu().numpy());modelnotes.append(dict(arm=rec['arm'],seed=rec['seed'],gate=float(torch.tanh(m.gate)),code_rank=int(torch.linalg.matrix_rank(e)),code_norm_min=float(e.norm(dim=1).min()),code_norm_max=float(e.norm(dim=1).max())))
  if c.get('directional'):modelnotes[-1]['directional_beta']=m.directional_beta.detach().cpu().tolist()
 ref=np.load(c['baseline_predictions']);assert np.array_equal(ref['truth'],truth) and np.array_equal(ref['origin'],test['origin'])
 for seed in c['seeds']:
  y=ref[f'TimeBase_s{seed}'];saved[f'TimeBase_s{seed}']=y;scores.append(dict(seed=seed,arm='TimeBase',mse=float(np.mean((y-truth)**2)),mae=float(np.mean(abs(y-truth))),n=len(y)))
 np.savez_compressed(r/'predictions.npz',**saved);csvsave(r/'scores.csv',scores);csvsave(r/'model_notes.csv',modelnotes)
 aggregate=[]
 for arm in ['TimeBase']+c['arms']:
  ar=[x for x in scores if x['arm']==arm];aggregate.append(dict(arm=arm,mse=float(np.mean([x['mse'] for x in ar])),mae=float(np.mean([x['mae'] for x in ar])),mse_sd=float(np.std([x['mse'] for x in ar],ddof=1))))
 csvsave(r/'aggregate.csv',aggregate)
 errors={arm:np.mean([((saved[f'{arm}_s{s}'].astype(float)-truth)**2).mean(1) for s in c['seeds']],axis=0) for arm in ['TimeBase']+c['arms']}
 days=np.unique(test['origin']);block=np.arange(len(days))//7;blocks=np.unique(block);rng=np.random.default_rng(c['bootstrap_seed']);choice=rng.integers(len(blocks),size=(2000,len(blocks)));counts=np.array([(block==b).sum() for b in blocks]);contrasts=[]
 primary=c.get('primary','LLM')
 for other in ['TimeBase']+[x for x in c['arms'] if x!=primary]:
  diff=errors[primary]-errors[other];daily=np.array([diff[test['origin']==d].mean() for d in days]);sums=np.array([daily[block==b].sum() for b in blocks]);draw=sums[choice].sum(1)/counts[choice].sum(1);lo,hi=np.quantile(draw,[.025,.975]);contrasts.append(dict(contrast=primary+'-'+other,difference=float(diff.mean()),lo95=float(lo),hi95=float(hi),blocks=len(blocks),origins=len(days)))
 csvsave(r/'contrasts.csv',contrasts)
 cases=[]
 for other in ['TimeBase']+[x for x in c['arms'] if x!=primary]:
  delta=errors[primary]-errors[other]
  for label,ix in [('largest_gain',np.argsort(delta)[:10]),('largest_harm',np.argsort(delta)[-10:][::-1])]:
   cases.extend(dict(comparison=primary+'-'+other,selection=label,index=int(i),origin=int(test['origin'][i]),channel=int(test['channel'][i]),primary_mse=float(errors[primary][i]),other_mse=float(errors[other][i]),difference=float(delta[i])) for i in ix)
 csvsave(r/'gain_and_failure_cases.csv',cases)
 assert all(sha(x['path'])==x['sha256'] for x in selected)
 dump(r/'completion.json',dict(commit=commit(),selected=len(selected),candidate_fits=sum(len(d['records']) for d in completions),new_fits=sum('reused_from' not in x for d in completions for x in d['records']),n_instances=len(truth),n_origins=len(days),blocks=len(blocks),sum_fit_seconds=sum(x['seconds'] for d in completions for x in d['records'] if 'reused_from' not in x),all_selected_hashes_unchanged=True,test_status=c['test_status']))
 print('AGGREGATE',json.dumps(aggregate),flush=True);print('CONTRASTS',json.dumps(contrasts),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','evaluate']);p.add_argument('--config',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda:3');a=p.parse_args();c=json.loads(Path(a.config).read_text());r=Path(c['run'])
 try:
  if a.action=='prepare':prepare(c,r)
  elif a.action=='train':train(c,r,a.seed,a.device)
  else:evaluate(c,r,a.device)
 except BaseException as ex:
  if r.exists():dump(r/f'failure_{a.action}_s{a.seed}.json',dict(error=str(ex),traceback=traceback.format_exc()))
  raise
