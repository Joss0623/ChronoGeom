"""Small ClockBase experiment inside the pinned TSLib checkout; no generic framework."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,copy,csv,hashlib,json,math,subprocess,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from layers.ChronoGeomTimeBaseOfficial import Model as TimeBase
ROOT=Path(__file__).resolve().parents[1]

def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def csvsave(p,rows):
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def make_base(c):return TimeBase(SimpleNamespace(seq_len=c['seq_len'],pred_len=c['pred_len'],enc_in=1,period_len=c['period'],basis_num=c['basis'],individual=False,use_period_norm=True,use_orthogonal=False))

def shift(x,delta):
 """Real orthogonal circular action; DC/Nyquist carried unchanged, paired modes rotate.
 Subtract identity in frequency space so zero transport is bitwise identity with live gradients.
 """
 p=x.shape[-1];omega=2*torch.pi*torch.fft.rfftfreq(p,device=x.device)
 if p%2==0:omega[-1]=0 # a single real Nyquist coordinate cannot carry continuous rotations
 rotation=torch.exp(-1j*delta[...,None]*omega)
 return x+torch.fft.irfft(torch.fft.rfft(x,dim=-1)*(rotation-1),n=p,dim=-1)

def correlations(x,p):
 with torch.no_grad():
  a=x.transpose(1,2).reshape(x.shape[0],x.shape[2],-1,p)[:,:,-4:]
  a=a-a.mean(-1,keepdim=True);a=a/a.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-4)
  return torch.stack([(a[:,:,1:]*torch.roll(a[:,:,:-1],lag,-1)).mean((-1,-2)) for lag in range(-4,5)],-1)

class ClockForecaster(nn.Module):
 def __init__(self,c,arm,body):
  super().__init__();self.body=copy.deepcopy(body);self.arm=arm;self.c=c
  if arm=='ClockBase':self.head=nn.Linear(9,1)
  elif arm=='SinCos':self.head=nn.Linear(4,2)
  elif arm in ['MatchedMix','PhaseMix']:self.head=nn.Conv1d(1,1,9)
  else:self.head=None
  if self.head is not None:
   nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x,hour,oracle=None,return_speed=False):
  b,l,ch=x.shape;p=self.c['period'];n=l//p;m=self.c['pred_len']//p
  v=x.new_zeros((b,ch));a=x
  if self.arm in ['ClockBase','OracleClock']:
   v=self.c['clock_max_shift']*torch.tanh(self.head(correlations(x,p)).squeeze(-1)) if self.arm=='ClockBase' else oracle.reshape(b,ch)
   past=torch.arange(1-n,1,device=x.device,dtype=x.dtype)
   z=x.transpose(1,2).reshape(b,ch,n,p);a=shift(z,-v[...,None]*past).reshape(b,ch,l).transpose(1,2)
  elif self.arm in ['MatchedMix','PhaseMix']:
   z=x.transpose(1,2).reshape(-1,1,p);a=(z+self.head(F.pad(z,(4,4),mode='circular'))).reshape(b,ch,l).transpose(1,2)
  y=self.body(a)
  if self.arm in ['ClockBase','OracleClock']:
   future=torch.arange(1,m+1,device=x.device,dtype=x.dtype)
   y=shift(y.transpose(1,2).reshape(b,ch,m,p),v[...,None]*future).reshape(b,ch,-1).transpose(1,2)
  elif self.arm=='SinCos':
   mean=x.mean(1);sd=x.std(1,unbiased=False);last=x[:,-1]-mean;slope=x[:,l//2:].mean(1)-x[:,:l//2].mean(1)
   coeff=self.head(torch.stack([mean,sd,last,slope],-1))
   theta=2*torch.pi*hour/p;y=y+torch.sin(theta)[...,None]*coeff[...,0][:,None,:]+torch.cos(theta)[...,None]*coeff[...,1][:,None,:]
  return (y,v) if return_speed else y

def synth(c,split):
 rng=np.random.default_rng(c['data_seed']+{'train':0,'val':1,'test':2}[split]);xs=[];ys=[];clean=[];hours=[];speeds=[];regimes=[]
 p=c['period'];n=c['seq_len']//p;m=c['pred_len']//p;j=np.arange(1-n,m+1);s=np.arange(p)
 count=c[split+'_per_regime']
 for ri,name in enumerate(c['synthetic_regimes']):
  for _ in range(count):
   theta=rng.uniform(-np.pi,np.pi);b=rng.uniform(-.5,.5);amplitude=rng.uniform(.7,1.3);a2=rng.uniform(-np.pi,np.pi);a3=rng.uniform(-np.pi,np.pi)
   v=rng.uniform(-2,2) if 'drift' in name else 0.;gamma=rng.uniform(-.04,.04) if name=='amplitude_only' else 0.
   if name=='continuous_drift':phase=theta+2*np.pi*(j[:,None]*p+s[None,:])/(p+v)
   else:phase=theta+2*np.pi*(s[None,:]-v*j[:,None])/p
   signal=b+amplitude*(1+gamma*j[:,None])*(np.cos(phase)+.35*np.sin(2*phase+a2)+.20*np.cos(3*phase+a3))
   noisy=signal+rng.normal(0,.05,signal.shape)
   xs.append(noisy[:n].reshape(-1,1));ys.append(noisy[n:].reshape(-1,1));clean.append(signal[:n].reshape(-1,1));speeds.append(v);regimes.append(ri)
   h0=rng.integers(0,p);hours.append((h0+np.arange(m*p))%p)
 return {k:np.asarray(v,dtype='int64' if k=='regime' else 'float32') for k,v in dict(x=xs,y=ys,clean=clean,hour=hours,speed=speeds,regime=regimes).items()}

def tensors(a,device):return {k:torch.as_tensor(v,device=device) for k,v in a.items()}
def select(a,ix):return {k:v[ix] for k,v in a.items()}
def pack_real(v,h,lo,hi,c,stride):
 l=c['seq_len'];H=c['pred_len'];origins=np.arange(max(lo,l),hi-H+1,stride);ch=v.shape[1]
 x=np.concatenate([v[o-l:o].T for o in origins])[:,:,None];y=np.concatenate([v[o:o+H].T for o in origins])[:,:,None]
 hour=np.concatenate([np.tile(h[o:o+H],(ch,1)) for o in origins])
 return dict(x=x.astype('float32'),y=y.astype('float32'),hour=hour.astype('float32'),origin=np.repeat(origins,ch),channel=np.tile(np.arange(ch),len(origins)))

def prepare(c,r,stage):
 if stage=='synthetic':
  r.mkdir(parents=True,exist_ok=False);dump(r/'config.json',c);dump(r/'source.json',dict(commit=commit(),script_sha256=sha(__file__),timebase_sha256=sha(ROOT/'layers/ChronoGeomTimeBaseOfficial.py'),upstream=c['upstream_tslib']))
  d=r/'synthetic';d.mkdir()
  for split in ['train','val']:np.savez_compressed(d/(split+'.npz'),**synth(c,split))
 else:
  assert (r/'synthetic/evaluation_completion.json').exists(),'Synthetic experiment must finish first'
  import pandas as pd
  d=r/'real';d.mkdir(exist_ok=False)
  df=pd.read_csv(c['real_path'],usecols=['date']+c['real_columns'],nrows=c['real_val_end']);dates=pd.to_datetime(df['date']);assert (dates.diff().dropna()==pd.Timedelta(hours=1)).all()
  v=df[c['real_columns']].to_numpy(dtype='float32');mean=v[:c['real_train_end']].mean(0);sd=v[:c['real_train_end']].std(0);assert (sd>1e-6).all();v=(v-mean)/sd
  h=dates.dt.hour.to_numpy();np.savez_compressed(d/'train_val.npz',values=v,hour=h,mean=mean,sd=sd)
  dump(d/'data_manifest.json',dict(path=c['real_path'],sha256=sha(c['real_path']),columns=c['real_columns'],rows=c['real_rows'],train=[0,c['real_train_end']],val=[c['real_train_end'],c['real_val_end']],test=[c['real_val_end'],c['real_rows']],first_timestamp=str(dates.iloc[0]),scope='Fixed first16 Traffic sensors, no semantic weekday claim; no test parsed at preparation'))
 print('prepared',stage,flush=True)

def predict(model,p):
 out=[];vs=[]
 model.eval()
 with torch.no_grad():
  for i in range(0,len(p['x']),256):
   b=select(p,slice(i,i+256));y,v=model(b['x'],b['hour'],b.get('speed'),True);out.append(y.cpu());vs.append(v.cpu())
 return torch.cat(out).numpy(),torch.cat(vs).numpy()

def fit(c,r,stage,seed,device,refine=False):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(seed);torch.cuda.set_per_process_memory_fraction(.10,device)
 d=r/stage;out=d/(f'seed{seed}_lr0003' if refine else f'seed{seed}');out.mkdir(exist_ok=False);start=time.monotonic();cap=c['syn_seconds_per_seed'] if stage=='synthetic' else c['real_seconds_per_seed']
 if stage=='synthetic':
  tr=tensors(dict(np.load(d/'train.npz')),device);va=tensors(dict(np.load(d/'val.npz')),device)
  def sample(rng):return select(tr,torch.as_tensor(rng.integers(len(tr['x']),size=c['batch']),device=device))
 else:
  a=np.load(d/'train_val.npz');values=torch.as_tensor(a['values'],device=device);hours=torch.as_tensor(a['hour'],device=device)
  va=tensors(pack_real(a['values'],a['hour'],c['real_train_end'],c['real_val_end'],c,24),device)
  def sample(rng):
   o=torch.as_tensor(rng.integers(c['seq_len'],c['real_train_end']-c['pred_len']+1,size=c['batch']),device=device);ch=torch.as_tensor(rng.integers(len(c['real_columns']),size=c['batch']),device=device)
   ti=o[:,None]+torch.arange(-c['seq_len'],0,device=device);fi=o[:,None]+torch.arange(c['pred_len'],device=device)
   return dict(x=values[ti,ch[:,None]][:,:,None],y=values[fi,ch[:,None]][:,:,None],hour=hours[fi].float())
 records=[]
 if refine:
  assert stage=='real' and not (d/'test_lock.json').exists()
  start-=json.loads((d/f'seed{seed}/completion.json').read_text())['seconds']
 def one(model,label,steps):
  opt=torch.optim.Adam(model.parameters(),lr=c['lr']);rng=np.random.default_rng(seed+271);curve=[];best=float('inf');beststate=None;beststep=0;t0=time.monotonic()
  for step in range(steps+1):
   if time.monotonic()-start>cap:raise TimeoutError('ClockBase bounded stage/seed budget exceeded')
   if step%c['validation_every']==0:
    pred,_=predict(model,va);val=float(np.mean((pred-va['y'].cpu().numpy())**2));assert np.isfinite(val)
    curve.append(dict(step=step,val_mse=val))
    if val<best:best=val;beststate=copy.deepcopy(model.state_dict());beststep=step
   if step==steps:break
   model.train();b=sample(rng);y=model(b['x'],b['hour'],b.get('speed'));loss=F.mse_loss(y,b['y'])
   if not torch.isfinite(loss):raise FloatingPointError(label+' nonfinite loss')
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
  model.load_state_dict(beststate);path=out/(label+'.pt');torch.save(beststate,path);dump(out/(label+'_curve.json'),curve)
  rec=dict(stage=stage,seed=seed,arm=label,parameters=sum(p.numel() for p in model.parameters()),steps_attempted=steps,best_step=beststep,val_mse=best,seconds=time.monotonic()-t0,sha256=sha(path),path=str(path));records.append(rec);print(json.dumps(rec),flush=True)
 base=make_base(c).to(device);warm=ClockForecaster(c,'TimeBase',base).to(device)
 if refine:warm.load_state_dict(torch.load(d/f'seed{seed}/Pretrain.pt',map_location=device,weights_only=True))
 else:one(warm,'Pretrain',c['syn_pretrain_steps'] if stage=='synthetic' else c['real_pretrain_steps'])
 base=warm.body
 arms=c['arms']+(['OracleClock'] if stage=='synthetic' else [])
 for arm in arms:
  model=ClockForecaster(c,arm,base).to(device);one(model,arm,c['syn_steps'] if stage=='synthetic' else c['real_steps'])
 counts={x['parameters'] for x in records if x['arm'] in ['SinCos','MatchedMix','PhaseMix','ClockBase']};assert len(counts)==1
 dump(out/'completion.json',dict(commit=commit(),device=device,seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,records=records));csvsave(out/'fits.csv',records)

def metrics(y,t,p):
 err=(y-t)**2;phasep=np.fft.rfft(y.reshape(len(y),-1,p),axis=-1)[:,:,1];phaset=np.fft.rfft(t.reshape(len(t),-1,p),axis=-1)[:,:,1]
 valid=(abs(phasep)*2/p>=.1)&(abs(phaset)*2/p>=.1);angles=np.rad2deg(abs(np.angle(phasep*np.conj(phaset))))
 return dict(mse=float(err.mean()),mae=float(abs(y-t).mean()),phase_error_degrees=float(angles[valid].mean()) if valid.any() else None,phase_valid=int(valid.sum()),phase_total=int(valid.size),amplitude_absolute_error=float((abs(abs(phasep)-abs(phaset))*2/p).mean()))

def evaluate(c,r,stage,device):
 d=r/stage;arms=c['arms']+(['OracleClock'] if stage=='synthetic' else []);locks={};paths={};selections=[]
 for seed in c['seeds']:
  assert (d/f'seed{seed}/completion.json').exists()
  records=json.loads((d/f'seed{seed}/completion.json').read_text())['records']
  if stage=='real':
   assert not (d/'test_lock.json').exists(),'Real test already opened'
   records+=json.loads((d/f'seed{seed}_lr0003/completion.json').read_text())['records']
  for arm in arms:
   chosen=min([v for v in records if v['arm']==arm],key=lambda v:v['val_mse']);key=f'{arm}_s{seed}';paths[key]=chosen['path'];locks[key]=sha(chosen['path']);selections.append(chosen)
 dump(d/'selection_before_test.json',selections)
 dump(d/'test_lock.json',dict(commit=commit(),models=locks,paths=paths))
 if stage=='synthetic':test=synth(c,'test');np.savez_compressed(d/'test.npz',**test)
 else:
  import pandas as pd
  a=np.load(d/'train_val.npz');df=pd.read_csv(c['real_path'],usecols=['date']+c['real_columns']);assert len(df)==c['real_rows'];dates=pd.to_datetime(df['date']);v=(df[c['real_columns']].to_numpy(dtype='float32')-a['mean'])/a['sd'];test=pack_real(v,dates.dt.hour.to_numpy(),c['real_val_end'],len(v),c,c['test_stride'])
 t=tensors(test,device);truth=test['y'][:,:,0];saved=dict(truth=truth,origin=test.get('origin',np.arange(len(truth))),regime=test.get('regime',np.zeros(len(truth),dtype=int)));scores=[];instance=[];speed=[]
 for seed in c['seeds']:
  for arm in arms:
   model=ClockForecaster(c,arm,make_base(c)).to(device);path=Path(paths[f'{arm}_s{seed}']);assert sha(path)==locks[f'{arm}_s{seed}'];model.load_state_dict(torch.load(path,map_location=device,weights_only=True));y,v=predict(model,t);y=y[:,:,0];saved[f'{arm}_s{seed}']=y
   groups=c['synthetic_regimes'] if stage=='synthetic' else ['Traffic16']
   for ri,name in enumerate(groups):
    ix=(test['regime']==ri) if stage=='synthetic' else np.ones(len(y),dtype=bool)
    scores.append(dict(stage=stage,seed=seed,arm=arm,group=name,n=int(ix.sum()),**metrics(y[ix],truth[ix],c['period'])))
    if arm=='ClockBase' and stage=='synthetic':speed.append(dict(seed=seed,group=name,estimated_mean=float(v[ix].mean()),true_mean=float(test['speed'][ix].mean()),speed_rmse=float(np.sqrt(np.mean((v[ix,0]-test['speed'][ix])**2)))))
   mse=((y-truth)**2).mean(1)
   for i,e in enumerate(mse):instance.append(dict(seed=seed,arm=arm,index=i,origin=int(saved['origin'][i]),group=int(saved['regime'][i]),channel=int(test['channel'][i]) if stage=='real' else 0,mse=float(e)))
   if arm=='ClockBase':np.savez_compressed(d/f'clock_speed_s{seed}.npz',speed=v[:,0])
   print(stage,seed,arm,'test_mse',float(mse.mean()),flush=True)
 np.savez_compressed(d/'predictions.npz',**saved);csvsave(d/'scores.csv',scores);csvsave(d/'per_instance.csv',instance)
 if speed:csvsave(d/'speed_accuracy.csv',speed)
 dump(d/'evaluation_completion.json',dict(commit=commit(),n_instances=len(truth),n_models=len(locks),test_opened_once=True))

def selfcheck(c):
 torch.set_num_threads(2);torch.manual_seed(0);x=torch.randn(4,c['seq_len'],1);hour=torch.arange(c['pred_len']).repeat(4,1).float()%c['period'];base=make_base(c)
 a=ClockForecaster(c,'TimeBase',base)(x,hour)
 for arm in c['arms'][1:]:assert torch.equal(a,ClockForecaster(c,arm,base)(x,hour)),arm
 z=torch.randn(4,3,24);u=torch.randn(4,3);v=torch.randn(4,3)
 inverse=float((shift(shift(z,u),-u)-z).abs().max());composition=float((shift(shift(z,u),v)-shift(z,u+v)).abs().max());energy=float((shift(z,u).square().sum(-1)-z.square().sum(-1)).abs().max())
 assert inverse<1e-5 and composition<1e-5 and energy<1e-4
 wave=torch.cos(2*torch.pi*torch.arange(24)/24)[None,:];assert torch.allclose(shift(wave,torch.ones(1)),torch.roll(wave,1,-1),atol=1e-6)
 counts={a:sum(p.numel() for p in ClockForecaster(c,a,base).parameters()) for a in c['arms']}
 assert len({counts[a] for a in c['arms'][1:]})==1
 print(json.dumps(dict(initial_predictions_exact=True,inverse_error=inverse,composition_error=composition,energy_error=energy,parameters=counts)),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['check','prepare','fit','evaluate']);p.add_argument('--stage',choices=['synthetic','real'],default='synthetic');p.add_argument('--config',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cpu');p.add_argument('--refine-lr',action='store_true');a=p.parse_args();c=json.loads(Path(a.config).read_text());r=Path(c['run'])
 if a.action=='check':selfcheck(c);return
 try:
  if a.action=='prepare':prepare(c,r,a.stage)
  elif a.action=='fit':
   if a.refine_lr:c['lr']=.0003
   fit(c,r,a.stage,a.seed,a.device,a.refine_lr)
  else:evaluate(c,r,a.stage,a.device)
 except BaseException as ex:
  if r.exists():dump(r/(f'failure_{a.action}_{a.stage}_s{a.seed}.json'),dict(error=str(ex),traceback=traceback.format_exc()))
  raise
if __name__=='__main__':main()
