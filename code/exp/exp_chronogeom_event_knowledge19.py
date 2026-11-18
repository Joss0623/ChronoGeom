"""Small event-knowledge teacher pilot in TSLib; frozen official TimeBase/Qwen."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TORCH_DEVICE_BACKEND_AUTOLOAD='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import argparse,copy,hashlib,json,subprocess,time,traceback,re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from exp.exp_chronogeom_downstream import CalendarTimeBase,predict
from exp.exp_chronogeom_residual03 import numeric_summary

def save(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(): return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def scores(p,y):
 e=np.asarray(p,dtype=float)-np.asarray(y,dtype=float)
 return dict(mse=float(np.mean(e**2)),mae=float(np.mean(abs(e))),level=float(np.mean(e.mean(1)**2)),shape=float(np.mean((e-e.mean(1,keepdims=True))**2)))
def extract(c,r,gpu):
 from transformers import AutoTokenizer,AutoModelForCausalLM
 t=time.monotonic(); device=f'cuda:{gpu}'
 assert torch.cuda.mem_get_info(device)[0]>=21*2**30
 torch.cuda.set_per_process_memory_fraction(22*2**30/torch.cuda.get_device_properties(device).total_memory,device)
 tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':device},attn_implementation='sdpa').eval()
 for p in model.parameters(): p.requires_grad_(False)
 out={}
 for name,domain in c['domains'].items():
  prompt=c['prompt'].replace('{DOMAIN}',domain)
  text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  cached=r/f'{name}_generation.json'
  if cached.exists():
   record=read(cached);assert record['prompt']==prompt and record['chat']==text;ans=record['answer']
  else:
   inputs=tok(text,return_tensors='pt').to(device)
   with torch.inference_mode():
    generated=model.generate(**inputs,do_sample=False,max_new_tokens=512,pad_token_id=tok.eos_token_id)
   ans=tok.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
   record=dict(prompt=prompt,chat=text,answer=ans,tokens=generated[0].tolist())
   save(cached,record)
  match=re.search(r'\{.*\}',ans,re.S); assert match,ans
  obj=json.loads(match.group()); bank=np.array([obj[k] for k in ['workday','weekend','holiday']],dtype=float)
  assert bank.shape==(3,24) and np.isfinite(bank).all() and (bank>=0).all(),obj
  original_max=float(bank.max());bank=bank*10/max(original_max,1e-8)
  save(r/f'{name}_scale_validation.json',dict(original_max=original_max,prompt_range_violation=original_max>10,uniform_scale=10/max(original_max,1e-8),note='Global scale cancels exactly under downstream centered RMS normalization; no clipping/prompt revision.'))
  assert (np.abs(bank-bank.mean(1,keepdims=True)).sum(1)>0).all()
  out[name]=bank.tolist()
  assert time.monotonic()-t<c['extract_seconds']
  print('GENERATED',name,ans,flush=True)
 save(r/'profiles.json',out)
 save(r/'extraction.json',dict(seconds=time.monotonic()-t,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,model=c['model'],revision=c['revision'],git=git(),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),gpu=gpu))

def data(c,r):
 f=pd.read_csv(c['data']);ts=pd.to_datetime(f.dteday)+pd.to_timedelta(f.hr,unit='h')
 assert np.all((ts.dt.dayofweek.to_numpy()+1)%7==f.weekday.to_numpy())
 assert np.all(f.workingday.to_numpy()==((ts.dt.dayofweek.to_numpy()<5)&(f.holiday.to_numpy()==0)))
 assert not f.groupby('dteday').holiday.nunique().gt(1).any()
 grid=pd.date_range('2011-01-01','2013-01-01',freq='h',inclusive='left'); v=pd.Series(f.cnt.to_numpy(),index=ts).reindex(grid).to_numpy(dtype=float)
 holiday=f.groupby('dteday').holiday.first();hol=np.array([holiday.get(str(d.date()),np.nan) for d in grid]);week=(grid.dayofweek>=5).astype(int);kind=np.where(hol==1,2,week)
 scale=np.load(c['scale']); mu=float(scale['mean'][0]);sd=float(scale['sd'][0]);v=(v-mu)/sd
 h=grid.hour.to_numpy();wd=grid.dayofweek.to_numpy();a=2*np.pi*(grid.dayofyear.to_numpy()-1)/366
 raw=np.stack([hol,week,((week==0)&(hol==0)),np.sin(wd*2*np.pi/7),np.cos(wd*2*np.pi/7),np.sin(a),np.cos(a)],1).astype(float)
 base=np.stack([fn(2*np.pi*k*h/24) for k in range(1,5) for fn in [np.sin,np.cos]],1)
 common=np.concatenate([base*(kind==k)[:,None] for k in range(3)],1)
 packs={};counts={}
 for name,(a,b) in c['splits'].items():
  lo=grid.get_loc(pd.Timestamp(a));hi=len(grid) if b=='2013-01-01' else grid.get_loc(pd.Timestamp(b))
  origins=np.array([o for o in range(lo,hi-96+1,24) if np.isfinite(v[o-96:o+96]).all() and np.isfinite(raw[o:o+96]).all()])
  past=origins[:,None]+np.arange(-96,0);future=origins[:,None]+np.arange(96)
  packs[name]=dict(x=torch.tensor(v[past,None],dtype=torch.float32),y=v[future,None],ids=torch.tensor((wd[future]*24+h[future]),dtype=torch.long),origin=origins,future=future)
  counts[name]=dict(windows=len(origins),unique_target_hours=len(np.unique(future)),holiday_window_count=int((hol[future].sum(1)>0).sum()),holiday_dates=sorted(set(grid[np.unique(future[hol[future]==1])].strftime('%Y-%m-%d'))))
 # Only fitting-period target hours construct the empirical template; no repeated-window weighting.
 a,b=c['splits']['fit']; mask=(grid>=a)&(grid<b)&np.isfinite(v)
 hist=np.zeros((3,24)); histcounts=np.zeros((3,24),int)
 for k in range(3):
  for hour in range(24):
   idx=mask&(kind==k)&(h==hour);histcounts[k,hour]=int(idx.sum());fallback=mask&(h==hour)
   hist[k,hour]=v[idx].mean() if idx.any() else v[fallback].mean()
 save(r/'availability.json',dict(counts=counts,mean=mu,sd=sd,historical_counts=histcounts.tolist(),data_sha=sha(c['data'])))
 np.savez_compressed(r/'data.npz',**{f'{name}_{key}':value.numpy() if torch.is_tensor(value) else value for name,p in packs.items() for key,value in p.items()})
 return packs,grid,raw,common,kind,h,hol,hist,sd

def knowledge(bank,kind,h):
 bank=bank-bank.mean(1,keepdims=True);bank=bank/max(float(np.sqrt(np.mean(bank**2))),1e-8)
 return np.stack([bank[0,h],(kind==1)*(bank[1,h]-bank[0,h]),(kind==2)*(bank[2,h]-bank[0,h])],1)
class Head(nn.Module):
 def __init__(self,seed):
  super().__init__();torch.manual_seed(seed+19000)
  self.net=nn.Sequential(nn.Linear(42,16),nn.GELU(),nn.Linear(16,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):
  q=self.net(x).double();return q-q.mean(1,keepdim=True)
def forward(head,z,base):return base.double()+head(z)

def fit(c,r):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);start=time.monotonic()
 packs,grid,raw,common,kind,h,hol,hist,sd=data(c,r)
 profiles=read(r/'profiles.json');task=np.array(profiles['Task']);wrong=np.array(profiles['Wrong']);perm=np.random.default_rng(2024).permutation(24)
 shuffled=task.copy();shuffled[1:]=task[0]+(task[1:]-task[0])[:,perm]
 K={'Fourier':np.stack([np.sin(10*np.pi*h/24),np.cos(10*np.pi*h/24),np.cos(12*np.pi*h/24)],1),'HistoricalTemplate':knowledge(hist,kind,h),'QwenTask':knowledge(task,kind,h),'QwenWrong':knowledge(wrong,kind,h),'QwenShuffle':knowledge(shuffled,kind,h)}
 np.savez_compressed(r/'knowledge.npz',**K,historical_profiles=hist,shuffle_permutation=perm)
 manifest=read(c['manifest'])['domains']['capital']['models'];baserecords=[];bases={};originals={};zs={};normalizers={};diagnostics=[]
 for seed in c['seeds']:
  bases[seed]={};originals[seed]={};zs[seed]={};normalizers[seed]={}
  for arm in ['Original','NoCalendar']:
   rec=next(x for x in manifest if x['seed']==seed and x['arm']==arm);assert sha(rec['path'])==rec['sha256'];baserecords.append(rec)
   model=CalendarTimeBase(c,seed,None if arm=='Original' else np.zeros((168,168),np.float32));model.load_state_dict(torch.load(rec['path'],map_location='cpu',weights_only=True));model.eval()
   for p in model.parameters():p.requires_grad_(False)
   for split,p in packs.items():
    pred=predict(model,p)
    (originals if arm=='Original' else bases)[seed][split]=torch.tensor(pred)
  # Shared normalizers estimated only from teacher fitting windows.
  features={}
  for split,p in packs.items():
   nums=numeric_summary(p['x'],bases[seed][split]).numpy();idx=p['future'];features[split]=np.concatenate([nums,raw[idx],common[idx]],-1)
  mu=features['fit'].mean((0,1));scale=features['fit'].std((0,1));scale=np.maximum(scale,1e-6)
  for arm,k in K.items():
   kfit=k[packs['fit']['future']];km=kfit.mean((0,1));ks=max(float(np.sqrt(np.mean((kfit-km)**2))),1e-6)
   zs[seed][arm]={split:torch.tensor(np.concatenate([(features[split]-mu)/scale,(k[p['future']]-km)/ks],-1),dtype=torch.float32) for split,p in packs.items()}
   normalizers[seed][arm]=dict(mean=mu.tolist(),sd=scale.tolist(),knowledge_mean=km.tolist(),knowledge_rms=ks)
   mat=zs[seed][arm]['fit'].numpy().reshape(-1,42);sv=np.linalg.svd(mat-mat.mean(0),compute_uv=False);w=sv**2/(sv**2).sum();er=float(np.exp(-np.sum(w[w>0]*np.log(w[w>0]))))
   diagnostics.append(dict(seed=seed,arm=arm,features=42,parameters=sum(p.numel() for p in Head(seed).parameters()),rank=int(np.linalg.matrix_rank(mat-mat.mean(0))),effective_rank=er,knowledge_rms=float(np.sqrt(np.mean(mat[:,-3:]**2)))))
 save(r/'normalizers.json',normalizers);save(r/'feature_diagnostics.json',diagnostics)
 save(r/'frozen_assets.json',baserecords)
 states={};curves=[];invariants=[]
 for seed in c['seeds']:
  states[seed]={};p=packs['fit'];target=torch.tensor(p['y']);base=bases[seed]['fit'];norm=scores(base.numpy(),p['y'])
  batches=np.random.default_rng(seed).integers(0,len(target),size=(c['steps'],c['batch']))
  for arm in K:
   head=Head(seed);opt=torch.optim.Adam(head.parameters(),lr=c['lr']);states[seed][arm]={}
   z=zs[seed][arm]['fit'];init=forward(head,z,base);assert torch.equal(init,base.double())
   opt.zero_grad();loss=((init-target)**2).mean();loss.backward();assert head.net[-1].weight.grad.abs().sum()>0
   grad=float(head.net[-1].weight.grad.abs().sum());head.zero_grad()
   for step in range(c['steps']+1):
    if step%100==0:
     with torch.no_grad():pv=forward(head,zs[seed][arm]['val'],bases[seed]['val']).numpy()
     met=scores(pv,packs['val']['y']);curves.append(dict(seed=seed,arm=arm,step=step,**met));states[seed][arm][step]=copy.deepcopy(head.state_dict())
    if step==c['steps']:break
    idx=batches[step];pred=forward(head,z[idx],base[idx]);err=pred-target[idx]
    loss=.5*(err.square().mean()/norm['mse']+err.abs().mean()/norm['mae']);opt.zero_grad();loss.backward();opt.step()
    assert time.monotonic()-start<c['fit_seconds'],'CPU fitting time budget exceeded'
   invariants.append(dict(seed=seed,arm=arm,initial_prediction_max_error=0.,initial_output_gradient_l1=grad))
   print('FIT',seed,arm,flush=True)
 # Select before computing evaluation losses. All arms receive seven checkpoints, pooled three seeds.
 chosen={}
 for arm in K:
  def criterion(step):
   vals=[]
   for seed in c['seeds']:
    b=scores(bases[seed]['val'].numpy(),packs['val']['y']);q=next(q for q in curves if q['seed']==seed and q['arm']==arm and q['step']==step);vals.append(.5*(q['mse']/b['mse']+q['mae']/b['mae']))
   return np.mean(vals)
  chosen[arm]=min(range(0,c['steps']+1,100),key=criterion)
 vm={arm:{key:float(np.mean([next(q for q in curves if q['seed']==seed and q['arm']==arm and q['step']==chosen[arm])[key] for seed in c['seeds']])) for key in ['mse','mae']} for arm in K}
 vb={key:float(np.mean([scores(bases[s]['val'].numpy(),packs['val']['y'])[key] for s in c['seeds']])) for key in ['mse','mae']}
 gate=all(vm['QwenTask'][key]<other[key] for key in ['mse','mae'] for other in [vb,vm['Fourier'],vm['HistoricalTemplate']]) and sum(next(q for q in curves if q['seed']==s and q['arm']=='QwenTask' and q['step']==chosen['QwenTask'])['mse']<scores(bases[s]['val'].numpy(),packs['val']['y'])['mse'] for s in c['seeds'])>=2
 save(r/'selection.json',dict(chosen=chosen,val=vm,base_val=vb,teacher_gate=gate,git=git(),note='Locked before outer metric computation; evaluation is previously viewed development, not independent.'))
 save(r/'curves.json',curves);torch.save(states,r/'all_checkpoints.pt')
 rows=[];predictions={};group_rows=[]
 for seed in c['seeds']:
  for split,p in packs.items():
   predictions[f'{seed}_{split}_y']=p['y'];predictions[f'{seed}_{split}_origin']=p['origin']
   out={'FrozenTimeBase':bases[seed][split].double().numpy(),'OriginalTimeBase':originals[seed][split].double().numpy()}
   for arm in K:
    head=Head(seed);head.load_state_dict(states[seed][arm][chosen[arm]])
    with torch.no_grad():pred=forward(head,zs[seed][arm][split],bases[seed][split]).numpy()
    out[arm]=pred;meanerr=float(np.max(abs((pred-bases[seed][split].numpy()).mean(1))));assert meanerr<1e-10
    replay=Head(seed);replay.load_state_dict(states[seed][arm][chosen[arm]])
    with torch.no_grad():assert np.array_equal(pred,forward(replay,zs[seed][arm][split],bases[seed][split]).numpy())
    invariants.append(dict(seed=seed,arm=arm,split=split,mean_error=meanerr,replay_exact=True))
   idx=p['future'];hours=h[idx];days=kind[idx]
   groups={'all':np.ones(idx.shape,bool),'holiday':days==2,'weekend':days==1,'workday':days==0,'morning7_9':(hours>=7)&(hours<=9),'evening16_19':(hours>=16)&(hours<=19)}
   for arm,pred in out.items():
    predictions[f'{seed}_{split}_{arm}']=pred;rows.append(dict(seed=seed,split=split,arm=arm,step=chosen.get(arm),**scores(pred,p['y'])))
    for label,mask in groups.items():
     err=(pred-p['y'])[...,0][mask]
     if len(err):group_rows.append(dict(seed=seed,split=split,arm=arm,group=label,hours=int(mask.sum()),unique_hours=int(len(np.unique(idx[mask]))),windows=int(mask.any(1).sum()),mse=float(np.mean(err**2)),mae=float(np.mean(abs(err)))))
 np.savez_compressed(r/'predictions.npz',**predictions);save(r/'metrics.json',rows);save(r/'subgroups.json',group_rows);save(r/'invariants.json',invariants)
 for rec in baserecords:assert sha(rec['path'])==rec['sha256']
 save(r/'status.json',dict(seconds=time.monotonic()-start,teacher_gate=gate,kd_runs=0,kd_status='Teacher gate passed; separate bounded KD stage required' if gate else 'Not run: validation teacher gate failed',git=git()))
 print('SELECTION',json.dumps(read(r/'selection.json')),flush=True)
 print('TEST',json.dumps([q for q in rows if q['split']=='test']),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--stage',choices=['extract','fit'],required=True);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--resume',action='store_true');a=ap.parse_args();c=read(a.config);r=Path(c['run'])
 if a.stage=='extract':
  if a.resume:
   assert r.is_dir() and read(r/'config.json')==c
   save(r/'resume_provenance.json',dict(git=git(),source_sha=sha(__file__),note='Reuse original generated text; only finish missing Wrong; accept arbitrary positive scale by global rescaling before RMS; no forecast scores inspected.'))
  else:
   r.mkdir(parents=True,exist_ok=False);save(r/'config.json',c);save(r/'provenance.json',dict(git=git(),source_sha=sha(__file__),config_sha=sha(a.config)))
  extract(c,r,a.gpu)
 else:fit(c,r)
if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();raise
