"""One bounded TimeBase research experiment; no changes to public TSLib trainers."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,copy,csv,hashlib,json,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from layers.ChronoGeomTimeBaseOfficial import Model as TimeBase

def dump(p,a):p.write_text(json.dumps(a,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def csvsave(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))),lineterminator='\n');w.writeheader();w.writerows(rows)
def normalize_codes(a):
 a=a-a.mean(0,keepdims=True)
 return a/torch.linalg.vector_norm(a,dim=1,keepdim=True).clamp_min(1e-6)
def scaled_codes(a):
 e=normalize_codes(a);g=e@e.T;s=g.square().mean().sqrt().clamp_min(1e-6)
 return e/s.sqrt()
def cfgbase(c):return SimpleNamespace(seq_len=c['seq_len'],pred_len=c['pred_len'],enc_in=7,period_len=c['period_len'],basis_num=c['basis_num'],individual=False,use_period_norm=True,use_orthogonal=False)
def teacher_tables(c,device):
 if 'frozen_tables' in c:
  a=np.load(c['frozen_tables']);return {k:torch.as_tensor(a[k],device=device) for k in a.files}
 d=np.repeat(np.arange(7),24);h=np.tile(np.arange(24),7)
 f4=np.column_stack((np.cos(2*np.pi*h/24),np.sin(2*np.pi*h/24),np.cos(2*np.pi*d/7),np.sin(2*np.pi*d/7)))
 cols=[]
 for x,p,ks in [(h,24,range(1,7)),(d,7,range(1,4))]:
  for k in ks:cols.extend((np.cos(2*np.pi*k*x/p),np.sin(2*np.pi*k*x/p)))
 cols.extend(f4[:,i]*f4[:,j] for i in [0,1] for j in [2,3])
 raw={'LLM':np.load(c['lookup_path'])['llm4'],'F4':f4,'F22':np.column_stack(cols)}
 tables={k:scaled_codes(torch.tensor(v,dtype=torch.float32,device=device)) for k,v in raw.items()}
 u,s,_=torch.linalg.svd(tables['F22'],full_matrices=False)
 projected=u[:,:4]*s[:4]
 tables['F22r4']=projected/(projected@projected.T).square().mean().sqrt().clamp_min(1e-6).sqrt()
 tables['Shuffle']=tables['LLM'][torch.as_tensor(np.random.default_rng(1729).permutation(168),device=device)]
 return tables

class RelationStudent(nn.Module):
 def __init__(self,backbone,seed,rank=4,hidden=16):
  super().__init__();self.backbone=copy.deepcopy(backbone)
  for p in self.backbone.parameters():p.requires_grad_(False)
  torch.manual_seed(seed+10000)
  self.time_mlp=nn.Sequential(nn.Linear(2,hidden),nn.GELU(),nn.Linear(hidden,rank))
  self.gamma=nn.Parameter(torch.zeros(()))
  d=torch.arange(168)//24;h=torch.arange(168)%24
  self.register_buffer('calendar',torch.stack((d/3.-1.,h/11.5-1.),1))
 def codes(self):return scaled_codes(self.time_mlp(self.calendar))
 def forward(self,x,ids):
  e=self.codes();p=self.backbone(x);L=x.shape[1]
  delta=torch.bmm(e[ids[:,L:]],torch.bmm(e[ids[:,:L]].transpose(1,2),x-x.mean(1,keepdim=True)))/L
  return p+self.gamma*delta,e

def pack(v,ids,lo,hi,L,H,stride):
 origins=np.arange(max(lo,L),hi-H+1,stride)
 return {'x':np.stack([v[o-L:o] for o in origins]),'y':np.stack([v[o:o+H] for o in origins]),'ids':np.stack([ids[o-L:o+H] for o in origins]),'origin':origins}
def tensors(p,device):return {k:torch.as_tensor(v,device=device) for k,v in p.items()}
def prepare(c,r):
 import pandas as pd
 r.mkdir(exist_ok=False,parents=True)
 df=pd.read_csv(c['data_path'],nrows=c['val_end'],usecols=['date']+c['columns']);dates=pd.to_datetime(df['date'])
 assert (dates.diff().dropna()==pd.Timedelta(hours=1)).all()
 v=df[c['columns']].to_numpy(dtype='float32');mu=v[:c['train_end']].mean(0);sd=v[:c['train_end']].std(0);assert (sd>1e-6).all()
 ids=(dates.dt.dayofweek.to_numpy()*24+dates.dt.hour.to_numpy()).astype('int64');v=(v-mu)/sd
 np.savez_compressed(r/'train_val.npz',values=v,ids=ids,mean=mu,sd=sd)
 dump(r/'config.json',c);dump(r/'source.json',{'commit':commit(),'script_sha256':sha(__file__),'upstream_tslib':'4e938a1767106324dd753b2a44832bf870a0252e','timebase_revision':'369b330f3d77371fcc7e8c75c808d01330c40899','data_sha256':sha(c['data_path']),'lookup_sha256':sha(c['lookup_path']),'test_opened':False})
 print('prepared training/validation only',flush=True)
def data(r,c,device):
 a=np.load(r/'train_val.npz');v=a['values'];ids=a['ids']
 return [tensors(pack(v,ids,lo,hi,c['seq_len'],c['pred_len'],st),device) for lo,hi,st in [(c['seq_len'],c['train_end'],1),(c['train_end'],c['val_end'],24)]]
def pred(model,p,baseline=False):
 out=[]
 with torch.no_grad():
  for i in range(0,len(p['x']),128):
   x=p['x'][i:i+128];ids=p['ids'][i:i+128];out.append((model(x) if baseline else model(x,ids)[0]).detach())
 return torch.cat(out)
def train(c,r,seed,device):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.1,device)
 out=r/f'seed{seed}';out.mkdir(exist_ok=True);tr,va=data(r,c,device);teachers=teacher_tables(c,device);torch.manual_seed(seed)
 cached={}
 for log in r.glob(f'fit_seed{seed}*.log'):
  for line in log.read_text().splitlines():
   if line.startswith('{'):
    try:
     a=json.loads(line)
     if 'arm' in a and 'path' in a and sha(a['path'])==a['sha256']:cached[(a['arm'],a['lambda'])]=a
    except (ValueError,KeyError,FileNotFoundError):pass
 backbone=TimeBase(cfgbase(c)).to(device);clock=time.monotonic()-sum(a['seconds'] for a in cached.values())-(30 if cached else 0);records=[]
 def fit(model,arm,lam,target=None,mode='relation',baseline=False):
  if (arm,lam) in cached:
   rec=cached[(arm,lam)];model.load_state_dict(torch.load(rec['path'],map_location=device,weights_only=True));records.append(rec);print('reused',arm,lam,flush=True);return rec
  opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=c['lr']);rng=np.random.default_rng(seed+233)
  best=float('inf');beststate=None;curve=[];steps=c['baseline_steps'] if baseline else c['student_steps'];begin=time.monotonic()
  for step in range(steps+1):
   if time.monotonic()-clock>c['seed_train_seconds']:raise TimeoutError('Bounded per-seed training budget exceeded')
   if step%100==0:
    v=float((pred(model,va,baseline)-va['y']).square().mean());assert np.isfinite(v)
    curve.append({'step':step,'val_mse':v})
    if v<best:best=v;beststate=copy.deepcopy(model.state_dict());beststep=step
   if step==steps:break
   ix=torch.as_tensor(rng.integers(len(tr['x']),size=c['batch_size']),device=device);x=tr['x'][ix];y=tr['y'][ix]
   if baseline:p=model(x);aux=x.new_zeros(())
   else:
    p,e=model(x,tr['ids'][ix]);aux=x.new_zeros(())
    if target is not None:
     if mode=='feature':aux=F.mse_loss(normalize_codes(e),normalize_codes(target).detach())
     else:aux=F.mse_loss(e@e.T,(target@target.T).detach())
   loss=F.mse_loss(p,y)+lam*aux
   if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss {arm} at step{step}')
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
  model.load_state_dict(beststate);label=f'{arm}_l{lam}';path=out/f'{label}.pt';torch.save(beststate,path)
  rec={'arm':arm,'lambda':lam,'seed':seed,'best_step':beststep,'val_mse':best,'seconds':time.monotonic()-begin,'path':str(path),'sha256':sha(path),'parameters':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'device':device}
  dump(out/f'{label}_curve.json',curve);records.append(rec);print(json.dumps(rec),flush=True);return rec
 base=fit(backbone,'B0',0,baseline=True);selected=[base]
 persist=float((va['x'][:,-1:, :]-va['y']).square().mean())
 for arm in ['S0','LLM','F4','F22','F22r4','Shuffle','Feature']:
  candidates=[]
  for lam in ([0] if arm=='S0' else c['lambdas']):
   m=RelationStudent(backbone,seed,c['rank'],c['hidden']).to(device)
   candidates.append(fit(m,arm,lam,None if arm=='S0' else teachers['LLM' if arm=='Feature' else arm], 'feature' if arm=='Feature' else 'relation'))
  selected.append(min(candidates,key=lambda a:a['val_mse']))
 # Direct frozen kernels have only one train-fitted scalar, no validation selection.
 direct=[]
 for name in ['LLM','F4','F22','F22r4']:
  num=den=0.;e=teachers[name]
  with torch.no_grad():
   for i in range(0,len(tr['x']),128):
    x=tr['x'][i:i+128];ids=tr['ids'][i:i+128];delta=torch.bmm(e[ids[:,c['seq_len']:]],torch.bmm(e[ids[:,:c['seq_len']]].transpose(1,2),x-x.mean(1,keepdim=True)))/c['seq_len']
    residual=tr['y'][i:i+128]-backbone(x);num+=float((delta*residual).sum());den+=float(delta.square().sum())
  direct.append({'arm':'Direct'+name,'gamma':num/max(den,1e-12),'seed':seed,'teacher_rank':int(torch.linalg.matrix_rank(e).item())})
 csvsave(out/'candidate_fits.csv',records);dump(out/'completion.json',{'commit':commit(),'selected':selected,'direct':direct,'persistence_val':persist,'seconds':time.monotonic()-clock,'peak_allocated':torch.cuda.max_memory_allocated(device)})
 print('seed complete',seed,flush=True)
def lock(c,r):
 a=[json.loads((r/f'seed{s}/completion.json').read_text()) for s in c['seeds']]
 base=[next(x for x in z['selected'] if x['arm']=='B0')['val_mse'] for z in a];persist=a[0]['persistence_val']
 ok=float(np.mean(base))<=.98*persist and sum(v<persist for v in base)>=2
 for z in a:
  for m in z['selected']:assert sha(m['path'])==m['sha256']
 dump(r/'lock.json',{'source_commit':commit(),'seeds':a,'validation_gate':ok,'baseline_val':base,'persistence_val':persist,'test_opened':False})
 print('validation_gate',ok,base,persist,flush=True)
def evaluate(c,r,device):
 import pandas as pd
 torch.set_num_threads(2);lockfile=json.loads((r/'lock.json').read_text());assert lockfile['validation_gate'],'Validation failed: test stays sealed'
 a=np.load(r/'train_val.npz');lo=c['test_start']-c['seq_len']
 df=pd.read_csv(c['data_path'],skiprows=range(1,lo+1),nrows=c['test_end']-lo,usecols=['date']+c['columns']);dates=pd.to_datetime(df['date'])
 assert len(df)==c['test_end']-lo and (dates.diff().dropna()==pd.Timedelta(hours=1)).all()
 v=(df[c['columns']].to_numpy(dtype='float32')-a['mean'])/a['sd'];ids=(dates.dt.dayofweek.to_numpy()*24+dates.dt.hour.to_numpy()).astype('int64')
 pn=pack(v,ids,c['seq_len'],len(v),c['seq_len'],c['pred_len'],24);pn['origin']+=lo;p=tensors(pn,device);tables=teacher_tables(c,device)
 predictions={'persistence':np.repeat(pn['x'][:,-1:,:],c['pred_len'],axis=1),'seasonal24':pn['x'][:,-24:,:]};rows=[];modelnotes=[];responses=[]
 for z in lockfile['seeds']:
  seed=z['selected'][0]['seed'];b=TimeBase(cfgbase(c)).to(device);br=next(x for x in z['selected'] if x['arm']=='B0');b.load_state_dict(torch.load(br['path'],map_location=device,weights_only=True));bp=pred(b,p,True)
  for rec in z['selected']:
   assert sha(rec['path'])==rec['sha256'];arm=rec['arm']
   if arm=='B0':out=bp
   else:
    m=RelationStudent(b,seed,c['rank'],c['hidden']).to(device);m.load_state_dict(torch.load(rec['path'],map_location=device,weights_only=True));out=pred(m,p)
    with torch.no_grad():
     e=m.codes();g=e@e.T;target=tables['LLM' if arm in ['S0','Feature'] else arm];modelnotes.append({'seed':seed,'arm':arm,'gamma':float(m.gamma),'relation_mse':float((g-target@target.T).square().mean()),'code_rank':int(torch.linalg.matrix_rank(e).item()),'correction_rms':float((out-bp).square().mean().sqrt())})
     for shift in [0,-24,24]:
      pp=dict(p,ids=(p['ids']+shift)%168);sp=pred(m,pp)
      responses.append({'seed':seed,'arm':arm,'shift':shift,'delta_rms':float((sp-out).square().mean().sqrt()),'delta_max':float((sp-out).abs().max()),'shifted_mse':float((sp-p['y']).square().mean())})
   predictions[f'{arm}_s{seed}']=out.cpu().numpy()
  for rec in z['direct']:
   e=tables[rec['arm'][6:]]
   with torch.no_grad():
    delta=torch.bmm(e[p['ids'][:,c['seq_len']:]],torch.bmm(e[p['ids'][:,:c['seq_len']]].transpose(1,2),p['x']-p['x'].mean(1,keepdim=True)))/c['seq_len']
   predictions[f"{rec['arm']}_s{seed}"]=(bp+rec['gamma']*delta).cpu().numpy()
 for key,out in predictions.items():rows.append({'method':key,'mse':float(np.mean((out-pn['y'])**2)),'mae':float(np.mean(abs(out-pn['y'])))})
 np.savez_compressed(r/'test_predictions.npz',**predictions,truth=pn['y'],origin=pn['origin']);csvsave(r/'test_scores.csv',rows);csvsave(r/'model_notes.csv',modelnotes);csvsave(r/'clock_responses.csv',responses)
 assert all(a['delta_max']==0 for a in responses if a['shift']==0)
 # Weekly, paired, weighted blocks; one draw shared by all seeds/channels/horizons.
 errors={arm:np.stack([((predictions[f'{arm}_s{s}']-pn['y'])**2).mean((1,2)) for s in c['seeds']]).mean(0) for arm in ['B0','S0','LLM','F4','F22','F22r4','Shuffle','Feature','DirectLLM','DirectF4','DirectF22','DirectF22r4']}
 block=(pn['origin']-c['test_start'])//168;uniq=np.unique(block);rng=np.random.default_rng(c['bootstrap_seed']);draw=rng.integers(len(uniq),size=(c['bootstrap_replicates'],len(uniq)));cnt=np.array([(block==k).sum() for k in uniq]);contrasts=[]
 for other in ['B0','S0','F4','F22','F22r4','Shuffle','Feature','DirectLLM','DirectF4','DirectF22','DirectF22r4']:
  d=errors['LLM']-errors[other];sums=np.array([d[block==k].sum() for k in uniq]);boot=sums[draw].sum(1)/cnt[draw].sum(1)
  contrasts.append({'comparison':'LLM-'+other,'difference':float(d.mean()),'lo95':float(np.quantile(boot,.025)),'hi95':float(np.quantile(boot,.975)),'seed_differences':[float(np.mean((predictions[f'LLM_s{s}']-pn['y'])**2)-np.mean((predictions[f'{other}_s{s}']-pn['y'])**2)) for s in c['seeds']]})
 dump(r/'contrasts.json',contrasts);dump(r/'completion.json',{'source_commit':commit(),'n_test_origins':len(pn['origin']),'n_channels':7,'n_week_blocks':len(uniq),'test_interval':[c['test_start'],c['test_end']],'data_sha256':sha(c['data_path']),'zero_noops':sum(x['shift']==0 for x in responses),'successful':True})
 print(json.dumps(rows),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','fit','lock','evaluate']);ap.add_argument('--config',default='docs/chronogeom/timebase_relation01_config.json');ap.add_argument('--seed',type=int,default=0);ap.add_argument('--device',default='cuda:3');args=ap.parse_args();c=json.loads(Path(args.config).read_text());r=Path(c['run'])
 if args.stage!='prepare':c=json.loads((r/'config.json').read_text())
 if args.stage=='prepare':prepare(c,r)
 elif args.stage=='fit':train(c,r,args.seed,args.device)
 elif args.stage=='lock':lock(c,r)
 else:evaluate(c,r,args.device)
if __name__=='__main__':main()
