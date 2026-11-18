"""Small GPT-J time-geometry learning experiment, not a forecasting trainer."""
import os
os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse,copy,csv,hashlib,importlib.util,importlib.metadata as md,json,re,subprocess,time,traceback
from pathlib import Path
DAYS=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def csvsave(p,rows):
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=sorted(set().union(*(x.keys() for x in rows))));w.writeheader();w.writerows(rows)
def record(d,h,delta,template,split,parts=None):
 d,h,delta=int(d),int(h),int(delta);source=f'{DAYS[d]} {h:02d}:00';signed=f'{delta:+d}'
 if template==0:text=f'Time: {source}. Shift: {signed} hours. Result:'
 elif template==1:
  source=f'{DAYS[d]} {h%12 or 12}:00 {"AM" if h<12 else "PM"}';text=f'Starting at {source}, add {signed} hours. The new time is'
 elif template==2:text=f'The clock reads {source}. After a change of {signed} hours, it reads'
 elif template==3:
  source=f'{h%12 or 12}:00 {"AM" if h<12 else "PM"} on {DAYS[d]}';text=f'At {source}, move the clock {abs(delta)} hours {"forward" if delta>=0 else "backward"}. Result:'
 elif template==4:
  source=f'{DAYS[d]} at {h%12 or 12}:00 {"AM" if h<12 else "PM"}';text=f'{source}: calculate the time {abs(delta)} hours {"later" if delta>=0 else "earlier"}. Answer:'
 elif template==5:text=f'Time: {source}. First shift {parts[0]:+d} hours, then {parts[1]:+d} hours. Result:'
 else:raise ValueError(template)
 absolute=(d*24+h+delta)%168;answer=f' {DAYS[absolute//24]} {absolute%24:02d}:00'
 return dict(day=d,hour=h,delta=delta,source_id=d*24+h,identity=[d,h,delta],template=template,split=split,text=text,source_end=text.index(source)+len(source),answer=answer,answer_day=absolute//24,answer_hour=absolute%24,carry=(h+delta)//24!=0,parts=parts)
def dataset(c):
 import numpy as np
 rng=np.random.default_rng(c['seed']);seen=[(d,h) for d in range(7) for h in range(24) if h%4!=3];offsets=c['train_offsets']
 pool=[(d,h,k) for d,h in seen for k in offsets];rng.shuffle(pool);chosen=[];used=set()
 for d,h in seen:
  k=offsets[int(rng.integers(len(offsets)))];key=(d,h,k);chosen.append(key);used.add(key)
 chosen+= [x for x in pool if x not in used][:512-len(chosen)];used=set(chosen);left=[x for x in pool if x not in used]
 rows=[record(*x,i%2,'train') for i,x in enumerate(chosen)]
 for name,count,templates in [('val',64,[2]),('id',48,[0,1]),('expression',48,[3,4])]:
  take,left=left[:count],left[count:];rows.extend(record(*x,templates[i%len(templates)],name) for i,x in enumerate(take));used.update(take)
 ood=[(d,h,k) for d in range(7) for h in range(24) if h%4==3 for k in offsets];rng.shuffle(ood)
 rows.extend(record(*x,i%2,'unseen_source') for i,x in enumerate(ood[:48]));used.update(ood[:48])
 unseen=[(d,h,k) for d,h in seen for k in c['test_offsets']];rng.shuffle(unseen)
 carry=[x for x in unseen if (x[1]+x[2])//24!=0];flat=[x for x in unseen if (x[1]+x[2])//24==0];take=carry[:24]+flat[:24];rng.shuffle(take)
 rows.extend(record(*x,i%2,'unseen_offset') for i,x in enumerate(take));used.update(take)
 candidates=[]
 for d,h in seen:
  for a in [-6,-3,-1,1,3,6]:
   for b in [-6,-3,-1,1,3,6]:
    k=a+b
    if k not in offsets and k!=0 and (d,h,k) not in used:candidates.append((d,h,k,a,b))
 rng.shuffle(candidates);unique=[];seenkeys=set()
 for x in candidates:
  if x[:3] not in seenkeys:unique.append(x);seenkeys.add(x[:3])
 carry=[x for x in unique if (x[1]+x[2])//24!=0];flat=[x for x in unique if (x[1]+x[2])//24==0];take=carry[:24]+flat[:24];rng.shuffle(take)
 rows.extend(record(x[0],x[1],x[2],5,'composition',parts=list(x[3:])) for x in take)
 assert len(rows)==816 and len({tuple(x['identity']) for x in rows})==len(rows)
 assert set(x['source_id'] for x in rows if x['split']=='train')==set(d*24+h for d,h in seen)
 canonical=[record(d,h,0,0,'canonical') for d in range(7) for h in range(24)]
 for i,row in enumerate(rows):row['id']=i
 return rows,canonical

def bootstrap():
 pairs={'torch':'torch','numpy':'numpy','packaging':'packaging','safetensors':'safetensors','tokenizers':'tokenizers','huggingface_hub':'huggingface_hub','accelerate':'accelerate','requests':'requests','filelock':'filelock','regex':'regex','tqdm':'tqdm','yaml':'PyYAML','typing_extensions':'typing_extensions','transformers':'transformers','jinja2':'Jinja2','pandas':'pandas','peft':'peft','psutil':'psutil'}
 mapping={m:[md.distribution(d).metadata['Name']] for m,d in pairs.items() if importlib.util.find_spec(m)};md.packages_distributions=lambda:dict(mapping)
 from transformers import utils as hu
 from transformers.utils import import_utils as hi
 for name in ['is_torchvision_available','is_torchvision_v2_available','is_vision_available','is_librosa_available','is_torchaudio_available','is_sklearn_available']:
  for module in [hu,hi]:
   if hasattr(module,name):setattr(module,name,lambda:False)
 return {m:md.version(d) for m,d in pairs.items() if importlib.util.find_spec(m)}

def prepare(c,r):
 r.mkdir(parents=True,exist_ok=False);rows,canonical=dataset(c);dump(r/'config.json',c);dump(r/'examples.json',rows);dump(r/'canonical.json',canonical)
 counts=[]
 for split in ['train','val','id','expression','unseen_source','unseen_offset','composition']:
  rr=[x for x in rows if x['split']==split];counts.append(dict(split=split,n=len(rr),source_ids=len(set(x['source_id'] for x in rr)),carry=sum(x['carry'] for x in rr),noncarry=sum(not x['carry'] for x in rr)))
 csvsave(r/'dataset_counts.csv',counts);dump(r/'source.json',dict(commit=commit(),script_sha256=sha(__file__),examples_sha256=sha(r/'examples.json'),model_revision=c['model_revision'],model_bytes=Path(c['model_path'],'pytorch_model.bin').stat().st_size,base_weight_mtime=Path(c['model_path'],'pytorch_model.bin').stat().st_mtime,expected_previously_verified_model_sha256=c['model_sha256']))
 print('PREPARED',json.dumps(counts),flush=True)

def run(c,r,device):
 import numpy as np
 versions=bootstrap();dump(r/'packages.json',versions)
 import torch
 from torch.nn import functional as F
 from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
 from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
 from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
 torch.set_num_threads(2);torch.manual_seed(c['seed']);torch.use_deterministic_algorithms(True)
 gpu=int(device.split(':')[1]);snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);free={int(z.split(',')[0]):int(z.split(',')[1]) for z in snapshot.splitlines()}
 assert free[gpu]>=c['min_free_mib'],'Insufficient free GPU memory; no other process touched'
 dump(r/'gpu.json',dict(device=device,snapshot=snapshot));torch.cuda.set_per_process_memory_fraction(c['memory_fraction'],device)
 tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(c['model_path'])/'tokenizer.json'),eos_token='<|endoftext|>',pad_token='<|endoftext|>')
 rows=json.loads((r/'examples.json').read_text());canonical=json.loads((r/'canonical.json').read_text());encoded={}
 for item in rows+canonical:
  enc=tok(item['text'],add_special_tokens=False,return_offsets_mapping=True);source=[i for i,(a,b) in enumerate(enc['offset_mapping']) if a<item['source_end'] and b>=item['source_end']];assert len(source)==1
  answer=tok.encode(item['answer'],add_special_tokens=False)+[tok.eos_token_id];assert len(enc['input_ids'])+len(answer)<=c['max_length']
  encoded[item['text']]=dict(prompt=enc['input_ids'],answer=answer,site=source[0])
 def collate(items,answer=False):
  seqs=[encoded[x['text']]['prompt']+(encoded[x['text']]['answer'] if answer else []) for x in items];length=max(map(len,seqs));inp=torch.full((len(items),length),tok.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(inp);labels=torch.full_like(inp,-100)
  for i,(item,seq) in enumerate(zip(items,seqs)):
   inp[i,:len(seq)]=torch.tensor(seq,device=device);mask[i,:len(seq)]=1
   if answer:
    n=len(encoded[item['text']]['prompt']);labels[i,n:len(seq)]=inp[i,n:len(seq)]
  positions=mask.cumsum(-1)-1;positions.masked_fill_(mask==0,0)
  data=dict(input_ids=inp,attention_mask=mask,position_ids=positions)
  if answer:data['labels']=labels
  sites=torch.tensor([encoded[x['text']]['site'] for x in items],device=device)
  return data,sites
 print('Loading existing GPT-J BF16',device,flush=True);load_start=time.monotonic()
 base=GPTJForCausalLM.from_pretrained(c['model_path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',low_cpu_mem_usage=True).to(device);base.config.use_cache=False;base.eval();load_seconds=time.monotonic()-load_start;assert load_seconds<c['load_cap_seconds']
 clock=time.monotonic()
 def limit():
  if time.monotonic()-clock>c['active_cap_seconds']:raise TimeoutError('Frozen active GPU pilot budget exceeded')
 captured={}
 def hook(module,args,out):captured['hidden']=out[0] if isinstance(out,tuple) else out
 handle=base.transformer.h[c['geometry_layer']].register_forward_hook(hook)
 def phi(items):
  h=torch.tensor([x['hour'] for x in items],device=device)*2*torch.pi/24;d=torch.tensor([x['day'] for x in items],device=device)*2*torch.pi/7
  return torch.stack((h.cos(),h.sin(),d.cos(),d.sin()),-1)/np.sqrt(2)
 def capture(items):
  hs=[];base.eval()
  with torch.no_grad():
   for i in range(0,len(items),c['eval_batch']):
    limit();batch=items[i:i+c['eval_batch']];enc,site=collate(batch);base(**enc,use_cache=False);hs.append(captured.pop('hidden')[torch.arange(len(batch),device=device),site].float().cpu().numpy())
  return np.concatenate(hs)
 train=[x for x in rows if x['split']=='train'];val=[x for x in rows if x['split']=='val'];test=[x for x in rows if x['split'] not in ['train','val']]
 print('Calibrating a fixed readout from training states only',flush=True);h=capture(train);np.savez_compressed(r/'base_train_hidden.npz',hidden=h.astype('float16'))
 # Orthogonal nuisance removal and low-dimensional source-state calibration; then freeze forever.
 h64=h.astype(float);mu=h64.mean(0);templates=np.array([x['template'] for x in train]);res=h64.copy();nuisance=h64[templates==1].mean(0)-h64[templates==0].mean(0);nuisance/=max(np.linalg.norm(nuisance),1e-12)
 for t in [0,1]:res[templates==t]-=h64[templates==t].mean(0)
 res-=res@nuisance[:,None]*nuisance[None,:]
 day=np.array([x['day'] for x in train]);hour=np.array([x['hour'] for x in train]);wd=np.stack([res[day==d].mean(0) for d in range(7)]);wd-=wd.mean(0);_,sw,vw=np.linalg.svd(wd,full_matrices=False);uw=vw[:2].T
 hrs=sorted(set(hour));hc=np.stack([res[hour==v].mean(0) for v in hrs]);hc-=hc.mean(0);hc-=hc@uw@uw.T;_,sh,vh=np.linalg.svd(hc,full_matrices=False);uh=vh[:2].T;u=np.c_[uh,uw];assert np.max(abs(u.T@u-np.eye(4)))<1e-6
 q=(h64-mu)@u;design=np.c_[q,np.ones(len(q))];target=phi(train).cpu().numpy().astype(float);coef=np.linalg.lstsq(design,target,rcond=1e-8)[0];weight=u@coef[:4];bias=coef[4]-mu@weight
 np.savez_compressed(r/'fixed_readout.npz',weight=weight,bias=bias,basis=u,mu=mu,calibration=coef);readout_sha=sha(r/'fixed_readout.npz');w=torch.tensor(weight,dtype=torch.float32,device=device);b=torch.tensor(bias,dtype=torch.float32,device=device)
 dump(r/'readout.json',dict(calibration='training source states only, template residualized PCA2 hour/PCA2 weekday + fixed affine calibration',train_mse=float(np.mean((design@coef-target)**2)),hour_singular_values=sh[:4].tolist(),weekday_singular_values=sw[:4].tolist(),basis_orthogonality=float(abs(u.T@u-np.eye(4)).max()),sha256=readout_sha))
 model=get_peft_model(base,LoraConfig(r=c['rank'],lora_alpha=c['alpha'],target_modules=['q_proj','v_proj'],layers_to_transform=c['lora_layers'],layers_pattern='h',bias='none',lora_dropout=0.,task_type='CAUSAL_LM'));trainable=[p for p in model.parameters() if p.requires_grad];count=sum(p.numel() for p in trainable);assert count==262144
 initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()};torch.save(initial,r/'initial_adapter.pt')
 def step_forward(items,geom=False):
  enc,site=collate(items,answer=True);out=model(**enc,use_cache=False);hidden=captured.pop('hidden')[torch.arange(len(items),device=device),site].float();z=hidden@w+b;geo=F.mse_loss(z,phi(items));return out.loss,geo
 def validate():
  model.eval();total=0.;n=0
  with torch.no_grad():
   for i in range(0,len(val),c['eval_batch']):
    limit();batch=val[i:i+c['eval_batch']];ce,_=step_forward(batch);total+=float(ce)*len(batch);n+=len(batch)
  return total/n
 sample=train[:2];enc,_=collate(sample,answer=True);model.eval()
 with torch.no_grad():
  logits=model(**enc,use_cache=False).logits;captured.clear()
  with model.disable_adapter():reference=model(**enc,use_cache=False).logits
  captured.clear();maxzero=float((logits-reference).abs().max());assert maxzero==0.
 ce,geo=step_forward(sample);geo.backward();geo_grad=float(sum(p.grad.float().square().sum() for p in trainable if p.grad is not None).sqrt());assert geo_grad>0 and np.isfinite(geo_grad);model.zero_grad(set_to_none=True)
 ce,geo=step_forward(sample);ce.backward();ce_grad=float(sum(p.grad.float().square().sum() for p in trainable if p.grad is not None).sqrt());assert ce_grad>0 and np.isfinite(ce_grad);model.zero_grad(set_to_none=True)
 dump(r/'preflight.json',dict(trainable_lora=count,readout_trainable=False,initial_adapter_logits_maxdiff=maxzero,geometry_to_lora_gradient_norm=geo_grad,ce_to_lora_gradient_norm=ce_grad,load_seconds=load_seconds,peak_gib=torch.cuda.max_memory_allocated(device)/2**30))
 records=[]
 for arm,lam in [('SFT',0.),('Circle',c['geometry_lambda'])]:
  set_peft_model_state_dict(model,initial);model.zero_grad(set_to_none=True);opt=torch.optim.AdamW(trainable,lr=c['lr'],weight_decay=0.);order=np.random.default_rng(c['seed']).permutation(len(train));curve=[];best=float('inf');bestpath=None;out=r/arm;out.mkdir(exist_ok=False);begin=time.monotonic()
  for step in range(c['updates']+1):
   limit()
   if step%c['validation_every']==0:
    v=validate();path=out/f'adapter_step{step}.pt';state={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()};torch.save(state,path);curve.append(dict(step=step,val_ce=v,path=str(path),sha256=sha(path)))
    if v<best:best=v;bestpath=path;beststep=step
    dump(out/'curve.json',curve);print('VALIDATION',arm,step,v,flush=True)
   if step==c['updates']:break
   model.train();opt.zero_grad(set_to_none=True);totalce=totalgeo=0.
   for micro in range(c['accumulation']):
    beginix=(step*c['accumulation']+micro)*c['micro_batch'];ix=order[beginix:beginix+c['micro_batch']];assert len(ix)==c['micro_batch'];items=[train[i] for i in ix];ce,geo=step_forward(items);loss=(ce+lam*geo)/c['accumulation'];assert torch.isfinite(loss);loss.backward();totalce+=float(ce.detach());totalgeo+=float(geo.detach())
   norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);opt.step()
   if (step+1)%16==0:print('TRAIN',arm,step+1,totalce/c['accumulation'],totalgeo/c['accumulation'],float(norm),flush=True)
  rec=dict(arm=arm,path=str(bestpath),sha256=sha(bestpath),best_step=beststep,val_ce=best,seconds=time.monotonic()-begin,trainable_parameters=count);records.append(rec);dump(r/'training_progress.json',records)
 # Lock every choice before opening held-out test/cohort results.
 dump(r/'test_lock.json',dict(commit=commit(),models=records,readout_sha256=readout_sha,selection='minimum validation answer-token CE, no test selection'))
 def gethidden(items):
  hs=[];model.eval()
  with torch.no_grad():
   for i in range(0,len(items),c['eval_batch']):
    limit();batch=items[i:i+c['eval_batch']];enc,site=collate(batch);model(**enc,use_cache=False);hs.append(captured.pop('hidden')[torch.arange(len(batch),device=device),site].float().cpu().numpy())
  return np.concatenate(hs)
 def generate(items):
  outs=[];model.eval()
  for i in range(0,len(items),c['eval_batch']):
   limit();batch=items[i:i+c['eval_batch']];seqs=[encoded[x['text']]['prompt'] for x in batch];length=max(map(len,seqs));inp=torch.full((len(batch),length),tok.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(inp)
   for j,seq in enumerate(seqs):inp[j,-len(seq):]=torch.tensor(seq,device=device);mask[j,-len(seq):]=1
   with torch.no_grad():out=model.generate(input_ids=inp,attention_mask=mask,max_new_tokens=c['max_new_tokens'],do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   captured.clear()
   for item,tokens in zip(batch,out[:,length:].cpu().tolist()):
    text=tok.decode(tokens,skip_special_tokens=True);match=re.fullmatch(r'\s*('+'|'.join(DAYS)+r')\s+(\d{1,2}):00\s*[.!]?\s*',text);valid=match is not None and int(match.group(2))<24;correct=valid and DAYS.index(match.group(1))==item['answer_day'] and int(match.group(2))==item['answer_hour']
    outs.append(dict(id=item['id'],split=item['split'],source_id=item['source_id'],carry=item['carry'],expected=item['answer'].strip(),text=text,tokens=tokens,valid_format=bool(valid),strict_exact=text.strip()==item['answer'].strip(),correct=bool(correct),weekday_correct=bool(valid and DAYS.index(match.group(1))==item['answer_day']),hour_correct=bool(valid and int(match.group(2))==item['answer_hour'])))
  return outs
 allrows=[];metrics=[];geoms=[]
 for arm in ['Base','SFT','Circle']:
  if arm=='Base':set_peft_model_state_dict(model,initial)
  else:
   rec=next(x for x in records if x['arm']==arm);assert sha(rec['path'])==rec['sha256'];set_peft_model_state_dict(model,torch.load(rec['path'],map_location='cpu',weights_only=True))
  print('EVALUATE',arm,flush=True);results=generate(test);allrows.extend(dict(arm=arm,**x) for x in results);ht=gethidden(test);hc=gethidden(canonical);np.savez_compressed(r/f'hidden_{arm}.npz',test=ht.astype('float16'),canonical=hc.astype('float16'))
  for split in ['id','expression','unseen_source','unseen_offset','composition']:
   for group in ['all','carry','noncarry']:
    rr=[x for x in results if x['split']==split and (group=='all' or x['carry']==(group=='carry'))];n=len(rr)
    metrics.append(dict(arm=arm,split=split,group=group,n=n,correct=sum(x['correct'] for x in rr),accuracy=sum(x['correct'] for x in rr)/n if n else None,strict_exact=sum(x['strict_exact'] for x in rr),valid_format=sum(x['valid_format'] for x in rr),weekday_correct=sum(x['weekday_correct'] for x in rr),hour_correct=sum(x['hour_correct'] for x in rr)))
  z=ht.astype(float)@weight+bias;phit=phi(test).cpu().numpy();probe_mse=float(np.mean((z-phit)**2));native=hc.astype(float).reshape(7,24,-1)
  for factor,v in [('hour',native.mean(0)),('weekday',native.mean(1))]:
   v=v-v.mean(0);uu,ss,vv=np.linalg.svd(v,full_matrices=False);q=uu[:,:2]*ss[:2];power=(abs(np.fft.fft(v,axis=0))**2).sum(1);radius=np.linalg.norm(q,axis=1);first=float((power[1]+power[-1])/power[1:].sum());geoms.append(dict(arm=arm,factor=factor,n=len(v),native_first_harmonic_energy=first,native_pca2_energy=float((ss[:2]**2).sum()/(ss*ss).sum()),native_pca2_radius_cv=float(radius.std()/radius.mean()),heldout_fixed_readout_mse=probe_mse))
  dump(r/'all_predictions.json',allrows);csvsave(r/'metrics.csv',metrics);csvsave(r/'geometry.csv',geoms)
 handle.remove();assert readout_sha==sha(r/'fixed_readout.npz')
 for rec in records:assert sha(rec['path'])==rec['sha256']
 assert Path(c['model_path'],'pytorch_model.bin').stat().st_mtime==json.loads((r/'source.json').read_text())['base_weight_mtime']
 dump(r/'completion.json',dict(commit=commit(),active_seconds=time.monotonic()-clock,load_seconds=load_seconds,selected=records,train_rows=len(train),validation_rows=len(val),test_rows=len(test),trainable_lora_parameters=count,base_checkpoint_unchanged=True,fixed_readout_unchanged=True,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,expected_model_sha256=c['model_sha256'],device=device))
 print('COMPLETE',json.dumps(json.loads((r/'completion.json').read_text())),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);ap.add_argument('--config',required=True);ap.add_argument('--device',default='cuda:3');args=ap.parse_args();c=json.loads(Path(args.config).read_text());r=Path(c['run'])
 try:
  if args.action=='prepare':prepare(c,r)
  else:run(c,r,args.device)
 except BaseException as ex:
  if r.exists():dump(r/f'failure_{args.action}.json',dict(error=str(ex),traceback=traceback.format_exc()))
  raise
