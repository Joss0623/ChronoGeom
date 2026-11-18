"""Approved bounded continuation of GPT-J SFT, with fresh identity test and exact-answer selection."""
from scripts.chronogeom.clock_teacher_lora01 import bootstrap,record,DAYS,dump,sha,csvsave,commit
import argparse,json,re,time,subprocess,traceback
from pathlib import Path

def new_tests(c,old):
 import numpy as np
 rng=np.random.default_rng(c['data_seed']);used={tuple(x['identity']) for x in old};prior=used.copy();seen=[(d,h) for d in range(7) for h in range(24) if h%4!=3];tests=[]
 pool=[(d,h,k) for d,h in seen for k in c['train_offsets'] if (d,h,k) not in used];rng.shuffle(pool)
 for split,templates in [('id',[0,1]),('expression',[3,4])]:
  take,pool=pool[:48],pool[48:];assert len(take)==48
  tests.extend(record(*x,templates[i%len(templates)],split) for i,x in enumerate(take));used.update(take)
 pool=[(d,h,k) for d in range(7) for h in range(24) if h%4==3 for k in c['train_offsets'] if (d,h,k) not in used];rng.shuffle(pool);take=pool[:48];assert len(take)==48
 tests.extend(record(*x,i%2,'unseen_source') for i,x in enumerate(take));used.update(take)
 pool=[(d,h,k) for d,h in seen for k in c['test_offsets'] if (d,h,k) not in used];rng.shuffle(pool);carry=[x for x in pool if (x[1]+x[2])//24!=0];flat=[x for x in pool if (x[1]+x[2])//24==0];take=carry[:24]+flat[:24];assert len(take)==48;rng.shuffle(take)
 tests.extend(record(*x,i%2,'unseen_offset') for i,x in enumerate(take));used.update(take)
 pool=[];keys=set()
 for d,h in seen:
  for a in [-6,-3,-1,1,3,6]:
   for b in [-6,-3,-1,1,3,6]:
    k=a+b;key=(d,h,k)
    if k not in c['train_offsets'] and k!=0 and key not in used and key not in keys:pool.append((d,h,k,a,b));keys.add(key)
 rng.shuffle(pool);carry=[x for x in pool if (x[1]+x[2])//24!=0];flat=[x for x in pool if (x[1]+x[2])//24==0];take=carry[:24]+flat[:24];assert len(take)==48;rng.shuffle(take)
 tests.extend(record(x[0],x[1],x[2],5,'composition',parts=list(x[3:])) for x in take)
 assert len(tests)==240 and len({tuple(x['identity']) for x in tests})==240 and not ({tuple(x['identity']) for x in tests}&prior)
 for i,x in enumerate(tests):x['id']=max(v['id'] for v in old)+1+i
 return tests

def prepare(c,r):
 r.mkdir(parents=True,exist_ok=False);old=json.loads(Path(c['previous_run'],'examples.json').read_text());train=[x for x in old if x['split']=='train'];val=[x for x in old if x['split']=='val'];test=new_tests(c,old);assert len(train)==512 and len(val)==64
 assert sha(c['warm_adapter'])==c['warm_adapter_sha256'];dump(r/'config.json',c);dump(r/'examples.json',train+val+test)
 counts=[]
 for split in ['train','val','id','expression','unseen_source','unseen_offset','composition']:
  rr=[x for x in train+val+test if x['split']==split];counts.append(dict(split=split,n=len(rr),source_ids=len(set(x['source_id'] for x in rr)),carry=sum(x['carry'] for x in rr),noncarry=sum(not x['carry'] for x in rr)))
 csvsave(r/'dataset_counts.csv',counts);dump(r/'source.json',dict(commit=commit(),script_sha256=sha(__file__),old_examples_sha256=sha(Path(c['previous_run'],'examples.json')),examples_sha256=sha(r/'examples.json'),warm_sha256=sha(c['warm_adapter']),base_checkpoint_bytes=Path(c['model_path'],'pytorch_model.bin').stat().st_size,base_checkpoint_mtime=Path(c['model_path'],'pytorch_model.bin').stat().st_mtime,optimizer_restart=True,new_test_identities_disjoint_all816=True))
 print('PREPARED',json.dumps(counts),flush=True)

def run(c,r,device):
 import numpy as np
 versions=bootstrap();dump(r/'packages.json',versions)
 import torch
 from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
 from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
 from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
 torch.set_num_threads(2);torch.manual_seed(c['seed']);torch.use_deterministic_algorithms(True);gpu=int(device.split(':')[1]);snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);free={int(x.split(',')[0]):int(x.split(',')[1]) for x in snapshot.splitlines()};assert free[gpu]>=c['min_free_mib']
 dump(r/'gpu.json',dict(device=device,snapshot=snapshot));props=torch.cuda.get_device_properties(device);torch.cuda.set_per_process_memory_fraction(c['allocator_cap_gib']*2**30/props.total_memory,device)
 rows=json.loads((r/'examples.json').read_text());train=[x for x in rows if x['split']=='train'];val=[x for x in rows if x['split']=='val'];test=[x for x in rows if x['split'] not in ['train','val']];canonical=[record(d,h,0,0,'canonical') for d in range(7) for h in range(24)]
 tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(c['model_path'])/'tokenizer.json'),eos_token='<|endoftext|>',pad_token='<|endoftext|>');encoded={}
 for row in rows+canonical:
  t=tok(row['text'],add_special_tokens=False,return_offsets_mapping=True);sites=[i for i,(a,b) in enumerate(t['offset_mapping']) if a<row['source_end'] and b>=row['source_end']];assert len(sites)==1;answer=tok.encode(row['answer'],add_special_tokens=False)+[tok.eos_token_id];assert len(t['input_ids'])+len(answer)<=c['max_length'];encoded[row['text']]=dict(prompt=t['input_ids'],answer=answer,site=sites[0])
 def pack(items,answer=True):
  seqs=[encoded[x['text']]['prompt']+(encoded[x['text']]['answer'] if answer else []) for x in items];n=max(map(len,seqs));ids=torch.full((len(items),n),tok.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
  for i,(row,seq) in enumerate(zip(items,seqs)):
   ids[i,:len(seq)]=torch.tensor(seq,device=device);mask[i,:len(seq)]=1
   if answer:
    start=len(encoded[row['text']]['prompt']);labels[i,start:len(seq)]=ids[i,start:len(seq)]
  pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0);data=dict(input_ids=ids,attention_mask=mask,position_ids=pos)
  if answer:data['labels']=labels
  return data
 print('Loading existing GPT-J BF16',device,flush=True);t=time.monotonic();base=GPTJForCausalLM.from_pretrained(c['model_path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',low_cpu_mem_usage=True).to(device);load_seconds=time.monotonic()-t;assert load_seconds<c['load_cap_seconds'];base.config.use_cache=False
 model=get_peft_model(base,LoraConfig(r=c['rank'],lora_alpha=c['alpha'],target_modules=['q_proj','v_proj'],layers_to_transform=c['lora_layers'],layers_pattern='h',bias='none',lora_dropout=0.,task_type='CAUSAL_LM'));warm=torch.load(c['warm_adapter'],map_location='cpu',weights_only=True);assert sha(c['warm_adapter'])==c['warm_adapter_sha256'];set_peft_model_state_dict(model,warm);loaded=get_peft_model_state_dict(model);assert all(torch.equal(loaded[k].cpu(),warm[k]) for k in warm)
 params=[x for x in model.parameters() if x.requires_grad];assert sum(x.numel() for x in params)==262144;begin=time.monotonic();captured={}
 def hook(module,args,out):captured['hidden']=out[0] if isinstance(out,tuple) else out
 handle=base.transformer.h[26].register_forward_hook(hook)
 def limit():
  if time.monotonic()-begin>c['active_cap_seconds']:raise TimeoutError('Approved600s active budget exceeded')
 def ce(items):
  out=model(**pack(items),use_cache=False);captured.clear();return out.loss
 def generate(items):
  outputs=[];model.eval()
  for i in range(0,len(items),c['eval_batch']):
   limit();batch=items[i:i+c['eval_batch']];seq=[encoded[x['text']]['prompt'] for x in batch];n=max(map(len,seq));ids=torch.full((len(batch),n),tok.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
   for j,ss in enumerate(seq):ids[j,-len(ss):]=torch.tensor(ss,device=device);mask[j,-len(ss):]=1
   with torch.no_grad():y=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=c['max_new_tokens'],do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   captured.clear()
   for row,yy in zip(batch,y[:,n:].cpu().tolist()):
    text=tok.decode(yy,skip_special_tokens=True);m=re.fullmatch(r'\s*('+'|'.join(DAYS)+r')\s+(\d{1,2}):00\s*[.!]?\s*',text);valid=m is not None and int(m.group(2))<24;correct=bool(valid and DAYS.index(m.group(1))==row['answer_day'] and int(m.group(2))==row['answer_hour']);outputs.append(dict(id=row['id'],split=row['split'],source_id=row['source_id'],carry=row['carry'],expected=row['answer'].strip(),text=text,tokens=yy,correct=correct,valid_format=bool(valid),strict_exact=text.strip()==row['answer'].strip(),weekday_correct=bool(valid and DAYS.index(m.group(1))==row['answer_day']),hour_correct=bool(valid and int(m.group(2))==row['answer_hour'])))
  return outputs
 def disk():
  size=sum(p.stat().st_size for p in r.rglob('*') if p.is_file());assert size<=c['disk_cap_mb']*2**20;return size
 def validate(step):
  model.eval();pred=generate(val);total=0.
  with torch.no_grad():
   for i in range(0,len(val),c['eval_batch']):limit();batch=val[i:i+c['eval_batch']];total+=float(ce(batch))*len(batch)
  path=Path(c['warm_adapter']) if step==0 else r/'adapters'/f'adapter_total{128+step}.pt'
  if step:
   torch.save({k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()},path)
  dump(r/f'val_total{128+step}.json',pred);rec=dict(additional_updates=step,total_updates=128+step,n=64,correct=sum(x['correct'] for x in pred),valid_format=sum(x['valid_format'] for x in pred),carry_n=sum(x['carry'] for x in pred),carry_correct=sum(x['correct'] for x in pred if x['carry']),answer_token_ce=total/len(val),path=str(path),sha256=sha(path));curve.append(rec);dump(r/'validation_curve.json',curve);disk();assert torch.cuda.max_memory_allocated(device)<13*2**30;print('VALIDATE',json.dumps(rec),flush=True)
 (r/'adapters').mkdir(exist_ok=False);curve=[];model.eval();sample=pack(train[:2])
 with torch.no_grad():a=model(**sample,use_cache=False).logits;captured.clear();b=model(**sample,use_cache=False).logits;captured.clear();assert float((a-b).abs().max())==0.
 dump(r/'preflight.json',dict(trainable_lora_parameters=262144,warm_loaded_exact=True,repeat_logits_maxdiff=0.,load_seconds=load_seconds,optimizer_restart=True,allocator_cap_gib=c['allocator_cap_gib']))
 validate(0);opt=torch.optim.AdamW(params,lr=c['lr'],weight_decay=0.);orders={epoch:np.random.default_rng(c['seed']+epoch).permutation(512) for epoch in range(1,8)};actual=0;budget_stop=False
 for step in range(c['additional_updates_max']):
  limit()
  if time.monotonic()-begin>c['active_cap_seconds']-c['evaluation_reserve_seconds']:budget_stop=True;break
  model.train();opt.zero_grad(set_to_none=True);epoch=step//128+1;offset=step%128
  for micro in range(c['accumulation']):
   ii=(offset*c['accumulation']+micro)*c['micro_batch'];ix=orders[epoch][ii:ii+c['micro_batch']];assert len(ix)==2;loss=ce([train[j] for j in ix])/c['accumulation'];assert torch.isfinite(loss);loss.backward()
  norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();actual=step+1
  if actual%32==0:print('TRAIN',actual,128+actual,float(loss.detach())*c['accumulation'],float(norm),flush=True)
  if actual%c['validation_every']==0:validate(actual)
 selected=min(curve,key=lambda x:(-x['correct'],x['answer_token_ce'],x['total_updates']));assert sha(selected['path'])==selected['sha256'];dump(r/'test_lock.json',dict(commit=commit(),selected=selected,warm_path=c['warm_adapter'],warm_sha256=c['warm_adapter_sha256'],additional_updates_completed=actual,budget_stop=budget_stop,selection=c['selection']))
 allpred=[];metrics=[];geometry=[];reader=np.load(Path(c['previous_run'])/'fixed_readout.npz')
 def hidden(items):
  arr=[];model.eval()
  with torch.no_grad():
   for i in range(0,len(items),c['eval_batch']):
    limit();batch=items[i:i+c['eval_batch']];model(**pack(batch,answer=False),use_cache=False);site=torch.tensor([encoded[x['text']]['site'] for x in batch],device=device);arr.append(captured.pop('hidden')[torch.arange(len(batch),device=device),site].float().cpu().numpy())
  return np.concatenate(arr)
 for arm,path in [('Warm128',c['warm_adapter']),('Selected',selected['path'])]:
  set_peft_model_state_dict(model,torch.load(path,map_location='cpu',weights_only=True));print('EVALUATE_NEW_TEST',arm,flush=True);result=generate(test);allpred.extend(dict(arm=arm,**x) for x in result)
  for cohort in ['id','expression','unseen_source','unseen_offset','composition']:
   for group in ['all','carry','noncarry']:
    rr=[x for x in result if x['split']==cohort and (group=='all' or x['carry']==(group=='carry'))];n=len(rr);metrics.append(dict(arm=arm,cohort=cohort,group=group,n=n,correct=sum(x['correct'] for x in rr),accuracy=sum(x['correct'] for x in rr)/n if n else None,valid_format=sum(x['valid_format'] for x in rr),strict_exact=sum(x['strict_exact'] for x in rr),weekday_correct=sum(x['weekday_correct'] for x in rr),hour_correct=sum(x['hour_correct'] for x in rr)))
  hh=hidden(canonical);ht=hidden(test);np.savez_compressed(r/f'hidden_{arm}.npz',canonical=hh.astype('float16'),test=ht.astype('float16'));z=ht.astype(float)@reader['weight']+reader['bias'];d=np.array([x['day'] for x in test]);h=np.array([x['hour'] for x in test]);target=np.c_[np.cos(h*2*np.pi/24),np.sin(h*2*np.pi/24),np.cos(d*2*np.pi/7),np.sin(d*2*np.pi/7)]/np.sqrt(2)
  for name,v in [('hour',hh.reshape(7,24,-1).mean(0)),('weekday',hh.reshape(7,24,-1).mean(1))]:
   v=v.astype(float)-v.mean(0);u,s,vt=np.linalg.svd(v,full_matrices=False);power=(abs(np.fft.fft(v,axis=0))**2).sum(1);q=u[:,:2]*s[:2];radius=np.linalg.norm(q,axis=1);geometry.append(dict(arm=arm,factor=name,native_first_harmonic_energy=float((power[1]+power[-1])/power[1:].sum()),native_pca2_energy=float(np.sum(s[:2]**2)/np.sum(s*s)),native_radius_cv=float(radius.std()/radius.mean()),fixed_readout_mse=float(np.mean((z-target)**2))))
  dump(r/'new_test_predictions.json',allpred);csvsave(r/'metrics.csv',metrics);csvsave(r/'geometry.csv',geometry);disk()
 set_peft_model_state_dict(model,torch.load(selected['path'],map_location='cpu',weights_only=True));traincheck=generate(train[:64]);dump(r/'selected_train64_predictions.json',traincheck);dump(r/'selected_train64.json',dict(status='descriptive fixed first64 training examples, not used for selection',n=64,correct=sum(x['correct'] for x in traincheck),valid_format=sum(x['valid_format'] for x in traincheck)))
 handle.remove();assert sha(c['warm_adapter'])==c['warm_adapter_sha256'] and sha(selected['path'])==selected['sha256'];assert Path(c['model_path'],'pytorch_model.bin').stat().st_mtime==json.loads((r/'source.json').read_text())['base_checkpoint_mtime']
 dump(r/'completion.json',dict(commit=commit(),load_seconds=load_seconds,active_seconds=time.monotonic()-begin,additional_updates=actual,total_updates=128+actual,selected=selected,budget_stop=budget_stop,trainable_parameters=262144,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,disk_bytes=disk(),new_test_identities_disjoint=True,warm_unchanged=True,base_file_unchanged=True,device=device));print('COMPLETE',json.dumps(json.loads((r/'completion.json').read_text())),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:5');a=p.parse_args();c=json.loads(Path(a.config).read_text());r=Path(c['run'])
 try:
  if a.action=='prepare':prepare(c,r)
  else:run(c,r,a.device)
 except BaseException as ex:
  if r.exists():dump(r/f'failure_{a.action}.json',dict(error=str(ex),traceback=traceback.format_exc()))
  raise
