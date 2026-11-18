"""ClockTeacher03: bounded decomposed SFT vs fixed-frame daily/weekly Circle."""
from scripts.chronogeom.clock_teacher_lora01 import bootstrap,DAYS,dump,sha,csvsave,commit
from pathlib import Path
import argparse,datetime as dt,json,math,random,re,subprocess,time,traceback
COHORTS=['id','unseen_offset','composition','expression','crossweek']

def row(d,h,k,split,idx,parts=None):
 d,h,k=int(d),int(h),int(k);t=d*24+h;dest=(t+k)%168;q=(h+k)//24;source=f'{DAYS[d]} {h:02d}:00'
 if parts: text=f'Time: {source}. First shift {parts[0]:+d} hours, then {parts[1]:+d} hours.'
 elif split=='expression':text=f'Starting at {source}, move the clock {abs(k)} hours {"forward" if k>=0 else "backward"}.'
 else:text=f'Time: {source}. Shift: {k:+d} hours.'
 return dict(id=idx,day=d,hour=h,delta=k,t=t,destination=dest,source_id=t,identity=[d,h,k],split=split,parts=parts,
  text=text+' Answer (carry hour day):',hour_text=text+' Answer (carry hour):',source_end=text.index(source)+len(source),
  answer=f' {q:+d} {dest%24:02d} {DAYS[dest//24]}',hour_answer=f' {q:+d} {dest%24:02d}',answer_carry=q,
  answer_hour=dest%24,answer_day=dest//24,carry=q!=0,crossweek=not 0<=t+k<168)

def data(c):
 rng=random.Random(c['data_seed']);old={tuple(x['identity']) for p in c['old_runs'] for x in json.loads((Path(p)/'examples.json').read_text())};used=set(old)
 pool=[(d,h,k) for d in range(7) for h in range(24) for k in c['train_offsets'] if (d,h,k) not in used];rng.shuffle(pool)
 cross=[];remaining={sid:sum(24*x[0]+x[1]==sid for x in pool) for sid in range(168)}
 for sign in [-1,1]:
  candidates=[x for x in pool if (24*x[0]+x[1]+x[2]<0 if sign<0 else 24*x[0]+x[1]+x[2]>=168)]
  assert len(candidates)>=24;chosen=[]
  for x in candidates:
   sid=24*x[0]+x[1]
   if remaining[sid]>1:chosen.append(x);remaining[sid]-=1
   if len(chosen)==24:break
  assert len(chosen)==24;cross+=chosen
 used.update(cross)
 def take(n,pool,coverage=False):
  candidates=[x for x in pool if x not in used];rng.shuffle(candidates);chosen=[];quota={False:n//2,True:n//2}
  if coverage:
   sources=[(d,h) for d in range(7) for h in range(24)];rng.shuffle(sources)
   for d,h in sources:
    options=[x for x in candidates if x[:2]==(d,h) and quota[(x[1]+x[2])//24!=0]>0 and x not in chosen];assert options
    x=options[0];chosen.append(x);quota[(x[1]+x[2])//24!=0]-=1
  for x in candidates:
   carry=(x[1]+x[2])//24!=0
   if x not in chosen and quota[carry]>0:chosen.append(x);quota[carry]-=1
  assert len(chosen)==n and all(v==0 for v in quota.values());used.update(chosen);rng.shuffle(chosen);return chosen
 groups={'train':take(512,pool,True),'val':take(64,pool),'id':take(48,pool),'expression':take(48,pool),'crossweek':cross}
 other=[(d,h,k) for d in range(7) for h in range(24) for k in c['test_offsets']];groups['unseen_offset']=take(48,other)
 comps={}
 for d in range(7):
  for h in range(24):
   for a in c['composition_parts']:
    for b in c['composition_parts']:
     k=a+b
     if k not in c['train_offsets'] and k!=0 and (d,h,k) not in used:comps.setdefault((d,h,k),(a,b))
 groups['composition']=take(48,list(comps))
 rows=[]
 for split in ['train','val']+COHORTS:
  for key in groups[split]:
   rows.append(row(*key,split,len(rows),parts=comps[key] if split=='composition' else None))
 assert len(rows)==816 and len({tuple(x['identity']) for x in rows})==816
 assert not ({tuple(x['identity']) for x in rows}&old)
 assert len({x['source_id'] for x in rows if x['split']=='train'})==168
 for x in rows:
  t=dt.datetime(2024,1,1)+dt.timedelta(days=x['day'],hours=x['hour']+x['delta'])
  assert (t.weekday(),t.hour)==(x['answer_day'],x['answer_hour'])
  if x['parts']:assert sum(x['parts'])==x['delta']
 return rows,len(old)

def targets(t):
 import numpy as np
 t=np.asarray(t)
 return np.stack([np.cos(2*np.pi*t/24),np.sin(2*np.pi*t/24),np.cos(2*np.pi*t/168),np.sin(2*np.pi*t/168)],axis=-1)/np.sqrt(2)

def prepare(c,r):
 import numpy as np
 r.mkdir(parents=True,exist_ok=False);rows,nold=data(c);dump(r/'config.json',c);dump(r/'examples.json',rows)
 hh=np.load(c['canonical_cache'])['canonical'].astype('float64');assert hh.shape==(168,4096)
 ids=sorted({x['source_id'] for x in rows if x['split']=='train'});x=hh[ids];mean=x.mean(0);x=x-mean;y=targets(ids);gram=x@x.T
 ridge=np.trace(gram)/len(gram)*1e-4;b=x.T@np.linalg.solve(gram+ridge*np.eye(len(gram)),y)
 blocks=[]
 for j in [0,2]:
  v=b[:,j:j+2].copy()
  if blocks:v-=blocks[0]@(blocks[0].T@v)
  ss=np.linalg.svd(v,compute_uv=False);assert ss[-1]>ss[0]*1e-6
  q=np.linalg.qr(v)[0][:,:2];u,_,vt=np.linalg.svd((x@q).T@y[:,j:j+2]);q=q@(u@vt);blocks.append(q)
 q=np.column_stack(blocks);assert np.max(abs(q.T@q-np.eye(4)))<1e-8
 scale=np.array([np.sqrt(np.mean(np.sum((x@q[:,j:j+2])**2,axis=1))) for j in [0,2]]);assert min(scale)>1e-5
 np.savez(r/'fixed_frame.npz',q=q.astype('float32'),source_mean=mean.astype('float32'),source_scale=scale.astype('float32'))
 counts=[]
 for split in ['train','val']+COHORTS:
  rr=[x for x in rows if x['split']==split];counts.append(dict(split=split,n=len(rr),sources=len({x['source_id'] for x in rr}),carry=sum(x['carry'] for x in rr),crossweek=sum(x['crossweek'] for x in rr),positive_delta=sum(x['delta']>0 for x in rr),negative_delta=sum(x['delta']<0 for x in rr)))
 csvsave(r/'dataset_counts.csv',counts);assert sha(c['warm_adapter'])==c['warm_adapter_sha256']
 dump(r/'source.json',dict(commit=commit(),script_sha256=sha(__file__),examples_sha256=sha(r/'examples.json'),frame_sha256=sha(r/'fixed_frame.npz'),canonical_cache_sha256=sha(c['canonical_cache']),warm_sha256=sha(c['warm_adapter']),base_mtime=Path(c['model_path'],'pytorch_model.bin').stat().st_mtime,old_identities_excluded=nold,all816_new_identities=True,calibration_uses_training_source_ids=True,training_source_ids=ids))
 print('PREPARED',json.dumps(counts),flush=True)

def parse(text,row):
 m=re.fullmatch(r'\s*([+-]\d+)\s+(\d{1,2})\s+('+'|'.join(DAYS)+r')\s*',text)
 valid=bool(m and int(m.group(2))<24)
 qok=bool(valid and int(m.group(1))==row['answer_carry']);hok=bool(valid and int(m.group(2))==row['answer_hour']);dok=bool(valid and DAYS.index(m.group(3))==row['answer_day'])
 return dict(valid_format=valid,carry_correct=qok,hour_correct=hok,day_correct=dok,correct=qok and hok and dok,clock_correct=hok and dok)

def run(c,r,device):
 import numpy as np
 dump(r/'packages.json',bootstrap())
 import torch
 import torch.nn.functional as F
 from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
 from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
 from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
 torch.set_num_threads(2);torch.manual_seed(c['seed']);torch.use_deterministic_algorithms(True);gpu=int(device.split(':')[1])
 snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);free={int(x.split(',')[0]):int(x.split(',')[1]) for x in snapshot.splitlines()};assert free[gpu]>=c['min_free_mib'];dump(r/'gpu.json',dict(device=device,snapshot=snapshot))
 torch.cuda.set_per_process_memory_fraction(c['allocator_cap_gib']*2**30/torch.cuda.get_device_properties(device).total_memory,device)
 rows=json.loads((r/'examples.json').read_text());train=[x for x in rows if x['split']=='train'];val=[x for x in rows if x['split']=='val'];test=[x for x in rows if x['split'] in COHORTS]
 canonical=[row(d,h,0,'canonical',10000+24*d+h) for d in range(7) for h in range(24)]
 tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(c['model_path'])/'tokenizer.json'),eos_token='<|endoftext|>',pad_token='<|endoftext|>');encoded={}
 def enc(x,stage='full'):
  key=(x['text'],stage)
  if key not in encoded:
   text=x['text'] if stage=='full' else x['hour_text'];t=tok(text,add_special_tokens=False,return_offsets_mapping=True);site=[i for i,(a,b) in enumerate(t['offset_mapping']) if a<x['source_end'] and b>=x['source_end']];assert len(site)==1
   a=tok.encode(x['answer'] if stage=='full' else x['hour_answer'],add_special_tokens=False)+[tok.eos_token_id];assert len(t['input_ids'])+len(a)<=c['max_length']
   encoded[key]=dict(prompt=t['input_ids'],answer=a,site=site[0])
  return encoded[key]
 def pack(items,stage='full',answers=True):
  es=[enc(x,stage) for x in items];seq=[e['prompt']+(e['answer'] if answers else []) for e in es];n=max(map(len,seq));ids=torch.full((len(items),n),tok.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids);lab=torch.full_like(ids,-100)
  for i,(e,s) in enumerate(zip(es,seq)):
   ids[i,:len(s)]=torch.tensor(s,device=device);mask[i,:len(s)]=1
   if answers:lab[i,len(e['prompt']):len(s)]=ids[i,len(e['prompt']):len(s)]
  pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0);p=dict(input_ids=ids,attention_mask=mask,position_ids=pos)
  if answers:p['labels']=lab
  src=torch.tensor([e['site'] for e in es],device=device);qry=torch.tensor([len(e['prompt'])-1 for e in es],device=device)
  return p,src,qry
 print('LOAD existing GPT-J',device,flush=True);load=time.monotonic();base=GPTJForCausalLM.from_pretrained(c['model_path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',low_cpu_mem_usage=True).to(device);load=time.monotonic()-load;assert load<c['load_cap_seconds'];base.config.use_cache=False
 model=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],layers_to_transform=[24,25,26,27],layers_pattern='h',bias='none',lora_dropout=0.,task_type='CAUSAL_LM'))
 warm=torch.load(c['warm_adapter'],map_location='cpu',weights_only=True);set_peft_model_state_dict(model,warm);assert sha(c['warm_adapter'])==c['warm_adapter_sha256'];assert all(torch.equal(v.detach().cpu(),warm[k]) for k,v in get_peft_model_state_dict(model).items())
 params=[x for x in model.parameters() if x.requires_grad];assert sum(x.numel() for x in params)==262144
 start=time.monotonic();captured={};patch={}
 def limit():
  if time.monotonic()-start>c['active_cap_seconds']:raise TimeoutError('600s GPU active cap reached')
 def disk():
  size=sum(p.stat().st_size for p in r.rglob('*') if p.is_file());assert size<c['disk_cap_mb']*2**20;return size
 def hook(module,args,out):
  h=out[0] if isinstance(out,tuple) else out
  if patch and h.shape[1]>1:
   h=h.clone();ii=torch.arange(h.shape[0],device=device);h[ii,patch['positions']]=patch['vectors'].to(h.dtype)
   out=(h,)+out[1:] if isinstance(out,tuple) else h
  captured['hidden']=h;return out
 handle=base.transformer.h[26].register_forward_hook(hook)
 frame=np.load(r/'fixed_frame.npz');q=torch.tensor(frame['q'],device=device);mus={'source':torch.tensor(frame['source_mean'],device=device)};scales={'source':torch.tensor(frame['source_scale'].repeat(2)*np.sqrt(2),device=device)}
 def extract(items,stage='full'):
  hs=[];hq=[];model.eval()
  with torch.no_grad():
   for i in range(0,len(items),c['eval_batch']):
    limit();bb=items[i:i+c['eval_batch']];pp,ss,tt=pack(bb,stage,False);model(**pp,use_cache=False);h=captured.pop('hidden');ii=torch.arange(len(bb),device=device);hs.append(h[ii,ss].float().cpu().numpy());hq.append(h[ii,tt].float().cpu().numpy())
  return np.concatenate(hs),np.concatenate(hq)
 _,query0=extract(train);query_mu=query0.mean(0);queryraw=(query0-query_mu)@frame['q'];query_scale=np.array([np.sqrt(np.mean(np.sum(queryraw[:,j:j+2]**2,axis=1))) for j in [0,2]]);assert min(query_scale)>1e-6
 mus['query']=torch.tensor(query_mu,device=device);scales['query']=torch.tensor(query_scale.repeat(2)*np.sqrt(2),device=device);np.savez(r/'query_calibration.npz',mean=query_mu,scale=query_scale)
 def z(h,slot):return ((h.float()-mus[slot])@q)/scales[slot]
 def rotate(v,delta):
  angle=delta[:,None]*torch.tensor([2*math.pi/24,2*math.pi/168],device=device)[None,:];p=v.reshape(-1,2,2);a,b=p[:,:,0],p[:,:,1];co,si=angle.cos(),angle.sin()
  return torch.stack([a*co-b*si,a*si+b*co],dim=-1).reshape(-1,4)
 def forward(items,stage):
  pp,ss,tt=pack(items,stage);ans=model(**pp,use_cache=False);h=captured.pop('hidden');ii=torch.arange(len(items),device=device);zin=z(h[ii,ss],'source');zout=z(h[ii,tt],'query');n=2 if stage=='hour' else 4
  yin=torch.tensor(targets([x['t'] for x in items]),device=device,dtype=torch.float32);yout=torch.tensor(targets([x['destination'] for x in items]),device=device,dtype=torch.float32)
  state=((zin[:,:n]-yin[:,:n])**2).mean()+((zout[:,:n]-yout[:,:n])**2).mean();rot=((zout[:,:n]-rotate(zin,torch.tensor([x['delta'] for x in items],device=device))[:,:n])**2).mean()
  return ans.loss,state,rot
 model.eval();batch=train[:2];pp,ss,tt=pack(batch)
 with torch.no_grad():
  model(**pp,use_cache=False);original=captured.pop('hidden')[torch.arange(2,device=device),tt].clone();changed={k:v.clone() for k,v in pp.items()};changed['input_ids'][changed['labels']!=-100]=tok.eos_token_id;model(**changed,use_cache=False);causal_error=float((captured.pop('hidden')[torch.arange(2,device=device),tt]-original).abs().max());assert causal_error==0.
 model.train();ce,st,ro=forward(train[:2],'full')
 norms=[]
 for loss in [ce,st,ro]:
  gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);norms.append(float(torch.sqrt(sum((g.float()**2).sum() for g in gs if g is not None))))
 assert min(norms)>0;weights=[c['aux_gradient_ratio']*norms[0]/norms[1],c['aux_gradient_ratio']*norms[0]/norms[2]]
 del ce,st,ro,gs;model.zero_grad(set_to_none=True)
 dump(r/'preflight.json',dict(trainable_parameters=262144,load_seconds=load,causal_answer_perturbation_maxdiff=causal_error,gradient_norms=norms,fixed_aux_weights=weights,orthogonality_maxerror=float(np.max(abs(frame['q'].T@frame['q']-np.eye(4)))),query_calibration_train_n=512))
 def generate(items,ivectors=None):
  result=[];model.eval()
  for i in range(0,len(items),c['eval_batch']):
   limit();bb=items[i:i+c['eval_batch']];ee=[enc(x) for x in bb];n=max(len(e['prompt']) for e in ee);ids=torch.full((len(bb),n),tok.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
   for j,e in enumerate(ee):ids[j,-len(e['prompt']):]=torch.tensor(e['prompt'],device=device);mask[j,-len(e['prompt']):]=1
   if ivectors is not None:
    patch.update(positions=torch.tensor([n-len(e['prompt'])+e['site'] for e in ee],device=device),vectors=torch.tensor(ivectors[i:i+len(bb)],device=device))
   try:
    with torch.no_grad():yy=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=c['max_new_tokens'],do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   finally:patch.clear();captured.clear()
   for x,tokens in zip(bb,yy[:,n:].cpu().tolist()):
    text=tok.decode(tokens,skip_special_tokens=True);result.append(dict(id=x['id'],split=x['split'],source_id=x['source_id'],carry=x['carry'],text=text,tokens=tokens,expected=x['answer'].strip(),**parse(text,x)))
  return result
 curves=[];trainlog=[];orders={epoch:np.random.default_rng(c['seed']+epoch).permutation(512) for epoch in range(3)}
 for arm in ['SFT','Circle']:
  (r/arm).mkdir(exist_ok=False);set_peft_model_state_dict(model,warm);opt=torch.optim.AdamW(params,lr=c['lr'],weight_decay=0.);tarm=time.monotonic()
  for step in range(c['updates']):
   limit();assert time.monotonic()-start<c['active_cap_seconds']-c['eval_reserve_seconds'];stage='hour' if step<c['hour_updates'] else 'full';model.train();opt.zero_grad(set_to_none=True)
   for micro in range(2):
    ix=orders[step//128][(step%128)*4+micro*2:(step%128)*4+micro*2+2];bb=[train[j] for j in ix];a,b,cc=forward(bb,stage);loss=(a+(weights[0]*b+weights[1]*cc if arm=='Circle' else 0))/2;assert torch.isfinite(loss);loss.backward()
   norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));opt.step()
   if (step+1)%32==0:
    rec=dict(arm=arm,step=step+1,stage=stage,ce=float(a.detach()),state=float(b.detach()),rotation=float(cc.detach()),grad_norm=norm);trainlog.append(rec);print('TRAIN',json.dumps(rec),flush=True)
   if step+1 in c['validation_steps']:
    pred=generate(val);vce=0.
    with torch.no_grad():
     for i in range(0,64,c['eval_batch']):
      limit();a,_,_=forward(val[i:i+c['eval_batch']],'full');vce+=float(a)*len(val[i:i+c['eval_batch']])/64
    path=r/arm/f'adapter_step{step+1}.pt';torch.save({k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()},path);dump(r/arm/f'val_step{step+1}.json',pred)
    rec=dict(arm=arm,step=step+1,n=64,correct=sum(x['correct'] for x in pred),clock_correct=sum(x['clock_correct'] for x in pred),carry_correct=sum(x['carry_correct'] for x in pred),format=sum(x['valid_format'] for x in pred),ce=vce,path=str(path),sha256=sha(path));curves.append(rec);dump(r/'validation_curve.json',curves);csvsave(r/'train_curve.csv',trainlog);disk();print('VALIDATE',json.dumps(rec),flush=True)
  dump(r/arm/'training_complete.json',dict(updates=c['updates'],seconds=time.monotonic()-tarm));del opt
 selected={arm:min([x for x in curves if x['arm']==arm],key=lambda x:(-x['correct'],x['ce'],x['step'])) for arm in ['SFT','Circle']}
 dump(r/'test_lock.json',dict(commit=commit(),selected=selected,test_ids=[x['id'] for x in test],patch_ids=[x['id'] for x in test[:24]],selection='val full generated triple accuracy, CE, earlier step'))
 allpred=[];metrics=[];geo=[];patchrows=[];trainchecks=[]
 def summarize(arm,rr):
  for group in ['all']+COHORTS+['carry','noncarry']:
   vv=[x for x in rr if group=='all' or x['split']==group or group=='carry' and x['carry'] or group=='noncarry' and not x['carry']]
   metrics.append(dict(arm=arm,group=group,n=len(vv),sources=len({x['source_id'] for x in vv}),**{k:sum(x[k] for x in vv) for k in ['correct','clock_correct','valid_format','carry_correct','hour_correct','day_correct']}))
 def logit_contrast(items,donors,vectors):
  vals=[]
  for i in range(0,len(items),c['eval_batch']):
   limit();bb=items[i:i+c['eval_batch']];dd=donors[i:i+len(bb)];seq=[];a=[];b=[]
   for x,y in zip(bb,dd):
    ea,eb=enc(x)['answer'],enc(y)['answer'];j=next(j for j,(u,v) in enumerate(zip(ea,eb)) if u!=v);seq.append(enc(x)['prompt']+ea[:j]);a.append(ea[j]);b.append(eb[j])
   n=max(map(len,seq));ids=torch.full((len(bb),n),tok.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
   for j,s in enumerate(seq):ids[j,:len(s)]=torch.tensor(s,device=device);mask[j,:len(s)]=1
   pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0)
   if vectors is not None:patch.update(positions=torch.tensor([enc(x)['site'] for x in bb],device=device),vectors=torch.tensor(vectors[i:i+len(bb)],device=device))
   try:
    with torch.no_grad():
     logits=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).logits[torch.arange(len(bb),device=device),torch.tensor([len(s)-1 for s in seq],device=device)].float()
     ld=logits[torch.arange(len(bb),device=device),torch.tensor(b,device=device)]-logits[torch.arange(len(bb),device=device),torch.tensor(a,device=device)];vals+=ld.cpu().tolist()
   finally:patch.clear();captured.clear()
  return vals
 for arm in ['SFT','Circle']:
  state=torch.load(selected[arm]['path'],map_location='cpu',weights_only=True);assert sha(selected[arm]['path'])==selected[arm]['sha256'];assert all(bool(torch.isfinite(v).all()) for v in state.values());set_peft_model_state_dict(model,state);model.eval();print('TEST',arm,flush=True)
  pred=generate(test);allpred.extend(dict(arm=arm,**x) for x in pred);summarize(arm,pred);dump(r/'test_predictions.json',allpred);csvsave(r/'metrics.csv',metrics)
  check=generate(train[:64]);trainchecks.extend(dict(arm=arm,**x) for x in check)
  hs,hout=extract(test);hc,_=extract(canonical);np.savez_compressed(r/f'hidden_{arm}.npz',canonical=hc.astype('float16'),source=hs.astype('float16'),query=hout.astype('float16'))
  zs=((hs-frame['source_mean'])@frame['q'])/frame['source_scale'].repeat(2)/np.sqrt(2);zo=((hout-query_mu)@frame['q'])/query_scale.repeat(2)/np.sqrt(2);yin=targets([x['t'] for x in test]);yout=targets([x['destination'] for x in test]);angle=np.stack([np.array([x['delta'] for x in test])*2*np.pi/p for p in [24,168]],axis=-1);p=zs.reshape(-1,2,2);a,b=p[:,:,0],p[:,:,1];rz=np.stack([a*np.cos(angle)-b*np.sin(angle),a*np.sin(angle)+b*np.cos(angle)],axis=-1).reshape(-1,4)
  native=hc.astype('float64').reshape(7,24,-1);hour=native.mean(0);hour-=hour.mean(0);pw=(abs(np.fft.fft(hour,axis=0))**2).sum(1);week=hc.astype('float64')-hc.mean(0);pww=(abs(np.fft.fft(week,axis=0))**2).sum(1)
  for slot,zz,yy in [('source',zs,yin),('query',zo,yout)]:
   for j,name in [(0,'daily'),(2,'weekly')]:
    radii=np.linalg.norm(zz[:,j:j+2],axis=1);ang=np.arctan2(zz[:,j+1],zz[:,j])-np.arctan2(yy[:,j+1],yy[:,j]);err=np.abs(np.arctan2(np.sin(ang),np.cos(ang)))
    geo.append(dict(arm=arm,slot=slot,period=name,state_mse=float(np.mean((zz[:,j:j+2]-yy[:,j:j+2])**2)),angle_mae_degrees=float(np.mean(err)*180/np.pi),radius_cv=float(radii.std()/radii.mean()),rotation_mse=float(np.mean((zo[:,j:j+2]-rz[:,j:j+2])**2)),native_hour_harmonic=float((pw[1]+pw[-1])/pw[1:].sum()),native_week_harmonic=float((pww[1]+pww[-1])/pww[1:].sum())))
  # All fixed cases are retained, whether the model originally answers correctly or not.
  recipients=test[:24];donors=[]
  for i,x in enumerate(recipients):
   s=3 if i%2==0 else -3;t=(x['t']+s)%168;donors.append(row(t//24,t%24,x['delta'],'donor',20000+i))
  hr,_=extract(recipients);hd,_=extract(donors);qq=frame['q'];rng=np.random.default_rng(c['random_subspace_seed']);qr=np.linalg.qr(rng.normal(size=(4096,4)))[0].astype('float32');delta=hd-hr;circle_delta=delta@qq@qq.T;rand=delta@qr@qr.T;rand*=np.linalg.norm(circle_delta,axis=1,keepdims=True)/np.maximum(np.linalg.norm(rand,axis=1,keepdims=True),1e-12)
  zz=((hr-frame['source_mean'])@qq)/frame['source_scale'].repeat(2)/np.sqrt(2);dr=np.array([3 if i%2==0 else -3 for i in range(24)]);angles=dr[:,None]*np.array([2*np.pi/24,2*np.pi/168])[None,:];p=zz.reshape(-1,2,2);a,b=p[:,:,0],p[:,:,1];rot=np.stack([a*np.cos(angles)-b*np.sin(angles),a*np.sin(angles)+b*np.cos(angles)],axis=-1).reshape(-1,4);rotvec=hr+((rot-zz)*frame['source_scale'].repeat(2)*np.sqrt(2))@qq.T
  controls={'Noop':hr,'FullNatural':hd,'CircleNatural':hr+circle_delta,'CircleRotate':rotvec,'RandomMatchedNorm':hr+rand}
  np.savez_compressed(r/f'patch_vectors_{arm}.npz',source=hr.astype('float16'),donor=hd.astype('float16'),**{k:x.astype('float16') for k,x in controls.items()})
  for cond,vectors in controls.items():
   out=generate(recipients,None if cond=='Noop' else vectors);ld=logit_contrast(recipients,donors,None if cond=='Noop' else vectors)
   for i,(p,x,y) in enumerate(zip(out,recipients,donors)):
    dv=vectors[i]-hr[i];near=float(np.min(np.linalg.norm(hc-vectors[i],axis=1)));old=parse(p['text'],x);new=parse(p['text'],y)
    patchrows.append(dict(arm=arm,condition=cond,id=x['id'],source_id=x['source_id'],shift=int(dr[i]),original_expected=x['answer'].strip(),counterfactual_expected=y['answer'].strip(),text=p['text'],original_correct=old['correct'],counterfactual_correct=new['correct'],cf_logit_difference=ld[i],perturbation_norm=float(np.linalg.norm(dv)),activation_norm=float(np.linalg.norm(vectors[i])),nearest_canonical_distance=near))
   print('PATCH',arm,cond,sum(parse(p['text'],y)['correct'] for p,y in zip(out,donors)),'/24',flush=True);dump(r/'patch_predictions.json',patchrows)
  original={x['id']:x['text'] for x in pred};noop=[x for x in patchrows if x['arm']==arm and x['condition']=='Noop'];assert all(x['text']==original[x['id']] for x in noop)
  dump(r/'test_predictions.json',allpred);dump(r/'train64_predictions.json',trainchecks);csvsave(r/'metrics.csv',metrics);csvsave(r/'geometry.csv',geo);disk()
 oracle=[]
 for x in test:
  before=targets([x['t']])[0];angles=np.array([2*np.pi*x['delta']/24,2*np.pi*x['delta']/168]);a,b=before.reshape(2,2).T;after=np.stack([a*np.cos(angles)-b*np.sin(angles),a*np.sin(angles)+b*np.cos(angles)],axis=-1).reshape(4);error=float(np.max(abs(after-targets([x['destination']])[0])));assert error<1e-12
  hour=int(round((np.arctan2(after[1],after[0])%(2*np.pi))*24/(2*np.pi)))%24;week=int(round((np.arctan2(after[3],after[2])%(2*np.pi))*168/(2*np.pi)))%168;assert hour==x['answer_hour'] and week//24==x['answer_day']
  oracle.append(dict(id=x['id'],split=x['split'],correct=True,carry=x['answer_carry'],rotation_maxerror=error))
 dump(r/'analytical_sincos.json',oracle);handle.remove();assert sha(c['warm_adapter'])==c['warm_adapter_sha256'];assert Path(c['model_path'],'pytorch_model.bin').stat().st_mtime==json.loads((r/'source.json').read_text())['base_mtime']
 dump(r/'completion.json',dict(commit=commit(),load_seconds=load,active_seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,disk_bytes=disk(),updates_per_arm=c['updates'],trainable_parameters=262144,selected=selected,all_noops_exact=True,warm_unchanged=True,base_mtime_unchanged=True,device=device));print('COMPLETE',json.dumps(json.loads((r/'completion.json').read_text())),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:4');a=p.parse_args();c=json.loads(Path(a.config).read_text());r=Path(c['run'])
 try:
  if a.action=='prepare':prepare(c,r)
  else:run(c,r,a.device)
 except BaseException as e:
  if r.exists():dump(r/f'failure_{a.action}.json',dict(error=str(e),traceback=traceback.format_exc()))
  raise
