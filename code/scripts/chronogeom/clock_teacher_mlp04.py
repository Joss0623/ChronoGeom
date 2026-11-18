"""ClockTeacher04: equal-capacity Q/V14 vs Q/V4+existing-MLP4, same frozen Circle03 task."""
from scripts.chronogeom.clock_teacher_circle03 import row,targets,parse,COHORTS
from scripts.chronogeom.clock_teacher_lora01 import bootstrap,DAYS,dump,sha,csvsave,commit
from pathlib import Path
import argparse,json,math,subprocess,time,traceback
ARMS=['QV14','QV_MLP4']

def prepare(c,r):
 import shutil
 p=Path(c['previous_run']);r.mkdir(parents=True,exist_ok=False);dump(r/'config.json',c);copied=[]
 for name in ['examples.json','fixed_frame.npz','query_calibration.npz','dataset_counts.csv','analytical_sincos.json']:
  shutil.copy2(p/name,r/name);assert sha(p/name)==sha(r/name);copied.append(dict(name=name,sha256=sha(r/name),bytes=(r/name).stat().st_size))
 source=json.loads((p/'source.json').read_text())
 dump(r/'source.json',dict(commit=commit(),script_sha256=sha(__file__),previous_run=str(p),copies=copied,examples_sha256=sha(r/'examples.json'),frame_sha256=sha(r/'fixed_frame.npz'),warm_sha256=sha(c['warm_adapter']),base_mtime=source['base_mtime'],same_previously_viewed_test=True,new_split=False,fixed_aux_weights=c['fixed_aux_weights'],helper_script_sha256=sha(Path('scripts/chronogeom/clock_teacher_circle03.py'))))
 assert c['fixed_aux_weights']==json.loads((p/'preflight.json').read_text())['fixed_aux_weights']
 print('PREPARED exact Circle03 data/frame/calibration/weights reused',flush=True)

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
 start=time.monotonic()
 model=get_peft_model(base,LoraConfig(r=14,lora_alpha=28,target_modules=['q_proj','v_proj'],layers_to_transform=[24,25,26,27],layers_pattern='h',bias='none',lora_dropout=0.,task_type='CAUSAL_LM'),adapter_name='QV14')
 model.add_adapter('QV_MLP4',LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj','fc_in','fc_out'],layers_to_transform=[24,25,26,27],layers_pattern='h',bias='none',lora_dropout=0.,task_type='CAUSAL_LM'))
 warm=torch.load(c['warm_adapter'],map_location='cpu',weights_only=True);assert sha(c['warm_adapter'])==c['warm_adapter_sha256'];initial={};init_manifest=[]
 for arm in ARMS:
  model.set_adapter(arm);state={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model,adapter_name=arm).items()}
  for k,v in warm.items():
   if arm=='QV14':
    if k.endswith('lora_A.weight'):state[k][:4].copy_(v)
    else:state[k].zero_();state[k][:,:4].copy_(v)
   else:state[k].copy_(v)
  if arm=='QV_MLP4':assert all(torch.count_nonzero(v)==0 for k,v in state.items() if 'fc_' in k and 'lora_B' in k)
  # Padding preserves the effective warm Q/V map; extra capacity starts at zero contribution.
  maxerr=0.
  for k,v in warm.items():
   if k.endswith('lora_A.weight'):
    kb=k.replace('lora_A.weight','lora_B.weight');probe=torch.linspace(-1,1,v.shape[1]).reshape(1,-1)
    maxerr=max(maxerr,float((((probe@state[k].T)@state[kb].T)-((probe@v.T)@warm[kb].T)).abs().max()))
  assert maxerr<1e-5
  set_peft_model_state_dict(model,state,adapter_name=arm);initial[arm]=state
  params=[x for x in model.parameters() if x.requires_grad];assert sum(x.numel() for x in params)==917504
  path=r/f'initial_{arm}.pt';torch.save(state,path);init_manifest.append(dict(arm=arm,trainable_parameters=917504,tensors=len(state),warm_function_probe_maxdiff=maxerr,path=str(path),sha256=sha(path)))
 dump(r/'initialization.json',init_manifest);model.set_adapter('QV14');params=[x for x in model.parameters() if x.requires_grad]
 captured={};patch={}
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
 query_ref=np.load(Path(c['previous_run'])/'query_calibration.npz');query_mu=query_ref['mean'];query_scale=query_ref['scale']
 assert sha(r/'query_calibration.npz')==sha(Path(c['previous_run'])/'query_calibration.npz')
 mus['query']=torch.tensor(query_mu,device=device);scales['query']=torch.tensor(query_scale.repeat(2)*np.sqrt(2),device=device)
 def z(h,slot):return ((h.float()-mus[slot])@q)/scales[slot]
 def rotate(v,delta):
  angle=delta[:,None]*torch.tensor([2*math.pi/24,2*math.pi/168],device=device)[None,:];p=v.reshape(-1,2,2);a,b=p[:,:,0],p[:,:,1];co,si=angle.cos(),angle.sin()
  return torch.stack([a*co-b*si,a*si+b*co],dim=-1).reshape(-1,4)
 def forward(items,stage):
  pp,ss,tt=pack(items,stage);ans=model(**pp,use_cache=False);h=captured.pop('hidden');ii=torch.arange(len(items),device=device);zin=z(h[ii,ss],'source');zout=z(h[ii,tt],'query');n=2 if stage=='hour' else 4
  yin=torch.tensor(targets([x['t'] for x in items]),device=device,dtype=torch.float32);yout=torch.tensor(targets([x['destination'] for x in items]),device=device,dtype=torch.float32)
  state=((zin[:,:n]-yin[:,:n])**2).mean()+((zout[:,:n]-yout[:,:n])**2).mean();rot=((zout[:,:n]-rotate(zin,torch.tensor([x['delta'] for x in items],device=device))[:,:n])**2).mean()
  return ans.loss,state,rot
 weights=c['fixed_aux_weights'];assert weights==json.loads((Path(c['previous_run'])/'preflight.json').read_text())['fixed_aux_weights']
 gradient_checks=[];initial_logits={}
 for arm in ARMS:
  model.set_adapter(arm);set_peft_model_state_dict(model,initial[arm],adapter_name=arm);params=[x for x in model.parameters() if x.requires_grad];assert sum(x.numel() for x in params)==917504
  model.eval();batch=train[:2];pp,ss,tt=pack(batch)
  with torch.no_grad():
   logits=model(**pp,use_cache=False).logits.float().cpu();original=captured.pop('hidden')[torch.arange(2,device=device),tt].clone();changed={k:v.clone() for k,v in pp.items()};changed['input_ids'][changed['labels']!=-100]=tok.eos_token_id;model(**changed,use_cache=False);causal_error=float((captured.pop('hidden')[torch.arange(2,device=device),tt]-original).abs().max());assert causal_error==0.;initial_logits[arm]=logits
  model.train();ce,st,ro=forward(batch,'full');norms=[];group_norms={}
  for label,loss in [('ce',ce),('state',st),('rotation',ro)]:
   gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);norms.append(float(torch.sqrt(sum((g.float()**2).sum() for g in gs if g is not None))))
   named=[n for n,p in model.named_parameters() if p.requires_grad]
   for group in ['qv','mlp']:
    terms=[(g.float()**2).sum() for n,g in zip(named,gs) if g is not None and (('fc_' in n)==(group=='mlp'))];group_norms[label+'_'+group]=float(torch.sqrt(sum(terms))) if terms else 0.
  assert min(norms)>0
  if arm=='QV_MLP4':assert group_norms['ce_mlp']>0 and group_norms['state_mlp']>0 and group_norms['rotation_mlp']>0
  gradient_checks.append(dict(arm=arm,trainable_parameters=917504,causal_answer_perturbation_maxdiff=causal_error,gradient_norms=norms,group_gradient_norms=group_norms))
  del ce,st,ro,gs,loss;model.zero_grad(set_to_none=True)
 initial_maxdiff=float((initial_logits['QV14']-initial_logits['QV_MLP4']).abs().max());initial_rms=float(torch.sqrt(torch.mean((initial_logits['QV14']-initial_logits['QV_MLP4'])**2)));assert initial_maxdiff<.5 and initial_rms<.01
 del initial_logits
 dump(r/'preflight.json',dict(trainable_parameters=917504,load_seconds=load,checks=gradient_checks,fixed_aux_weights=weights,weights_reused_exact=True,query_calibration_reused_exact=True,initial_arm_logit_maxdiff=initial_maxdiff,initial_arm_logit_rms=initial_rms,orthogonality_maxerror=float(np.max(abs(frame['q'].T@frame['q']-np.eye(4))))))
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
 for arm in ARMS:
  (r/arm).mkdir(exist_ok=False);model.set_adapter(arm);set_peft_model_state_dict(model,initial[arm],adapter_name=arm);params=[x for x in model.parameters() if x.requires_grad];assert sum(x.numel() for x in params)==917504;opt=torch.optim.AdamW(params,lr=c['lr'],weight_decay=0.);tarm=time.monotonic()
  for step in range(c['updates']):
   limit();assert time.monotonic()-start<c['active_cap_seconds']-c['eval_reserve_seconds'];stage='hour' if step<c['hour_updates'] else 'full';model.train();opt.zero_grad(set_to_none=True)
   for micro in range(2):
    ix=orders[step//128][(step%128)*4+micro*2:(step%128)*4+micro*2+2];bb=[train[j] for j in ix];a,b,cc=forward(bb,stage);loss=(a+weights[0]*b+weights[1]*cc)/2;assert torch.isfinite(loss);loss.backward()
   norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));opt.step()
   if (step+1)%32==0:
    rec=dict(arm=arm,step=step+1,stage=stage,ce=float(a.detach()),state=float(b.detach()),rotation=float(cc.detach()),grad_norm=norm);trainlog.append(rec);print('TRAIN',json.dumps(rec),flush=True)
   if step+1 in c['validation_steps']:
    pred=generate(val);vce=0.
    with torch.no_grad():
     for i in range(0,64,c['eval_batch']):
      limit();a,_,_=forward(val[i:i+c['eval_batch']],'full');vce+=float(a)*len(val[i:i+c['eval_batch']])/64
    path=r/arm/f'adapter_step{step+1}.pt';torch.save({k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model,adapter_name=arm).items()},path);dump(r/arm/f'val_step{step+1}.json',pred)
    rec=dict(arm=arm,step=step+1,n=64,correct=sum(x['correct'] for x in pred),clock_correct=sum(x['clock_correct'] for x in pred),carry_correct=sum(x['carry_correct'] for x in pred),format=sum(x['valid_format'] for x in pred),ce=vce,path=str(path),sha256=sha(path));curves.append(rec);dump(r/'validation_curve.json',curves);csvsave(r/'train_curve.csv',trainlog);disk();print('VALIDATE',json.dumps(rec),flush=True)
  dump(r/arm/'training_complete.json',dict(updates=c['updates'],seconds=time.monotonic()-tarm));del opt
 selected={arm:min([x for x in curves if x['arm']==arm],key=lambda x:(-x['correct'],x['ce'],x['step'])) for arm in ARMS}
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
 for arm in ARMS:
  state=torch.load(selected[arm]['path'],map_location='cpu',weights_only=True);assert sha(selected[arm]['path'])==selected[arm]['sha256'];assert all(bool(torch.isfinite(v).all()) for v in state.values());model.set_adapter(arm);set_peft_model_state_dict(model,state,adapter_name=arm);model.eval();print('TEST',arm,flush=True)
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
 dump(r/'completion.json',dict(commit=commit(),load_seconds=load,active_seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,disk_bytes=disk(),updates_per_arm=c['updates'],trainable_parameters=917504,selected=selected,all_noops_exact=True,warm_unchanged=True,base_mtime_unchanged=True,device=device));print('COMPLETE',json.dumps(json.loads((r/'completion.json').read_text())),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:2');a=p.parse_args();c=json.loads(Path(a.config).read_text());r=Path(c['run'])
 try:
  if a.action=='prepare':prepare(c,r)
  else:run(c,r,a.device)
 except BaseException as e:
  if r.exists():dump(r/f'failure_{a.action}.json',dict(error=str(e),traceback=traceback.format_exc()))
  raise
