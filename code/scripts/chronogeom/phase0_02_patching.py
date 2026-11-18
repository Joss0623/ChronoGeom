"""Independent local-weight activation patching; never downloads or trains a model."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'2')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import sys,json,time,argparse,hashlib,subprocess
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.chronogeom.representation_probe import basis

def block_input_hook(positions, capture=None, replacement=None):
 def hook(module,args,kwargs):
  hidden=kwargs.get('hidden_states',args[0] if args else None)
  if hidden is None:raise ValueError('hidden_states unavailable')
  ix=torch.arange(len(hidden),device=hidden.device);pos=torch.as_tensor(positions,device=hidden.device)
  if capture is not None:capture.append(hidden[ix,pos].detach().float().cpu())
  if replacement is None:return None
  changed=hidden.clone();changed[ix,pos]=replacement.to(hidden.device,hidden.dtype)
  if 'hidden_states' in kwargs:
   kwargs=dict(kwargs);kwargs['hidden_states']=changed;return args,kwargs
  return (changed,)+args[1:],kwargs
 return hook

def replacement(kind,corrupt,clean,subspace=None):
 if kind=='noop':return corrupt.clone()
 if kind=='full':return clean.clone()
 if subspace is None:raise ValueError('subspace required')
 u=subspace.to(corrupt.device,corrupt.dtype)
 return corrupt+(clean-corrupt)@u.T@u

def canonical_integer(value):
 if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or value!=int(value):raise ValueError('integer identity required')
 return int(value)

def same_token_layout(c,r):
 return bool(c['valid'] and r['valid'] and len(c['ids'])==len(r['ids']) and c['pos']==r['pos'] and [x for i,x in enumerate(c['ids']) if i!=c['pos']]==[x for i,x in enumerate(r['ids']) if i!=r['pos']])

def generate_pairs(seed,count,eval_values,b_max):
 rng=np.random.default_rng(seed);all_pairs=[(a,ap,b) for a in eval_values for ap in eval_values if a!=ap for b in range(b_max)]
 ix=rng.choice(len(all_pairs),size=min(count,len(all_pairs)),replace=False)
 return [{'id':i,'clean_a':int(all_pairs[j][0]),'corrupt_a':int(all_pairs[j][1]),'b':int(all_pairs[j][2])} for i,j in enumerate(ix)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();cfg=json.loads(args.config.read_text());args.output.mkdir(parents=True,exist_ok=False)
 cfg['fit_values']=[canonical_integer(v) for v in cfg['fit_values']];cfg['eval_values']=[canonical_integer(v) for v in cfg['eval_values']]
 if not cfg['fit_values'] or not cfg['eval_values'] or set(cfg['fit_values'])&set(cfg['eval_values']):raise ValueError('nonempty disjoint identity sets required')
 torch.set_num_threads(cfg.get('threads',2));torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);start=time.monotonic()
 from transformers import AutoModelForCausalLM,AutoTokenizer
 path=Path(cfg['model_path']).expanduser();tok=AutoTokenizer.from_pretrained(path,local_files_only=True);tok.pad_token=tok.eos_token
 device=cfg.get('device','cpu');dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32
 model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=dtype,attn_implementation='eager').to(device).eval()
 layers=cfg['layers'];assert max(layers)<len(model.transformer.h)
 def encode(a,b):
  text=f'Output ONLY a number.\n{a}+{b}=';begin=text.index('\n')+1;end=begin+len(str(a));r=tok(text,return_offsets_mapping=True,add_special_tokens=False)
  inds=[i for i,(s,e) in enumerate(r['offset_mapping']) if e>begin and s<end];ans=tok(str(a+b),add_special_tokens=False)['input_ids']
  return {'a':a,'b':b,'text':text,'ids':r['input_ids'],'pos':inds[0] if len(inds)==1 else None,'answer_ids':ans,'valid':len(inds)==1 and len(ans)==1}
 fit_values=cfg['fit_values'];eval_values=cfg['eval_values'];assert not set(fit_values)&set(eval_values)
 rng=np.random.default_rng(cfg['seed']);fit=[encode(a,int(b)) for a in fit_values for b in rng.choice(100,size=cfg['fit_b_per_a'],replace=False)]
 pairs=generate_pairs(cfg['seed'],cfg['candidate_pairs'],eval_values,cfg['b_max']);problems={}
 for pair in pairs:
  for role in ['clean','corrupt']:
   key=f"{pair[role+'_a']}+{pair['b']}";pair[role+'_key']=key;problems[key]=encode(pair[role+'_a'],pair['b'])
 for pair in pairs:
  c=problems[pair['clean_key']];r=problems[pair['corrupt_key']];pair['token_layout_valid']=same_token_layout(c,r)
 (args.output/'candidate_pairs.json').write_text(json.dumps(pairs,indent=2)+'\n')
 def forward(records, patch_layer=None, exchange=None, capture_all=False):
  n=max(len(r['ids']) for r in records);ids=torch.full((len(records),n),tok.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids);positions=[]
  for i,r in enumerate(records):
   offset=n-len(r['ids']);ids[i,offset:]=torch.tensor(r['ids'],device=device);mask[i,offset:]=1;positions.append(offset+r['pos'])
  posids=(mask.cumsum(1)-1).clamp(min=0);caps={};handles=[]
  try:
   if capture_all:
    for l in layers:
     caps[l]=[];handles.append(model.transformer.h[l].register_forward_pre_hook(block_input_hook(positions,capture=caps[l]),with_kwargs=True))
   if patch_layer is not None:handles.append(model.transformer.h[patch_layer].register_forward_pre_hook(block_input_hook(positions,replacement=exchange),with_kwargs=True))
   with torch.inference_mode():logits=model(input_ids=ids,attention_mask=mask,position_ids=posids,use_cache=False).logits[:,-1].float().cpu()
  finally:
   for handle in handles:handle.remove()
  return logits,{l:torch.cat(v) for l,v in caps.items()}
 def batches(records):
  for i in range(0,len(records),cfg['batch_size']):yield records[i:i+cfg['batch_size']]
 (args.output/'fit_tokenization.json').write_text(json.dumps(fit,indent=2)+'\n')
 fit=[r for r in fit if r['valid']]
 if not fit:raise ValueError('No valid fit prompts; refusing empty geometric fit')
 fit_h={l:[] for l in layers}
 for batch in batches(fit):
  _,cap=forward(batch,capture_all=True)
  for l in layers:fit_h[l].append(cap[l])
 print('fitting activations collected',len(fit),flush=True)
 if cfg.get('audit_cache'):
  old=torch.load(Path(cfg['audit_cache']).expanduser(),weights_only=True,map_location='cpu');oa=old['a'].numpy();fa=np.array([r['a'] for r in fit]);ids=np.unique(fa);alignment=[]
  for l in layers:
   new=torch.cat(fit_h[l]).numpy().astype(float);old_y=old['hidden_states'][:,l].float().numpy().astype(float)
   target=np.stack([old_y[oa==a].mean(0) for a in ids]);pred=np.stack([new[fa==a].mean(0) for a in ids]);den=np.sum((target-target.mean(0))**2)
   alignment.append(dict(cache_index=l,current_hook='transformer.h[layer] input at a',n_identities=len(ids),r2_to_old_cache=float(1-np.sum((target-pred)**2)/den),relative_l2=float(np.linalg.norm(target-pred)/np.linalg.norm(target)),max_absolute_difference=float(np.abs(target-pred).max()),warning='consistency check with current FP16-source/BF16-run checkpoint, not proof of historical generation revision'))
  (args.output/'current_vs_old_cache_alignment.json').write_text(json.dumps(alignment,indent=2)+'\n');del old
 bases={};rankmap={};fitted_maps={};methods=['noop','full','helix','pca','random']+cfg.get('extra_conditions',[])
 assert set(methods)<=set(['noop','full','helix','pca','random','helix_fit_replacement','pca_reconstruction8','pca_reconstruction9'])
 for l in layers:
  y=torch.cat(fit_h[l]).numpy().astype(float);des=basis([r['a'] for r in fit],'helix');coef=np.linalg.lstsq(des,y,rcond=1e-10)[0]
  _,s,v=np.linalg.svd(coef[:-1],full_matrices=False);rank=int(np.sum(s>s[0]*1e-8));rankmap[l]=rank
  _,_,pca=np.linalg.svd(y-y.mean(0),full_matrices=False);q,_=np.linalg.qr(np.random.default_rng(cfg['seed']+l).normal(size=(y.shape[1],rank)))
  fitted_maps[l]=(torch.tensor(coef,dtype=torch.float32),torch.tensor(y.mean(0),dtype=torch.float32))
  bases[l]={'pca9':torch.tensor(pca[:9],dtype=torch.float32),'helix':torch.tensor(v[:rank],dtype=torch.float32),'pca':torch.tensor(pca[:rank],dtype=torch.float32),'random':torch.tensor(q.T,dtype=torch.float32)}
 states={};eligibility=[];valid=[]
 for key,r in problems.items():
  if r['valid']:r['key']=key;valid.append(r)
  else:eligibility.append(dict(key=key,valid=False,reason='multi_token_a_or_answer'))
 for batch in batches(valid):
  logits,cap=forward(batch,capture_all=True)
  for i,r in enumerate(batch):
   prediction=int(logits[i].argmax());correct=prediction==r['answer_ids'][0]
   eligibility.append(dict(key=r['key'],valid=True,correct=correct,target=r['a']+r['b'],prediction_token=prediction,prediction_text=tok.decode([prediction]),prompt=r['text'],a_token_position=r['pos'],a_token_text=tok.decode([r['ids'][r['pos']]])))
   states[r['key']]={'record':r,'logits':logits[i],'h':{l:cap[l][i] for l in layers},'correct':correct}
 (args.output/'eligibility.json').write_text(json.dumps(eligibility,indent=2)+'\n')
 selected=[]
 for p in pairs:
  p['eligible']=p['token_layout_valid'] and all(k in states and states[k]['correct'] for k in [p['clean_key'],p['corrupt_key']]);p['reason']='both_correct' if p['eligible'] else 'one_or_both_invalid_or_incorrect'
  if p['eligible']:selected.append(p)
 (args.output/'pair_screening.json').write_text(json.dumps(pairs,indent=2)+'\n');print('eligible pairs',len(selected),'of',len(pairs),flush=True)
 if cfg.get('plumbing_only',False):
  selected=[p for p in pairs if p['token_layout_valid'] and p['clean_key'] in states and p['corrupt_key'] in states][:2]
 results=[]
 for l in layers:
  for method in methods:
   for batch in batches(selected):
    records=[states[p['corrupt_key']]['record'] for p in batch];base_logits,base_caps=forward(records,capture_all=True);hc=torch.stack([states[p['clean_key']]['h'][l] for p in batch]);hr=base_caps[l];ex=None
    if method=='helix_fit_replacement':ex=torch.tensor(basis([p['clean_a'] for p in batch],'helix'),dtype=torch.float32)@fitted_maps[l][0]
    elif method.startswith('pca_reconstruction'):
     comp=bases[l]['pca9' if method.endswith('9') else 'pca'];mean=fitted_maps[l][1];ex=(hc-mean)@comp.T@comp+mean
    elif method!='noop':ex=replacement(method,hr,hc,bases[l].get(method))
    logits,_=forward(records,patch_layer=l,exchange=ex)
    for i,p in enumerate(batch):
     clean=states[p['clean_key']];corrupt=states[p['corrupt_key']];ct=clean['record']['answer_ids'][0];rt=corrupt['record']['answer_ids'][0];base=base_logits[i];log=logits[i];pred=int(log.argmax())
     results.append(dict(pair_id=p['id'],layer=l,method=method,rank=9 if method=='pca_reconstruction9' else (0 if method=='noop' else (hr.shape[1] if method=='full' else rankmap[l])),clean_answer=clean['record']['a']+p['b'],corrupt_answer=corrupt['record']['a']+p['b'],baseline_text=tok.decode([int(base.argmax())]),baseline_still_correct=int(base.argmax())==rt,patched_text=tok.decode([pred]),patched_token=pred,clean_token=ct,corrupt_token=rt,clean_target_logit_gain=float(log[ct]-base[ct]),baseline_logit_difference=float(base[ct]-base[rt]),patched_logit_difference=float(log[ct]-log[rt]),logit_difference_gain=float((log[ct]-log[rt])-(base[ct]-base[rt])),restores_clean=pred==ct,patch_l2=float(torch.linalg.vector_norm((ex[i].to(dtype).float()-hr[i]) if ex is not None else torch.zeros_like(hr[i]))),rounding_l2=float(torch.linalg.vector_norm(ex[i].to(dtype).float()-ex[i])) if ex is not None else 0.,full_delta_l2=float(torch.linalg.vector_norm(hc[i]-hr[i])),max_logit_change=float(torch.max(torch.abs(log-base)))))
   print('patched',l,method,flush=True)
 (args.output/'per_pair_results.json').write_text(json.dumps(results,indent=2)+'\n')
 summary=[]
 for l in layers:
  for method in methods:
   rs=[r for r in results if r['layer']==l and r['method']==method]
   summary.append(dict(layer=l,method=method,n=len(rs),rank=9 if method=='pca_reconstruction9' else (0 if method=='noop' else (model.config.hidden_size if method=='full' else rankmap[l])),mean_gain=float(np.mean([r['logit_difference_gain'] for r in rs])) if rs else None,mean_target_gain=float(np.mean([r['clean_target_logit_gain'] for r in rs])) if rs else None,restoration=float(np.mean([r['restores_clean'] for r in rs])) if rs else None,max_noop_error=max([r['max_logit_change'] for r in rs],default=None) if method=='noop' else None))
 (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 meta=dict(config=cfg,run_dtype=str(dtype),actual_parameter_dtype=str(next(model.parameters()).dtype),model_type=model.config.model_type,model_config=model.config.to_dict(),tokenizer_class=type(tok).__name__,eligible_pairs=sum(p['eligible'] for p in pairs),patched_pairs=len(selected),candidate_pairs=len(pairs),fit_rows=len(fit),seed=cfg['seed'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.monotonic()-start,torch=torch.__version__,python=sys.version,cuda_peak_allocated=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,cuda_peak_reserved=torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else None,evidence='incorrect_allowed_plumbing_only' if cfg.get('plumbing_only',False) else 'correct_pair_causal_experiment',protocol='block input at a token; additive projected clean-minus-corrupt replacement; fit/eval a disjoint; joint correct screen before patch; one next-token output; predeclared optional full fit/PCA reconstruction controls')
 (args.output/'metadata.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')
if __name__=='__main__':main()
