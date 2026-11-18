"""Frozen GPT-J conditional-coordinate exchange and rescue, offline <=600s."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import argparse,json,sys,time,subprocess,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.chronogeom.phase0_02_patching import block_input_hook
from scripts.chronogeom.phase0_03_run import bf16_match
from scripts.chronogeom.phase0_04_subspaces import orth

SWAP=['helix7','fourier7','pca7','shared','unique_h','unique_f','linear_increment','pca1','random1','random7','full','noop']
CONDITIONAL=SWAP+['pca6','random6','helix8']
RESCUE=['shared','unique_h','unique_f','linear_increment','helix7','fourier7','pca7','random7','helix8','full','noop']

def project(x,u):return (x@u.T)@u

def control_exchange(operation,method,applied):
 # Rescue baseline is already damaged. Omitting this hook would restore clean.
 return None if method=='noop' and operation!='rescue' else applied

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--repair-rescue-noop',action='store_true');args=ap.parse_args();cfg=json.loads(args.config.read_text());out=args.output;out.mkdir(exist_ok=False);cfg['repair_rescue_noop']=args.repair_rescue_noop;start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42)
 source=Path(cfg['source']);audit=Path(cfg['audit']);oldcfg=json.loads((source/'config.json').read_text());sc=json.loads((source/'screening.json').read_text());pairs=[p for p in sc['cohorts']['fixed'] if p['selected']];fit=json.loads((Path(oldcfg['source_run'])/'fit_tokenization.json').read_text());cached=np.load(source/'fixed_paired_hidden.npz');train=np.load(source/'fit_activations.npz')['hidden'][:,:,0];spaces=np.load(audit/'subspaces.npz');layers=oldcfg['layers']
 assert len(pairs)==85 and all(p['clean_a'] not in oldcfg['fit_values'] and p['corrupt_a'] not in oldcfg['fit_values'] for p in pairs)
 state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);available=[int(r.split(',')[0]) for r in state.splitlines() if float(r.split(',')[1])>=22000 and float(r.split(',')[2])<=5];assert available,'No idle GPU';device='cuda:'+str(available[0]);cfg['actual_device']=device;(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');(out/'preflight.json').write_text(json.dumps(dict(gpu_state=state,selected=device,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('phase0_04_*.py')}),indent=2)+'\n')
 from transformers import AutoTokenizer,AutoModelForCausalLM
 tok=AutoTokenizer.from_pretrained(oldcfg['model_path'],local_files_only=True);tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(oldcfg['model_path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to(device).eval()
 def encode(a,b):
  text=f'Output ONLY a number.\n{a}+{b}=';begin=text.index('\n')+1;z=tok(text,return_offsets_mapping=True,add_special_tokens=False);positions=[i for i,(s,e) in enumerate(z['offset_mapping']) if e>begin and s<begin+len(str(a))];ans=tok(str(a+b),add_special_tokens=False)['input_ids'];assert len(positions)==len(ans)==1;return dict(ids=z['input_ids'],pos=positions[0],answer=ans[0])
 clean=[encode(p['clean_a'],p['b']) for p in pairs];corrupt=[encode(p['corrupt_a'],p['b']) for p in pairs];ct=torch.tensor([r['answer'] for r in clean]);rt=torch.tensor([r['answer'] for r in corrupt])
 def forward(records,layer=None,exchange=None,capture=False):
  n=max(map(lambda r:len(r['ids']),records));ids=torch.full((len(records),n),tok.pad_token_id,device=device,dtype=torch.long);mask=torch.zeros_like(ids);pos=[];handles=[];caps={}
  for i,r in enumerate(records):ids[i,-len(r['ids']):]=torch.tensor(r['ids'],device=device);mask[i,-len(r['ids']):]=1;pos.append(n-len(r['ids'])+r['pos'])
  def hook(l):
   def cap(module,a,kw):
    h=kw.get('hidden_states',a[0] if a else None);caps[l]=h[torch.arange(len(h),device=device),pos].detach().float().cpu()
   return cap
  try:
   if capture:
    for l in layers:handles.append(model.transformer.h[l].register_forward_pre_hook(hook(l),with_kwargs=True))
   if layer is not None:handles.append(model.transformer.h[layer].register_forward_pre_hook(block_input_hook(pos,replacement=None if exchange is None else torch.from_numpy(exchange.astype(np.float32))),with_kwargs=True))
   with torch.inference_mode():log=model(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(1)-1).clamp(min=0),use_cache=False).logits[:,-1].float().cpu()
  finally:
   for h in handles:h.remove()
  return log,np.stack([caps[l].numpy() for l in layers],axis=1) if capture else None
 def batches(records,layer=None,exchange=None,capture=False):
  logs=[];hs=[]
  for i in range(0,len(records),8):
   if time.monotonic()-start>cfg['wall_limit']-15:raise TimeoutError('600s preregistered hard cap, partial rows preserved')
   log,h=forward(records[i:i+8],layer,None if exchange is None else exchange[i:i+8],capture);logs.append(log)
   if capture:hs.append(h)
  return torch.cat(logs),np.concatenate(hs) if capture else None
 cl,ch=batches(clean,capture=True);rl,rh=batches(corrupt,capture=True);cache_error=max(float(np.max(np.abs(ch-cached['clean'][:,:,0]))),float(np.max(np.abs(rh-cached['corrupt'][:,:,0]))));assert cache_error==0,'Cached/current activations differ, stop before intervention'
 assert torch.all(cl.argmax(1)==ct) and torch.all(rl.argmax(1)==rt),'Fixed baseline drift; do not reselect'
 fixed_ix=np.flatnonzero(spaces[f'{layers[0]}_reference_indices']>=0);subset_clean=[clean[i] for i in fixed_ix];subset_cl,subset_ch=batches(subset_clean,capture=True);assert torch.all(subset_cl.argmax(1)==ct[fixed_ix]);subset_cache_error=float(np.max(np.abs(subset_ch-ch[fixed_ix])));np.savez(out/'subset_hidden.npz',hidden=subset_ch,pair_ids=np.array([pairs[i]['id'] for i in fixed_ix]));subset_by_layer=[dict(layer=l,max_abs=float(np.max(np.abs(subset_ch[:,j]-ch[fixed_ix,j]))),n_rows_changed=int(np.sum(np.any(subset_ch[:,j]!=ch[fixed_ix,j],axis=1)))) for j,l in enumerate(layers)]
 rows=0;writer=(out/'interventions.jsonl').open('x');all_control=[]
 for li,l in enumerate(layers):
  u={k.split('_',1)[1]:spaces[k] for k in spaces.files if k.startswith(str(l)+'_')};mean=u['mean'];refs=u['reference_indices'];ix=np.flatnonzero(refs>=0);assert len(ix)==82
  # Current-shape controls for the 82 subset prevent batch-shape effects being
  # counted as patching. Hidden cache is exact original baseline, not recomputed
  # with different batch boundaries for intervention construction.
  assert np.array_equal(ix,fixed_ix);local_clean=subset_ch[:,li].astype(float);baseclean=ch[:,li].astype(float);basecorrupt=rh[:,li].astype(float);ref=train[refs[ix],li].astype(float)
  def cast(x):return torch.from_numpy(np.asarray(x,dtype=np.float32)).bfloat16().float().numpy()
  damaged=cast(local_clean+project(ref-local_clean,u['helix8']));damlog,_=batches(subset_clean,l,damaged)
  def execute(operation,method,mode='raw',seed=42):
   nonlocal rows
   selected=np.arange(85) if operation=='swap' else ix;p=[pairs[i] for i in selected];base=basecorrupt if operation=='swap' else (local_clean if operation=='conditional' else damaged.astype(float));donor=baseclean if operation=='swap' else (ref if operation=='conditional' else local_clean);baseline=rl if operation=='swap' else (subset_cl if operation=='conditional' else damlog);records=corrupt if operation=='swap' else subset_clean
   axes=u.get(method);target=None
   if method.startswith('random') and seed!=42:axes=np.linalg.qr(np.random.default_rng(seed+l*101).normal(size=(4096,7)))[0].T[:int(method[6:])]
   if method=='full':ideal=donor.copy()
   elif method=='noop':ideal=base.copy()
   else:ideal=base+project(donor-base,axes)
   if mode=='equal_linear_increment':
    target=np.linalg.norm(cast(base+project(donor-base,u['linear_increment']))-base,axis=1);ideal,zero=bf16_match(base.astype(np.float32),ideal.astype(np.float32),target.astype(np.float32))
   else:zero=np.zeros(len(p),dtype=bool)
   applied=cast(ideal);log,_=batches(records,l,control_exchange(operation,method,applied));changes=applied-base;norm=np.linalg.norm(changes,axis=1);mu=applied-mean;diag=project(mu,u['diag_axes']);md=np.mean(((mu@u['diag_axes'].T)**2)/u['diag_var'],axis=1);off=np.mean((mu-diag)**2,axis=1)
   recons={name:np.mean((mu-project(mu,u[name]))**2,axis=1) for name in ['helix7','fourier7','pca7']}
   for j,global_i in enumerate(selected):
    pair=p[j];clean_id=int(ct[global_i]);corrupt_id=int(rt[global_i]);v=log[j];b=baseline[j];before=float(b[clean_id]-b[corrupt_id]);after=float(v[clean_id]-v[corrupt_id]);gain=after-before;pred=int(v.argmax());candidates=[k for k,r in enumerate(fit) if r['b']==pair['b']];nn=float(np.min(np.linalg.norm(train[candidates,li]-applied[j],axis=1))) if candidates else None;nn0=float(np.min(np.linalg.norm(train[candidates,li]-base[j],axis=1))) if candidates else None
    row=dict(pair_id=pair['id'],clean_a=pair['clean_a'],corrupt_a=pair['corrupt_a'],b=pair['b'],layer=l,site='a',operation=operation,method=method,rank=4096 if method=='full' else (0 if method=='noop' else len(axes)),mode=mode,control_seed=seed,natural_reference_row=int(refs[global_i]) if operation!='swap' else None,baseline_ld=before,patched_ld=after,ld_change=gain,effect=-gain if operation=='conditional' else gain,target_correct=pred==clean_id,baseline_clean_correct=int(b.argmax())==clean_id,baseline_text=tok.decode([int(b.argmax())]),predicted_text=tok.decode([pred]),prediction_changed=pred!=int(b.argmax()),max_logit_change=float(torch.max(torch.abs(v-b))),patch_l2=float(norm[j]),rounding_l2=float(np.linalg.norm(applied[j]-ideal[j])),requested_l2=None if target is None else float(target[j]),relative_norm_error=None if target is None else float(abs(norm[j]-target[j])/max(target[j],1e-12)),zero_direction=bool(zero[j]),train_pca16_regularized_score_m2=float(md[j]),train_pca16_complement_mse=float(off[j]),nearest_same_b_train_l2=nn,baseline_nearest_same_b_train_l2=nn0,nearest_distance_ratio=None if nn0 is None or nn0<=1e-12 else nn/nn0,helix7_reconstruction_mse=float(recons['helix7'][j]),fourier7_reconstruction_mse=float(recons['fourier7'][j]),pca7_reconstruction_mse=float(recons['pca7'][j]),clean_ld=float(cl[global_i,clean_id]-cl[global_i,corrupt_id]) if operation=='swap' else float(subset_cl[j,clean_id]-subset_cl[j,corrupt_id]),damage_ld=float(damlog[j,clean_id]-damlog[j,corrupt_id]) if operation=='rescue' else None)
    writer.write(json.dumps(row)+'\n');rows+=1
   writer.flush();print(l,operation,method,mode,seed,'rows',rows,flush=True)
   if method=='noop':all_control.append(dict(layer=l,operation=operation,max_logit_change=float(torch.max(torch.abs(log-baseline)))))
  if args.repair_rescue_noop:
   execute('rescue','noop');continue
  for method in SWAP:execute('swap',method)
  for method in ['linear_increment','unique_h','unique_f','pca1','random1']:execute('swap',method,'equal_linear_increment')
  for method in CONDITIONAL:execute('conditional',method)
  for method in RESCUE:execute('rescue',method)
  if l in [1,9]:
   for seed in [43,44]:
    execute('swap','random1','equal_linear_increment',seed);execute('conditional','random6',seed=seed);execute('rescue','random7',seed=seed)
 writer.close();meta=dict(rows=rows,seconds=time.monotonic()-start,cache_max_error=cache_error,subset_batchshape_hidden_max_difference=subset_cache_error,subset_batchshape_by_layer=subset_by_layer,repair_rescue_noop=args.repair_rescue_noop,baseline_drift=0,noop_controls=all_control,peak_allocated=torch.cuda.max_memory_allocated(device),dtype=str(next(model.parameters()).dtype),gpu=device,python=sys.version,torch=torch.__version__,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print('COMPLETE',json.dumps(meta),flush=True)
if __name__=='__main__':main()
