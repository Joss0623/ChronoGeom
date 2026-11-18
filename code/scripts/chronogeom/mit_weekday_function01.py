"""Frozen GPT-J weekday functional test. No training, downloads, or forecasting fits."""
import os
os.environ.update(USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0')
import argparse, csv, hashlib, importlib.util, importlib.metadata as md, json, subprocess, time, traceback
from pathlib import Path
DAYS=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
NUM=['Zero days','One day','Two days','Three days','Four days','Five days','Six days','Seven days']
SHOTS="Let's do some days of the week math. One day from Monday is Tuesday.\nLet's do some days of the week math. Two days from Thursday is Saturday.\nLet's do some days of the week math. Three days from Saturday is Tuesday.\n"
METHODS=['noop','full','phase2','pca2','random2','pca2_norm','random2_norm','learned_transition','analytic_rotation']

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def records():
    rows=[]
    for context in range(2):
        for split,offsets in [('fit',[4,5]),('test',[1,2,3])]:
            for k in offsets:
                for d,day in enumerate(DAYS):
                    text=(SHOTS if context else '')+f"Let's do some days of the week math. {NUM[k]} from {day} is"
                    rows.append(dict(id=len(rows),context=context,split=split,offset=k,day=d,answer=(d+k)%7,text=text))
    return rows

def rotation(delta):
    import numpy as np
    t=delta*2*np.pi/7
    return np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])

def norm_match(delta,target):
    import numpy as np
    n=np.linalg.norm(delta); t=np.linalg.norm(target)
    if n<1e-12:return delta.copy(),False
    return delta*(t/n),True

def fit_space(h,rows,layer):
    import numpy as np
    day=np.array([r['day'] for r in rows]);ctx=np.array([r['context'] for r in rows]);off=np.array([r['offset'] for r in rows])
    phase=np.c_[np.cos(day*2*np.pi/7),np.sin(day*2*np.pi/7)]
    nuisance=np.c_[np.ones(len(h)),ctx,off==5]
    residual=h-nuisance@np.linalg.lstsq(nuisance,h,rcond=1e-10)[0]
    coef=np.linalg.lstsq(phase,residual,rcond=1e-10)[0]
    _,s,_=np.linalg.svd(coef,full_matrices=False)
    if s[-1]<s[0]*1e-8:raise ValueError('Phase fit rank deficient')
    u=np.linalg.qr(coef.T)[0][:,:2]
    _,ps,pv=np.linalg.svd(residual,full_matrices=False);pca=pv[:2].T
    random=np.linalg.qr(np.random.default_rng(1729+layer).normal(size=(h.shape[1],2)))[0]
    mu=h.mean(0);cent=np.stack([h[day==d].mean(0)-mu for d in range(7)])
    q=cent@u;d=np.c_[np.cos(np.arange(7)*2*np.pi/7),np.sin(np.arange(7)*2*np.pi/7)]
    a=np.linalg.lstsq(d,q,rcond=1e-10)[0]
    if np.linalg.cond(a)>1e6:raise ValueError('Ill-conditioned analytic calibration')
    learned={str(delta):np.linalg.lstsq(q,q[(np.arange(7)+delta)%7],rcond=1e-10)[0] for delta in [-1,1]}
    analytic={str(delta):np.linalg.solve(a,rotation(delta)@a) for delta in [-1,1]}
    energies=np.sum(abs(np.fft.rfft(cent,axis=0)[1:])**2,axis=1)
    _,cs,cv=np.linalg.svd(cent,full_matrices=False);native=cent@cv[:2].T
    rng=np.random.default_rng(1729);null=[]
    for _ in range(256):
        e=np.sum(abs(np.fft.rfft(cent[rng.permutation(7)],axis=0)[1:])**2,axis=1);null.append(float(e[0]/e.sum()))
    rec={}
    for name,b in [('phase2',u),('pca2',pca),('random2',random)]:
        rec[name]=dict(rank=int(np.linalg.matrix_rank(b)),orthogonality_error=float(abs(b.T@b-np.eye(2)).max()),fit_reconstruction_error=float(np.sum((residual-residual@b@b.T)**2)/np.sum(residual**2)))
    share=float(energies[0]/energies.sum())
    geometry=dict(layer=layer,native_centroid_rank=int(np.sum(cs>cs[0]*1e-8)),native_first_harmonic_share=share,permutation_p=(1+sum(x>=share for x in null))/257,native_pca2_variance_share=float((cs[:2]**2).sum()/(cs**2).sum()),native_pca2_radius_cv=float(np.linalg.norm(native,axis=1).std()/np.linalg.norm(native,axis=1).mean()),phase_fit_singular_values=s.tolist(),pca_fit_singular_values=ps[:5].tolist(),calibration_condition=float(np.linalg.cond(a)),centroid_circle_residual=float(np.sum((q-d@a)**2)/np.sum(q**2)),phase_pca_principal_angles_degrees=np.rad2deg(np.arccos(np.clip(np.linalg.svd(u.T@pca,compute_uv=False),0,1))).tolist(),spaces=rec,learned_analytic_relative_distance={str(delta):float(np.linalg.norm(learned[str(delta)]-analytic[str(delta)])/np.linalg.norm(analytic[str(delta)])) for delta in [-1,1]})
    return dict(mu=mu,phase2=u,pca2=pca,random2=random,centroids=cent,calibration=a,learned=learned,analytic=analytic),geometry

def self_check():
    import numpy as np
    d=np.c_[np.cos(np.arange(7)*2*np.pi/7),np.sin(np.arange(7)*2*np.pi/7)]
    assert np.allclose(d@rotation(1),d[(np.arange(7)+1)%7])
    assert np.allclose(np.linalg.matrix_power(rotation(1),7),np.eye(2))
    v,ok=norm_match(np.array([3.,4.]),np.array([0.,2.]));assert ok and np.isclose(np.linalg.norm(v),2)
    assert len(records())==70 and sum(r['split']=='test' for r in records())==42
    print('Algebra/dataset checks passed: rotation direction, composition, norm matching, fixed disjoint offsets.')

def main(args):
    import numpy as np
    cfg=json.loads(Path(args.config).read_text());run=Path(args.run);run.mkdir(parents=True,exist_ok=False)
    dump(run/'config.json',cfg);dump(run/'source.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha(__file__),config_sha256=sha(args.config),official_commit=cfg['official_commit']))
    original=md.packages_distributions
    pairs={'torch':'torch','numpy':'numpy','packaging':'packaging','safetensors':'safetensors','tokenizers':'tokenizers','huggingface_hub':'huggingface_hub','accelerate':'accelerate','requests':'requests','filelock':'filelock','regex':'regex','tqdm':'tqdm','yaml':'PyYAML','typing_extensions':'typing_extensions','transformers':'transformers','jinja2':'Jinja2','pandas':'pandas'}
    mapping={};versions={}
    for module,dist in pairs.items():
        if importlib.util.find_spec(module) is not None:
            provider=md.distribution(dist);mapping[module]=[provider.metadata['Name']];versions[module]=provider.version
    md.packages_distributions=lambda:dict(mapping)
    dump(run/'packages.json',versions)
    try:
        import torch
        from transformers import utils as hu
        from transformers.utils import import_utils as hi
        for name in ['is_torchvision_available','is_torchvision_v2_available','is_vision_available','is_librosa_available','is_torchaudio_available','is_sklearn_available']:
            for module in [hu,hi]:
                if hasattr(module,name):setattr(module,name,lambda:False)
        from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
        from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
        torch.set_num_threads(2);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
        snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        choices=[tuple(map(int,x.split(','))) for x in snapshot.splitlines()];choices=[x for x in choices if x[1]>=18*1024]
        if not choices:raise RuntimeError('No GPU has 18GiB free; no other processes touched')
        gpu=min(choices,key=lambda x:(x[2],-x[1]))[0];device='cuda:'+str(gpu);dump(run/'gpu.json',dict(gpu=gpu,snapshot=snapshot))
        torch.cuda.set_per_process_memory_fraction(.60,device)
        path=Path(cfg['model_path']);tok=PreTrainedTokenizerFast(tokenizer_file=str(path/'tokenizer.json'),eos_token='<|endoftext|>',pad_token='<|endoftext|>')
        ids=[tok.encode(' '+d,add_special_tokens=False) for d in DAYS]
        if any(len(x)!=1 for x in ids):raise ValueError('Weekday answer is not a single token')
        answer_ids=torch.tensor([x[0] for x in ids],device=device)
        rows=records();dump(run/'prompts.json',rows);dump(run/'answer_token_ids.json',dict(zip(DAYS,[x[0] for x in ids])))
        print('Loading frozen GPT-J on',device,flush=True);start=time.monotonic()
        model=GPTJForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16,attn_implementation='eager',low_cpu_mem_usage=True).to(device).eval()
        model.requires_grad_(False);loaded=time.monotonic()-start;begin=time.monotonic();print('Loaded',loaded,'seconds; forward cap',cfg['inference_seconds'],flush=True)
        layers=cfg['layers'];raw=np.empty((len(rows),len(layers),model.config.n_embd),dtype=np.float32);base=[]
        def encode(items):
            enc=tok([r['text'] for r in items],return_tensors='pt',padding=True,return_offsets_mapping=True,return_token_type_ids=False,add_special_tokens=False)
            offsets=enc.pop('offset_mapping');positions=[]
            for i,r in enumerate(items):
                at=r['text'].rindex(' from ')+6;end=at+len(DAYS[r['day']]);ix=[j for j,(a,b) in enumerate(offsets[i].tolist()) if b>at and a<end]
                if len(ix)!=1:raise ValueError('Source weekday must be exactly one token')
                positions.append(ix[0])
            enc['position_ids']=enc['attention_mask'].cumsum(-1)-1;enc['position_ids'].masked_fill_(enc['attention_mask']==0,0)
            return {k:v.to(device) for k,v in enc.items()},positions
        def limit():
            if time.monotonic()-begin>cfg['inference_seconds']:raise TimeoutError('Predefined inference wall cap')
        def output(enc):
            limit()
            with torch.inference_mode():
                logits=model(**enc,use_cache=False).logits
                last=enc['attention_mask'].sum(-1)-1
                out=logits[torch.arange(len(last),device=device),last].float()
                seven=out[:,answer_ids].cpu().numpy();top=out.argmax(-1).cpu().numpy();prob=out.softmax(-1)[:,answer_ids].cpu().numpy()
            return seven,top,prob
        for a in range(0,len(rows),7):
            batch=rows[a:a+7];enc,pos=encode(batch);handles=[]
            for li,l in enumerate(layers):
                def hook(module,inputs,result,li=li):
                    h=result[0] if isinstance(result,tuple) else result
                    raw[a:a+len(batch),li]=h[torch.arange(len(batch),device=device),pos].float().cpu().numpy()
                handles.append(model.transformer.h[l].register_forward_hook(hook))
            try:seven,top,prob=output(enc)
            finally:
                for h in handles:h.remove()
            for i,r in enumerate(batch):base.append(dict(id=r['id'],context=r['context'],split=r['split'],day=r['day'],offset=r['offset'],answer=r['answer'],candidate_prediction=int(seven[i].argmax()),global_token=int(top[i]),global_text=tok.decode([int(top[i])]),candidate_correct=bool(seven[i].argmax()==r['answer']),global_correct=bool(top[i]==ids[r['answer']][0]),weekday_logits=seven[i].tolist(),weekday_prob=prob[i].tolist()))
        np.savez_compressed(run/'raw_activations.npz',hidden=raw);dump(run/'baseline.json',base)
        lookup={(r['context'],r['offset'],r['day']):r for r in rows};fit=np.array([r['split']=='fit' for r in rows]);fitrows=[r for r in rows if r['split']=='fit'];spaces={};geoms=[]
        for li,l in enumerate(layers):
            space,geom=fit_space(raw[fit,li].astype(float),fitrows,l);spaces[l]=space;geoms.append(geom)
            np.savez_compressed(run/f'space_layer{l}.npz',**{k:v for k,v in space.items() if k not in ['learned','analytic']},**{'learned_'+k:v for k,v in space['learned'].items()},**{'analytic_'+k:v for k,v in space['analytic'].items()})
        dump(run/'geometry.json',geoms)
        pairs=[]
        for r in rows:
            if r['split']!='test':continue
            for delta in [-1,1]:
                corrupted=lookup[(r['context'],r['offset'],(r['day']-delta)%7)]
                clean_ids=tok.encode(r['text'],add_special_tokens=False);corrupt_ids=tok.encode(corrupted['text'],add_special_tokens=False)
                assert len(clean_ids)==len(corrupt_ids) and sum(a!=b for a,b in zip(clean_ids,corrupt_ids))==1
                pairs.append(dict(id=len(pairs),clean_id=r['id'],corrupt_id=corrupted['id'],delta=delta,context=r['context'],day=r['day'],offset=r['offset'],answer=r['answer'],corrupt_answer=corrupted['answer'],joint_candidate_correct=base[r['id']]['candidate_correct'] and base[corrupted['id']]['candidate_correct'],joint_global_correct=base[r['id']]['global_correct'] and base[corrupted['id']]['global_correct']))
        dump(run/'pairs.json',pairs);allout=[]
        for li,l in enumerate(layers):
            sp=spaces[l];work=[];replacements=[]
            for p in pairs:
                hc=raw[p['clean_id'],li].astype(float);hb=raw[p['corrupt_id'],li].astype(float);diff=hc-hb;phase=diff@sp['phase2']@sp['phase2'].T
                for name in METHODS:
                    matched=True
                    if name=='noop':delta=np.zeros_like(hb)
                    elif name=='full':delta=diff
                    elif name in ['phase2','pca2','random2']:delta=diff@sp[name]@sp[name].T
                    elif name.endswith('_norm'):
                        b=sp[name[:-5]];delta,matched=norm_match(diff@b@b.T,phase)
                    else:
                        trans=sp['learned' if name=='learned_transition' else 'analytic'][str(p['delta'])];q=(hb-sp['mu'])@sp['phase2'];delta=(q@trans-q)@sp['phase2'].T
                    replacement=(hb+delta).astype(np.float16);actual=replacement.astype(float)-hb
                    projected=(replacement.astype(float)-sp['mu'])@sp['phase2'];qclean=(hc-sp['mu'])@sp['phase2']
                    score=dict(pair_id=p['id'],method=name,layer=l,context=p['context'],day=p['day'],offset=p['offset'],delta=p['delta'],joint_candidate_correct=p['joint_candidate_correct'],joint_global_correct=p['joint_global_correct'],patch_norm=float(np.linalg.norm(actual)),requested_patch_norm=float(np.linalg.norm(delta)),relative_patch_norm=float(np.linalg.norm(actual)/np.linalg.norm(hb)),phase_donor_distance=float(np.linalg.norm(projected-qclean)),activation_donor_distance=float(np.linalg.norm(replacement.astype(float)-hc)),norm_match_valid=matched)
                    work.append((p,score));replacements.append(replacement)
            replacements=np.stack(replacements);np.savez_compressed(run/f'patch_vectors_layer{l}.npz',replacement=replacements)
            for a in range(0,len(work),cfg['batch']):
                group=work[a:a+cfg['batch']];items=[rows[p['corrupt_id']] for p,_ in group];enc,pos=encode(items)
                ref,ref_top,ref_prob=output(enc)
                rep=torch.from_numpy(replacements[a:a+len(group)]).to(device)
                def intervene(module,inputs,result):
                    h=result[0] if isinstance(result,tuple) else result;changed=h.clone()
                    for j,(_,r) in enumerate(group):
                        if r['method']!='noop':changed[j,pos[j]]=rep[j]
                    return (changed,)+result[1:] if isinstance(result,tuple) else changed
                handle=model.transformer.h[l].register_forward_hook(intervene)
                try:pred,top,prob=output(enc)
                finally:handle.remove()
                for j,(p,r) in enumerate(group):
                    target=p['answer'];wrong=p['corrupt_answer'];ld=float(pred[j,target]-pred[j,wrong]);before=float(ref[j,target]-ref[j,wrong]);cleanbase=base[p['clean_id']]['weekday_logits'];cleanld=cleanbase[target]-cleanbase[wrong]
                    r.update(logit_difference=ld,corrupt_logit_difference=before,clean_logit_difference=float(cleanld),logit_difference_change=ld-before,candidate_prediction=int(pred[j].argmax()),global_token=int(top[j]),global_text=tok.decode([int(top[j])]),candidate_target_correct=bool(pred[j].argmax()==target),global_target_correct=bool(top[j]==ids[target][0]),corrupt_candidate_target_correct=bool(ref[j].argmax()==target),corrupt_global_target_correct=bool(ref_top[j]==ids[target][0]),target_probability=float(prob[j,target]),max_logit_change=float(abs(pred[j]-ref[j]).max()))
                    allout.append(r)
                if a%168==0:print('Layer',l,'interventions',min(a+len(group),len(work)),'/',len(work),'elapsed',time.monotonic()-begin,flush=True)
            with open(run/'interventions.csv','w') as f:
                w=csv.DictWriter(f,fieldnames=list(allout[0]));w.writeheader();w.writerows(allout)
        elapsed=time.monotonic()-begin;summary=dict(n_prompts=len(rows),n_pairs=len(pairs),n_interventions=len(allout),load_seconds=loaded,inference_seconds=elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,noop_max_logit_change=max(r['max_logit_change'] for r in allout if r['method']=='noop'),baseline_test_candidate_accuracy=float(np.mean([b['candidate_correct'] for b in base if b['split']=='test'])),baseline_test_global_accuracy=float(np.mean([b['global_correct'] for b in base if b['split']=='test'])),joint_candidate_pairs=sum(p['joint_candidate_correct'] for p in pairs),joint_global_pairs=sum(p['joint_global_correct'] for p in pairs))
        dump(run/'completion.json',summary);print(json.dumps(summary),flush=True)
    except BaseException as e:
        dump(run/'failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise
    finally:md.packages_distributions=original

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--run');parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:self_check()
    else:main(args)
