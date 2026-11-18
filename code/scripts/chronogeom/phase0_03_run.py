"""Bounded pretrained GPT-J specificity/ablation experiment, offline and non-training."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'2')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import argparse,json,time,sys,hashlib,subprocess,csv
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.chronogeom.phase0_02_patching import block_input_hook,canonical_integer,same_token_layout
from scripts.chronogeom.phase0_03_geometry import fit_spaces,make_change,subspace_error
from scripts.chronogeom.representation_probe import basis

SITES=('a','b','last')

def additional_pairs(values,n,seed,mode='uniform'):
    rng=np.random.default_rng(seed);seen=set();out=[]
    while len(out)<n:
        a,c=map(int,rng.choice(values,2,replace=False));i=len(out)
        b=int(rng.integers((i%4)*25,(i%4+1)*25)) if mode=='b_stratified' else int(rng.integers(100))
        if (a,c,b) in seen:continue
        seen.add((a,c,b));out.append(dict(id=i,clean_a=a,corrupt_a=c,b=b))
    return out

def bf16_match(base,ideal,target,iterations=4):
    delta=ideal-base;norm=np.linalg.norm(delta,axis=1);zero=norm<=1e-12
    factor=np.divide(target,norm,out=np.zeros_like(norm),where=~zero);delta*=factor[:,None]
    for _ in range(iterations):
        applied=torch.from_numpy(base+delta).bfloat16().float().numpy();actual=np.linalg.norm(applied-base,axis=1)
        factor=np.divide(target,actual,out=np.ones_like(actual),where=actual>1e-12);delta*=factor[:,None]
    return base+delta,zero

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(args.config.read_text());out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    source=Path(cfg['source_run']);old_pairs=json.loads((source/'candidate_pairs.json').read_text());old_screen=json.loads((source/'pair_screening.json').read_text());old_eligible={p['id'] for p in old_screen if p['eligible']}
    fit=json.loads((source/'fit_tokenization.json').read_text());fit_values=set(map(int,cfg['fit_values']));test_values=list(map(int,cfg['eval_values']));restricted=[x for x in fit_values if x<80];lo,hi=min(restricted),max(restricted)
    cohorts={'fixed':old_pairs,'train_ids':additional_pairs(sorted(fit_values),32,45),'range_in':additional_pairs([x for x in test_values if lo<=x<=hi],32,46),'range_out':additional_pairs([x for x in test_values if x>=80],32,47),'seed43_uniform':additional_pairs(test_values,32,43),'seed44_b_stratified':additional_pairs(test_values,32,44,'b_stratified')}
    (out/'candidate_manifest.json').write_text(json.dumps(cohorts,indent=2)+'\n')
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True);tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(cfg['model_path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to(cfg['device']).eval();layers=cfg['layers'];device=cfg['device']
    def encode(a,b):
        a,b=canonical_integer(a),canonical_integer(b);text=f'Output ONLY a number.\n{a}+{b}=';begin=text.index('\n')+1;second=begin+len(str(a))+1
        r=tok(text,return_offsets_mapping=True,add_special_tokens=False);off=r['offset_mapping'];pa=[i for i,(s,e) in enumerate(off) if e>begin and s<begin+len(str(a))];pb=[i for i,(s,e) in enumerate(off) if e>second and s<second+len(str(b))];ans=tok(str(a+b),add_special_tokens=False)['input_ids']
        return dict(a=a,b=b,text=text,ids=r['input_ids'],pos=pa[0] if len(pa)==1 else None,positions=[pa[0],pb[0],len(off)-1] if len(pa)==len(pb)==1 else None,answer_ids=ans,valid=len(pa)==len(pb)==len(ans)==1)
    fit=[encode(r['a'],r['b']) for r in fit];assert len(fit)==316 and all(r['valid'] for r in fit)
    problems={}
    for pairs in cohorts.values():
        for p in pairs:
            for role in ('clean','corrupt'):
                key=f"{p[role+'_a']}+{p['b']}";p[role+'_key']=key
                if key not in problems:problems[key]=encode(p[role+'_a'],p['b'])
            p['layout_valid']=same_token_layout(problems[p['clean_key']],problems[p['corrupt_key']])
    def budget():
        if time.monotonic()-start>cfg['gpu_wall_limit']-15:raise TimeoutError('predeclared600s wall budget; partial rows retained')
    def forward(records, capture=False,layer=None,site=0,exchange=None,which_layers=None):
        n=max(len(r['ids']) for r in records);ids=torch.full((len(records),n),tok.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids);positions=[];handles=[];caps={};use_layers=layers if which_layers is None else which_layers
        for i,r in enumerate(records):
            offset=n-len(r['ids']);ids[i,offset:]=torch.tensor(r['ids'],device=device);mask[i,offset:]=1;positions.append([offset+p for p in r['positions']])
        def capture_hook(container):
            def hook(module,a,kw):
                h=kw.get('hidden_states',a[0] if a else None);ix=torch.arange(len(h),device=h.device)[:,None];pos=torch.tensor(positions,device=h.device);container.append(h[ix,pos].detach().float().cpu())
            return hook
        try:
            if capture:
                for l in use_layers:
                    caps[l]=[];handles.append(model.transformer.h[l].register_forward_pre_hook(capture_hook(caps[l]),with_kwargs=True))
            if layer is not None:handles.append(model.transformer.h[layer].register_forward_pre_hook(block_input_hook([p[site] for p in positions],replacement=None if exchange is None else torch.from_numpy(exchange)),with_kwargs=True))
            with torch.inference_mode():logits=model(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(1)-1).clamp(min=0),use_cache=False).logits[:,-1].float().cpu()
        finally:
            for h in handles:h.remove()
        hidden=torch.stack([torch.cat(caps[l]) for l in use_layers],dim=1).numpy() if capture else None
        return logits,hidden
    def collect(records):
        logs=[];hs=[]
        for i in range(0,len(records),8):
            budget();log,h=forward(records[i:i+8],True);logs.append(log);hs.append(h)
        return torch.cat(logs),np.concatenate(hs)
    _,fit_h=collect(fit);a_fit=np.array([r['a'] for r in fit]);fit_mask=a_fit<80;space={};inf={};space_rows=[]
    for li,l in enumerate(layers):
        for si,site in enumerate(SITES):
            for seed in [42,43,44]:
                if seed!=42 and (l not in cfg['rank_sensitivity_layers'] or site!='a'):continue
                sp,ii=fit_spaces(a_fit,fit_h[:,li,si],seed+l*101+si*17);space[l,site,seed,'primary']=sp;inf[l,site,seed,'primary']=ii
            sp,ii=fit_spaces(a_fit[fit_mask],fit_h[fit_mask,li,si],42+l*101+si*17);space[l,site,42,'range']=sp;inf[l,site,42,'range']=ii
    print('fit316 collected; spaces ready',flush=True)
    global_state={};eligible={};screen_rows=[]
    # Preserve legacy baseline order/batch boundaries for the fixed cohort.
    seen=set()
    for cohort,pairs in cohorts.items():
        keys=[]
        for p in pairs:
            for role in ('clean','corrupt'):
                k=p[role+'_key']
                if k not in seen and problems[k]['valid']:keys.append(k);seen.add(k)
        if keys:
            logs,hidden=collect([problems[k] for k in keys])
            for i,k in enumerate(keys):global_state[k]=(logs[i],hidden[i]);rr=problems[k];pred=int(logs[i].argmax());screen_rows.append(dict(cohort_first_seen=cohort,key=k,a=rr['a'],b=rr['b'],correct=pred==rr['answer_ids'][0],prediction=tok.decode([pred]),tokenization_valid=True))
        selected=[]
        for p in pairs:
            current=p['layout_valid'] and all(k in global_state and int(global_state[k][0].argmax())==problems[k]['answer_ids'][0] for k in [p['clean_key'],p['corrupt_key']]);p['current_joint_correct']=bool(current)
            p['selected']=(p['id'] in old_eligible) if cohort=='fixed' else bool(current)
            p['reason']='legacy_frozen_eligible' if cohort=='fixed' and p['selected'] else ('joint_correct' if current else 'one_or_both_incorrect_or_invalid')
            if p['selected']:selected.append(p)
        eligible[cohort]=selected;print('cohort',cohort,'selected',len(selected),'of',len(pairs),flush=True)
    (out/'screening.json').write_text(json.dumps({'cohorts':cohorts,'unique_prompts':screen_rows,'fixed_selection_unchanged':len(eligible['fixed'])==85,'fixed_baseline_drift':sum(not p['current_joint_correct'] for p in eligible['fixed']),'range_fit_values':sorted(set(a_fit[fit_mask].tolist())),'restricted_observed_bounds':[lo,hi]},indent=2)+'\n')
    np.savez(out/'fit_activations.npz',a=a_fit,hidden=fit_h)
    paired={}
    for cohort,pairs in eligible.items():
        if not pairs:continue
        clean=[problems[p['clean_key']] for p in pairs];corrupt=[problems[p['corrupt_key']] for p in pairs];cl,ch=collect(clean);rl,rh=collect(corrupt);paired[cohort]=(cl,ch,rl,rh)
        np.savez(out/f'{cohort}_paired_hidden.npz',clean=ch,corrupt=rh)
    unsupported=[];record_count=0;writer=(out/'interventions.jsonl').open('x')
    def execute(cohort,l,site,method,rank,operation,mode='raw',seed=42,profile='primary'):
        nonlocal record_count
        pairs=eligible[cohort]
        if not pairs:return
        li=layers.index(l);si=SITES.index(site);sp=space[l,site,seed,profile];ii=inf[l,site,seed,profile];full=method=='full';noop=method=='noop'
        if not full and not noop and len(sp[method])<rank:
            unsupported.append(dict(cohort=cohort,layer=l,site=site,method=method,requested_rank=rank,native_rank=len(sp[method]),operation=operation,mode=mode));return
        cl,ch,rl,rh=paired[cohort];is_ablate=operation!='swap';base_all=ch[:,li,si] if is_ablate else rh[:,li,si];donor=ch[:,li,si];baseline=cl if is_ablate else rl;records=[problems[p['clean_key'] if is_ablate else p['corrupt_key']] for p in pairs]
        if noop:ideal=base_all.copy();zero=np.zeros(len(pairs),bool)
        elif full:ideal,zero=make_change(base_all,donor,None,'full_'+('swap' if operation=='swap' else mode),mean=ii['helix']['mean'])
        else:ideal,zero=make_change(base_all,donor,sp[method][:rank],operation,mean=ii[method]['mean'])
        unscaled_norm=np.linalg.norm(ideal-base_all,axis=1)
        requested=None
        if mode=='equal_helix':
            hu=sp['helix'][:rank];href,_=make_change(base_all,donor,hu,operation,mean=ii['helix']['mean']);ha=torch.from_numpy(href).bfloat16().float().numpy();requested=np.linalg.norm(ha-base_all,axis=1)
            if method=='helix':ideal=href
            else:ideal,zero=bf16_match(base_all,ideal,requested)
        applied=torch.from_numpy(ideal).bfloat16().float().numpy();patch_norm=np.linalg.norm(applied-base_all,axis=1)
        train_y=fit_h[fit_mask,li,si] if profile=='range' else fit_h[:,li,si]
        recon_train=None if full or noop else subspace_error(train_y,ii[method]['mean'],sp[method][:rank]);recon_test=None if full or noop else subspace_error(ch[:,li,si],ii[method]['mean'],sp[method][:rank])
        numeric_mse=None
        if not full and not noop and ii[method]['coef'] is not None:
            kind='helix' if method=='shuffled' else method
            numeric=basis([p['clean_a'] for p in pairs],kind,degree=8)@ii[method]['coef']
            numeric=ii[method]['mean']+((numeric-ii[method]['mean'])@sp[method][:rank].T)@sp[method][:rank]
            numeric_mse=float(np.mean((numeric-ch[:,li,si])**2))
        if cohort=='fixed' and seed==42:
            space_rows.append(dict(layer=l,site=site,method=method,rank=rank,profile=profile,native_rank=None if full or noop else len(sp[method]),train_mse=recon_train,test_mse=recon_test,design_rank=None if full or noop else ii[method]['design_rank']))
        for start_i in range(0,len(pairs),8):
            budget();batch=records[start_i:start_i+8];logits,_=forward(batch,layer=l,site=si,exchange=None if noop else applied[start_i:start_i+8])
            for j in range(len(batch)):
                i=start_i+j;p=pairs[i];ct=problems[p['clean_key']]['answer_ids'][0];rt=problems[p['corrupt_key']]['answer_ids'][0];b=baseline[i];v=logits[j];pred=int(v.argmax());before=float(b[ct]-b[rt]);after=float(v[ct]-v[rt]);gain=after-before
                row=dict(cohort=cohort,pair_id=p['id'],clean_a=p['clean_a'],corrupt_a=p['corrupt_a'],b=p['b'],layer=l,site=site,method=method,rank=rank,operation=operation,mode=mode,control_seed=seed,fit_profile=profile,baseline_ld=before,patched_ld=after,ld_change=gain,effect=-gain if is_ablate else gain,clean_target_logit_change=float(v[ct]-b[ct]),baseline_correct=int(b.argmax())==(ct if is_ablate else rt),target_correct=pred==ct,predicted_text=tok.decode([pred]),baseline_text=tok.decode([int(b.argmax())]),prediction_changed=pred!=int(b.argmax()),max_logit_change=float(torch.max(torch.abs(v-b))),patch_l2=float(patch_norm[i]),unscaled_ideal_l2=float(unscaled_norm[i]),ideal_l2=float(np.linalg.norm(ideal[i]-base_all[i])),rounding_l2=float(np.linalg.norm(applied[i]-ideal[i])),requested_l2=None if requested is None else float(requested[i]),relative_norm_error=None if requested is None else float(abs(patch_norm[i]-requested[i])/max(requested[i],1e-12)),zero_direction=bool(zero[i]),train_reconstruction_mse=recon_train,test_reconstruction_mse=recon_test,number_to_hidden_mse=numeric_mse,carry_clean=(p['clean_a']%10+p['b']%10)>=10,carry_corrupt=(p['corrupt_a']%10+p['b']%10)>=10)
                writer.write(json.dumps(row)+'\n');record_count+=1
            writer.flush()
        print('done',cohort,l,site,operation,method,rank,mode,seed,profile,flush=True)
    # Main all-six-layer rank7 comparisons; centered coordinate replacement is primary ablation.
    comparable=['helix','pca','fourier','polynomial','random','shuffled']
    for l in layers:
        for method in comparable:
            execute('fixed',l,'a',method,7,'swap');execute('fixed',l,'a',method,7,'swap','equal_helix');execute('fixed',l,'a',method,7,'ablate_mean')
        for method in ['pca','random']:execute('fixed',l,'a',method,7,'ablate_mean','equal_helix')
        execute('fixed',l,'a','linear',1,'ablate_mean')
        execute('fixed',l,'a','full',4096,'swap');execute('fixed',l,'a','noop',0,'swap');execute('fixed',l,'a','noop',0,'ablate_mean')
        execute('fixed',l,'a','full',4096,'ablate_mean','mean');execute('fixed',l,'a','full',4096,'ablate_mean','zero')
    # Rank sensitivity: all methods truly rank1, and natural rank8 methods, at two predefined indices.
    for l in cfg['rank_sensitivity_layers']:
        for method in comparable+['linear']:execute('fixed',l,'a',method,1,'swap')
        for method in ['helix','pca','polynomial','random','shuffled']:
            execute('fixed',l,'a',method,8,'swap');execute('fixed',l,'a',method,8,'ablate_mean')
    # All six locations in depth, not only high-effect sites, at contextual b and last-token positions.
    for site in ['b','last']:
        for l in layers:
            for method in ['helix','pca']:
                execute('fixed',l,site,method,7,'swap');execute('fixed',l,site,method,7,'ablate_mean')
            execute('fixed',l,site,'random',7,'swap','equal_helix');execute('fixed',l,site,'random',7,'ablate_mean','equal_helix')
            execute('fixed',l,site,'full',4096,'swap');execute('fixed',l,site,'noop',0,'swap')
    for cohort in [c for c in cohorts if c!='fixed']:
        profile='range' if cohort.startswith('range_') else 'primary'
        for l in cfg['rank_sensitivity_layers']:
            for method in ['helix','pca','polynomial']:
                execute(cohort,l,'a',method,7,'swap',profile=profile);execute(cohort,l,'a',method,7,'ablate_mean',profile=profile)
            execute(cohort,l,'a','full',4096,'swap',profile=profile);execute(cohort,l,'a','noop',0,'swap',profile=profile)
    for seed in [43,44]:
        for l in cfg['rank_sensitivity_layers']:
            for method in ['random','shuffled']:
                execute('fixed',l,'a',method,7,'swap','equal_helix',seed=seed);execute('fixed',l,'a',method,7,'ablate_mean',seed=seed)
    writer.close();(out/'unsupported.json').write_text(json.dumps(unsupported,indent=2)+'\n')
    (out/'subspaces.json').write_text(json.dumps(space_rows,indent=2)+'\n')
    meta=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ['candidate_pairs.json','pair_screening.json','fit_tokenization.json']},code_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['phase0_03_run.py','phase0_03_geometry.py','phase0_03_analysis.py','phase0_02_patching.py','representation_probe.py']},main_seconds=time.monotonic()-start,rows=record_count,actual_dtype=str(next(model.parameters()).dtype),gpu_peak_allocated=torch.cuda.max_memory_allocated(device),gpu_peak_reserved=torch.cuda.max_memory_reserved(device),python=sys.version,torch=torch.__version__,fixed_pairs=len(eligible['fixed']),fit_rows=len(fit),cache_bytes=sum(p.stat().st_size for p in out.glob('*.npz')))
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    subprocess.run([sys.executable,'scripts/chronogeom/phase0_03_analysis.py','--run',str(out),'--output',str(out/'analysis')],check=True)
    if not json.loads((out/'analysis/quality.json').read_text())['execution_gate']:raise RuntimeError('primary quality gates failed; optional FFT not run')
    if time.monotonic()-start<cfg['optional_fft_if_elapsed_below']:
        standalone=[]
        for a in range(361):
            ids=tok(str(a),add_special_tokens=False)['input_ids'];standalone.append(dict(a=a,b=0,ids=ids,positions=[len(ids)-1]*3))
        logs=[];hs=[]
        for i in range(0,361,8):
            budget();_,h=forward(standalone[i:i+8],capture=True,which_layers=[0,1]);hs.append(h[:,:,0])
        np.savez(out/'standalone_0_360.npz',values=np.arange(361),hidden=np.concatenate(hs),token_lengths=np.array([len(r['ids']) for r in standalone]))
        print('optional standalone361 done',flush=True)
    else:(out/'fft_skipped.json').write_text(json.dumps({'reason':'predeclared wall-time margin','elapsed':time.monotonic()-start})+'\n')
    meta['total_seconds']=time.monotonic()-start;(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print('COMPLETE',record_count,'rows',meta['total_seconds'],'seconds',flush=True)
if __name__=='__main__':main()
