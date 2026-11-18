"""Context teacher39: fixed history prototypes and frozen LM relation cache only."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0",CUBLAS_WORKSPACE_CONFIG=":4096:8")
import argparse,copy,time,json
from pathlib import Path
import numpy as np,torch
from exp.exp_chronogeom_event_knowledge19 import read,save,sha,git

def conditional_channels(table,state,w,h):
    k=state[:,None];q=table[k,w,h]
    return np.stack([q-table[k,1-w,h],q-table[k,w,np.full_like(h,3)],q-table[k,w,np.full_like(h,12)]],-1)
def observed_ratio(x,origin,scale,c):
    raw=np.maximum(x[:,:,0].astype(float)*scale["sd"]+scale["mean"],0)
    hours=(origin[:,None]+np.arange(-96,0)[None,:])%24
    night=np.isin(hours,c["night_hours"]);day=np.isin(hours,c["day_hours"])
    ratio=(raw*night).sum(1)/night.sum(1)/np.maximum((raw*day).sum(1)/day.sum(1),c["raw_ratio_floor"])
    assert np.isfinite(ratio).all()
    return raw,ratio

def prepare(c,config):
    from transformers import AutoTokenizer
    r=Path(c["root"]);r.mkdir(parents=True,exist_ok=False);src=Path(c["source"])
    save(r/"config.json",c);sel=read(src/"selection_buildings.json");save(r/"selection_buildings.json",sel);contexts=[];state_records=[]
    for meta in sel["selected"]:
        name=meta["building"];d=src/name;out=r/name;out.mkdir();b=np.load(d/"fit_inputs.npz");scale=read(d/"scale.json")
        fx,fr=observed_ratio(b["fit_x"],b["fit_origin"],scale,c);vx,vr=observed_ratio(b["val_x"],b["val_origin"],scale,c)
        assert b["fit_future"].max()<8472 and b["val_future"].max()<8472
        cuts=np.quantile(fr,[1/3,2/3]);basefit=np.searchsorted(cuts,fr,side="right");baseval=np.searchsorted(cuts,vr,side="right");active=np.unique(basefit)
        fit=np.argmin(abs(basefit[:,None]-active[None,:]),axis=1);val=np.argmin(abs(baseval[:,None]-active[None,:]),axis=1)
        protos=[]
        for k in range(len(active)):
            ix=np.flatnonzero(fit==k);median=float(np.median(fr[ix]));p=int(ix[np.argmin(abs(fr[ix]-median))]);origin=int(b["fit_origin"][p]);z=fx[p]/max(float(fx[p].mean()),1e-6)
            days=[];start=origin-96
            for day in range(4):
                wd=int(((start+day*24)//24+4)%7);days.append(dict(weekday=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][wd],values=[float(round(t,3)) for t in z[day*24:(day+1)*24]]))
            proto=dict(state=k,original_bin=int(active[k]),fit_index=p,origin=origin,ratio=float(fr[p]),days=days,normalization="observed96h divided by its own mean")
            protos.append(proto)
            history="Past 96 hours in chronological order; each row contains 24 relative values for recorded local hours 00 through 23. Night/day observed mean ratio="+format(fr[p],".3f")+".\n"
            history+="\n".join(x["weekday"]+": "+",".join(format(v,".3f") for v in x["values"]) for x in days)
            domains={"Generic":c["response_contexts"]["Generic"],"Task":c["response_contexts"][meta["category"]],"Wrong":c["response_contexts"][c["wrong_context"][meta["category"]]]}
            for label,domain in domains.items():contexts.append(dict(building=name,state=k,context=label,history=history,domain=domain))
        np.savez_compressed(out/"states.npz",fit=fit,val=val,fit_ratio=fr,val_ratio=vr,thresholds=cuts)
        record=dict(building=name,category=meta["category"],thresholds=cuts.tolist(),original_active_bins=active.tolist(),K=len(active),fit_counts=np.bincount(fit,minlength=len(active)).tolist(),val_counts=np.bincount(val,minlength=len(active)).tolist(),prototypes=protos,history_only_selection=True)
        save(out/"states.json",record);state_records.append(record)
    tok=AutoTokenizer.from_pretrained(c["model"],local_files_only=True);tok.padding_side="left"
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    ids=[tok.encode(w,add_special_tokens=False) for w in ["Higher","Similar","Lower"]];assert all(len(x)==1 for x in ids)
    groups=[];requests=[];maxlen=0;total=0;prefixwork=0;suffixwork=0
    for group in contexts:
        texts=[];fields=[];pref=c["prompt_prefix"].format(DOMAIN=group["domain"],HISTORY=group["history"])
        for day in [0,1]:
            dtype="ordinary non-holiday weekday" if day==0 else "ordinary non-holiday weekend day";other="ordinary non-holiday weekend day" if day==0 else "ordinary non-holiday weekday"
            for hour in range(24):
                a=f"{dtype} at {hour:02d}:00"
                for rel,bb in enumerate([f"{other} at {hour:02d}:00",f"{dtype} at 03:00",f"{dtype} at 12:00"]):
                    prompt=pref+c["prompt_suffix"].format(A=a,B=bb)
                    chat=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    tokens=tok.encode(chat,add_special_tokens=False);maxlen=max(maxlen,len(tokens));total+=len(tokens);texts.append(tokens);fields.append(dict(daytype=day,hour=hour,relation=rel,prompt_sha=__import__("hashlib").sha256(prompt.encode()).hexdigest()))
        n=0
        while n<min(map(len,texts)) and len({x[n] for x in texts})==1:n+=1
        assert n>0
        prefixwork+=n;gi=len(groups);indices=[]
        for tokens,field in zip(texts,fields):
            assert tokens[:n]==texts[0][:n]
            indices.append(len(requests));suffixwork+=len(tokens)-n
            requests.append(dict(group=gi,**field,suffix_ids=tokens[n:]))
        groups.append(dict(**group,prefix_ids=texts[0][:n],request_indices=indices))
    assert len(requests)<=7776 and maxlen<=c["max_prompt_tokens"]
    save(r/"groups.json",groups);save(r/"requests.json",requests);save(r/"state_summary.json",state_records)
    save(r/"prepared.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),queries=len(requests),groups=len(groups),maximum_tokens=maxlen,full_prefill_token_count=total,prefix_reuse_token_count=prefixwork+suffixwork,word_token_ids=[x[0] for x in ids],pad_token_id=tok.pad_token_id,new2017_access=False))
    print("PREPARED",read(r/"prepared.json"),flush=True)

def extract(c):
    from transformers import AutoModelForCausalLM
    start=time.monotonic();r=Path(c["root"]);prep=read(r/"prepared.json");groups=read(r/"groups.json");requests=read(r/"requests.json");device="cuda:0"
    assert torch.cuda.mem_get_info()[0]>=22*2**30
    torch.set_num_threads(2);torch.manual_seed(2024);torch.cuda.set_per_process_memory_fraction(c["LLM_memory_gib"]*2**30/torch.cuda.get_device_properties(device).total_memory)
    model=AutoModelForCausalLM.from_pretrained(c["model"],local_files_only=True,dtype=torch.bfloat16,device_map={"":device},attn_implementation="sdpa").eval()
    for p in model.parameters():p.requires_grad_(False)
    load_seconds=time.monotonic()-start;logits=np.zeros((len(requests),3));logpart=np.zeros(len(requests));probed=False;probe=None;batchcalls=0
    def packed(seqs):
        L=max(map(len,seqs));x=torch.full((len(seqs),L),prep["pad_token_id"],dtype=torch.long,device=device);m=torch.zeros_like(x)
        for i,seq in enumerate(seqs):x[i,-len(seq):]=torch.tensor(seq,device=device);m[i,-len(seq):]=1
        return x,m
    with torch.inference_mode():
        for gi,g in enumerate(groups):
            pids=torch.tensor([g["prefix_ids"]],device=device);plen=pids.shape[1]
            base=model(input_ids=pids,attention_mask=torch.ones_like(pids),position_ids=torch.arange(plen,device=device)[None,:],use_cache=True,logits_to_keep=1).past_key_values
            assert base.get_seq_length()==plen
            for j in range(0,len(g["request_indices"]),c["extract_batch"]):
                ix=g["request_indices"][j:j+c["extract_batch"]];suffix=[requests[t]["suffix_ids"] for t in ix];x,mask=packed(suffix)
                att=torch.cat([torch.ones((len(ix),plen),dtype=mask.dtype,device=device),mask],1);pos=(att.cumsum(1)-1)[:,-x.shape[1]:].masked_fill(mask==0,0)
                kv=copy.deepcopy(base);kv.batch_repeat_interleave(len(ix))
                full=model(input_ids=x,attention_mask=att,position_ids=pos,past_key_values=kv,use_cache=True,logits_to_keep=1).logits[:,-1].float()
                selected=full[:,prep["word_token_ids"]];part=torch.logsumexp(full,-1)
                if not probed:
                    xx,mm=packed([g["prefix_ids"]+s for s in suffix]);pp=(mm.cumsum(1)-1).masked_fill(mm==0,0)
                    standard=model(input_ids=xx,attention_mask=mm,position_ids=pp,use_cache=False,logits_to_keep=1).logits[:,-1].float()[:,prep["word_token_ids"]]
                    dl=float(abs(standard-selected).max());dp=float(abs(standard.softmax(-1)-selected.softmax(-1)).max());probe=dict(logit_max_difference=dl,probability_max_difference=dp,queries=len(ix),cache_logits=selected.cpu().tolist(),full_logits=standard.cpu().tolist())
                    save(r/"prefix_probe.json",probe);assert dl<=c["prefix_logit_tolerance"] and dp<=c["prefix_probability_tolerance"],"Prefix/full numerical check failed; preserve outputs"
                    probed=True;print("PREFIX_CHECK",dl,dp,flush=True)
                logits[ix]=selected.cpu().numpy();logpart[ix]=part.cpu().numpy();batchcalls+=1
                del kv,full
                assert base.get_seq_length()==plen
                assert time.monotonic()-start<c["extract_seconds"],"Frozen extraction activity cap reached"
            np.savez_compressed(r/"partial_response_cache.npz",logits=logits,full_vocab_log_partition=logpart,completed_groups=gi+1)
            save(r/"extract_progress.json",dict(groups=gi+1,total_groups=len(groups),seconds=time.monotonic()-start))
            print("CONTEXT_GROUP",gi+1,len(groups),g["building"],g["state"],g["context"],flush=True)
            del base
    xx=np.exp(logits-logits.max(1,keepdims=True));prob=xx/xx.sum(1,keepdims=True);response=prob[:,0]-prob[:,2];allowed=np.exp(np.log(xx.sum(1))+logits.max(1)-logpart).clip(0,1)
    banks={}
    for g in groups:
        b=g["building"];banks.setdefault(b,{})
        z=response[g["request_indices"]].reshape(2,24,3)
        banks[b].setdefault(g["context"],{})[g["state"]]=z
    features={}
    for b,bs in banks.items():
        arrays={label:np.stack([d[k] for k in sorted(d)]) for label,d in bs.items()}
        features[b]={"HistoryQwen":(arrays["Task"]-arrays["Generic"]).tolist(),"WrongContext":(arrays["Wrong"]-arrays["Generic"]).tolist()}
        features[b]["HistoryShuffle"]=np.roll(np.array(features[b]["HistoryQwen"]),-1,axis=0).tolist()
    save(r/"context_features.json",features)
    save(r/"raw_context_responses.json",{b:{label:{k:v.tolist() for k,v in table.items()} for label,table in bs.items()} for b,bs in banks.items()})
    np.savez_compressed(r/"response_cache.npz",logits=logits,probabilities=prob,full_vocab_log_partition=logpart,valid_answer_mass=allowed,response=response)
    self_ix=[i for i,q in enumerate(requests) if (q["relation"]==1 and q["hour"]==3) or (q["relation"]==2 and q["hour"]==12)]
    save(r/"extraction.json",dict(complete=True,seconds=time.monotonic()-start,load_seconds=load_seconds,queries=len(requests),prefix_forwards=len(groups),suffix_batch_forwards=batchcalls,numerical_probe=probe,allowed_mass_min=float(allowed.min()),allowed_mass_mean=float(allowed.mean()),self_similar_fraction=float((prob[self_ix].argmax(1)==1).mean()),no_question_filtering=True,peak_gib=torch.cuda.max_memory_allocated()/2**30,trainable_parameters=0,model_revision=c["model_revision"],new2017_access=False))
    print("EXTRACTION_COMPLETE",read(r/"extraction.json"),flush=True)


def extract_full_shard(c,shard):
    from transformers import AutoModelForCausalLM
    start=time.monotonic();r=Path(c["root"]);prep=read(r/"prepared.json");groups=read(r/"groups.json");requests=read(r/"requests.json");device="cuda:0"
    assert torch.cuda.mem_get_info()[0]>=22*2**30
    torch.set_num_threads(2);torch.manual_seed(2024);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(c["LLM_memory_gib"]*2**30/torch.cuda.get_device_properties(device).total_memory)
    model=AutoModelForCausalLM.from_pretrained(c["model"],local_files_only=True,dtype=torch.bfloat16,device_map={"":device},attn_implementation="sdpa").eval()
    for p in model.parameters():p.requires_grad_(False)
    load=time.monotonic()-start;L=np.zeros((len(requests),3));Z=np.zeros(len(requests));done=np.zeros(len(requests),bool);batchcalls=0;batchsize=c["extract_batch"]
    def infer(indices):
        seqs=[groups[requests[t]["group"]]["prefix_ids"]+requests[t]["suffix_ids"] for t in indices];length=max(map(len,seqs));x=torch.full((len(seqs),length),prep["pad_token_id"],dtype=torch.long,device=device);m=torch.zeros_like(x)
        for i,seq in enumerate(seqs):x[i,-len(seq):]=torch.tensor(seq,device=device);m[i,-len(seq):]=1
        pos=(m.cumsum(1)-1).masked_fill(m==0,0)
        with torch.inference_mode():full=model(input_ids=x,attention_mask=m,position_ids=pos,use_cache=False,logits_to_keep=1).logits[:,-1].float()
        return full[:,prep["word_token_ids"]].cpu().numpy(),torch.logsumexp(full,-1).cpu().numpy()
    probe,_=infer(list(range(8)))
    for gi,g in enumerate(groups):
        if gi%c["extract_shards"]!=shard:continue
        j=0;ids=g["request_indices"]
        while j<len(ids):
            ix=ids[j:j+batchsize]
            try:q,z=infer(ix)
            except torch.cuda.OutOfMemoryError:
                if batchsize<=8:raise
                batchsize//=2;torch.cuda.empty_cache();print("INFERENCE_BATCH_REDUCED_FOR_MEMORY",batchsize,flush=True);continue
            L[ix]=q;Z[ix]=z;done[ix]=True;j+=len(ix);batchcalls+=1
            assert time.monotonic()-start<c["extract_seconds"],"Standard full-prefill shard activity cap reached"
        np.savez_compressed(r/f"full_shard{shard}_response.npz",logits=L,logpart=Z,done=done,probe=probe)
        save(r/f"full_shard{shard}_progress.json",dict(completed_queries=int(done.sum()),seconds=time.monotonic()-start,batch=batchsize,group=gi,total_queries=len(requests)))
        print("FULL_CONTEXT_GROUP",shard,gi,g["building"],g["state"],g["context"],int(done.sum()),round(time.monotonic()-start,2),flush=True)
    save(r/f"full_shard{shard}_status.json",dict(complete=True,seconds=time.monotonic()-start,load_seconds=load,queries=int(done.sum()),batch_forwards=batchcalls,actual_batch=batchsize,peak_gib=torch.cuda.max_memory_allocated()/2**30,model_revision=c["model_revision"],trainable_parameters=0))
    print("FULL_SHARD_COMPLETE",shard,flush=True)

def finalize_full(c):
    r=Path(c["root"]);groups=read(r/"groups.json");requests=read(r/"requests.json");N=len(requests);logits=np.zeros((N,3));logpart=np.zeros(N);count=np.zeros(N,int);stats=[];probes=[]
    for shard in range(c["extract_shards"]):
        st=read(r/f"full_shard{shard}_status.json");assert st["complete"];stats.append(st);a=np.load(r/f"full_shard{shard}_response.npz");mask=a["done"];logits[mask]=a["logits"][mask];logpart[mask]=a["logpart"][mask];count+=mask.astype(int);probes.append(a["probe"])
    assert np.all(count==1)
    probe_error=max(float(np.max(abs(x-probes[0]))) for x in probes);save(r/"cross_device_probe.json",dict(max_word_logit_difference=probe_error,queries=8));assert probe_error<1e-6,"Cross-device same-full-forward mismatch"
    xx=np.exp(logits-logits.max(1,keepdims=True));prob=xx/xx.sum(1,keepdims=True);response=prob[:,0]-prob[:,2];allowed=np.exp(np.log(xx.sum(1))+logits.max(1)-logpart).clip(0,1);banks={}
    for g in groups:
        b=g["building"];banks.setdefault(b,{})
        banks[b].setdefault(g["context"],{})[g["state"]]=response[g["request_indices"]].reshape(2,24,3)
    features={}
    for b,bs in banks.items():
        arrays={label:np.stack([d[k] for k in sorted(d)]) for label,d in bs.items()};features[b]={"HistoryQwen":(arrays["Task"]-arrays["Generic"]).tolist(),"WrongContext":(arrays["Wrong"]-arrays["Generic"]).tolist()}
        features[b]["HistoryShuffle"]=np.roll(np.array(features[b]["HistoryQwen"]),-1,axis=0).tolist()
    save(r/"context_features.json",features);save(r/"raw_context_responses.json",{b:{label:{k:v.tolist() for k,v in table.items()} for label,table in bs.items()} for b,bs in banks.items()})
    np.savez_compressed(r/"response_cache.npz",logits=logits,probabilities=prob,full_vocab_log_partition=logpart,valid_answer_mass=allowed,response=response)
    selfix=[i for i,q in enumerate(requests) if (q["relation"]==1 and q["hour"]==3) or (q["relation"]==2 and q["hour"]==12)]
    save(r/"extraction.json",dict(complete=True,seconds=max(x["seconds"] for x in stats),aggregate_gpu_seconds=sum(x["seconds"] for x in stats),queries=N,extra_common_probe_queries=8*len(stats),standard_full_prefill=True,prefix_reuse=False,shards=stats,cross_device_probe_error=probe_error,valid_answer_mass_min=float(allowed.min()),valid_answer_mass_mean=float(allowed.mean()),self_similar_fraction=float((prob[selfix].argmax(1)==1).mean()),all_queries_retained=True,trainable_parameters=0,model_revision=c["model_revision"],new2017_access=False))
    print("FULL_EXTRACTION_COMPLETE",read(r/"extraction.json"),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--stage",choices=["prepare","extract","full","finalize"],required=True);p.add_argument("--shard",type=int,default=0);a=p.parse_args();c=read(a.config)
    if a.stage=="prepare":prepare(c,a.config)
    elif a.stage=="extract":extract(c)
    elif a.stage=="full":extract_full_shard(c,a.shard)
    else:finalize_full(c)
