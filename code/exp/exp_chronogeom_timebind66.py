"""TimeBind66: bounded structured temporal-understanding pilot. No forecasting or LLM training."""
import os
os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                 PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
import argparse, hashlib, json, math, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

ARMS = ["TimeBindQwen", "PatchConcatQwen", "TimeBindFourier", "TimeBindShuffled"]
OPS = ["mean_order", "peak_order", "change_order"]
PERIODS = ["overnight (00:00-05:59)", "morning (06:00-11:59)",
           "afternoon (12:00-17:59)", "evening (18:00-23:59)"]
def save(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False))
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(2**20),b""): h.update(b)
    return h.hexdigest()
def git(): return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
def texts():
    return [f"The {['first','second'][w]} week, {['weekday (Monday to Friday)','weekend (Saturday and Sunday)'][d]} {PERIODS[h]}. Select the observed values in this temporal role."
            for w in range(2) for d in range(2) for h in range(4)]
def role_cal():
    out=[]
    for w in range(2):
        for d in range(2):
            for h in range(4):
                angle=2*np.pi*(h+.5)/4
                v=list(np.eye(16)[w*8+d*4+h])
                for k in range(1,5): v.extend([np.sin(k*angle),np.cos(k*angle)])
                for k in range(1,3): v.extend([np.sin(k*np.pi*d),np.cos(k*np.pi*d)])
                for k in range(1,3): v.extend([np.sin(k*np.pi*w),np.cos(k*np.pi*w)])
                out.append(v)
    return np.array(out,dtype=np.float32)
def energy(z):
    z=np.asarray(z,np.float64); z=z-z.mean(0,keepdims=True)
    z=z/np.maximum(np.sqrt(np.mean(z*z,axis=1,keepdims=True)),1e-8)
    return z.astype(np.float32)
def definition(c):
    pairs=[(a,b) for a in range(8) for b in range(a+1,8)]
    order=np.random.default_rng(c["composition_seed"]).permutation(len(pairs))
    held={op:{pairs[int(j)] for j in order[k*7:(k+1)*7]} for k,op in enumerate(OPS)}
    return pairs,held
def scalar_label(d,tol):
    return 2 if d>tol else 0 if d < -tol else 1
def case_rows(means,peaks,held,tol):
    rows=[]
    for op in range(2):
        stats=means if op==0 else peaks
        for a in range(16):
            for b in range(16):
                if a==b: continue
                ba,bb=a%8,b%8
                ood=int(tuple(sorted((ba,bb))) in held[OPS[op]])
                rows.append([op,a,b,a,b,scalar_label(stats[a]-stats[b],tol),ood])
    for a in range(8):
        for b in range(8):
            if a==b: continue
            diff=(means[a+8]-means[a])-(means[b+8]-means[b])
            ood=int(tuple(sorted((a,b))) in held[OPS[2]])
            rows.append([2,a,a+8,b,b+8,scalar_label(diff,tol),ood])
    return np.array(rows,np.int64)
def patch_calendar(idx):
    h=np.asarray(idx.hour); dow=np.asarray(idx.dayofweek); n=len(idx)
    week=np.arange(n)//168
    day=(dow>=5).astype(int)
    # Known calendar only; no targets, weather or holiday guesses.
    out=[np.eye(2)[week],np.eye(2)[day],np.eye(4)[h//6],np.eye(7)[dow]]
    for p in [24,168]:
        t=h if p==24 else dow*24+h
        out.append(np.stack([v for k in range(1,5) for v in
                            (np.sin(2*np.pi*k*t/p),np.cos(2*np.pi*k*t/p))],-1))
    out.append((np.arange(n)/(n-1))[:,None])
    return np.concatenate(out,-1).astype(np.float32)
def load_series(path,domain):
    a=pd.read_csv(path)
    if domain=="Capital":
        a["time"]=pd.to_datetime(a["dteday"])+pd.to_timedelta(a["hr"],unit="h")
        y=a.groupby("time")["cnt"].mean()
    else:
        a["time"]=pd.to_datetime(a["date_time"])
        # Weather rows can duplicate the same traffic observation.
        grouped=a.groupby("time")["traffic_volume"]
        conflicts=int((grouped.nunique()>1).sum())
        y=grouped.mean()
        return y.sort_index(),dict(raw_rows=len(a),unique_hours=len(y),duplicate_conflict_hours=conflicts,
                                    duplicate_policy="mean per native timestamp")
    return y.sort_index(),dict(raw_rows=len(a),unique_hours=len(y),duplicate_conflict_hours=0)
def prepare(c,config):
    r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False)
    save(r/"config.json",c)
    protected=[]
    for dirname in c["protect"]:
        for p in sorted(Path(dirname).iterdir()):
            if p.is_file(): protected.append(dict(path=str(p),sha=sha(p)))
    save(r/"protected_assets.json",protected)
    pairs,held=definition(c)
    meta=dict(roles=texts(),operations=OPS,labels={"0":"less","1":"similar","2":"greater"},
              relative_tolerance=c["tolerance"],windows={},holdout={k:[list(v) for v in sorted(s)] for k,s in held.items()},
              important="Structured choice understanding, not open QA, forecasting, causal attribution, or new independent raw datasets.")
    for domain,path in c["data"].items():
        series,clean=load_series(path,domain)
        first=series.index.min().normalize()
        first+=pd.Timedelta(days=(-first.dayofweek)%7)
        if first<series.index.min(): first+=pd.Timedelta(days=7)
        starts=pd.date_range(first,series.index.max()-pd.Timedelta(hours=335),freq="14D")
        values=[];valids=[];cals=[];masks=[];roles=[];stats=[];origins=[];skipped=[]
        for start in starts:
            idx=pd.date_range(start,periods=336,freq="h")
            y=series.reindex(idx).to_numpy(float);v=np.isfinite(y)
            if v.mean()<c["coverage"]:
                skipped.append(dict(start=str(start),reason="window coverage",coverage=float(v.mean())));continue
            day=np.asarray(idx.dayofweek>=5,dtype=int);week=np.arange(336)//168;hb=np.asarray(idx.hour)//6
            rid=week*8+day*4+hb
            complete=all(v[rid==j].mean()>=c["role_coverage"] for j in range(16))
            if not complete:
                skipped.append(dict(start=str(start),reason="role coverage",coverage=float(v.mean())));continue
            m=float(y[v].mean());sd=max(float(y[v].std()),1e-3)
            z=np.where(v,(y-m)/sd,0).reshape(56,6);vv=v.reshape(56,6)
            means=np.array([((y[rid==j][v[rid==j]]-m)/sd).mean() for j in range(16)])
            peaks=np.array([((y[rid==j][v[rid==j]]-m)/sd).max() for j in range(16)])
            numeric=[]
            for zz,vm in zip(z,vv):
                numeric.append([float(zz[vm].mean()) if vm.any() else 0.,
                                float(zz[vm].max()) if vm.any() else 0.,
                                float(zz[vm].std()) if vm.any() else 0.])
            values.append(np.c_[z,vv.astype(float),np.array(numeric)])
            valids.append(vv.any(1));cal=patch_calendar(idx)[np.arange(0,336,6)]
            cals.append(cal);pr=rid.reshape(56,6)[:,0]
            masks.append(np.stack([(pr==j)&vv.any(1) for j in range(16)]))
            roles.append(pr);stats.append([m,sd]);origins.append(str(start))
        n=len(values);assert n>=20,(domain,n)
        nf=int(n*.6);nv=int(n*.2)
        part=np.array([0]*nf+[1]*nv+[2]*(n-nf-nv))
        xx=np.array(values,dtype=np.float32);st=np.array(stats)
        # Absolute level/scale normalize using training windows only.
        smu=st[part==0].mean(0);ssd=np.maximum(st[part==0].std(0),1e-3)
        ss=((st-smu)/ssd).astype(np.float32)
        xx=np.concatenate([xx,np.repeat(ss[:,None,:],56,axis=1),np.array(cals)],-1)
        all_cases=[]
        for wi,(start,ms) in enumerate(zip(origins,masks)):
            # Recompute labels from original observations, never from model estimates.
            idx=pd.date_range(start,periods=336,freq="h");y=series.reindex(idx).to_numpy(float)
            m,sd=st[wi];valid=np.isfinite(y)
            rid=np.repeat(np.array(roles[wi]),6)
            mu=np.array([np.mean((y[(rid==j)&valid]-m)/sd) for j in range(16)])
            pk=np.array([np.max((y[(rid==j)&valid]-m)/sd) for j in range(16)])
            cr=case_rows(mu,pk,held,c["tolerance"])
            all_cases.append(np.c_[np.full(len(cr),wi),cr])
        cases=np.concatenate(all_cases)
        np.savez_compressed(r/(domain+"_data.npz"),x=xx,valid=np.array(valids),role_masks=np.array(masks),
                            cases=cases,window_split=part,origins=np.array(origins),raw_stats=st)
        m=dict(cleaning=clean,source_sha=sha(path),retained_windows=n,windows_by_split=np.bincount(part).tolist(),
               skipped=skipped,origins=origins,train_only_level_mean=smu.tolist(),train_only_level_sd=ssd.tolist(),
               input_dimension=int(xx.shape[-1]),cases={})
        for phase,pi in [("fit",0),("selection",1),("test",2)]:
            for kind,ood in [("seen",0),("unseen",1)]:
                k=(part[cases[:,0]]==pi)&(cases[:,-1]==ood);rr=cases[k]
                m["cases"][phase+"_"+kind]=dict(total=len(rr),by_operation={OPS[op]:np.bincount(rr[rr[:,1]==op,-2],minlength=3).tolist() for op in range(3)})
        meta["windows"][domain]=m
    save(r/"task_definition.json",meta)
    save(r/"protocol_lock.json",dict(source_commit=git(),source_sha=sha(__file__),config_sha=sha(config),
                                    test_metrics_read=False,scope="bounded pilot; no LLM training",
                                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    print("PREPARED", {k:v["cases"] for k,v in meta["windows"].items()},flush=True)
def extract(c):
    from transformers import AutoTokenizer,AutoModel
    r=Path(c["run"]);assert sha(__file__)==read(r/"protocol_lock.json")["source_sha"]
    t0=time.monotonic();torch.set_num_threads(2)
    # Only a few short forward passes. BF16 trunk sharded with embedding resident on GPU0.
    # CUDA_VISIBLE_DEVICES controls the three explicitly checked physical cards.
    devmap={"embed_tokens":0,"norm":2,"rotary_emb":2}
    for i in range(36):devmap[f"layers.{i}"]=0 if i<10 else 1 if i<23 else 2
    for i in range(3):
        assert torch.cuda.mem_get_info(i)[0] > c["extract_min_free_gib"]*2**30
        torch.cuda.set_per_process_memory_fraction(c["extract_memory_gib"]*2**30/torch.cuda.get_device_properties(i).total_memory,i)
    tok=AutoTokenizer.from_pretrained(c["model"],local_files_only=True)
    model=AutoModel.from_pretrained(c["model"],local_files_only=True,dtype=torch.bfloat16,
                                     device_map=devmap,attn_implementation="sdpa").eval()
    for p in model.parameters():p.requires_grad_(False)
    prompts=[];vectors=[]
    for role in texts():
        assert time.monotonic()-t0<c["extract_seconds"]
        prompt=tok.apply_chat_template([{"role":"user","content":"Represent the precise temporal meaning of this role, without assuming any demand level: "+role}],
                                       tokenize=False,add_generation_prompt=True,enable_thinking=False)
        batch=tok(prompt,return_tensors="pt")
        with torch.inference_mode():
            out=model(**batch,output_hidden_states=True,use_cache=False)
            vec=out.hidden_states[c["qwen_layer"]][0,-1].float().cpu().numpy()
        vectors.append(vec);prompts.append(prompt)
        print("QWEN_ROLE",len(vectors),round(time.monotonic()-t0,2),flush=True)
    full=np.array(vectors)
    # Label-free, fixed projection; roles all seen individually, only compositions are held out.
    rp=np.random.default_rng(c["projection_seed"]).normal(size=(full.shape[1],32))/np.sqrt(full.shape[1])
    q=energy(full@rp);fourier=energy(role_cal())
    permutation=np.random.default_rng(c["shuffle_seed"]).permutation(16)
    np.savez_compressed(r/"role_features.npz",qwen=q,fourier=fourier,shuffled=q[permutation],raw_qwen=full,
                        random_projection=rp.astype(np.float32),permutation=permutation)
    save(r/"feature_extraction.json",dict(model=c["model"],revision=c["revision"],frozen=True,
        non_thinking=True,generation_calls=0,forward_calls=len(vectors),layer=c["qwen_layer"],
        representation="last input-token hidden state, fixed Gaussian projection to32, per-role RMS1",
        prompts=prompts,seconds=time.monotonic()-t0,device_map={k:str(v) for k,v in model.hf_device_map.items()},
        physical_gpu_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"),
        peak_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(3)],
        ranks={k:int(np.linalg.matrix_rank(v)) for k,v in [("Qwen",q),("Fourier",fourier),("Shuffle",q[permutation])]},
        ))
    print("EXTRACTION_DONE",time.monotonic()-t0,flush=True)

class Block(nn.Module):
    def __init__(self,d):
        super().__init__();self.attn=nn.MultiheadAttention(d,4,batch_first=True,dropout=0)
        self.ln1=nn.LayerNorm(d);self.ln2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d))
    def forward(self,q,memory,valid,joint):
        if joint:
            n=memory.shape[1];z=torch.cat([memory,q],1)
            mask=torch.cat([~valid,torch.zeros((len(q),4),dtype=torch.bool,device=q.device)],1)
            out,w=self.attn(z,z,z,key_padding_mask=mask,need_weights=True)
            z=self.ln1(z+out);z=self.ln2(z+self.ff(z))
            return z[:,n:],z[:,:n],w[:,n:,:n]
        out,w=self.attn(q,memory,memory,key_padding_mask=~valid,need_weights=True)
        q=self.ln1(q+out);q=self.ln2(q+self.ff(q))
        return q,memory,w
class Model(nn.Module):
    def __init__(self,indim,bank,joint=False,d=32):
        super().__init__();self.register_buffer("bank",torch.as_tensor(bank,dtype=torch.float32))
        self.patch=nn.Sequential(nn.Linear(indim,d),nn.GELU(),nn.Linear(d,d))
        self.query=nn.Linear(32,d);self.blocks=nn.ModuleList([Block(d),Block(d)])
        self.op=nn.Embedding(3,d)
        self.head=nn.Sequential(nn.Linear(d*5,64),nn.GELU(),nn.Linear(64,3))
        self.joint=joint
    def forward(self,x,valid,roles,op,truth=None):
        memory=self.patch(x);q=self.query(self.bank[roles]);attentions=[]
        for layer in self.blocks:
            q,memory,w=layer(q,memory,valid,self.joint);attentions.append(w)
        logits=self.head(torch.cat([q.flatten(1),self.op(op)],-1))
        aux=torch.zeros((),device=x.device)
        if truth is not None:
            # Same calendar-derived role supervision in all four arms, including PatchConcat.
            mass=(attentions[-1]*truth).sum(-1).clamp_min(1e-8)
            aux=-mass.log().mean()
        return logits,aux,attentions[-1]
def get_batch(data,idx,device):
    cases=data["cases"][idx];wi=cases[:,0];roles=cases[:,2:6]
    def tensor(v,dtype=None):return torch.as_tensor(v,device=device,dtype=dtype)
    x=tensor(data["x"][wi]);valid=tensor(data["valid"][wi])
    rr=tensor(roles,torch.long);op=tensor(cases[:,1],torch.long);y=tensor(cases[:,6],torch.long)
    truth=data["role_masks"][wi[:,None],roles]
    return x,valid,rr,op,y,tensor(truth,torch.float32)
def metrics(prob,rows):
    pred=prob.argmax(-1);y=rows[:,6];correct=pred==y
    out=dict(n=len(rows),accuracy=float(correct.mean()),by_operation={})
    bas=[]
    for op in range(3):
        m=rows[:,1]==op
        aa=float(correct[m].mean()) if m.any() else 0.
        recalls=[float(correct[m&(y==cl)].mean()) for cl in range(3) if (m&(y==cl)).any()]
        out["by_operation"][OPS[op]]=dict(n=int(m.sum()),accuracy=aa,balanced_accuracy=float(np.mean(recalls)),
                                                     labels=np.bincount(y[m],minlength=3).tolist())
        bas.append(float(np.mean(recalls)))
    out["macro_operation_accuracy"]=float(np.mean([v["accuracy"] for v in out["by_operation"].values()]))
    out["macro_balanced_accuracy"]=float(np.mean(bas))
    return out
@torch.no_grad()
def evaluate(model,data,idx,device,batch=512):
    model.eval();pp=[];mass=[]
    for a in range(0,len(idx),batch):
        x,v,r,op,y,truth=get_batch(data,idx[a:a+batch],device)
        logits,aux,attn=model(x,v,r,op,truth)
        pp.append(logits.softmax(-1).cpu().numpy());mass.append((attn*truth).sum(-1).mean(-1).cpu().numpy())
    return np.concatenate(pp),np.concatenate(mass)
def indices(data,phase,ood):
    ca=data["cases"];return np.flatnonzero((data["window_split"][ca[:,0]]==phase)&(ca[:,-1]==ood))
def fit(c,config):
    r=Path(c["run"]);lock=read(r/"protocol_lock.json")
    assert sha(__file__)==lock["source_sha"] and sha(config)==lock["config_sha"]
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_device(0);assert torch.cuda.mem_get_info(0)[0]>c["fit_memory_gib"]*2**30
    torch.cuda.set_per_process_memory_fraction(c["fit_memory_gib"]*2**30/torch.cuda.get_device_properties(0).total_memory)
    device="cuda:0";features=np.load(r/"role_features.npz");curves=[];t0=time.monotonic()
    params={};init_hashes={}
    for domain in c["data"]:
        a=np.load(r/(domain+"_data.npz"));data={k:a[k] for k in a.files}
        tr=indices(data,0,0);va=indices(data,1,0)
        rows=data["cases"][tr];ct=np.zeros((3,3))
        for op in range(3):
            for y in range(3):ct[op,y]=((rows[:,1]==op)&(rows[:,6]==y)).sum()
        weight=1/np.maximum(ct,1);weight=weight/(weight[rows[:,1],rows[:,6]].mean())
        for seed in c["seeds"]:
            batches=np.random.default_rng(seed).integers(0,len(tr),size=(c["steps"],c["batch"]))
            for arm in ARMS:
                assert time.monotonic()-t0<c["fit_seconds"],"Global fit budget reached; retain completed and failed logs"
                torch.manual_seed(seed);np.random.seed(seed)
                bank=features["fourier" if arm=="TimeBindFourier" else "shuffled" if arm=="TimeBindShuffled" else "qwen"]
                model=Model(data["x"].shape[-1],bank,arm=="PatchConcatQwen").to(device)
                params[arm]=sum(p.numel() for p in model.parameters())
                h=hashlib.sha256()
                for name,p in model.named_parameters():h.update(p.detach().cpu().numpy().tobytes())
                init_hashes[domain+"_"+str(seed)+"_"+arm]=h.hexdigest()
                optimizer=torch.optim.AdamW(model.parameters(),lr=c["lr"],weight_decay=c["weight_decay"])
                td=r/"fits"/domain/str(seed)/arm;td.mkdir(parents=True,exist_ok=False)
                ww=torch.tensor(weight,device=device,dtype=torch.float32);beg=time.monotonic()
                for step in range(1,c["steps"]+1):
                    model.train();x,v,roles,op,y,truth=get_batch(data,tr[batches[step-1]],device)
                    logits,aux,_=model(x,v,roles,op,truth)
                    ce=(F.cross_entropy(logits,y,reduction="none")*ww[op,y]).mean()
                    loss=ce+c["binding_weight"]*aux
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
                    if step%c["save_every"]==0:
                        prob,attn=evaluate(model,data,va,device)
                        met=metrics(prob,data["cases"][va])
                        curve=dict(domain=domain,seed=seed,arm=arm,step=step,fit_loss=float(loss.detach()),
                                   selection=met,role_mass=float(attn.mean()),seconds=time.monotonic()-beg)
                        curves.append(curve);save(r/"curves.json",curves)
                        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},td/f"step{step}.pt")
                        print("FIT",domain,seed,arm,step,round(met["macro_balanced_accuracy"],4),round(time.monotonic()-t0,1),flush=True)
                        assert time.monotonic()-t0<c["fit_seconds"],"Global fit budget reached"
                save(td/"completed.json",dict(steps=c["steps"],seconds=time.monotonic()-beg,parameters=params[arm]))
                del model,optimizer;torch.cuda.empty_cache()
    assert len(set(params.values()))==1,params
    for domain in c["data"]:
        for seed in c["seeds"]:
            assert len({init_hashes[domain+"_"+str(seed)+"_"+arm] for arm in ARMS})==1
    chosen={}
    for domain in c["data"]:
        chosen[domain]={}
        for arm in ARMS:
            vals=[]
            for step in range(c["save_every"],c["steps"]+1,c["save_every"]):
                subset=[z for z in curves if z["domain"]==domain and z["arm"]==arm and z["step"]==step]
                assert len(subset)==len(c["seeds"])
                vals.append((np.mean([z["selection"]["macro_balanced_accuracy"] for z in subset]),-step,step))
            chosen[domain][arm]=max(vals)[2]
    save(r/"selection_lock.json",dict(selection=chosen,rule="maximize mean3seed validation-seen macro operation/class balanced accuracy; tie earliest; no unseen-composition/test scores",
                                     source_commit=git(),parameters=params,initial_trainable_sha=init_hashes,
                                     fit_seconds=time.monotonic()-t0,features_sha=sha(r/"role_features.npz"),
                                     test_metrics_read=False))
    print("FIT_DONE",chosen,"parameters",params,flush=True)
def assess(c):
    r=Path(c["run"]);sel=read(r/"selection_lock.json");device="cpu";torch.set_num_threads(2)
    features=np.load(r/"role_features.npz");out=[];predictions={};paired=[]
    for domain in c["data"]:
        a=np.load(r/(domain+"_data.npz"));data={k:a[k] for k in a.files}
        train=data["cases"][indices(data,0,0)]
        majority={op:int(np.bincount(train[train[:,1]==op,6],minlength=3).argmax()) for op in range(3)}
        for seed in c["seeds"]:
            for arm in ARMS:
                bank=features["fourier" if arm=="TimeBindFourier" else "shuffled" if arm=="TimeBindShuffled" else "qwen"]
                model=Model(data["x"].shape[-1],bank,arm=="PatchConcatQwen")
                path=r/"fits"/domain/str(seed)/arm/f"step{sel['selection'][domain][arm]}.pt"
                model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True));model.eval()
                for name,phase,ood in [("test_seen",2,0),("test_unseen",2,1),("selection_unseen",1,1)]:
                    ix=indices(data,phase,ood);rows=data["cases"][ix]
                    prob,mass=evaluate(model,data,ix,device)
                    mm=metrics(prob,rows)
                    mm.update(domain=domain,seed=seed,arm=arm,phase=name,step=sel["selection"][domain][arm],
                              role_mass=float(mass.mean()),parameters=sel["parameters"][arm],
                              train_majority_accuracy=float(np.mean([majority[int(v)] for v in rows[:,1]]==rows[:,6])))
                    out.append(mm);key=f"{domain}_{seed}_{arm}_{name}"
                    predictions[key+"_prob"]=prob;predictions[key+"_rows"]=rows;predictions[key+"_mass"]=mass
                print("SCORE",domain,seed,arm,out[-2]["macro_operation_accuracy"],flush=True)
    np.savez_compressed(r/"predictions.npz",**predictions);save(r/"metrics.json",out)
    summary={}
    rng=np.random.default_rng(c["bootstrap_seed"])
    for domain in c["data"]:
        summary[domain]={}
        for arm in ARMS:
            mm=[z for z in out if z["domain"]==domain and z["arm"]==arm and z["phase"]=="test_unseen"]
            summary[domain][arm]={k:float(np.mean([z[k] for z in mm])) for k in
                                 ["accuracy","macro_operation_accuracy","macro_balanced_accuracy","role_mass"]}
            summary[domain][arm]["per_seed"]=[{k:z[k] for k in ["seed","accuracy","macro_operation_accuracy","macro_balanced_accuracy"]} for z in mm]
            summary[domain][arm]["n_cases_per_seed"]=mm[0]["n"]
        for comparator in ARMS[1:]:
            wins=[]
            for seed in c["seeds"]:
                ka=f"{domain}_{seed}_TimeBindQwen_test_unseen";kb=f"{domain}_{seed}_{comparator}_test_unseen"
                rows=predictions[ka+"_rows"];assert np.array_equal(rows,predictions[kb+"_rows"])
                aa=(predictions[ka+"_prob"].argmax(1)==rows[:,6]).astype(float)
                bb=(predictions[kb+"_prob"].argmax(1)==rows[:,6]).astype(float)
                perwindow=[]
                for wi in np.unique(rows[:,0]):
                    z=[]
                    for op in range(3):
                        m=(rows[:,0]==wi)&(rows[:,1]==op);z.append(float((aa[m]-bb[m]).mean()))
                    perwindow.append(np.mean(z))
                wins.append(perwindow)
            block=np.array(wins).mean(0);n=len(block)
            boot=block[rng.integers(0,n,size=(c["bootstrap_repeats"],n))].mean(1)
            paired.append(dict(domain=domain,comparator=comparator,estimand="mean-seed macro-operation accuracy difference; resample whole14day windows",
                               n_windows=n,difference=float(block.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    protected=read(r/"protected_assets.json")
    for v in protected:assert sha(v["path"])==v["sha"]
    save(r/"paired_intervals.json",paired)
    save(r/"summary.json",dict(results=summary,paired=paired,scope="First structured understanding pilot, same raw datasets previously explored; not free-form QA or proof of LLM semantic contribution",
                              protected_assets=len(protected),source_commit=read(r/"protocol_lock.json")["source_commit"],
                              code_sha=sha(__file__),fit_seconds=sel["fit_seconds"],selection=sel["selection"]))
    print("SUMMARY",json.dumps(summary),flush=True)
def verify(c):
    r=Path(c["run"]);a=np.load(r/"predictions.npz");saved=read(r/"metrics.json");maxerr=0.
    for row in saved:
        key=f"{row['domain']}_{row['seed']}_{row['arm']}_{row['phase']}"
        mm=metrics(a[key+"_prob"],a[key+"_rows"])
        maxerr=max(maxerr,abs(mm["accuracy"]-row["accuracy"]),abs(mm["macro_operation_accuracy"]-row["macro_operation_accuracy"]))
    for domain in c["data"]:
        d=np.load(r/(domain+"_data.npz"));dates=pd.to_datetime(d["origins"])
        assert all((dates[1:]-dates[:-1])>=pd.Timedelta(days=14))
        assert d["cases"].shape[1]==8
        split=d["window_split"];ca=d["cases"]
        seen=set(tuple(v) for v in ca[(split[ca[:,0]]==0)&(ca[:,-1]==0),1:6])
        new=set(tuple(v) for v in ca[(split[ca[:,0]]==2)&(ca[:,-1]==1),1:6])
        assert not seen&new
        assert np.isfinite(d["x"]).all()
    assert maxerr<1e-12
    save(r/"verification.json",dict(metric_rows=len(saved),max_metric_error=maxerr,
                                    chronological_disjoint_windows=True,unseen_query_compositions_absent_from_fit=True,
                                    protected_unchanged=True,complete=True))
    print("VERIFIED",len(saved),maxerr,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["prepare","extract","fit","assess","verify"]);p.add_argument("--config",required=True)
    args=p.parse_args();cfg=read(args.config)
    try:
        if args.stage in ["prepare","fit"]:globals()[args.stage](cfg,args.config)
        else:globals()[args.stage](cfg)
    except Exception:
        import traceback
        run=Path(cfg["run"])
        if run.exists():(run/f"failure_{args.stage}_{int(time.time())}.log").write_text(traceback.format_exc())
        raise

