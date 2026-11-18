"""TimeBind67: warm-start Patch interaction with post-interaction semantic soft bias."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",
                 PYTHONDONTWRITEBYTECODE="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
import argparse, csv, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from exp import exp_chronogeom_timebind66 as old
ARMS=["PlainPatch","QwenSoft","FourierSoft","ShuffleSoft"]
save,read,sha,git=old.save,old.read,old.sha,old.git

def geometry(bank):
    z=torch.as_tensor(bank,dtype=torch.float32)
    z=F.normalize(z,dim=-1)
    return z@z.T

class SoftModel(old.Model):
    def __init__(self,indim,query_bank,bias_bank=None):
        super().__init__(indim,query_bank,joint=True)
        self.alpha=nn.Parameter(torch.zeros(()))
        rel=torch.zeros(16,16) if bias_bank is None else geometry(bias_bank)
        self.register_buffer("relation",rel)
    def prior(self,roles,patch_roles,valid):
        b=self.relation[roles[:,:,None],patch_roles[:,None,:]]
        vv=valid[:,None,:].to(b.dtype)
        n=vv.sum(-1,keepdim=True).clamp_min(1)
        mu=(b*vv).sum(-1,keepdim=True)/n
        sd=(((b-mu)**2*vv).sum(-1,keepdim=True)/n).sqrt().clamp_min(1e-6)
        return (b-mu)/sd
    def forward(self,x,valid,roles,op,truth=None,patch_roles=None):
        assert patch_roles is not None
        memory=self.patch(x);q=self.query(self.bank[roles])
        q,memory,_=self.blocks[0](q,memory,valid,True)
        n=memory.shape[1];z=torch.cat([memory,q],1);batch,length,d=z.shape
        block=self.blocks[1];heads=block.attn.num_heads
        bias=self.prior(roles,patch_roles,valid)
        mask=torch.zeros((batch,length,length),device=z.device,dtype=z.dtype)
        mask[:,n:,:n]=self.alpha*bias
        keyvalid=torch.cat([valid,torch.ones((batch,4),device=z.device,dtype=torch.bool)],1)
        mask=mask.masked_fill(~keyvalid[:,None,:],float("-inf"))
        mask=mask[:,None].expand(-1,heads,-1,-1).reshape(batch*heads,length,length)
        out,w=block.attn(z,z,z,attn_mask=mask,need_weights=True)
        z=block.ln1(z+out);z=block.ln2(z+block.ff(z))
        q=z[:,n:];attn=w[:,n:,:n]
        logits=self.head(torch.cat([q.flatten(1),self.op(op)],-1))
        aux=-(attn*truth).sum(-1).clamp_min(1e-8).log().mean() if truth is not None else logits.new_zeros(())
        return logits,aux,attn

def dataset(c,domain):
    a=np.load(Path(c["source"])/(domain+"_data.npz"))
    return {k:a[k] for k in a.files}
def batch(data,idx,device):
    x,v,r,op,y,truth=old.get_batch(data,idx,device)
    wi=data["cases"][idx,0]
    # Role identity is derived only from existing legal calendar, not numeric labels.
    patch_roles=np.argmax(data["role_masks"][wi],axis=1)
    pr=torch.as_tensor(patch_roles,device=device,dtype=torch.long)
    return x,v,r,op,y,truth,pr
def create(c,domain,seed,arm,device,features):
    bank=None if arm=="PlainPatch" else features[{"QwenSoft":"qwen","FourierSoft":"fourier","ShuffleSoft":"shuffled"}[arm]]
    model=SoftModel(dataset(c,domain)["x"].shape[-1],features["qwen"],bank).to(device)
    src=Path(c["source"])/"fits"/domain/str(seed)/"PatchConcatQwen"/f"step{c['warm_step']}.pt"
    state=torch.load(src,map_location=device,weights_only=True)
    missing,unexpected=model.load_state_dict(state,strict=False)
    assert set(missing)=={"alpha","relation"} and not unexpected
    return model,src,state

@torch.no_grad()
def evaluate(model,data,idx,device,batch_size=512):
    model.eval();pp=[];mm=[]
    for a in range(0,len(idx),batch_size):
        x,v,r,op,y,truth,pr=batch(data,idx[a:a+batch_size],device)
        logits,aux,attn=model(x,v,r,op,truth,pr)
        pp.append(logits.softmax(-1).cpu().numpy());mm.append((attn*truth).sum(-1).mean(-1).cpu().numpy())
    return np.concatenate(pp),np.concatenate(mm)

def prepare(c,config):
    r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c)
    src=Path(c["source"]);protected=read(src/"protected_assets.json")
    files=[src/"Capital_data.npz",src/"Metro_data.npz",src/"role_features.npz",src/"selection_lock.json",
           src/"protocol_lock.json",Path(old.__file__)]
    for domain in c["domains"]:
        for seed in c["seeds"]:
            files.append(src/"fits"/domain/str(seed)/"PatchConcatQwen"/f"step{c['warm_step']}.pt")
    seen={v["path"] for v in protected}
    for p in files:
        if str(p) not in seen:protected.append(dict(path=str(p),sha=sha(p)));seen.add(str(p))
    # Preserve all original66 assets as well as22/51.
    for p in src.rglob("*"):
        if p.is_file() and str(p) not in seen:protected.append(dict(path=str(p),sha=sha(p)));seen.add(str(p))
    save(r/"protected_assets.json",protected)
    save(r/"protocol_lock.json",dict(source_commit=git(),source_sha=sha(__file__),config_sha=sha(config),
         timebind66_sha=sha(old.__file__),feature_sha=sha(src/"role_features.npz"),
         source_split_sha={d:sha(src/(d+"_data.npz")) for d in c["domains"]},
         test_previously_viewed=True,test_used_for_selection=False,new_llm_calls=0,model_downloads=0,
         timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    print("PREPARED67 protected",len(protected),flush=True)
def fit(c,config):
    r=Path(c["run"]);lock=read(r/"protocol_lock.json")
    assert sha(__file__)==lock["source_sha"] and sha(config)==lock["config_sha"]
    assert sha(old.__file__)==lock["timebind66_sha"]
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.set_device(0);assert torch.cuda.mem_get_info(0)[0]>c["memory_gib"]*2**30
    torch.cuda.set_per_process_memory_fraction(c["memory_gib"]*2**30/torch.cuda.get_device_properties(0).total_memory)
    device="cuda:0";features=np.load(Path(c["source"])/"role_features.npz")
    curves=[];inits=[];params={};t0=time.monotonic()
    for domain in c["domains"]:
        data=dataset(c,domain);tr=old.indices(data,0,0);va=old.indices(data,1,0)
        rows=data["cases"][tr];ct=np.zeros((3,3))
        for op in range(3):
            for y in range(3):ct[op,y]=((rows[:,1]==op)&(rows[:,6]==y)).sum()
        weight=1/np.maximum(ct,1);weight/=weight[rows[:,1],rows[:,6]].mean()
        ww=torch.as_tensor(weight,device=device,dtype=torch.float32)
        for seed in c["seeds"]:
            batches=np.random.default_rng(seed).integers(0,len(tr),size=(c["steps"],c["batch"]))
            for arm in ARMS:
                assert time.monotonic()-t0<c["fit_seconds"],"Global fit budget exhausted"
                torch.manual_seed(seed);np.random.seed(seed)
                model,src,state=create(c,domain,seed,arm,device,features)
                params[arm]=sum(p.numel() for p in model.parameters())
                # Compare initial predictions on fixed selection examples BEFORE any update.
                ref=old.Model(data["x"].shape[-1],features["qwen"],joint=True).to(device)
                ref.load_state_dict(state);ref.eval();model.eval()
                ii=va[np.linspace(0,len(va)-1,min(len(va),256),dtype=int)]
                with torch.no_grad():
                    x,v,rr,op,y,truth,pr=batch(data,ii,device)
                    zref=ref(x,v,rr,op,truth)[0];znew=model(x,v,rr,op,truth,pr)[0]
                    err=float((zref-znew).abs().max())
                assert err<c["warm_atol"],(domain,seed,arm,err)
                h=hashlib.sha256()
                for name,p in model.named_parameters():h.update(p.detach().cpu().numpy().tobytes())
                inits.append(dict(domain=domain,seed=seed,arm=arm,source=str(src),source_sha=sha(src),
                                  max_initial_logit_error=err,initial_trainable_sha=h.hexdigest()))
                save(r/"warmstart_verification.json",inits);del ref
                td=r/"fits"/domain/str(seed)/arm;td.mkdir(parents=True,exist_ok=False)
                optimizer=torch.optim.AdamW(model.parameters(),lr=c["lr"],weight_decay=c["weight_decay"])
                began=time.monotonic()
                for step in range(0,c["steps"]+1):
                    if step:
                        model.train();x,v,rr,op,y,truth,pr=batch(data,tr[batches[step-1]],device)
                        logits,aux,_=model(x,v,rr,op,truth,pr)
                        ce=(F.cross_entropy(logits,y,reduction="none")*ww[op,y]).mean()
                        loss=ce+c["binding_weight"]*aux
                        optimizer.zero_grad(set_to_none=True);loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
                    if step%c["save_every"]==0:
                        pp,mass=evaluate(model,data,va,device)
                        met=old.metrics(pp,data["cases"][va])
                        z=dict(domain=domain,seed=seed,arm=arm,step=step,selection=met,
                               alpha=float(model.alpha.detach()),role_mass=float(mass.mean()),
                               seconds=time.monotonic()-began)
                        curves.append(z);save(r/"curves.json",curves)
                        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},td/f"step{step}.pt")
                        print("FIT67",domain,seed,arm,step,round(met["macro_balanced_accuracy"],4),
                              "alpha",round(z["alpha"],5),"sec",round(time.monotonic()-t0,1),flush=True)
                        assert time.monotonic()-t0<c["fit_seconds"],"Global fit budget exhausted"
                save(td/"completed.json",dict(actual_updates=c["steps"],seconds=time.monotonic()-began,
                    nominal_parameters=params[arm],active_parameters=params[arm]-(arm=="PlainPatch"),
                    peak_gib=torch.cuda.max_memory_allocated()/2**30))
                del model,optimizer;torch.cuda.empty_cache()
    assert len(set(params.values()))==1
    for domain in c["domains"]:
        for seed in c["seeds"]:
            assert len({v["initial_trainable_sha"] for v in inits if v["domain"]==domain and v["seed"]==seed})==1
    chosen={}
    for domain in c["domains"]:
        chosen[domain]={}
        for arm in ARMS:
            vals=[]
            for step in range(0,c["steps"]+1,c["save_every"]):
                mm=[v for v in curves if v["domain"]==domain and v["arm"]==arm and v["step"]==step]
                assert len(mm)==len(c["seeds"])
                vals.append((np.mean([v["selection"]["macro_balanced_accuracy"] for v in mm]),-step,step))
            chosen[domain][arm]=max(vals)[2]
    save(r/"selection_lock.json",dict(selection=chosen,rule="same3seed checkpoint: maximize earlier selection-seen macro balanced accuracy, tie earliest; includes0",
         source_commit=git(),parameters=params,source_warm_step=c["warm_step"],actual_fits=24,
         actual_updates=c["steps"]*24,fit_seconds=time.monotonic()-t0,peak_gib=torch.cuda.max_memory_allocated()/2**30,
         test_used_for_selection=False,source66_frozen=True,
         inactive_parameter_note="PlainPatch alpha multiplies a zero prior;31396 nominal/31395 active, biased arms31396 active"))
    print("FIT_DONE67",chosen,flush=True)

def assess(c):
    r=Path(c["run"]);sel=read(r/"selection_lock.json");features=np.load(Path(c["source"])/"role_features.npz")
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    mm=[];pred={};summary={};paired=[];fail=[];rng=np.random.default_rng(c["bootstrap_seed"])
    for domain in c["domains"]:
        data=dataset(c,domain);origins=data["origins"];summary[domain]={}
        for seed in c["seeds"]:
            for arm in ARMS:
                model,_,_=create(c,domain,seed,arm,"cpu",features)
                path=r/"fits"/domain/str(seed)/arm/f"step{sel['selection'][domain][arm]}.pt"
                model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True));model.eval()
                for phase,pi,ood in [("test_seen",2,0),("test_unseen",2,1),("selection_unseen",1,1)]:
                    ix=old.indices(data,pi,ood);rows=data["cases"][ix]
                    prob,mass=evaluate(model,data,ix,"cpu")
                    met=old.metrics(prob,rows)
                    met.update(domain=domain,seed=seed,arm=arm,phase=phase,step=sel["selection"][domain][arm],
                               alpha=float(model.alpha.detach()),role_mass=float(mass.mean()))
                    mm.append(met);key=f"{domain}_{seed}_{arm}_{phase}"
                    pred[key+"_prob"]=prob;pred[key+"_rows"]=rows;pred[key+"_mass"]=mass
                print("SCORE67",domain,seed,arm,mm[-2]["macro_operation_accuracy"],flush=True)
        for arm in ARMS:
            z=[v for v in mm if v["domain"]==domain and v["arm"]==arm and v["phase"]=="test_unseen"]
            summary[domain][arm]={k:float(np.mean([v[k] for v in z])) for k in
                                 ["accuracy","macro_operation_accuracy","macro_balanced_accuracy","alpha","role_mass"]}
            summary[domain][arm]["per_seed"]=[{k:v[k] for k in ["seed","accuracy","macro_operation_accuracy","macro_balanced_accuracy","alpha"]} for v in z]
            summary[domain][arm]["by_operation"]={op:{k:float(np.mean([v["by_operation"][op][k] for v in z])) for k in ["accuracy","balanced_accuracy"]} for op in old.OPS}
            summary[domain][arm]["n_cases_per_seed"]=z[0]["n"]
        for comparator in ["PlainPatch","FourierSoft","ShuffleSoft"]:
            perseed=[];perseed_change=[]
            for seed in c["seeds"]:
                ka=f"{domain}_{seed}_QwenSoft_test_unseen";kb=f"{domain}_{seed}_{comparator}_test_unseen"
                rows=pred[ka+"_rows"];assert np.array_equal(rows,pred[kb+"_rows"])
                aa=(pred[ka+"_prob"].argmax(1)==rows[:,6]).astype(float)
                bb=(pred[kb+"_prob"].argmax(1)==rows[:,6]).astype(float)
                block=[];change=[]
                for wi in np.unique(rows[:,0]):
                    ops=[float((aa[(rows[:,0]==wi)&(rows[:,1]==op)]-bb[(rows[:,0]==wi)&(rows[:,1]==op)]).mean()) for op in range(3)]
                    block.append(np.mean(ops));change.append(ops[2])
                perseed.append(block);perseed_change.append(change)
                for op in range(3):
                    for category,mask in [("qwen_wrong_control_right",(aa==0)&(bb==1)),("qwen_right_control_wrong",(aa==1)&(bb==0))]:
                        ix=np.flatnonzero(mask&(rows[:,1]==op))
                        if len(ix):
                            j=int(ix[0]);v=rows[j]
                            fail.append(dict(domain=domain,seed=seed,comparator=comparator,operation=old.OPS[op],
                                category=category,selection="earliest fixed-index example, no tuning",window_start=str(origins[v[0]]),
                                roles=v[2:6].tolist(),target=int(v[6]),qwen_prediction=int(pred[ka+"_prob"][j].argmax()),
                                comparator_prediction=int(pred[kb+"_prob"][j].argmax()),row=j))
            block=np.mean(perseed,axis=0);change=np.mean(perseed_change,axis=0);n=len(block)
            ix=rng.integers(0,n,size=(c["bootstrap_repeats"],n))
            paired.append(dict(domain=domain,comparator=comparator,n_windows=n,
                 macro_difference=float(block.mean()),macro_ci95=np.quantile(block[ix].mean(1),[.025,.975]).tolist(),
                 change_difference=float(change.mean()),change_ci95=np.quantile(change[ix].mean(1),[.025,.975]).tolist(),
                 unit="whole14day window; average seeds first"))
    save(r/"metrics.json",mm);np.savez_compressed(r/"predictions.npz",**pred)
    save(r/"paired_intervals.json",paired);save(r/"failure_examples.json",fail)
    flat=[]
    for v in mm:
        for op,t in v["by_operation"].items():
            flat.append({k:v[k] for k in ["domain","seed","arm","phase","step","alpha","role_mass"]}|
                        dict(operation=op,n=t["n"],accuracy=t["accuracy"],balanced_accuracy=t["balanced_accuracy"]))
    with open(r/"task_metrics.csv","w") as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    save(r/"summary.json",dict(results=summary,paired=paired,source_commit=read(r/"protocol_lock.json")["source_commit"],
        source_sha=sha(__file__),source66=str(c["source"]),selection=sel["selection"],actual_fits=24,
        actual_updates=sel["actual_updates"],fit_seconds=sel["fit_seconds"],peak_fit_gib=sel["peak_gib"],
        new_llm_calls=0,model_downloads=0,test_previously_viewed=True,independent_confirmation=False,
        comparison="common inherited Qwen role-input interface; vary ONLY additional semantic prior, not complete LLM removal",
        parameters=sel["parameters"],inactive_parameter_note=sel["inactive_parameter_note"],
        scope="structured3class choices on observed14day windows; no natural-language generation, forecasting or LLMSFT"))
    print("SUMMARY67",json.dumps(summary),flush=True)

def verify(c):
    r=Path(c["run"]);pp=np.load(r/"predictions.npz");metrics=read(r/"metrics.json");maxerr=0.
    for z in metrics:
        key=f"{z['domain']}_{z['seed']}_{z['arm']}_{z['phase']}"
        rows=pp[key+"_rows"];pr=pp[key+"_prob"]
        # Independent direct recomputation, without calling the original metric function.
        correct=pr.argmax(1)==rows[:,6]
        aa=float(correct.mean());macro=float(np.mean([correct[rows[:,1]==op].mean() for op in range(3)]))
        maxerr=max(maxerr,abs(aa-z["accuracy"]),abs(macro-z["macro_operation_accuracy"]))
    protected=read(r/"protected_assets.json")
    for z in protected:assert sha(z["path"])==z["sha"],z["path"]
    assert maxerr<1e-12
    init=read(r/"warmstart_verification.json");assert len(init)==24
    assert max(v["max_initial_logit_error"] for v in init)<c["warm_atol"]
    for d in c["domains"]:
        for seed in c["seeds"]:
            for arm in ARMS:assert read(r/"fits"/d/str(seed)/arm/"completed.json")["actual_updates"]==c["steps"]
    save(r/"verification.json",dict(complete=True,metric_rows=len(metrics),max_metric_error=maxerr,
        protected_assets=len(protected),protected_unchanged=True,warm_start_cases=len(init),
        max_initial_logit_error=max(v["max_initial_logit_error"] for v in init),actual_fits=24,actual_updates=24*c["steps"]))
    print("VERIFIED67",len(metrics),"metrics",len(protected),"protected",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["prepare","fit","assess","verify"]);ap.add_argument("--config",required=True)
    a=ap.parse_args();c=read(a.config)
    try:
        if a.stage in ["prepare","fit"]:globals()[a.stage](c,a.config)
        else:globals()[a.stage](c)
    except Exception:
        r=Path(c["run"])
        if r.exists():(r/f"failure_{a.stage}_{int(time.time())}.log").write_text(traceback.format_exc())
        raise

