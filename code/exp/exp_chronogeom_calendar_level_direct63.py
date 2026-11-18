"""Calendar-aware level and KD-isolated full shape63; Metro-only minimal extension."""
import os
os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                  TORCH_DEVICE_BACKEND_AUTOLOAD="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
import argparse, copy, csv, json, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from layers.ChronoGeomTimeBaseOfficial import Model as OfficialTimeBase
from exp.exp_chronogeom_event_knowledge19 import Head, knowledge, read, save, sha, git, scores
from exp.exp_chronogeom_teacher_content40 import calendar_arrays, check22
from exp.exp_chronogeom_residual03 import numeric_summary

def center(y):
    return y-y.mean(1, keepdim=True)

def writecsv(p, rows):
    if rows:
        with Path(p).open("w", newline="") as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def statecpu(m):
    return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

def source_base(c, domain, seed):
    manifest=read(c["frozen22"])
    if domain=="Metro":
        return Path(c["domains"][domain]["source"])/f"FrozenTimeBase_seed{seed}.pt"
    suffix="qwen_capital01_20261027_01/fit/NoCalendar.pt" if seed==2024 else f"task_context02_20261028_01/capital/seed{seed}/NoCalendar.pt"
    return next(Path(a["path"]) for a in manifest["base_assets"] if a["path"].endswith(suffix))

def collapse_base(state):
    assert not torch.count_nonzero(state["table"])
    with torch.no_grad():
        v=F.linear(torch.zeros(1,168),state["time_mlp.0.weight"],state["time_mlp.0.bias"])
        v=F.linear(F.gelu(v),state["time_mlp.2.weight"],state["time_mlp.2.bias"])
        q=F.linear(v,state["readout.weight"],state["readout.bias"])[0]
    body={k[5:]:v for k,v in state.items() if k.startswith("body.")}
    return body, q

class DirectStudent(nn.Module):
    """Full prediction is a joint reader output; no stored base forecast or target."""
    def __init__(self, seed, base_state, numeric_mean, numeric_sd):
        super().__init__()
        cfg=SimpleNamespace(seq_len=96,pred_len=96,enc_in=1,period_len=24,basis_num=4,
                            individual=False,use_period_norm=True,use_orthogonal=False)
        self.body=OfficialTimeBase(cfg)
        body,q=collapse_base(base_state);self.body.load_state_dict(body)
        self.affine=nn.Parameter(q.clone())
        torch.manual_seed(seed+61000)
        self.codes=nn.Embedding(72,3);nn.init.normal_(self.codes.weight,std=.02)
        self.reader=nn.Sequential(nn.Linear(42,16),nn.GELU(),nn.Linear(16,1))
        self.register_buffer("numeric_mean",torch.tensor(numeric_mean,dtype=torch.float32))
        self.register_buffer("numeric_sd",torch.tensor(numeric_sd,dtype=torch.float32))
        # GELU(z)-GELU(-z)=z gives an identity initialization without a skip path.
        with torch.no_grad():
            self.reader[0].weight[:2].zero_();self.reader[0].bias[:2].zero_()
            self.reader[0].weight[0,7]=1.;self.reader[0].weight[1,7]=-1.
            self.reader[2].weight.zero_()
            self.reader[2].weight[0,0]=self.numeric_sd[7]
            self.reader[2].weight[0,1]=-self.numeric_sd[7]
            self.reader[2].bias[0]=self.numeric_mean[7]
        self.setup_calendar_level()
    def numerical_prediction(self, x):
        p=self.body(x)
        return p*(1+torch.tanh(self.affine[0]))+self.affine[1]
    def setup_calendar_level(self):
        self.calendar_level=nn.Linear(31,1)
        nn.init.zeros_(self.calendar_level.weight)
        nn.init.zeros_(self.calendar_level.bias)
    def level_from_prediction(self,numerical,calendar31):
        return numerical.mean(1,keepdim=True)+self.calendar_level(calendar31.mean(1))[:,None,:]
    def level_prediction(self,x,calendar31):
        return self.level_from_prediction(self.numerical_prediction(x),calendar31)
    def components(self,x,calendar31,ids):
        numerical=self.numerical_prediction(x)
        level=self.level_from_prediction(numerical,calendar31)
        # Sole isolation boundary: temporal supervision cannot shift numeric level.
        num=(numeric_summary(x,numerical.detach())-self.numeric_mean)/self.numeric_sd
        raw=self.reader(torch.cat([num,calendar31,self.codes(ids)],-1))
        shape=raw-raw.mean(1,keepdim=True)
        return level,shape
    def forward(self,x,calendar31,ids):
        level,shape=self.components(x,calendar31,ids)
        return level+shape

def to_pack(d, split, grid, raw, common, ids, device, indices=None):
    x=d[split+"_x"]; future=d[split+"_future"]
    if indices is not None:
        x=x[indices];future=future[indices]
    return dict(x=torch.tensor(x,dtype=torch.float32,device=device),
                calendar=torch.tensor(np.concatenate([raw[future],common[future]],-1),dtype=torch.float32,device=device),
                ids=torch.tensor(ids[future],dtype=torch.long,device=device),
                future=future,origin=d[split+"_origin"] if indices is None else d[split+"_origin"][indices])
def pred(m,p):
    m.eval()
    with torch.no_grad():
        return m(p["x"],p["calendar"],p["ids"]).cpu().numpy()
def interval(d,seed):
    # Predictions first average over training seeds; daily overlapping H96 windows
    # resampled in contiguous7origin blocks, not hourly independence.
    rng=np.random.default_rng(seed);n=len(d);length=min(7,n)
    starts=rng.integers(0,n,size=(2000,int(np.ceil(n/length))))
    ix=((starts[:,:,None]+np.arange(length))%n).reshape(2000,-1)[:,:n]
    means=d[ix].mean(1)
    return dict(delta=float(d.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)),
                origins=n,block_origins=length,blocks_approx=n/length,conditional_on_three_fits=True)

def prepare(c,config):
    r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c)
    records=check22(c)
    roots=[Path(c["source51"]),Path(c["source61"]),Path(c["source62"])]
    records += [dict(path=str(p),sha=sha(p)) for root in roots for p in root.iterdir()
                if p.is_file() and p.suffix in [".json",".pt",".npz"]]
    for spec in c["domains"].values():
        root=Path(spec["source"])
        for name in ["config.json","data.npz","predictions.npz","normalizers.json","knowledge.npz"]:
            p=root/name;records.append(dict(path=str(p),sha=sha(p)))
    root=Path(c["teacher40"])
    for name in ["teacher_predictions.npz","teacher_checkpoints.pt","teacher_selection.json","Capital/profiles.json","Metro/profiles.json"]:
        p=root/name;records.append(dict(path=str(p),sha=sha(p)))
    records += [dict(path=str(p),sha=sha(p)) for p in [Path("exp/exp_chronogeom_direct_forecast61.py"),Path("docs/chronogeom/direct_forecast61_config.json")]]
    save(r/"protected_assets.json",records)
    save(r/"protocol_lock.json",dict(config_sha=sha(config),source_sha=sha(__file__),git=git(),
        source_path=str(Path(__file__).resolve()),before_any_training=True,independent_confirmation=False,
        all_arms_same_parameters=True,all_arms_same_updates=True))
    print("PREPARED",str(r),len(records),flush=True)

def run(c,config):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    r=Path(c["run"]);assert read(r/"config.json")==c
    assert not (r/"all_checkpoints.pt").exists()
    lock=read(r/"protocol_lock.json")
    assert sha(config)==lock["config_sha"] and sha(__file__)==lock["source_sha"]
    device=torch.device(c["device"])
    assert torch.cuda.is_available()
    assert torch.cuda.mem_get_info(device)[0]>=c["gpu_memory_gib"]*2**30
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(c["gpu_memory_gib"]*2**30/torch.cuda.get_device_properties(device).total_memory,device)
    started=time.monotonic()
    save(r/"provenance.json",dict(git=git(),python=sys.version,torch=torch.__version__,numpy=np.__version__,
        config_sha=sha(config),source_sha=sha(__file__),device=str(device),
        gpu_name=torch.cuda.get_device_name(device),gpu_uuid=str(torch.cuda.get_device_properties(device).uuid),
        pretrained_LLM_loaded=False,source_commit=lock["git"],thread_count=2))
    protected=read(r/"protected_assets.json")
    for record in protected:assert sha(record["path"])==record["sha"],record["path"]
    tp=np.load(Path(c["teacher40"])/"teacher_predictions.npz")
    ts=torch.load(Path(c["teacher40"])/"teacher_checkpoints.pt",map_location="cpu",weights_only=True)
    teacher_selection=read(Path(c["teacher40"])/"teacher_selection.json")
    domain_data={};all_states={};curves=[];losses=[];preflight=[];teacher_checks=[];norm_records={};table_records={};time_splits={}
    specnames=list(c["arms"])
    for domain,spec in c["domains"].items():
        src=Path(spec["source"]);packs=np.load(src/"data.npz");cached=np.load(src/"predictions.npz")
        grid,raw,common,kind,h=calendar_arrays(read(src/"config.json"),domain);ids=kind*24+h
        # No calibration target can occur in selection history.
        fit_end=int(packs["fit_future"].max())
        selected=np.flatnonzero(packs["val_origin"]-96>fit_end)
        assert len(selected)>=9
        valblocks=np.array_split(np.arange(len(selected)),3)
        time_splits[domain]=dict(fit_windows=len(packs["fit_x"]),fit_target_end=str(grid[fit_end]),
            internal_selection_windows=len(selected),dropped_val_windows=int(len(packs["val_x"])-len(selected)),
            selection_indices=selected.tolist(),selection_origins=grid[packs["val_origin"][selected]].astype(str).tolist(),
            selection_block_sizes=[len(b) for b in valblocks],
            evaluation_split=spec["eval_split"],selection_target_after_fit=True,selection_history_after_fit_targets=True,
            development_only=True,historical_teacher_hyperparameters_used_old_val=True)
        fit=to_pack(packs,"fit",grid,raw,common,ids,device)
        val=to_pack(packs,"val",grid,raw,common,ids,device,selected)
        yf=torch.tensor(packs["fit_y"],dtype=torch.float32,device=device)
        yv=packs["val_y"][selected]
        profiles=read(Path(c["teacher40"])/domain/"profiles.json")
        ck=np.repeat(np.arange(3),24);ch=np.tile(np.arange(24),3)
        delta=knowledge(np.array(profiles["Task"]),ck,ch)-knowledge(np.array(profiles["Generic"]),ck,ch)
        fourier=np.stack([np.sin(10*np.pi*ch/24),np.cos(10*np.pi*ch/24),np.cos(12*np.pi*ch/24)],-1)
        perm=np.random.default_rng(61001).permutation(24)
        perm72=np.concatenate([k*24+perm for k in range(3)])
        original_tables=dict(Delta=delta,Fourier=fourier,ShuffledDelta=delta[perm72])
        tables={};table_records[domain]={}
        for key,value in original_tables.items():
            kt=value[ids[packs["fit_future"]]];km=kt.mean((0,1))
            rms=max(float(np.sqrt(np.mean((kt-km)**2))),1e-6)
            table=(value-km)/rms;tables[key]=torch.tensor(table,dtype=torch.float32,device=device)
            sv=np.linalg.svd(table-table.mean(0),compute_uv=False)
            table_records[domain][key]=dict(target=table.tolist(),mean=km.tolist(),rms=rms,
                fit_energy=float(np.mean(((kt-km)/rms)**2)),rank=int(np.sum(sv>sv[0]*1e-6)),
                normalized_target_rms=float(np.sqrt(np.mean(table**2))))
        table_records[domain]["shuffle_permutation"]=perm72.tolist()
        norms=read(src/"normalizers.json")
        data={};all_states[domain]={};norm_records[domain]={}
        for seed in c["seeds"]:
            base_state=torch.load(source_base(c,domain,seed),map_location="cpu",weights_only=True)
            initial=DirectStudent(seed,base_state,np.zeros(8),np.ones(8)).to(device)
            with torch.no_grad():bp=initial.numerical_prediction(fit["x"])
            base_cached=cached[f"{seed}_fit_FrozenTimeBase"]
            collapse_error=float(np.max(abs(bp.cpu().numpy()-base_cached)))
            assert collapse_error<3e-6
            numeric=numeric_summary(fit["x"],bp).detach().cpu().numpy()
            mu=numeric.mean((0,1));sd=np.maximum(numeric.std((0,1)),1e-3)
            norm_records[domain][seed]=dict(mean=mu.tolist(),sd=sd.tolist(),base_path=str(source_base(c,domain,seed)))
            initial=DirectStudent(seed,base_state,mu,sd).to(device)
            initstate=statecpu(initial);initp=pred(initial,fit)
            err=float(np.max(abs(initp-base_cached)));assert err<4e-6
            initval=pred(initial,val)
            # Removing/change of all calendar inputs cannot change initial forecast.
            fake=dict(val);fake["calendar"]=torch.zeros_like(val["calendar"]);fake["ids"]=(val["ids"]+17)%72
            assert np.array_equal(initval,pred(initial,fake))
            assert sum(p.numel() for p in initial.parameters())==995
            initial.train()
            ll,ss=initial.components(fit["x"][:2],fit["calendar"][:2],fit["ids"][:2])
            initial.zero_grad()
            ss.square().mean().backward()
            shape_body_grads=[v.grad for v in initial.body.parameters()]
            assert all(g is None or torch.count_nonzero(g)==0 for g in shape_body_grads)
            assert initial.affine.grad is None or torch.count_nonzero(initial.affine.grad)==0
            assert all(v.grad is None or torch.count_nonzero(v.grad)==0 for v in initial.calendar_level.parameters())
            initial.zero_grad(set_to_none=True)
            ll,ss=initial.components(fit["x"][:2],fit["calendar"][:2],fit["ids"][:2])
            (ll-yf[:2].mean(1,keepdim=True)).square().mean().backward()
            bodygrad=sum(float(v.grad.square().sum()) for v in initial.body.parameters() if v.grad is not None)**.5
            assert bodygrad>0
            levelgrad=float(initial.affine.grad[1].abs());assert levelgrad>0
            assert float(initial.calendar_level.weight.grad.abs().sum())>0
            initial.zero_grad(set_to_none=True)
            preflight.append(dict(domain=domain,seed=seed,collapse_error=collapse_error,initial_base_error=err,
                initial_calendar_invariant=True,parameters=995,trainable=995,body_gradient=bodygrad,
                level_gradient=levelgrad,no_additive_forecast_skip=True,trainable_numeric_level=True,calendar_can_change_level=True,Qwen_codes_cannot_change_level=True))
            frozen_ref=cached[f"{seed}_val_FrozenTimeBase"][selected]
            teacher_targets={}
            for key in ["Delta","Fourier","ShuffledDelta"]:
                if key!="ShuffledDelta":
                    target=tp[f"{domain}_{seed}_fit_{key}"]
                    assert np.array_equal(packs["fit_y"],tp[f"{domain}_{seed}_fit_y"])
                else:
                    pp=norms[str(seed)]["Fourier"];ix=packs["fit_future"]
                    ft=np.concatenate([numeric,raw[ix],common[ix]],-1)
                    z39=(ft-np.array(pp["mean"]))/np.array(pp["sd"])
                    z=np.concatenate([z39,tables[key][fit["ids"]].detach().cpu().numpy()],-1)
                    teacher=Head(seed);teacher.load_state_dict(ts[domain][seed]["Delta"][teacher_selection[domain]["Delta"]]);teacher.eval()
                    with torch.no_grad():target=base_cached+teacher(torch.tensor(z,dtype=torch.float32)).numpy()
                teacher_targets[key]=torch.tensor(target,dtype=torch.float32,device=device).detach()
            # Reproduce normal teacher to prove target semantics/cache alignment.
            for key in ["Delta","Fourier"]:
                pp=norms[str(seed)]["Fourier"];ix=packs["fit_future"]
                oldnum=numeric_summary(torch.tensor(packs["fit_x"]),torch.tensor(base_cached,dtype=torch.float32)).numpy()
                ft=np.concatenate([oldnum,raw[ix],common[ix]],-1)
                z39=(ft-np.array(pp["mean"]))/np.array(pp["sd"])
                z3=tables[key][fit["ids"]].detach().cpu().numpy()
                head=Head(seed);head.load_state_dict(ts[domain][seed][key][teacher_selection[domain][key]])
                with torch.no_grad():replay=base_cached+head(torch.tensor(np.concatenate([z39,z3],-1),dtype=torch.float32)).numpy()
                gap=float(abs(replay-tp[f"{domain}_{seed}_fit_{key}"]).max())
                assert gap<3e-6
                teacher_checks.append(dict(domain=domain,seed=seed,target=key,replay_error=gap))
            data[seed]=dict(base_state=base_state,mu=mu,sd=sd,initial=initstate,initial_val=initval,
                norm=scores(frozen_ref,yv),teacher_targets=teacher_targets)
            all_states[domain][seed]={}
            del initial
        domain_data[domain]=dict(packs=packs,cached=cached,grid=grid,raw=raw,common=common,ids=ids,
            fit=fit,val=val,selected=selected,blocks=valblocks,yfit=yf,yval=yv,data=data,tables=tables)
    save(r/"preflight.json",preflight);save(r/"teacher_checks.json",teacher_checks)
    save(r/"numeric_normalizers.json",norm_records);save(r/"knowledge_tables.json",table_records);save(r/"time_splits.json",time_splits)
    for domain,ds in domain_data.items():
        for seed,d in ds["data"].items():
            batches=np.random.default_rng(seed).integers(0,len(ds["yfit"]),size=(c["steps"],c["batch"]))
            for arm,spec in c["arms"].items():
                model=DirectStudent(seed,d["base_state"],d["mu"],d["sd"]).to(device);model.load_state_dict(d["initial"])
                assert np.array_equal(pred(model,ds["val"]),d["initial_val"])
                others=[p for name,p in model.named_parameters() if name!="codes.weight"]
                opt=torch.optim.Adam([dict(params=others,lr=c["lr"]),dict(params=model.codes.parameters(),lr=c["map_lr"])])
                all_states[domain][seed][arm]={}
                all72=torch.arange(72,device=device)
                for step in range(c["steps"]+1):
                    if time.monotonic()-started>c["seconds_cap"]:raise TimeoutError("bounded direct-output pilot time cap")
                    if step in c["checkpoints"]:
                        all_states[domain][seed][arm][step]=statecpu(model)
                        pv=pred(model,ds["val"])
                        for bi,block in enumerate(ds["blocks"]):
                            curves.append(dict(domain=domain,seed=seed,arm=arm,step=step,block=bi,
                                               windows=len(block),**scores(pv[block],ds["yval"][block])))
                    if step==c["steps"]:break
                    ib=torch.tensor(batches[step],device=device)
                    model.train()
                    level,forecast_shape=model.components(ds["fit"]["x"][ib],ds["fit"]["calendar"][ib],ds["fit"]["ids"][ib])
                    p=level+forecast_shape
                    sup_level=(level-ds["yfit"][ib].mean(1,keepdim=True)).square().mean()
                    sup_shape=(forecast_shape-center(ds["yfit"][ib])).square().mean()
                    sup=sup_level+sup_shape
                    # Identical computations/optimizer steps; NoKD auxiliary weights zero.
                    table_target=ds["tables"][spec["representation"] or "Delta"]
                    rep=(model.codes(all72)-table_target).square().mean()
                    target=ds["data"][seed]["teacher_targets"][spec["shape"] or "Delta"][ib]
                    shape=(forecast_shape-center(target)).square().mean()
                    assert abs(float((sup-(p-ds["yfit"][ib]).square().mean()).detach()))<1e-6
                    loss=sup+c["representation_weight"]*int(spec["representation"] is not None)*rep+c["shape_weight"]*int(spec["shape"] is not None)*shape
                    assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True);loss.backward()
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    opt.step()
                    if (step+1)%100==0:
                        losses.append(dict(domain=domain,seed=seed,arm=arm,step=step+1,
                            supervised=float(sup.detach()),level_loss=float(sup_level.detach()),shape_loss=float(sup_shape.detach()),representation=float(rep.detach()),shape=float(shape.detach()),loss=float(loss.detach())))
                changed=any(not torch.equal(model.state_dict()[k].detach().cpu(),d["initial"][k]) for k in d["initial"] if k.startswith("body."))
                assert changed
                print("FIT63",domain,seed,arm,"steps",c["steps"],"body_changed",changed,flush=True)
                del model,opt
    independence=[]
    for domain in domain_data:
        for seed in c["seeds"]:
            names=[k for k in all_states[domain][seed]["DirectNoKD"][0] if k.startswith(("body.","calendar_level.")) or k=="affine"]
            for step in c["checkpoints"]:
                ref=all_states[domain][seed]["DirectNoKD"][step]
                maxdiff=max(float((all_states[domain][seed][arm][step][k]-ref[k]).abs().max())
                            for arm in c["arms"] for k in names)
                assert maxdiff==0,(domain,seed,step,maxdiff)
                independence.append(dict(domain=domain,seed=seed,step=step,max_body_level_parameter_difference=maxdiff))
    save(r/"level_parameter_independence.json",independence)
    selection={};selected_states={}
    for domain,ds in domain_data.items():
        selection[domain]={};selected_states[domain]={s:{} for s in c["seeds"]}
        for arm in c["arms"]:
            options=[]
            for step in c["checkpoints"]:
                risks=[]
                for seed in c["seeds"]:
                    row=[]
                    for bi,block in enumerate(ds["blocks"]):
                        ref=scores(ds["cached"][f"{seed}_val_FrozenTimeBase"][ds["selected"]][block],ds["yval"][block])
                        q=next(v for v in curves if (v["domain"],v["seed"],v["arm"],v["step"],v["block"])==(domain,seed,arm,step,bi))
                        row.append(.5*(q["mse"]/ref["mse"]+q["mae"]/ref["mae"]))
                    risks.append(row)
                risks=np.array(risks)
                options.append(dict(step=step,score=float((risks.mean()+risks.mean(1).max()+risks.mean(0).max())/3)))
            chosen=min(options,key=lambda v:(v["score"],v["step"]))["step"]
            selection[domain][arm]=dict(step=chosen,options=options)
            for seed in c["seeds"]:selected_states[domain][seed][arm]=all_states[domain][seed][arm][chosen]
    save(r/"selection.json",selection);save(r/"curves.json",curves);save(r/"losses.json",losses)
    torch.save(all_states,r/"all_checkpoints.pt");torch.save(selected_states,r/"students.pt")
    evlock=dict(selection_sha=sha(r/"selection.json"),students_sha=sha(r/"students.pt"),source_sha=sha(__file__),
        before_outer_arrays_accessed=True,outer_previously_viewed=True,independent_confirmation=False)
    save(r/"evaluation_lock.json",evlock)
    # No evaluation array read until this point, after all30 fits and selections.
    rows=[];windowrows=[];predictions={};replaychecks=[];fixed600=[];paired=[];matched=[]
    for domain,ds in domain_data.items():
        spec=c["domains"][domain];ev=to_pack(ds["packs"],spec["eval_split"],ds["grid"],ds["raw"],ds["common"],ds["ids"],device)
        ye=ds["packs"][spec["eval_split"]+"_y"]
        subsets={"internal":(ds["val"],ds["yval"]),"development":(ev,ye)}
        for label,(pack,y) in subsets.items():
            predictions[f"{domain}_{label}_y"]=y;predictions[f"{domain}_{label}_origin"]=pack["origin"]
        for seed,d in ds["data"].items():
            for arm in c["arms"]:
                model=DirectStudent(seed,d["base_state"],d["mu"],d["sd"]).to(device);model.load_state_dict(selected_states[domain][seed][arm])
                for label,(pack,y) in subsets.items():
                    p=pred(model,pack);predictions[f"{domain}_{seed}_{label}_{arm}"]=p
                    met=scores(p,y);assert abs(met["mse"]-met["level"]-met["shape"])<1e-12
                    rows.append(dict(domain=domain,seed=seed,split=label,arm=arm,step=selection[domain][arm]["step"],**met))
                    if label=="development":
                        for j in range(len(y)):
                            windowrows.append(dict(domain=domain,seed=seed,arm=arm,
                                origin=str(ds["grid"][pack["origin"][j]]),**scores(p[j:j+1],y[j:j+1])))
                reload=DirectStudent(seed,d["base_state"],d["mu"],d["sd"]).to(device)
                disk=torch.load(r/"students.pt",map_location="cpu",weights_only=True)
                reload.load_state_dict(disk[domain][seed][arm]);rp=pred(reload,ev);pp=pred(model,ev)
                replay_error=float(abs(rp-pp).max());assert replay_error==0
                initial=d["initial"]
                mapchange=float((model.codes.weight.detach().cpu()-initial["codes.weight"]).abs().max())
                bodychange=max(float((model.state_dict()[k].detach().cpu()-initial[k]).abs().max()) for k in initial if k.startswith("body."))
                with torch.no_grad():
                    numpred=model.level_prediction(ev["x"],ev["calendar"]).cpu().numpy()
                mean_difference=float(abs((pp-numpred).mean(1)).max())
                assert mean_difference<2e-6
                replaychecks.append(dict(domain=domain,seed=seed,arm=arm,reload_error=replay_error,
                    map_parameter_change=mapchange,body_parameter_change=bodychange,
                    forecast_vs_calendar_numeric_level_difference=mean_difference))
                model.load_state_dict(all_states[domain][seed][arm][600])
                fixed600.append(dict(domain=domain,seed=seed,arm=arm,step=600,**scores(pred(model,ev),ye)))
                # Output-KD effect also measured at identical representation-only checkpoint.
                commonstep=selection[domain]["QwenRepresentation"]["step"]
                if arm in ["QwenRepresentation","QwenRepresentationShapeKD"]:
                    model.load_state_dict(all_states[domain][seed][arm][commonstep])
                    matched.append(dict(domain=domain,seed=seed,arm=arm,step=commonstep,**scores(pred(model,ev),ye)))
                del model,reload
        for arm in ["DirectNoKD","QwenRepresentation","FourierRepresentationShapeKD","ShuffleRepresentationShapeKD"]:
            for metric in ["mse","mae"]:
                diffs=[]
                for seed in c["seeds"]:
                    q=predictions[f"{domain}_{seed}_development_QwenRepresentationShapeKD"]-ye
                    b=predictions[f"{domain}_{seed}_development_{arm}"]-ye
                    diffs.append(((q*q-b*b) if metric=="mse" else (abs(q)-abs(b))).mean((1,2)))
                paired.append(dict(domain=domain,reference=arm,metric=metric,**interval(np.mean(diffs,axis=0),61002)))
    save(r/"metrics.json",rows);writecsv(r/"metrics.csv",rows);writecsv(r/"window_metrics.csv",windowrows)
    save(r/"fixed600_metrics.json",fixed600);save(r/"matched_checkpoint_outputKD.json",matched)
    save(r/"paired_intervals.json",paired);np.savez_compressed(r/"predictions.npz",**predictions)
    summary={domain:{label:{arm:{k:float(np.mean([v[k] for v in rows if (v["domain"],v["split"],v["arm"])==(domain,label,arm)])) for k in ["mse","mae","level","shape"]} for arm in c["arms"]} for label in ["internal","development"]} for domain in c["domains"]}
    save(r/"summary.json",summary)
    protected_error=[]
    for record in protected:
        if sha(record["path"])!=record["sha"]:protected_error.append(record["path"])
    assert not protected_error
    check22(c)
    assert sha(r/"selection.json")==evlock["selection_sha"] and sha(r/"students.pt")==evlock["students_sha"]
    torch.cuda.synchronize(device)
    save(r/"verification.json",dict(protected_assets=len(protected),protected_changes=protected_error,
        fixed22_unchanged=True,source51_unchanged=True,no_LLM_at_inference=True,all_predictions_finite=all(np.isfinite(a).all() for a in predictions.values()),
        all995_parameters_trainable=True,all_body_weights_updated=all(q["body_parameter_change"]>0 for q in replaychecks),
        reload_max_error=max(q["reload_error"] for q in replaychecks),checks=replaychecks))
    save(r/"status.json",dict(complete=True,seconds=time.monotonic()-started,model_fits=len(c["domains"])*len(c["seeds"])*len(c["arms"]),updates_each=c["steps"],
        total_updates=len(c["domains"])*len(c["seeds"])*len(c["arms"])*c["steps"],parameters=995,device=str(device),peak_GPU_GiB=torch.cuda.max_memory_allocated(device)/2**30,
        old22_control_fits=0,old51_fits=0,LLM_calls=0,new_model_downloads=0,independent_confirmation=False))
    print("SUMMARY63",json.dumps(summary),flush=True);print("STATUS63",json.dumps(read(r/"status.json")),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args();config=read(args.config)
    try:
        if args.prepare:prepare(config,args.config)
        else:run(config,args.config)
    except Exception:
        r=Path(config["run"])
        if r.exists():(r/("failure_"+str(time.time_ns())+".log")).write_text(traceback.format_exc())
        raise
