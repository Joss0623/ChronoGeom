"""Analytical response anchor + residual Qwen response KD; common Fourier warm start."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
import argparse,json,time,shutil,traceback
from pathlib import Path
import numpy as np
import torch
from exp import exp_chronogeom_phase_bilinear14 as bil
p13=bil.prior;old=bil.old;rt=bil.rt;csvsave=bil.csvsave
CONFIG=Path("docs/chronogeom/phase_opd17_config.json")
HIDX=bil.HIDX;ZIDX=bil.ZIDX
MAP=dict(Fourier="SeasonFourier32",Task="QwenTask",Wrong="QwenWrong",Shuffle="QwenTrainShuffle")
def teacher_labels(p,c):
    grid=np.array(c["angles_hours"])*p13.W
    preds=np.stack([p13.rotate(p,np.full(len(p["b"]),a)) for a in grid],1)
    mse=((preds-p["y"][:,None])**2).mean(2);mae=abs(preds-p["y"][:,None]).mean(2)
    ms=float(p["base_mse"].mean());ma=float(abs(p["b"]-p["y"]).mean());assert ms>0 and ma>0
    keep=np.arange(len(grid))!=len(grid)//2
    yy=np.c_[((mse-mse[:,len(grid)//2,None])/ms)[:,keep],((mae-mae[:,len(grid)//2,None])/ma)[:,keep]]
    joint=.5*((mse-mse[:,len(grid)//2,None])/ms+(mae-mae[:,len(grid)//2,None])/ma)
    return yy,ms,ma,max(float(np.median(np.ptp(joint,axis=1))),1e-4)

def plain_response_fit(x,p,lam,c):
    yy,ms,ma,temp=teacher_labels(p,c);mu=x.mean(0);sd=np.maximum(x.std(0),.001)
    xx=np.c_[np.ones(len(x)),(x-mu)/sd];reg=np.eye(xx.shape[1])*lam;reg[0,0]=0
    coef=np.linalg.solve(xx.T@xx/len(x)+reg,xx.T@yy/len(x))
    assert coef.size==1312
    return dict(mu=mu,sd=sd,coef=coef,mse_scale=np.array(ms),mae_scale=np.array(ma),temperature=np.array(temp))

def plain_response_grid(m,x):
    v=np.c_[np.ones(len(x)),(x-m["mu"])/m["sd"]]@m["coef"]
    mse=np.insert(v[:,:8],4,0,axis=1);mae=np.insert(v[:,8:],4,0,axis=1)
    return .5*(mse+mae)


def analytical_x(x):
    a=x[:,:47];state=24*a[:,16]+a[:,17];annual=np.arctan2(a[:,20],a[:,21]);parts=[]
    for period in [24.,168.]:
        for k in range(1,5):parts.extend([np.sin(2*np.pi*k*state/period),np.cos(2*np.pi*k*state/period)])
    for k in range(1,9):parts.extend([np.sin(k*annual),np.cos(k*annual)])
    return np.c_[a,np.stack(parts,1),x[:,79:81]]

def response_fit(x,p,lam,c):
    # Shared analytical anchor; matched residual heads explain its response errors.
    ax=analytical_x(x);yy,ms,ma,temp=teacher_labels(p,c)
    anchor=plain_response_fit(ax,p,c["anchor_regularizer"],c)
    oo=np.zeros_like(yy);idx=np.arange(len(x))
    for hold in np.array_split(idx,3):
        fit=idx[(idx+4<=hold[0])|(idx>=hold[-1]+4)];assert len(fit)>=4
        mm=plain_response_fit(ax[fit],{k:v[fit] for k,v in p.items()},c["anchor_regularizer"],c)
        value=np.c_[np.ones(len(hold)),(ax[hold]-mm["mu"])/mm["sd"]]@mm["coef"]
        value[:,:8]*=float(mm["mse_scale"])/ms;value[:,8:]*=float(mm["mae_scale"])/ma
        oo[hold]=value
    result=plain_response_fit(x,p,lam,c);xx=np.c_[np.ones(len(x)),(x-result["mu"])/result["sd"]]
    reg=np.eye(82)*lam;reg[0,0]=0
    result["coef"]=np.linalg.solve(xx.T@xx/len(x)+reg,xx.T@(yy-oo)/len(x))
    result.update(anchor_mu=anchor["mu"],anchor_sd=anchor["sd"],anchor_coef=anchor["coef"])
    return result

def response_grid(m,x):
    anchor=dict(mu=m["anchor_mu"],sd=m["anchor_sd"],coef=m["anchor_coef"])
    return plain_response_grid(anchor,analytical_x(x))+plain_response_grid(m,x)

def plain_response_cv(x,p,origins,c):
    cost=np.zeros((len(x),9));temp=np.zeros(len(x))
    for hold in np.array_split(np.arange(len(x)),3):
        lo=origins[hold].min();hi=origins[hold].max()+96
        fit=np.where((origins+96<=lo)|(origins>=hi))[0];assert len(fit)>=8
        model=plain_response_fit(x[fit],{k:v[fit] for k,v in p.items()},c["anchor_regularizer"],c)
        cost[hold]=plain_response_grid(model,x[hold]);temp[hold]=float(model["temperature"])
    return cost,temp


def calibrate_response_temperature(cost,temp,p,c):
    yy,_,_,_=teacher_labels(p,c)
    truth=.5*(np.insert(yy[:,:8],4,0,axis=1)+np.insert(yy[:,8:],4,0,axis=1))
    labels=truth.argmin(1)
    spread=np.ptp(truth,axis=1);weights=np.clip(spread/max(float(spread.mean()),1e-12),.1,10.)
    scores=[]
    for factor in c["temperature_factors"]:
        logits=-cost/(temp[:,None]*factor)
        logits-=logits.max(1,keepdims=True)
        logp=logits-np.log(np.exp(logits).sum(1,keepdims=True))
        scores.append((float(np.average(-logp[np.arange(len(labels)),labels],weights=weights)),factor))
    best=min(scores,key=lambda a:(a[0],-a[1]))
    assert best[0]<=next(a[0] for a in scores if a[1]==1)+1e-12
    return temp*best[1],dict(factor=best[1],nll=best[0],grid=[dict(factor=f,nll=n) for n,f in scores],
        scope="same-training-period purged response-head OOF only; not full-pipeline nested OOF")

def response_cv(x,p,origins,lam,c):
    cost=np.zeros((len(x),9));temp=np.zeros(len(x));blocks=[]
    for hold in np.array_split(np.arange(len(x)),3):
        lo=origins[hold].min();hi=origins[hold].max()+96
        fit=np.where((origins+96<=lo)|(origins>=hi))[0];assert len(fit)>=8
        model=response_fit(x[fit],{k:v[fit] for k,v in p.items()},lam,c)
        cost[hold]=response_grid(model,x[hold]);temp[hold]=float(model["temperature"])
        blocks.append(dict(held=hold.tolist(),fit=fit.tolist(),target_overlap=False))
    temp,calibration=calibrate_response_temperature(cost,temp,p,c)
    blocks.append(dict(temperature_calibration=calibration))
    return cost,temp,blocks

def joint_metrics(p,angle,ms,ma):
    pred=p13.rotate(p,angle)
    mse=float(np.mean((pred-p["y"])**2));mae=float(np.mean(abs(pred-p["y"])))
    return mse,mae,.5*(mse/ms+mae/ma)

def choose(rows):
    out={}
    for arm in sorted(set(v["arm"] for v in rows)):
        scores=[]
        for lam in sorted(set(v["lam"] for v in rows)):
            a=[v for v in rows if v["arm"]==arm and v["lam"]==lam]
            assert len(a)==9
            scores.append(dict(lam=lam,joint=float(np.mean([v["joint"] for v in a]))))
        winner=min(scores,key=lambda v:(v["joint"],-v["lam"]))
        out[arm]=dict(**winner,grid=scores)
    return out

def query(cost,angles,c):
    grid=torch.tensor(c["angles_hours"],dtype=torch.float64)*p13.W
    position=((angles-grid[0])/(grid[1]-grid[0])).clamp(0,8)
    idx=torch.floor(position).long().clamp(max=7);frac=position-idx
    return torch.gather(cost,1,idx)*(1-frac)+torch.gather(cost,1,idx+1)*frac

def fit_student(x,p,lam,c,seed,mode="NoKD",feedback=None,initial=None):
    mu=x.mean(0);sd=np.maximum(x.std(0),.001);xx=torch.tensor((x-mu)/sd,dtype=torch.float64)
    hh=xx[:,HIDX];zz=xx[:,ZIDX];rng=np.random.default_rng(seed+140014)
    init=rng.normal(size=49);init/=np.linalg.norm(init)
    w=torch.zeros(81,dtype=torch.float64,requires_grad=True);bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    left=torch.tensor(init,dtype=torch.float64,requires_grad=True);right=torch.zeros(32,dtype=torch.float64,requires_grad=True)
    if initial is not None:
        assert abs(initial["mu"]-mu).max()<1e-10 and abs(initial["sd"]-sd).max()<1e-10
        with torch.no_grad():
            w.copy_(torch.tensor(initial["coef"]));bias.copy_(torch.tensor(initial["bias"]))
            left.copy_(torch.tensor(initial["left"]));right.copy_(torch.tensor(initial["right"]))
    opt=torch.optim.Adam([w,bias,left,right],lr=c["learning_rate"])
    base=torch.tensor(p["b"]);uu=torch.tensor(p["u"]);vv=torch.tensor(p["v"]);target=torch.tensor(p["y"])
    ms=float(p["base_mse"].mean());ma=float(abs(p["b"]-p["y"]).mean())
    noise_rng=torch.Generator().manual_seed(seed+150015)
    if feedback is not None:
        cost=torch.tensor(feedback[0]);temp=torch.tensor(feedback[1])[:,None]
    trace=[];queries=0
    sigma=c["policy_sigma_hours"]*p13.W
    for step in range(1,c["steps"]+1):
        opt.zero_grad()
        delta=c["delta_max"]*torch.tanh(xx@w+bias+(hh@left)*(zz@right))
        pred=base+uu*(torch.cos(delta)[:,None]-1)+vv*torch.sin(delta)[:,None]
        mse=(pred-target).square().mean();mae=(pred-target).abs().mean()
        supervised=.5*(mse/ms+mae/ma)
        kd=torch.zeros((),dtype=torch.float64)
        if mode!="NoKD" and step>c["warmup_steps"]:
            with torch.no_grad():
                if mode=="StaticKD":
                    angles=(torch.tensor(c["angles_hours"],dtype=torch.float64)*p13.W)[None].repeat(len(x),1)
                else:
                    # Fresh student-dependent local states, not a stale fixed response target.
                    center=delta.detach()+sigma*torch.randn(len(x),generator=noise_rng,dtype=torch.float64)
                    offsets=torch.tensor([-1.,0.,1.],dtype=torch.float64)*sigma
                    angles=(center[:,None]+offsets).clamp(-c["delta_max"],c["delta_max"])
                logqt=torch.log_softmax((-query(cost,angles,c)/temp).clamp(-30,30),dim=1)
                queries+=angles.numel()
            logps=torch.log_softmax(-.5*((angles-delta[:,None])/sigma).square(),dim=1)
            ps=logps.exp();kd=(ps*(logps-logqt)).sum(1).mean()
        if initial is None:
            penalty=w.square().sum()+left.square().sum()*right.square().sum()
        else:
            w0=torch.tensor(initial["coef"]);a0=torch.tensor(initial["left"]);c0=torch.tensor(initial["right"])
            matrix_change=left.square().sum()*right.square().sum()+a0.square().sum()*c0.square().sum()-2*(left@a0)*(right@c0)
            penalty=(w-w0).square().sum()+matrix_change+(bias-float(initial["bias"])).square()
        loss=supervised+lam*penalty+c["kd_weight"]*kd
        loss.backward()
        if step==1:assert float(right.grad.norm())>0
        opt.step()
        if step in [1,25,50,51,75,100]:
            trace.append(dict(step=step,mse=float(mse.detach()),mae=float(mae.detach()),joint=float(supervised.detach()),
                reverse_kl=float(kd.detach()),phase_mean_hours=float((delta.detach()/p13.W).mean()),
                phase_sd_hours=float((delta.detach()/p13.W).std())))
    m=dict(mu=mu,sd=sd,coef=w.detach().numpy(),bias=bias.detach().numpy(),left=left.detach().numpy(),right=right.detach().numpy())
    assert bil.count(m)==163
    return m,trace,queries

def select(c,r):
    r.mkdir(parents=True,exist_ok=False);rt.save(r/"config.json",c);shutil.copy2(__file__,r/Path(__file__).name)
    tick=time.monotonic();torch.set_num_threads(2)
    try:
        data,h=p13.training_data(c,r);src=rt.read(r/"source_train.json")
        for seed in c["seeds"]:
            assert np.diff(data[seed,"train"]["origin"]).min()>=24
            f=Path(c["warm_source"])/f"student_s{seed}_Fourier_StaticKD.npz";src["sources"][str(f)]=rt.sha(f)
        f=Path(c["warm_source"])/"selection_lock.json";src["sources"][str(f)]=rt.sha(f)
        for f in [Path(__file__),Path(bil.__file__),Path(p13.__file__),CONFIG]:src["sources"][str(f)]=rt.sha(f)
        rt.save(r/"source_train.json",src)
        folds,meta=p13.make_folds(c,data,h);rt.save(r/"crossfit_inner.json",meta)
        for f in folds.values():
            assert abs(analytical_x(f["mat"]["train"]["SeasonFourier32"])-f["mat"]["train"]["SeasonFourier32"]).max()<1e-10
        records=[]
        for (seed,name),f in folds.items():
            pv=f["valid"];ms=float(pv["base_mse"].mean());ma=float(abs(pv["b"]-pv["y"]).mean())
            for arm in c["teacher_arms"]:
                for lam in c["lambdas"]:
                    m=response_fit(f["mat"]["train"][arm],f["train"],lam,c)
                    cost=response_grid(m,f["mat"]["inner"][arm]);angle=np.array(c["angles_hours"])[cost.argmin(1)]*p13.W
                    mse,mae,joint=joint_metrics(pv,angle,ms,ma)
                    records.append(dict(seed=seed,fold=name,arm=arm,lam=lam,mse=mse,mae=mae,joint=joint))
        tc=choose(records);csvsave(r/"teacher_inner.csv",records);rt.save(r/"teacher_choice.json",tc)
        feedback={};cfmeta=[]
        for (seed,name),f in folds.items():
            feedback[seed,name]={}
            for arm in c["teacher_arms"]:
                costs,temperatures,blocks=response_cv(f["mat"]["train"][arm],f["train"],data[seed,"train"]["origin"][f["ii"]],tc[arm]["lam"],c)
                feedback[seed,name][arm]=(costs,temperatures);cfmeta.append(dict(seed=seed,fold=name,arm=arm,blocks=blocks))
        rt.save(r/"teacher_crossfit_inner.json",cfmeta);records=[];queries=0
        for (seed,name),f in folds.items():
            ms=float(f["valid"]["base_mse"].mean());ma=float(abs(f["valid"]["b"]-f["valid"]["y"]).mean())
            warm_feedback=plain_response_cv(f["mat"]["train"]["SeasonFourier32"],f["train"],data[seed,"train"]["origin"][f["ii"]],c)
            warmcfg=dict(c,warmup_steps=50)
            initial,_,_=fit_student(f["mat"]["train"]["SeasonFourier32"],f["train"],c["warm_regularizer"],warmcfg,seed,"StaticKD",warm_feedback)
            for arm in c["student_arms"]:
                mode="NoKD" if arm=="NoKD" else arm.split("_")[1]
                teach=None if arm=="NoKD" else MAP[arm.split("_")[0]]
                for lam in c["lambdas"]:
                    model,_,count=fit_student(f["mat"]["train"]["SeasonFourier32"],f["train"],lam,c,seed,mode,
                        None if teach is None else feedback[seed,name][teach],initial=initial)
                    queries+=count;delta=bil.predict(model,f["mat"]["inner"]["SeasonFourier32"],c)
                    mse,mae,joint=joint_metrics(f["valid"],delta,ms,ma)
                    records.append(dict(seed=seed,fold=name,arm=arm,lam=lam,mse=mse,mae=mae,joint=joint))
        sc=choose(records);csvsave(r/"student_inner.csv",records);rt.save(r/"student_choice.json",sc)
        routes={}
        for name in MAP:
            options=[name+"_StaticKD",name+"_OPD"];routes[name]=min(options,key=lambda x:(sc[x]["joint"],x))
        rt.save(r/"route_choice.json",routes);curves=[];fullcf=[]
        for seed in c["seeds"]:
            q=data[seed,"train"];ii=np.arange(len(q["origin"]));b,cf=p13.crossfit_base(q,ii,c);p=p13.pack(b,q["y"])
            mat,pca=old.matrices(c,data,h,seed,ii,dict(train=ii));mat=mat["train"]
            for arm in mat:mat[arm]=np.c_[mat[arm],p["coords"]]
            bm=p13.base_fit(q,ii,c);ref=np.load(Path(c["reuse_phase_cache"])/f"teacher_s{seed}_SeasonFourier32.npz")
            assert max(float(abs(bm[k]-ref[k]).max()) for k in bm)<1e-10
            np.savez_compressed(r/f"frozen_base_s{seed}.npz",**bm)
            for arm,(mu,w,info) in pca.items():np.savez_compressed(r/f"pca_s{seed}_{arm}.npz",mu=mu,w=w)
            initial=np.load(Path(c["warm_source"])/f"student_s{seed}_Fourier_StaticKD.npz")
            fb={}
            for arm in c["teacher_arms"]:
                model=response_fit(mat[arm],p,tc[arm]["lam"],c);np.savez_compressed(r/f"teacher_s{seed}_{arm}.npz",**model)
                cost,temperature,blocks=response_cv(mat[arm],p,q["origin"],tc[arm]["lam"],c);fb[arm]=(cost,temperature)
                fullcf.append(dict(seed=seed,arm=arm,blocks=blocks))
            for arm in c["student_arms"]:
                mode="NoKD" if arm=="NoKD" else arm.split("_")[1];teach=None if arm=="NoKD" else MAP[arm.split("_")[0]]
                model,trace,count=fit_student(mat["SeasonFourier32"],p,sc[arm]["lam"],c,seed,mode,None if teach is None else fb[teach],initial=initial)
                queries+=count;np.savez_compressed(r/f"student_s{seed}_{arm}.npz",**model)
                curves.append(dict(seed=seed,arm=arm,trace=trace))
        rt.save(r/"curves.json",curves);rt.save(r/"teacher_crossfit_full.json",fullcf)
        assert all(rt.sha(p)==v for p,v in src["sources"].items())
        assert time.monotonic()-tick<c["budget"]["max_fit_seconds"]
        rt.save(r/"selection_lock.json",dict(files={str(p):rt.sha(p) for p in r.iterdir() if p.is_file()},seconds=time.monotonic()-tick,
            response_action_queries=queries,routes=routes,no_outer_files_opened=True,LLM_forward_calls=0))
        print("SELECTED",json.dumps(dict(teacher={a:v["lam"] for a,v in tc.items()},student={a:v["lam"] for a,v in sc.items()},routes=routes)),flush=True)
    except BaseException:rt.save(r/"failure_select.json",dict(traceback=traceback.format_exc()));raise

def evaluate(c,r):
    tick=time.monotonic();lock=rt.read(r/"selection_lock.json");assert all(rt.sha(p)==v for p,v in lock["files"].items())
    pc=rt.read(Path(c["reliability_run"])/"config.json");cache=Path(pc["reuse_run"])
    sf=Path(c["reliability_run"])/"shared_summaries.json";sources={str(sf):rt.sha(sf)}
    sums={(v["seed"],v["stage"],v["row"]):v["summary"] for v in rt.read(sf)}
    rows=[];cases=[];outputs={};conservation={};teachers=[]
    for seed in c["seeds"]:
        bm=np.load(r/f"frozen_base_s{seed}.npz")
        for stage in c["stages"]:
            f=cache/f"seed{seed}_{stage}.npz";z=np.load(f);q={k:z[k] for k in z.files};sources[str(f)]=rt.sha(f)
            q["features"]=old.annual_features(np.stack([p13.s.flatten(sums[seed,stage,j]) for j in range(len(q["origin"]))]),q["origin"],c)
            b=p13.base_predict(q,np.arange(len(q["origin"])),bm);p=p13.pack(b,q["y"]);x=np.c_[p13.base_phi(q),p["coords"]]
            np.savez_compressed(r/f"deployment_s{seed}_{stage}.npz",x=x,b=b,u=p["u"],v=p["v"],y=q["y"],origin=q["origin"])
            angles=dict(FrozenBase=np.zeros(len(x)),FrozenFourierKD=bil.predict(np.load(Path(c["warm_source"])/f"student_s{seed}_Fourier_StaticKD.npz"),x,c))
            for arm in c["student_arms"]:angles[arm]=bil.predict(np.load(r/f"student_s{seed}_{arm}.npz"),x,c)
            for arm,angle in angles.items():
                pred,met,check=p13.forecast_metrics(p,angle)
                for k,v in check.items():conservation[k]=max(conservation.get(k,0.),v)
                outputs[f"s{seed}_{stage}_{arm}"]=pred;outputs[f"a{seed}_{stage}_{arm}"]=angle
                rows.append(dict(seed=seed,stage=stage,arm=arm,n=len(x),
                    **{k:float(met[k].mean()) for k in ["mse","mae","level","shape"]},
                    phase_mean_hours=float((angle/p13.W).mean()),phase_sd_hours=float((angle/p13.W).std())))
                gain=p["base_mse"]-met["mse"]
                for j in range(len(x)):
                    cases.append(dict(seed=seed,stage=stage,arm=arm,origin=int(q["origin"][j]),mse=float(met["mse"][j]),mae=float(met["mae"][j]),
                        gain=float(gain[j]),shift_hours=float(angle[j]/p13.W)))
            hm={"SeasonFourier32":x}
            for arm in c["contexts"]:
                f=Path(c["reuse_phase_cache"])/"evaluation_extraction"/f"hidden_{seed}_{arm}_{stage}.npy";h=np.load(f).astype(float);sources[str(f)]=rt.sha(f)
                basis=np.load(r/f"pca_s{seed}_{arm}.npz");hm[arm]=np.c_[q["features"],(h-basis["mu"])@basis["w"].T,p["coords"]]
            hm["QwenTrainShuffle"]=hm["QwenTask"]
            for arm in c["teacher_arms"]:
                model=np.load(r/f"teacher_s{seed}_{arm}.npz");cost=response_grid(model,hm[arm]);angle=np.array(c["angles_hours"])[cost.argmin(1)]*p13.W
                _,met,_=p13.forecast_metrics(p,angle)
                teachers.append(dict(seed=seed,stage=stage,arm=arm,mse=float(met["mse"].mean()),mae=float(met["mae"].mean())))
    csvsave(r/"by_seed_period.csv",rows);csvsave(r/"cases.csv",cases);csvsave(r/"teacher_by_seed_period.csv",teachers);np.savez_compressed(r/"predictions.npz",**outputs)
    fields=["mse","mae","level","shape","phase_mean_hours","phase_sd_hours"];macro=[];summary=[]
    for seed in c["seeds"]:
        for arm in ["FrozenBase","FrozenFourierKD"]+c["student_arms"]:
            a=[v for v in rows if v["seed"]==seed and v["arm"]==arm];macro.append(dict(seed=seed,arm=arm,**{k:float(np.mean([v[k] for v in a])) for k in fields}))
    for arm in ["FrozenBase","FrozenFourierKD"]+c["student_arms"]:
        a=[v for v in macro if v["arm"]==arm];summary.append(dict(arm=arm,**{k:float(np.mean([v[k] for v in a])) for k in fields}))
    csvsave(r/"macro_by_seed.csv",macro);csvsave(r/"summary.csv",summary)
    assert all(rt.sha(p)==v for p,v in lock["files"].items())
    src=rt.read(r/"source_train.json");assert all(rt.sha(p)==v for p,v in src["sources"].items())
    rt.save(r/"source_evaluation.json",dict(sources=sources,selection_checked_before_outer=True))
    assert time.monotonic()-tick<c["budget"]["max_eval_seconds"]
    rt.save(r/"completion.json",dict(fit_seconds=lock["seconds"],evaluation_seconds=time.monotonic()-tick,student_parameters=163,
        teacher_coefficients=2624,steps=100,warmup_steps=0,new_LLM_calls=0,LLM_updates=0,TimeBase_updates=0,
        conservation=conservation,old_sources_and_selection_unchanged=True,response_queries=lock["response_action_queries"]))
    print("RESULT",json.dumps(summary),flush=True)

def preflight(c):
    p13.preflight(c);torch.set_num_threads(2);rng=np.random.default_rng(15)
    x=rng.normal(size=(40,81));b=np.cos(p13.W*np.arange(96))[None]*np.ones((40,1));p=p13.pack(b,b)
    target=p13.rotate(p,.2*np.tanh(x[:,0]*x[:,47]));p=p13.pack(b,target)
    t=response_fit(x,p,.1,c);cost=response_grid(t,x);assert cost.shape==(40,9) and np.all(cost[:,4]==0)
    grid=torch.tensor(c["angles_hours"],dtype=torch.float64)[None].repeat(40,1)*p13.W
    assert torch.max(torch.abs(query(torch.tensor(cost),grid,c)-torch.tensor(cost)))<1e-12
    teacher=np.array([.2,.5,.3]);student=np.array([.4,.35,.25])
    exact=np.sum(student*(np.log(student)-np.log(teacher)))
    ps=torch.tensor(student);qt=torch.tensor(teacher)
    assert abs(float((ps*(ps.log()-qt.log())).sum())-exact)<1e-12 and exact>0
    model,trace,queries=fit_student(x,p,.1,c,2024,"OPD",(cost,np.full(40,float(t["temperature"]))))
    assert bil.count(model)==163 and queries==40*3*(c["steps"]-c["warmup_steps"]) and trace[-1]["reverse_kl"]>=-1e-12
    print("RESPONSE_OPD_PREFLIGHT_PASS",json.dumps(dict(parameters=bil.count(model),teacher_coefficients=t["coef"].size,queries=queries)),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--preflight",action="store_true");ap.add_argument("--select",action="store_true");ap.add_argument("--evaluate",action="store_true");a=ap.parse_args()
    c=rt.read(CONFIG);r=Path(c["run"])
    if a.preflight:preflight(c)
    elif a.select:select(c,r)
    elif a.evaluate:evaluate(c,r)
    else:raise ValueError("choose phase")
if __name__=="__main__":main()
