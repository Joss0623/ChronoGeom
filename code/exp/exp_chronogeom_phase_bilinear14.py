"""Rank-one bilinear phase readout. Existing caches only; conditional KD."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
import argparse,json,time,traceback,shutil
from pathlib import Path
import numpy as np
import torch
from exp import exp_chronogeom_phase_rotation13 as prior
old=prior.old;rt=prior.rt;csvsave=prior.csvsave
HIDX=np.r_[np.arange(47),79,80];ZIDX=np.arange(47,79)
TEACHER=prior.TEACHER
CONFIG=Path("docs/chronogeom/phase_bilinear14_config.json")

def numpy_logits(m,x,conditional=True):
    xx=(x-m["mu"])/m["sd"]
    if not conditional:return np.full(len(x),float(m["bias"]))
    return xx@m["coef"]+float(m["bias"])+(xx[:,HIDX]@m["left"])*(xx[:,ZIDX]@m["right"])

def predict(m,x,c,conditional=True):
    return c["delta_max"]*np.tanh(numpy_logits(m,x,conditional))

def count(m):return sum(np.size(m[k]) for k in ["coef","bias","left","right"])

def fit_head(x,p,lam,c,seed,teacher=None):
    assert x.shape[1]==81
    mu=x.mean(0);sd=np.maximum(x.std(0),.001)
    xx=torch.tensor((x-mu)/sd,dtype=torch.float64);hh=xx[:,HIDX];zz=xx[:,ZIDX]
    aa=torch.tensor(p["A"]);bb=torch.tensor(p["B"]);ee=torch.tensor(p["E"])
    rng=np.random.default_rng(seed+140014);initial=rng.normal(size=49);initial/=np.linalg.norm(initial)
    w=torch.zeros(81,dtype=torch.float64,requires_grad=True)
    bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    left=torch.tensor(initial,dtype=torch.float64,requires_grad=True)
    right=torch.zeros(32,dtype=torch.float64,requires_grad=True)
    target=None if teacher is None else torch.tensor(teacher,dtype=torch.float64)
    opt=torch.optim.Adam([w,bias,left,right],lr=c["learning_rate"]);trace=[]
    for step in range(1,c["steps"]+1):
        opt.zero_grad();interaction=(hh@left)*(zz@right)
        delta=c["delta_max"]*torch.tanh(xx@w+bias+interaction)
        sup=(4*aa*torch.sin(delta/2)**2-2*bb*torch.sin(delta)).mean()
        kd=torch.zeros((),dtype=torch.float64) if target is None else (4*ee*torch.sin((delta-target)/2)**2).mean()
        # Exact squared Frobenius norm of the rank-one interaction matrix.
        penalty=w.square().sum()+left.square().sum()*right.square().sum()
        loss=sup+lam*penalty+c["kd_weight"]*kd
        loss.backward()
        if step==1:
            assert float(right.grad.norm())>0,"zero initial interaction gradient"
        opt.step()
        if step in [1,25,50,100]:
            trace.append(dict(step=step,supervised_without_constant=float(sup.detach()),kd=float(kd.detach()),
                interaction_rms=float(interaction.detach().square().mean().sqrt()),
                right_gradient=float(right.grad.norm()),penalty=float(penalty.detach())))
    m=dict(mu=mu,sd=sd,coef=w.detach().numpy(),bias=bias.detach().numpy(),
           left=left.detach().numpy(),right=right.detach().numpy())
    assert count(m)==163 and all(np.isfinite(v).all() for v in m.values())
    return m,trace

def constant_angle(p,c):
    angle=float(np.arctan2(p["B"].mean(),p["A"].mean()))
    candidates=[0.,-c["delta_max"],c["delta_max"]]
    if abs(angle)<=c["delta_max"]:candidates.append(angle)
    return min(candidates,key=lambda x:prior.mse_vector(p,np.full(len(p["A"]),x)).mean())

def save_lock(r,c,name,extra):
    rt.save(r/name,dict(files={str(p):rt.sha(p) for p in r.iterdir() if p.is_file()},
       parameters=163,steps=100,no_outer_files_opened=True,**extra))

def gate(records,choices,constant):
    # Same past folds only. This is a development gate, not an unbiased hypothesis test.
    keys=sorted(set((v["seed"],v["fold"]) for v in records))
    losses={}
    for arm,v in choices.items():
        losses[arm]={(q["seed"],q["fold"]):q["ratio"] for q in records if q["arm"]==arm and q["lam"]==v["lam"]}
    losses["FrozenBase"]={k:1. for k in keys}
    losses["TrainConstantPhase"]={(v["seed"],v["fold"]):v["ratio"] for v in constant}
    comparisons=[]
    for control in ["FrozenBase","TrainConstantPhase","SeasonFourier32","QwenTrainShuffle"]:
        byseed={str(seed):float(np.mean([losses["QwenTask"][k]-losses[control][k] for k in keys if k[0]==seed])) for seed in sorted(set(k[0] for k in keys))}
        delta=float(np.mean(list(byseed.values())))
        comparisons.append(dict(control=control,mean_normalized_mse_difference=delta,seed_differences=byseed,
            improved_seeds=sum(v<0 for v in byseed.values()),passed=delta<0 and sum(v<0 for v in byseed.values())>=2))
    return dict(passed=all(v["passed"] for v in comparisons),comparisons=comparisons,
      rule="Task lower mean past-fold normalized MSE and >=2/3 improved seeds versus frozen base, training-optimal constant angle, equal-capacity Fourier/NoKD, and shuffle. Wrong retained and reported, not an extra gate.",
      source="training-internal only",not_independent_validation=True)

def select(c,r):
    r.mkdir(parents=True,exist_ok=False);rt.save(r/"config.json",c);shutil.copy2(__file__,r/Path(__file__).name)
    tick=time.monotonic();torch.set_num_threads(2)
    try:
        data,h=prior.training_data(c,r)
        src=rt.read(r/"source_train.json")
        for f in [Path(__file__),Path(prior.__file__),CONFIG]:src["sources"][str(f)]=rt.sha(f)
        rt.save(r/"source_train.json",src)
        folds,meta=prior.make_folds(c,data,h);rt.save(r/"crossfit_inner.json",meta)
        records=[];constant=[];saved_fold_predictions={}
        for (seed,name),f in folds.items():
            delta=constant_angle(f["train"],c)
            const=float(prior.mse_vector(f["valid"],np.full(len(f["vi"]),delta)).mean())
            base=float(f["valid"]["base_mse"].mean())
            constant.append(dict(seed=seed,fold=name,angle=delta,mse=const,ratio=const/base))
            for arm in c["teacher_arms"]:
                for lam in c["lambdas"]:
                    m,_=fit_head(f["mat"]["train"][arm],f["train"],lam,c,seed)
                    delta=predict(m,f["mat"]["inner"][arm],c)
                    mse=prior.mse_vector(f["valid"],delta)
                    records.append(dict(seed=seed,fold=name,arm=arm,lam=lam,mse=float(mse.mean()),ratio=float(mse.mean()/base),
                        angle_mean_hours=float((delta/prior.W).mean()),angle_sd_hours=float((delta/prior.W).std())))
                    saved_fold_predictions[seed,name,arm,lam]=delta
        choice=old.choose(records);csvsave(r/"teacher_inner.csv",records);csvsave(r/"constant_inner.csv",constant)
        rt.save(r/"teacher_choice.json",choice);decision=gate(records,choice,constant);rt.save(r/"teacher_gate.json",decision)
        detail=[]
        for (seed,name),f in folds.items():
            for arm in c["teacher_arms"]:
                delta=saved_fold_predictions[seed,name,arm,choice[arm]["lam"]]
                gain=f["valid"]["base_mse"]-prior.mse_vector(f["valid"],delta)
                theta=np.arctan2(f["valid"]["B"],f["valid"]["A"])
                for j in range(len(delta)):
                    detail.append(dict(seed=seed,fold=name,arm=arm,origin=int(data[seed,"train"]["origin"][f["vi"][j]]),
                       shift_hours=float(delta[j]/prior.W),mse=float(f["valid"]["base_mse"][j]-gain[j]),
                       gain=float(gain[j]),oracle_direction=int(np.sign(theta[j])),phase_energy=float(f["valid"]["E"][j])))
        csvsave(r/"inner_cases.csv",detail)
        save_lock(r,c,"teacher_gate_lock.json",dict(gate=decision["passed"],teacher_choice={a:v["lam"] for a,v in choice.items()}))
        sc=None
        if decision["passed"]:
            targets={}
            for key,f in folds.items():
                targets[key]={}
                for arm in c["teacher_arms"]:
                    m,_=fit_head(f["mat"]["train"][arm],f["train"],choice[arm]["lam"],c,key[0])
                    targets[key][arm]=predict(m,f["mat"]["train"][arm],c)
            studentrecords=[]
            for key,f in folds.items():
                seed,name=key
                for arm in c["student_arms"]:
                    for lam in c["lambdas"]:
                        m,_=fit_head(f["mat"]["train"]["SeasonFourier32"],f["train"],lam,c,seed,
                            None if arm=="NoKD" else targets[key][TEACHER[arm]])
                        delta=predict(m,f["mat"]["inner"]["SeasonFourier32"],c)
                        score=float(prior.mse_vector(f["valid"],delta).mean())
                        studentrecords.append(dict(seed=seed,fold=name,arm=arm,lam=lam,mse=score,ratio=score/float(f["valid"]["base_mse"].mean())))
            sc=old.choose(studentrecords);csvsave(r/"student_inner.csv",studentrecords);rt.save(r/"student_choice.json",sc)
        curves=[];metadata=[]
        for seed in c["seeds"]:
            q=data[seed,"train"];ii=np.arange(len(q["origin"]));b,cf=prior.crossfit_base(q,ii,c);p=prior.pack(b,q["y"])
            mat,pca=old.matrices(c,data,h,seed,ii,dict(train=ii));mat=mat["train"]
            for arm in mat:mat[arm]=np.c_[mat[arm],p["coords"]]
            metadata.append(dict(seed=seed,crossfit=cf))
            bm=prior.base_fit(q,ii,c);ref=np.load(Path(c["reuse_phase_cache"])/f"teacher_s{seed}_SeasonFourier32.npz")
            assert max(float(abs(bm[k]-ref[k]).max()) for k in bm)<1e-10
            np.savez_compressed(r/f"frozen_base_s{seed}.npz",**bm)
            for arm,(mu,w,info) in pca.items():np.savez_compressed(r/f"pca_s{seed}_{arm}.npz",mu=mu,w=w)
            targets={}
            for arm in c["teacher_arms"]:
                m,trace=fit_head(mat[arm],p,choice[arm]["lam"],c,seed);targets[arm]=predict(m,mat[arm],c)
                np.savez_compressed(r/f"teacher_s{seed}_{arm}.npz",**m);curves.append(dict(seed=seed,arm=arm,role="teacher",trace=trace))
            if sc is not None:
                for arm in c["student_arms"]:
                    m,trace=fit_head(mat["SeasonFourier32"],p,sc[arm]["lam"],c,seed,
                       None if arm=="NoKD" else targets[TEACHER[arm]])
                    np.savez_compressed(r/f"student_s{seed}_{arm}.npz",**m);curves.append(dict(seed=seed,arm=arm,role="student",trace=trace))
        rt.save(r/"crossfit_full.json",metadata);rt.save(r/"curves.json",curves)
        assert all(rt.sha(p)==v for p,v in src["sources"].items())
        gate_lock=rt.read(r/"teacher_gate_lock.json");assert all(rt.sha(p)==v for p,v in gate_lock["files"].items())
        assert time.monotonic()-tick<c["budget"]["max_fit_seconds"]
        save_lock(r,c,"selection_lock.json",dict(seconds=time.monotonic()-tick,kd_started=decision["passed"],
            teacher_choice={a:v["lam"] for a,v in choice.items()},
            student_choice=None if sc is None else {a:v["lam"] for a,v in sc.items()}))
        print("TEACHER_GATE",json.dumps(decision),flush=True)
    except BaseException:
        rt.save(r/"failure_select.json",dict(traceback=traceback.format_exc()));raise

def evaluate(c,r):
    tick=time.monotonic();lock=rt.read(r/"selection_lock.json")
    assert all(rt.sha(p)==v for p,v in lock["files"].items())
    pc=rt.read(Path(c["reliability_run"])/"config.json");cache=Path(pc["reuse_run"])
    sums_path=Path(c["reliability_run"])/"shared_summaries.json"
    sums={(v["seed"],v["stage"],v["row"]):v["summary"] for v in rt.read(sums_path)}
    sources={str(sums_path):rt.sha(sums_path)};rows=[];cases=[];outputs={};conservation={};replays=0
    for seed in c["seeds"]:
        bm=np.load(r/f"frozen_base_s{seed}.npz")
        for stage in c["stages"]:
            f=cache/f"seed{seed}_{stage}.npz";raw=np.load(f);q={k:raw[k] for k in raw.files};sources[str(f)]=rt.sha(f)
            q["features"]=old.annual_features(np.stack([prior.s.flatten(sums[seed,stage,j]) for j in range(len(q["origin"]))]),q["origin"],c)
            b=prior.base_predict(q,np.arange(len(q["origin"])),bm);p=prior.pack(b,q["y"]);x=np.c_[prior.base_phi(q),p["coords"]]
            matrices={"SeasonFourier32":x}
            for arm in c["contexts"]:
                f=Path(c["reuse_phase_cache"])/"evaluation_extraction"/f"hidden_{seed}_{arm}_{stage}.npy";h=np.load(f).astype(float)
                sources[str(f)]=rt.sha(f);basis=np.load(r/f"pca_s{seed}_{arm}.npz")
                matrices[arm]=np.c_[q["features"],(h-basis["mu"])@basis["w"].T,p["coords"]]
            matrices["QwenTrainShuffle"]=matrices["QwenTask"]
            angles={"FrozenBase":np.zeros(len(x))};states={}
            for arm in c["teacher_arms"]:
                m=np.load(r/f"teacher_s{seed}_{arm}.npz");label="NoKD_Fourier" if arm=="SeasonFourier32" else "Teacher_"+arm
                angles[label]=predict(m,matrices[arm],c);states[label]=(m,matrices[arm])
            if lock["kd_started"]:
                for arm in c["student_arms"]:
                    m=np.load(r/f"student_s{seed}_{arm}.npz");angles["Student_"+arm]=predict(m,x,c);states["Student_"+arm]=(m,x)
            for label,delta in angles.items():
                pred,met,checks=prior.forecast_metrics(p,delta)
                for k,v in checks.items():conservation[k]=max(conservation.get(k,0.),v)
                outputs[f"s{seed}_{stage}_{label}"]=pred;outputs[f"a{seed}_{stage}_{label}"]=delta
                if label in states:
                    m,z=states[label];assert count(m)==163
                    zz=(z-m["mu"])/m["sd"]
                    explicit=c["delta_max"]*np.tanh(zz@m["coef"]+float(m["bias"])+(zz[:,HIDX]@m["left"])*(zz[:,ZIDX]@m["right"]))
                    assert np.array_equal(delta,explicit);replays+=1
                    np.savez_compressed(r/f"deployment_s{seed}_{stage}_{label}.npz",x=z,b=b,u=p["u"],v=p["v"])
                    bilinear=(zz[:,HIDX]@m["left"])*(zz[:,ZIDX]@m["right"])
                    bilinear_rms=float(np.mean(bilinear**2)**.5)
                else:bilinear_rms=0.
                fields={k:float(met[k].mean()) for k in ["mse","mae","level","shape","amplitude_error"]}
                rows.append(dict(seed=seed,stage=stage,arm=label,n=len(delta),**fields,phase_error_hours=float(np.sum(met["phase"]*met["weight"])/max(float(met["weight"].sum()),1e-12)),
                  shift_mean_hours=float((delta/prior.W).mean()),shift_sd_hours=float((delta/prior.W).std()),bilinear_logit_rms=bilinear_rms))
                gain=p["base_mse"]-met["mse"]
                theta=np.arctan2(p["B"],p["A"])
                for j in range(len(delta)):
                    cases.append(dict(seed=seed,stage=stage,arm=label,origin=int(q["origin"][j]),shift_hours=float(delta[j]/prior.W),
                      mse=float(met["mse"][j]),mae=float(met["mae"][j]),level=float(met["level"][j]),shape=float(met["shape"][j]),
                      gain=float(gain[j]),oracle_direction=int(np.sign(theta[j])),phase_energy=float(p["E"][j])))
    csvsave(r/"by_seed_period.csv",rows);csvsave(r/"cases.csv",cases);np.savez_compressed(r/"predictions.npz",**outputs)
    fields=["mse","mae","level","shape","amplitude_error","phase_error_hours","shift_mean_hours","shift_sd_hours","bilinear_logit_rms"]
    macro=[];summary=[]
    for seed in c["seeds"]:
        for arm in sorted(set(v["arm"] for v in rows)):
            a=[v for v in rows if v["seed"]==seed and v["arm"]==arm]
            macro.append(dict(seed=seed,arm=arm,**{k:float(np.mean([v[k] for v in a])) for k in fields}))
    for arm in sorted(set(v["arm"] for v in rows)):
        a=[v for v in macro if v["arm"]==arm];summary.append(dict(arm=arm,**{k:float(np.mean([v[k] for v in a])) for k in fields}))
    csvsave(r/"macro_by_seed.csv",macro);csvsave(r/"summary.csv",summary)
    assert all(rt.sha(p)==v for p,v in lock["files"].items())
    src=rt.read(r/"source_train.json");assert all(rt.sha(p)==v for p,v in src["sources"].items())
    rt.save(r/"source_evaluation.json",dict(sources=sources,selection_checked_before_outer=True))
    assert time.monotonic()-tick<c["budget"]["max_eval_seconds"]
    rt.save(r/"completion.json",dict(fit_seconds=lock["seconds"],evaluation_seconds=time.monotonic()-tick,
        new_LLM_calls=0,LLM_updates=0,TimeBase_updates=0,kd_started=lock["kd_started"],parameters=163,steps=100,
        conservation=conservation,explicit_numpy_angle_checks=replays,source_and_selection_hashes_unchanged=True))
    print("RESULT",json.dumps(summary),flush=True)

def preflight(c):
    prior.preflight(c)
    rng=np.random.default_rng(14);x=rng.normal(size=(40,81));truth=.2*np.tanh(x[:,0]*x[:,47])
    b=np.cos(prior.W*np.arange(96))[None]*np.ones((40,1));p=prior.pack(b,b)
    y=prior.rotate(p,truth);p=prior.pack(b,y)
    m,trace=fit_head(x,p,.01,c,2024)
    assert count(m)==163 and trace[0]["right_gradient"]>0
    assert trace[0]["interaction_rms"]==0 and trace[-1]["interaction_rms"]>0
    assert np.all(abs(predict(m,x,c))<=c["delta_max"])
    print("BILINEAR_PREFLIGHT_PASS",json.dumps(dict(parameters=count(m),initial_interaction_gradient=trace[0]["right_gradient"],final_interaction_rms=trace[-1]["interaction_rms"])),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--preflight",action="store_true");ap.add_argument("--select",action="store_true");ap.add_argument("--evaluate",action="store_true");args=ap.parse_args()
    c=rt.read(CONFIG);r=Path(c["run"])
    if args.preflight:preflight(c)
    elif args.select:select(c,r)
    elif args.evaluate:evaluate(c,r)
    else:raise ValueError("choose phase")
if __name__=="__main__":main()
