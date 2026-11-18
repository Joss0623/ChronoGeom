"""Metro teacher64: reuse strong63TimeBase, update only the existing prediction reader."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",
 TORCH_DEVICE_BACKEND_AUTOLOAD="0",CUBLAS_WORKSPACE_CONFIG=":4096:8")
import argparse,copy,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from exp.exp_chronogeom_calendar_level_direct63 import DirectStudent,source_base,to_pack,pred,center,statecpu,writecsv,interval
from exp.exp_chronogeom_teacher_content40 import calendar_arrays,check22
from exp.exp_chronogeom_event_knowledge19 import read,save,sha,git,scores
from exp.exp_chronogeom_residual03 import numeric_summary

class PredictionTeacher(nn.Module):
    """Original strong numerical/calendar prior, SAME705reader, no new layer."""
    def __init__(self,base,table):
        super().__init__()
        self.body=copy.deepcopy(base.body)
        self.affine=nn.Parameter(base.affine.detach().clone(),requires_grad=False)
        self.calendar_level=copy.deepcopy(base.calendar_level)
        self.original_codes=nn.Embedding.from_pretrained(base.codes.weight.detach().clone(),freeze=True)
        for p in self.body.parameters():p.requires_grad_(False)
        for p in self.calendar_level.parameters():p.requires_grad_(False)
        self.reader=copy.deepcopy(base.reader)
        for p in self.reader.parameters():p.requires_grad_(True)
        self.register_buffer("numeric_mean",base.numeric_mean.detach().clone())
        self.register_buffer("numeric_sd",base.numeric_sd.detach().clone())
        self.register_buffer("prior_calendar_weight",base.reader[0].weight[:,39:].detach().clone())
        self.register_buffer("knowledge_table",table.detach().clone().to(base.affine))
        with torch.no_grad():self.reader[0].weight[:,39:].zero_()
    def features(self,x,calendar,ids):
        numerical=self.body(x)*(1+torch.tanh(self.affine[0]))+self.affine[1]
        level=numerical.mean(1,keepdim=True)+self.calendar_level(calendar.mean(1))[:,None,:]
        num=(numeric_summary(x,numerical)-self.numeric_mean)/self.numeric_sd
        z=torch.cat([num,calendar,self.knowledge_table[ids]],-1)
        offset=F.linear(self.original_codes(ids),self.prior_calendar_weight)
        return z,offset,level
    def read_features(self,z,offset,level):
        hidden=F.linear(z,self.reader[0].weight,self.reader[0].bias)+offset
        raw=self.reader[2](F.gelu(hidden))
        return level+center(raw)
    def forward(self,x,calendar,ids):
        return self.read_features(*self.features(x,calendar,ids))

def load_base(c,seed,device,state=None):
    if state is None:
        state=torch.load(Path(c["source63"])/"students.pt",map_location="cpu",weights_only=True)["Metro"][seed]["DirectNoKD"]
    old=torch.load(source_base(c,"Metro",seed),map_location="cpu",weights_only=True)
    model=DirectStudent(seed,old,state["numeric_mean"].cpu().numpy(),state["numeric_sd"].cpu().numpy())
    model.load_state_dict(state);model.to(device);model.eval()
    return model

def choose(c,curves,baselines,y,blocks,arm):
    options=[]
    for step in c["checkpoints"]:
        risk=[]
        for seed in c["seeds"]:
            per=[]
            for bi,b in enumerate(blocks):
                ref=scores(baselines[seed][b],y[b])
                v=next(v for v in curves if (v["seed"],v["arm"],v["step"],v["block"])==(seed,arm,step,bi))
                per.append(.5*(v["mse"]/ref["mse"]+v["mae"]/ref["mae"]))
            risk.append(per)
        risk=np.array(risk)
        options.append(dict(step=step,score=float((risk.mean()+risk.mean(1).max()+risk.mean(0).max())/3)))
    return dict(step=min(options,key=lambda x:(x["score"],x["step"]))["step"],options=options)

def prepare(c,config):
    r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c)
    assets=check22(c);seen={v["path"] for v in assets}
    for name in ["source51","source61","source62","source63"]:
        for p in Path(c[name]).iterdir():
            if p.is_file() and str(p) not in seen:
                assets.append(dict(path=str(p),sha=sha(p)));seen.add(str(p))
    paths=[Path(c["teacher40"])/"Metro/profiles.json",Path(c["teacher40"])/"teacher_predictions.npz"]
    paths += [Path(c["domains"]["Metro"]["source"])/n for n in ["data.npz","config.json"]]
    paths += [Path(f"exp/exp_chronogeom_{s}.py") for s in ["direct_forecast61","level_shape_direct62","calendar_level_direct63"]]
    paths += [Path(f"docs/chronogeom/{s}_config.json") for s in ["direct_forecast61","level_shape_direct62","calendar_level_direct63"]]
    for p in paths:
        if str(p) not in seen:assets.append(dict(path=str(p),sha=sha(p)));seen.add(str(p))
    save(r/"protected_assets.json",assets)
    save(r/"protocol_lock.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),before_training=True))
    print("PREPARED64",len(assets),flush=True)

def execute(c,config,resume_teachers=False):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    r=Path(c["run"]);assert read(r/"config.json")==c
    lock=read(r/"protocol_lock.json")
    assert sha(config)==lock["config_sha"]
    if not resume_teachers:
        assert sha(__file__)==lock["source_sha"]
    else:
        resume=read(r/"technical_resume.json")
        assert sha(__file__)==resume["fixed_source_sha"]
        for name,value in resume["preserved_teacher_SHA"].items():assert sha(r/name)==value
        assert not (r/"teacher_gate.json").exists()
    active_source_sha=sha(__file__)
    assert resume_teachers or not (r/"all_teacher_checkpoints.pt").exists()
    device=torch.device(c["device"]);assert torch.cuda.mem_get_info(device)[0]>=c["gpu_memory_gib"]*2**30
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(c["gpu_memory_gib"]*2**30/torch.cuda.get_device_properties(device).total_memory,device)
    started=time.monotonic()
    save(r/("resume_provenance.json" if resume_teachers else "provenance.json"),dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),torch=torch.__version__,
        numpy=np.__version__,device=str(device),GPU=torch.cuda.get_device_name(device),inherited63NoKD_selected_updates=400,
        historical63maximum_updates=600,new_LLM_calls=0))
    assets=read(r/"protected_assets.json")
    for a in assets:assert sha(a["path"])==a["sha"],a["path"]
    old=Path(c["source63"]);src=Path(c["domains"]["Metro"]["source"]);packs=np.load(src/"data.npz")
    grid,raw,common,kind,h=calendar_arrays(read(src/"config.json"),"Metro")
    splits=read(old/"time_splits.json")["Metro"];selected=np.array(splits["selection_indices"])
    assert len(packs["fit_x"])==73 and len(selected)==29
    assert np.array_equal(selected,np.flatnonzero(packs["val_origin"]-96>packs["fit_future"].max()))
    blocks=np.array_split(np.arange(len(selected)),3)
    save(r/"time_split.json",splits)
    fit=to_pack(packs,"fit",grid,raw,common,kind*24+h,device)
    val=to_pack(packs,"val",grid,raw,common,kind*24+h,device,selected)
    yf=torch.tensor(packs["fit_y"],device=device,dtype=torch.float32);yv=packs["val_y"][selected]
    knowledge=read(old/"knowledge_tables.json")["Metro"]
    tables={name:torch.tensor(knowledge[name]["target"],device=device,dtype=torch.float32) for name in ["Delta","Fourier","ShuffledDelta"]}
    prior_states=torch.load(old/"students.pt",map_location="cpu",weights_only=True)["Metro"]
    oldpred=np.load(old/"predictions.npz")
    data={};baselines={};states={};curves=[];trainlogs=[];preflight=[];diagnostics=[];table_export={}
    for seed in c["seeds"]:
        base=load_base(c,seed,device);frozen=statecpu(base)
        baseline=pred(base,val);ref=oldpred[f"Metro_{seed}_internal_DirectNoKD"]
        gap=float(abs(baseline-ref).max());assert gap<3e-6
        baselines[seed]=baseline
        with torch.no_grad():calendar=base.codes.weight.detach().clone()
        kfit=calendar[fit["ids"]];mean=kfit.mean((0,1));rms=(kfit-mean).square().mean().sqrt().clamp_min(1e-6)
        tables_seed=dict(tables);tables_seed["LearnedCalendar"]=(calendar-mean)/rms
        table_export[seed]={k:v.detach().cpu().numpy().tolist() for k,v in tables_seed.items()}
        data[seed]=dict(prior=frozen,base=base,features={},tables=tables_seed)
        for arm,key in c["teacher_arms"].items():
            model=PredictionTeacher(base,tables_seed[key]).to(device)
            assert sum(p.numel() for p in model.parameters())==995
            assert sum(p.numel() for p in model.parameters() if p.requires_grad)==705
            with torch.no_grad():
                initial=pred(model,val)
                diff=float(abs(initial-baseline).max());assert diff<4e-6
                features=model.features(fit["x"],fit["calendar"],fit["ids"])
                data[seed]["features"][arm]=tuple(v.detach() for v in features)
                features_v=model.features(val["x"],val["calendar"],val["ids"])
                data[seed].setdefault("val_features",{})[arm]=tuple(v.detach() for v in features_v)
            model.zero_grad(set_to_none=True)
            p=model.read_features(*(v[:2] for v in features));((p-yf[:2])**2).mean().backward()
            grads=float(model.reader[0].weight.grad[:,39:].abs().sum());assert grads>0
            assert all(v.grad is None for name,v in model.named_parameters() if not name.startswith("reader."))
            preflight.append(dict(seed=seed,arm=arm,prior_replay_error=gap,initial_prediction_error=diff,
                                  nominal_params=995,trainable_params=705,additional_knowledge_gradient=grads))
            k=tables_seed[key][fit["ids"]].cpu().numpy().reshape(-1,3)
            sv=np.linalg.svd(k-k.mean(0),compute_uv=False);pwr=sv**2;pwr=pwr/pwr.sum()
            diagnostics.append(dict(seed=seed,arm=arm,dimensions=3,rank=int((sv>sv[0]*1e-6).sum()),
                effective_rank=float(np.exp(-np.sum(pwr[pwr>0]*np.log(pwr[pwr>0])))),fit_RMS=float(np.sqrt(np.mean(k*k)))))
        del model
    save(r/"preflight.json",preflight);save(r/"knowledge_diagnostics.json",diagnostics);save(r/"knowledge_tables.json",table_export)
    torch.save({seed:v["prior"] for seed,v in data.items()},r/"frozen_strong_priors.pt")
    if resume_teachers:
        states=torch.load(r/"all_teacher_checkpoints.pt",map_location="cpu",weights_only=True)
        curves=read(r/"curves.json");trainlogs=read(r/"train_loss.json")
        print("RESUME64_REUSE_ALL12_TEACHERS_NO_REFIT",flush=True)
    else:
        for seed,d in data.items():
            states[seed]={};batches=np.random.default_rng(seed).integers(0,len(yf),size=(c["steps"],c["batch"]))
            for arm,key in c["teacher_arms"].items():
                teacher=PredictionTeacher(d["base"],d["tables"][key]).to(device);optimizer=torch.optim.Adam(teacher.reader.parameters(),lr=c["lr"])
                states[seed][arm]={}
                for step in range(c["steps"]+1):
                    assert time.monotonic()-started<c["teacher_cap"],"teacher64 time cap"
                    if step in c["checkpoints"]:
                        states[seed][arm][step]=statecpu(teacher)
                        with torch.no_grad():pv=teacher.read_features(*d["val_features"][arm]).cpu().numpy()
                        for bi,b in enumerate(blocks):
                            curves.append(dict(seed=seed,arm=arm,step=step,block=bi,windows=len(b),**scores(pv[b],yv[b])))
                    if step==c["steps"]:break
                    ix=torch.tensor(batches[step],device=device)
                    out=teacher.read_features(*(v[ix] for v in d["features"][arm]))
                    loss=(out-yf[ix]).square().mean();assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in teacher.reader.parameters())
                    optimizer.step()
                    if (step+1)%100==0:trainlogs.append(dict(seed=seed,arm=arm,step=step+1,loss=float(loss.detach())))
                print("TEACHER64",seed,arm,"600steps",flush=True)
                del optimizer,teacher
    selection={arm:choose(c,curves,baselines,yv,blocks,arm) for arm in c["teacher_arms"]}
    chosen={seed:{arm:states[seed][arm][selection[arm]["step"]] for arm in c["teacher_arms"]} for seed in c["seeds"]}
    if resume_teachers:
        assert selection==read(r/"teacher_selection.json")
        disk=torch.load(r/"teachers.pt",map_location="cpu",weights_only=True)
        assert all(torch.equal(chosen[seed][arm][key],disk[seed][arm][key])
                   for seed in chosen for arm in chosen[seed] for key in chosen[seed][arm])
    else:
        torch.save(states,r/"all_teacher_checkpoints.pt");torch.save(chosen,r/"teachers.pt")
        save(r/"teacher_selection.json",selection);save(r/"curves.json",curves);save(r/"train_loss.json",trainlogs)
    teacher_preds={};internal_rows=[];block_rows=[]
    for seed,d in data.items():
        teacher_preds[(seed,"Frozen63NoKD")]=baselines[seed]
        for arm,key in c["teacher_arms"].items():
            t=PredictionTeacher(d["base"],d["tables"][key]).to(device);t.load_state_dict(chosen[seed][arm])
            with torch.no_grad():
                teacher_preds[(seed,arm)]=t.read_features(*d["val_features"][arm]).cpu().numpy()
                d.setdefault("fit_teacher_targets",{})[arm]=t.read_features(*d["features"][arm]).detach()
        for arm in ["Frozen63NoKD"]+list(c["teacher_arms"]):
            pr=teacher_preds[(seed,arm)];internal_rows.append(dict(seed=seed,arm=arm,**scores(pr,yv)))
            for bi,b in enumerate(blocks):block_rows.append(dict(seed=seed,arm=arm,block=bi,**scores(pr[b],yv[b])))
    summary_internal={arm:{k:float(np.mean([v[k] for v in internal_rows if v["arm"]==arm])) for k in ["mse","mae","level","shape"]}
                      for arm in ["Frozen63NoKD"]+list(c["teacher_arms"])}
    checks={}
    for ref in ["Frozen63NoKD","UpdatedNoKD","FourierTeacher"]:
        q=summary_internal["QwenTeacher"];b=summary_internal[ref]
        metric_ok=all(q[k]<b[k] for k in ["mse","mae","shape"])
        wins=sum(scores(teacher_preds[(s,"QwenTeacher")],yv)["mse"]<scores(teacher_preds[(s,ref)],yv)["mse"] for s in c["seeds"])
        bwins=sum(np.mean([v["mse"] for v in block_rows if v["arm"]=="QwenTeacher" and v["block"]==bi])<
                  np.mean([v["mse"] for v in block_rows if v["arm"]==ref and v["block"]==bi]) for bi in range(3))
        checks[ref]=dict(all3metrics_improved=metric_ok,seed_MSE_wins=wins,block_MSE_wins=int(bwins))
    passed=selection["QwenTeacher"]["step"]>0 and all(v["all3metrics_improved"] and v["seed_MSE_wins"]>=2 and v["block_MSE_wins"]>=2 for v in checks.values())
    gate=dict(pass_=bool(passed),checks=checks,internal_summary=summary_internal,uses_outer=False,
              Qwen_step=selection["QwenTeacher"]["step"],KD_authorized_if_pass=True)
    save(r/"teacher_gate.json",gate);save(r/"internal_teacher_metrics.json",internal_rows);save(r/"internal_blocks.json",block_rows)
    save(r/"teacher_lock.json",dict(teachers_sha=sha(r/"teachers.pt"),selection_sha=sha(r/"teacher_selection.json"),gate_sha=sha(r/"teacher_gate.json"),
        before_outer_scoring=True,source_commit=git()))
    teacher_seconds=time.monotonic()-started
    kd_states={};kd_selection={};kd_curves=[];student_selected={};kd_fits=0
    if passed:
        kd_started=time.monotonic();student_arms=dict(NoKD=None,QwenOutputKD="QwenTeacher",FourierOutputKD="FourierTeacher",ShuffleOutputKD="ShuffleTeacher")
        for seed,d in data.items():
            kd_states[seed]={};batches=np.random.default_rng(seed).integers(0,len(yf),size=(c["steps"],c["batch"]))
            for arm,targetname in student_arms.items():
                student=load_base(c,seed,device,state=d["prior"])
                optim=torch.optim.Adam([dict(params=[p for n,p in student.named_parameters() if n!="codes.weight"],lr=c["lr"]),
                                        dict(params=student.codes.parameters(),lr=c["map_lr"])])
                kd_states[seed][arm]={}
                for step in range(c["steps"]+1):
                    assert time.monotonic()-kd_started<c["student_cap"] and time.monotonic()-started<c["seconds_cap"]
                    if step in c["checkpoints"]:
                        kd_states[seed][arm][step]=statecpu(student);p=pred(student,val)
                        for bi,b in enumerate(blocks):kd_curves.append(dict(seed=seed,arm=arm,step=step,block=bi,**scores(p[b],yv[b])))
                    if step==c["steps"]:break
                    ix=torch.tensor(batches[step],device=device)
                    level,shape=student.components(fit["x"][ix],fit["calendar"][ix],fit["ids"][ix])
                    loss=(level-yf[ix].mean(1,keepdim=True)).square().mean()+(shape-center(yf[ix])).square().mean()
                    if targetname:loss=loss+c["shape_weight"]*(shape-center(d["fit_teacher_targets"][targetname][ix])).square().mean()
                    optim.zero_grad(set_to_none=True);loss.backward();optim.step();assert torch.isfinite(loss)
                kd_fits+=1;print("STUDENT64",seed,arm,"600steps",flush=True)
        kd_selection={arm:choose(c,kd_curves,baselines,yv,blocks,arm) for arm in student_arms}
        student_selected={s:{arm:kd_states[s][arm][kd_selection[arm]["step"]] for arm in student_arms} for s in c["seeds"]}
        torch.save(kd_states,r/"all_student_checkpoints.pt");torch.save(student_selected,r/"students.pt");save(r/"student_selection.json",kd_selection)
        save(r/"student_curves.json",kd_curves)
        save(r/"student_lock.json",dict(students_sha=sha(r/"students.pt"),selection_sha=sha(r/"student_selection.json"),before_outer_scoring=True))
    save(r/"KD_status.json",dict(performed=bool(passed),student_fits=kd_fits,reason="internal teacher gate passed" if passed else "No teacher value beyond matchedNoKD/Fourier;0studenttraining"))
    # All teacher and conditional student choices fixed before reading oldviewed outer arrays.
    ev=to_pack(packs,"confirm",grid,raw,common,kind*24+h,device);ye=packs["confirm_y"]
    subsets=dict(internal=(val,yv),development=(ev,ye))
    rows=[];predictions={};windows=[];replay=[];paired=[]
    for label,(pack,y) in subsets.items():
        predictions[f"{label}_y"]=y;predictions[f"{label}_origin"]=pack["origin"]
        for seed,d in data.items():
            outputs={"Frozen63NoKD":pred(d["base"],pack)}
            for arm,key in c["teacher_arms"].items():
                t=PredictionTeacher(d["base"],d["tables"][key]).to(device);t.load_state_dict(chosen[seed][arm]);t.eval()
                outputs[arm]=pred(t,pack)
                tr=PredictionTeacher(d["base"],d["tables"][key]).to(device)
                tr.load_state_dict(torch.load(r/"teachers.pt",map_location="cpu",weights_only=True)[seed][arm])
                error=float(abs(pred(tr,pack)-outputs[arm]).max());assert error==0
                with torch.no_grad():level=d["base"].level_prediction(pack["x"],pack["calendar"]).cpu().numpy()
                mean_error=float(abs(outputs[arm].mean(1,keepdims=True)-level).max());assert mean_error<2e-6
                replay.append(dict(seed=seed,arm=arm,split=label,max_error=error,mean_error=mean_error))
            if passed:
                for arm in kd_selection:
                    st=load_base(c,seed,device,state=student_selected[seed][arm])
                    outputs["Student"+arm]=pred(st,pack)
            for arm,p in outputs.items():
                predictions[f"{seed}_{label}_{arm}"]=p;met=scores(p,y)
                assert abs(met["mse"]-met["level"]-met["shape"])<1e-12
                rows.append(dict(seed=seed,split=label,arm=arm,**met))
                for j in range(len(y)):windows.append(dict(seed=seed,split=label,arm=arm,origin=str(grid[pack["origin"][j]]),**scores(p[j:j+1],y[j:j+1])))
    for ref in ["Frozen63NoKD","UpdatedNoKD","FourierTeacher","ShuffleTeacher"]:
        for key in ["mse","mae","shape"]:
            differences=[]
            for seed in c["seeds"]:
                q=predictions[f"{seed}_development_QwenTeacher"]-ye;b=predictions[f"{seed}_development_{ref}"]-ye
                if key=="shape":q=q-q.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
                difference=(abs(q)-abs(b)) if key=="mae" else (q*q-b*b)
                differences.append(difference.mean((1,2)))
            paired.append(dict(reference=ref,metric=key,**interval(np.mean(differences,axis=0),64001)))
    arms=list(dict.fromkeys(v["arm"] for v in rows))
    summary={label:{arm:{key:float(np.mean([v[key] for v in rows if v["split"]==label and v["arm"]==arm])) for key in ["mse","mae","level","shape"]} for arm in arms} for label in subsets}
    save(r/"summary.json",summary);save(r/"metrics.json",rows);writecsv(r/"metrics.csv",rows);writecsv(r/"window_metrics.csv",windows)
    np.savez_compressed(r/"predictions.npz",**predictions);save(r/"paired_intervals.json",paired)
    for a in assets:assert sha(a["path"])==a["sha"],a["path"]
    check22(c);assert sha(__file__)==active_source_sha
    save(r/"verification.json",dict(protected_assets=len(assets),all_unchanged=True,frozen22and51and63_unchanged=True,
        teacher_replays=len(replay),max_replay_error=max(v["max_error"] for v in replay),
        max_mean_error=max(v["mean_error"] for v in replay),reader_only_training=True,checks=replay))
    torch.cuda.synchronize(device)
    save(r/"status.json",dict(complete=True,seconds=time.monotonic()-started,teacher_seconds=teacher_seconds,technical_resume=resume_teachers,\
        original_failed_wall_seconds=read(r/"technical_resume.json")["original_failed_wall_seconds"] if resume_teachers else 0.,
        teacher_fits=12,teacher_updates=7200,student_fits=kd_fits,student_updates=kd_fits*c["steps"],
        KD_performed=bool(passed),trainable_teacher_parameters=705,total_teacher_parameters=995,
        extra_fixed_prior_weight_buffer=48,extra_knowledge_buffer=216,LLM_calls=0,downloads=0,
        peak_gpu_GiB=torch.cuda.max_memory_allocated(device)/2**30,independent_confirmation=False))
    print("GATE64",json.dumps(gate),flush=True);print("SUMMARY64",json.dumps(summary),flush=True);print("STATUS64",json.dumps(read(r/"status.json")),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--prepare",action="store_true");ap.add_argument("--resume-teachers",action="store_true")
    args=ap.parse_args();c=read(args.config)
    try:
        if args.prepare:prepare(c,args.config)
        else:execute(c,args.config,args.resume_teachers)
    except Exception:
        r=Path(c["run"])
        if r.exists():(r/("failure_"+str(time.time_ns())+".log")).write_text(traceback.format_exc())
        raise
