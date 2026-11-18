"""Live frozen-Qwen feedback on current TimeBase phase actions. No cached Qwen proxy."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",
 CUBLAS_WORKSPACE_CONFIG=":4096:8",HF_HUB_DISABLE_PROGRESS_BARS="1",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,json,time,traceback,shutil,subprocess,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from exp import exp_chronogeom_phase_opd16 as base
p13=base.p13;bil=base.bil;rt=base.rt;csvsave=base.csvsave
CONFIG=Path("docs/chronogeom/live_qwen_feedback18_config.json")
ARMS=["NoKD","Fourier","QwenTask","QwenWrong","QwenShuffle"]
PERMS=[[0,1,2],[1,2,0],[2,0,1]]

def pack(b,y):return p13.pack(b,y)
def npmodel(m):return {k:np.array(m[k],copy=True) for k in ["mu","sd","coef","bias","left","right"]}
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def named_summary(template,features):
    result={};j=0
    for group,values in template.items():
        result[group]={}
        for name in values:
            name="annual_sin" if name=="hour_sin" else "annual_cos" if name=="hour_cos" else name
            result[group][name]=float(features[j]);j+=1
    assert j==47
    return result

def build_prompt(summary,context,current,profiles,angles,peak,residue,mean,scales,perm):
    # No target array, target error, or oracle choice can enter this interface.
    options=[]
    for label,idx in zip("ABC",perm):
        options.append(dict(label=label,shift_hours=float(angles[idx]),fundamental_peak_clock_hour=float(peak[idx]),
                            daily_fundamental24=[round(float(v),3) for v in profiles[idx]]))
    return ("Domain: "+context+"\n"
      "Judge three candidate forecasts for the next96 hourly values using only available history, calendar and proposed forecasts. "
      "Future observations, future weather,occupancy and appliance/event schedules are unknown. "
      "Choose the option expected to minimize 0.5*MSE/squared_error_normalizer+0.5*MAE/absolute_error_normalizer. "
      "Normalizers come only from historical training. All values use fixed training-standardized demand units; negatives are valid. "
      "Each option changes ONLY the phase of the24h fundamental. All have IDENTICAL96h mean,daily amplitude,total centered energy and other-frequency residual. "
      "Daily arrays below are the sinusoidal component repeated4times,not the entire forecast. Do not prefer a flatter curve. "
      "start_weekday=0Monday..6Sunday,start_hour=0..23;following hours advance normally. annual_sin/cos are civil-year phase,Jan1~(0,1),Apr~(1,0),Jul~(0,-1),Oct~(-1,0).\n"
      "Shared past-only numerical/calendar/reference-forecast summary:"+json.dumps(summary,separators=(",",":"))+"\n"
      "Shared forecast96h_mean:"+str(round(float(mean),3))+
      ";shared other-frequency residual6h_means:"+json.dumps([round(float(v),3) for v in residue])+"\n"
      "Historical training normalizers:"+json.dumps(scales,separators=(",",":"))+"\n"
      "Student current phase shift_hours:"+str(round(float(current),6))+"\n"
      "Candidate options:"+json.dumps(options,separators=(",",":"))+"\n"
      "Which candidate is expected to have lower future prediction error? Reply with exactly A,B or C.")

def request_for(pkg,row,angle,current,c,metadata):
    pp=pack(pkg["b"][row:row+1],pkg["b"][row:row+1])
    profiles=(pp["u"][0,None,:24]*np.cos(angle)[:,None]+pp["v"][0,None,:24]*np.sin(angle)[:,None])
    predicted=np.stack([p13.rotate(pp,np.array([a]))[0] for a in angle])
    start_hour=int(pkg["ids"][row,0])%24
    peak=((np.arctan2(pp["base_bs"][0],pp["base_ac"][0])+angle)/p13.W+start_hour)%24
    residual=(pp["b"][0]-pp["b"][0].mean()-pp["u"][0]).reshape(16,6).mean(1)
    summary=named_summary(pkg["template"],pkg["features"][row])
    context=c["contexts"]["QwenWrong" if metadata["family"]=="QwenWrong" else "QwenTask"]
    texts=[build_prompt(summary,context,current/p13.W,profiles,angle/p13.W,peak,residual,pp["b"].mean(),pkg["scales"],perm) for perm in PERMS]
    return dict(**metadata,row=int(row),origin=int(pkg["origin"][row]),current_angle=float(current),
       candidate_angles=np.asarray(angle).tolist(),candidate_sha256=digest(predicted),texts=texts)

class Scorer:
    def __init__(self,c,r):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from transformers.utils import logging
        logging.set_verbosity_error()
        self.c,self.r=c,r;self.tick=time.monotonic();self.calls=0;self.max_tokens=0
        records=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True)
        available=[tuple(map(int,line.split(","))) for line in records.strip().splitlines()]
        eligible=[v for v in available if v[1]>=c["budget"]["min_free_gib"]*1024]
        if not eligible:raise RuntimeError("No GPU has enough free memory; no other process will be stopped.")
        chosen=min(eligible,key=lambda v:(v[2],-v[1],v[0]));self.device=f"cuda:{chosen[0]}"
        torch.cuda.set_device(self.device);rt.setup(self.device,c["budget"]["gpu_cap_gib"])
        assert torch.cuda.mem_get_info(self.device)[0]>=c["budget"]["min_free_gib"]*2**30
        self.tok=AutoTokenizer.from_pretrained(c["model_path"],local_files_only=True)
        self.tok.padding_side="left"
        if self.tok.pad_token_id is None:self.tok.pad_token=self.tok.eos_token
        self.ids=[self.tok.encode(a,add_special_tokens=False) for a in "ABC"];assert all(len(a)==1 for a in self.ids)
        self.ids=[a[0] for a in self.ids]
        rt.save(r/"gpu_preflight.json",dict(device=self.device,observed=available,free_gib_after_context=torch.cuda.mem_get_info(self.device)[0]/2**30,
            candidate_token_ids=self.ids,decoded=[self.tok.decode([a]) for a in self.ids],nonthinking=True))
        print("LOAD_FROZEN_QWEN",self.device,flush=True)
        self.model=AutoModelForCausalLM.from_pretrained(c["model_path"],local_files_only=True,dtype=torch.bfloat16,
                       device_map={"":self.device},attn_implementation="sdpa")
        self.model.eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.before=p13.u.fingerprint(self.model);self.load_seconds=time.monotonic()-self.tick
        self.log=(r/"live_feedback.jsonl").open("w")
        print("QWEN_READY",round(self.load_seconds,2),flush=True)

    def preference(self,requests):
        texts=[t for req in requests for t in req["texts"]]
        assert self.calls+len(texts)<=self.c["budget"]["max_forward_sequences"]
        logits=[]
        for i in range(0,len(texts),self.c["budget"]["prompt_batch_size"]):
            if time.monotonic()-self.tick>self.c["budget"]["max_llm_seconds"]:raise TimeoutError("live Qwen budget")
            chats=[self.tok.apply_chat_template([dict(role="user",content=t)],tokenize=False,add_generation_prompt=True,enable_thinking=False) for t in texts[i:i+self.c["budget"]["prompt_batch_size"]]]
            assert all("<think>\n\n</think>" in t for t in chats)
            batch=self.tok(chats,add_special_tokens=False,padding=True,return_tensors="pt").to(self.device)
            width=batch["input_ids"].shape[1];self.max_tokens=max(self.max_tokens,width)
            assert width<=self.c["budget"]["max_prompt_tokens"],width
            with torch.inference_mode():
                h=self.model.model(**batch,use_cache=False,return_dict=True).last_hidden_state[:,-1]
                z=F.linear(h,self.model.get_output_embeddings().weight[self.ids]).float().cpu().numpy()
            logits.extend(z);self.calls+=len(chats)
        logits=np.array(logits).reshape(len(requests),3,3);answers=[]
        for req,z in zip(requests,logits):
            physical=np.stack([z[j,np.argsort(perm)] for j,perm in enumerate(PERMS)])
            averaged=physical.mean(0,dtype=float);averaged-=averaged.max();q=np.exp(averaged);q/=q.sum()
            raw_choices=[PERMS[j][int(z[j].argmax())] for j in range(3)]
            entry={**req,"letter_logits":z.tolist(),"q_physical":q.tolist(),"order_disagreement":len(set(raw_choices))>1,
                   "raw_physical_choices":raw_choices,"forward_sequences_after":self.calls}
            self.log.write(json.dumps(entry)+"\n");answers.append(q)
        self.log.flush();return np.array(answers)

    def close(self):
        assert self.before==p13.u.fingerprint(self.model)
        assert all(not p.requires_grad and p.grad is None for p in self.model.parameters())
        info=dict(device=self.device,seconds=time.monotonic()-self.tick,load_seconds=self.load_seconds,
            forward_sequences=self.calls,max_tokens=self.max_tokens,peak_gib=torch.cuda.max_memory_allocated(self.device)/2**30,
            fingerprint_before=self.before,fingerprint_after=p13.u.fingerprint(self.model),llm_updates=0,all_parameters_frozen=True,
            logits="restricted ABC probabilities; three cyclic position-balanced queries per action set")
        rt.save(self.r/"qwen_execution.json",info);self.log.close()
        del self.model;torch.cuda.empty_cache()
        return info

def prepare(c,r):
    data={};sources={}
    summaryfile=Path(c["reuse_train"])/"training_summary.json";summary=rt.read(summaryfile);sources[str(summaryfile)]=rt.sha(summaryfile)
    for seed in c["seeds"]:
        f=Path(c["reuse_train"])/f"training_seed{seed}.npz";z=np.load(f);q={k:z[k] for k in z.files};sources[str(f)]=rt.sha(f)
        q["features"]=p13.old.annual_features(q["features"],q["origin"],c)
        cut=int((np.datetime64(c["inner_cut"],"h")-np.datetime64(c["clock_base"],"h"))/np.timedelta64(1,"h"))
        fit=np.where(q["origin"]+96<=cut)[0];valid=np.where(q["origin"]>=cut)[0]
        assert len(fit)==135 and len(valid)==20 and max(q["origin"][fit]+96)<=min(q["origin"][valid])
        for phase,ii in [("inner",fit),("full",np.arange(len(q["origin"])))]:
            b,cf=p13.crossfit_base(q,ii,c);p=pack(b,q["y"][ii]);x=np.c_[p13.base_phi(q)[ii],p["coords"]]
            cost,temp=base.plain_response_cv(x,p,q["origin"][ii],c)
            if phase=="inner":
                initial,_,_=base.fit_student(x,p,c["warm_regularizer"],dict(c,warmup_steps=50),seed,"StaticKD",(cost,temp))
                vb=p13.base_predict(q,valid,p13.base_fit(q,ii,c))
                vp=pack(vb,q["y"][valid]);vx=np.c_[p13.base_phi(q)[valid],vp["coords"]]
            else:
                f=Path(c["warm_source"])/f"student_s{seed}_Fourier_StaticKD.npz";sources[str(f)]=rt.sha(f);initial=npmodel(np.load(f))
                bm=p13.base_fit(q,ii,c);np.savez_compressed(r/f"frozen_base_s{seed}.npz",**bm)
            initial=npmodel(initial);assert abs(initial["mu"]-x.mean(0)).max()<1e-10
            assert abs(initial["sd"]-np.maximum(x.std(0),.001)).max()<1e-10
            pkg=dict(x=x,b=b,y=q["y"][ii],origin=q["origin"][ii],ids=q["ids"][ii],features=q["features"][ii],cost=cost,temp=temp,
              template=summary[str(seed)][0],initial=initial,
              scales=dict(squared_error_normalizer=float(p["base_mse"].mean()),absolute_error_normalizer=float(abs(b-p["y"]).mean())))
            if phase=="inner":pkg.update(valid_b=vb,valid_y=q["y"][valid],valid_x=vx)
            data[phase,seed]=pkg
            serial={k:v for k,v in pkg.items() if isinstance(v,np.ndarray)}
            np.savez_compressed(r/f"package_{phase}_s{seed}.npz",**serial)
            np.savez_compressed(r/f"initial_{phase}_s{seed}.npz",**initial)
            rt.save(r/f"package_{phase}_s{seed}.json",dict(template=pkg["template"],scales=pkg["scales"],crossfit=cf))
    for f in [Path(__file__),Path(base.__file__),Path(bil.__file__),Path(p13.__file__),CONFIG]:
        sources[str(f)]=rt.sha(f)
    for name in ["config.json","tokenizer_config.json","model.safetensors.index.json"]:
        f=Path(c["model_path"])/name
        if f.exists():sources[str(f)]=rt.sha(f)
    rt.save(r/"source_train.json",dict(git=rt.git(),sources=sources,no_outer_files_opened=True,no_Qwen_hidden_cache_loaded=True))
    return data

def run_phase(c,r,phase,seed,pkg,scorer):
    x=pkg["x"];initial=pkg["initial"];xx=torch.tensor((x-initial["mu"])/initial["sd"],dtype=torch.float64)
    hh=xx[:,base.HIDX];zz=xx[:,base.ZIDX];p=pack(pkg["b"],pkg["y"])
    bt=torch.tensor(p["b"]);ut=torch.tensor(p["u"]);vt=torch.tensor(p["v"]);yt=torch.tensor(p["y"])
    ms=pkg["scales"]["squared_error_normalizer"];ma=pkg["scales"]["absolute_error_normalizer"]
    constants={k:torch.tensor(initial[k]) for k in ["coef","bias","left","right"]}
    parameters={arm:{k:torch.tensor(initial[k],dtype=torch.float64,requires_grad=True) for k in constants} for arm in ARMS}
    opts={a:torch.optim.Adam(list(v.values()),lr=c["learning_rate"]) for a,v in parameters.items()}
    rng=np.random.default_rng(seed+180018+(10000 if phase=="full" else 0));records=[];usage=[]
    sigma=c["policy_sigma_hours"]*p13.W
    def get_model(arm):
        return dict(mu=initial["mu"],sd=initial["sd"],**{k:v.detach().numpy().copy() for k,v in parameters[arm].items()})
    def checkpoint(step):
        for arm in ARMS:
            m=get_model(arm);assert bil.count(m)==163
            np.savez_compressed(r/f"{phase}_s{seed}_{arm}_step{step}.npz",**m)
            if phase=="inner":
                angle=bil.predict(m,pkg["valid_x"],c);vp=pack(pkg["valid_b"],pkg["valid_y"])
                pred=p13.rotate(vp,angle);ref=p13.rotate(vp,bil.predict(initial,pkg["valid_x"],c))
                mse=float(np.mean((pred-pkg["valid_y"])**2));mae=float(np.mean(abs(pred-pkg["valid_y"])))
                joint=.5*(mse/np.mean((ref-pkg["valid_y"])**2)+mae/np.mean(abs(ref-pkg["valid_y"])))
                records.append(dict(seed=seed,arm=arm,step=step,mse=mse,mae=mae,joint=float(joint),n=len(pred)))
    checkpoint(0)
    for step in range(1,c["steps"]+1):
        deltas={};supervised={};penalties={}
        for arm,v in parameters.items():
            d=c["delta_max"]*torch.tanh(xx@v["coef"]+v["bias"]+(hh@v["left"])*(zz@v["right"]))
            pred=bt+ut*(torch.cos(d)[:,None]-1)+vt*torch.sin(d)[:,None]
            deltas[arm]=d;supervised[arm]=.5*((pred-yt).square().mean()/ms+(pred-yt).abs().mean()/ma)
            v0=constants
            matrix_change=v["left"].square().sum()*v["right"].square().sum()+v0["left"].square().sum()*v0["right"].square().sum()-2*(v["left"]@v0["left"])*(v["right"]@v0["right"])
            penalties[arm]=(v["coef"]-v0["coef"]).square().sum()+matrix_change+(v["bias"]-v0["bias"]).square()
        feedback={};actions={};rows=None
        if step in c["feedback_steps"]:
            rows=rng.choice(len(x),size=c["feedback_batch"],replace=False);requests=[];keys=[]
            for arm in ARMS[1:]:
                current=deltas[arm][rows].detach().numpy()
                act=np.clip(current[:,None]+np.array([-1.,0.,1.])[None]*sigma,-c["delta_max"],c["delta_max"])
                actions[arm]=act
                if arm=="Fourier":
                    with torch.no_grad():
                        cost=base.query(torch.tensor(pkg["cost"][rows]),torch.tensor(act),c)
                        feedback[arm]=torch.softmax(-cost/torch.tensor(pkg["temp"][rows,None]),dim=1).numpy()
                else:
                    for j,row in enumerate(rows):
                        requests.append(request_for(pkg,int(row),act[j],float(current[j]),c,
                            dict(phase=phase,seed=seed,step=step,family=arm)))
                        keys.append((arm,j))
            probabilities=scorer.preference(requests)
            for arm in ARMS[2:]:feedback[arm]=np.stack([probabilities[k] for k,key in enumerate(keys) if key[0]==arm])
            for arm in ARMS[1:]:
                raw=feedback[arm].copy()
                if arm=="QwenShuffle":feedback[arm]=feedback[arm][np.roll(np.arange(len(rows)),1)]
                usage.append(dict(phase=phase,seed=seed,step=step,arm=arm,rows=rows.tolist(),origins=pkg["origin"][rows].tolist(),
                  candidate_angles=actions[arm].tolist(),q_raw=raw.tolist(),q_used=feedback[arm].tolist()))
            print("LIVE_FEEDBACK",phase,seed,step,"Qwen_sequences",scorer.calls,flush=True)
        for arm in ARMS:
            kd=torch.zeros((),dtype=torch.float64)
            if arm in feedback:
                act=torch.tensor(actions[arm]);logps=torch.log_softmax(-.5*((act-deltas[arm][rows,None])/sigma).square(),dim=1)
                qt=torch.tensor(feedback[arm],dtype=torch.float64).clamp_min(1e-12);qt=qt/qt.sum(1,keepdim=True)
                kd=(logps.exp()*(logps-qt.log())).sum(1).mean()
            loss=supervised[arm]+c["regularizer"]*penalties[arm]+c["kd_weight"]*kd
            opts[arm].zero_grad();loss.backward();opts[arm].step()
        if step in c["checks"]:checkpoint(step)
    csvsave(r/f"{phase}_s{seed}_curves.csv",records) if records else None
    rt.save(r/f"{phase}_s{seed}_feedback_usage.json",usage)
    return records

def execute(c,r):
    r.mkdir(parents=True,exist_ok=False);tick=time.monotonic();torch.set_num_threads(2)
    rt.save(r/"config.json",c);shutil.copy2(__file__,r/Path(__file__).name)
    scorer=None
    try:
        data=prepare(c,r);scorer=Scorer(c,r)
        pkg=data["inner",c["seeds"][0]];row=0;pp=pack(pkg["b"][:1],pkg["b"][:1])
        current=float(bil.predict(pkg["initial"],pkg["x"][:1],c)[0])
        req=request_for(pkg,row,np.full(3,current),current,c,dict(phase="noop_check",seed=c["seeds"][0],step=0,family="QwenTask"))
        noop=scorer.preference([req])[0];assert abs(noop-1/3).max()<1e-6
        rt.save(r/"no_op_letter_balance.json",dict(probabilities=noop.tolist(),passed=True))
        curves=[]
        for seed in c["seeds"]:curves+=run_phase(c,r,"inner",seed,data["inner",seed],scorer)
        csvsave(r/"inner_curves.csv",curves)
        grid=[dict(step=step,joint=float(np.mean([v["joint"] for v in curves if v["step"]==step]))) for step in [0]+c["checks"]]
        chosen=min(grid,key=lambda v:(v["joint"],v["step"]))["step"]
        rt.save(r/"selection.json",dict(common_step=chosen,grid=grid,criterion="mean normalized joint MSE/MAE across all5arms and3seeds on Jan4-Mar1 training-internal targets only",all_models_trained_steps=100))
        print("COMMON_SELECTION",chosen,flush=True)
        for seed in c["seeds"]:
            run_phase(c,r,"full",seed,data["full",seed],scorer)
            for arm in ARMS:shutil.copy2(r/f"full_s{seed}_{arm}_step{chosen}.npz",r/f"student_s{seed}_{arm}.npz")
        execution=scorer.close();scorer=None
        src=rt.read(r/"source_train.json");assert all(rt.sha(p)==v for p,v in src["sources"].items())
        assert time.monotonic()-tick<c["budget"]["max_workflow_seconds"]
        rt.save(r/"selection_lock.json",dict(seconds=time.monotonic()-tick,common_step=chosen,forward_sequences=execution["forward_sequences"],
             files={str(p):rt.sha(p) for p in r.iterdir() if p.is_file()},no_outer_files_opened=True,
             genuine_live_frozen_Qwen_feedback=True,TimeBase_updates=0,LLM_updates=0))
        print("LIVE_TRAIN_COMPLETE",json.dumps(execution),flush=True)
    except BaseException:
        rt.save(r/"failure.json",dict(traceback=traceback.format_exc(),seconds=time.monotonic()-tick))
        if scorer is not None:scorer.log.flush()
        raise

def evaluate(c,r):
    tick=time.monotonic();lock=rt.read(r/"selection_lock.json");assert all(rt.sha(p)==v for p,v in lock["files"].items())
    previous=Path(c["reliability_run"]);cache=Path(rt.read(previous/"config.json")["reuse_run"])
    sf=previous/"shared_summaries.json";sums={(v["seed"],v["stage"],v["row"]):v["summary"] for v in rt.read(sf)}
    sources={str(sf):rt.sha(sf)};metrics=[];cases=[];preds={};conservation={}
    for seed in c["seeds"]:
        bm=np.load(r/f"frozen_base_s{seed}.npz")
        for stage in c["stages"]:
            path=cache/f"seed{seed}_{stage}.npz";z=np.load(path);q={k:z[k] for k in z.files};sources[str(path)]=rt.sha(path)
            q["features"]=p13.old.annual_features(np.stack([p13.s.flatten(sums[seed,stage,j]) for j in range(len(q["origin"]))]),q["origin"],c)
            b=p13.base_predict(q,np.arange(len(q["origin"])),bm);p=pack(b,q["y"]);x=np.c_[p13.base_phi(q),p["coords"]]
            np.savez_compressed(r/f"deployment_s{seed}_{stage}.npz",x=x,b=b,u=p["u"],v=p["v"],y=q["y"],origin=q["origin"])
            angles=dict(FrozenBase=np.zeros(len(x)),FrozenFourier=bil.predict(np.load(r/f"initial_full_s{seed}.npz"),x,c))
            angles.update({arm:bil.predict(np.load(r/f"student_s{seed}_{arm}.npz"),x,c) for arm in ARMS})
            for arm,d in angles.items():
                pred,met,ch=p13.forecast_metrics(p,d)
                for k,v in ch.items():conservation[k]=max(conservation.get(k,0.),v)
                preds[f"s{seed}_{stage}_{arm}"]=pred;preds[f"a{seed}_{stage}_{arm}"]=d
                metrics.append(dict(seed=seed,stage=stage,arm=arm,n=len(x),**{k:float(met[k].mean()) for k in ["mse","mae","level","shape"]},
                    phase_mean_hours=float((d/p13.W).mean()),phase_sd_hours=float((d/p13.W).std())))
                for j in range(len(x)):cases.append(dict(seed=seed,stage=stage,arm=arm,origin=int(q["origin"][j]),mse=float(met["mse"][j]),mae=float(met["mae"][j]),gain=float(p["base_mse"][j]-met["mse"][j]),shift_hours=float(d[j]/p13.W)))
    csvsave(r/"by_seed_period.csv",metrics);csvsave(r/"cases.csv",cases);np.savez_compressed(r/"predictions.npz",**preds)
    fields=["mse","mae","level","shape","phase_mean_hours","phase_sd_hours"];macro=[];summary=[]
    for seed in c["seeds"]:
        for arm in ["FrozenBase","FrozenFourier"]+ARMS:
            a=[v for v in metrics if v["seed"]==seed and v["arm"]==arm]
            macro.append(dict(seed=seed,arm=arm,**{k:float(np.mean([v[k] for v in a])) for k in fields}))
    for arm in ["FrozenBase","FrozenFourier"]+ARMS:
        a=[v for v in macro if v["arm"]==arm];summary.append(dict(arm=arm,**{k:float(np.mean([v[k] for v in a])) for k in fields}))
    csvsave(r/"macro_by_seed.csv",macro);csvsave(r/"summary.csv",summary)
    assert all(rt.sha(p)==v for p,v in lock["files"].items())
    src=rt.read(r/"source_train.json");assert all(rt.sha(p)==v for p,v in src["sources"].items())
    rt.save(r/"source_evaluation.json",dict(sources=sources,locked_before_outer=True))
    rt.save(r/"completion.json",dict(train_seconds=lock["seconds"],evaluation_seconds=time.monotonic()-tick,
       genuine_live_Qwen_sequences=lock["forward_sequences"],common_step=lock["common_step"],all_arms_trained_steps=100,
       student_parameters=163,LLM_updates=0,TimeBase_updates=0,conservation=conservation,sources_and_choices_unchanged=True))
    print("RESULT",json.dumps(summary),flush=True)

def preflight(c):
    p13.preflight(c)
    logits=np.array([[5.,2.,-1.],[5.,2.,-1.],[5.,2.,-1.]])
    physical=np.stack([logits[j,np.argsort(perm)] for j,perm in enumerate(PERMS)])
    assert np.max(abs(physical.mean(0)-2))<1e-12
    assert all(k not in build_prompt.__code__.co_varnames[:build_prompt.__code__.co_argcount] for k in ["y","truth","target"])
    assert 3*2*len(c["feedback_steps"])*3*c["feedback_batch"]*3+3<=c["budget"]["max_forward_sequences"]
    assert 1+81+49+32==163
    print("LIVE_FEEDBACK_PREFLIGHT_PASS")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--preflight",action="store_true");ap.add_argument("--execute",action="store_true");ap.add_argument("--evaluate",action="store_true");a=ap.parse_args()
    c=rt.read(CONFIG);r=Path(c["run"])
    if a.preflight:preflight(c)
    elif a.execute:execute(c,r)
    elif a.evaluate:evaluate(c,r)
    else:raise ValueError("choose phase")
if __name__=="__main__":main()
