"""Bounded fixed temporal shape/history amplitude pilot; only saved 2016 inputs."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",CUBLAS_WORKSPACE_CONFIG=":4096:8",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,time,json,csv
from pathlib import Path
import numpy as np,torch
from torch import nn
from exp.exp_chronogeom_event_knowledge19 import Head,forward,scores,read,save,sha,git
from exp.exp_chronogeom_downstream import CalendarTimeBase,predict
from exp.exp_chronogeom_residual03 import numeric_summary

def center(x):return x-x.mean(1,keepdim=True)
def raw_calendar(ix):
    h=ix%24;wd=(ix//24+4)%7;w=(wd>=5).astype(int);phase=2*np.pi*(ix//24)/366
    raw=np.stack([np.zeros(ix.shape),w,1-w,np.sin(2*np.pi*wd/7),np.cos(2*np.pi*wd/7),np.sin(phase),np.cos(phase)],-1)
    hh=np.stack([fn(2*np.pi*k*h/24) for k in range(1,5) for fn in [np.sin,np.cos]],-1)
    common=np.concatenate([hh*(w==k)[...,None] for k in range(3)],-1)
    return h,w,raw,common
def potential(profile):
    D=[];b=[];edges=[]
    for day in range(2):
        for h in range(24):
            for rel,ref in enumerate([(1-day,h),(day,3),(day,12)]):
                a=np.zeros(48);a[day*24+h]+=1;a[ref[0]*24+ref[1]]-=1
                D.append(a);b.append(profile[day,h,rel]);edges.append([day,h,*ref])
    D=np.array(D);b=np.array(b);A=np.concatenate([D,np.ones((1,48))]);u,_,rank,sv=np.linalg.lstsq(A,np.r_[b,0.],rcond=1e-12)
    assert rank==48 and abs(u.mean())<1e-10
    du=D@u;mask=abs(b)>1e-6;energy=float(np.mean(b*b));rms=max(float(np.sqrt(np.mean(u*u))),1e-12)
    return (u/rms).reshape(2,24),dict(rank=int(rank),incidence_rank=int(np.linalg.matrix_rank(D)),condition=float(sv[0]/sv[-1]),edge_rmse=float(np.sqrt(np.mean((du-b)**2))),edge_relative_residual=float(np.mean((du-b)**2)/max(energy,1e-30)),sign_agreement_nonzero=float(np.mean(np.sign(du[mask])==np.sign(b[mask]))) if mask.any() else None,potential_raw_rms=rms,self_edge_count=int((abs(D).sum(1)==0).sum()),potential=u.tolist(),edges=edges)
def fourier(h,w,H):
    cols=[np.ones(h.shape)]+[fn(2*np.pi*k*h/24) for k in range(1,H+1) for fn in [np.sin,np.cos]]
    return np.stack([col*(w==day) for day in range(2) for col in cols],-1)
class Amplitude(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(8,1);nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)
    def forward(self,u,phi,ref):
        a=self.linear(u).double()[:,None,:]
        return ref+center(a*phi),a
def summary8(x,b):
    return torch.stack([x[:,:,0].mean(1),x[:,:,0].std(1,unbiased=False),x[:,-1,0],x[:,-24:,0].mean(1),x[:,-24:,0].std(1,unbiased=False),x[:,-24:,0].mean(1)-x[:,:24,0].mean(1),b[:,:,0].mean(1),b[:,:,0].std(1,unbiased=False)],1)
def slice_scores(p,y,origin,cut):
    masks={"val":np.ones(len(origin),bool),"cal":origin+96<=cut,"check":origin>=cut}
    return {name:dict(windows=int(mask.sum()),**scores(p[mask],y[mask])) for name,mask in masks.items()}

def run(c,config):
    started=time.monotonic();src=Path(c["source"]);r=Path(c["root"]);r.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.mem_get_info()[0]>3*2**30;torch.cuda.set_per_process_memory_fraction(c["memory_fraction"]);device=torch.device("cuda:0")
    save(r/"config.json",c);selection=read(src/"selection_buildings.json");save(r/"selection_buildings.json",selection)
    assets=read(src/"old_assets.json")
    for meta in selection["selected"]:
        d=src/meta["building"]
        for name in ["head_checkpoints.pt","fit_inputs.npz","normalizers.json","loss_scales.json","config.json","curves.json"]+[f"FrozenTimeBase_{s}.pt" for s in c["seeds"]]:assets.append(dict(path=str(d/name),sha=sha(d/name)))
    assets.append(dict(path=str(src/"profiles.json"),sha=sha(src/"profiles.json")))
    for a in assets:assert sha(a["path"])==a["sha"]
    save(r/"old_assets.json",assets);save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),only_saved_2016_arrays=True,LLM_calls=0))
    profiles=read(src/"profiles.json");tables={};diagnostics={}
    for cat in ["Office","Education","Healthcare"]:tables[cat],diagnostics[cat]=potential(np.array(profiles[cat]))
    save(r/"graph_diagnostics.json",diagnostics);save(r/"fixed_shapes.json",{k:v.tolist() for k,v in tables.items()})
    cut=int((np.datetime64(c["selection_cut"],"h")-np.datetime64("2016-01-01","h"))/np.timedelta64(1,"h"))
    rows=[];curves=[];bank={};fits=[];initial_errors=[];mean_errors=[];replay_errors=[]
    def guard(limit):assert time.monotonic()-started<limit,"Activity budget; preserve partial run"
    for meta in selection["selected"]:
        name=meta["building"];d=src/name;dr=r/name;dr.mkdir();cc=read(d/"config.json");inp=np.load(d/"fit_inputs.npz");states=torch.load(d/"head_checkpoints.pt",map_location="cpu",weights_only=True)
        oldnorm=read(d/"normalizers.json");norms=read(d/"loss_scales.json");case={};coeffs={};preps={};weights={}
        for seed in c["seeds"]:
            base=CalendarTimeBase(cc,seed,np.zeros((168,168),np.float32)).to(device);base.load_state_dict(torch.load(d/f"FrozenTimeBase_{seed}.pt",map_location=device,weights_only=True));base.eval()
            for p in base.parameters():p.requires_grad_(False)
            sd={};oldheads={}
            for arm in ["TeacherFourier","TeacherQwen","TeacherWrong","NoKD","FourierKD","WrongKD","QwenTaskKD"]:
                m=Head(seed).to(device);m.load_state_dict(states[seed][arm][600 if arm.startswith("Teacher") else 500]);m.eval()
                for p in m.parameters():p.requires_grad_(False)
                oldheads[arm]=m
            for split in ["fit","val"]:
                x=torch.tensor(inp[f"{split}_x"],dtype=torch.float32,device=device);ids=torch.tensor(inp[f"{split}_ids"],device=device);y=inp[f"{split}_y"];ix=inp[f"{split}_future"];origin=inp[f"{split}_origin"]
                assert ix.max()<8472 and ix.min()>=0 and not np.intersect1d(inp["fit_future"].ravel(),inp["val_future"].ravel()).size
                b=torch.tensor(predict(base,dict(x=x,ids=ids)),device=device);h,w,raw,common=raw_calendar(ix);num=numeric_summary(x,b).cpu().numpy();f=np.concatenate([num,raw,common],-1);z={}
                for fa in ["Fourier","Qwen","Wrong"]:
                    n=oldnorm[str(seed)][fa]
                    K=np.stack([np.sin(10*np.pi*h/24),np.cos(10*np.pi*h/24),np.cos(12*np.pi*h/24)],-1) if fa=="Fourier" else np.array(profiles[cc["category"] if fa=="Qwen" else cc["wrong_context"][cc["category"]]])[w,h]
                    z[fa]=torch.tensor(np.concatenate([(f-n["mean"])/n["sd"],(K-n["km"])/n["ks"]],-1),dtype=torch.float32,device=device)
                with torch.no_grad():
                    ref=forward(oldheads["TeacherFourier"],z["Fourier"],b)
                    oldpred={a:forward(m,z["Qwen" if a=="TeacherQwen" else "Wrong" if a=="TeacherWrong" else "Fourier"],b).cpu().numpy() for a,m in oldheads.items()}
                sd[split]=dict(x=x,b=b,y=torch.tensor(y,dtype=torch.float64,device=device),ref=ref,h=h,w=w,origin=origin,z=z["Fourier"],u=summary8(x,b),old=oldpred)
            u=sd["fit"]["u"];um=u.mean(0);us=u.std(0,unbiased=False).clamp_min(1e-6)
            # Only 2016 fit targets construct the strong extra Fourier fixed-shape control.
            B=fourier(sd["fit"]["h"],sd["fit"]["w"],c["fourier_harmonics"]);Bc=B-B.mean(1,keepdims=True);target=center(sd["fit"]["y"]-sd["fit"]["ref"]).cpu().numpy().reshape(-1)
            beta,_,rank,_=np.linalg.lstsq(Bc.reshape(-1,Bc.shape[-1]),target,rcond=1e-12);coeffs[seed]=dict(coefficients=beta.tolist(),effective_rank=int(rank),fitted_coefficients=len(beta))
            phi={a:{} for a in ["QwenShape","WrongShape","FourierShape"]}
            for split,p in sd.items():
                p["u"]=(p["u"]-um)/us
                arrs={"QwenShape":tables[cc["category"]][p["w"],p["h"]],"WrongShape":tables[cc["wrong_context"][cc["category"]]][p["w"],p["h"]],"FourierShape":fourier(p["h"],p["w"],c["fourier_harmonics"])@beta}
                for a,v in arrs.items():phi[a][split]=torch.tensor(v[:,:,None],dtype=torch.float64,device=device);phi[a][split]=center(phi[a][split])
            scales={}
            for arm in phi:
                scale=max(float(phi[arm]["fit"].square().mean().sqrt()),1e-12);scales[arm]=scale
                for split in sd:phi[arm][split]/=scale
            preps[seed]=dict(history_mean=um.cpu().tolist(),history_sd=us.cpu().tolist(),shape_scales=scales)
            for a,p in sd["val"]["old"].items():
                for period,met in slice_scores(p,sd["val"]["y"].cpu().numpy(),sd["val"]["origin"],cut).items():rows.append(dict(building=name,category=cc["category"],seed=seed,arm="Old"+a,period=period,**met))
                curve=next(t for t in read(d/"curves.json") if t["seed"]==seed and t["arm"]==a and t["step"]==(600 if a.startswith("Teacher") else 500))
                replay_errors.append(max(abs(scores(p,sd["val"]["y"].cpu().numpy())[k]-curve[k]) for k in ["mse","mae","level","shape"]))
            weights[seed]={};teachers={}
            for arm in phi:
                m=Amplitude().to(device);assert sum(p.numel() for p in m.parameters())==9
                with torch.no_grad():init,_=m(sd["fit"]["u"],phi[arm]["fit"],sd["fit"]["ref"])
                initial_errors.append(float(abs(init-sd["fit"]["ref"]).max()));assert initial_errors[-1]==0
                opt=torch.optim.Adam(m.parameters(),lr=c["teacher_lr"],foreach=False);batches=np.random.default_rng(seed).integers(0,len(u),size=(600,32));norm=norms[str(seed)]
                for step in range(601):
                    if step%100==0:
                        with torch.no_grad():pv,av=m(sd["val"]["u"],phi[arm]["val"],sd["val"]["ref"])
                        curves.append(dict(building=name,seed=seed,arm=arm,step=step,**scores(pv.cpu().numpy(),sd["val"]["y"].cpu().numpy())))
                    if step==600:break
                    ix=torch.as_tensor(batches[step],device=device);pr,_=m(sd["fit"]["u"][ix],phi[arm]["fit"][ix],sd["fit"]["ref"][ix]);e=pr-sd["fit"]["y"][ix]
                    loss=.5*(e.square().mean()/norm["mse"]+e.abs().mean()/norm["mae"]);opt.zero_grad();loss.backward();opt.step();guard(c["teacher_seconds"])
                weights[seed][arm]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};teachers[arm]={}
                for split,p in sd.items():
                    with torch.no_grad():pv,av=m(p["u"],phi[arm][split],p["ref"])
                    teachers[arm][split]=pv.detach();mean_errors.append(float(abs((pv-p["ref"]).mean(1)).max()))
                    if split=="val":
                        pp=pv.cpu().numpy()
                        for period,met in slice_scores(pp,p["y"].cpu().numpy(),p["origin"],cut).items():rows.append(dict(building=name,category=cc["category"],seed=seed,arm=arm,period=period,**met))
                        np.savez_compressed(dr/f"{seed}_{arm}_val.npz",prediction=pp,amplitude=av.cpu().numpy(),y=p["y"].cpu().numpy(),origin=p["origin"])
                fits.append(dict(building=name,seed=seed,arm=arm,role="teacher",parameters=9,updates=600,step=600))
                print("TEACHER",name,seed,arm,flush=True)
            case[seed]=dict(sd=sd,teachers=teachers,norm=norms[str(seed)])
        torch.save(weights,dr/"amplitude_weights.pt");save(dr/"preprocessors.json",preps);save(dr/"fourier_coefficients.json",coeffs);bank[name]=case
        save(r/"teacher_progress.json",dict(fits=len(fits),seconds=time.monotonic()-started));guard(c["teacher_seconds"])
    assert len(fits)==54;save(r/"teacher_metrics.json",rows);save(r/"teacher_curves.json",curves)
    lock=[dict(path=str(p),sha=sha(p)) for p in r.glob("*/*") if p.name in ["amplitude_weights.pt","preprocessors.json","fourier_coefficients.json"]];save(r/"teacher_lock.json",dict(files=lock,before_KD_decision=True))
    def pooled(arm,period):
        v=[x for x in rows if x["arm"]==arm and x["period"]==period]
        return {k:float(np.mean([x[k] for x in v])) for k in ["mse","mae","level","shape"]}
    def ratios(arm,ref,period,seed=None):
        v=[x for x in rows if x["arm"]==arm and x["period"]==period and (seed is None or x["seed"]==seed)]
        return {k:float(np.mean([x[k]/next(b for b in rows if b["arm"]==ref and b["period"]==period and b["seed"]==x["seed"] and b["building"]==x["building"])[k] for x in v])) for k in ["mse","mae"]}
    tests=[]
    for ref in ["OldTeacherFourier","FourierShape"]:
        rr=ratios("QwenShape",ref,"cal");sr={s:ratios("QwenShape",ref,"cal",s) for s in c["seeds"]}
        tests.append(dict(reference=ref,ratios=rr,seed_ratios=sr,pass_=all(v<1 for v in rr.values()) and sum(all(v<1 for v in z.values()) for z in sr.values())>=2))
    gate=all(t["pass_"] for t in tests);save(r/"KD_decision.json",dict(pass_=gate,tests=tests,period="cal",no_per_building_switch=True));print("KD_DECISION",gate,tests,flush=True)
    if gate:
        for name,case in bank.items():
            dr=r/name;studentstates={}
            for seed,rec in case.items():
                sd=rec["sd"];n=rec["norm"];studentstates[seed]={}
                for arm,teacher in [("QwenShapeKD","QwenShape"),("FourierShapeKD","FourierShape"),("WrongShapeKD","WrongShape")]:
                    model=Head(seed).to(device);opt=torch.optim.Adam(model.parameters(),lr=c["student_lr"],foreach=False);batches=np.random.default_rng(seed).integers(0,len(sd["fit"]["y"]),size=(600,32));target=center(rec["teachers"][teacher]["fit"]);selected=None
                    for step in range(601):
                        if step%100==0:
                            with torch.no_grad():pv=forward(model,sd["val"]["z"],sd["val"]["b"]).cpu().numpy()
                            curves.append(dict(building=name,seed=seed,arm=arm,step=step,**scores(pv,sd["val"]["y"].cpu().numpy())))
                            if step==c["student_step"]:selected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        if step==600:break
                        ix=torch.as_tensor(batches[step],device=device);p=forward(model,sd["fit"]["z"][ix],sd["fit"]["b"][ix]);e=p-sd["fit"]["y"][ix]
                        loss=.5*(e.square().mean()/n["mse"]+e.abs().mean()/n["mae"])+c["kd_weight"]*(center(p)-target[ix]).square().mean()/n["mse"]
                        opt.zero_grad();loss.backward();opt.step();guard(c["total_seconds"])
                    model.load_state_dict(selected);studentstates[seed][arm]=selected
                    with torch.no_grad():pv=forward(model,sd["val"]["z"],sd["val"]["b"]).cpu().numpy()
                    for period,met in slice_scores(pv,sd["val"]["y"].cpu().numpy(),sd["val"]["origin"],cut).items():rows.append(dict(building=name,category=next(m["category"] for m in selection["selected"] if m["building"]==name),seed=seed,arm=arm,period=period,**met))
                    np.savez_compressed(dr/f"{seed}_{arm}_val.npz",prediction=pv,y=sd["val"]["y"].cpu().numpy(),origin=sd["val"]["origin"])
                    fits.append(dict(building=name,seed=seed,arm=arm,role="student",parameters=705,updates=600,step=500));print("STUDENT",name,seed,arm,flush=True)
            torch.save(studentstates,dr/"student_weights.pt")
    for a in assets+lock:assert sha(a["path"])==a["sha"]
    assert max(initial_errors)==0 and max(mean_errors)<1e-10
    save(r/"metrics.json",rows);save(r/"curves.json",curves);save(r/"fits.json",fits)
    arms=sorted({x["arm"] for x in rows});summary={period:{arm:pooled(arm,period) for arm in arms} for period in ["val","cal","check"]}
    save(r/"summary.json",summary)
    with (r/"by_seed.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(r/"verification.json",dict(initial_error=max(initial_errors),window_mean_error=max(mean_errors),old_curve_replay_max_error=max(replay_errors),old_assets_checked=len(assets),teacher_amplitude_parameters=9,teacher_Fourier_template_fitted_coefficients=26,old_teacher_reference_parameters=705,students_parameters=705,LLM_calls=0,new_2017_access=False,old_assets_unchanged=True,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30))
    save(r/"status.json",dict(complete=True,teacher_fits=54,student_fits=len(fits)-54,teacher_updates=32400,student_updates=(len(fits)-54)*600,KD_gate=gate,seconds=time.monotonic()-started,git=git(),only2016=True))
    print("COMPLETE",read(r/"status.json"),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args();c=read(a.config);run(c,a.config)
