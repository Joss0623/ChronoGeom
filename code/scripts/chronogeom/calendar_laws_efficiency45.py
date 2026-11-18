"""Calendar laws, function-retention bound and CPU inference; no training."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import argparse,csv,json,time,platform
from pathlib import Path
import numpy as np,torch
from exp.exp_chronogeom_event_knowledge19 import Head,knowledge,read,save,sha,git
from exp.exp_chronogeom_capital_representation_transfer44 import CalendarMap,predict
from exp.exp_chronogeom_downstream import CalendarTimeBase
from exp.exp_chronogeom_teacher_content40 import calendar_arrays
from exp.exp_chronogeom_residual03 import numeric_summary

def linear_shift_checks(name,z,shifts,pairs):
 z=np.asarray(z,float);D=z.shape[-1];h=np.tile(np.arange(24),3);rows=[];compositions=[]
 for context in ["pooled",0,1,2]:
  a=z.reshape(72,D) if context=="pooled" else z[context];hours=h if context=="pooled" else np.arange(24)
  sv=np.linalg.svd(a-a.mean(0),compute_uv=False);rank=int(np.sum(sv>max(sv[0],1e-30)*1e-10))
  for delta in shifts:
   target=np.roll(z,-delta,axis=1).reshape(72,D) if context=="pooled" else np.roll(a,-delta,axis=0);p=np.zeros_like(target)
   for parity in [0,1]:
    train=hours%2==parity;test=~train;X=np.c_[a[train],np.ones(train.sum())];A=np.linalg.lstsq(X,target[train],rcond=1e-10)[0];p[test]=np.c_[a[test],np.ones(test.sum())]@A
   sse=float(np.sum((p-target)**2));sst=float(np.sum((target-target.mean(0))**2));rows.append(dict(name=name,context=context,shift=delta,dimension=D,rank=rank,cv_residual_ratio=sse/max(sst,1e-30),cv_R2=1-sse/max(sst,1e-30)))
  if context=="pooled":
   maps={}
   for delta in range(24):
    target=np.roll(z,-delta,axis=1).reshape(72,D);A=np.linalg.lstsq(np.c_[a,np.ones(len(a))],target,rcond=1e-10)[0];M=np.zeros((D+1,D+1));M[:,:D]=A;M[-1,-1]=1;maps[delta]=M
   for u,v in pairs:
    lhs=(np.c_[a,np.ones(len(a))]@maps[u%24]@maps[v%24])[:,:D];rhs=(np.c_[a,np.ones(len(a))]@maps[(u+v)%24])[:,:D]
    compositions.append(dict(name=name,delta1=u,delta2=v,relative_composition_error=float(np.linalg.norm(lhs-rhs)/max(np.linalg.norm(a-a.mean(0)),1e-15))))
 return rows,compositions

def run(c,config):
 start=time.monotonic();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",c)
 old=Path(c["transfer44"]);oc=read(old/"config.json");src=Path(oc["source"]);teacher=Path(oc["teacher40"]);assets=read(old/"input_assets.json")+[dict(path=str(p),sha=sha(p)) for p in old.iterdir() if p.is_file() and p.suffix in [".pt",".npz",".json"]]
 save(r/"inputs.json",assets);save(r/"provenance.json",dict(git=git(),source_sha=sha(__file__),config_sha=sha(config),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__))
 profiles=read(teacher/"Capital/profiles.json");kind=np.repeat(np.arange(3),24);h=np.tile(np.arange(24),3);target=knowledge(np.array(profiles["Task"]),kind,h)-knowledge(np.array(profiles["Generic"]),kind,h);norm=read(old/"teacher_normalization.json")["2024"]["Delta"];target=(target-np.array(norm["mean"]))/norm["rms"]
 bundle=torch.load(old/"student_inference_bundle.pt",map_location="cpu",weights_only=True);students=bundle["students"];Z={"TeacherDelta3":target.reshape(3,24,3)}
 for seed in c["seeds"]:Z["StudentDelta3_"+str(seed)]=students[seed]["QwenTransferShapeKD"]["mapper"]["codes.weight"].double().numpy().reshape(3,24,3)
 Z["LegacyFourier3"]=np.stack([np.sin(10*np.pi*h/24),np.cos(10*np.pi*h/24),np.cos(12*np.pi*h/24)],-1).reshape(3,24,3)
 Z["SinCos2"]=np.stack([np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24)],-1).reshape(3,24,2)
 Z["PairedFourier6"]=np.stack([f(2*np.pi*k*h/24) for k in [1,2,3] for f in [np.sin,np.cos]],-1).reshape(3,24,6)
 rows=[];comp=[];spectra=[];context=[]
 for name,z in Z.items():
  rr,cc=linear_shift_checks(name,z,c["phase_shifts"],c["composition_pairs"]);rows.extend(rr);comp.extend(cc)
  for k in range(3):
   a=z[k]-z[k].mean(0);f=np.fft.rfft(a,axis=0);energy=np.sum(abs(f)**2,axis=1);energy[1:-1]*=2;energy[0]=0;energy=energy/max(energy.sum(),1e-30)
   sv=np.linalg.svd(a,compute_uv=False);spectra.append(dict(name=name,context=k,first_harmonic_fraction=float(energy[1]),first_two_fraction=float(energy[1:3].sum()),first_four_fraction=float(energy[1:5].sum()),energy=energy.tolist(),rank=int(np.sum(sv>max(sv[0],1e-30)*1e-10))))
  if z.shape[-1]==3 and "Delta" in name:
   context.append(dict(name=name,base_channel_context_difference_max=float(abs(z[:,:,0]-z[0:1,:,0]).max()),weekend_channel_unintended_max=float(abs(z[[0,2],:,1]).max()),holiday_channel_unintended_max=float(abs(z[[0,1],:,2]).max())))
 save(r/"phase_shift_cv.json",rows);save(r/"composition.json",comp);save(r/"spectra.json",spectra);save(r/"context_factorization.json",context)
 # Q is a symmetric orthogonal projector; bound representation errors under shared reader.
 Q=np.eye(96)-np.ones((96,96))/96;projection=dict(symmetry_max=float(abs(Q-Q.T).max()),idempotence_max=float(abs(Q@Q-Q).max()),constant_null_max=float(abs(Q@np.ones(96)).max()),operator_norm=float(np.linalg.norm(Q,2)))
 bounds=[];preds=np.load(old/"predictions.npz");packs=np.load(src/"data.npz");cfg=read(src/"config.json");grid,raw,common,daykind,hour=calendar_arrays(cfg,"Capital");calendar_ids=(daykind*24+hour).astype(int)
 Lgelu=1+1/np.sqrt(2*np.pi*np.e)
 for seed in c["seeds"]:
  st=students[seed]["QwenTransferShapeKD"]["head"];Wtime=st["net.0.weight"][:,-3:].double().numpy();Wout=st["net.2.weight"].double().numpy();lip=float(Lgelu*np.linalg.norm(Wtime,2)*np.linalg.norm(Wout,2));codes=students[seed]["QwenTransferShapeKD"]["mapper"]["codes.weight"].double().numpy();dz=codes-target
  for split,key in [("fit","fit"),("val","val"),("test","eval")]:
   dx=dz[calendar_ids[packs[split+"_future"]]];upper=lip*np.sqrt(np.sum(dx**2,axis=(1,2)));err=preds[f"{seed}_{key}_QwenTransferShapeKD"]-preds[f"{seed}_{key}_ExactCopy_Delta"];actual=np.sqrt(np.sum(err**2,axis=(1,2)));assert np.all(actual<=upper+2e-6)
   bounds.append(dict(seed=seed,split=key,lipschitz_bound=lip,max_representation_error=float(abs(dz).max()),max_prediction_l2_error=float(actual.max()),max_prediction_l2_bound=float(upper.max()),maximum_ratio=float(np.max(actual/np.maximum(upper,1e-30))),verified=bool(np.all(actual<=upper+2e-6))))
 save(r/"projection.json",projection);save(r/"function_retention_bound.json",bounds)
 # Benchmarks: no teacher/Qwen loads, CPU only, fixed24 calendar states from existing eval metadata.
 seed=2024;base=CalendarTimeBase(cfg,seed,np.zeros((168,168),np.float32));base.load_state_dict(bundle["frozen_bases"][seed]);base.eval();mappers={};heads={}
 for arm in ["QwenTransferShapeKD","AnalyticNoKD"]:
  mappers[arm]=CalendarMap(seed);mappers[arm].load_state_dict(students[seed][arm]["mapper"]);mappers[arm].eval();heads[arm]=Head(seed);heads[arm].load_state_dict(students[seed][arm]["head"]);heads[arm].eval()
 exact=CalendarMap(seed);exact.codes.weight.data.copy_(torch.tensor(target,dtype=torch.float32));exact.eval()
 pp=read(old/"deployment_preprocessing.json")["common39"][str(seed)];mu=torch.tensor(pp["mean"],dtype=torch.float64);sd=torch.tensor(pp["sd"],dtype=torch.float64);latency=[]
 for batch in [1,32]:
  x=torch.tensor(packs["test_x"][:batch],dtype=torch.float32);ids168=torch.tensor(packs["test_ids"][:batch],dtype=torch.long);ix=packs["test_future"][:batch];cid=torch.tensor(calendar_ids[ix],dtype=torch.long);cal=torch.tensor(np.concatenate([raw[ix],common[ix]],-1),dtype=torch.float64)
  def full(arm,use_exact=False):
   bp=base(x,ids168);z39=((torch.cat([numeric_summary(x,bp).double(),cal],-1)-mu)/sd).float()
   return predict(exact if use_exact else mappers[arm],heads[arm],z39,cid,bp)
  funcs={"OriginalTimeBase40":lambda:base.body(x),"FrozenBase6570":lambda:base(x,ids168),"QwenStudent7491":lambda:full("QwenTransferShapeKD"),"AnalyticStudent7491":lambda:full("AnalyticNoKD"),"CachedTeacher7491":lambda:full("QwenTransferShapeKD",True)}
  samples={k:[] for k in funcs};rng=np.random.default_rng(2024)
  with torch.inference_mode():
   for fn in funcs.values():
    for _ in range(30):fn()
   for round in range(5):
    for name in rng.permutation(list(funcs)):
     for _ in range(60):
      t=time.perf_counter_ns();funcs[name]();samples[name].append((time.perf_counter_ns()-t)/1e6)
      assert time.monotonic()-start<180
  for name,v in samples.items():latency.append(dict(model=name,batch=batch,repeats=len(v),median_ms=float(np.median(v)),p10_ms=float(np.quantile(v,.1)),p90_ms=float(np.quantile(v,.9)),median_ms_per_window=float(np.median(v)/batch)))
 save(r/"latency_cpu.json",latency)
 cpu=next((l.split(":",1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")),"unknown")
 for a in assets:assert sha(a["path"])==a["sha"]
 save(r/"status.json",dict(complete=True,seconds=time.monotonic()-start,CPU=cpu,threads=torch.get_num_threads(),training_updates=0,optimizer_steps=0,LLM_calls=0,GPU_loaded=False,old_assets_unchanged=True,assets=len(assets),CPU_benchmark_excludes_data_IO=True))
 print("PHASE",json.dumps([x for x in rows if x["context"]=="pooled"]));print("SPECTRA",json.dumps([x for x in spectra if x["name"]=="TeacherDelta3"]));print("LATENCY",json.dumps(latency));print("BOUNDS",json.dumps(bounds));print("STATUS",read(r/"status.json"))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args();run(read(a.config),a.config)
