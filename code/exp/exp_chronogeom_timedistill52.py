"""Native TimeDistill model/full-loss pilot; separate process isolates its TSLib fork."""
import os,sys,json,argparse,time,copy,ast,hashlib,subprocess
from pathlib import Path
a=argparse.ArgumentParser();a.add_argument("--config",required=True);a.add_argument("--phase",choices=["preflight","fit","evaluate"],required=True);args=a.parse_args();cfg=json.loads(Path(args.config).read_text())
os.environ.update(CUDA_VISIBLE_DEVICES=str(cfg["gpu_index"]),PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",TORCH_DEVICE_BACKEND_AUTOLOAD="0")
import numpy as np,torch
from torch import nn
from types import SimpleNamespace
torch.set_num_threads(2)
root=Path(cfg["reference_root"]);sys.path.insert(0,str(root))
from models import DMLP,iTransformer,Regressor
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+"\n")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(p,y):
 e=p-y
 return dict(mse=float(np.mean(e*e)),mae=float(np.mean(abs(e))),level=float(np.mean(e.mean(1)**2)),shape=float(np.mean((e-e.mean(1,keepdims=True))**2)))
tree=ast.parse((root/"exp/exp_long_term_forecasting.py").read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="multi_scale_process_inputs")
ns={"torch":torch,"nn":nn};exec(compile(ast.Module(body=[node],type_ignores=[]),str(root/"exp/exp_long_term_forecasting.py"),"exec"),ns);scales=ns["multi_scale_process_inputs"]()
def native_args(width,teacher=False):
 p=dict(task_name="long_term_forecast",seq_len=96,pred_len=96,enc_in=1,c_out=1,d_model=width,d_ff=cfg["teacher"]["d_ff"] if teacher else width,e_layers=cfg["teacher"]["e_layers"],n_heads=cfg["teacher"]["n_heads"],dropout=cfg["teacher"]["dropout"],factor=3,activation="gelu",embed="timeF",freq="h",moving_avg=cfg["moving_avg"],norm=cfg["norm"],output_attention=False,use_norm=True,num_class=1)
 return SimpleNamespace(**p)
def check_timekd():
 base=root.parent/"TimeKD";ns={"__name__":"timekd_applicability"}
 exec(compile((base/"layers/TS_Pos_Enc.py").read_text(),str(base/"layers/TS_Pos_Enc.py"),"exec"),ns)
 code=(base/"layers/Sub_CA.py").read_text().replace("from layers.TS_Pos_Enc import *","")
 exec(compile(code,str(base/"layers/Sub_CA.py"),"exec"),ns);SCA=ns["SCA"];rows=[]
 for channels in [1,3]:
  torch.manual_seed(2024);m=SCA(d_model=channels,n_heads=1,d_ff=3072,norm="LayerNorm",attn_dropout=.1,dropout=.1,pre_norm=True,res_attention=True,n_layers=1).eval()
  for p in m.parameters():p.requires_grad_(False)
  q=torch.randn(4,16,channels,requires_grad=True);k=torch.randn_like(q);v=torch.randn_like(q);out=m(q,k,v);other=m(torch.randn_like(q),torch.randn_like(q),torch.randn_like(q));grad=torch.autograd.grad(out.sum(),q)[0]
  rows.append(dict(channels=channels,input_sensitivity=float((out-other).abs().max()),input_jacobian_norm=float(grad.norm()),tokens=16))
 assert rows[0]["input_sensitivity"]==0 and rows[0]["input_jacobian_norm"]==0
 assert rows[1]["input_sensitivity"]>0 and rows[1]["input_jacobian_norm"]>0
 return dict(official_commit="42ee42da68f5b3a33f1a7fe759269c7731ae1392",rows=rows,native_single_variable_comparison_valid=False,reason="Official SCA applies LayerNorm over num_nodes. At C=1 it removes all input dependence; native attention correlations also collapse. Requires a genuine multivariate task, not relabelled outputKD.",LLM_loaded=False,new_weights=False)
def loss_full(outputs,features,ot,ft,regressor,y):
 # Exact official four KD terms, temperature0.5/scales0..3.
 rt=regressor(ft)
 po=torch.softmax(abs(torch.fft.rfft(outputs,dim=1))[:,1:,:]/.5,dim=1)
 pt=torch.softmax(abs(torch.fft.rfft(ot,dim=1))[:,1:,:]/.5,dim=1)
 out_freq=cfg["alpha"]*torch.nn.functional.kl_div(torch.log(po+1e-8),pt,reduction="mean")
 oo=scales(outputs.permute(0,2,1));tt=scales(ot.permute(0,2,1));out_scale=sum(cfg["alpha"]*(aa-bb).square().mean() for aa,bb in zip(oo,tt))/len(oo)
 fs=features[-2];lp=torch.log_softmax(abs(torch.fft.rfft(fs.permute(0,2,1),dim=1))[:,1:,:]/.5,dim=1);lt=torch.log_softmax(abs(torch.fft.rfft(rt.permute(0,2,1),dim=1))[:,1:,:]/.5,dim=1)
 feat_freq=cfg["beta"]*torch.nn.functional.kl_div(lp,lt,reduction="mean",log_target=True)
 ss=scales(fs);ts=scales(rt);feat_scale=sum(cfg["beta"]*(aa-bb).square().mean() for aa,bb in zip(ss,ts))/len(ss)
 gt=(outputs-y).square().mean()
 return gt+out_freq+out_scale+feat_freq+feat_scale,torch.stack([gt,out_freq,out_scale,feat_freq,feat_scale])
def forward(m,x,mark):return m(x,mark,None,None)
def choose(curves,base,variants):
 options=[]
 for variant in variants:
  for step in range(0,cfg["steps"]+1,cfg["check_every"]):
   risks=[]
   for seed in cfg["seeds"]:
    v=next(q for q in curves if q["seed"]==seed and q["variant"]==variant and q["step"]==step);risks.append([.5*(a["mse"]/b["mse"]+a["mae"]/b["mae"]) for a,b in zip(v["blocks"],base[seed])])
   a=np.array(risks);score=float((a.mean()+a.mean(0).max()+a.mean(1).max())/3);options.append(dict(variant=variant,step=step,score=score))
 best=min(options,key=lambda q:(q["score"],q["variant"],q["step"]));return best,options
def fit():
 start=time.monotonic();r=Path(cfg["run"]);r.mkdir(parents=True,exist_ok=False);save(r/"config.json",cfg);save(r/"timekd_applicability.json",check_timekd());refs=read(root.parent/"sources.json")
 for rec in refs:assert sha(rec["path"])==rec["sha256"]
 save(r/"reference_assets.json",refs);save(r/"provenance.json",dict(git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),source_sha=sha(__file__),config_sha=sha(args.config),torch=torch.__version__))
 main=Path(cfg["main_run"]);assert (main/"model_lock.json").exists();assert not (main/"confirmation_data.npz").exists()
 d=np.load(main/"baseline_data.npz");dev="cuda:0";assert torch.cuda.mem_get_info()[0]>cfg["memory_gib"]*2**30
 torch.cuda.set_per_process_memory_fraction(cfg["memory_gib"]*2**30/torch.cuda.get_device_properties(0).total_memory)
 x=torch.tensor(d["train_x"],dtype=torch.float32,device=dev);y=torch.tensor(d["train_y"],dtype=torch.float32,device=dev);mark=torch.tensor(d["train_mark"],dtype=torch.float32,device=dev);vx=torch.tensor(d["val_x"],dtype=torch.float32,device=dev);vm=torch.tensor(d["val_mark"],dtype=torch.float32,device=dev);vy=d["val_y"]
 blocks=np.array_split(np.arange(len(vy)),3);base={s:[metric(d[f"val_bp_{s}"][ii],vy[ii]) for ii in blocks] for s in cfg["seeds"]}
 teacher_states={};tc=[];params=[]
 for seed in cfg["seeds"]:
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);m=iTransformer.Model(native_args(cfg["teacher"]["d_model"],True)).to(dev);opt=torch.optim.Adam(m.parameters(),lr=cfg["learning_rate"]);rng=np.random.default_rng(seed+271);states={}
  for step in range(cfg["steps"]+1):
   if step%cfg["check_every"]==0:
    m.eval()
    with torch.no_grad():p,_=forward(m,vx,vm);pr=p.cpu().numpy()
    tc.append(dict(seed=seed,variant="iTransformer",step=step,blocks=[metric(pr[ii],vy[ii]) for ii in blocks]));states[step]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
   if step==cfg["steps"]:break
   m.train();ids=rng.integers(0,len(x),cfg["batch"]);pred,_=forward(m,x[ids],mark[ids]);loss=(pred-y[ids]).square().mean();opt.zero_grad();loss.backward();opt.step();assert time.monotonic()-start<cfg["max_seconds"]
  teacher_states[seed]=states;params.append(dict(seed=seed,model="Teacher_iTransformer",parameters=sum(p.numel() for p in m.parameters())));print("TEACHER_DONE",seed,flush=True)
 teacherpick,toptions=choose(tc,base,["iTransformer"]);save(r/"teacher_selection.json",dict(chosen=teacherpick,options=toptions));save(r/"teacher_curves.json",tc);teacher_cache={}
 for seed in cfg["seeds"]:
  m=iTransformer.Model(native_args(cfg["teacher"]["d_model"],True)).to(dev);m.load_state_dict(teacher_states[seed][teacherpick["step"]]);m.eval()
  for p in m.parameters():p.requires_grad_(False)
  with torch.no_grad():ot,ft=forward(m,x,mark);teacher_cache[seed]=(ot.detach(),ft[-2][:,:1,:].detach())
 del m
 states_s={};curves=[];loss_rows=[]
 for width in cfg["student_widths"]:
  for seed in cfg["seeds"]:
   torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);m=DMLP.Model(native_args(width)).to(dev);reg=Regressor.Model(native_args(width),cfg["teacher"]["d_model"],width).to(dev);opt=torch.optim.Adam(list(m.parameters())+list(reg.parameters()),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"]);rng=np.random.default_rng(seed);states={};ot,ft=teacher_cache[seed]
   for step in range(cfg["steps"]+1):
    if step%cfg["check_every"]==0:
     m.eval()
     with torch.no_grad():p,_=forward(m,vx,vm);pr=p.cpu().numpy()
     curves.append(dict(seed=seed,variant=width,step=step,blocks=[metric(pr[ii],vy[ii]) for ii in blocks]));states[step]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    if step==cfg["steps"]:break
    m.train();reg.train();ids=rng.integers(0,len(x),cfg["batch"]);pred,features=forward(m,x[ids],mark[ids]);loss,terms=loss_full(pred,features,ot[ids],ft[ids],reg,y[ids]);assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
    if step%100==0:loss_rows.append(dict(width=width,seed=seed,step=step,gt=float(terms[0]),output_frequency=float(terms[1]),output_multiscale=float(terms[2]),feature_frequency=float(terms[3]),feature_multiscale=float(terms[4])))
    assert time.monotonic()-start<cfg["max_seconds"]
   states_s[(width,seed)]=states;params.append(dict(seed=seed,model="DMLP",width=width,parameters=sum(p.numel() for p in m.parameters()),training_regressor_parameters=sum(p.numel() for p in reg.parameters())));print("STUDENT_DONE",width,seed,flush=True)
 pick,options=choose(curves,base,cfg["student_widths"]);save(r/"selection.json",dict(chosen=pick,options=options,shared3seeds=True,before2018=True));save(r/"student_curves.json",curves);save(r/"loss_terms.json",loss_rows);save(r/"parameters.json",params)
 torch.save(dict(width=pick["variant"],step=pick["step"],students={s:states_s[(pick["variant"],s)][pick["step"]] for s in cfg["seeds"]}),r/"students.pt");torch.save(dict(teachers=teacher_states,students=states_s),r/"all_states.pt")
 save(r/"model_lock.json",dict(model_sha=sha(r/"students.pt"),selection_sha=sha(r/"selection.json"),official_commit=cfg["official_commit"],before2018=True,native_architecture=True,full_four_term_KD=True,native_driver_budget_and_checkpoint_adaptation=True))
 save(r/"status.json",dict(complete=True,seconds=time.monotonic()-start,teacher_fits=3,student_fits=6,updates=9*cfg["steps"],new_control22_fits=0,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,LLM_loaded=False,TimeKD_native_run=False,TimeKD_single_variable_invalid=True))
 print("TIMEDISTILL_LOCKED",pick,flush=True)
def evaluate():
 r=Path(cfg["run"]);main=Path(cfg["main_run"]);lock=read(r/"model_lock.json");assert sha(r/"students.pt")==lock["model_sha"] and sha(r/"selection.json")==lock["selection_sha"]
 d=np.load(main/"confirmation_data.npz");pack=torch.load(r/"students.pt",map_location="cpu",weights_only=True);rows=[];preds={};replays=[];dev="cuda:0"
 for s in cfg["seeds"]:
  m=DMLP.Model(native_args(pack["width"])).to(dev);m.load_state_dict(pack["students"][s]);m.eval();x=torch.tensor(d["x"],dtype=torch.float32,device=dev);mark=torch.tensor(d["mark"],dtype=torch.float32,device=dev);outputs=[]
  with torch.no_grad():
   for i in range(0,len(x),32):q,_=forward(m,x[i:i+32],mark[i:i+32]);outputs.append(q.cpu().numpy())
  pred=np.concatenate(outputs);rows.append(dict(seed=s,model="TimeDistill_native",**metric(pred,d["y"])));preds[f"{s}_TimeDistill_native"]=pred
  m2=DMLP.Model(native_args(pack["width"])).to(dev);m2.load_state_dict(pack["students"][s]);m2.eval()
  with torch.no_grad():q,_=forward(m2,x[:8],mark[:8])
  err=float(abs(q.cpu().numpy()-pred[:8]).max());assert err<3e-6;replays.append(dict(seed=s,error=err))
 np.savez_compressed(r/"predictions.npz",**preds,y=d["y"],origin=d["origin"]);save(r/"metrics.json",rows);save(r/"summary.json",{k:float(np.mean([v[k] for v in rows])) for k in ["mse","mae","level","shape"]});save(r/"replay.json",replays)
 assert sha(r/"students.pt")==lock["model_sha"] and sha(r/"selection.json")==lock["selection_sha"];print("TIMEDISTILL_CONFIRM",rows,flush=True)
def loss_fidelity():
 torch.manual_seed(2024);y=torch.randn(4,96,1,device="cuda");outputs=torch.randn_like(y);ot=torch.randn_like(y);features=[torch.randn(4,1,32,device="cuda"),outputs];ft=torch.randn(4,1,256,device="cuda");reg=Regressor.Model(native_args(32),256,32).cuda()
 mine,terms=loss_full(outputs,features,ot,ft,reg,y)
 code=(root/"exp/exp_long_term_forecasting.py").read_text()
 i=code.index("                        # logit level")
 j=code.index("                    else:",code.index("                        loss = loss_gt + loss_logit + loss_feature",i))
 block=code[i:j];import textwrap
 scope=dict(self=SimpleNamespace(args=SimpleNamespace(alpha=cfg["alpha"],beta=cfg["beta"]),regressor=reg),torch=torch,F=torch.nn.functional,criterion=nn.MSELoss(),outputs=outputs,outputs_t=[ot],features=features,features_t=[ft],batch_y=y,multi_scale_process_inputs=ns["multi_scale_process_inputs"])
 exec(compile(textwrap.dedent(block),"official_TimeDistill_loss_block","exec"),scope)
 error=float(abs(mine-scope["loss"]));assert error<1e-6,error
 return error

if args.phase=="preflight":
 print("REFERENCE_IMPORTS_OK",check_timekd());print("OFFICIAL_LOSS_FIDELITY",loss_fidelity())
 m=DMLP.Model(native_args(32)).cuda();t=iTransformer.Model(native_args(cfg["teacher"]["d_model"],True)).cuda();rr=Regressor.Model(native_args(32),cfg["teacher"]["d_model"],32).cuda()
 x=torch.randn(4,96,1,device="cuda");mark=torch.randn(4,96,7,device="cuda");a,f=forward(m,x,mark);b,ff=forward(t,x,mark);loss,terms=loss_full(a,f,b.detach(),ff[-2][:,:1,:].detach(),rr,torch.randn_like(a));loss.backward();assert torch.isfinite(terms).all();print("COMPLETE_FOUR_LOSS_FORWARD_BACKWARD",terms.detach().cpu().tolist())
elif args.phase=="fit":fit()
else:evaluate()
