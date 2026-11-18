"""Frozen teacher first-token calibration; no student or LLM training."""
import os
os.environ.update(OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from exp import exp_chronogeom_ordinal_opd01 as old
def main():
 tick=time.monotonic();c=old.read("docs/chronogeom/teacher_calibration01_config.json");r=Path(c["run"]);r.mkdir(parents=True,exist_ok=False);old.save(r/"config.json",c)
 data=np.load(c["data"]);groups={}
 for root in c["cache_runs"]:
  for line in (Path(root)/"qwen_feedback.jsonl").read_text().splitlines():
   z=json.loads(line)
   if z["phase"]!="inner" or z["family"]!="QwenOPD":continue
   for ix,p in zip(z["rows"],z["probabilities"]):groups.setdefault(int(ix),[]).append(p[0])
 indices=np.array(sorted(groups));p=np.array([np.mean(groups[i],axis=0) for i in indices]);p/=p.sum(1,keepdims=True);lp=np.log(np.maximum(p,1e-9))
 labels=old.true_states(data["y"][indices],dict(band=.25))[:,0]
 origin=data["origin"][indices];days=np.unique(origin//24);cut=days[len(days)*7//10]*24;fit=np.where(origin+96<=cut)[0];test=np.where(origin>=cut)[0]
 assert len(fit)>0 and len(test)>0 and max(origin[fit]+96)<=min(origin[test])
 old.save(r/"split.json",dict(cut=int(cut),fit_indices=indices[fit].tolist(),check_indices=indices[test].tolist(),excluded_indices=indices[(origin<cut)&(origin+96>cut)].tolist(),fit_origin_days=len(np.unique(origin[fit]//24)),check_origin_days=len(np.unique(origin[test]//24)),targets_disjoint=True))
 oh=np.eye(3)[labels];L=lp[fit];Y=oh[fit]
 def objective(theta):
  alpha=theta[0];bias=np.array([theta[1],0,theta[2]]);z=alpha*L+bias
  ce=(logsumexp(z,axis=1)-(z*Y).sum(1)).mean()
  prob=softmax(z,axis=1);diff=(prob-Y)/len(fit)
  reg=.001*((alpha-1)**2+(bias**2).sum())
  grad=np.array([(diff*L).sum()+.002*(alpha-1),diff[:,0].sum()+.002*bias[0],diff[:,2].sum()+.002*bias[2]])
  return ce+reg,grad
 result=minimize(objective,np.array([1.,0.,0.]),jac=True,method="L-BFGS-B",bounds=[(0,20),(-10,10),(-10,10)],options=dict(maxiter=300,ftol=1e-12,gtol=1e-8))
 assert result.success,result.message
 alpha,low,high=result.x;cal=softmax(alpha*lp+np.array([low,0,high]),axis=1)
 prior=(np.bincount(labels[fit],minlength=3)+1)/(len(fit)+3);prior=np.repeat(prior[None],len(indices),axis=0)
 x=np.concatenate([old.old.fourier(data["ids"][indices,3:4])[:,0],data["statistics"][indices]],axis=1);mu=x[fit].mean(0);sd=np.maximum(x[fit].std(0),.001);x=(x-mu)/sd;x=np.c_[x,np.ones(len(x))]
 def linear_obj(w):
  w=w.reshape(39,3);z=x[fit]@w
  diff=(softmax(z,axis=1)-Y)/len(fit);penalty=w.copy();penalty[-1]=0
  loss=np.mean(logsumexp(z,axis=1)-(z*Y).sum(1))+.1*(penalty**2).sum()
  return loss,(x[fit].T@diff+.2*penalty).ravel()
 ff=minimize(linear_obj,np.zeros(117),jac=True,method="L-BFGS-B",options=dict(maxiter=300,ftol=1e-12))
 assert ff.success,ff.message
 fourier=softmax(x@ff.x.reshape(39,3),axis=1)
 old.save(r/"calibration.json",dict(alpha=float(alpha),temperature=float(1/alpha) if alpha>0 else None,bias=[float(low),0.,float(high)],optimizer_success=bool(result.success),fit_regularizer=.001,parameters=3,coefficient_space_convex=True,old_llm_frozen=True))
 np.savez_compressed(r/"predictions.npz",indices=indices,origin=origin,labels=labels,raw=p,calibrated=cal,prior=prior,fourier=fourier,fit=fit,check=test)
 rows=[]
 for part,ii in [("fit",fit),("check",test)]:
  y=labels[ii]
  for name,prob in [("RawQwen",p),("CalibratedQwen",cal),("ClassPrior",prior),("FourierHistory117",fourier)]:
   q=prob[ii];pred=q.argmax(1);rec=[np.mean(pred[y==k]==k) for k in range(3) if np.any(y==k)]
   row=dict(part=part,arm=name,windows=len(ii),accuracy=float(np.mean(pred==y)),balanced_accuracy=float(np.mean(rec)),nll=float(-np.log(np.maximum(q[np.arange(len(ii)),y],1e-9)).mean()),brier=float(np.mean(np.sum((q-oh[ii])**2,axis=1))),mid_probability=float(q[:,1].mean()),true_mid_frequency=float(np.mean(y==1)))
   rows.append(row)
 old.csvsave(r/"scores.csv",rows)
 old.save(r/"completion.json",dict(seconds=time.monotonic()-tick,git=old.rt.git(),unique_windows=len(indices),first_token_only=True,llm_calls=0,student_updates=0,source_hashes={str(Path(k)/"qwen_feedback.jsonl"):old.rt.sha(Path(k)/"qwen_feedback.jsonl") for k in c["cache_runs"]}))
 print(json.dumps(dict(parameters=old.read(r/"calibration.json"),split=old.read(r/"split.json"),scores=rows),indent=2))
if __name__=="__main__":main()
