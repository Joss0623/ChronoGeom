"""Freeze only development-validation-derived rules; predict heldout before evaluation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,json,sys,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from data_provider.chronogeom_synthetic import FAMILIES,task_spec,generate
from scripts.chronogeom.phase1_representations import T_ENCODINGS,encode,rank
FEATURES=['alignment_nmse','effective_rank','native_dimension']
def diagnostic(spec,encoding):
 t,y,_=generate(spec,'train');e=encode(t,encoding,track='T',period=spec['period'])
 if encoding=='shuffled_fourier':
  e=e.copy()
  for i in range(len(e)):
   rng=np.random.default_rng(72000+i);e[i,:16]=e[i,rng.permutation(16)];e[i,16:]=e[i,16+rng.permutation(8)]
 x=np.column_stack([e.reshape(-1,e.shape[-1]),np.ones(e.shape[0]*e.shape[1])]);target=y.ravel();coef=np.linalg.lstsq(x,target,rcond=1e-8)[0]
 return [float(np.mean((x@coef-target)**2)/np.var(target)),float(rank(e[:,:16])),float(e.shape[-1])]
def fit_ridge(x,y,alpha):
 mean=x.mean(0);std=x.std(0);std=np.where(std<1e-8,1,std);z=np.column_stack([(x-mean)/std,np.ones(len(x))]);reg=np.eye(z.shape[1])*alpha;reg[-1,-1]=0;coef=np.linalg.solve(z.T@z+reg,z.T@y);return dict(mean=mean.tolist(),std=std.tolist(),coef=coef.tolist(),alpha=alpha)
def predict(model,x):return np.column_stack([(x-model['mean'])/model['std'],np.ones(len(x))])@np.array(model['coef'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--development',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output;out.mkdir(exist_ok=False);dev=pd.read_csv(args.development/'metrics.csv');d=dev.groupby(['family','backbone','encoding'],as_index=False)[['mse']+FEATURES].mean();rules={};cv=[];pred=[]
 for b in ['DLinear','TSMixer']:
  q=d[d.backbone==b].copy();q['log_mse']=np.log(q.mse);x=q[FEATURES].to_numpy();y=q.log_mse.to_numpy()
  # Ensure the independent train-only diagnostic matches inference provenance.
  for row in q.itertuples():np.testing.assert_allclose(diagnostic(task_spec('development',row.family),row.encoding),[getattr(row,f) for f in FEATURES],atol=1e-10)
  for name,alpha in [('constant',None),('ridge_0.1',.1),('ridge_1',1.),('ridge_10',10.)]:
   errors=[]
   for family in FAMILIES:
    tr=q.family!=family;te=~tr
    if alpha is None:
     lookup=q[tr].groupby('encoding').log_mse.mean().to_dict();p=np.array([lookup[e] for e in q[te].encoding])
    else:p=predict(fit_ridge(x[tr],y[tr],alpha),x[te])
    errors.extend((p-y[te])**2)
   cv.append(dict(backbone=b,candidate=name,lofo_log_mse=float(np.mean(errors))))
  chosen=min([v for v in cv if v['backbone']==b],key=lambda v:v['lofo_log_mse'])['candidate'];rule=dict(type=chosen,features=FEATURES)
  if chosen=='constant':rule['per_encoding_log_mse']=q.groupby('encoding').log_mse.mean().to_dict()
  else:rule.update(fit_ridge(x,y,float(chosen.split('_')[1])))
  rules[b]=rule
  for f in FAMILIES:
   raw=np.array([diagnostic(task_spec('heldout',f),e) for e in T_ENCODINGS]);score=np.array([rule['per_encoding_log_mse'][e] for e in T_ENCODINGS]) if chosen=='constant' else predict(rule,raw);winner=T_ENCODINGS[int(np.argmin(score))]
   for e,z,sc in zip(T_ENCODINGS,raw,score):pred.append(dict(family=f,backbone=b,encoding=e,alignment_nmse=z[0],effective_rank=z[1],native_dimension=z[2],predicted_log_mse=float(sc),chosen=e==winner))
 pd.DataFrame(cv).to_csv(out/'validation_rule_cv.csv',index=False);pd.DataFrame(pred).to_csv(out/'heldout_predictions_before_eval.csv',index=False)
 lock=dict(rules=rules,validation_metrics_sha256=hashlib.sha256((args.development/'metrics.csv').read_bytes()).hexdigest(),predictions_sha256=hashlib.sha256((out/'heldout_predictions_before_eval.csv').read_bytes()).hexdigest(),test_outcomes_used=False,features_scope='heldout task training data only; no heldout validation/test targets',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(out/'locked_rule.json').write_text(json.dumps(lock,indent=2)+'\n');print(pd.DataFrame(cv).to_string(index=False));print('LOCKED', {b:r['type'] for b,r in rules.items()})
if __name__=='__main__':main()
