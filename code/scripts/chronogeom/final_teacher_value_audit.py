"""Cache-only final teacher audit: matrix decompositions, no torch/models/data CSVs."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='2'
import argparse,csv,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def csvsave(p,rows):
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def centered(x):return x-x.mean(0,keepdims=True)
def normalized(x):
 x=centered(np.asarray(x,dtype=float));return x/max(np.sqrt(np.mean(np.sum(x*x,axis=1))),1e-15)
def basis(x,tol):
 u,s,_=np.linalg.svd(centered(np.asarray(x,dtype=float)),full_matrices=False);rank=int(np.sum(s>s[0]*tol)) if len(s) and s[0]>1e-12 else 0
 return u[:,:rank],s[:rank]
def relative(a,b):return float(np.sum((a-b)**2)/max(np.sum(a*a),1e-30))
def cosine(a,b):return float(np.sum(a*b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
def kernel(x):
 x=centered(x);g=x@x.T;return g/max(np.sqrt(np.mean(g*g)),1e-15)
def overlap(a,b,tol):
 u,_=basis(a,tol);v,_=basis(b,tol);s=np.linalg.svd(u.T@v,compute_uv=False);angles=np.rad2deg(np.arccos(np.clip(s,0,1)))
 return dict(rank_a=u.shape[1],rank_b=v.shape[1],squared_overlap=float(np.sum(s*s)),normalized_projection_distance=float(np.sqrt(max(u.shape[1]+v.shape[1]-2*np.sum(s*s),0)/max(u.shape[1]+v.shape[1],1))),angles_degrees=angles.tolist())
def make_tables(llm):
 d=np.repeat(np.arange(7),24);h=np.tile(np.arange(24),7);t=np.arange(168)
 circ=lambda x,p:np.c_[np.cos(2*np.pi*x/p),np.sin(2*np.pi*x/p)]
 day=circ(h,24);week=circ(d,7);f4=np.c_[day,week]
 f22=np.column_stack([circ(h*k,24) for k in range(1,7)]+[circ(d*k,7) for k in range(1,4)]+[day[:,i:i+1]*week[:,j:j+1] for i in range(2) for j in range(2)])
 complete=np.column_stack([np.ones(168)]+[circ(t*k,168) for k in range(1,84)]+[np.cos(np.pi*t)])
 return {k:normalized(v) for k,v in dict(LLM4=llm,Fourier4=f4,Fourier22=f22,AdditiveCalendar32=np.c_[np.eye(24)[h],np.eye(7)[d],d>=5],JointCalendar168=np.eye(168),CompleteWeeklyFourier168=complete,WorkdayWeekend=np.c_[d<5,d>=5],Fourier4Weekend=np.c_[f4,d>=5],WeekdayCircle2=week,ContinuousWeekFourier4=np.c_[day,circ(t,168)]).items()}
def table_audit(tables,out,cfg):
 target=tables['LLM4'];tol=cfg['svd_relative_tolerance'];projections=[];responses=[];states=np.arange(168);summary=[]
 for name,x in tables.items():
  u,s=basis(x,tol);p=u@u.T;fit=p@target
  summary.append(dict(teacher=name,dimension=x.shape[1],centered_rank=len(s),energy_rms=float(np.sqrt(np.mean(np.sum(x*x,axis=1)))),min_nonzero_singular_value=float(s[-1]),unique_states=len(np.unique(np.round(x,6),axis=0))))
  projections.append(dict(teacher=name,llm_explained_energy=1-relative(target,fit),llm_residual_energy=relative(target,fit),llm_kernel_explained_energy=1-relative(kernel(target),p@kernel(target)@p),**overlap(target,x,tol)))
  g=kernel(x)
  for delta in cfg['shifts_hours']:
   ix=(states+delta)%168;response=target[ix]-target;fitresponse=fit[ix]-fit;op=np.linalg.lstsq(x,x[ix],rcond=tol)[0]
   responses.append(dict(teacher=name,shift_hours=delta,llm_response_unexplained_energy=relative(response,fitresponse) if abs(delta)!=168 else 0.,self_best_linear_shift_residual=relative(x[ix],x@op),kernel_joint_shift_relative_distance=float(np.linalg.norm(g[ix][:,ix]-g)/np.linalg.norm(g)),state_step_energy=float(np.mean(np.sum((x[ix]-x)**2,axis=1)))))
 csvsave(out/'representation_capacity.csv',summary);dump(out/'llm_projection.json',projections);csvsave(out/'transformation_response.csv',responses)
 pairs=[]
 for i,(an,a) in enumerate(tables.items()):
  for bn,b in list(tables.items())[i+1:]:pairs.append(dict(a=an,b=bn,**overlap(a,b,tol)))
 dump(out/'pairwise_subspaces.json',pairs)
 fft=np.fft.fft2(target.reshape(7,24,4),axes=(0,1),norm='ortho');e=(abs(fft)**2).sum(-1);total=e.sum()
 decomp=dict(dc=float(e[0,0]/total),hour_only=float(e[0,1:].sum()/total),weekday_only=float(e[1:,0].sum()/total),interaction=float(e[1:,1:].sum()/total))
 dump(out/'llm_2d_fourier_energy.json',decomp)
 # Mathematical completeness checks, not evidence of statistical predictive value.
 assert relative(target,basis(tables['JointCalendar168'],tol)[0]@basis(tables['JointCalendar168'],tol)[0].T@target)<1e-20
 assert relative(target,basis(tables['CompleteWeeklyFourier168'],tol)[0]@basis(tables['CompleteWeeklyFourier168'],tol)[0].T@target)<1e-20
 return projections

def template_audit(run,out,cfg):
 raw=np.load(run/'llm/raw_spans.npz')['hidden'];prompts=json.loads((run/'llm/prompts.json').read_text());lookup=np.load(run/'llm/lookup.npz')['llm4'];records=[];full=[];residual_stability=[];analytic=make_tables(lookup)
 for li,layer in enumerate(cfg['layers']):
  a=np.load(run/f'llm/geometry_layer{layer}.npz');z=(raw[:,li].astype(float)-a['center'])@a['projection'].T/float(a['scale'])
  ft=[]
  for template in [0,1]:
   ix=[i for i,r in enumerate(prompts) if r['purpose']=='lookup' and r['template']==template]
   assert [(prompts[i]['day'],prompts[i]['hour']) for i in ix]==[(d,h) for d in range(7) for h in range(24)]
   ft.append(z[ix])
  ft=np.stack(ft);mu=ft.mean(0).mean(0);scale=np.sqrt(np.mean(np.sum((ft.mean(0)-mu)**2,axis=1)));ft=(ft-mu)/scale
  if layer==cfg['selected_layer']:assert np.max(abs(ft.mean(0)-lookup))<1e-6
  full.append(dict(layer=layer,scope='two in-fit templates, full168 states',coordinate_nrmse=float(np.sqrt(relative(ft[0],ft[1]))),centered_coordinate_nrmse=float(np.sqrt(relative(centered(ft[0]),centered(ft[1])))),kernel_relative_distance=float(np.linalg.norm(kernel(ft[0])-kernel(ft[1]))/np.linalg.norm(kernel(ft[0]))),kernel_cosine=cosine(kernel(ft[0]),kernel(ft[1])),**overlap(ft[0],ft[1],cfg['svd_relative_tolerance'])))
  for name in ['Fourier4','Fourier22','AdditiveCalendar32']:
   q,_=basis(analytic[name],cfg['svd_relative_tolerance']);p=q@q.T
   rr=[centered(f)-p@centered(f) for f in ft]
   residual_stability.append(dict(layer=layer,basis=name,scope='full168 only two in-fit templates',template0_residual_energy=relative(centered(ft[0]),p@centered(ft[0])),template1_residual_energy=relative(centered(ft[1]),p@centered(ft[1])),residual_cosine=cosine(rr[0],rr[1]),residual_nrmse=float(np.sqrt(relative(rr[0],rr[1])))))
  for delta in [1,24]:
   ix=(np.arange(168)+delta)%168;a0=ft[0][ix]-ft[0];a1=ft[1][ix]-ft[1]
   full.append(dict(layer=layer,scope=f'in-fit full168 shift{delta}h response',coordinate_nrmse=float(np.sqrt(relative(a0,a1))),centered_coordinate_nrmse=float(np.sqrt(relative(centered(a0),centered(a1)))),kernel_relative_distance=float(np.linalg.norm(kernel(a0)-kernel(a1))/np.linalg.norm(kernel(a0))),kernel_cosine=cosine(kernel(a0),kernel(a1)),**overlap(a0,a1,cfg['svd_relative_tolerance'])))
  grids=[]
  for template in range(8):
   ix=[i for i,r in enumerate(prompts) if r['purpose']=='geometry' and r['template']==template]
   assert [(prompts[i]['day'],prompts[i]['hour']) for i in ix]==[(d,h) for d in range(7) for h in [0,6,12,18]]
   grids.append((z[ix]-mu)/scale)
  ref=np.mean(grids[:4],axis=0)
  for template,g in enumerate(grids):
   rec=dict(layer=layer,template=template,stage='fit' if template<4 else 'validation' if template<6 else 'audit',n_states=28,coordinate_nrmse=float(np.sqrt(relative(ref,g))),centered_coordinate_nrmse=float(np.sqrt(relative(centered(ref),centered(g)))),kernel_relative_distance=float(np.linalg.norm(kernel(ref)-kernel(g))/np.linalg.norm(kernel(ref))),kernel_cosine=cosine(kernel(ref),kernel(g)))
   for delta in cfg['template_shifts_hours']:
    ix=(np.arange(28)+delta//6)%28;dr=ref[ix]-ref;dg=g[ix]-g
    rec[f'response_{delta}h_cosine']=cosine(dr,dg);rec[f'response_{delta}h_nrmse']=float(np.sqrt(relative(dr,dg)))
   records.append(rec)
 csvsave(out/'template_grid_stability.csv',records);dump(out/'fullweek_template_stability.json',full);csvsave(out/'residual_cross_template.csv',residual_stability)

def saved_forecast_audit(path,predfile,scorefile,out,prefix,labelkey):
 a=np.load(path/predfile);truth=a['truth'].astype(float);previous=list(csv.DictReader((path/scorefile).open()));idx={r[labelkey]:r for r in previous};rows=[]
 for k in a.files:
  if k in ['truth','origin','channel']:continue
  mse=float(np.mean((a[k].astype(float)-truth)**2));old=float(idx[k]['mse']);assert abs(mse-old)<1e-5
  rows.append(dict(method=k,recomputed_mse_float64=mse,original_mse=old,absolute_error=abs(mse-old),n_origins=len(np.unique(a['origin']))))
 csvsave(out/(prefix+'_saved_scores.csv'),rows)
 return rows

def weekday_audit(run,out):
 rows=list(csv.DictReader((run/'interventions.csv').open()));summary=[]
 for ctx in [0,1]:
  for layer in [6,14,20]:
   a=[r for r in rows if int(r['context'])==ctx and int(r['layer'])==layer];index={(int(r['pair_id']),r['method']):r for r in a};learn=[r for r in a if r['method']=='learned_transition'];analytic=[index[(int(r['pair_id']),'analytic_rotation')] for r in learn]
   summary.append(dict(context=ctx,layer=layer,n_pairs=len(learn),learned_target_hits=sum(r['global_target_correct']=='True' for r in learn),analytic_target_hits=sum(r['global_target_correct']=='True' for r in analytic),mean_ld_increment=float(np.mean([float(x['logit_difference_change'])-float(y['logit_difference_change']) for x,y in zip(learn,analytic)])),argmax_disagreements=sum(x['global_token']!=y['global_token'] for x,y in zip(learn,analytic))))
 csvsave(out/'weekday_learned_analytic.csv',summary)

def main(args):
 start=time.monotonic();cfg=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False);r=Path(cfg['distill_run']);t=Path(cfg['relation_run']);w=Path(cfg['weekday_run'])
 inputs=[r/'llm/lookup.npz',r/'llm/raw_spans.npz',r/'llm/prompts.json',t/'teacher_tables.npz',r/'pilot_predictions.npz',t/'test_predictions.npz',w/'interventions.csv',r/'timedistill_optimization_diagnostics.json',r/'all_candidate_fits.csv',t/'all_candidate_fits.csv']+[r/f'llm/geometry_layer{l}.npz' for l in cfg['layers']]
 before={str(p):sha(p) for p in inputs};dump(out/'config.json',cfg);dump(out/'source.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha(__file__),input_sha256=before))
 tables=make_tables(np.load(r/'llm/lookup.npz')['llm4']);np.savez_compressed(out/'analytical_tables.npz',**tables);proj=table_audit(tables,out,cfg)
 # Audit actual frozen TimeBase kernels separately from the global-RMS tables.
 frozen=np.load(t/'teacher_tables.npz');actual=[]
 for name in frozen.files:
  x=frozen[name].astype(float);u,s=basis(x,cfg['svd_relative_tolerance']);g=x@x.T;actual.append(dict(teacher=name,rank=int(np.linalg.matrix_rank(x,tol=1e-6)),kernel_rms=float(np.sqrt(np.mean(g*g))),diagonal_cv=float(np.diag(g).std()/np.diag(g).mean()),llm_centered_subspace=overlap(frozen['LLM'],x,cfg['svd_relative_tolerance']),shift_1h_kernel_distance=float(np.linalg.norm(g[(np.arange(168)+1)%168][:,(np.arange(168)+1)%168]-g)/np.linalg.norm(g)),shift_24h_kernel_distance=float(np.linalg.norm(g[(np.arange(168)+24)%168][:,(np.arange(168)+24)%168]-g)/np.linalg.norm(g))))
 dump(out/'actual_relation_kernels.json',actual);template_audit(r,out,cfg)
 saved_forecast_audit(r,'pilot_predictions.npz','pilot_summary.csv',out,'distill','label');saved_forecast_audit(t,'test_predictions.npz','test_scores.csv',out,'relation','method');weekday_audit(w,out)
 after={str(p):sha(p) for p in inputs};assert before==after,'Historical inputs changed'
 elapsed=time.monotonic()-start;assert elapsed<cfg['cpu_seconds']
 dump(out/'completion.json',dict(seconds=elapsed,historical_inputs_unchanged=True,n_inputs=len(inputs),new_model_training=0,new_llm_forwards=0,new_forecast_data_reads=0,selected_layer=cfg['selected_layer'],projection=proj))
 print(json.dumps(dict(seconds=elapsed,projection=[{k:x[k] for k in ['teacher','llm_explained_energy','llm_residual_energy']} for x in proj]),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);main(p.parse_args())
