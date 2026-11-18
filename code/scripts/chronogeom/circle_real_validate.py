"""Integrity/GPU replay and explicitly supplemental real residual-fidelity audit."""
import argparse,json,hashlib,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data_provider.chronogeom_circle_real import RealData,args_for
from scripts.chronogeom.circle_real_pilot import load_model,capture,metrics
from scripts.chronogeom.tslib_circle_transfer import forward_with_patch

def main(r):
    output=r/'validation.json'
    if output.exists():raise FileExistsError(output)
    start=time.monotonic();cfg=json.loads((r/'config.json').read_text());data=RealData(cfg['dataset']);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.15);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert hashlib.sha256(Path(cfg['dataset']).read_bytes()).hexdigest()==cfg['dataset_sha256']
    assert not subprocess.check_output(['git','diff','4e938a1767106324dd753b2a44832bf870a0252e','--','models/Transformer.py','layers/Embed.py','layers/Transformer_EncDec.py','layers/SelfAttention_Family.py'])
    for key in data.origins:
        p=data.split(key);stored=np.load(r/'data'/f'{key}.npz')
        for name,val in p.items():np.testing.assert_array_equal(val,stored[name])
    d=pd.read_csv(r/'interventions.csv');assert len(d)==84960 and d.groupby(['seed','site','arm','shift','condition']).size().eq(59).all()
    numeric=d.select_dtypes(include='number').drop(columns=['norm_match_error']);assert np.isfinite(numeric).all().all()
    assert d.norm_match_error.dropna().max()<.01
    predictions=list((r/'interventions/predictions').glob('*.npz'));assert len(predictions)==1440
    for p in predictions:
        a=np.load(p)['prediction'];assert a.shape==(59,24,1) and np.isfinite(a).all()
    controls=json.loads((r/'control_checks.json').read_text());assert len(controls)==60 and max(v['max_error'] for v in controls)==0
    orth=[]
    for p in (r/'interventions/bases').glob('[0-2]_[0-3].npz'):
        a=np.load(p)
        for name in ['circle','raw_pca','residual_pca','random0','random1','random2','shuffled','cv_phase','recon_match','info_match','joint_match']:
            q=a[name];assert q.shape==(16,2);orth.append(float(abs(q.T@q-np.eye(2)).max()))
    replay=[];test=data.split('test')
    for seed in cfg['seeds']:
        model=load_model(cfg,r,seed,'best');cp,ca=capture(model,test,cfg['sites'])
        bad=data.pack(test['origins'],clock_shift=6);bp,ba=capture(model,bad,cfg['sites'])
        donor=data.pack(test['origins']+6);dp,da=capture(model,donor,cfg['sites'])
        for arm,pack,basepred,acts,targetpred,targetacts,target in [('clock',bad,bp,ba,cp,ca,test),('natural',test,cp,ca,dp,da,donor)]:
            for si,site in enumerate(cfg['sites']):
                b=np.load(r/'interventions/bases'/f'{seed}_{si}.npz');base=acts[site][:,-24:].astype(float);other=targetacts[site][:,-24:].astype(float)
                ref=np.linalg.norm(((other-base)@b['circle'])@b['circle'].T,axis=(1,2))
                for name in ['circle','info_match_norm']:
                    if time.monotonic()-start>60:raise TimeoutError('60s replay cap')
                    q=b['circle' if name=='circle' else 'info_match'];delta=((other-base)@q)@q.T
                    if name.endswith('_norm'):
                        n=np.linalg.norm(delta,axis=(1,2));fac=np.divide(ref,n,out=np.ones_like(n),where=n>1e-10);delta*=fac[:,None,None]
                    patch=acts[site].copy();patch[:,-24:]=(base+delta).astype(np.float32)
                    pred=forward_with_patch(model,site,args_for(pack,'cuda'),torch.from_numpy(patch).to('cuda'),mode='full').cpu().numpy()
                    old=np.load(r/'interventions/predictions'/f'{seed}_{si}_{arm}_6_{name}.npz')['prediction'];err=float(abs(pred-old).max());assert err==0
                    saved=d[(d.seed==seed)&d.site.eq(site)&d.arm.eq(arm)&d['shift'].eq(6)&d.condition.eq(name)].set_index('trajectory').loc[test['ids']]
                    mm=metrics(pred,basepred,targetpred,test['y'],target['y']);me=max(float(abs(saved[k].to_numpy()-v).max()) for k,v in mm.items() if v.dtype!=bool);assert me<1e-12
                    replay.append(dict(seed=seed,site=site,arm=arm,condition=name,prediction_error=err,metric_error=me))
        assert all(not m._forward_hooks for m in model.modules())
    retained=list((r/'interventions_failed_attempt01/predictions').glob('*.npz'));assert len(retained)==1
    for p in retained:np.testing.assert_array_equal(np.load(p)['prediction'],np.load(r/'interventions/predictions'/p.name)['prediction'])
    # Explicit post-protocol descriptive supplement: real nonharmonic signal need not be noise/artifact.
    # Retain discrepancy from baseline and from observations rather than reward smoothing.
    h=np.arange(1,25);z=np.column_stack([np.ones(24),h,np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24)]);rp=np.eye(24)-z@np.linalg.pinv(z)
    truth_res=test['y'][...,0]@rp.T;supp=[]
    for path in predictions:
        seed,si,arm,shift,cond=path.stem.split('_',4);a=np.load(path)['prediction'][...,0]@rp.T
        noop=np.load(path.parent/f'{seed}_{si}_{arm}_{shift}_noop.npz')['prediction'][...,0]@rp.T
        er=np.mean((a-noop)**2,axis=1);et=np.mean((a-truth_res)**2,axis=1)
        for i,identity in enumerate(test['ids']):supp.append(dict(seed=int(seed),site=cfg['sites'][int(si)],arm=arm,shift=int(shift),condition=cond,trajectory=int(identity),
            residual_change_mse=float(er[i]),residual_error_observed_mse=float(et[i])))
    sp=pd.DataFrame(supp);sp.to_csv(r/'summary/residual_fidelity_cases_posthoc.csv',index=False)
    sp.groupby(['arm','site','condition'])[['residual_change_mse','residual_error_observed_mse']].mean().reset_index().to_csv(r/'summary/residual_fidelity_posthoc.csv',index=False)
    result=dict(source_protocol_commit='b2656f3',analysis_repair_commit='ffba596',data_replays=4,prediction_packages=len(predictions),raw_rows=len(d),noops_and_final_full=len(controls),max_control_error=0,
        norm_matched_records=int(d.norm_match_error.notna().sum()),max_norm_error=float(d.norm_match_error.max()),max_orthogonality_error=max(orth),
        exact_replays=len(replay),replays=replay,retained_failure_prediction_exact=True,checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'states').glob('*.pt')},
        supplementary='Post-protocol residual change and actual-observation residual error; all cases; not a revised primary metric',wall_s=time.monotonic()-start)
    output.write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k not in ['replays','checkpoint_sha256']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(Path(p.parse_args().run))
