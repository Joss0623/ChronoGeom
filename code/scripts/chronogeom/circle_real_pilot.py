"""Real-data Circle transfer in pinned TSLib. All models use unchanged official Transformer."""
import argparse,json,time,hashlib,subprocess,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from models.Transformer import Model
from data_provider.chronogeom_circle_real import RealData,args_for,nuisance,purged_folds
from scripts.chronogeom.circle_pilot02_math import native_audit,centroids,basis,apply_residual,decode_metrics
from scripts.chronogeom.circle_real_math import calibrate_real
from scripts.chronogeom.tslib_circle_transfer import forward_with_patch

H=24

def js(path,v):path.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False))
def phase(t):return np.stack([np.cos(2*np.pi*t/24),np.sin(2*np.pi*t/24)],axis=-1)
def capture(model,pack,sites,device='cuda'):
    cache={};handles=[];mods=dict(model.named_modules())
    for site in sites:
        def hook(m,a,o,key=site):cache[key]=o.detach().cpu().numpy().copy()
        handles.append(mods[site].register_forward_hook(hook))
    try:
        with torch.inference_mode():p=model(*args_for(pack,device)).cpu().numpy()
    finally:
        for h in handles:h.remove()
    return p,cache

def load_model(cfg,run,seed,stage):
    model=Model(SimpleNamespace(**cfg['model'])).eval()
    state=torch.load(run/'states'/f'{stage}_{seed}.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['state']);return model.to('cuda')

def prepare(cfg,run,data):
    run.mkdir();(run/'states').mkdir();(run/'data').mkdir();(run/'cache').mkdir()
    js(run/'config.json',cfg)
    for name in data.origins:np.savez_compressed(run/'data'/f'{name}.npz',**data.split(name))
    folds=purged_folds(data.origins['fit'])
    js(run/'dataset.json',dict(source=json.loads(Path(cfg['source']).read_text()),mean_train=data.mu,std_train=data.sd,
        rows=len(data.df),official_borders=[0,8640,11520,14400],unused_tail=len(data.df)-14400,
        analysis_counts={k:len(v) for k,v in data.origins.items()},test_blocks=int(len(np.unique(data.split('test')['block']))),
        purged_cv=[dict(train_origins=data.origins['fit'][a].tolist(),test_origins=data.origins['fit'][b].tolist()) for a,b in folds]))
    js(run/'provenance.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        upstream='4e938a1767106324dd753b2a44832bf870a0252e',python=__import__('platform').python_version(),
        torch=torch.__version__,numpy=np.__version__,physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),device=torch.cuda.get_device_name()))

def train(cfg,run,data):
    start=time.monotonic();hist=[];v=data.split('model_val');test=data.split('test');forecasts=[]
    for seed in cfg['seeds']:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
        model=Model(SimpleNamespace(**cfg['model'])).to('cuda');opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
        torch.save(dict(state={k:v.cpu().clone() for k,v in model.state_dict().items()}),run/'states'/f'init_{seed}.pt')
        best=np.inf;best_step=0
        for step in range(1,cfg['steps']+1):
            if time.monotonic()-start>cfg['train_cap_s']:raise TimeoutError('Training cap')
            pack=data.pack(rng.choice(data.train_origins,cfg['batch'],replace=True));model.train();opt.zero_grad(set_to_none=True)
            pred=model(*args_for(pack,'cuda'));loss=((pred-torch.as_tensor(pack['y'],device='cuda'))**2).mean();loss.backward();opt.step()
            if step%cfg['validate_every']==0:
                model.eval()
                with torch.inference_mode():pv=model(*args_for(v,'cuda')).cpu().numpy()
                mse=float(np.mean((pv-v['y'])**2));hist.append(dict(seed=seed,step=step,train_mse=float(loss.detach()),val_mse=mse,wall_s=time.monotonic()-start))
                if mse<best:
                    best=mse;best_step=step
                    torch.save(dict(state={k:v.cpu().clone() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),step=step,val_mse=mse,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=rng.bit_generator.state),run/'states'/f'best_{seed}.pt')
        del model;model=load_model(cfg,run,seed,'best');pred,acts=capture(model,test,cfg['sites']);np.savez_compressed(run/'cache'/f'test_{seed}.npz',prediction=pred)
        forecasts.append(dict(seed=seed,step=best_step,params=sum(p.numel() for p in model.parameters()),val_mse=best,test_mse=float(np.mean((pred-test['y'])**2))))
        print('trained',seed,best_step,best,'wall',round(time.monotonic()-start,2),flush=True)
    pd.DataFrame(hist).to_csv(run/'training.csv',index=False);pd.DataFrame(forecasts).to_csv(run/'forecast.csv',index=False)
    base=[]
    for name,pred in [('persistence',np.repeat(test['x'][:,-1:],H,axis=1)),('seasonal24',test['x'][:,-24:]),('train_mean',np.zeros_like(test['y']))]:
        np.savez_compressed(run/'cache'/f'baseline_{name}.npz',prediction=pred)
        base.append(dict(method=name,test_mse=float(np.mean((pred-test['y'])**2))))
    pd.DataFrame(base).to_csv(run/'baselines.csv',index=False)
    js(run/'training_completion.json',dict(complete=True,wall_s=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))

def waveform(pred):
    z=np.column_stack([np.ones(H),np.arange(1,H+1),np.sin(2*np.pi*np.arange(1,H+1)/24),np.cos(2*np.pi*np.arange(1,H+1)/24)])
    p=pred[...,0];coef=p@np.linalg.pinv(z).T
    return coef,np.hypot(coef[:,2],coef[:,3]),np.arctan2(coef[:,3],coef[:,2]),np.mean((p-coef@z.T)**2,axis=1)

def metrics(pred,base,target,truth,donor_truth):
    co,amp,ang,nh=waveform(pred);cb,ab,ba,_=waveform(base);ct,at,ta,_=waveform(target);cy,ay,ya,_=waveform(truth)
    wrap=lambda a:np.angle(np.exp(1j*a))
    return dict(target_pred_mse=np.mean((pred-target)**2,axis=(1,2)),base_target_mse=np.mean((base-target)**2,axis=(1,2)),
        recipient_truth_mse=np.mean((pred-truth)**2,axis=(1,2)),base_truth_mse=np.mean((base-truth)**2,axis=(1,2)),
        donor_truth_mse=np.mean((pred-donor_truth)**2,axis=(1,2)),prediction_change_mse=np.mean((pred-base)**2,axis=(1,2)),
        phase_error_target=np.abs(wrap(ang-ta)),phase_error_observed=np.abs(wrap(ang-ya)),phase_change=wrap(ang-ba),
        target_phase_change=wrap(ta-ba),phase_defined=(amp>=1e-3)&(at>=1e-3),observed_phase_defined=(amp>=1e-3)&(ay>=1e-3),
        amplitude=amp,amplitude_change=np.abs(amp-ab),amplitude_error_observed=np.abs(amp-ay),
        offset_change=np.abs(co[:,0]-cb[:,0]),trend_change=np.abs(co[:,1]-cb[:,1]),
        offset_error_observed=np.abs(co[:,0]-cy[:,0]),trend_error_observed=np.abs(co[:,1]-cy[:,1]),nonharmonic_mse=nh)

def native(cfg,run,data):
    out=run/'native';out.mkdir();start=time.monotonic();rows=[]
    f=data.split('fit');test=data.split('test');h=f['t'].ravel()%24;th=test['t'].ravel()%24
    for seed in cfg['seeds']:
        for stage in ['init','best']:
            model=load_model(cfg,run,seed,stage);fp,fa=capture(model,f,cfg['sites']);tp,ta=capture(model,test,cfg['sites'])
            for si,site in enumerate(cfg['sites']):
                if time.monotonic()-start>cfg['analysis_cap_s']:raise TimeoutError('Native cap')
                x=fa[site][:,-H:].reshape(-1,16).astype(float);tx=ta[site][:,-H:].reshape(-1,16).astype(float)
                m,a=native_audit(x,h,tx,th,len(f['ids']),cfg['permutations'],0,1729+seed)
                # Cluster resampling for time-dependent real windows, not independent-window bootstrap.
                blocks=(f['origins']-f['origins'].min())//336;keys=np.unique(blocks);rng=np.random.default_rng(1729+seed);angs=[]
                for _ in range(cfg['pca_bootstrap']):
                    ids=np.concatenate([np.where(blocks==b)[0] for b in rng.choice(keys,len(keys),replace=True)])
                    cc=centroids(x.reshape(len(f['ids']),H,16)[ids].reshape(-1,16),h.reshape(-1,H)[ids].ravel())
                    q=basis(cc-cc.mean(0));angs.append(np.degrees(np.arccos(np.clip(np.linalg.svd(a['pca_basis'].T@q,compute_uv=False),0,1))))
                m['stability_max_angle_deg']=float(np.max(angs));a['stability_angles']=np.array(angs)
                rows.append(dict(seed=seed,stage=stage,site=site,fit_blocks=len(keys),**m));np.savez_compressed(out/f'{seed}_{stage}_{si}.npz',**a)
            print('geometry',seed,stage,round(time.monotonic()-start,2),flush=True)
    pd.DataFrame(rows).to_csv(run/'native_geometry.csv',index=False);js(run/'native_completion.json',dict(complete=True,wall_s=time.monotonic()-start))

def analyze(cfg,run,data):
    out=run/'interventions';out.mkdir();(out/'bases').mkdir();(out/'predictions').mkdir();start=time.monotonic()
    f=data.split('fit');v=data.split('val');test=data.split('test');nf=nuisance(f);nv=nuisance(v);nt=nuisance(test)
    h=f['t'].ravel()%24;vh=v['t'].ravel()%24;th=test['t'].ravel()%24;rows=[];cal=[];scores=[];controls=[];masks=[]
    for seed in cfg['seeds']:
        model=load_model(cfg,run,seed,'best');cp,ca=capture(model,test,cfg['sites']);_,fa=capture(model,f,cfg['sites']);_,va=capture(model,v,cfg['sites'])
        from scripts.chronogeom.circle_pilot02 import input_mode
        for mode in ['baseline','calendar_zero','position_zero','both_zero']:
            with input_mode(model,mode):mp,_=capture(model,test,cfg['sites'])
            masks.append(dict(seed=seed,mode=mode,mse=float(np.mean((mp-test['y'])**2))))
        pairs={}
        for shift in cfg['shifts']:
            corrupted=data.pack(test['origins'],clock_shift=shift);bp,ba=capture(model,corrupted,cfg['sites'])
            natural=data.pack(test['origins']+shift);dp,da=capture(model,natural,cfg['sites'])
            pairs[('clock',shift)]=(corrupted,bp,ba,test,cp,ca)
            pairs[('natural',shift)]=(test,cp,ca,natural,dp,da)
        for si,site in enumerate(cfg['sites']):
            if time.monotonic()-start>cfg['analysis_cap_s']:raise TimeoutError('Analysis cap')
            x=fa[site][:,-H:].reshape(-1,16).astype(float);vx=va[site][:,-H:].reshape(-1,16).astype(float);tx=ca[site][:,-H:].reshape(-1,16).astype(float)
            fit,b,bases,dec,diag,pool=calibrate_real(x,h,nf,vx,vh,nv,len(f['ids']),cfg,seed,f['origins'])
            np.savez_compressed(out/'bases'/f'{seed}_{si}.npz',nuisance_coef=b,mean=fit.mean,circle_phase_coef=fit.full_coef[-2:],**bases)
            np.savez_compressed(out/'bases'/f'pool_{seed}_{si}.npz',**pool)
            for row in diag:cal.append(dict(seed=seed,site=site,**row))
            rt=apply_residual(tx,nt,b)
            for name,q in bases.items():
                r2,err,_=decode_metrics(rt,phase(th),q,dec[name]);scores.append(dict(seed=seed,site=site,space=name,test_phase_r2=r2,test_capture=float(np.sum((rt@q)**2)/np.sum(rt**2))))
            sample=np.linspace(0,len(x)-1,min(512,len(x))).astype(int);tree=cKDTree(x[sample]);hc=centroids(x,h);within=x-hc[h]
            cov=within.T@within/len(x);ridge=max(np.trace(cov)/16*.01,np.var(hc,axis=0).sum()/16*.01,1e-6);ci=np.linalg.pinv(cov+np.eye(16)*ridge)
            scale=max(np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=1))),1e-6)
            for (arm,shift),(recipient,bp,ba,target,dp,da) in pairs.items():
                base=ba[site][:,-H:].astype(float);donor=da[site][:,-H:].astype(float);n=len(base)
                changes={'noop':base.copy(),'full_donor':donor.copy()};matched={}
                for name,q in bases.items():changes[name]=base+((donor-base)@q)@q.T
                ref=np.linalg.norm(changes['circle']-base,axis=(1,2))
                for name in bases:
                    if name=='circle':continue
                    delta=changes[name]-base;dn=np.linalg.norm(delta,axis=(1,2));fac=np.divide(ref,dn,out=np.ones_like(dn),where=dn>1e-10)
                    changes[name+'_norm']=base+delta*fac[:,None,None];matched[name+'_norm']=ref
                hd=(phase(target['t'])-phase(recipient['t']))@fit.full_coef[-2:];changes['harmonic_mean_delta']=base+hd
                dn=np.linalg.norm(hd,axis=(1,2));fac=np.divide(ref,dn,out=np.ones_like(dn),where=dn>1e-10);changes['harmonic_mean_delta_norm']=base+hd*fac[:,None,None];matched['harmonic_mean_delta_norm']=ref
                for name in ['circle','residual_pca','info_match']:
                    q=bases[name];changes[name+'_ablate']=base+((fit.mean-base)@q)@q.T
                q=bases['random0'];delta=((fit.mean-base)@q)@q.T;dn=np.linalg.norm(delta,axis=(1,2));rr=np.linalg.norm(changes['circle_ablate']-base,axis=(1,2))
                fac=np.divide(rr,dn,out=np.ones_like(dn),where=dn>1e-10);changes['random0_ablate_norm']=base+delta*fac[:,None,None];matched['random0_ablate_norm']=rr
                changes['full_ablate']=np.broadcast_to(fit.mean,base.shape).copy()
                for name,changed in changes.items():
                    if time.monotonic()-start>cfg['analysis_cap_s']:raise TimeoutError('Analysis cap')
                    changed=changed.astype(np.float32);patch=ba[site].copy();patch[:,-H:]=changed
                    pred=forward_with_patch(model,site,args_for(recipient,'cuda'),torch.from_numpy(patch).to('cuda'),mode='full').cpu().numpy()
                    np.savez_compressed(out/'predictions'/f'{seed}_{si}_{arm}_{shift}_{name}.npz',prediction=pred)
                    met=metrics(pred,bp,dp,test['y'],target['y']);flat=changed.reshape(-1,16).astype(float);dn=np.linalg.norm(changed-base,axis=(1,2))
                    nn=tree.query(flat,k=1,workers=2)[0].reshape(n,H).mean(1)/scale
                    delta=flat-hc[target['t'].ravel()%24];mahal=np.einsum('nd,de,ne->n',delta,ci,delta).reshape(n,H).mean(1)
                    mismatch=abs(dn-matched[name])/np.maximum(matched[name],1e-10) if name in matched else np.full(n,np.nan)
                    for i,identity in enumerate(test['ids']):
                        rows.append(dict(seed=seed,site=site,arm=arm,shift=shift,condition=name,trajectory=int(identity),cohort=int(test['cohort'][i]),block=int(test['block'][i]),
                            **{k:bool(v[i]) if v.dtype==bool else float(v[i]) for k,v in met.items()},delta_norm=float(dn[i]),norm_match_error=float(mismatch[i]),
                            relative_delta=float(dn[i]/max(np.linalg.norm(base[i]),1e-10)),nn_state_distance=float(nn[i]),target_centroid_distance=float(np.linalg.norm(delta,axis=1).reshape(n,H).mean(1)[i]/scale),mahal=float(mahal[i])))
                    if name=='noop':controls.append(dict(seed=seed,site=site,arm=arm,shift=shift,kind='noop',max_error=float(abs(pred-bp).max())))
                    if name=='full_donor' and si==3:controls.append(dict(seed=seed,site=site,arm=arm,shift=shift,kind='final_full',max_error=float(abs(pred-dp).max())))
            print('intervened',seed,site,'wall',round(time.monotonic()-start,2),flush=True)
    pd.DataFrame(rows).to_csv(run/'interventions.csv',index=False);pd.DataFrame(cal).to_csv(run/'calibration.csv',index=False);pd.DataFrame(scores).to_csv(run/'test_space_scores.csv',index=False);pd.DataFrame(masks).to_csv(run/'input_controls.csv',index=False)
    js(run/'control_checks.json',controls);assert max(c['max_error'] for c in controls)==0
    js(run/'analysis_completion.json',dict(complete=True,wall_s=time.monotonic()-start,records=len(rows),conditions=len(changes),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','geometry','analyze']);p.add_argument('--config',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());run=Path(a.run);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    prior=sum(json.loads(p.read_text())['wall_s'] for p in run.glob('*_completion.json')) if run.exists() else 0
    remaining=max(0,cfg['total_cap_s']-prior);cfg['train_cap_s']=min(cfg['train_cap_s'],remaining);cfg['analysis_cap_s']=min(cfg['analysis_cap_s'],remaining)
    data=RealData(cfg['dataset']);assert hashlib.sha256(Path(cfg['dataset']).read_bytes()).hexdigest()==cfg['dataset_sha256']
    try:
        if a.stage=='train':prepare(cfg,run,data);train(cfg,run,data)
        elif a.stage=='geometry':native(cfg,run,data)
        else:analyze(cfg,run,data)
    except Exception:
        if run.exists():(run/(a.stage+'_failure.txt')).write_text(traceback.format_exc())
        raise
if __name__=='__main__':main()
