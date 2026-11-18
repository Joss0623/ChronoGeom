"""User-requested GPU portability check of fixed Pilot02 conditions; no training."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.chronogeom.circle_pilot02 import get_model,metrics
from scripts.chronogeom.tslib_circle_transfer import forward_with_patch
from data_provider.chronogeom_circle_pilot02 import test_pack,args_for

def capture(model,pack,sites):
    cache={};modules=dict(model.named_modules());handles=[]
    for site in sites:
        def hook(m,a,o,key=site):cache[key]=o.detach().cpu().numpy().copy()
        handles.append(modules[site].register_forward_hook(hook))
    try:
        with torch.inference_mode():pred=model(*args_for(pack,'cuda')).cpu().numpy()
    finally:
        for h in handles:h.remove()
    return pred,cache

def main(root):
    out=root/'gpu_replay';out.mkdir();start=time.monotonic()
    cfg=json.loads((root/'config.json').read_text());torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.04,0) # <1 GiB tensor allocator on the 24 GiB card; CUDA context additional.
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    conditions=['noop','full_donor','circle','cv_phase_norm','info_match_norm','harmonic_mean_delta_norm']
    (out/'protocol.json').write_text(json.dumps(dict(reason='User explicitly requests GPU execution after CPU main analysis; supplemental portability check only',
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),cap_s=60,allocator_fraction=.04,conditions=conditions,
        seeds=cfg['seeds'],sites=cfg['sites'],shifts=cfg['shifts'],tolerance=1e-4,new_training=0),indent=2))
    rows=[];errors=[];controls=[];clean=test_pack(cfg['n_test_a'])
    phase=lambda t:np.stack([np.cos(2*np.pi*t/24),np.sin(2*np.pi*t/24)],axis=-1)
    for seed in cfg['seeds']:
        model=get_model(cfg['old_run'],seed,'trained').to('cuda');cp,ca=capture(model,clean,cfg['sites'])
        for shift in cfg['shifts']:
            bad=test_pack(cfg['n_test_a'],shift);bp,ba=capture(model,bad,cfg['sites'])
            for si,site in enumerate(cfg['sites']):
                basis=np.load(root/'interventions/bases'/f'{seed}_{si}.npz')
                base=ba[site][:,-24:].astype(float);donor=ca[site][:,-24:].astype(float)
                delta=((donor-base)@basis['circle'])@basis['circle'].T;ref=np.linalg.norm(delta,axis=(1,2))
                changes=dict(noop=base.copy(),full_donor=donor,circle=base+delta)
                for name in ['cv_phase','info_match']:
                    q=basis[name];dd=((donor-base)@q)@q.T;dn=np.linalg.norm(dd,axis=(1,2))
                    fac=np.divide(ref,dn,out=np.ones_like(dn),where=dn>1e-10);changes[name+'_norm']=base+dd*fac[:,None,None]
                hd=(phase(clean['t'])-phase(bad['t']))@basis['circle_phase_coef'];dn=np.linalg.norm(hd,axis=(1,2))
                fac=np.divide(ref,dn,out=np.ones_like(dn),where=dn>1e-10);changes['harmonic_mean_delta_norm']=base+hd*fac[:,None,None]
                for name in conditions:
                    if time.monotonic()-start>60:raise TimeoutError('GPU replay60s cap')
                    patch=ba[site].copy();patch[:,-24:]=changes[name].astype(np.float32)
                    pred=forward_with_patch(model,site,args_for(bad,'cuda'),torch.from_numpy(patch).to('cuda'),mode='full').cpu().numpy()
                    cpu=np.load(root/'interventions/predictions'/f'{seed}_{si}_{shift}_{name}.npz')['prediction']
                    err=float(abs(pred-cpu).max());errors.append(dict(seed=seed,site=site,shift=shift,condition=name,cpu_gpu_max_error=err,passed=err<=1e-4))
                    ctrl=float(abs(pred-bp).max()) if name=='noop' else float(abs(pred-cp).max()) if name=='full_donor' and si==3 else None
                    if ctrl is not None:controls.append(ctrl)
                    np.savez_compressed(out/f'{seed}_{si}_{shift}_{name}.npz',prediction=pred)
                    met=metrics(pred,cp,bp,clean,bad)
                    for i,identity in enumerate(clean['ids']):
                        rows.append(dict(seed=seed,site=site,shift=shift,condition=name,trajectory=int(identity),**{k:bool(v[i]) if k=='phase_defined' else float(v[i]) for k,v in met.items()}))
        del model
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(errors).to_csv(out/'comparison.csv',index=False)
    d=pd.DataFrame(rows);rec=d.groupby(['site','condition'])[['clean_pred_mse','base_clean_pred_mse']].mean();rec['recovery']=1-rec.clean_pred_mse/rec.base_clean_pred_mse
    rec.to_csv(out/'recovery.csv');summary=dict(complete=True,wall_s=time.monotonic()-start,groups=len(errors),rows=len(rows),
        device=torch.cuda.get_device_name(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),max_tensor_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_tensor_reserved_bytes=torch.cuda.max_memory_reserved(),cpu_gpu_max_error=max(v['cpu_gpu_max_error'] for v in errors),
        tolerance=1e-4,failed_groups=sum(not v['passed'] for v in errors),noop_final_full_max=max(controls),new_training=0)
    (out/'completion.json').write_text(json.dumps(summary,indent=2));print(summary)
    assert max(controls)==0 and all(v['passed'] for v in errors)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);main(Path(ap.parse_args().run))
