"""Post-execution integrity/replay audit. Does not train or select interventions."""
import argparse, hashlib, json, platform, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.chronogeom.circle_pilot02 import get_model, capture_all, packs, metrics
from scripts.chronogeom.tslib_circle_transfer import forward_with_patch
from data_provider.chronogeom_circle_pilot02 import test_pack,args_for

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main(root):
    target=root/'validation.json'
    if target.exists():raise FileExistsError(target)
    cfg=json.loads((root/'config.json').read_text());p=json.loads((root/'provenance.json').read_text())
    torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
    old=Path(cfg['old_run']);manifest=json.loads((old/'run_manifest.json').read_text())
    for name,record in manifest['files'].items():
        f=old/name;assert f.stat().st_size==record['bytes'] and digest(f)==record['sha256'],name
    for name,sha in p['checkpoint_hashes'].items():assert digest(Path(name))==sha,name
    # The protocol and executable scientific core remain the preregistered files.
    frozen=['docs/chronogeom/CIRCLE_PILOT02_PROTOCOL.md','docs/chronogeom/circle_transfer/pilot02_config.json',
            'data_provider/chronogeom_circle_pilot02.py','scripts/chronogeom/circle_pilot02_math.py','scripts/chronogeom/circle_pilot02.py']
    hashes={}
    for name in frozen:
        recorded=subprocess.check_output(['git','show',p['source_commit']+':'+name]);current=Path(name).read_bytes()
        assert recorded==current,name;hashes[name]=hashlib.sha256(current).hexdigest()
    for name,pack in {**packs(cfg),'test_6':test_pack(cfg['n_test_a'],6),'test_-6':test_pack(cfg['n_test_a'],-6)}.items():
        stored=np.load(root/'data'/f'{name}.npz')
        for key,val in pack.items():np.testing.assert_array_equal(stored[key],val)
    table=pd.read_csv(root/'interventions.csv');assert len(table)==165888
    primary=['clean_pred_mse','base_clean_pred_mse','clean_truth_mse','phase_error','amplitude','delta_norm','mahal']
    assert np.isfinite(table[primary]).all().all()
    assert table.groupby(['seed','site','shift','condition']).size().eq(192).all()
    assert len(table.groupby(['seed','site','shift','condition']))==864
    assert table[['trajectory','cohort']].drop_duplicates().groupby('cohort').size().to_dict()=={2:96,3:96}
    assert all(len(set(packs(cfg)[a]['ids'])&set(packs(cfg)[b]['ids']))==0 for a,b in [('fit','val'),('fit','test'),('val','test')])
    pred_files=list((root/'interventions/predictions').glob('*.npz'));assert len(pred_files)==864
    for path in pred_files:
        arr=np.load(path)['prediction'];assert arr.shape==(192,24,1) and np.isfinite(arr).all(),path
    control=json.loads((root/'control_checks.json').read_text())
    nm=max(v['noop_max'] for v in control if v['noop_max'] is not None)
    fm=max(v['full_max'] for v in control if v['full_max'] is not None)
    assert nm==0 and fm==0
    norm=table.norm_match_error.dropna();assert norm.max()<.01
    replays=[];identity=[];ortherr=[]
    clean=test_pack(cfg['n_test_a'])
    for seed in cfg['seeds']:
        model=get_model(cfg['old_run'],seed,'trained');cp,ca=capture_all(model,clean,cfg['sites'])
        np.testing.assert_array_equal(cp,np.load(root/'cache'/f'{seed}_trained_baseline_test.npz')['prediction'])
        for shift in cfg['shifts']:
            bad=test_pack(cfg['n_test_a'],shift);bp,ba=capture_all(model,bad,cfg['sites'])
            for si,site in enumerate(cfg['sites']):
                basis=np.load(root/'interventions/bases'/f'{seed}_{si}.npz')
                base=ba[site][:,-24:].astype(float);donor=ca[site][:,-24:].astype(float)
                circle_delta=((donor-base)@basis['circle'])@basis['circle'].T
                reference=np.linalg.norm(circle_delta,axis=(1,2))
                for cond in ['circle','info_match_norm']:
                    q=basis['circle' if cond=='circle' else 'info_match']
                    ortherr.append(float(np.max(abs(q.T@q-np.eye(2)))))
                    delta=((donor-base)@q)@q.T
                    if cond.endswith('_norm'):
                        n=np.linalg.norm(delta,axis=(1,2));factor=np.divide(reference,n,out=np.ones_like(n),where=n>1e-10)
                        delta=delta*factor[:,None,None]
                    patch=ba[site].copy();patch[:,-24:]=(base+delta).astype(np.float32)
                    pred=forward_with_patch(model,site,args_for(bad,'cpu'),torch.from_numpy(patch),mode='full').numpy()
                    prior=np.load(root/'interventions/predictions'/f'{seed}_{si}_{shift}_{cond}.npz')['prediction']
                    err=float(np.max(abs(pred-prior)));assert err==0
                    vals=metrics(pred,cp,bp,clean,bad)
                    frame=table[(table.seed==seed)&table.site.eq(site)&(table['shift']==shift)&table.condition.eq(cond)].set_index('trajectory').loc[clean['ids']]
                    me=max(float(np.max(abs(frame[key].to_numpy()-arr))) for key,arr in vals.items() if arr.dtype!=bool)
                    assert me<1e-12
                    replays.append(dict(seed=seed,site=site,shift=shift,condition=cond,max_prediction_error=err,max_metric_error=me))
                if shift==cfg['shifts'][0]:
                    cent=np.load(root/'native'/f'{seed}_trained_baseline_{si}.npz')['fit_centroids']
                    q=basis['circle'];rem=(cent-cent.mean(0))@(np.eye(16)-q@q.T)
                    dist=np.linalg.norm(rem[:,None]-rem[None,:],axis=-1);np.fill_diagonal(dist,np.inf)
                    identity.append(dict(seed=seed,site=si,min_centroid_separation=float(dist.min()),all24_centroids_distinct=bool(dist.min()>1e-6),
                        note='post-hoc; exact finite-support identity claim only at hour lookup'))
        assert all(not m._forward_hooks for m in model.modules())
    posthoc=root/'summary/complement_identity_posthoc.csv'
    if posthoc.exists():
        previous=pd.read_csv(posthoc)
        np.testing.assert_allclose(previous['min_centroid_separation'],pd.DataFrame(identity)['min_centroid_separation'],atol=1e-12)
    else:
        pd.DataFrame(identity).to_csv(posthoc,index=False)
    result=dict(old_manifest_files_verified=len(manifest['files']),checkpoint_hashes_verified=len(p['checkpoint_hashes']),
        frozen_source_commit=p['source_commit'],frozen_source_hashes=hashes,data_packages_exact=5,raw_rows=len(table),prediction_packages=len(pred_files),
        noop_max=nm,final_full_max=fm,norm_match_records=len(norm),max_relative_norm_error=float(norm.max()),
        max_basis_orthogonality_error=max(ortherr),replay_groups=len(replays),replays=replays,
        posthoc_complement_identity_audit=identity,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',new_training=0)
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False));print({k:v for k,v in result.items() if k not in ['replays','frozen_source_hashes','posthoc_complement_identity_audit']})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);main(Path(ap.parse_args().run))
