"""Bounded GPU4 recovery: audit -> selection -> locked qualification -> one blind test."""
import argparse,json,time,hashlib,subprocess,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data_provider.chronogeom_recovery import RecoveryData
from data_provider.chronogeom_circle_real import args_for
from exp.exp_chronogeom_recovery import make_model
from scripts.chronogeom.circle_real_pilot import waveform

def js(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False,ensure_ascii=False))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mse(a,b):return float(np.mean((a-b)**2))
def pred(model,pack):
    model.eval()
    with torch.inference_mode():return model(*args_for(pack,'cuda')).cpu().numpy()
def score(a,pack):
    err=a-pack['y'];bias=err.mean(1,keepdims=True)
    return dict(mse=mse(a,pack['y']),mae=float(abs(err).mean()),bias_mse=float((bias**2).mean()),centered_error_mse=float(((err-bias)**2).mean()),
        mean_forecast=float(a.mean()),mean_observed=float(pack['y'].mean()))
def load(cfg,r,tag):
    item=json.loads((r/'training_lock.json').read_text())['models'][tag]
    p=r/item['path'];assert digest(p)==item['sha256'];m=make_model(cfg['model'],item['variant']).to('cuda')
    m.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['state']);return m.eval()

def audit(cfg,r,d):
    out=r/'audit';out.mkdir();rows=[];pack=d.split('select');base=np.repeat(pack['x'][:,-1:],24,axis=1)
    rows.append(dict(model='persistence',**score(base,pack)))
    from models.Transformer import Model
    from types import SimpleNamespace
    for seed in cfg['seeds']:
        model=Model(SimpleNamespace(**cfg['model'])).to('cuda');state=torch.load(Path(cfg['old_run'])/'states'/f'best_{seed}.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['state']);p=pred(model,pack)
        rows.append(dict(model=f'old_seed{seed}',**score(p,pack)));np.savez_compressed(out/f'old_{seed}.npz',prediction=p)
        anchored=p+(pack['x'][:,-1:]-p[:,:1]);rows.append(dict(model=f'old_seed{seed}_history_anchor_diagnostic',**score(anchored,pack)))
        shifted={k:v.copy() for k,v in pack.items()};shifted['x']+=1;shifted['xd'][:,:48]+=1
        shifted_pred=pred(model,shifted);rows[-2]['translation_error_rms']=float(np.sqrt(np.mean((shifted_pred-p-1)**2)))
    pd.DataFrame(rows).to_csv(out/'old_validation_audit.csv',index=False)
    # One-batch fit on training data only; not a candidate selected for forecasting.
    torch.manual_seed(7);model=make_model(cfg['model'],'raw').to('cuda');fixed=d.pack(np.random.default_rng(7).choice(d.train_origins,8,replace=False));a=args_for(fixed,'cuda');y=torch.as_tensor(fixed['y'],device='cuda');opt=torch.optim.Adam(model.parameters(),lr=.001);losses=[]
    for step in range(cfg['overfit_steps']+1):
        model.train();opt.zero_grad(set_to_none=True);loss=((model(*a)-y)**2).mean()
        if step%25==0:losses.append(dict(step=step,mse=float(loss.detach())))
        if step<cfg['overfit_steps']:loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    pd.DataFrame(losses).to_csv(out/'one_batch.csv',index=False)
    js(out/'alignment.json',dict(history_last_index=int(fixed['origins'][0]-1),target_first_index=int(fixed['origins'][0]),future_decoder_zeros=bool((fixed['xd'][:,-24:]==0).all()),
        input_shape=list(fixed['x'].shape),target_shape=list(fixed['y'].shape),one_batch_initial=losses[0]['mse'],one_batch_final=losses[-1]['mse'],csv_rows_read=d.read_limit,old_test_evaluated=False,blind_evaluated=False))

def train(cfg,r,d):
    start=time.monotonic();logs=[];results=[];alltrain=d.pack(d.train_origins);ta=args_for(alltrain,'cuda');ty=torch.as_tensor(alltrain['y'],device='cuda');v=d.split('select');va=args_for(v,'cuda');vy=torch.as_tensor(v['y'],device='cuda')
    def fit(variant,seed,steps,tag):
        if time.monotonic()-start>cfg['train_cap_s']:raise TimeoutError('Training cap')
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
        model=make_model(cfg['model'],variant).to('cuda');opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']);best=np.inf;last_best=0;best_state=None
        if tag.startswith('final_clock'):torch.save(dict(state={k:v.cpu().clone() for k,v in model.state_dict().items()},variant=variant),r/'states'/f'{tag}_init.pt')
        for step in range(1,steps+1):
            if time.monotonic()-start>cfg['train_cap_s']:raise TimeoutError('Training cap; retain logged candidates, no blind evaluation')
            ids=torch.as_tensor(rng.integers(len(ty),size=cfg['batch']),device='cuda');model.train();opt.zero_grad(set_to_none=True)
            forecast=model(*(t[ids] for t in ta));loss=((forecast-ty[ids])**2).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            if step%cfg['validate_every']==0:
                model.eval()
                with torch.inference_mode():pv=model(*va);val=float(((pv-vy)**2).mean())
                logs.append(dict(tag=tag,variant=variant,seed=seed,step=step,train_mse=float(loss.detach()),val_mse=val,grad_norm=float(norm),wall_s=time.monotonic()-start))
                if val<best:
                    best=val;last_best=step;best_state=dict(state={k:v.cpu().clone() for k,v in model.state_dict().items()},step=step,variant=variant,seed=seed,val_mse=val,optimizer=opt.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
                if step>=cfg['min_steps'] and step-last_best>=cfg['patience_steps']:break
        path=r/'states'/f'{tag}.pt';torch.save(best_state,path);model.load_state_dict(best_state['state']);pv=pred(model,v);np.savez_compressed(r/'predictions'/f'{tag}_select.npz',prediction=pv)
        row=dict(tag=tag,variant=variant,seed=seed,step=last_best,updates=step,params=sum(p.numel() for p in model.parameters()),val_mse=best,path=str(path.relative_to(r)),sha256=digest(path));results.append(row)
        pd.DataFrame(logs).to_csv(r/'training.csv',index=False);pd.DataFrame(results).to_csv(r/'selection.csv',index=False);print('fit',tag,'val',round(best,6),'best',last_best,'updates',step,'wall',round(time.monotonic()-start,1),flush=True)
        return row
    screens=[fit(v,0,cfg['screen_steps'],'screen_'+v) for v in cfg['variants']]
    compatible=[x for x in screens if x['variant'] in ['raw','instance','nonstationary']];selected=min(compatible,key=lambda x:x['val_mse'])['variant']
    js(r/'family_selection.json',dict(source='selection validation only',selected_clock=selected,candidates=screens,csv_rows_read=d.read_limit,old_test_used=False))
    for seed in cfg['seeds']:fit(selected,seed,cfg['final_steps'],f'final_clock_{seed}')
    for seed in cfg['seeds']:fit('dlinear',seed,cfg['final_steps'],f'final_linear_{seed}')
    models={x['tag']:x for x in results if x['tag'].startswith('final_')}
    js(r/'training_lock.json',dict(models=models,selected_clock=selected,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),csv_rows_read=d.read_limit,
        selection_only=True,old_test_used=False,clock_labels_in_student=False))
    js(r/'training_completion.json',dict(complete=True,wall_s=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))

def qualify(cfg,r,d):
    if (r/'qualification.json').exists():raise FileExistsError('Qualification is one-shot')
    pack=d.split('calibration');baseline=np.repeat(pack['x'][:,-1:],24,axis=1);risk=mse(baseline,pack['y']);rows=[]
    for family in ['clock','linear']:
        for seed in cfg['seeds']:
            tag=f'final_{family}_{seed}';p=pred(load(cfg,r,tag),pack);np.savez_compressed(r/'predictions'/f'{tag}_qualification.npz',prediction=p)
            rows.append(dict(family=family,seed=seed,**score(p,pack),persistence=risk))
    table=pd.DataFrame(rows);table.to_csv(r/'qualification.csv',index=False)
    result={}
    for family in ['clock','linear']:
        v=table[table.family==family];result[family+'_qualified']=bool(v.mse.mean()<=cfg['qualification_ratio']*risk and (v.mse<risk).sum()>=2)
        result[family+'_mean_mse']=float(v.mse.mean())
    result.update(persistence=risk,threshold_ratio=cfg['qualification_ratio'],csv_rows_read=d.read_limit,old_test_used=False,unseen_tail_still_sealed=True)
    js(r/'qualification.json',result);print(result)

def blind(cfg,r,d):
    if (r/'blind_completion.json').exists():raise FileExistsError('Blind evaluation already consumed')
    gate=json.loads((r/'qualification.json').read_text());pack=d.split('test');np.savez_compressed(r/'data/blind.npz',**pack);rows=[]
    for name,p in [('persistence',np.repeat(pack['x'][:,-1:],24,axis=1)),('seasonal24',pack['x'][:,-24:])]:
        rows.append(dict(model=name,seed=-1,**score(p,pack)));np.savez_compressed(r/'predictions'/f'{name}_blind.npz',prediction=p)
    for family in ['clock','linear']:
        if not gate[family+'_qualified']:continue
        for seed in cfg['seeds']:
            tag=f'final_{family}_{seed}';p=pred(load(cfg,r,tag),pack);np.savez_compressed(r/'predictions'/f'{tag}_blind.npz',prediction=p);rows.append(dict(model=family,seed=seed,**score(p,pack)))
    pd.DataFrame(rows).to_csv(r/'blind_results.csv',index=False);risk=rows[0]['mse'];clock=[x['mse'] for x in rows if x['model']=='clock']
    js(r/'blind_completion.json',dict(complete=True,csv_rows_read=d.read_limit,test_target_range=[14400,15912],final_reserve=[15912,17420],
        clock_blind_pass=bool(len(clock)==3 and np.mean(clock)<risk and sum(x<risk for x in clock)>=2),blind_evaluation_count=1))
    print(pd.DataFrame(rows).to_string(index=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','train','qualify','blind']);p.add_argument('--config',required=True);p.add_argument('--run',required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text());r=Path(a.run)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.08);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    access={'audit':'develop','train':'develop','qualify':'qualification','blind':'blind'}[a.stage];d=RecoveryData(cfg['dataset'],r,access)
    if a.stage=='audit':
        r.mkdir();[ (r/name).mkdir() for name in ['states','predictions','data'] ];js(r/'config.json',cfg);js(r/'provenance.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),device=torch.cuda.get_device_name(),physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,numpy=np.__version__))
    if a.stage=='train':cfg['train_cap_s']-=json.loads((r/'audit_completion.json').read_text())['wall_s']
    started=time.monotonic()
    try:
        {'audit':audit,'train':train,'qualify':qualify,'blind':blind}[a.stage](cfg,r,d)
        if a.stage=='audit':js(r/'audit_completion.json',dict(wall_s=time.monotonic()-started,complete=True))
    except Exception:
        if r.exists():(r/(a.stage+'_failure.txt')).write_text(traceback.format_exc())
        raise
if __name__=='__main__':main()
