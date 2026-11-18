"""One declared selection-only amendment, before qualification/blind. No extra trainable layers."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from data_provider.chronogeom_recovery import RecoveryData
from data_provider.chronogeom_circle_real import args_for
from exp.exp_chronogeom_recovery import make_model
from scripts.chronogeom.recover_etth1 import js,digest,pred,mse

def main(r,cfg):
    assert not (r/'qualification.json').exists() and not (r/'blind_completion.json').exists()
    old=json.loads((r/'training_lock_stage1.json').read_text());elapsed=json.loads((r/'training_completion.json').read_text())['wall_s']+json.loads((r/'audit_completion.json').read_text())['wall_s']
    budget=600-elapsed;start=time.monotonic();d=RecoveryData(cfg['dataset']);pack=d.pack(d.train_origins);a=args_for(pack,'cuda');y=torch.as_tensor(pack['y'],device='cuda');v=d.split('select');va=args_for(v,'cuda');vy=torch.as_tensor(v['y'],device='cuda');logs=[];results=[]
    def fit(seed,steps,tag):
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=make_model(cfg['model'],'last_anchor').to('cuda');opt=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');last=0;state=None
        if tag.startswith('anchor_final'):torch.save(dict(state={k:v.cpu().clone() for k,v in model.state_dict().items()},variant='last_anchor'),r/'states'/f'{tag}_init.pt')
        for step in range(1,steps+1):
            if time.monotonic()-start>budget:raise TimeoutError('Shared600s training budget reached')
            ids=torch.as_tensor(rng.integers(len(y),size=32),device='cuda');model.train();opt.zero_grad(set_to_none=True);loss=((model(*(x[ids] for x in a))-y[ids])**2).mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            if step%100==0:
                model.eval()
                with torch.inference_mode():risk=float(((model(*va)-vy)**2).mean())
                logs.append(dict(tag=tag,seed=seed,step=step,train_mse=float(loss.detach()),val_mse=risk,grad_norm=float(norm)))
                pd.DataFrame(logs).to_csv(r/'anchor_training.csv',index=False)
                if risk<best:
                    best=risk;last=step;state=dict(state={k:v.cpu().clone() for k,v in model.state_dict().items()},variant='last_anchor',seed=seed,step=step,val_mse=risk,optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=rng.bit_generator.state)
                if step>=300 and step-last>=300:break
        p=r/'states'/f'{tag}.pt';torch.save(state,p);row=dict(tag=tag,variant='last_anchor',seed=seed,step=last,updates=step,val_mse=best,path=str(p.relative_to(r)),sha256=digest(p),params=sum(p.numel() for p in model.parameters()));results.append(row);pd.DataFrame(results).to_csv(r/'anchor_selection.csv',index=False)
        model.load_state_dict(state['state']);np.savez_compressed(r/'predictions'/f'{tag}_select.npz',prediction=pred(model,v));print(tag,best,'wall',round(time.monotonic()-start,2),flush=True);return row
    screen=fit(0,600,'anchor_screen');baseline=mse(np.repeat(v['x'][:,-1:],24,axis=1),v['y']);oldscreen=min(x['val_mse'] for x in json.loads((r/'family_selection.json').read_text())['candidates'] if x['variant'] in ['raw','instance','nonstationary'])
    selected=screen['val_mse']<min(baseline,oldscreen);models=old['models']
    if selected:
        for seed in cfg['seeds']:
            row=fit(seed,800,f'anchor_final_{seed}');models[f'final_clock_{seed}']=row
    lock=dict(models=models,selected_clock='last_anchor' if selected else old['selected_clock'],selection_only=True,old_test_used=False,phase_supervision_in_forecast_loss=False,
        amendment_selected=bool(selected),screen=screen,source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip())
    js(r/'training_lock.json',lock);js(r/'anchor_completion.json',dict(complete=True,wall_s=time.monotonic()-start,prior_train_s=elapsed,remaining_budget_start=budget,selected=bool(selected)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--config',required=True);a=p.parse_args();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.08);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;main(Path(a.run),json.loads(Path(a.config).read_text()))
