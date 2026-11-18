"""Bounded temporal-response distillation pilot in the pinned THUML TSLib checkout.
No downloads, no LLM training, no modifications to public TSLib models/trainers.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse,copy,csv,hashlib,json,subprocess,time,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
DAYS=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
TEMPLATES=[
 'An event is scheduled for {day} at {hour}:00.',
 'The meeting takes place on {day} at {hour}:00.',
 'The calendar shows {day}, {hour}:00, for this appointment.',
 'A reminder says: {day} at {hour}:00.',
 'The planned departure is on {day} at {hour}:00.',
 'The next activity begins on {day} at {hour}:00.',
 'We noted the time as {day} at {hour}:00.',
 'The appointment time is {day} at {hour}:00.'
]
def dump(p,x):
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def cfgload(p):return json.loads(Path(p).read_text())
def tensor(x,device):return torch.as_tensor(x,dtype=torch.float32,device=device)
def save_csv(path,rows):
    if not rows:return
    with open(path,'w') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
def r2(y,p):return float(1-np.sum((y-p)**2)/max(np.sum((y-y.mean(0))**2),1e-12))
def phase_design(days):return np.column_stack([np.cos(2*np.pi*days/7),np.sin(2*np.pi*days/7)])
def raw_table(kind):
    d=np.repeat(np.arange(7),24);h=np.tile(np.arange(24),7)
    if kind=='raw':return np.column_stack([h/23-.5,d/6-.5]).astype('float32')
    if kind=='fourier4':return np.column_stack([np.cos(2*np.pi*h/24),np.sin(2*np.pi*h/24),np.cos(2*np.pi*d/7),np.sin(2*np.pi*d/7)]).astype('float32')
    if kind=='fourier22':
        cols=[]
        for x,p,ks in [(h,24,range(1,7)),(d,7,range(1,4))]:
            for k in ks:cols.extend([np.cos(2*np.pi*k*x/p),np.sin(2*np.pi*k*x/p)])
        day=raw_table('fourier4')[:,:2];week=raw_table('fourier4')[:,2:]
        cols.extend([day[:,i]*week[:,j] for i in range(2) for j in range(2)])
        return np.column_stack(cols).astype('float32')
    if kind=='calendar32':return np.column_stack([np.eye(24)[h],np.eye(7)[d],d>=5]).astype('float32')
    raise ValueError(kind)
def normalize_table(a):
    a=a.astype('float64');mu=a.mean(0);scale=np.sqrt(np.mean(np.sum((a-mu)**2,axis=1)));return ((a-mu)/max(scale,1e-12)).astype('float32'),mu,scale

def extract(cfg,run,device):
    out=run/'llm';out.mkdir(exist_ok=False);start=time.monotonic()
    # Pure-text GPT-J does not use image/audio backends. Disable optional discovery
    # only in this process; never edit installed packages or model parameters.
    from transformers import utils as hf_utils
    from transformers.utils import import_utils as hf_imports
    for name in ['is_torchvision_available','is_torchvision_v2_available','is_vision_available','is_librosa_available','is_torchaudio_available','is_sklearn_available']:
        if hasattr(hf_imports,name):setattr(hf_imports,name,lambda:False)
        if hasattr(hf_utils,name):setattr(hf_utils,name,lambda:False)
    from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
    modelpath=Path(cfg['model_path']);tok=PreTrainedTokenizerFast(tokenizer_file=str(modelpath/'tokenizer.json'),eos_token='<|endoftext|>',pad_token='<|endoftext|>');assert tok.eos_token_id==50256
    if device.startswith('cuda'):
        free,total=torch.cuda.mem_get_info(device)
        if free<cfg['llm_min_free_gib']*2**30:raise RuntimeError('Insufficient free memory; no other process will be killed')
        torch.cuda.set_per_process_memory_fraction(.60,device)
    torch.set_num_threads(2)
    print('Loading existing frozen GPT-J',device,flush=True)
    model=GPTJForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.float16,attn_implementation='eager',low_cpu_mem_usage=True).to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    load_seconds=time.monotonic()-start;inference_start=time.monotonic()
    print('Loaded GPT-J; extracting span means',flush=True)
    records=[]
    for template in range(8):
        for day in range(7):
            for hour in [0,6,12,18]:records.append(dict(template=template,day=day,hour=hour,purpose='geometry'))
    for template in [0,1]:
        for day in range(7):
            for hour in range(24):records.append(dict(template=template,day=day,hour=hour,purpose='lookup'))
    assert len(records)==560
    layers=cfg['layers'];caps={l:[] for l in layers};lastcaps={l:[] for l in layers}
    for a in range(0,len(records),cfg['llm_batch']):
        if time.monotonic()-inference_start>cfg['llm_seconds']:raise TimeoutError('LLM extraction wall cap reached; partial files retained')
        batch=records[a:a+cfg['llm_batch']];texts=[TEMPLATES[r['template']].format(day=DAYS[r['day']],hour=f"{r['hour']:02d}") for r in batch]
        enc=tok(texts,return_tensors='pt',padding=True,return_offsets_mapping=True,add_special_tokens=False)
        offsets=enc.pop('offset_mapping');spans=[]
        for i,(r,text) in enumerate(zip(batch,texts)):
            begin=text.index(DAYS[r['day']]);hour=f"{r['hour']:02d}:00";end=text.index(hour)+len(hour)
            inds=[j for j,(s,e) in enumerate(offsets[i].tolist()) if e>begin and s<end]
            r.update(text=text,token_ids=enc['input_ids'][i][enc['attention_mask'][i].bool()].tolist(),span_positions=inds,span_text=tok.decode(enc['input_ids'][i,inds].tolist()))
            if not inds:raise ValueError('Empty target time span')
            spans.append(inds)
        handles=[];got={}
        for l in layers:
            def hook(m,args,result,l=l):
                h=result[0] if isinstance(result,tuple) else result
                got[l]=(torch.stack([h[i,ix].mean(0) for i,ix in enumerate(spans)]).float().cpu().numpy(),torch.stack([h[i,ix[-1]] for i,ix in enumerate(spans)]).float().cpu().numpy())
            handles.append(model.transformer.h[l].register_forward_hook(hook))
        try:
            with torch.inference_mode():model.transformer(**{k:v.to(device) for k,v in enc.items()},use_cache=False)
        finally:
            for h in handles:h.remove()
        for l in layers:caps[l].append(got[l][0]);lastcaps[l].append(got[l][1])
        if a%64==0:print('captured',min(a+cfg['llm_batch'],len(records)),len(records),flush=True)
    allh=np.stack([np.concatenate(caps[l]) for l in layers],axis=1);alllast=np.stack([np.concatenate(lastcaps[l]) for l in layers],axis=1)
    np.savez_compressed(out/'raw_spans.npz',hidden=allh,last_token=alllast)
    dump(out/'prompts.json',records);days=np.array([r['day'] for r in records]);templates=np.array([r['template'] for r in records]);geom=np.array([r['purpose']=='geometry' for r in records]);fit=geom&(templates<4);val=geom&(templates>=4)&(templates<6);test=geom&(templates>=6)
    diagnostics=[];spaces=[]
    for li,l in enumerate(layers):
        h=allh[:,li].astype('float64');mu=h[fit].mean(0);cent=np.stack([h[fit&(days==d)].mean(0) for d in range(7)]);cent-=cent.mean(0)
        _,sv,v=np.linalg.svd(cent,full_matrices=False);uc=v[:2]
        resid=h[fit]-mu;resid-=resid@uc.T@uc;_,sr,vr=np.linalg.svd(resid,full_matrices=False)
        u=np.concatenate([uc,vr[:2]]);assert np.max(np.abs(u@u.T-np.eye(4)))<1e-6
        z=(h-mu)@u.T;scale=np.sqrt(np.mean(z[fit]**2));z/=max(scale,1e-12)
        X=np.column_stack([z[fit],np.ones(sum(fit))]);coef=np.linalg.lstsq(X,phase_design(days[fit]),rcond=1e-8)[0]
        vp=np.column_stack([z[val],np.ones(sum(val))])@coef;tp=np.column_stack([z[test],np.ones(sum(test))])@coef
        energies=np.sum(np.abs(np.fft.rfft(cent,axis=0)[1:])**2,axis=1);share=float(energies[0]/max(energies.sum(),1e-12))
        shuffled=[];rng=np.random.default_rng(413)
        for _ in range(128):
            q=cent[rng.permutation(7)];e=np.sum(abs(np.fft.rfft(q,axis=0)[1:])**2,axis=1);shuffled.append(float(e[0]/max(e.sum(),1e-12)))
        distances=np.linalg.norm(cent[:,None]-cent[None,:],axis=-1);adj=np.mean([distances[d,(d+1)%7] for d in range(7)])
        native=cent@uc.T;angles=np.unwrap(np.arctan2(native[:,1],native[:,0]));delta=np.diff(np.r_[angles,angles[0]+np.sign(np.median(np.diff(angles)))*2*np.pi]);radius=np.linalg.norm(native,axis=1)
        rec=dict(layer=l,hook=f'transformer.h[{l}] output before next block',phase_val_r2=r2(phase_design(days[val]),vp),phase_test_r2=r2(phase_design(days[test]),tp),native_first_harmonic_share=share,label_permutation_mean=float(np.mean(shuffled)),label_permutation_p=(1+sum(s>=share for s in shuffled))/129,adjacent_distance=float(adj),all_pair_distance=float(distances[np.triu_indices(7,1)].mean()),radius_cv=float(radius.std()/max(radius.mean(),1e-12)),centroid_rank=int(np.sum(sv>sv[0]*1e-8)),effective_rank=int(np.linalg.matrix_rank(z[fit],tol=1e-6)),angular_step_sign_consistency=float(max(np.mean(delta>0),np.mean(delta<0))))
        diagnostics.append(rec);spaces.append((mu,u,scale,z));np.savez_compressed(out/f'geometry_layer{l}.npz',centroids=cent,distances=distances,projection=u,center=mu,scale=scale)
    # Layer choice sees only validation prompt geometry, not audit templates or forecasts.
    li=int(np.argmax([r['phase_val_r2'] for r in diagnostics]));mu,u,scale,z=spaces[li]
    table=np.zeros((168,4))
    for d in range(7):
        for h in range(24):
            ix=[i for i,r in enumerate(records) if r['purpose']=='lookup' and r['day']==d and r['hour']==h]
            table[d*24+h]=z[ix].mean(0)
    table,tm,ts=normalize_table(table)
    np.savez_compressed(out/'lookup.npz',llm4=table,selected_layer=layers[li])
    save_csv(out/'geometry.csv',diagnostics)
    dump(out/'completion.json',dict(source_commit=commit(),model_path=str(modelpath),model_bytes=(modelpath/'pytorch_model.bin').stat().st_size,expected_model_sha256=cfg['model_sha256'],dtype=str(next(model.parameters()).dtype),layers=layers,selected_layer=layers[li],selection='max validation-template phase R2; no forecast labels',prompts=len(records),seconds=time.monotonic()-start,load_seconds=load_seconds,inference_seconds=time.monotonic()-inference_start,device=device,peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device),torch=torch.__version__))
    print('LLM cache complete',layers[li],round(time.monotonic()-start,1),flush=True)

def audit(cfg,run):
    import pandas as pd
    p=Path(cfg['data_path']);n=0;h=hashlib.sha256()
    with open(p,'rb') as f:
        for line in f:h.update(line);n+=1
    n-=1
    head=pd.read_csv(p,nrows=1344);datecol=cfg['date_column'];dates=pd.to_datetime(head[datecol]);diff=dates.diff().dropna()
    if not bool((diff==pd.Timedelta(hours=1)).all()):raise ValueError('Nonhourly calendar; cannot assume calendar structure')
    cols=[c for c in head.columns if c!=datecol and pd.to_numeric(head[c],errors='coerce').notna().all() and float(head[c].std())>1e-8][:16]
    if len(cols)!=16:raise ValueError('Need 16 prespecified complete nonconstant columns')
    train_end=8*7*24;val_start=int(n*.7);val_end=int(n*.8);pilot_end=int(n*.9)
    if n-pilot_end<4*7*24 or val_start<train_end+168:raise ValueError('Insufficient sealed confirmation interval')
    a=dict(path=str(p.resolve()),sha256=h.hexdigest(),n_rows=n,columns=cols,first_timestamp=str(dates.iloc[0]),train_end=train_end,val_start=val_start,val_end=val_end,pilot_end=pilot_end,confirmation=[pilot_end,n],date_column=datecol,calendar_claim='CSV-declared civil dates; hourly cadence checked; geography/timezone/DST not independently established',source_commit=commit())
    dump(run/'data_manifest.json',a);print(json.dumps(a,ensure_ascii=False),flush=True)

def load_data(run,stage):
    import pandas as pd
    a=cfgload(run/'data_manifest.json');end=a['val_end'] if stage=='fit' else a['pilot_end']
    if stage!='fit' and not (run/'final_lock.json').exists():raise RuntimeError('Pilot data access requires frozen final model hashes')
    df=pd.read_csv(a['path'],nrows=end,usecols=[a['date_column']]+a['columns'])
    dates=pd.to_datetime(df[a['date_column']]);assert bool((dates.diff().dropna()==pd.Timedelta(hours=1)).all())
    values=df[a['columns']].to_numpy(dtype='float32');assert np.isfinite(values).all();mean=values[:a['train_end']].mean(0);sd=values[:a['train_end']].std(0);assert np.all(sd>1e-8)
    values=(values-mean)/sd;ids=(dates.dt.dayofweek.to_numpy()*24+dates.dt.hour.to_numpy()).astype('int64')
    return a,values,ids

def pack(values,ids,lo,hi,L=168,H=24,stride=24):
    origins=np.arange(max(lo,L),hi-H+1,stride,dtype='int64');nch=values.shape[1]
    xs=[];ys=[];cs=[]
    for o in origins:
        xs.append(values[o-L:o].T);ys.append(values[o:o+H].T);cs.append(np.tile(ids[o-L:o+H],(nch,1)))
    return dict(x=np.concatenate(xs),y=np.concatenate(ys),c=np.concatenate(cs),origin=np.repeat(origins,nch),channel=np.tile(np.arange(nch),len(origins)))
def get_tables(run):
    tables={k:normalize_table(raw_table(k))[0] for k in ['raw','fourier4','fourier22','calendar32']}
    if (run/'llm/lookup.npz').exists():tables['llm4']=np.load(run/'llm/lookup.npz')['llm4'].astype('float32')
    return tables

def shift_ids(c,delta):return (c+delta)%168

class Probe(nn.Module):
    def __init__(self,q,width=48,L=168,H=24):
        super().__init__();self.L=L;self.H=H;self.width=width
        self.fc1=nn.Linear(L+(L+H)*q,width);self.fc2=nn.Linear(width,width);self.out=nn.Linear(width,H)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x,c,table):
        anchor=x[:,-1:];v=torch.cat([x-anchor,table[c].flatten(1)],1)
        h=F.gelu(self.fc2(F.gelu(self.fc1(v))));return self.out(h)+anchor,h

def params_for(q,w,L=168,H=24):return w*w+(L+(L+H)*q+H+2)*w+H
def matched_width(q,target):return min(range(4,97),key=lambda w:abs(params_for(q,w)-target))
def predict(model,p,table,device):
    preds=[];model.eval()
    with torch.inference_mode():
        for b in range(0,len(p['x']),512):preds.append(model(tensor(p['x'][b:b+512],device),torch.as_tensor(p['c'][b:b+512],device=device),table)[0].cpu().numpy())
    return np.concatenate(preds)
def td_term(student,teacher):
    # Exact ingredients of the pinned TimeDistill objectives; no feature-axis physical claim.
    mse=[];s=student;t=teacher
    for i in range(3):
        mse.append(F.mse_loss(s,t))
        if i<2:s=F.avg_pool1d(s[:,None,:],2,2).squeeze(1);t=F.avg_pool1d(t[:,None,:],2,2).squeeze(1)
    sf=torch.fft.rfft(student,dim=-1).abs()[:,1:];tf=torch.fft.rfft(teacher,dim=-1).abs()[:,1:]
    kl=F.kl_div(F.log_softmax(sf/.5,dim=-1),F.softmax(tf/.5,dim=-1),reduction='mean')
    return torch.stack(mse).mean()+kl

def train_one(cfg,run,label,kind,width,seed,train,val,device,mode='supervised',teacher=None,teacher_kind=None,lam=0):
    torch.manual_seed(seed);torch.set_num_threads(2);rng=np.random.default_rng(seed);shift_rng=np.random.default_rng(seed+10000)
    tables=get_tables(run);table=tensor(tables[kind],device);model=Probe(table.shape[1],width).to(device)
    reg=nn.Linear(width,width).to(device) if mode=='timedistill' else None
    if reg is not None:nn.init.eye_(reg.weight);nn.init.zeros_(reg.bias)
    optimized=list(model.parameters())+([] if reg is None else list(reg.parameters()))
    opt=torch.optim.Adam(optimized,lr=cfg['lr']);xt=tensor(train['x'],device);yt=tensor(train['y'],device);ct=torch.as_tensor(train['c'],device=device)
    if teacher is not None:
        teacher.eval()
        for v in teacher.parameters():v.requires_grad_(False)
        tt=tensor(tables[teacher_kind],device)
    start=time.monotonic();curve=[];best=np.inf;beststate=None;beststep=0;bestreg=None
    for step in range(cfg['steps']+1):
        if step%100==0:
            vp=predict(model,val,table,device);vl=float(np.mean((vp-val['y'])**2));curve.append(dict(step=step,val_mse=vl))
            if vl<best:
                best=vl;beststep=step;beststate=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
                bestreg=None if reg is None else copy.deepcopy({k:v.cpu() for k,v in reg.state_dict().items()})
        if step==cfg['steps']:break
        if time.monotonic()-start>cfg['fit_seconds']:raise TimeoutError('Per-fit pilot cap exceeded')
        model.train();ix=rng.integers(len(xt),size=cfg['batch']);x=xt[ix];y=yt[ix];c=ct[ix]
        p,h=model(x,c,table);sup=F.mse_loss(p,y);extra=torch.zeros((),device=device)
        if mode=='response':
            delta=int(shift_rng.choice(cfg['shifts']));cp=shift_ids(c,delta);pp,_=model(x,cp,table)
            with torch.no_grad():tp,_=teacher(x,c,tt);tpp,_=teacher(x,cp,tt)
            extra=F.mse_loss(pp-p,tpp-tp)
        elif mode=='kd':
            with torch.no_grad():tp,_=teacher(x,c,tt)
            extra=F.mse_loss(p,tp)
        elif mode=='timedistill':
            with torch.no_grad():tp,th=teacher(x,c,tt)
            assert th.shape==h.shape
            extra=td_term(p,tp)+td_term(h,reg(th))
        loss=sup+lam*extra;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(optimized,1.);opt.step()
    model.load_state_dict(beststate)
    record=dict(label=label,kind=kind,width=width,seed=seed,mode=mode,lambda_=lam,teacher_kind=teacher_kind,val_mse=best,best_step=beststep,updates=cfg['steps'],parameters=sum(v.numel() for v in model.parameters()),training_only_parameters=0 if reg is None else sum(v.numel() for v in reg.parameters()),seconds=time.monotonic()-start,device=device)
    path=run/'states'/f'{label}.pt';torch.save(dict(state=beststate,training_only_regressor=bestreg,record=record),path)
    dump(run/'curves'/f'{label}.json',curve);print(json.dumps(record),flush=True);return model,record

def fit_seed(cfg,run,seed,device):
    seedout=run/f'seed{seed}';seedout.mkdir(exist_ok=False);(run/'states').mkdir(exist_ok=True);(run/'curves').mkdir(exist_ok=True)
    if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.08,device)
    a,values,ids=load_data(run,'fit');tr=pack(values,ids,0,a['train_end'],stride=1);va=pack(values,ids,a['val_start'],a['val_end']);start=time.monotonic();records=[];final=[];models={}
    target=params_for(4,48)
    for kind in ['llm4','fourier4','fourier22','calendar32']:
        q=get_tables(run)[kind].shape[1];width=48 if q==4 else matched_width(q,target)
        label=f'teacher_{kind}_s{seed}';path=run/'states'/f'{label}.pt'
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=True);rec=saved['record']
            assert rec['kind']==kind and rec['seed']==seed and rec['width']==width and rec['updates']==cfg['steps']
            model=Probe(q,width).to(device);model.load_state_dict(saved['state'])
        else:model,rec=train_one(cfg,run,label,kind,width,seed,tr,va,device)
        models[kind]=model;records.append(rec)
    start-=sum(r['seconds'] for r in records)
    strong=min(['fourier22','calendar32'],key=lambda k:next(r['val_mse'] for r in records if r['kind']==k));llm=models['llm4']
    dump(seedout/'teacher_selection.json',dict(strong=strong,records=records,train_examples=len(tr['x']),train_origins=len(np.unique(tr['origin'])),val_examples=len(va['x']),val_origins=len(np.unique(va['origin'])),persistence_val=float(np.mean((va['x'][:,-1:]-va['y'])**2)),source_commit=commit()))
    arms=[('S0','raw','supervised',None),('S2','raw','response',strong),('S3','raw','response','llm4'),('S4','raw','kd','llm4'),('S5','raw','timedistill','llm4'),('S8','fourier4','supervised',None),('S10','llm4','supervised',None),('S9','raw','response','fourier4')]
    for arm,kind,mode,tk in arms:
        width=48 if kind=='raw' else matched_width(get_tables(run)[kind].shape[1],params_for(2,48))
        candidates=[]
        for lam in ([0] if mode=='supervised' else cfg['lambdas']):
            if time.monotonic()-start>cfg['seed_seconds']:raise TimeoutError('Per-seed total pilot cap exceeded')
            label=f'{arm}_s{seed}_l{lam}'
            model,rec=train_one(cfg,run,label,kind,width,seed,tr,va,device,mode,models.get(tk),tk,lam);records.append(rec);candidates.append(rec)
        chosen=min(candidates,key=lambda r:r['val_mse']);final.append(dict(arm=arm,**chosen))
    save_csv(seedout/'all_fits.csv',records);save_csv(seedout/'selected.csv',final);dump(seedout/'completion.json',dict(source_commit=commit(),seconds=time.monotonic()-start,final=final,strong=strong,peak_allocated=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None))

def lock(run):
    files=[]
    for seed in [0,1,2]:
        c=cfgload(run/f'seed{seed}/completion.json')
        for r in c['final']:
            p=run/'states'/f"{r['label']}.pt";files.append(dict(seed=seed,arm=r['arm'],path=str(p),sha256=sha(p),record=r))
        for kind in ['llm4','fourier4','fourier22','calendar32']:
            p=run/'states'/f'teacher_{kind}_s{seed}.pt';files.append(dict(seed=seed,arm='teacher_'+kind,path=str(p),sha256=sha(p)))
    dump(run/'final_lock.json',dict(source_commit=commit(),models=files,test_access='pilot only; confirmation remains sealed',time=time.strftime('%Y-%m-%dT%H:%M:%S%z')))

def diagnose_calendar_tables(run):
    rows=[]
    for kind,z in get_tables(run).items():
        z=z.astype('float64');s=np.linalg.svd(z-z.mean(0),compute_uv=False)
        c=(np.arange(168)[:,None]+np.arange(-168,24)[None,:])%168
        flat=z[c].reshape(168,-1);f=np.linalg.svd(flat-flat.mean(0),compute_uv=False)
        energy=f*f;prob=energy/max(energy.sum(),1e-30);prob=prob[prob>1e-15]
        rows.append(dict(kind=kind,dimension=z.shape[1],table_rank=int(np.sum(s>s[0]*1e-6)),calendar_sequence_rank_1e6=int(np.sum(f>f[0]*1e-6)),calendar_sequence_rank_1e4=int(np.sum(f>f[0]*1e-4)),calendar_sequence_entropy_rank=float(np.exp(-np.sum(prob*np.log(prob)))),total_vector_rms=float(np.sqrt(np.mean(np.sum(z*z,axis=1)))),flattened_width=flat.shape[1]))
    save_csv(run/'encoding_rank_energy.csv',rows)

def response_analysis(run,p,tables,states,preds,device):
    rows=[]
    def restore(item):
        saved=torch.load(item['path'],map_location='cpu',weights_only=True);r=saved['record']
        model=Probe(tables[r['kind']].shape[1],r['width']).to(device);model.load_state_dict(saved['state'])
        return model,tensor(tables[r['kind']],device)
    for seed in [0,1,2]:
        ti=next(i for i in states if i['seed']==seed and i['arm']=='teacher_llm4')
        teacher,tt=restore(ti);base=preds[f'teacher_llm4_s{seed}'];target={}
        for delta in [0,-24,-1,1,24]:
            shifted=dict(p,c=shift_ids(p['c'],delta));target[delta]=predict(teacher,shifted,tt,device)-base
        for item in [i for i in states if i['seed']==seed and i['arm'].startswith('S')]:
            model,st=restore(item);base=preds[item['arm']+f'_s{seed}']
            for delta,td in target.items():
                shifted=dict(p,c=shift_ids(p['c'],delta));sd=predict(model,shifted,st,device)-base
                den=np.linalg.norm(td)*np.linalg.norm(sd)
                rows.append(dict(arm=item['arm'],seed=seed,shift=delta,response_mse_to_llm=float(np.mean((sd-td)**2)),teacher_response_rms=float(np.sqrt(np.mean(td**2))),student_response_rms=float(np.sqrt(np.mean(sd**2))),response_cosine=float(np.sum(sd*td)/den) if den>1e-12 else None,max_noop=float(np.max(np.abs(sd))) if delta==0 else None,n_examples=len(sd)))
        print('response audit seed',seed,flush=True)
    save_csv(run/'conditional_responses.csv',rows)
    assert all(r['max_noop']==0 for r in rows if r['shift']==0),'No-op response failed; report before claiming transfer'

def evaluate(cfg,run,device):
    import pandas as pd
    a,values,ids=load_data(run,'pilot');diagnose_calendar_tables(run);p=pack(values,ids,a['val_end'],a['pilot_end']);tables=get_tables(run);results=[];errors={};fidelity=[];states=cfgload(run/'final_lock.json')['models']
    preds={'persistence':np.repeat(p['x'][:,-1:],24,axis=1),'seasonal24':p['x'][:,-24:],'seasonal168':p['x'][:,:24]}
    for item in states:
        path=Path(item['path']);assert sha(path)==item['sha256'];state=torch.load(path,map_location='cpu',weights_only=True);r=state['record'];model=Probe(tables[r['kind']].shape[1],r['width']).to(device);model.load_state_dict(state['state']);preds[item['arm']+f"_s{item['seed']}"]=predict(model,p,tensor(tables[r['kind']],device),device)
    # Compute on the raw standardized 24-hour target, preserving both periodic and nonperiodic parts.
    t=np.arange(24);B=np.column_stack([np.ones(24),t-t.mean(),np.cos(2*np.pi*t/24),np.sin(2*np.pi*t/24)]);Bi=np.linalg.pinv(B);cy=p['y']@Bi.T
    for label,pred in preds.items():
        e=np.mean((pred-p['y'])**2,axis=1);errors[label]=e;cp=pred@Bi.T;amp=np.linalg.norm(cp[:,2:],axis=1);trueamp=np.linalg.norm(cy[:,2:],axis=1);valid=(trueamp>=.1)&(amp>=.1)
        phase=np.abs(np.angle(np.exp(1j*(np.arctan2(cp[:,3],cp[:,2])-np.arctan2(cy[:,3],cy[:,2])))))
        resid=pred-cp@B.T;residy=p['y']-cy@B.T
        results.append(dict(label=label,mse=float(e.mean()),mae=float(np.abs(pred-p['y']).mean()),n_examples=len(e),n_days=len(np.unique(p['origin'])),phase_defined_n=int(valid.sum()),truth_phase_eligible_n=int((trueamp>=.1).sum()),predicted_phase_small_n=int((amp<.1).sum()),phase_error_deg=float(np.degrees(phase[valid]).mean()) if valid.any() else None,amplitude_abs_error=float(np.abs(amp-trueamp).mean()),offset_error=float(np.abs(cp[:,0]-cy[:,0]).mean()),trend_error=float(np.abs(cp[:,1]-cy[:,1]).mean()),nonperiodic_mse=float(np.mean((resid-residy)**2))))
    np.savez_compressed(run/'pilot_predictions.npz',**preds,truth=p['y'],origin=p['origin'],channel=p['channel'])
    save_csv(run/'pilot_summary.csv',results)
    contrasts=[];blocks=(p['origin']-a['val_end'])//168;unique=np.unique(blocks);rng=np.random.default_rng(203)
    for base in ['S2','S9','S8','S10','S4','S5','S0']:
        ds=np.stack([errors[f'S3_s{s}']-errors[f'{base}_s{s}'] for s in [0,1,2]])
        block_sums=np.array([ds[:,blocks==b].sum() for b in unique]);block_counts=np.array([ds[:,blocks==b].size for b in unique]);ix=rng.integers(len(unique),size=(2000,len(unique)));boots=block_sums[ix].sum(1)/block_counts[ix].sum(1)
        contrasts.append(dict(comparison='S3-'+base,mean_mse_difference=float(ds.mean()),seed_differences=ds.mean(1).tolist(),low95=float(np.quantile(boots,.025)),high95=float(np.quantile(boots,.975)),low975=float(np.quantile(boots,.0125)),high975=float(np.quantile(boots,.9875)),n_week_blocks=len(unique),bootstrap='paired 7-day blocks, all channels/seeds retained jointly; conditional on three fitted seeds'))
    dump(run/'paired_contrasts.json',contrasts)
    response_analysis(run,p,tables,states,preds,device)
    dump(run/'pilot_completion.json',dict(source_commit=commit(),n_examples=len(p['x']),n_days=len(np.unique(p['origin'])),confirmation_unparsed=[a['pilot_end'],a['n_rows']],date_first=str(pd.to_datetime(cfgload(run/'data_manifest.json')['first_timestamp'])+pd.Timedelta(hours=int(p['origin'].min()))),date_last=str(pd.to_datetime(a['first_timestamp'])+pd.Timedelta(hours=int(p['origin'].max()+23))),resource='no teacher/LLM at student inference'))
    print(json.dumps(results,ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['audit','extract','fit','lock','pilot']);ap.add_argument('--config',required=True);ap.add_argument('--run',required=True);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--device',default='cpu');a=ap.parse_args();cfg=cfgload(a.config);run=Path(a.run);run.mkdir(exist_ok=True,parents=True);torch.set_num_threads(2)
    if a.stage=='audit':audit(cfg,run)
    elif a.stage=='extract':extract(cfg,run,a.device)
    elif a.stage=='fit':fit_seed(cfg,run,a.seed,a.device)
    elif a.stage=='lock':lock(run)
    else:evaluate(cfg,run,a.device)
if __name__=='__main__':main()
