"""Optional Phase0-03 standalone capture; offline, 120s external cap, no training."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse,time,json,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output;out.mkdir(exist_ok=False);start=time.monotonic();cfg=json.loads(args.config.read_text());torch.set_num_threads(2);torch.manual_seed(42)
assert json.loads((out.parent/'analysis_retry1/quality.json').read_text())['execution_gate']
assert json.loads((out.parent/'execution/metadata.json').read_text())['main_seconds']+120<600
state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
available=[int(r.split(',')[0]) for r in state.splitlines() if float(r.split(',')[1])>=22000 and float(r.split(',')[2])<=5]
assert available,'No idle GPU';device='cuda:'+str(available[0]);(out/'preflight.json').write_text(json.dumps(dict(gpu_state=state,device=device,budget_seconds=120,seed=42,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
tok=AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True);tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(cfg['model_path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to(device).eval();records=[tok(str(a),add_special_tokens=False)['input_ids'] for a in range(361)];captured=[]
for i in range(0,361,8):
    assert time.monotonic()-start<110,'Time margin exhausted'
    batch=records[i:i+8];n=max(map(len,batch));ids=torch.full((len(batch),n),tok.pad_token_id,device=device,dtype=torch.long);mask=torch.zeros_like(ids);caps={};handles=[]
    for j,r in enumerate(batch):ids[j,-len(r):]=torch.tensor(r,device=device);mask[j,-len(r):]=1
    def hook(l):
        def capture(module,a,kw):
            h=kw.get('hidden_states',a[0] if a else None);caps[l]=h[:,-1].detach().float().cpu()
        return capture
    try:
        for l in [0,1]:handles.append(model.transformer.h[l].register_forward_pre_hook(hook(l),with_kwargs=True))
        with torch.inference_mode():model(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(1)-1).clamp(min=0),use_cache=False)
    finally:
        for h in handles:h.remove()
    captured.append(torch.stack([caps[l] for l in [0,1]],dim=1).numpy())
np.savez(out/'standalone_0_360.npz',values=np.arange(361),hidden=np.concatenate(captured),token_lengths=np.array(list(map(len,records))))
(out/'metadata.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,device=device,shape=list(np.concatenate(captured).shape),dtype='BF16 forward, FP32 cache',model_revision=cfg['model_revision'],peak_allocated=torch.cuda.max_memory_allocated(device)),indent=2))
print('COMPLETE',time.monotonic()-start,flush=True)
