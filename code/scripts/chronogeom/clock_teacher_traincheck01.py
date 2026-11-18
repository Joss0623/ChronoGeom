"""Post-hoc train-fit diagnostic on fixed first64 train rows; no parameter updates."""
from scripts.chronogeom.clock_teacher_lora01 import bootstrap,dump,sha,DAYS
import argparse,json,re,time
from pathlib import Path

def main(c,r,device):
 bootstrap()
 import torch
 from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
 from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
 from peft import LoraConfig,get_peft_model,set_peft_model_state_dict
 torch.set_num_threads(2);torch.manual_seed(c['seed']);torch.cuda.set_per_process_memory_fraction(c['memory_fraction'],device)
 rows=[x for x in json.loads((r/'examples.json').read_text()) if x['split']=='train'][:64];tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(c['model_path'])/'tokenizer.json'),eos_token='<|endoftext|>',pad_token='<|endoftext|>')
 t=time.monotonic();base=GPTJForCausalLM.from_pretrained(c['model_path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',low_cpu_mem_usage=True).to(device);loaded=time.monotonic()-t
 model=get_peft_model(base,LoraConfig(r=c['rank'],lora_alpha=c['alpha'],target_modules=['q_proj','v_proj'],layers_to_transform=c['lora_layers'],layers_pattern='h',bias='none',lora_dropout=0.,task_type='CAUSAL_LM'));records=json.loads((r/'test_lock.json').read_text())['models'];allrows=[];metrics=[];begin=time.monotonic()
 for arm in ['SFT','Circle']:
  rec=next(x for x in records if x['arm']==arm);assert sha(rec['path'])==rec['sha256'];set_peft_model_state_dict(model,torch.load(rec['path'],map_location='cpu',weights_only=True));model.eval();outrows=[]
  for i in range(0,64,c['eval_batch']):
   assert time.monotonic()-begin<180
   batch=rows[i:i+c['eval_batch']];seq=[tok.encode(x['text'],add_special_tokens=False) for x in batch];n=max(map(len,seq));ids=torch.full((len(batch),n),tok.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
   for j,s in enumerate(seq):ids[j,-len(s):]=torch.tensor(s,device=device);mask[j,-len(s):]=1
   with torch.no_grad():output=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=c['max_new_tokens'],do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   for row,tokens in zip(batch,output[:,n:].cpu().tolist()):
    text=tok.decode(tokens,skip_special_tokens=True);m=re.fullmatch(r'\s*('+'|'.join(DAYS)+r')\s+(\d{1,2}):00\s*[.!]?\s*',text);valid=m is not None and int(m.group(2))<24;correct=valid and DAYS.index(m.group(1))==row['answer_day'] and int(m.group(2))==row['answer_hour'];outrows.append(dict(arm=arm,id=row['id'],carry=row['carry'],expected=row['answer'].strip(),text=text,valid_format=bool(valid),correct=bool(correct),tokens=tokens))
  allrows+=outrows;metrics.append(dict(arm=arm,n=64,correct=sum(x['correct'] for x in outrows),valid_format=sum(x['valid_format'] for x in outrows),carry_n=sum(x['carry'] for x in outrows),carry_correct=sum(x['correct'] for x in outrows if x['carry'])))
 dump(r/'train_fit_predictions.json',allrows);dump(r/'train_fit_diagnostic.json',dict(status='Post-hoc fixed first64 training examples, descriptive, no training or checkpoint selection',metrics=metrics,load_seconds=loaded,inference_seconds=time.monotonic()-begin,peak_gib=torch.cuda.max_memory_allocated(device)/2**30));print('TRAIN-FIT DIAGNOSTIC',json.dumps(metrics),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:3');a=p.parse_args();c=json.loads(Path(a.config).read_text());main(c,Path(c['run']),a.device)
