"""Read-only offline asset/GPU check. Does not reserve, download or launch inference."""
import json,hashlib,subprocess,argparse
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();p=args.model_dir/'pytorch_model.bin'
 r={'model_directory':str(args.model_dir),'weight_exists':p.exists(),'weight_bytes':p.stat().st_size if p.exists() else None,'verified':False,'eligible_gpu_indices':[]}
 if p.exists() and p.stat().st_size==12106053103:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  r['sha256']=h.hexdigest();r['verified']=h.hexdigest()=='319b73584bcdd552fec914980a406cc2ee4f84affaf34bd8ed87fbe14bbba28e'
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 r['gpu_snapshot']=raw.splitlines()
 mapping=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
 uuids={int(line.split(',')[0]):line.split(',')[1].strip() for line in mapping.splitlines()}
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
 counts={idx:sum(line.strip()==uuid for line in apps) for idx,uuid in uuids.items()};r['compute_process_count_by_gpu']=counts
 for line in raw.splitlines():
  idx,free,util=map(int,line.split(','))
  if free>=22000 and util<=5 and counts.get(idx,1)==0:r['eligible_gpu_indices'].append(idx)
 r['ready']=r['verified'] and bool(r['eligible_gpu_indices']);r['warning']='snapshot does not reserve GPU; recheck immediately before approved execution'
 with args.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps(r))
if __name__=='__main__':main()
