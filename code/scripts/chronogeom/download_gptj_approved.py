"""Explicitly approved fixed GPT-J asset fetch; range-bounded, SHA-verified."""
from pathlib import Path
import urllib.request,concurrent.futures,json,hashlib,time,argparse,os
SIZE=12106053103
SHA='319b73584bcdd552fec914980a406cc2ee4f84affaf34bd8ed87fbe14bbba28e'
REV='b71ae8bc86cac13154e03e92b5855203086b722e'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args();root=args.model_dir;root.mkdir(parents=True,exist_ok=True)
 if (root/'pytorch_model.bin').exists():raise FileExistsError('verified final asset already exists; do not overwrite')
 start=time.monotonic();parts=root/'download_parts';parts.mkdir(exist_ok=True);prefix=root/'pytorch_model.bin.partial';entries=[];offset=0
 if (parts/'part000').exists():
  n=(parts/'part000').stat().st_size;entries.append((0,n-1,parts/'part000'));offset=n
 elif prefix.exists():
  n=prefix.stat().st_size
  if not 0<n<SIZE:raise ValueError('unexpected prefix size')
  os.rename(prefix,parts/'part000');entries.append((0,n-1,parts/'part000'));offset=n
 for i,beg in enumerate(range(offset,SIZE,384*1024*1024),1):entries.append((beg,min(SIZE-1,beg+384*1024*1024-1),parts/f'part{i:03d}'))
 def fetch(entry):
  beg,end,path=entry
  if path.exists() and path.stat().st_size==end-beg+1:return
  for attempt in range(3):
   if time.monotonic()-start>3600:raise TimeoutError('one-hour acquisition limit')
   have=path.stat().st_size if path.exists() else 0;first=beg+have
   url=f'https://hf-mirror.com/EleutherAI/gpt-j-6b/resolve/{REV}/pytorch_model.bin?download=true&chronogeom_range={first}_{end}'
   try:
    req=urllib.request.Request(url,headers={'Range':f'bytes={first}-{end}'})
    with urllib.request.urlopen(req,timeout=45) as response:
     expected=f'bytes {first}-{end}/{SIZE}'
     if response.status!=206 or response.headers.get('Content-Range')!=expected:raise RuntimeError('server did not honor bounded Range')
     remaining=end-first+1
     with path.open('ab' if have else 'xb') as out:
      while remaining:
       if time.monotonic()-start>3600:raise TimeoutError('one-hour acquisition limit')
       chunk=response.read(min(4*1024*1024,remaining))
       if not chunk:raise EOFError('truncated member')
       out.write(chunk);remaining-=len(chunk)
    if path.stat().st_size!=end-beg+1:raise ValueError('part size mismatch')
    print('completed',path.name,path.stat().st_size,'elapsed',round(time.monotonic()-start),flush=True);return
   except Exception as e:
    print('retry',path.name,attempt,type(e).__name__,str(e),flush=True)
    if attempt==2:raise
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(fetch,entries))
 total=0;digest=hashlib.sha256()
 with prefix.open('xb') as out:
  for beg,end,path in entries:
   with path.open('rb') as src:
    for chunk in iter(lambda:src.read(4*1024*1024),b''):out.write(chunk);digest.update(chunk);total+=len(chunk)
 if total!=SIZE or digest.hexdigest()!=SHA:raise ValueError('final SHA/size mismatch; retained partial')
 os.rename(prefix,root/'pytorch_model.bin')
 (root/'acquisition.json').write_text(json.dumps({'upstream':'EleutherAI/gpt-j-6b','revision':REV,'sha256':SHA,'bytes':SIZE,'transport':'server hf-mirror.com plus Mac official fixed-revision HTTP ranges; final verified against official Hub LFS metadata','this_invocation_seconds':time.monotonic()-start,'resume_note':'download parts accumulated across invocations; this duration includes final assembly only when all parts already exist','verified_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'workers':args.workers,'approval':'user explicitly approved Phase0-02 GPT-J download and controlled experiment'},indent=2)+'\n')
 print('VERIFIED',SHA,total,flush=True)
if __name__=='__main__':main()
