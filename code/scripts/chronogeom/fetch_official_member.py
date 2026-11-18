"""Fetch only pre-inventoried ZIP members via strict HTTP ranges.

The checked manifest provides ZIP offsets, uncompressed size and CRC32. No archive
is fully downloaded. Refuse ignored ranges, oversized files and existing outputs.
Use an approved reachable network; this script does not change proxy/TLS settings.
"""
from pathlib import Path
import argparse,hashlib,json,re,struct,urllib.request,zlib


def read_range(url,start,length):
    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{start+length-1}','User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(request,timeout=30) as response:
        if response.status!=206:raise RuntimeError('Server ignored Range; refusing full download')
        expected=f'bytes {start}-{start+length-1}/'
        if not response.headers.get('Content-Range','').startswith(expected):raise RuntimeError('Incorrect byte range')
        data=response.read(length+1)
        if len(data)!=length:raise RuntimeError('Incorrect response size')
    return data


def fetch(entry,output,max_bytes):
    if entry['size']>max_bytes or entry['compressed_size']>max_bytes:raise ValueError('Member exceeds budget')
    dest=output/Path(entry['name']).name
    if dest.exists() or dest.with_suffix(dest.suffix+'.partial').exists():raise FileExistsError(dest)
    header=struct.unpack('<4s5H3I2H',read_range(entry['source'],entry['offset'],30))
    if header[0]!=b'PK\x03\x04':raise ValueError('Invalid ZIP local header')
    if header[3]!=entry['method'] or header[2]&1:raise ValueError('Compression mismatch or encrypted member')
    position=entry['offset']+30+header[-2]+header[-1]
    remaining=entry['compressed_size'];decompress=zlib.decompressobj(-15)
    crc=0;size=0;sha=hashlib.sha256();partial=dest.with_suffix(dest.suffix+'.partial')
    with open(partial,'xb') as handle:
        while remaining:
            count=min(4*1024*1024,remaining);chunk=read_range(entry['source'],position,count)
            if entry['method']==8:raw=decompress.decompress(chunk,max_bytes-size+1)
            elif entry['method']==0:raw=chunk
            else:raise ValueError('Unsupported compression')
            size+=len(raw)
            if size>entry['size'] or size>max_bytes or decompress.unconsumed_tail:raise ValueError('Decompression exceeds budget')
            handle.write(raw);crc=zlib.crc32(raw,crc);sha.update(raw);position+=count;remaining-=count
    if entry['method']==8 and not decompress.eof:raise ValueError('Incomplete deflate stream')
    if size!=entry['size'] or crc&0xffffffff!=entry['crc32']:raise ValueError('ZIP integrity mismatch')
    if sha.hexdigest()!=entry['sha256']:raise ValueError('SHA256 mismatch with pinned member')
    partial.rename(dest)
    return {'path':str(dest),'sha256':sha.hexdigest(),'bytes':size}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-mib',type=int,default=128)
    args=parser.parse_args();entries=json.loads(args.manifest.read_text());args.output.mkdir(parents=True,exist_ok=True)
    for entry in entries:print(json.dumps(fetch(entry,args.output,args.max_mib*1024*1024)),flush=True)


if __name__=='__main__':main()
