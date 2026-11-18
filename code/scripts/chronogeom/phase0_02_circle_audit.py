"""Audit fixed cluster membership and source token alignment, no SAE training."""
from pathlib import Path
import argparse,ast,pickletools,pickle,json,hashlib
import numpy as np

def reconstruct_cluster(values,features,tokens,cluster,decoder):
 values,features,tokens=map(np.asarray,(values,features,tokens))
 if not(len(values)==len(features)==len(tokens)):raise ValueError('sparse columns not aligned')
 if np.any(np.diff(tokens)<0):raise ValueError('token groups must be sorted')
 keep=np.isin(features,list(cluster));ids=np.unique(tokens[keep]);out=np.zeros((len(ids),decoder.shape[1]))
 if len(ids):np.add.at(out,np.searchsorted(ids,tokens[keep]),values[keep,None]*decoder[features[keep]])
 return out,ids.tolist()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 path=args.root/'sae_multid_feature_discovery/gpt-2_layer_7_clusters_spectral_n1000.pkl';raw=path.read_bytes()
 ops=list(pickletools.genops(raw));assert not any(op.name in ['GLOBAL','STACK_GLOBAL','REDUCE','BUILD','INST','OBJ','NEWOBJ','NEWOBJ_EX'] for op,_,_ in ops)
 clusters=pickle.loads(raw);assert isinstance(clusters,dict) and all(type(k)==int and isinstance(v,list) and all(type(i)==int for i in v) for k,v in clusters.items())
 source=args.root/'sae_multid_feature_discovery/gpt2_days_months_years.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_cluster_activations');ns={'np':np,'tqdm':lambda x,**kw:x};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
 dec=np.zeros((1,768));dec[0,0]=1;data={'sparse_sae_values':np.array([2.,3.]),'sparse_sae_indices':np.array([0,0]),'all_token_indices':np.array([2,5])}
 old,oldids=ns['get_cluster_activations'](data,{0},dec);new,newids=reconstruct_cluster(data['sparse_sae_values'],data['sparse_sae_indices'],data['all_token_indices'],{0},dec)
 result={'cluster_count':len(clusters),'sha256':hashlib.sha256(raw).hexdigest(),'weekday':{'cluster_id':138,'features':clusters[138]},'month':{'cluster_id':251,'features':clusters[251]},'source_function_sha256':hashlib.sha256(ast.get_source_segment(source.read_text(),fn).encode()).hexdigest(),'synthetic_alignment_test':{'input_token_ids':[2,5],'official_returned_token_ids':[int(i) for i in oldids],'official_first_coordinate':old[:,0].tolist(),'corrected_token_ids':newids,'corrected_first_coordinate':new[:,0].tolist()},'evidence':'source/schema audit and synthetic software test, not actual LLM circular representation'}
 (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
