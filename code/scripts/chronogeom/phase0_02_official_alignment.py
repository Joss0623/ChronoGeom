"""Execute fixed official fitting function with injected cached data, no model import."""
from pathlib import Path
import argparse,json,ast,warnings,time
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--official-root',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--random-audit',action='store_true');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 nb=json.loads((args.official_root/'experimentation/helix_fitting.ipynb').read_text());ns={'np':np,'torch':torch,'LinearRegression':LinearRegression,'warnings':warnings,'ConvergenceWarning':ConvergenceWarning}
 for cell in nb['cells']:
  if cell['cell_type']!='code':continue
  text=''.join(cell['source'])
  if not ('def create_features(' in text or 'def helix_fit(' in text):continue
  tree=ast.parse(text);fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['create_features','get_train_test_split','get_train_mask','helix_fit']]
  exec(compile(ast.Module(body=fns,type_ignores=[]),'official_fixed_cells','exec'),ns)
 d=torch.load(args.cache,weights_only=True,map_location='cpu');a,b,s=d['a'],d['b'],d['a+b'];y=d['hidden_states'][:,1].float().numpy();pca=PCA(n_components=100,svd_solver='full').fit(y);z=pca.transform(y)
 ns['get_pca_coords']=lambda *args,**kw:(a,b,s,z);ns['get_nums_hss']=lambda *args,**kw:(a,b,s,d['hidden_states']);ns['get_pca']=lambda *args,**kw:{'components':pca.components_,'mean':pca.mean_}
 results=[]
 if args.random_audit:
  class Spy(LinearRegression):
   first_fit=None;first_predict=None
   def fit(self,x,y,**kwargs):
    if Spy.first_fit is None:Spy.first_fit=x
    return super().fit(x,y,**kwargs)
   def predict(self,x):
    if Spy.first_predict is None:Spy.first_predict=x
    return super().predict(x)
  ns['LinearRegression']=Spy
  natural=ns['create_features']('helix',np.array([a]))
  for split in [False,True]:
   Spy.first_fit=None;Spy.first_predict=None;np.random.seed(42)
   _,pred,mse,mask=ns['helix_fit'](1,'helix_a',save=False,traintest=split,random=True)
   results.append(dict(traintest=split,training_rows=len(Spy.first_fit),prediction_rows=len(Spy.first_predict),fit_and_predict_design_share_memory=bool(np.shares_memory(Spy.first_fit,Spy.first_predict)),predict_design_matches_natural_numerals=bool(np.array_equal(Spy.first_predict,natural)),source_returned_full_mse=float(mse)))
  (args.output/'random_alignment.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results));return
 for kind in ['helix','circle','poly']:
  np.random.seed(42);nums,pred,mse,mask=ns['helix_fit'](1,kind+'_a',save=False,traintest=True)
  x=ns['create_features'](kind,np.array([a]));coef,_,rank,sv=np.linalg.lstsq(x[mask],z[mask],rcond=None);ours=x@coef@pca.components_+pca.mean_
  results.append({'fit_type':kind,'layer':1,'seed':42,'train_rows':int(mask.sum()),'test_rows':int((~mask).sum()),'official_returned_full_mse':float(mse),'official_heldout_full_mse':float(np.mean((pred[~mask]-y[~mask])**2)),'numpy_default_effective_rank':int(rank),'numpy_default_max_prediction_difference':float(np.abs(ours-pred).max()),'source_evaluation':'original sklearn function; all-cache sklearn PCA100;75% full-grid identity split; polynomial9 raw'})
 (args.output/'alignment.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))
if __name__=='__main__':main()
