import unittest
import numpy as np
import torch
from scripts.chronogeom.phase1_representations import encode,fit_interface,transform,rank,split_ids,T_ENCODINGS
from data_provider.chronogeom_synthetic import generate,task_spec,datasets
from exp.exp_chronogeom_synthetic import CovariateReadout
class Phase1Tests(unittest.TestCase):
 def test_aliasing_and_linear_coordinate_distinction(self):
  for k in ['sincos','fourier','ttw_phase']:np.testing.assert_allclose(encode([0,100],k)[0],encode([0,100],k)[1],atol=1e-12)
  self.assertGreater(np.linalg.norm(encode([0,100],'helix')[0]-encode([0,100],'helix')[1]),1)
 def test_isometric_lift_preserves_standardized_distance_and_rank(self):
  x=encode(np.arange(80),'fourier');s=fit_interface(x,9);a=(x-s['mean'])/s['std'];b=transform(x,s)
  np.testing.assert_allclose(np.linalg.norm(a[1:]-a[:-1],axis=1),np.linalg.norm(b[1:]-b[:-1],axis=1),atol=2e-6);self.assertEqual(rank(x),7)
 def test_identity_splits_and_independent_trajectory_ids(self):
  ids=split_ids();self.assertEqual(len(set(ids['train'])&set(ids['interpolation'])),0);spec=task_spec('development','periodic');a=generate(spec,'train');b=generate(spec,'val');self.assertFalse(set(v['instance_id'] for v in a[2])&set(v['instance_id'] for v in b[2]));self.assertEqual(a[1].shape,(128,24))
 def test_backbones_have_uniform_parameters_and_future_target_not_input(self):
  torch.set_num_threads(2);spec=task_spec('development','mixed');all_counts={}
  for backbone in ['DLinear','TSMixer']:
   all_counts[backbone]=[]
   for e in T_ENCODINGS:
    ds,_,_,_,_=datasets(spec,e,'val');m=CovariateReadout(backbone).eval();all_counts[backbone].append(sum(p.numel() for p in m.parameters()));test=ds[1]
    with torch.no_grad():a=m(test.x[:2],test.x_mark[:2],test.y_mark[:2]);test.y[:2]+=1000;b=m(test.x[:2],test.x_mark[:2],test.y_mark[:2])
    torch.testing.assert_close(a,b,atol=0,rtol=0);self.assertEqual(a.shape,(2,8,1))
   self.assertEqual(len(set(all_counts[backbone])),1)
 def test_raw_observations_identical_between_time_conditions(self):
  spec=task_spec('heldout','regime');d1,*_=datasets(spec,'none','test');d2,*_=datasets(spec,'helix','test');torch.testing.assert_close(d1[0].x,d2[0].x);torch.testing.assert_close(d1[1].y,d2[1].y)
if __name__=='__main__':unittest.main()
