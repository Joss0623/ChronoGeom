import unittest
import numpy as np
import torch
from exp.exp_chronogeom_distill import raw_table,normalize_table,shift_ids,Probe,params_for,td_term

class DistillationMinimalTests(unittest.TestCase):
 def test_calendar_shift_preserves_hour_weekday_and_wrap(self):
  a=np.array([[0,23,24,167]])
  np.testing.assert_array_equal(shift_ids(a,1),[[1,24,25,0]])
  np.testing.assert_array_equal(shift_ids(shift_ids(a,24),-24),a)
 def test_equal_total_encoding_energy_not_per_coordinate(self):
  for kind,q in [('raw',2),('fourier4',4),('fourier22',22),('calendar32',32)]:
   z,mu,s=normalize_table(raw_table(kind));self.assertEqual(z.shape,(168,q));self.assertAlmostEqual(float(np.mean(np.sum(z*z,axis=1))),1,places=6)
 def test_zero_head_is_persistence_and_time_noop(self):
  m=Probe(4,48);x=torch.randn(3,168);c=torch.arange(192).repeat(3,1)%168;z=torch.tensor(normalize_table(raw_table('fourier4'))[0]);p,h=m(x,c,z)
  self.assertTrue(torch.equal(p,x[:,-1:].expand(3,24)));self.assertEqual(sum(v.numel() for v in m.parameters()),params_for(4,48))
  self.assertEqual(float((m(x,shift_ids(c,0),z)[0]-p).abs().max()),0)
 def test_timedistill_self_target_zero_and_finite_gradient(self):
  torch.manual_seed(1);x=torch.randn(8,48,requires_grad=True);self.assertLess(float(td_term(x,x.detach()).abs()),1e-6)
  loss=td_term(x,torch.randn_like(x));loss.backward();self.assertTrue(torch.isfinite(x.grad).all())
if __name__=='__main__':unittest.main()
