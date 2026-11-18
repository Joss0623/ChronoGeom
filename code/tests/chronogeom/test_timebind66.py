import unittest
import numpy as np
import torch
from exp.exp_chronogeom_timebind66 import case_rows,definition,energy,role_cal,Model,scalar_label,texts

class TimeBindTest(unittest.TestCase):
 def test_composition_holdouts(self):
  _,h=definition({"composition_seed":66001})
  self.assertTrue(all(len(v)==7 for v in h.values()))
  self.assertFalse(h["mean_order"]&h["peak_order"])
 def test_order_and_change_semantics(self):
  _,h=definition({"composition_seed":66001})
  m=np.arange(16,dtype=float);m[8:]=m[:8]+np.arange(8)
  rows=case_rows(m,m,h,.1)
  # Change role7 by7 and role0 by0: first change larger.
  v=rows[(rows[:,0]==2)&(rows[:,1]==7)&(rows[:,3]==0)][0]
  self.assertEqual(v[5],2)
  self.assertEqual(scalar_label(.05,.1),1)
  self.assertEqual(scalar_label(-2.,.1),0)
 def test_energy_and_identity(self):
  z=energy(role_cal())
  self.assertEqual(z.shape,(16,32))
  np.testing.assert_allclose(np.mean(z*z,1),1,atol=1e-6)
  self.assertEqual(len(np.unique(z,axis=0)),16)
 def test_matched_capacity_and_initialization(self):
  z=energy(role_cal());states=[];counts=[]
  for joint in [False,True]:
   torch.manual_seed(2024);m=Model(49,z,joint)
   counts.append(sum(p.numel() for p in m.parameters()));states.append(dict(m.named_parameters()))
  self.assertEqual(counts[0],counts[1])
  for k in states[0]:self.assertTrue(torch.equal(states[0][k],states[1][k]))
 def test_mask_and_gradient(self):
  torch.manual_seed(2);m=Model(49,energy(role_cal()))
  x=torch.randn(2,56,49);v=torch.ones(2,56,dtype=torch.bool);v[:,55]=False
  rr=torch.tensor([[0,1,8,9],[1,2,9,10]]);op=torch.tensor([0,2])
  target=torch.zeros(2,4,56);target[:,:,:4]=1
  logits,aux,attn=m(x,v,rr,op,target)
  self.assertEqual(logits.shape,(2,3))
  self.assertTrue(torch.equal(attn[:,:,55],torch.zeros_like(attn[:,:,55])))
  (logits.square().mean()+aux).backward()
  self.assertTrue(torch.isfinite(m.patch[0].weight.grad).all())
  self.assertGreater(float(m.query.weight.grad.norm()),0.)
 def test_role_text_definitions(self):
  t=texts();self.assertEqual(len(t),16)
  self.assertIn("weekday",t[0]);self.assertIn("weekend",t[4]);self.assertIn("second",t[8])
if __name__=="__main__":unittest.main()

