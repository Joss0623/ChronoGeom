import unittest
import numpy as np
import torch
from exp.exp_chronogeom_timebind66 import Model,role_cal,energy
from exp.exp_chronogeom_timebind_soft67 import SoftModel

class SoftBindingTest(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(67)
  self.bank=energy(role_cal());self.old=Model(49,self.bank,joint=True).eval()
  self.x=torch.randn(3,56,49);self.valid=torch.ones(3,56,dtype=torch.bool);self.valid[:,-1]=False
  self.roles=torch.tensor([[0,8,4,12],[1,9,5,13],[2,10,6,14]])
  self.op=torch.tensor([0,1,2]);self.pr=torch.arange(56).repeat(3,1)%16
  self.truth=(self.roles[:,:,None]==self.pr[:,None,:]).float()*self.valid[:,None,:]
 def model(self,bank):
  m=SoftModel(49,self.bank,bank).eval()
  missing,unexpected=m.load_state_dict(self.old.state_dict(),strict=False)
  self.assertEqual(set(missing),{"alpha","relation"});self.assertFalse(unexpected)
  return m
 def runm(self,m):
  return m(self.x,self.valid,self.roles,self.op,self.truth,self.pr)
 def test_zero_initialization_exact_reference(self):
  with torch.no_grad():ref=self.old(self.x,self.valid,self.roles,self.op,self.truth)[0]
  for bank in [None,self.bank,self.bank[::-1].copy()]:
   m=self.model(bank)
   with torch.no_grad():z=self.runm(m)[0]
   torch.testing.assert_close(z,ref,atol=1e-6,rtol=1e-6)
 def test_complete_weight_inheritance(self):
  m=self.model(self.bank)
  for name,v in self.old.state_dict().items():self.assertTrue(torch.equal(v,m.state_dict()[name]),name)
 def test_alpha_has_gradient_at_zero(self):
  m=self.model(self.bank);z,aux,_=self.runm(m)
  (z.square().mean()+.1*aux).backward()
  self.assertTrue(torch.isfinite(m.alpha.grad));self.assertGreater(float(m.alpha.grad.abs()),1e-8)
  self.assertGreater(float(m.blocks[0].attn.in_proj_weight.grad.norm()),0.)
 def test_prior_valid_key_normalization(self):
  m=self.model(self.bank);p=m.prior(self.roles,self.pr,self.valid)
  for v in p.reshape(-1,56):
   self.assertAlmostEqual(float(v[:-1].mean()),0.,places=6)
   self.assertAlmostEqual(float((v[:-1]**2).mean()),1.,places=5)
 def test_padding_and_bias_effect(self):
  m=self.model(self.bank)
  with torch.no_grad():
   z0=self.runm(m)[0];m.alpha.fill_(.5);z,aux,w=self.runm(m)
  self.assertGreater(float((z-z0).abs().max()),1e-6)
  self.assertTrue(torch.equal(w[:,:,-1],torch.zeros_like(w[:,:,-1])))
 def test_plain_prior_inactive_and_parameter_budget(self):
  m=self.model(None)
  with torch.no_grad():
   z0=self.runm(m)[0];m.alpha.fill_(5);z=self.runm(m)[0]
  torch.testing.assert_close(z,z0,atol=0,rtol=0)
  self.assertEqual(sum(v.numel() for v in m.parameters()),sum(v.numel() for v in self.old.parameters())+1)
if __name__=="__main__":unittest.main()

