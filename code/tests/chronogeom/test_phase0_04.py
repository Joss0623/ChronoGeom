import unittest
import numpy as np
from scripts.chronogeom.phase0_04_subspaces import orth,relation,decompose,spaces_for

class SubspaceAuditTests(unittest.TestCase):
 def test_overlap_and_increments_in_known_nested_spaces(self):
  e=np.eye(10);angle=.4;h=np.vstack([e[:6],np.cos(angle)*e[6]+np.sin(angle)*e[7]]);f=e[:7]
  s,uh,uf,_=decompose(h,f);self.assertEqual((len(s),len(uh),len(uf)),(6,1,1))
  self.assertAlmostEqual(relation(h,f)['projector_frobenius'],np.sqrt(2)*np.sin(angle))
  np.testing.assert_allclose(uh@s.T,0,atol=1e-12);np.testing.assert_allclose(uf@s.T,0,atol=1e-12)
  self.assertAlmostEqual(relation(uh,uf)['cosines'][0],np.cos(angle))
 def test_rank_and_collinearity_do_not_create_directions(self):
  e=np.eye(5);self.assertEqual(len(orth(np.vstack([e[0],e[0]+1e-9*e[1]]))),1);self.assertEqual(len(orth(np.zeros((3,5)))),0)
  self.assertEqual(len(decompose(e[:2],e[:2])[0]),2)
 def test_rotation_invariance_and_unequal_rank_distance(self):
  rng=np.random.default_rng(5);q=np.linalg.qr(rng.normal(size=(12,8)))[0].T;rot=np.linalg.qr(rng.normal(size=(7,7)))[0]
  self.assertAlmostEqual(relation(q[:7],rot@q[:7])['projector_frobenius'],0,places=6)
  self.assertAlmostEqual(relation(q,q[:7])['projector_frobenius'],1,places=6)
 def test_linear_increment_orthogonal_to_fourier_and_rescue_identity(self):
  rng=np.random.default_rng(9);a=np.arange(100);y=rng.normal(size=(100,24));sp,d,_=spaces_for(a,y)
  self.assertEqual(d['shared_rank'],6);self.assertEqual(len(sp['linear_increment']),1)
  np.testing.assert_allclose(sp['linear_increment']@sp['fourier7'].T,0,atol=1e-6)
  base=rng.normal(size=(3,24));ref=rng.normal(size=(3,24));u=sp['helix8'];dam=base+((ref-base)@u.T)@u;rescue=dam+((base-dam)@u.T)@u;np.testing.assert_allclose(base,rescue,atol=1e-12)
 def test_rescue_noop_keeps_damaged_logits_on_real_tiny_model(self):
  import torch
  from transformers import GPTJConfig,GPTJForCausalLM
  from scripts.chronogeom.phase0_02_patching import block_input_hook
  from scripts.chronogeom.phase0_04_run import control_exchange
  torch.set_num_threads(2);torch.manual_seed(13)
  m=GPTJForCausalLM(GPTJConfig(vocab_size=32,n_positions=16,n_embd=32,n_layer=2,n_head=4,rotary_dim=8)).eval();x=torch.tensor([[1,2,3,4]]);damage=np.zeros((1,32),dtype=np.float32)
  def run(exchange):
   h=m.transformer.h[0].register_forward_pre_hook(block_input_hook([1],replacement=None if exchange is None else torch.from_numpy(exchange)),with_kwargs=True)
   try:
    with torch.inference_mode():return m(x,use_cache=False).logits
   finally:h.remove()
  damaged=run(damage);noop=run(control_exchange('rescue','noop',damage));torch.testing.assert_close(noop,damaged,rtol=0,atol=0)
  self.assertGreater(float(torch.max(torch.abs(run(None)-damaged))),0)
if __name__=='__main__':unittest.main()
