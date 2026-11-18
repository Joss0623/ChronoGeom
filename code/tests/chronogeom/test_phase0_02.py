import unittest
import numpy as np
import torch
from scripts.chronogeom.phase0_02_patching import block_input_hook,replacement,generate_pairs
from scripts.chronogeom.phase0_02_fits import global_pca
from scripts.chronogeom.representation_probe import metrics

class Phase002Tests(unittest.TestCase):
 def test_projected_swap_preserves_complement(self):
  h=torch.tensor([[1.,2.,3.]]);clean=torch.tensor([[4.,5.,6.]]);u=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
  patched=replacement('helix',h,clean,u)
  torch.testing.assert_close(patched,torch.tensor([[4.,5.,3.]]));torch.testing.assert_close(replacement('helix',patched,clean,u),patched)
  torch.testing.assert_close(replacement('noop',h,clean),h)
 def test_random_gptj_hook_noop_and_full(self):
  from transformers import GPTJConfig,GPTJForCausalLM
  torch.set_num_threads(2);torch.manual_seed(2)
  m=GPTJForCausalLM(GPTJConfig(vocab_size=32,n_positions=16,n_embd=32,n_layer=2,n_head=4,rotary_dim=8)).eval()
  a=torch.tensor([[1,2,3,4]]);b=torch.tensor([[1,5,3,4]]);cap=[]
  handle=m.transformer.h[0].register_forward_pre_hook(block_input_hook([1],capture=cap),with_kwargs=True)
  with torch.no_grad():la=m(a,use_cache=False).logits
  handle.remove();hb=[];handle=m.transformer.h[0].register_forward_pre_hook(block_input_hook([1],capture=hb),with_kwargs=True)
  with torch.no_grad():lb=m(b,use_cache=False).logits
  handle.remove()
  handle=m.transformer.h[0].register_forward_pre_hook(block_input_hook([1],replacement=hb[0]),with_kwargs=True)
  with torch.no_grad():noop=m(b,use_cache=False).logits
  handle.remove();torch.testing.assert_close(noop,lb,rtol=0,atol=0)
  handle=m.transformer.h[0].register_forward_pre_hook(block_input_hook([1],replacement=cap[0]),with_kwargs=True)
  with torch.no_grad():full=m(b,use_cache=False).logits
  handle.remove();torch.testing.assert_close(full,la,rtol=0,atol=0)
  self.assertGreater(float(torch.max(torch.abs(la-lb))),0)
 def test_circle_group_indices_and_final_flush(self):
  from scripts.chronogeom.phase0_02_circle_audit import reconstruct_cluster
  dec=np.eye(2);out,ids=reconstruct_cluster([2.,3.,4.],[0,1,0],[2,2,5],{0,1},dec)
  self.assertEqual(ids,[2,5]);np.testing.assert_array_equal(out,[[2.,3.],[4.,0.]])
 def test_integral_float_ids_do_not_become_decimal_prompts(self):
  from scripts.chronogeom.phase0_02_patching import canonical_integer
  self.assertEqual(str(canonical_integer(3.0)),'3');self.assertEqual(canonical_integer(np.float64(0.)),0)
  for v in [3.5,float('nan'),True,'3']:
   with self.assertRaises(ValueError):canonical_integer(v)
 def test_token_layout_rejects_other_context_changes(self):
  from scripts.chronogeom.phase0_02_patching import same_token_layout
  c={'valid':True,'ids':[1,2,3,4],'pos':1};r={'valid':True,'ids':[1,9,3,4],'pos':1}
  self.assertTrue(same_token_layout(c,r))
  self.assertFalse(same_token_layout(c,dict(r,ids=[1,9,8,4])))
  self.assertFalse(same_token_layout(c,dict(r,ids=[1,9,3,4,5])))
 def test_pairs_fixed_and_distinct(self):
  x=generate_pairs(42,20,list(range(5)),3);self.assertEqual(x,generate_pairs(42,20,list(range(5)),3));self.assertTrue(all(p['clean_a']!=p['corrupt_a'] for p in x));self.assertEqual(len({(p['clean_a'],p['corrupt_a'],p['b']) for p in x}),20)
 def test_gram_pca_matches_svd(self):
  y=np.random.default_rng(1).normal(size=(12,8));mean,c=global_pca(y,4);_,_,v=np.linalg.svd(y-y.mean(0),full_matrices=False)
  np.testing.assert_allclose(c.T@c,v[:4].T@v[:4],atol=1e-12)
 def test_full_hidden_r2_is_variance_weighted(self):
  from sklearn.metrics import r2_score
  y=np.random.default_rng(1).normal(size=(20,5))*np.arange(1,6);p=y+.2
  self.assertAlmostEqual(metrics(y,p)['r2'],r2_score(y,p,multioutput='variance_weighted'),places=12)
if __name__=='__main__':unittest.main()
