import unittest
import numpy as np
from scripts.chronogeom.phase0_03_geometry import fit_spaces,make_change,subspace_error,identity_weights,interval
from scripts.chronogeom.representation_probe import basis
from scripts.chronogeom.phase0_03_run import bf16_match,additional_pairs

class SpecificityTests(unittest.TestCase):
    def test_rank_and_known_geometry_recovery(self):
        a=np.tile(np.arange(100),2);rng=np.random.default_rng(2);y=basis(a,'helix')@rng.normal(size=(10,16))
        sp,info=fit_spaces(a,y)
        self.assertEqual(len(sp['helix']),8);self.assertEqual(len(sp['fourier']),7);self.assertEqual(len(sp['linear']),1)
        np.testing.assert_allclose(sp['helix']@sp['helix'].T,np.eye(8),atol=2e-6)
        self.assertLess(subspace_error(y,info['helix']['mean'],sp['helix']),1e-10)
    def test_constant_hidden_does_not_invent_pca_rank(self):
        sp,_=fit_spaces(np.arange(20),np.ones((20,12)))
        self.assertEqual(len(sp['pca']),0);self.assertEqual(len(sp['helix']),0)
    def test_centered_ablation_only_removes_projected_coordinates(self):
        base=np.array([[3.,5.,7.]],np.float32);mean=np.array([1.,1.,1.]);u=np.array([[1.,0.,0.],[0.,1.,0.]])
        result,_=make_change(base,None,u,'ablate_mean',mean)
        np.testing.assert_array_equal(result,[[1.,1.,7.]])
        result2,_=make_change(result,None,u,'ablate_mean',mean);np.testing.assert_array_equal(result,result2)
    def test_bf16_energy_match_and_zero_direction(self):
        base=np.zeros((2,16),np.float32);ideal=np.ones_like(base)*.1;ideal[1]=0
        target=np.array([2.,1.],np.float32);matched,zero=bf16_match(base,ideal,target)
        import torch
        actual=torch.from_numpy(matched).bfloat16().float().numpy()
        self.assertLess(abs(np.linalg.norm(actual[0])-2.),.02);self.assertTrue(zero[1]);self.assertEqual(np.linalg.norm(actual[1]),0)
    def test_shared_identity_weights_respect_reversed_roles(self):
        pairs=[dict(clean_a=1,corrupt_a=2,b=3),dict(clean_a=2,corrupt_a=1,b=3)]
        w=identity_weights(pairs,200,4);np.testing.assert_array_equal(w[:,0],w[:,1]);np.testing.assert_array_equal(w,identity_weights(pairs,200,4))
        ci=interval(np.array([0.,0.]),w);self.assertEqual(ci['low'],0);self.assertEqual(ci['high'],0)
    def test_sampling_is_fixed_and_b_strata_are_balanced(self):
        a=additional_pairs(list(range(10)),32,44,'b_stratified');self.assertEqual(a,additional_pairs(list(range(10)),32,44,'b_stratified'))
        self.assertTrue(all(p['clean_a']!=p['corrupt_a'] for p in a));self.assertEqual(np.bincount([p['b']//25 for p in a]).tolist(),[8]*4)
if __name__=='__main__':unittest.main()
