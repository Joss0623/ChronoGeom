import unittest
import numpy as np
from scripts.chronogeom.circle_pilot02_math import native_audit,calibrate,score_many
from data_provider.chronogeom_circle_pilot02 import make_pack,test_pack

class Pilot02Tests(unittest.TestCase):
    def test_native_circle_without_label_fitting_and_rotation(self):
        h=np.tile(np.arange(24),8);t=2*np.pi*h/24
        x=np.column_stack([np.cos(t),np.sin(t),np.zeros((len(h),2))])
        a,_=native_audit(x,h,x,h,8,permutations=12,boot=2)
        q=np.linalg.qr(np.random.default_rng(4).normal(size=(4,4)))[0]
        b,_=native_audit(x@q,h,x@q,h,8,permutations=12,boot=2)
        self.assertTrue(a["native_circle_gate"]);self.assertAlmostEqual(a["harmonic1_fraction"],1.)
        self.assertAlmostEqual(a["cyclic_distance_rho"],b["cyclic_distance_rho"],places=2)
        self.assertLess(a["radius_cv"],1e-10)

    def test_unstructured_codes_and_degenerate_line_not_called_circle(self):
        h=np.tile(np.arange(24),4);codes=np.random.default_rng(2).normal(size=(24,8))
        m,_=native_audit(codes[h],h,codes[h],h,4,permutations=12,boot=2)
        self.assertFalse(m["native_circle_gate"])
        line=np.column_stack([np.sin(2*np.pi*h/24),np.zeros((len(h),7))])
        m,_=native_audit(line,h,line,h,4,permutations=0,boot=0)
        self.assertEqual(m["centroid_rank"],1);self.assertFalse(m["native_circle_gate"])

    def test_independent_interfaces_and_conditional_donor(self):
        a=make_pack("fit",48);b=make_pack("val",48);c=test_pack(48)
        self.assertEqual(len(set(a["ids"])&set(b["ids"])),0);self.assertEqual(len(set(a["ids"])&set(c["ids"])),0)
        alt=test_pack(48,0,True);np.testing.assert_array_equal(c["c"],alt["c"]);np.testing.assert_array_equal(c["mean"],alt["mean"])
        self.assertFalse(np.array_equal(c["x"],alt["x"]))
        np.testing.assert_array_equal(a["xd"][:,-24:],0)

    def test_matched_pool_rank_and_calibration_no_forecast_predictions(self):
        from copy import deepcopy
        h=np.tile(np.arange(24),8);rng=np.random.default_rng(3);x=rng.normal(size=(len(h),8))
        x[:,:2]+=np.column_stack([np.cos(2*np.pi*h/24),np.sin(2*np.pi*h/24)])*3
        cfg=dict(ridge_grid=[1e-3,.1],candidate_random=32,candidate_complements=4,mix_angles=[.3,.8],
            match_overlap_max=.95,match_max_angle_min_deg=15,phase_tolerance=.02,reconstruction_tolerance=.02)
        _,_,qs,_,diag,_=calibrate(x,h,np.empty((len(h),0)),x,h,np.empty((len(h),0)),8,cfg,0)
        for q in qs.values():np.testing.assert_allclose(q.T@q,np.eye(2),atol=1e-10)
        self.assertTrue(all(np.isfinite(d["validation_phase_r2"]) for d in diag))

    def test_forward_phase_complement_zero_covariance_is_not_no_information(self):
        from scripts.chronogeom.tslib_circle_transfer import fit_circle
        h=np.arange(24);y=np.column_stack([np.cos(2*np.pi*h/24),np.sin(2*np.pi*h/24)])
        x=np.random.default_rng(18).normal(size=(24,8))
        fit=fit_circle(x,h,24);q=fit.circle_basis;rest=(x-x.mean(0))@(np.eye(8)-q@q.T)
        np.testing.assert_allclose(y.T@rest,0,atol=1e-10)
        distances=np.linalg.norm(rest[:,None]-rest[None,:],axis=-1);np.fill_diagonal(distances,np.inf)
        self.assertGreater(distances.min(),1e-3)

if __name__=="__main__":unittest.main()
