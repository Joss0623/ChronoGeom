import unittest
import numpy as np
from data_provider.chronogeom_circle_pilot import make_pack,L,H,LABEL

class CirclePilotDataTests(unittest.TestCase):
    def test_forecast_interface_never_contains_future_values(self):
        p=make_pack("train",48)
        np.testing.assert_array_equal(p["xd"][:,-H:],0)
        np.testing.assert_array_equal(p["xd"][:,:LABEL],p["x"][:,-LABEL:])
        self.assertEqual(p["dm"].shape,(48,LABEL+H,4))
        np.testing.assert_array_equal(p["dm"][:,:LABEL],p["xm"][:,-LABEL:])
        np.testing.assert_array_equal(np.bincount(p["t0"]%24),np.full(24,2))

    def test_splits_and_physical_counterfactual_pair(self):
        a=make_pack("test",48);b=make_pack("test",48,6);f=make_pack("fit",48)
        self.assertEqual(len(set(a["ids"])&set(f["ids"])),0)
        np.testing.assert_array_equal(a["c"],b["c"])
        np.testing.assert_array_equal(a["b"],b["b"])
        np.testing.assert_allclose(a["y"]-a["mean"],b["y"]-b["mean"],atol=2e-7)
        np.testing.assert_array_equal(b["t"]-a["t"],6)
        np.testing.assert_array_equal(b["dm"][:,-H:,3],b["t"]%24)

if __name__=="__main__":unittest.main()
