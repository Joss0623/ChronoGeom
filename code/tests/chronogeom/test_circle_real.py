import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from data_provider.chronogeom_circle_real import RealData,args_for,nuisance,purged_folds
from scripts.chronogeom.circle_real_pilot import waveform,metrics

class RealCircleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.path=Path(cls.tmp.name)/'ETTh1.csv'
        t=np.arange(14500);pd.DataFrame(dict(date=pd.date_range('2016-07-01',periods=len(t),freq='h'),OT=t*.001+np.sin(2*np.pi*t/24))).to_csv(cls.path,index=False)
        cls.data=RealData(cls.path)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_train_only_scaler_and_temporal_borders(self):
        a=self.data;d=pd.read_csv(self.path);d.loc[8640:,'OT']+=1e6;p=Path(self.tmp.name)/'modified.csv';d.to_csv(p,index=False);b=RealData(p)
        self.assertEqual(a.mu,b.mu);self.assertEqual(a.sd,b.sd)
        for name,lo,hi in [('model_val',8640,9216),('fit',9216,10368),('val',10368,11520),('test',11520,14400)]:
            ids=a.origins[name];self.assertTrue((ids>=lo).all());self.assertTrue((ids+24<=hi).all())
    def test_clock_corruption_has_identical_observations_and_no_future_inputs(self):
        a=self.data;ids=a.origins['test'];x=a.pack(ids);bad=a.pack(ids,6)
        for key in ['x','xd','y','actual_t']:np.testing.assert_array_equal(x[key],bad[key])
        np.testing.assert_array_equal(x['xd'][:,-24:],0);self.assertFalse(np.array_equal(x['xm'],bad['xm']))
        self.assertEqual(nuisance(x).shape[0],len(ids)*24)
        self.assertTrue(np.all(np.sort(x['t']%24,axis=1)==np.arange(24)))
    def test_eval_targets_disjoint_and_donor_targets_remain_test(self):
        a=self.data;ids=a.origins['test'];self.assertTrue((np.diff(ids)>=24).all())
        self.assertGreaterEqual((ids-6).min(),11520);self.assertLessEqual((ids+6+24).max(),14400)
    def test_purged_cv_has_no_shared_raw_observations(self):
        a=self.data;ids=a.origins['fit']
        for train,test in purged_folds(ids):
            for tr in ids[train]:
                self.assertTrue(all(tr+24<=te-96 or tr-96>=te+24 for te in ids[test]))
    def test_waveform_decomposition_and_noop(self):
        h=np.arange(1,25);y=(2+.01*h+3*np.sin(2*np.pi*h/24))[None,:,None]
        coef,amp,ang,nh=waveform(y);np.testing.assert_allclose(coef[0],[2,.01,3,0],atol=1e-10);self.assertLess(nh[0],1e-20)
        m=metrics(y,y,y,y,y);self.assertEqual(m['prediction_change_mse'][0],0);self.assertLess(m['amplitude_error_observed'][0],1e-10)
if __name__=='__main__':unittest.main()
