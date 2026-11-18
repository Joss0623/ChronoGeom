import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
from data_provider.chronogeom_recovery import RecoveryData
from data_provider.chronogeom_circle_real import args_for
from exp.exp_chronogeom_recovery import make_model

class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.csv=cls.root/'data.csv'
        n=17420;pd.DataFrame(dict(date=pd.date_range('2016-07-01',periods=n,freq='h'),OT=np.sin(np.arange(n)/24)+np.arange(n)*.001)).to_csv(cls.csv,index=False)
        cls.cfg=dict(task_name='long_term_forecast',seq_len=96,label_len=48,pred_len=24,d_model=16,n_heads=2,d_ff=32,e_layers=1,d_layers=1,dropout=0,embed='learned',freq='h',enc_in=1,dec_in=1,c_out=1,factor=1,activation='gelu',moving_avg=25,p_hidden_dims=[16,16],p_hidden_layers=2)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_access_caps_and_old_test_quarantine(self):
        d=RecoveryData(self.csv);self.assertEqual(len(d.df),10368)
        with self.assertRaises(PermissionError):d.pack([11520])
        with self.assertRaises(PermissionError):d.pack([14400])
        with self.assertRaises(PermissionError):RecoveryData(self.csv,self.root,'qualification')
        (self.root/'training_lock.json').write_text('{}');q=RecoveryData(self.csv,self.root,'qualification');self.assertEqual(len(q.df),11520)
        (self.root/'qualification.json').write_text(json.dumps(dict(clock_qualified=True,linear_qualified=False)))
        b=RecoveryData(self.csv,self.root,'blind');self.assertEqual(len(b.df),15912)
        with self.assertRaises(PermissionError):b.pack([15912])
    def test_scaler_and_legal_future_input(self):
        d=RecoveryData(self.csv);p=d.split('select');np.testing.assert_array_equal(p['xd'][:,-24:],0)
        self.assertTrue((p['origins']+24<=10368).all());self.assertTrue((np.diff(p['origins'])>=24).all())
        np.testing.assert_allclose(d.mu,pd.read_csv(self.csv,nrows=8640).OT.mean())
    def test_no_new_trainable_parameters_and_shift_equivariance(self):
        torch.set_num_threads(2);torch.manual_seed(0);raw=make_model(self.cfg,'raw');torch.manual_seed(0);instance=make_model(self.cfg,'instance').eval()
        self.assertEqual(set(raw.state_dict()),set(instance.state_dict()));self.assertEqual(sum(p.numel() for p in raw.parameters()),sum(p.numel() for p in instance.parameters()))
        pack=RecoveryData(self.csv).pack([1000,1500]);a=args_for(pack,'cpu')
        with torch.no_grad():
            y=instance(*a);x,xm,xd,dm=a;xd2=xd.clone();xd2[:,:48]+=2
            y2=instance(x+2,xm,xd2,dm)
        torch.testing.assert_close(y2,y+2,atol=3e-6,rtol=3e-6)
    def test_all_existing_models_have_correct_forecast_shape(self):
        p=RecoveryData(self.csv).pack([1000,1500])
        for v in ['dlinear','raw','instance','no_calendar','nonstationary']:
            with torch.no_grad():y=make_model(self.cfg,v).eval()(*args_for(p,'cpu'))
            self.assertEqual(tuple(y.shape),(2,24,1));self.assertTrue(torch.isfinite(y).all())
if __name__=='__main__':unittest.main()
