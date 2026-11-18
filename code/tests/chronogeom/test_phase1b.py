import unittest
import numpy as np
from data_provider.chronogeom_identifiability import *
from exp.exp_chronogeom_identifiability import fit_readout,predict_readout
class IdentifiabilityTests(unittest.TestCase):
 def test_oracle_risk_and_redundancy(self):
  for task in TASKS:
   p=generate(task,'test',32);gap=np.mean(((p['full_oracle']-p['history_oracle'])/p['unit_value'])**2)
   self.assertAlmostEqual(gap,.5 if task=='B' else 0,places=10)
  p=generate('B','test',32,'new_system');np.testing.assert_allclose(p['full_oracle'],p['history_oracle'],atol=1e-12)
 def test_no_future_value_in_design(self):
  p=generate('B','train',64);s=interface(p,'sincos');a=design(p,'sincos',s,'rms')[0];p['y']+=1000;b=design(p,'sincos',s,'rms')[0];np.testing.assert_array_equal(a,b)
 def test_energy_and_units(self):
  tr=generate('B','train',64)
  for kind in ENCODINGS:
   s=interface(tr,kind);_,_,e=design(tr,kind,s,'rms');self.assertAlmostEqual(np.sqrt(np.mean(np.sum(e**2,axis=-1))),0 if kind=='none' else 1,places=12)
   p=generate('B','test',16);u=generate('B','test',16,'units');np.testing.assert_allclose(design(p,kind,s,'rms')[0],design(u,kind,s,'rms')[0],atol=1e-6)
 def test_phase_misleading_changes_clock_not_signal(self):
  # Directly verify history oracle remains exact when supplied clock is wrong.
  p=generate('D','test',32,'phase');np.testing.assert_allclose(p['history_oracle'],p['full_oracle'],atol=1e-10)
  clock_prediction=np.sin(2*np.pi*p['future_time']/p['period'][:,None]);signal=p['full_oracle'];self.assertGreater(np.mean((clock_prediction-signal)**2),.1)
 def test_linear_decoder_can_read_sufficient_encoding(self):
  for task,kind in [('A','none'),('B','sincos'),('C','none')]:
   p=generate(task,'train',128);p['y']=p['full_oracle'].copy();fit=fit_readout(p,kind,'rms','linear',-1);test=generate(task,'test',64);result=predict_readout(fit,test);self.assertLess(result['clean_loss'].mean(),1e-10);self.assertEqual(fit['parameters'],832)
if __name__=='__main__':unittest.main()
