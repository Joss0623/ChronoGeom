import unittest
import numpy as np
from exp.exp_chronogeom_history_day_shape65 import condition_prompt,parse_answer,interpolate,profile_channels,history_fourier

class HistoryTeacherInputTests(unittest.TestCase):
 def test_profile_parser_preserves_all48_and_rejects_invalid(self):
  q,o=parse_answer('{"shape":'+str(np.arange(48).reshape(4,12).tolist())+'}')
  self.assertTrue(np.array_equal(q,np.arange(48)))
  with self.assertRaises(AssertionError):parse_answer('{"shape":[1,2,3]}')
  with self.assertRaises(AssertionError):parse_answer('{"shape":['+','.join(["NaN"]*48)+']}')
 def test_prompt_has_only_history_forecast_and_legal_calendar(self):
  c=dict(prompt="{INPUT}");x=np.arange(96,dtype=float);p=x[::-1]+3;days=[dict(date="2017-01-01",weekday=6,holiday=1)]*4
  a,m,s=condition_prompt(c,x,p,days,days,"past","future")
  b,_,_=condition_prompt(c,x*7+20,p*7+20,days,days,"past","future")
  import json
  aa=json.loads(a);bb=json.loads(b)
  self.assertTrue(np.allclose(aa["observed_history"],bb["observed_history"]))
  self.assertTrue(np.allclose(aa["timebase_forecast"],bb["timebase_forecast"]))
  self.assertNotIn("target",a);self.assertEqual(len(aa["observed_history"]),96)
 def test_interpolation_has_no_circular_day_boundary(self):
  q=np.arange(48)*2+.5;p=interpolate(q)
  self.assertTrue(np.allclose(p[1:95],np.arange(1,95)))
  self.assertEqual(p[0],.5);self.assertEqual(p[-1],94.5)
  k=profile_channels(p);self.assertEqual(k.shape,(96,3));self.assertAlmostEqual(float(k[:,0].mean()),0)
 def test_history_fourier_uses_history_and_same_forecast(self):
  t=np.arange(-96,96);cal=np.zeros((192,31))
  cal[:,0]=np.sin(t*2*np.pi/24);cal[:,1]=np.cos(t*2*np.pi/24)
  x=np.sin(t[:96]*2*np.pi/24)+.1*np.arange(96)/96;base=np.sin(t[96:]*2*np.pi/24)
  a=history_fourier(x,base,cal[:96],cal[96:],.1)
  b=history_fourier(x*5+3,base*5+3,cal[:96],cal[96:],.1)
  self.assertEqual(a.shape,(48,));self.assertTrue(np.isfinite(a).all());self.assertTrue(np.allclose(a,b))
  changed=history_fourier(x[::-1],base,cal[:96],cal[96:],.1)
  self.assertGreater(float(np.mean((changed-a)**2)),1e-5)
if __name__=="__main__":unittest.main()
