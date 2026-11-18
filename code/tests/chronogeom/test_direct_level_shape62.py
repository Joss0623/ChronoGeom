"""Level/shape independence is a scientific constraint of pilot62."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from exp.exp_chronogeom_downstream import CalendarTimeBase
from exp.exp_chronogeom_level_shape_direct62 import DirectStudent, center

class DirectLevelShapeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        source=CalendarTimeBase(dict(seq_len=96,pred_len=96),2024,np.zeros((168,168),np.float32))
        self.model=DirectStudent(2024,source.state_dict(),np.zeros(8),np.ones(8))
        torch.manual_seed(62001)
        self.x=torch.randn(3,96,1)
        self.calendar=torch.randn(3,96,31)
        self.ids=(torch.arange(96)[None].repeat(3,1)%72).long()
        self.target=torch.randn(3,96,1)
    def activate_reader(self):
        with torch.no_grad():
            self.model.reader[2].weight.normal_(std=.1)
            self.model.reader[2].bias.fill_(.7)
    def test_initial_identity_and_loss_decomposition(self):
        numerical=self.model.numerical_prediction(self.x)
        p=self.model(self.x,self.calendar,self.ids)
        self.assertLess(float((p-numerical).abs().max()),2e-6)
        self.activate_reader()
        level,shape=self.model.components(self.x,self.calendar,self.ids)
        a=((level+shape-self.target)**2).mean()
        b=((level-self.target.mean(1,keepdim=True))**2).mean()+((shape-center(self.target))**2).mean()
        self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=2e-6))
    def test_calendar_changes_shape_not_level(self):
        self.activate_reader()
        p=self.model(self.x,self.calendar,self.ids)
        q=self.model(self.x,-self.calendar,(self.ids+11)%72)
        level=self.model.numerical_prediction(self.x).mean(1,keepdim=True)
        self.assertTrue(torch.allclose(p.mean(1,keepdim=True),level,atol=1e-6,rtol=1e-6))
        self.assertTrue(torch.allclose(q.mean(1,keepdim=True),level,atol=1e-6,rtol=1e-6))
        self.assertGreater(float((p-q).square().mean()),1e-8)
    def test_shape_supervision_cannot_update_level(self):
        self.activate_reader()
        self.model.zero_grad(set_to_none=True)
        level,shape=self.model.components(self.x,self.calendar,self.ids)
        ((shape-center(self.target))**2).mean().backward()
        params=list(self.model.body.parameters())+[self.model.affine]
        self.assertTrue(all(p.grad is None or not torch.count_nonzero(p.grad) for p in params))
        self.assertGreater(float(self.model.codes.weight.grad.abs().sum()),0)
        self.assertGreater(float(self.model.reader[2].weight.grad.abs().sum()),0)
    def test_level_supervision_cannot_update_calendar_or_shape(self):
        self.activate_reader()
        self.model.zero_grad(set_to_none=True)
        level,shape=self.model.components(self.x,self.calendar,self.ids)
        ((level-(self.target.mean(1,keepdim=True)+1))**2).mean().backward()
        params=list(self.model.reader.parameters())+list(self.model.codes.parameters())
        self.assertTrue(all(p.grad is None or not torch.count_nonzero(p.grad) for p in params))
        self.assertGreater(float(self.model.affine.grad.abs().sum()),0)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),963)

if __name__=="__main__":
    unittest.main()
