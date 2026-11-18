"""Scientific interface checks; these do not claim forecasting effectiveness."""
import unittest
from types import SimpleNamespace
import torch
from models.ChronoGeomTimeBase import Model, timestamp_feature_loss, timestamp_change_loss


def config(**changes):
    d=dict(seq_len=168,pred_len=24,enc_in=2,period_len=24,basis_num=4,
           individual=False,use_period_norm=True,use_orthogonal=False,
           chrono_use_calendar=True,chrono_timestamp_dim=4,chrono_timestamp_hidden=16)
    d.update(changes)
    return SimpleNamespace(**d)


def marks(batch=3,horizon=24):
    z=torch.zeros(batch,horizon,4)
    z[...,0]=1;z[...,1]=1;z[...,2]=2
    z[...,3]=torch.arange(horizon)%24
    return z


class TimeBaseTimestampTests(unittest.TestCase):
    def test_upstream_forecast_parity_before_calendar_learning(self):
        torch.manual_seed(19)
        for individual in [False,True]:
            for period_norm in [False,True]:
                m=Model(config(seq_len=170,pred_len=25,individual=individual,use_period_norm=period_norm))
                x=torch.randn(3,170,2)
                self.assertTrue(torch.equal(m.backbone(x),m(x,x_mark_dec=marks(horizon=25))))

    def test_future_observations_are_not_consumed(self):
        m=Model(config());x=torch.randn(3,168,2);calendar=marks()
        a=m(x,x_dec=torch.zeros(3,24,2),x_mark_dec=calendar)
        b=m(x,x_dec=torch.full((3,24,2),float('nan')),x_mark_dec=calendar)
        self.assertTrue(torch.equal(a,b))

    def test_distillation_trains_timestamp_branch_not_teacher(self):
        m=Model(config());x=torch.randn(3,168,2)
        p,aux=m.forward_with_aux(x,marks());t=torch.randn_like(aux['timestamp_code'],requires_grad=True)
        loss=timestamp_feature_loss(aux['timestamp_code'],t)+p.square().mean()
        loss.backward()
        self.assertIsNone(t.grad)
        self.assertTrue(torch.isfinite(m.timestamp_mlp[0].weight.grad).all())
        self.assertGreater(m.timestamp_mlp[0].weight.grad.abs().sum().item(),0)
        self.assertGreater(m.calendar_readout.weight.grad.abs().sum().item(),0)
        self.assertFalse(any('teacher' in k for k in m.state_dict()))
        with torch.no_grad():m.calendar_readout.weight.fill_(1)
        p1=m(x,x_mark_dec=marks());p2=m(x,x_mark_dec=marks()+torch.tensor([0,0,1,0]))
        self.assertGreater((p1-p2).abs().max().item(),0)

    def test_marks_and_target_shape_errors_are_explicit(self):
        with self.assertRaises(ValueError):Model(config(chrono_calendar_format='timeF'))
        m=Model(config());x=torch.randn(3,168,2)
        with self.assertRaises(ValueError):m(x)
        z=marks();z[...,2]=.3
        with self.assertRaises(ValueError):m(x,x_mark_dec=z)
        with self.assertRaises(ValueError):timestamp_feature_loss(torch.zeros(2,4),torch.zeros(2,22))

    def test_displacement_noop_and_orthogonal_aux(self):
        z=torch.randn(3,24,4,requires_grad=True);t=torch.randn_like(z,requires_grad=True)
        loss=timestamp_change_loss(z,z,t,t)
        self.assertEqual(loss.item(),0);loss.backward();self.assertIsNone(t.grad)
        m=Model(config(use_orthogonal=True));x=torch.randn(3,168,2)
        p,aux=m.forward_with_aux(x,marks())
        self.assertEqual(p.shape,(3,24,2));self.assertTrue(torch.isfinite(aux['orthogonal_loss']))

    def test_parameter_accounting_and_no_calendar_path(self):
        m=Model(config());n=sum(p.numel() for p in m.parameters())
        base=sum(p.numel() for p in m.backbone.parameters())
        self.assertEqual(base,37);self.assertEqual(n-base,121);self.assertEqual(n,158)
        b=Model(config(chrono_use_calendar=False));x=torch.randn(3,168,2)
        self.assertTrue(torch.equal(b(x),b.backbone(x)))


if __name__=='__main__':unittest.main()
