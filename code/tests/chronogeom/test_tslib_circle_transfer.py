"""Engineering fixtures only: no trained TS/LLM mechanism claims."""
import unittest
import numpy as np
import torch
from scripts.chronogeom.tslib_circle_transfer import fit_circle,evaluate_circle,capture_site,forward_with_patch

class CircleTransferTests(unittest.TestCase):
    def test_train_test_phase_regression_and_rank(self):
        p = np.arange(48) % 12
        def a(z):
            t = 2*np.pi*z/12
            return np.column_stack([3*np.cos(t),2*np.sin(t),np.zeros(len(z))])
        fit = fit_circle(a(p),p,12)
        result = evaluate_circle(fit,a(p+.25),p+.25)
        self.assertAlmostEqual(result["circle_augmented_r2"],1.,places=12)
        self.assertEqual(result["effective_rank"],2)
        self.assertAlmostEqual(result["axis_ratio"],2/3,places=10)
        np.testing.assert_allclose(fit.circle_basis.T@fit.circle_basis,np.eye(2),atol=1e-12)

    def test_degenerate_phase_and_nuisance_rejected(self):
        p = np.zeros(16);x = np.ones((16,3))
        with self.assertRaisesRegex(ValueError,"not identifiable"):
            fit_circle(x,p,12)
        p = np.arange(16)
        nuisance=np.column_stack([np.cos(2*np.pi*p/12),np.sin(2*np.pi*p/12)])
        with self.assertRaisesRegex(ValueError,"not identifiable"):
            fit_circle(x,p,12,nuisance=nuisance)

    def test_zero_effect_not_forced_to_rank_two(self):
        p = np.tile(np.arange(12),4)
        fit = fit_circle(np.ones((48,3)),p,12)
        self.assertEqual(fit.circle_basis.shape,(3,0))

    def test_noop_full_and_subspace_patch(self):
        model=torch.nn.Sequential(torch.nn.Identity(),torch.nn.Linear(3,1,bias=False))
        with torch.no_grad():model[1].weight.copy_(torch.tensor([[1.,2.,3.]]))
        model.train();x=torch.tensor([[[1.,2.,3.],[4.,5.,6.]]])
        y,act=capture_site(model,"0",(x,),position=1)
        self.assertTrue(model.training)
        torch.testing.assert_close(act,x[:,1])
        noop=forward_with_patch(model,"0",(x,),mode="noop",position=1)
        torch.testing.assert_close(noop,y,rtol=0,atol=0)
        donor=torch.tensor([[8.,9.,10.]])
        full=forward_with_patch(model,"0",(x,),donor,position=1,mode="full")
        torch.testing.assert_close(full[:,1],model[1](donor))
        torch.testing.assert_close(full[:,0],y[:,0])
        q=np.array([[1.],[0.],[0.]])
        patched=forward_with_patch(model,"0",(x,),donor,q,position=1)
        torch.testing.assert_close(patched[:,1],y[:,1]+4)
        ablate=forward_with_patch(model,"0",(x,),torch.zeros_like(donor),q,position=1)
        torch.testing.assert_close(ablate[:,1],y[:,1]-4)
        self.assertEqual(len(model[0]._forward_hooks),0)

    def test_invalid_patch_cleans_up_and_preserves_model_mode(self):
        model=torch.nn.Sequential(torch.nn.Identity())
        model.eval();x=torch.zeros(2,3)
        with self.assertRaises(ValueError):
            forward_with_patch(model,"0",(x,),torch.zeros(1,3),np.eye(3))
        self.assertFalse(model.training)
        self.assertEqual(len(model[0]._forward_hooks),0)
        with self.assertRaises(ValueError):
            forward_with_patch(model,"0",(x,),x,np.ones((3,1)))
        self.assertEqual(len(model[0]._forward_hooks),0)

    def test_official_tslib_forecaster_hook_dry_run(self):
        # Untrained official Transformer is only an API/control fixture.
        from types import SimpleNamespace
        from models.Transformer import Model
        cfg=SimpleNamespace(task_name="long_term_forecast",pred_len=4,enc_in=1,dec_in=1,
            c_out=1,d_model=8,embed="learned",freq="h",dropout=0.,factor=1,n_heads=2,
            d_ff=16,activation="gelu",e_layers=1,d_layers=1)
        torch.manual_seed(1729)
        model=Model(cfg).eval()
        values=torch.zeros(1,8,1)
        marks=torch.zeros(1,8,4);marks[:,:,0:2]=1;marks[:,:,2]=1
        corrupted=marks.clone();corrupted[:,:,2]=2
        site="dec_embedding.temporal_embedding.weekday_embed"
        clean_args=(values,marks,values,marks)
        corrupted_args=(values,marks,values,corrupted)
        clean,donor=capture_site(model,site,clean_args)
        corrupt,_=capture_site(model,site,corrupted_args)
        noop=forward_with_patch(model,site,corrupted_args,mode="noop")
        full=forward_with_patch(model,site,corrupted_args,donor,mode="full")
        self.assertEqual(tuple(clean.shape),(1,4,1))
        torch.testing.assert_close(noop,corrupt,rtol=0,atol=0)
        torch.testing.assert_close(full,clean,rtol=0,atol=0)

if __name__=="__main__":unittest.main()
