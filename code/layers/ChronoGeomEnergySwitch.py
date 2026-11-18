"""Minimal energy switch prototype; not a trained forecasting result."""
import torch
from torch import nn

def centered(x):
    return x-x.mean(dim=1,keepdim=True)

def mix_by_cross_energy(reference,candidate,predicted_cross_energy):
    delta=centered(candidate-reference)
    energy=delta.square().mean(dim=(1,2),keepdim=True)
    estimate=predicted_cross_energy.reshape(-1,1,1)
    weight=torch.where(energy>1e-12,(estimate/energy.clamp_min(1e-12)).clamp(0.,1.),torch.zeros_like(energy))
    return reference+weight*delta,weight,energy

class EnergySwitch(nn.Module):
    """Five learned coefficients estimate cross energy from observable inputs.

    No target y or LLM call is accepted at inference. Zero initialization gives
    the reference prediction exactly. Cross-energy estimates require training;
    the clipping rule alone does not guarantee lower forecasting risk.
    """
    def __init__(self,period=24):
        super().__init__()
        self.period=period
        self.alignment=nn.Linear(4,1)
        nn.init.zeros_(self.alignment.weight)
        nn.init.zeros_(self.alignment.bias)
        self.register_buffer("feature_mean",torch.zeros(4))
        self.register_buffer("feature_scale",torch.ones(4))

    def observable_features(self,history,reference,candidate,history_weekend,future_weekend):
        assert history.ndim==reference.ndim==candidate.ndim==3
        assert reference.shape==candidate.shape
        H=reference.shape[1]
        delta=centered(candidate-reference)
        energy=delta.square().mean(dim=(1,2))
        tail=history[:,-self.period:]
        repeated=tail.repeat(1,(H+self.period-1)//self.period,1)[:,:H]
        proxy_cross=(centered(repeated-reference)*delta).mean(dim=(1,2))
        history_energy=centered(tail).square().mean(dim=(1,2))
        weekend_fraction=future_weekend.reshape(len(history),-1).to(reference.dtype).mean(dim=1)-history_weekend.reshape(len(history),-1).to(reference.dtype).mean(dim=1)
        return torch.stack([energy,proxy_cross,history_energy,weekend_fraction],dim=1)

    @torch.no_grad()
    def set_training_normalizer(self,features):
        self.feature_mean.copy_(features.mean(dim=0))
        self.feature_scale.copy_(features.std(dim=0,unbiased=False).clamp_min(1e-8))

    def forward(self,history,reference,candidate,history_weekend,future_weekend):
        f=self.observable_features(history,reference,candidate,history_weekend,future_weekend)
        prediction=self.alignment((f-self.feature_mean)/self.feature_scale)
        mixed,weight,energy=mix_by_cross_energy(reference,candidate,prediction)
        return mixed,dict(weight=weight,cross_energy_estimate=prediction,correction_energy=energy)
