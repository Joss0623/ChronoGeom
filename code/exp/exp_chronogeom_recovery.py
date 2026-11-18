"""Existing TSLib models and a standard history-only normalization control; no new encoder."""
from types import SimpleNamespace
import torch
from models.Transformer import Model as Transformer
from models.Nonstationary_Transformer import Model as Nonstationary
from models.DLinear import Model as DLinear

class InstanceTransformer(Transformer):
    # Same trainable architecture and state_dict as official Transformer.
    # This mirrors the history-only standardization used by official Nonstationary Transformer.
    def __init__(self,cfg,no_calendar=False,last_anchor=False):
        super().__init__(cfg);self.label_len=cfg.label_len;self.no_calendar=no_calendar;self.last_anchor=last_anchor
    def forward(self,x,xm,xd,dm,mask=None):
        mean=(x[:,-1:] if self.last_anchor else x.mean(1,keepdim=True)).detach();std=torch.sqrt(x.var(1,keepdim=True,unbiased=False)+1e-5).detach()
        xn=(x-mean)/std;dn=torch.cat([xn[:,-self.label_len:],torch.zeros_like(xd[:,-self.pred_len:])],dim=1)
        if self.no_calendar:xm=dm=None
        return super().forward(xn,xm,dn,dm,mask)*std+mean

def make_model(cfg,variant):
    c=SimpleNamespace(**cfg)
    if variant=='raw':return Transformer(c)
    if variant=='instance':return InstanceTransformer(c)
    if variant=='no_calendar':return InstanceTransformer(c,True)
    if variant=='nonstationary':return Nonstationary(c)
    if variant=='last_anchor':return InstanceTransformer(c,last_anchor=True)
    if variant=='dlinear':return DLinear(c)
    raise ValueError(variant)
