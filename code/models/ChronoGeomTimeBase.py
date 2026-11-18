"""TimeBase with one common timestamp branch, inside the pinned TSLib checkout.

The exact upstream numerical backbone is in layers/ChronoGeomTimeBaseOfficial.py.
This is TimeBase+Calendar, not an unchanged TimeBase architecture. All timestamp
teacher controls must use this same student, initialization and training budget.
No teacher is stored in the model or needed for inference.
"""
import torch
from torch import nn
from torch.nn import functional as F
from layers.ChronoGeomTimeBaseOfficial import Model as OfficialTimeBase


def timestamp_feature_loss(student, teacher):
    """MSE to a frozen, training-calendar-normalized teacher feature target."""
    if student.shape != teacher.shape:
        raise ValueError('Student and teacher timestamp features must have identical shapes')
    if not torch.isfinite(teacher).all():
        raise ValueError('Teacher timestamp features must be finite')
    return F.mse_loss(student, teacher.detach().to(student))


def timestamp_change_loss(student_before, student_after, teacher_before, teacher_after):
    """Optional displacement matching, not a physical counterfactual label."""
    if not (student_before.shape == student_after.shape == teacher_before.shape == teacher_after.shape):
        raise ValueError('All displacement feature shapes must match')
    return timestamp_feature_loss(student_after - student_before,
                                  teacher_after.detach() - teacher_before.detach())


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        if self.task_name not in ('long_term_forecast', 'short_term_forecast'):
            raise ValueError('ChronoGeomTimeBase currently supports forecasting only')
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.use_calendar = getattr(configs, 'chrono_use_calendar', True)
        self.calendar_format = getattr(configs, 'chrono_calendar_format', 'discrete')
        if self.use_calendar and self.calendar_format != 'discrete':
            raise ValueError('Use explicit TSLib discrete [month,day,weekday,hour] marks; timeF is not supported')
        self.backbone = OfficialTimeBase(configs)
        self.timestamp_dim = getattr(configs, 'chrono_timestamp_dim', 4)
        if self.use_calendar:
            hidden = getattr(configs, 'chrono_timestamp_hidden', 16)
            self.timestamp_mlp = nn.Sequential(nn.Linear(2, hidden), nn.GELU(),
                                               nn.Linear(hidden, self.timestamp_dim))
            # One shared residual across independent, training-standardized channels.
            # Zero initialization makes the initial forecast exactly the backbone.
            self.calendar_readout = nn.Linear(self.timestamp_dim, 1)
            nn.init.zeros_(self.calendar_readout.weight)
            nn.init.zeros_(self.calendar_readout.bias)

    def timestamp_inputs(self, marks, batch_size):
        if marks is None or marks.ndim != 3 or marks.shape[0] != batch_size:
            raise ValueError('Known future timestamp marks [B,T,4+] are required')
        if marks.shape[1] < self.pred_len or marks.shape[2] < 4:
            raise ValueError('Marks must contain the full forecast horizon and discrete calendar fields')
        future = marks[:, -self.pred_len:, :]
        weekday, hour = future[..., 2], future[..., 3]
        values = torch.stack((weekday, hour), -1)
        if not torch.isfinite(values).all() or not torch.equal(values, values.round()):
            raise ValueError('Weekday and hour fields must be finite integer-valued metadata')
        if not (((weekday >= 0) & (weekday <= 6)).all()
                and ((hour >= 0) & (hour <= 23)).all()):
            raise ValueError('Expected weekday0..6 and hour0..23; normalized timeF is invalid')
        return torch.stack((weekday / 3. - 1., hour / 11.5 - 1.), -1)

    def forward_with_aux(self, x_enc, x_mark_dec=None):
        if x_enc.ndim != 3 or x_enc.shape[-1] != self.enc_in:
            raise ValueError('Expected historical observations [B,L,enc_in]')
        base = self.backbone(x_enc)
        if isinstance(base, tuple):
            base, orthogonal_loss = base
        else:
            orthogonal_loss = x_enc.new_zeros(())
        code = None
        if self.use_calendar:
            inputs = self.timestamp_inputs(x_mark_dec, len(x_enc)).to(x_enc)
            code = self.timestamp_mlp(inputs)
            base = base + self.calendar_readout(code)
        return base, {'timestamp_code': code, 'orthogonal_loss': orthogonal_loss}

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # Decoder observations, including any future values, never enter the model.
        # A training harness must explicitly add the upstream orthogonal loss when
        # enabled; the ordinary TSLib forecast return is only the prediction tensor.
        return self.forward_with_aux(x_enc, x_mark_dec)[0]
