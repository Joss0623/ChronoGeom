"""ChronoGeom Phase-0 geometry primitives for THUML Time-Series-Library.

This file provides *only fixed representations*, not a forecasting model.
Place it at <Time-Series-Library>/layers/ChronoGeometry.py.

Each representation keeps the original leading tensor dimensions and appends
feature dimensions. All timestamps and periods must be known at forecast time.
No train/test statistics are estimated in this module.
"""
from __future__ import annotations

from typing import Iterable
import math
import torch


def _period_tensor(values: Iterable[float], reference: torch.Tensor) -> torch.Tensor:
    periods = torch.as_tensor(list(values), dtype=reference.dtype, device=reference.device)
    if periods.ndim != 1 or periods.numel() == 0:
        raise ValueError('periods must be a nonempty 1-D sequence')
    if not bool(torch.all(periods > 0).item()):
        raise ValueError('periods must all be strictly positive')
    return periods


def periodic_pairs(coordinate: torch.Tensor, periods: Iterable[float]) -> torch.Tensor:
    """[cos(2π coordinate/T_j), sin(2π coordinate/T_j)] for each T_j.

    Output: coordinate.shape + (2*K,).
    A periodic representation is non-injective when shifted by any common period.
    """
    if not coordinate.is_floating_point():
        coordinate = coordinate.float()
    p = _period_tensor(periods, coordinate)
    angles = (2 * math.pi) * coordinate.unsqueeze(-1) / p
    return torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(-2)


def decimal_fourier_number(value: torch.Tensor, powers: Iterable[int]) -> torch.Tensor:
    """FoNE-style decimal-scale numeric features: T in {10**k}.

    Only the input representation is reproduced; this is NOT the paper's
    full number-token substitution, digitwise output decoder or loss function.
    """
    exponents = list(powers)
    return periodic_pairs(value, (10.0 ** k for k in exponents))


def time_phase(timestamp: torch.Tensor, physical_periods: Iterable[float]) -> torch.Tensor:
    """Physical time-phase coordinates. timestamp and periods use the SAME unit.

    For example: timestamp in hours; periods=[24, 168].
    The time-phase T=24 has no relationship to decimal numeric T=10**k.
    """
    return periodic_pairs(timestamp, physical_periods)


def time_helix(timestamp: torch.Tensor,
               physical_periods: Iterable[float],
               time_scale: float) -> torch.Tensor:
    """Engineered time helix = [time/time_scale, physical time-phase features].

    This is a candidate experimental condition, NOT a replication of MIT
    activation geometry nor a proven optimal forecasting representation.
    """
    if time_scale <= 0:
        raise ValueError('time_scale must be positive')
    return torch.cat((timestamp.unsqueeze(-1) / float(time_scale),
                      time_phase(timestamp, physical_periods)), dim=-1)


def numerical_helix(value: torch.Tensor,
                     numeric_periods: Iterable[float],
                     magnitude_scale: float = 1.0) -> torch.Tensor:
    """MIT-form candidate [x/scale, cos(2πx/T_1), sin(...), ...].

    Important: when value is RevIN normalized, periods are in normalized units;
    fixed decimal periods then lose their original decimal-place meaning.
    """
    if magnitude_scale <= 0:
        raise ValueError('magnitude_scale must be positive')
    return torch.cat((value.unsqueeze(-1) / float(magnitude_scale),
                      periodic_pairs(value, numeric_periods)), dim=-1)
