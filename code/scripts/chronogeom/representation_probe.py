"""Offline numerical-basis regression; no model/tokenizer downloads or training.

PCA reconstructions use observed test activations and are diagnostic compression
controls, not comparable numerical predictors. Numeric regressions see only x.
"""
from __future__ import annotations
import numpy as np


def basis(values, kind, periods=(2., 5., 10., 100.), scale=99., degree=8):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('values must be a finite vector')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be finite and positive')
    parts = []
    if kind in ('circle', 'fourier', 'helix'):
        p = np.asarray(periods, dtype=np.float64)
        if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError('periods must be finite, positive, nonempty')
        if kind == 'circle' and len(p) != 1:
            raise ValueError('circle requires exactly one period')
        for period in p:
            angle = 2 * np.pi * x / period
            parts.extend([np.cos(angle), np.sin(angle)])
    if kind in ('linear', 'helix'):
        parts.append(x / scale)
    elif kind == 'polynomial':
        parts.extend([(x / scale) ** power for power in range(1, degree + 1)])
    elif kind not in ('circle', 'fourier'):
        raise ValueError('unknown basis: ' + kind)
    parts.append(np.ones_like(x))
    return np.column_stack(parts)


def fit_predict(x_train, y_train, x_eval, rcond=1e-10):
    x_train, y_train, x_eval = map(np.asarray, (x_train, y_train, x_eval))
    if x_train.ndim != 2 or y_train.ndim != 2 or len(x_train) != len(y_train):
        raise ValueError('design and target must be aligned matrices')
    coef, _, rank, singular = np.linalg.lstsq(x_train, y_train, rcond=rcond)
    return x_eval @ coef, {'columns': x_train.shape[1], 'effective_rank': int(rank),
                         'coefficient_entries': int(coef.size),
                         'singular_values': singular.tolist()}


def metrics(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape:
        raise ValueError('prediction shape mismatch')
    squared = np.square(target - prediction)
    denom = np.square(target - target.mean(axis=0)).sum()
    return {'mse': float(squared.mean()), 'rmse': float(np.sqrt(squared.mean())),
            'r2': float(1 - squared.sum() / denom) if denom > 0 else None}


def split_values(values, seed, fraction=.8):
    values = np.asarray(values)
    unique = np.unique(values)
    if len(unique) < 5 or not 0 < fraction < 1:
        raise ValueError('need >=5 identities and 0<fraction<1')
    selected = np.random.default_rng(seed).permutation(unique)[:int(len(unique) * fraction)]
    train = np.isin(values, selected)
    return train, ~train
