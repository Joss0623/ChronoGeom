"""MIT Circle-inspired analysis adapters for TSLib, not an encoder.

Source concept: JoshEngels/MultiDimensionalFeatures (MIT), pinned
ff8c7eee8f13d78805f590ced6f3fc2227dd1daa,
intervention/circle_finding_utils.py: do_regression and subspace interventions.
Adaptation uses float64 direct least squares instead of normal equations,
train-only bases, rank checks and forecasting outputs instead of logits.
Software fixtures do not constitute a learned-circle or forecasting result.
"""
from dataclasses import dataclass
import numpy as np
import torch

OFFICIAL_COMMIT = "ff8c7eee8f13d78805f590ced6f3fc2227dd1daa"


def _design(phase, period, nuisance=None):
    phase = np.asarray(phase, dtype=np.float64).reshape(-1)
    if not np.isfinite(period) or period <= 0 or not np.isfinite(phase).all():
        raise ValueError("Finite phases and positive physical period required")
    n = len(phase)
    z = np.empty((n, 0)) if nuisance is None else np.asarray(nuisance, dtype=np.float64)
    if z.ndim != 2 or len(z) != n or not np.isfinite(z).all():
        raise ValueError("Nuisance must be finite N x K")
    base = np.column_stack([np.ones(n), z])
    angle = 2 * np.pi * phase / period
    full = np.column_stack([base, np.cos(angle), np.sin(angle)])
    return base, full


def _acts(acts, n):
    x = np.asarray(acts, dtype=np.float64)
    if x.ndim != 2 or len(x) != n or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Activations must be finite N x D with D >= 2")
    return x


def _row_basis(matrix, rtol=1e-8):
    _, s, vt = np.linalg.svd(matrix, full_matrices=False)
    r = int(np.sum(s > max(1e-12, (s[0] if len(s) else 0) * rtol)))
    return vt[:r].T, s


@dataclass
class CircleFit:
    period: float
    base_coef: np.ndarray
    full_coef: np.ndarray
    mean: np.ndarray
    circle_basis: np.ndarray
    pca_basis: np.ndarray
    random_basis: np.ndarray
    singular_values: np.ndarray
    source_commit: str = OFFICIAL_COMMIT


def fit_circle(train_acts, train_phase, period, nuisance=None, seed=1729):
    """Caller must split trajectories/contexts before fitting; never fit test data."""
    base, full = _design(train_phase, period, nuisance)
    x = _acts(train_acts, len(base))
    if np.linalg.matrix_rank(full) - np.linalg.matrix_rank(base) != 2:
        raise ValueError("Phase pair is not identifiable given nuisance/train phases")
    b = np.linalg.lstsq(base, x, rcond=1e-10)[0]
    c = np.linalg.lstsq(full, x, rcond=1e-10)[0]
    q, s = _row_basis(c[-2:])
    mean = x.mean(0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    rank = q.shape[1]
    pca = vt[:rank].T
    rng = np.random.default_rng(seed)
    random = np.linalg.qr(rng.normal(size=(x.shape[1], rank)))[0] if rank else np.empty((x.shape[1], 0))
    return CircleFit(float(period), b, c, mean, q, pca, random, s)


def evaluate_circle(fit, acts, phase, nuisance=None):
    """Held-out reconstruction R2; not proof of a Euclidean circle or causality."""
    b, f = _design(phase, fit.period, nuisance)
    x = _acts(acts, len(b))
    if x.shape[1] != len(fit.mean) or b.shape[1] != len(fit.base_coef):
        raise ValueError("Train/evaluation feature schema differs")
    denominator = np.sum((x - x.mean(0)) ** 2)
    if denominator <= 1e-20:
        raise ValueError("R2 undefined for constant evaluation activations")
    r0 = 1 - np.sum((x - b @ fit.base_coef) ** 2) / denominator
    r1 = 1 - np.sum((x - f @ fit.full_coef) ** 2) / denominator
    s = fit.singular_values
    return dict(baseline_r2=float(r0), circle_augmented_r2=float(r1),
                incremental_r2=float(r1 - r0), effective_rank=fit.circle_basis.shape[1],
                axis_ratio=float(s[1] / s[0]) if len(s) > 1 and s[0] > 1e-12 else 0.)


def _module(model, site):
    modules = dict(model.named_modules())
    if site not in modules:
        raise ValueError("Unknown hook site: " + site)
    return modules[site]


def _select(output, position):
    if not isinstance(output, torch.Tensor):
        raise ValueError("Hook site must return a tensor, not attention tuples")
    if position is None:
        return output
    if output.ndim != 3 or not -output.shape[1] <= position < output.shape[1]:
        raise ValueError("Position requires B x T x D hook output")
    return output[:, position, :]


def capture_site(model, site, forward_args, position=None):
    captured = []
    def hook(module, args, output):
        captured.append(_select(output, position).detach().clone())
    handle = _module(model, site).register_forward_hook(hook)
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            prediction = model(*forward_args)
        if len(captured) != 1:
            raise ValueError("Site must execute exactly once per forecast")
        return prediction, captured[0]
    finally:
        handle.remove()
        model.train(was_training)


def forward_with_patch(model, site, forward_args, donor=None, basis=None,
                       position=None, mode="subspace"):
    """Donor replacement/mean ablation; caller controls pairs and fit/test split.

    mode=noop leaves output unchanged; full replaces selected activation;
    subspace replaces its Q-coordinates. Passing a train mean as donor performs
    centered ablation. Distribution preservation is not guaranteed by this API.
    """
    if mode not in ("noop", "full", "subspace"):
        raise ValueError("Unknown patch mode")
    calls = []
    def hook(module, args, output):
        x = _select(output, position)
        calls.append(1)
        if mode == "noop":
            return output
        d = torch.as_tensor(donor, dtype=x.dtype, device=x.device)
        if d.shape != x.shape:
            raise ValueError("Donor must match selected activation exactly")
        if mode == "full":
            changed = d
        else:
            q = torch.as_tensor(basis, dtype=x.dtype, device=x.device)
            if q.ndim != 2 or q.shape[0] != x.shape[-1]:
                raise ValueError("Basis shape mismatch")
            eye = torch.eye(q.shape[1], dtype=q.dtype, device=q.device)
            if not torch.allclose(q.T @ q, eye, atol=1e-5, rtol=1e-5):
                raise ValueError("Basis must be orthonormal")
            changed = x + ((d - x) @ q) @ q.T
        if position is None:
            return changed
        result = output.clone()
        result[:, position, :] = changed
        return result
    handle = _module(model, site).register_forward_hook(hook)
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            prediction = model(*forward_args)
        if len(calls) != 1:
            raise ValueError("Site must execute exactly once per forecast")
        return prediction
    finally:
        handle.remove()
        model.train(was_training)
