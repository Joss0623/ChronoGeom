"""Run from THUML Time-Series-Library root:
    python scripts/chronogeom/phase0_geometry_check.py

No real-world dataset, GPU, model checkpoints, or training is required.
This is a mathematical sanity check only, not a forecasting performance claim.
"""
from pathlib import Path
import sys

# Direct execution (python scripts/chronogeom/phase0_geometry_check.py)
# does not reliably put the TSLib repository root on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from layers.ChronoGeometry import (
    decimal_fourier_number, numerical_helix, time_phase, time_helix
)


def main():
    x = torch.tensor([3.0, 13.0], dtype=torch.float64)
    f = decimal_fourier_number(x, powers=[1])
    n = numerical_helix(x, numeric_periods=[10.0], magnitude_scale=10.0)
    periodic_collision = torch.linalg.vector_norm(f[0] - f[1]).item()
    helix_separation = torch.linalg.vector_norm(n[0] - n[1]).item()

    t = torch.tensor([2.0, 26.0], dtype=torch.float64)  # hours 2 and 26
    phase = time_phase(t, physical_periods=[24.0])
    helix = time_helix(t, physical_periods=[24.0], time_scale=24.0)
    same_phase_dist = torch.linalg.vector_norm(phase[0] - phase[1]).item()
    separate_time_dist = torch.linalg.vector_norm(helix[0] - helix[1]).item()

    assert periodic_collision < 1e-12, periodic_collision
    assert helix_separation > 0.9, helix_separation
    assert same_phase_dist < 1e-12, same_phase_dist
    assert separate_time_dist > 0.9, separate_time_dist

    print('FoNE-style T=10: x=3 and x=13 distance: ', round(periodic_collision, 12))
    print('Numerical helix: x=3 and x=13 distance: ', round(helix_separation, 12))
    print('Physical phase T=24h: t=2h and 26h distance:', round(same_phase_dist, 12))
    print('Time helix: t=2h and 26h distance:        ', round(separate_time_dist, 12))
    print('PASS: representation aliases and global coordinates behave as expected')
    print('NOTE: This does NOT validate forecasting performance or an LLM mechanism.')


if __name__ == '__main__':
    main()
