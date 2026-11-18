"""No datasets or model downloads are required. Run from TSLib root:
python -m unittest discover -s tests/chronogeom -p 'test_*.py' -v
"""
import unittest

import torch

from layers.ChronoGeometry import (
    periodic_pairs,
    decimal_fourier_number,
    numerical_helix,
    time_phase,
    time_helix,
)


class TestGeometry(unittest.TestCase):
    def test_pair_dimensions_and_dtype(self):
        values = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
        features = periodic_pairs(values, [2., 5., 10.])
        self.assertEqual(features.shape, (2, 2, 6))
        self.assertEqual(features.dtype, torch.float64)
        self.assertTrue(torch.isfinite(features).all().item())

    def test_integer_values_are_supported(self):
        x = torch.tensor([0, 1, 2], dtype=torch.int64)
        self.assertEqual(periodic_pairs(x, [7]).shape, (3, 2))
        self.assertTrue(periodic_pairs(x, [7]).is_floating_point())

    def test_periodic_alias_and_helix_separation(self):
        x = torch.tensor([3., 13.], dtype=torch.float64)
        circle = decimal_fourier_number(x, powers=[1])
        helix = numerical_helix(x, numeric_periods=[10.], magnitude_scale=10.)
        self.assertLess(torch.linalg.vector_norm(circle[0] - circle[1]).item(), 1e-12)
        self.assertAlmostEqual(torch.linalg.vector_norm(helix[0] - helix[1]).item(), 1., places=10)

    def test_time_phase_and_time_helix(self):
        t = torch.tensor([2., 26.], dtype=torch.float64)
        phase = time_phase(t, physical_periods=[24.])
        helix = time_helix(t, physical_periods=[24.], time_scale=24.)
        self.assertLess(torch.linalg.vector_norm(phase[0] - phase[1]).item(), 1e-12)
        self.assertAlmostEqual(torch.linalg.vector_norm(helix[0] - helix[1]).item(), 1., places=10)

    def test_invalid_periods(self):
        for periods in ([], [0.], [-1.], [1., 0.]):
            with self.subTest(periods=periods):
                with self.assertRaises(ValueError):
                    periodic_pairs(torch.tensor([1.]), periods)

    def test_invalid_scales(self):
        with self.assertRaises(ValueError):
            time_helix(torch.tensor([1.]), [24.], 0.)
        with self.assertRaises(ValueError):
            numerical_helix(torch.tensor([1.]), [10.], -1.)


if __name__ == '__main__':
    unittest.main()
