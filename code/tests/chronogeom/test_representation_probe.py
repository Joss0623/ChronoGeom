import unittest
import numpy as np
from scripts.chronogeom.representation_probe import basis, fit_predict, metrics, split_values


class TestRepresentationProbe(unittest.TestCase):
    def test_recover_known_helix_on_unseen_values(self):
        x = np.arange(100.)
        design = basis(x, 'helix')
        weights = np.random.default_rng(2).normal(size=(design.shape[1], 16))
        target = design @ weights
        train, test = split_values(x, seed=0)
        pred, info = fit_predict(design[train], target[train], design[test])
        self.assertLess(metrics(target[test], pred)['mse'], 1e-20)
        self.assertEqual(info['effective_rank'], 9)  # zero sin(pi*x) column

    def test_integer_nyquist_and_periodic_alias(self):
        x = np.arange(100.)
        matrix = basis(x, 'helix')
        self.assertLess(np.max(np.abs(matrix[:, 1])), 1e-12)
        self.assertEqual(np.linalg.matrix_rank(matrix, tol=1e-8), 9)
        circle = basis(np.array([1., 101.]), 'circle', periods=[100.])
        np.testing.assert_allclose(circle[0], circle[1], atol=1e-12)
        helix = basis(np.array([1., 101.]), 'helix')
        self.assertGreater(np.linalg.norm(helix[0] - helix[1]), 1.)

    def test_split_holds_out_identities_not_rows(self):
        x = np.repeat(np.arange(100), 3)
        train, test = split_values(x, 3)
        self.assertEqual(len(set(x[train]) & set(x[test])), 0)
        self.assertEqual(np.unique(x[train]).size, 80)

    def test_negative_control_breaks_known_mapping(self):
        x = np.arange(100.)
        design = basis(x, 'circle', periods=[10.])
        y = design[:, :2]
        train, test = split_values(x, 2)
        wrong = design[train][np.random.default_rng(3).permutation(train.sum())]
        pred, _ = fit_predict(wrong, y[train], design[test])
        self.assertLess(metrics(y[test], pred)['r2'], .5)

    def test_invalid_geometry_rejected(self):
        for p in ([], [0.], [np.inf]):
            with self.assertRaises(ValueError):basis(np.arange(4.), 'fourier', periods=p)
        with self.assertRaises(ValueError):basis(np.arange(4.), 'circle', periods=[2., 10.])


if __name__ == '__main__':
    unittest.main()
