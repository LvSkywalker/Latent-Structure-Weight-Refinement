import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from subspace_refinement import D3LHSS
from subspace_refinement.data import cases, dataset, split_indices, truth
from subspace_refinement.lhss import gaussian_kernel, pearson_gradient


class TestNumerics(unittest.TestCase):
    def test_gradient_finite_differences(self):
        rng = np.random.default_rng(17)
        de, nu, centers = [rng.normal(size=(n, 4)) for n in (23, 19, 7)]
        basis = np.linalg.qr(rng.normal(size=(4, 2)))[0]
        fitted = SimpleNamespace(alpha=rng.uniform(.01, .3, 7), sigma=1.2)

        def objective(u):
            p = gaussian_kernel(de@u, centers@u, fitted.sigma)@fitted.alpha
            q = gaussian_kernel(nu@u, centers@u, fitted.sigma)@fitted.alpha
            return np.mean(q) - .5*np.mean(p*p)

        numeric = np.empty_like(basis)
        for index in np.ndindex(basis.shape):
            perturbation = np.zeros_like(basis)
            perturbation[index] = 1e-5
            numeric[index] = (objective(basis + perturbation) - objective(basis - perturbation))/2e-5
        np.testing.assert_allclose(pearson_gradient(de, nu, centers, basis, fitted).T,
                                   numeric, rtol=2e-6, atol=2e-9)

    def test_event_split_is_disjoint(self):
        for case in cases():
            train, valid = split_indices(case['n'], case['seed'])
            self.assertEqual(len(np.intersect1d(train, valid)), 0)
            self.assertEqual(len(train) + len(valid), case['n'])

    def test_truth_uses_only_the_signal_coordinate(self):
        first = np.array([[.2, -4., 8.], [1., 2., -3.]])
        second = first.copy()
        second[:, 1:] += 100
        np.testing.assert_array_equal(truth(first)[0], truth(second)[0])
        self.assertTrue(np.all(truth(first)[0] >= .2))

    def test_native_fit_serialization_and_orthogonality(self):
        rng = np.random.default_rng(31)
        p, q = rng.normal(size=(90, 2)), rng.normal(scale=.6, size=(60, 2))
        model = D3LHSS(n_centers=12, max_dim=2).fit(p, q)
        np.testing.assert_allclose(model.projection_.T@model.projection_,
                                   np.eye(model.selected_dim_), atol=1e-10)
        before = model.predict_ratio(p)
        self.assertTrue(np.all(np.isfinite(before) & (before >= 0)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.npz'
            model.save(path)
            np.testing.assert_array_equal(before, D3LHSS.load(path).predict_ratio(p))

    def test_invalid_samples(self):
        with self.assertRaises(ValueError):
            D3LHSS().fit(np.ones((5, 2)), np.ones((5, 3)))
        with self.assertRaises(ValueError):
            D3LHSS().fit(np.ones((5, 2)), np.ones((5, 2)))
        with self.assertRaises(ValueError):
            D3LHSS().predict_ratio(np.ones((5, 2)))


if __name__ == '__main__':
    unittest.main()
