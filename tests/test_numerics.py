import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from subspace_refinement import D3LHSS
from subspace_refinement.data import cases, dataset, split_indices, truth
from subspace_refinement.lhss import gaussian_kernel, pearson_gradient
from subspace_refinement.report import CONFIRM_METHODS, confirm_report, validate_confirm
import pandas as pd


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

    def test_confirm_seeds_are_fresh(self):
        development = {case['seed'] for case in cases('full')}
        confirm = cases('confirm')
        self.assertEqual(len(confirm), 10)
        self.assertTrue(all(case['d'] == 30 for case in confirm))
        self.assertFalse(development & {case['seed'] for case in confirm})

    def test_invalid_samples(self):
        with self.assertRaises(ValueError):
            D3LHSS().fit(np.ones((5, 2)), np.ones((5, 3)))
        with self.assertRaises(ValueError):
            D3LHSS().fit(np.ones((5, 2)), np.ones((5, 2)))
        with self.assertRaises(ValueError):
            D3LHSS().predict_ratio(np.ones((5, 2)))


if __name__ == '__main__':
    unittest.main()


def confirm_rows(extra=()):
    rows = []
    for case in cases('confirm'):
        for i, method in enumerate(tuple(CONFIRM_METHODS) + tuple(extra)):
            rows.append({**case, 'method': method, 'c_rmse': .1 + .01*i + case['seed']/1e4})
    return pd.DataFrame(rows)


class TestConfirmReport(unittest.TestCase):
    def assert_rejected(self, rows, fragment):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'out'
            with self.assertRaisesRegex(ValueError, fragment):
                confirm_report(rows, target)
            self.assertFalse(target.exists(), 'Nothing must be written for invalid input.')

    def test_valid_input_and_optional_kernel_full(self):
        rows = confirm_rows(extra=('kernel_full',))
        with tempfile.TemporaryDirectory() as directory:
            text = confirm_report(rows, Path(directory))
            self.assertIn('5 of 5', text)
            self.assertTrue((Path(directory)/'report.md').is_file())
            saved = pd.read_csv(Path(directory)/'results.csv')
            self.assertEqual(len(saved), len(rows))

    def test_rejects_missing_column(self):
        self.assert_rejected(confirm_rows().drop(columns='seed'), 'required columns')

    def test_rejects_incomplete_cases(self):
        rows = confirm_rows()
        self.assert_rejected(rows[rows.seed != 211], 'incomplete')

    def test_rejects_missing_method(self):
        rows = confirm_rows()
        drop = (rows.method == 'nn_lhss') & (rows.seed == 131) & (rows.n == 2000)
        self.assert_rejected(rows[~drop], 'lacks confirm methods')

    def test_rejects_duplicates(self):
        rows = confirm_rows()
        self.assert_rejected(pd.concat([rows, rows.iloc[[0]]]), 'Duplicate')

    def test_rejects_non_finite_and_negative(self):
        rows = confirm_rows()
        rows.loc[0, 'c_rmse'] = np.nan
        self.assert_rejected(rows, 'finite')
        rows = confirm_rows()
        rows.loc[0, 'c_rmse'] = -.1
        self.assert_rejected(rows, 'non-negative')

    def test_rejects_zero_denominator(self):
        rows = confirm_rows()
        rows.loc[rows.method == 'nn_full', 'c_rmse'] = 0.
        self.assert_rejected(rows, 'strictly positive')

    def test_rejects_wrong_cases(self):
        rows = confirm_rows()
        rows.loc[0, 'd'] = 10
        self.assert_rejected(rows, 'expected')
        other_suite = pd.DataFrame([{**case, 'method': m, 'c_rmse': .1}
                                    for case in cases('full')[:2] for m in CONFIRM_METHODS])
        self.assert_rejected(other_suite, 'confirm suite')

    def test_rejects_unknown_method(self):
        self.assert_rejected(confirm_rows(extra=('other',)), 'Unexpected methods')

    def test_saved_confirm_results_are_valid(self):
        saved = Path(__file__).resolve().parents[1]/'results'/'confirm'/'results.csv'
        if saved.exists():
            validate_confirm(pd.read_csv(saved))
