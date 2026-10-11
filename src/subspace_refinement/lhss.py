"""Native NumPy/SciPy D³-LHSS estimator.

Mathematical reference: Sugiyama et al., Neural Networks 24 (2011), 183–198,
https://www.ms.k.u-tokyo.ac.jp/sugi/2011/LHSS.pdf.
The experimental protocol follows the authors' distributed LHSS software:
two-fold uLSIF selection, coefficient clipping, Grassmann rotations, and
automatic dimension selection. This module does not require MATLAB.

NumPy's PCG64 permutations differ from MATLAB's legacy random generator.
Consequently ordinary fits are algorithmic replications, not identical runs.
Private validation can supply matching random choices and initial subspaces.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.linalg import expm, qr, solve


def gaussian_kernel(x, centers, sigma):
    """Gaussian features; samples are rows throughout this module."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Kernel width must be finite and positive.')
    distances = (x*x).sum(axis=1)[:, None] + (centers*centers).sum(axis=1)[None, :] - 2*x@centers.T
    # Do not clip round-off in distances: retain the reference calculation.
    return np.exp(-distances / (2*sigma*sigma))


def median_width(x):
    x = x[:500]
    norms = (x*x).sum(axis=1)
    distances = norms[:, None] + norms[None, :] - 2*x@x.T
    positive = distances[np.triu_indices(len(x), 1)]
    positive = positive[positive > 0]
    if not len(positive):
        raise ValueError('Median bandwidth is undefined for constant projected data.')
    return float(np.sqrt(.5*np.median(positive)))


@dataclass
class RandomChoices:
    """Fixed random choices, optionally with initial bases for parity checks."""
    centers: np.ndarray
    numerator_order: np.ndarray
    denominator_order: np.ndarray
    initial_bases: dict | None = None


@dataclass
class RatioFit:
    alpha: np.ndarray
    sigma: float
    ridge: float
    divergence: float
    cv_score: float


def fit_ulsif(denominator, numerator, centers, widths, choices,
              ridges=None):
    """Select the Gaussian width/ridge using the reference two-fold risk."""
    ridges = np.logspace(-3, 1, 9) if ridges is None else np.asarray(ridges)
    n_de, n_nu = len(denominator), len(numerator)
    partitions = []
    for order, count in [(choices.denominator_order, n_de),
                         (choices.numerator_order, n_nu)]:
        folds = (np.arange(count)*2//count)
        partitions.append([(order[folds != f], order[folds == f]) for f in range(2)])
    de_folds, nu_folds = partitions
    eye = np.eye(len(centers))
    best = None
    for sigma in widths:
        kde = gaussian_kernel(denominator, centers, sigma)
        knu = gaussian_kernel(numerator, centers, sigma)
        fold_data = []
        for (de_train, de_valid), (nu_train, nu_valid) in zip(de_folds, nu_folds):
            train = kde[de_train]
            fold_data.append((train.T@train/len(train), knu[nu_train].mean(axis=0),
                              kde[de_valid], knu[nu_valid]))
        for ridge in ridges:
            risks = []
            for hessian, mean, de_valid, nu_valid in fold_data:
                alpha = np.maximum(solve(hessian + ridge*eye, mean,
                                         assume_a='gen', check_finite=False), 0)
                risks.append(.5*np.mean((de_valid@alpha)**2) - np.mean(nu_valid@alpha))
            score = float(np.mean(risks))
            if best is None or score < best[0]:
                best = (score, float(sigma), float(ridge), kde, knu)
    score, sigma, ridge, kde, knu = best
    mean = knu.mean(axis=0)
    alpha = np.maximum(solve(kde.T@kde/n_de + ridge*eye, mean,
                             assume_a='gen', check_finite=False), 0)
    return RatioFit(alpha, sigma, ridge, float(mean@alpha - .5), score)


def pearson_gradient(denominator, numerator, centers, basis, fit):
    """Derivative of the reference Pearson objective, shape (k, d)."""
    de, nu, ce = denominator@basis, numerator@basis, centers@basis
    kde = gaussian_kernel(de, ce, fit.sigma)
    knu = gaussian_kernel(nu, ce, fit.sigma)
    weighted_nu = knu*fit.alpha
    weighted_de = (kde*fit.alpha)*(kde@fit.alpha)[:, None]

    def scatter(projected, raw, weights):
        return ((projected*weights.sum(axis=1)[:, None]).T@raw
                - (projected.T@weights)@centers
                - ce.T@(weights.T@raw)
                + (ce*weights.sum(axis=0)[:, None]).T@centers)

    return (-scatter(nu, numerator, weighted_nu)/len(numerator)
            + scatter(de, denominator, weighted_de)/len(denominator))/fit.sigma**2


class D3LHSS:
    """Density ratio p_numerator/p_denominator in a learned linear subspace.

    Parameters retain the reference defaults. Set max_dim to the observed
    dimension to search every possible k, as in our full benchmark.
    """
    def __init__(self, n_centers=100, max_dim=5, random_state=0):
        self.n_centers = n_centers
        self.max_dim = max_dim
        self.random_state = random_state

    def _fixed_dimension(self, de, nu, k, choices):
        dim = de.shape[1]
        if choices.initial_bases is not None:
            basis = np.array(choices.initial_bases[k], dtype=float, copy=True)
        else:
            q, r = qr(np.ones((dim, k)), mode='full')
            basis = (q*np.sign(r[0, 0]))[:, :k]
        centers = nu[choices.centers[:self.n_centers]]

        def estimate(u):
            width = median_width(nu@u)
            return fit_ulsif(de@u, nu@u, centers@u,
                             width*np.array([.75, 1., 1.25]), choices)

        current = estimate(basis)
        best_basis, best_fit = basis.copy(), current
        bad_steps = 0
        for step in range(200, 0, -1):
            q, r = qr(basis, mode='full')
            q *= np.sign(r[0, 0])
            basis, complement = q[:, :k], q[:, k:]
            tangent = pearson_gradient(de, nu, centers, basis, current)@complement
            generator = np.zeros((dim, dim))
            generator[:k, k:] = tangent
            generator[k:, :k] = -tangent.T
            basis = (expm(.005*step*generator)@q.T)[:k].T
            current = estimate(basis)
            if current.divergence > best_fit.divergence:
                best_basis, best_fit = basis.copy(), current
                bad_steps = max(0, bad_steps - 1)
            else:
                bad_steps += 1
            if bad_steps > 20:
                break
        return best_basis, best_fit, centers

    def fit(self, denominator, numerator, *, choices=None):
        de, nu = np.asarray(denominator, dtype=float), np.asarray(numerator, dtype=float)
        if (de.ndim != 2 or nu.ndim != 2 or de.shape[1] != nu.shape[1]
                or de.shape[1] < 1 or min(len(de), len(nu)) < 2
                or not np.isfinite(de).all() or not np.isfinite(nu).all()):
            raise ValueError('Finite sample matrices with matching nonzero feature dimensions are required.')
        if not isinstance(self.n_centers, (int, np.integer)) or not 1 <= self.n_centers <= len(nu):
            raise ValueError('n_centers must be an integer between one and the numerator sample count.')
        if not isinstance(self.max_dim, (int, np.integer)) or self.max_dim < 1:
            raise ValueError('max_dim must be a positive integer.')
        if choices is None:
            center_rng = np.random.default_rng(self.random_state)
            fold_rng = np.random.default_rng(self.random_state)
            choices = RandomChoices(center_rng.permutation(len(nu)),
                                    fold_rng.permutation(len(nu)),
                                    fold_rng.permutation(len(de)))
        for order, size in [(choices.centers, len(nu)),
                            (choices.numerator_order, len(nu)),
                            (choices.denominator_order, len(de))]:
            if not np.array_equal(np.sort(order), np.arange(size)):
                raise ValueError('Random choices must contain valid sample permutations.')
        scores = []
        for k in range(1, min(de.shape[1], self.max_dim) + 1):
            _, fitted, _ = self._fixed_dimension(de, nu, k, choices)
            scores.append(fitted.cv_score)
        self.dimension_scores_ = np.array(scores)
        self.selected_dim_ = int(np.argmin(scores) + 1)
        self.projection_, fitted, self.centers_ = self._fixed_dimension(de, nu, self.selected_dim_, choices)
        self.alpha_, self.sigma_, self.ridge_ = fitted.alpha, fitted.sigma, fitted.ridge
        self.cv_score_, self.divergence_ = fitted.cv_score, fitted.divergence
        return self

    def predict_ratio(self, x, chunk_size=4096):
        if not hasattr(self, 'projection_'):
            raise ValueError('Fit the estimator before prediction.')
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.projection_.shape[0] or not np.isfinite(x).all():
            raise ValueError('Prediction samples must have the training feature dimension and finite values.')
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer.')
        centers = self.centers_@self.projection_
        result = np.empty(len(x))
        for start in range(0, len(x), chunk_size):
            block = x[start:start + chunk_size]@self.projection_
            result[start:start + len(block)] = gaussian_kernel(block, centers, self.sigma_)@self.alpha_
        return result

    def predict_refinement(self, x, mass_ratio):
        if not np.isfinite(mass_ratio) or mass_ratio < 0:
            raise ValueError('The negative/positive mass ratio must be finite and nonnegative.')
        ratio = mass_ratio*self.predict_ratio(x)
        return (1 - ratio)/(1 + ratio)

    def save(self, path):
        """Save fitted numerical state without pickle or a MATLAB dependency."""
        if not hasattr(self, 'projection_'):
            raise ValueError('Fit the estimator before saving.')
        np.savez_compressed(Path(path), projection=self.projection_, centers=self.centers_,
                            alpha=self.alpha_, sigma=self.sigma_, ridge=self.ridge_,
                            selected_dim=self.selected_dim_, dimension_scores=self.dimension_scores_,
                            cv_score=self.cv_score_, divergence=self.divergence_)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as state:
            result = cls(n_centers=len(state['centers']), max_dim=len(state['dimension_scores']))
            for key in ['projection', 'centers', 'alpha', 'dimension_scores']:
                setattr(result, key + '_', state[key].copy())
            for key in ['sigma', 'ridge', 'cv_score', 'divergence']:
                setattr(result, key + '_', float(state[key]))
            result.selected_dim_ = int(state['selected_dim'])
        return result
