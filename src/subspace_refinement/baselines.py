"""Simple comparison methods, selected using the external event holdout."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve
from .lhss import gaussian_kernel


def covariance_direction(x, weights):
    delta = np.cov(x[weights > 0], rowvar=False) - np.cov(x[weights < 0], rowvar=False)
    if np.ndim(delta) == 0:
        return np.ones((1, 1))
    values, vectors = np.linalg.eigh(delta)
    return vectors[:, np.argmax(np.abs(values)), None]


@dataclass
class KernelRatio:
    projection: np.ndarray | None
    centers: np.ndarray
    sigma: float
    alpha: np.ndarray
    mass_ratio: float

    def predict_refinement(self, x):
        z = x if self.projection is None else x@self.projection
        ratio = np.empty(len(z))
        for start in range(0, len(z), 4000):
            ratio[start:start + 4000] = np.maximum(
                gaussian_kernel(z[start:start + 4000], self.centers, self.sigma)@self.alpha, 0)
        ratio *= self.mass_ratio
        return (1 - ratio)/(1 + ratio)


def fit_kernel(data, seed, projection=None):
    z = data.train_x if projection is None else data.train_x@projection
    zv = data.valid_x if projection is None else data.valid_x@projection
    p, q = z[data.train_w > 0], z[data.train_w < 0]
    pv, qv = zv[data.valid_w > 0], zv[data.valid_w < 0]
    rng = np.random.default_rng(seed)
    centers = q[rng.choice(len(q), min(64, len(q)), replace=False)]
    reference = np.concatenate([p[:300], q[:100]])
    distances = np.maximum((reference*reference).sum(1)[:, None]
                           + (centers*centers).sum(1)[None, :] - 2*reference@centers.T, 0)
    width = np.sqrt(np.median(distances[distances > 1e-12]))
    best = None
    for factor in (.15, .3, .6, 1.2, 2.4):
        sigma = width*factor
        kp, kq = gaussian_kernel(p, centers, sigma), gaussian_kernel(q, centers, sigma)
        hessian, mean = kp.T@kp/len(p), kq.mean(0)
        kvp, kvq = gaussian_kernel(pv, centers, sigma), gaussian_kernel(qv, centers, sigma)
        for ridge in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
            alpha = solve(hessian + ridge*np.eye(len(centers)), mean, assume_a='pos')
            pp, qq = np.maximum(kvp@alpha, 0), np.maximum(kvq@alpha, 0)
            risk = .5*np.mean(pp*pp) - np.mean(qq)
            if best is None or risk < best[0]:
                best = (risk, KernelRatio(projection, centers, sigma, alpha, data.mass_ratio), ridge)
    return best[1], {'sigma': float(best[1].sigma), 'ridge': float(best[2]),
                     'external_validation_risk': float(best[0])}

