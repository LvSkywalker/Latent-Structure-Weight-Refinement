"""Self-contained, reproducible synthetic data; no physical event files needed.

The one-dimensional Gaussian setup follows Nachman and Noll's MIT-licensed
weight_refiner generator. Nuisance coordinates, rotation and radial control
are additions of this study. See THIRD_PARTY_NOTICES.md.
"""
from dataclasses import dataclass
import math
import numpy as np

SEEDS = (11, 29, 47, 71, 101)
# Fresh seeds, never used during method development, for the confirmatory run
# of the reduced-input neural comparison at 30 observed features.
CONFIRM_SEEDS = (131, 157, 173, 191, 211)


def sample(n, d, seed, radial=False):
    """Return observed coordinates, signed unit weights, raw coordinates, rotation."""
    if n < 8 or d < 1:
        raise ValueError('Require at least eight events and one feature.')
    legacy = np.random.RandomState(seed)
    n_negative = int(n*.25)
    n_positive = n - n_negative
    z = np.concatenate([legacy.normal(0, 1, (n_positive, 1)),
                        legacy.normal(0, .5, (n_negative, 1))])[:, 0]
    weights = np.concatenate([np.ones(n_positive), -np.ones(n_negative)])
    rng = np.random.default_rng(seed + 10000)
    rotation = np.linalg.qr(rng.normal(size=(d, d)))[0]
    raw = np.column_stack([z, rng.normal(size=(n, d - 1))])
    if radial:
        raw = rng.normal(size=(n, d))
        raw[weights < 0] *= .85
    return raw@rotation.T, weights, raw, rotation


def truth(raw, radial=False):
    """Return exact c and the mass-weighted negative/positive density ratio R."""
    if radial:
        log_ratio = -raw.shape[1]*np.log(.85) - .5*(1/.85**2 - 1)*(raw*raw).sum(axis=1)
    else:
        log_ratio = np.log(2) - 1.5*raw[:, 0]**2
    ratio = np.exp(log_ratio)/3
    return (1 - ratio)/(1 + ratio), ratio


def split_indices(n, seed, validation_fraction=.2):
    """Match sklearn's non-stratified ShuffleSplit for these integer seeds."""
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between zero and one.')
    order = np.random.RandomState(seed).permutation(n)
    count = math.ceil(n*validation_fraction)
    if count >= n:
        raise ValueError('The split must leave training events.')
    return order[count:], order[:count]


@dataclass
class Dataset:
    train_x: np.ndarray
    train_w: np.ndarray
    valid_x: np.ndarray
    valid_w: np.ndarray
    test_x: np.ndarray
    true_c: np.ndarray
    true_ratio: np.ndarray
    raw_test: np.ndarray
    rotation: np.ndarray

    @property
    def mass_ratio(self):
        return float(np.abs(self.train_w[self.train_w < 0]).sum()
                     / self.train_w[self.train_w > 0].sum())


def dataset(case, test_size=24000):
    radial = case['scenario'] == 'all_dimensions_matter'
    n, d, seed = int(case['n']), int(case['d']), int(case['seed'])
    x, weights, _, rotation = sample(n, d, seed, radial)
    _, _, raw_test, _ = sample(test_size, d, seed + 200000, radial)
    train, valid = split_indices(n, seed)
    c, ratio = truth(raw_test, radial)
    return Dataset(x[train], weights[train], x[valid], weights[valid],
                   raw_test@rotation.T, c, ratio, raw_test, rotation)


def cases(suite='full'):
    configs = [(n, d, 'one_latent_direction') for n in (2000, 8000) for d in (1, 10, 30)]
    configs += [(8000, 10, 'all_dimensions_matter')]
    result = [dict(id=f'{scenario}_{n}_{d}_{seed}', n=n, d=d, seed=seed, scenario=scenario)
              for n, d, scenario in configs for seed in SEEDS]
    if suite == 'confirm':
        return [dict(id=f'one_latent_direction_{n}_30_{seed}', n=n, d=30, seed=seed,
                     scenario='one_latent_direction')
                for n in (2000, 8000) for seed in CONFIRM_SEEDS]
    if suite == 'smoke':
        return [case for case in result if case['id'] == 'one_latent_direction_2000_10_11']
    if suite != 'full':
        raise ValueError('suite must be smoke, full or confirm.')
    return result

