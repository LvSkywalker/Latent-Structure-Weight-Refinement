"""Reproducible benchmark CLI with original-event validation and resumable runs."""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import tempfile
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .data import cases, dataset
from .baselines import covariance_direction, fit_kernel
from .lhss import D3LHSS

METHODS = ('d3lhss', 'nn_full', 'kernel_full', 'covariance_kernel', 'nn_lhss')


def metrics(prediction, data):
    residual = prediction - data.true_c
    ratio = (1 - prediction)/np.maximum(1 + prediction, 1e-12)
    return {'c_rmse': float(np.sqrt(np.mean(residual**2))),
            'r_rmse': float(np.sqrt(np.mean((ratio - data.true_ratio)**2))),
            'normalization_bias_over_mass': float(np.mean(residual)/.5),
            'negative_refined_fraction': float(np.mean(prediction < 0))}


def environment():
    packages = {}
    for name in ('numpy', 'scipy', 'pandas', 'matplotlib', 'threadpoolctl', 'tensorflow', 'keras'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}


def atomic_csv(rows, path):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.csv', delete=False) as file:
        pd.DataFrame(rows).to_csv(file, index=False)
        name = Path(file.name)
    name.replace(path)


def model_path(directory, case):
    return Path(directory)/'models'/f"{case['id']}.npz"


def fit_lhss(data, case, directory):
    path = model_path(directory, case)
    if path.exists():
        return D3LHSS.load(path)
    model = D3LHSS(max_dim=int(case['d']), random_state=0).fit(
        data.train_x[data.train_w > 0], data.train_x[data.train_w < 0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    model.save(temporary)
    temporary.replace(path)
    return model


def evaluate(case, method, directory, data=None):
    data = dataset(case) if data is None else data
    settings = {}
    started = time.perf_counter()
    if method == 'd3lhss':
        model = fit_lhss(data, case, directory)
        prediction = model.predict_refinement(data.test_x, data.mass_ratio)
        normalized = model.predict_ratio(data.test_x)
        valid_ratio = model.predict_ratio(data.valid_x)
        valid_positive = data.valid_w > 0
        n_positive = int(.75*len(data.test_x))
        settings = {'selected_k': model.selected_dim_, 'internal_cv_score': model.cv_score_,
                    'external_validation_risk': float(.5*np.mean(valid_ratio[valid_positive]**2)
                                                      - np.mean(valid_ratio[~valid_positive])),
                    'independent_ratio_risk': float(.5*np.mean(normalized[:n_positive]**2)
                                                     - np.mean(normalized[n_positive:])),
                    'sigma': model.sigma_, 'ridge': model.ridge_,
                    'alignment_squared': float(np.sum((model.projection_.T@data.rotation[:, 0])**2))}
    elif method in ('kernel_full', 'covariance_kernel'):
        projection = None if method == 'kernel_full' else covariance_direction(data.train_x, data.train_w)
        model, settings = fit_kernel(data, int(case['seed']), projection)
        prediction = model.predict_refinement(data.test_x)
        settings['selected_k'] = int(case['d']) if projection is None else 1
    elif method in ('nn_full', 'nn_lhss'):
        try:
            from .neural import fit_network, predict_refinement
            projection = None if method == 'nn_full' else fit_lhss(data, case, directory).projection_
            model, settings = fit_network(data, int(case['seed']), projection)
            prediction = predict_refinement(model, data.test_x, projection)
        except ModuleNotFoundError as error:
            if error.name == 'tensorflow':
                raise RuntimeError('Install the neural extra: pip install ".[neural]"') from error
            raise
        settings['selected_k'] = int(case['d']) if projection is None else projection.shape[1]
    else:
        raise ValueError(f'Unknown method: {method}')
    if not np.isfinite(prediction).all():
        raise ValueError(f'Non-finite predictions for {case["id"]}/{method}.')
    return {**case, 'method': method, **metrics(prediction, data), **settings,
            'seconds': time.perf_counter() - started}


def initialize_run(directory, suite):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = {'protocol': 'subspace-refinement-v1', 'suite': suite, 'cases': cases(suite),
              'lhss_random_state': 0, 'lhss_rng': 'NumPy PCG64', 'test_events': 24000,
              'train_fraction': .8, 'model_selection_uses_test_truth': False}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    target = directory/'run.json'
    if target.exists():
        previous = json.loads(target.read_text())
        if previous['configuration_sha256'] != signature:
            raise ValueError('This output directory contains a different protocol. Choose another --output.')
    else:
        source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in Path(__file__).parent.glob('*.py')}
        target.write_text(json.dumps({**config, 'configuration_sha256': signature,
                                      'environment': environment(), 'source_sha256': source_hashes}, indent=2)+'\n')
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    directory = initialize_run(args.output or Path('runs')/args.suite, args.suite)
    path = directory/'results.csv'
    rows = pd.read_csv(path).to_dict('records') if path.exists() else []
    done = {(row['id'], row['method']) for row in rows}
    with threadpool_limits(2):
        for case in cases(args.suite):
            data = dataset(case)
            for method in args.methods:
                if (case['id'], method) in done:
                    continue
                print(f"Running {case['id']} / {method}", flush=True)
                row = evaluate(case, method, directory, data)
                rows.append(row)
                atomic_csv(rows, path)
                print(json.dumps(row), flush=True)
    print(f'Results: {path}', flush=True)


if __name__ == '__main__':
    main()
