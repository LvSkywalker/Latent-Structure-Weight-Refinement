"""Generate and plot a small native D³-LHSS example without TensorFlow."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits
from subspace_refinement import D3LHSS
from subspace_refinement.data import dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/quickstart.png'))
    args = parser.parse_args()
    case = dict(n=1000, d=2, seed=11, scenario='one_latent_direction')
    data = dataset(case, test_size=6000)
    with threadpool_limits(2):
        model = D3LHSS(n_centers=100, max_dim=2).fit(
            data.train_x[data.train_w > 0], data.train_x[data.train_w < 0])
        predicted = model.predict_refinement(data.test_x, data.mass_ratio)
    error = np.sqrt(np.mean((predicted - data.true_c)**2))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for mask, color, label in [(data.train_w > 0, '#52647a', 'positive events'),
                               (data.train_w < 0, '#d38b51', 'negative events')]:
        axes[0].scatter(*data.train_x[mask].T, s=9, alpha=.35, color=color, label=label)
    axes[0].set(xlabel='Observed x₁', ylabel='Observed x₂', title='Full-dimensional samples')
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].scatter(data.true_c[::10], predicted[::10], s=9, alpha=.4, color='#167b8e')
    axes[1].plot([.2, 1], [.2, 1], '--', color='#a5a9ae')
    axes[1].set(xlabel='Exact refinement c', ylabel='Estimated refinement c',
                title=f'Learned k={model.selected_dim_} · RMSE={error:.3f}')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f'Selected k={model.selected_dim_}; c-RMSE={error:.6f}; figure: {args.output}')


if __name__ == '__main__':
    main()
