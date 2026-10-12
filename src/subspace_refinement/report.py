"""Create transparent result tables and scientific figures from per-case CSVs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import cases

STYLE = {
    'd3lhss': ('D³-LHSS', '#127d8e', '-'),
    'nn_full': ('Full-space neural refiner', '#435575', '-'),
    'covariance_kernel': ('Covariance direction + kernel', '#d08c48', '--'),
    'kernel_full': ('Full-space kernel', '#969ca6', ':'),
    'nn_lhss': ('Neural refiner on LHSS subspace', '#9165a3', '-'),
}


def theme():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#c2c7ce', 'axes.labelcolor': '#293548',
                         'text.color': '#293548', 'figure.facecolor': '#fbfcfd',
                         'axes.facecolor': '#fbfcfd', 'savefig.facecolor': '#fbfcfd'})


def banner(path):
    """Draw the research pipeline as a compact, neutral vector graphic."""
    theme()
    fig, ax = plt.subplots(figsize=(12, 1.65))
    fig.patch.set_facecolor('#f6f8fa')
    ax.set_facecolor('#f6f8fa')
    ax.set(xlim=(0, 12), ylim=(0, 1.65))
    ax.axis('off')
    columns = ((.45, '01', 'Observed events', 'Many measured variables'),
               (4.45, '02', 'Learned subspace', 'Keep information for the ratio'),
               (8.45, '03', 'Refined weights', 'Estimate R(x), then c(x)'))
    for x, number, title, subtitle in columns:
        ax.text(x, 1.19, number, fontsize=10, weight='bold', color='#127d8e')
        ax.text(x, .77, title, fontsize=16, weight='bold', color='#1f2937')
        ax.text(x, .35, subtitle, fontsize=10, color='#57606a')
    for x in (3.72, 7.72):
        ax.annotate('', xy=(x+.35, .81), xytext=(x-.2, .81),
                    arrowprops={'arrowstyle': '->', 'color': '#8c959f', 'lw': 1.3})
    fig.subplots_adjust(left=.01, right=.99, top=.98, bottom=.02)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format=path.suffix[1:], facecolor=fig.get_facecolor())
    plt.close(fig)


def tables(rows):
    return rows.groupby(['scenario', 'n', 'd', 'method']).c_rmse.agg(['mean', 'std', 'count'])


def value(summary, scenario, n, d, method):
    try:
        row = summary.loc[(scenario, n, d, method)]
    except KeyError:
        return '—'
    if row['count'] > 1:
        return f"{row['mean']:.3f} ± {row['std']:.3f}"
    return f"{row['mean']:.3f} (one run)"


def comparison_table(rows, summary):
    methods = [m for m in STYLE if m in set(rows.method) and m != 'nn_lhss']
    result = '| Events | Features | ' + ' | '.join(STYLE[m][0] for m in methods) + ' |\n'
    result += '|---:|---:|' + '---:|'*len(methods) + '\n'
    settings = rows[rows.scenario == 'one_latent_direction'][['n', 'd']].drop_duplicates().sort_values(['n', 'd'])
    for n, d in settings.itertuples(index=False, name=None):
        result += f'| {n:,} | {d} | ' + ' | '.join(value(summary, 'one_latent_direction', n, d, m) for m in methods) + ' |\n'
    return result


def plots(rows, directory):
    theme()
    directory.mkdir(parents=True, exist_ok=True)
    main = rows[rows.scenario == 'one_latent_direction']
    ns = sorted(main.n.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(6*len(ns), 4.8), squeeze=False, sharey=True)
    for ax, n in zip(axes[0], ns):
        for method in ('d3lhss', 'nn_full', 'covariance_kernel', 'kernel_full'):
            selected = main[(main.n == n) & (main.method == method)]
            if selected.empty:
                continue
            stats = selected.groupby('d').c_rmse.agg(['mean', 'std'])
            label, color, line = STYLE[method]
            ax.errorbar(stats.index, stats['mean'], yerr=stats['std'].fillna(0), marker='o',
                        color=color, linestyle=line, capsize=3, linewidth=2, label=label)
        ax.set(title=f'{n:,} development events', xlabel='Observed features', xticks=[1, 10, 30])
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', color='#e6e9ee')
    axes[0, 0].set_ylabel('Independent test RMSE of refinement c')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=9)
    title = 'Native Python benchmark: one informative linear direction' if len(ns) > 1 else 'Native Python: one informative direction'
    fig.suptitle(title, fontsize=14, y=.98)
    fig.text(.5, .15, 'Mean ± sample standard deviation across dataset seeds; lower is better',
             ha='center', color='#657183', fontsize=9)
    fig.tight_layout(rect=[0, .20, 1, .92])
    fig.savefig(directory/'benchmark.png', dpi=170)
    plt.close(fig)

    selected = rows[rows.method == 'd3lhss'].sort_values(['scenario', 'n', 'd', 'seed'])
    if not selected.empty:
        matrix = selected.pivot(index=['scenario', 'n', 'd'], columns='seed', values='selected_k')
        signal = matrix.loc[matrix.index.get_level_values(0) == 'one_latent_direction']
        control = matrix.loc[matrix.index.get_level_values(0) != 'one_latent_direction']
        matrix = pd.concat([signal, control])
        fig, ax = plt.subplots(figsize=(8, 4.8))
        image = ax.imshow(matrix, cmap='YlGnBu', vmin=1, vmax=max(2, np.nanmax(matrix)), aspect='auto')
        ax.set_xticks(range(len(matrix.columns)), [str(s) for s in matrix.columns])
        labels = [f'{n:,} events / {d}D' + (' · radial control' if scenario != 'one_latent_direction' else '')
                  for scenario, n, d in matrix.index]
        ax.set_yticks(range(len(matrix)), labels)
        ax.set_xlabel('Dataset seed')
        for (i, j), k in np.ndenumerate(matrix.to_numpy()):
            if np.isfinite(k):
                ax.text(j, i, str(int(k)), ha='center', va='center',
                        color='white' if k > (np.nanmax(matrix) + 1)/2 else '#293548', weight='bold')
        fig.colorbar(image, ax=ax, label='Selected subspace dimension k')
        ax.set_title('Dimension selection varies across independent samples', pad=14)
        fig.tight_layout()
        fig.savefig(directory/'dimension_selection.png', dpi=170)
        plt.close(fig)

    diagnostic = main[(main.d == 30) & main.method.isin(['d3lhss', 'nn_full', 'nn_lhss'])]
    if 'nn_lhss' in set(diagnostic.method):
        ns = sorted(diagnostic.n.unique())
        fig, ax = plt.subplots(figsize=(8, 4.7))
        for offset, method in zip([-.25, 0, .25], ['nn_full', 'd3lhss', 'nn_lhss']):
            stats = diagnostic[diagnostic.method == method].groupby('n').c_rmse.agg(['mean', 'std']).reindex(ns)
            ax.bar(np.arange(len(ns)) + offset, stats['mean'], width=.23,
                   yerr=stats['std'].fillna(0), capsize=3, color=STYLE[method][1], label=STYLE[method][0])
        ax.set_xticks(range(len(ns)), [f'{n:,} events' for n in ns])
        ax.set_ylabel('Independent test RMSE of c')
        ax.set_title('Neural refinement after LHSS (30 features)')
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.1), ncol=1, fontsize=9)
        ax.grid(axis='y', color='#e6e9ee')
        ax.set_axisbelow(True)
        fig.tight_layout(rect=[0, .12, 1, 1])
        fig.savefig(directory/'subspace_head.png', dpi=170)
        plt.close(fig)


def report(rows, output):
    if rows.duplicated(['id', 'method']).any():
        raise ValueError('Duplicate case/method rows in the input.')
    if not np.isfinite(rows.c_rmse).all():
        raise ValueError('Non-finite primary errors in the input.')
    summary = tables(rows)
    summary.to_csv(output/'summary.csv')
    plots(rows, output/'figures')
    main_table = comparison_table(rows, summary)
    selected = rows[rows.method == 'd3lhss']
    dimension_table = '| Scenario | Events | Features | Selected k, ordered by dataset seed |\n|---|---:|---:|---|\n'
    for (scenario, n, d), part in selected.groupby(['scenario', 'n', 'd']):
        values = ', '.join(str(int(k)) for k in part.sort_values('seed').selected_k)
        dimension_table += f'| {scenario} | {n:,} | {d} | {values} |\n'
    diagnostic_table = '| Method, 30 observed features | 2,000 events | 8,000 events |\n|---|---:|---:|\n'
    for method in ('nn_full', 'd3lhss', 'nn_lhss'):
        if method in set(rows.method):
            diagnostic_table += f'| {STYLE[method][0]} | ' + ' | '.join(
                value(summary, 'one_latent_direction', n, 30, method) for n in (2000, 8000)) + ' |\n'
    control_table = '| Method | Radial-control RMSE |\n|---|---:|\n'
    for method in STYLE:
        if method in set(rows.method):
            control_table += f'| {STYLE[method][0]} | {value(summary, "all_dimensions_matter", 8000, 10, method)} |\n'
    risk_table = '| Events | Features | Internal CV risk | Independent test risk |\n|---:|---:|---:|---:|\n'
    for (n, d), part in selected[selected.scenario == 'one_latent_direction'].groupby(['n', 'd']):
        risk_table += f'| {n:,} | {d} | {part.internal_cv_score.mean():.3f} | {part.independent_ratio_risk.mean():.3f} |\n'
    dimension_image = ('![Dimension selection](figures/dimension_selection.png)'
                       if (output/'figures/dimension_selection.png').exists()
                       else 'No LHSS dimension-selection results were included in this run.')
    diagnostic_image = ('![Neural refiner on the LHSS subspace](figures/subspace_head.png)'
                        if (output/'figures/subspace_head.png').exists()
                        else 'The neural-subspace diagnostic was not included in this run.')
    text = f'''# Native Python results

{rows.id.nunique()} dataset cases; {len(rows)} method evaluations.
All development and test samples are synthetic. Test truth is not used for
selection. Tables show mean ± sample standard deviation over dataset seeds;
the per-case CSV records every evaluation and its selected settings.

## Full-space and reduced-space comparison

{main_table}

![Benchmark](figures/benchmark.png)

The covariance baseline uses a supplied one-dimensional representation family
and is particularly well matched to this Gaussian rank-one covariance shift.
The learned method must also select its dimension and uses a different
selection protocol. The comparison does not isolate dimensionality alone.

## Selected dimensions

{dimension_table}

{dimension_image}

The true sufficient linear dimension is one in the main setting and ten in
the radial control. Choosing k>1 in the main setting can preserve the ratio;
the independent error, rather than k alone, determines whether it generalized.

## Neural refiner on the learned subspace

{diagnostic_table}

{diagnostic_image}

The learned subspace and selected k are frozen. Only the final estimator is
changed and selected on the external event holdout. This comparison was
designed after the first runs and reuses these test sets, so it was repeated
on fresh dataset seeds: see `results/confirm/report.md`. It supports a
representation benefit without demonstrating a unique advantage of kernel
estimation.

## Radial control

{control_table}

The radial control tests linear information loss. Its exact c is negative in
a central region, so it is not a demonstration of an everywhere nonnegative
physical refinement. A nonlinear radial feature would be sufficient.

## Internal selection versus independent risk

{risk_table}

Both columns use the normalized-ratio uLSIF risk (lower is better). Adaptive
reuse during subspace search can make internal scores optimistic. The
comparison does not identify a unique failure mechanism. No nested refitting
of the complete subspace search was added, to retain the reference protocol.

## Limits

- The favorable sufficient direction was constructed, not discovered in
  physical data. There is no collider-data evaluation.
- Five dataset seeds do not establish universal superiority. Error bars are
  standard deviations, not confidence intervals; optimizer RNG is fixed.
- Neural and kernel models have different selection budgets and regularization.
- Nonuniform absolute weights, resampling and uncertainty coverage are not tested.
- Local timings can include cached projections and are not a speed benchmark.

See the repository's `docs/protocol.md` and `docs/validation.md` for full details.
'''
    (output/'report.md').write_text(text)
    return main_table


CONFIRM_METHODS = ('d3lhss', 'nn_lhss', 'nn_full', 'covariance_kernel')
CONFIRM_OPTIONAL = ('kernel_full',)
CONFIRM_COLUMNS = ('id', 'n', 'd', 'seed', 'scenario', 'method', 'c_rmse')


def validate_confirm(rows):
    """Check that rows are exactly one complete confirm-suite run; raise ValueError otherwise."""
    missing = [c for c in CONFIRM_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f'Confirm results lack required columns: {", ".join(missing)}.')
    if rows.empty:
        raise ValueError('Confirm results are empty.')
    duplicated = rows[rows.duplicated(['id', 'method'], keep=False)]
    if not duplicated.empty:
        pairs = sorted(set(zip(duplicated.id, duplicated.method)))
        raise ValueError(f'Duplicate case/method rows in confirm results: {pairs[:3]}.')
    expected = {case['id']: case for case in cases('confirm')}
    unknown = sorted(set(rows.id) - set(expected))
    if unknown:
        raise ValueError(f'Rows do not belong to the confirm suite: {unknown[:3]}.')
    absent = sorted(set(expected) - set(rows.id))
    if absent:
        raise ValueError(f'Confirm results are incomplete; missing cases: {absent[:3]} '
                         f'({len(absent)} of {len(expected)}).')
    for row in rows.itertuples(index=False):
        case = expected[row.id]
        for key in ('n', 'd', 'seed', 'scenario'):
            if getattr(row, key) != case[key]:
                raise ValueError(f'Case {row.id} has {key}={getattr(row, key)!r}, '
                                 f'expected {case[key]!r}.')
    allowed = set(CONFIRM_METHODS) | set(CONFIRM_OPTIONAL)
    extra = sorted(set(rows.method) - allowed)
    if extra:
        raise ValueError(f'Unexpected methods in confirm results: {extra}.')
    for case_id in expected:
        present = set(rows.loc[rows.id == case_id, 'method'])
        lacking = [m for m in CONFIRM_METHODS if m not in present]
        if lacking:
            raise ValueError(f'Case {case_id} lacks confirm methods: {lacking}.')
    values = pd.to_numeric(rows.c_rmse, errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('c_rmse values must be finite and non-negative.')
    full = rows.loc[rows.method == 'nn_full', 'c_rmse'].astype(float)
    if not (full > 0).all():
        raise ValueError('nn_full c_rmse must be strictly positive for relative comparisons.')


def confirm_report(rows, output, source=None):
    """Summarize the fresh-seed run of the reduced-input neural comparison.

    All checks run before anything is written. The validated rows, including any
    optional kernel_full results, are copied to output unless that is the source.
    """
    validate_confirm(rows)
    validated = rows
    rows = rows[rows.method.isin(CONFIRM_METHODS)]
    methods = ['nn_full', 'nn_lhss', 'd3lhss', 'covariance_kernel']
    table = '| Events | ' + ' | '.join(STYLE[m][0] for m in methods) + ' |\n|---:|' + '---:|'*len(methods) + '\n'
    paired = ('| Events | Seeds where LHSS subspace < all features | Mean relative change |\n'
              '|---:|---:|---:|\n')
    for n, part in rows.groupby('n'):
        cells = []
        for m in methods:
            v = part[part.method == m].c_rmse
            cells.append(f'{v.mean():.3f} ± {v.std(ddof=1):.3f}')
        table += f'| {n:,} | ' + ' | '.join(cells) + ' |\n'
        full = part[part.method == 'nn_full'].set_index('seed').c_rmse
        sub = part[part.method == 'nn_lhss'].set_index('seed').c_rmse.reindex(full.index)
        wins = int((sub < full).sum())
        change = float((sub/full - 1).mean())
        paired += f'| {n:,} | {wins} of {len(full)} | {100*change:+.0f}% |\n'
    seeds = ', '.join(str(s) for s in sorted(rows.seed.unique()))
    text = f'''# Confirmatory run on fresh seeds

The reduced-input neural comparison at 30 observed features was repeated on
five dataset seeds never used during development ({seeds}). Each seed was
evaluated at two sample sizes, 2,000 and 8,000 events, with new training,
validation and independent test samples and the unchanged protocol. The two
sample sizes share the seed and are not independent replications. Values are
independent-test c-RMSE, mean ± standard deviation over the five seeds;
lower is better.

{table}
Paired comparison of the two neural refiners (same architecture, same
training protocol, different input representation):

{paired}
This run is separate from the original 35-case study in `results/report.md`.
Its software environment differs from the original one and is recorded in
`run.json` and `requirements-confirm.txt`.

Reproduce with:

```sh
python -m subspace_refinement.benchmark --suite confirm --methods d3lhss nn_lhss nn_full covariance_kernel --output runs/confirm
python -m subspace_refinement.report --confirm runs/confirm/results.csv --output runs/confirm
```
'''
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'report.md').write_text(text)
    target = output/'results.csv'
    if source is None or Path(source).resolve() != target.resolve():
        validated.to_csv(target, index=False)
    return table + '\n' + paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/results.csv'))
    parser.add_argument('--output', type=Path, default=Path('runs/report'))
    parser.add_argument('--confirm', type=Path, help='results.csv of the fresh-seed confirm suite')
    args = parser.parse_args()
    if args.confirm:
        print(confirm_report(pd.read_csv(args.confirm), args.output, source=args.confirm))
        print(f'Report: {args.output/"report.md"}')
        return
    args.output.mkdir(parents=True, exist_ok=True)
    print(report(pd.read_csv(args.input), args.output))
    print(f'Report: {args.output/"report.md"}')


if __name__ == '__main__':
    main()
