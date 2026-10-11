# Experimental protocol

## Question

Shared physical mechanisms motivate the hypothesis that the information
needed for weight refinement might admit a compact latent representation.
A low-dimensional manifold is one possibility, but neither its existence
nor sufficiency for the ratio follows from a common physical origin alone.
The concrete target here is R(x) approximately equal to g(T(x)), where T(x)
has fewer coordinates than x and preserves information relevant to the ratio.

Can learning a representation sufficient for a density ratio improve the
refinement of signed Monte Carlo weights in a finite-sample setting?
The proposed application is weight refinement. D³-LHSS itself is an existing
method; this project does not claim a new dimensionality-reduction algorithm.
We investigate learned linear subspaces on synthetic data, rather than
discovering a nonlinear manifold in physical event data.

For unit absolute event weights and a positive-label posterior p(x), the
refinement factor is c(x)=2p(x)-1. Equivalently, write

R(x) = [negative mass / positive mass] × p_negative(x)/p_positive(x),

c(x) = [1-R(x)]/[1+R(x)].

The estimators learn the normalized ratio (kernels) or the posterior (networks).
The empirical training mass ratio is used for conversion of kernel predictions.
The exact population mass ratio of 1/3 is used only for evaluating ground truth.

## Synthetic distributions

**One informative linear direction.** 75% of events have weight +1 and
z~N(0,1); 25% have weight -1 and z~N(0,0.5²). Append d-1 independent N(0,1)
coordinates shared by the two classes, then rotate all coordinates by a
seeded orthogonal matrix. The exact weighted ratio is
R(z)=(2/3)exp(-1.5z²). All coordinates can vary: the samples do not lie on a
one-dimensional manifold. The ratio is sufficient in one linear direction,
which is deliberately built into this benchmark. Here c≥0.2 everywhere.

**Radial control.** Positive samples follow N(0,I), negative samples
N(0,0.85²I), in ten dimensions with the same 75%/25% mixture. The ratio depends
on the full squared radius. No linear projection with k<10 preserves it
exactly; a nonlinear radial feature would be sufficient. In a central region
R>1 and c<0. This is a density-ratio/compression control, not a demonstration
that a locally negative signed measure can be exactly made positive.

## Samples and splits

| Setting | Values |
|---|---|
| Total development events | 2,000 and 8,000 |
| Observed features | 1, 10 and 30 |
| Dataset seeds | 11, 29, 47, 71, 101 |
| External split | 80% training / 20% validation, original events |
| Independent test events | 24,000 per case |
| Radial control | 8,000 events, 10 features, same five seeds |

There are 35 dataset cases. Test samples use seed+200000 and are transformed
with the training rotation. They contain 18,000 positive and 6,000 negative
events, matching the population mixture. Test truth is never used for model
selection. All input event weights have absolute value one. Nonuniform
within-class weights would require a corresponding extension of the ratio
estimators, beyond multiplying the final mass ratio.

## Methods

| Method | Representation | Selection |
|---|---|---|
| `d3lhss` | Learned linear k-dimensional subspace | Reference internal CV; all k=1,…,d |
| `nn_full` | All observed coordinates | External validation BCE checkpoint |
| `kernel_full` | All observed coordinates | External validation uLSIF risk |
| `covariance_kernel` | Leading eigenvector of class covariance difference | External validation uLSIF risk |
| `nn_lhss` | Frozen native D³-LHSS subspace | External validation BCE checkpoint |

The covariance baseline is important: our Gaussian construction has a rank-one
covariance difference, so this simple estimator is especially well matched.
The dimension-one assumption is provided to that baseline, but not to D³-LHSS.

### Complete native D³-LHSS

100 negative-class kernel centers; two-fold uLSIF CV at each update; kernel
widths at 0.75, 1 and 1.25 times the reference median-distance scale; nine
ridge values log-spaced between 0.001 and 10. Negative coefficients are
truncated to zero. Orthogonal subspace updates use a skew-symmetric matrix
exponential, a decreasing 200-step schedule, and the reference stopping rule.
Every candidate k is optimized; the minimum internal CV score selects k;
the selected dimension is refitted with the same deterministic random choices.
The distributed software's default maximum k is five; this study explicitly
extends that parameter to the observed dimension.

Initialization retains the full QR of an all-ones matrix. NumPy PCG64 is used
with internal seed zero. Its permutations differ from MATLAB's legacy RNG;
rank-deficient QR completions can also be library-dependent. The native suite
is a separate replication. MATLAB and Python parity is checked with matching
random choices and initial bases on four local reference cases.

The reference gradient and divergence/stopping conventions are retained,
including coefficient clipping. The independent finite-difference test
checks the analytic partial derivative with coefficients held fixed; it does
not assert differentiability through the clipping operation.

### Neural comparisons

Two hidden layers of 128 ReLU units, sigmoid output, Adam with learning rate
0.001, weighted binary cross-entropy, batch size 128, maximum 100 epochs.
Restore the checkpoint with minimum external validation BCE; stop after
20 non-improving epochs. Seeds are set per dataset. For these unit-weight
synthetics, sample weights in the loss are all one.

This preserves the upstream architecture, with an explicitly stated local
training budget. It is not a reproduction of the paper's large-data training
or a search for the strongest possible neural baseline. The original notebook's
10-epoch schedule would give far fewer updates on these small datasets, so
it is not used as the headline comparison.

### Simple kernel comparisons

64 numerator centers; five bandwidth factors (0.15, 0.3, 0.6, 1.2, 2.4)
relative to a training-distance scale; ridge values 1e-5, 1e-4, 1e-3, 1e-2,
1e-1. These baselines clip predicted ratios at zero, rather than clipping
coefficients. Hyperparameters minimize the external validation uLSIF risk.
They differ from D³-LHSS in both representation and selection procedure;
differences are not attributed exclusively to dimensionality.

## Outcomes and interpretation

Primary metric: RMSE of c on the independent test mixture. CSVs also record
weighted ratio RMSE, normalization bias and the fraction of negative refined
weights. Report means and sample standard deviations over the five dataset
seeds; these error bars are not confidence intervals. Model RNG seed zero
is held fixed, so this does not quantify variation over optimizer seeds.

The `nn_lhss` head is an exploratory diagnostic motivated by earlier MATLAB
and pilot results. It shares the existing test sets and is not a confirmatory
experiment on fresh data. A lower error here cannot establish kernel
superiority; the comparison helps assess representation versus final estimator.

Internal CV scores can be optimistic because the projection is optimized
adaptively on the same training samples. This protocol retains that behavior
for fidelity, rather than adding nested refitting per fold. Comparing its
risk to a completely independent test risk helps expose this limitation.
The risk comparison alone does not isolate a unique cause.

Reported seconds are local wall-clock diagnostics, not a hardware-controlled
speed benchmark. Subspace heads can reuse saved projections; their timings
do not include the projection's original training cost.

No physical event dataset is evaluated. There is no resampling, uncertainty
preservation or coverage study. Generalization to collider events, nonlinear
sufficient representations and nonuniform weights remains future work.
