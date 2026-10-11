# Validation and code audit

## Native numerical implementation

The complete native estimator was checked locally against the downloaded
authors' LHSS software on its `sep`, `les` and `dep` examples, plus our
2,000-event/10-feature synthetic case. Random permutations and initial bases
were matched to isolate numerical implementation differences. All selected
dimensions agree. The largest absolute discrepancy in the predicted
normalized density ratio is approximately **6.31×10⁻¹⁴**.

The comparison also checks Gaussian features, fitted uLSIF coefficients,
analytic gradients and dimension CV scores. Numerical results are in
[`matlab_parity.json`](../results/validation/matlab_parity.json).
The original MATLAB source, private replay fixtures and third-party example
arrays are excluded from this repository.

Ordinary Python fits use NumPy PCG64. They have their own recorded results;
parity under matched choices is not a claim that different RNG streams
produce identical fitted models. Full QR of the rank-deficient initial
all-ones matrix is another possible source of platform variation.

The local tests run without MATLAB:

```sh
python -m unittest discover -s tests -v
```

They check the analytic gradient against central finite differences, disjoint
event IDs in every benchmark split, the one-direction population truth,
native fitting/orthogonality, finite nonnegative ratios and model save/load.
The gradient test holds coefficients fixed, matching the derivative being
implemented; it does not differentiate through coefficient clipping.

## Generator and neural protocol

Every regenerated training, validation and independent-test matrix was
compared to the corresponding input from the preceding local study: all
35 cases match exactly. The packaged neural training loop reproduces the
preceding Python loop's independent c-RMSE exactly on all 35 cases in the
tested environment. The final release includes fresh native evaluations.

## Event overlap in the inspected reweighter

The public upstream
[`prepare_data_reweighter`](https://github.com/Nollde/weight_refiner/blob/4b3a41ea9ff45d1922068387b04c7c694ce1338a/refiner/model.py#L12)
duplicates each original feature vector to form the two terms of its loss.
If the random holdout is made after that duplication, copies of a single
event can enter both partitions. Independently choosing plot-validation
events from the original arrays can also reuse training events.

This repository includes a small event-ID illustration:

```sh
python -m subspace_refinement.audit
```

With 2,000 original events and seed 42, duplicate-then-split shares 634 unique
event IDs across training and validation; 373 of the 400 natural validation
events occur in the duplicated training set. Splitting the original IDs first
and duplicating within each partition yields zero shared IDs. Exact output
is retained in [`event_overlap.json`](../results/validation/event_overlap.json).

This observation concerns the inspected reweighter's split construction.
It does not establish the magnitude of performance inflation, invalidate
the refiner, or demonstrate an error in the published physics conclusions.
The present benchmark splits original events before all model fitting and
uses independently generated test events throughout.

## Numerical refinement conversion

For a sigmoid posterior p, use c=2p-1 directly. This avoids forming
(1/p)-1, which is infinite at p=0 even though c=-1 is well defined. The neural
comparisons use this stable form. No claim is made about how frequently exact
zeros occur in previously published models.
