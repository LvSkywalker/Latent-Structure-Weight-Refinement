# Confirmatory run on fresh seeds

The reduced-input neural comparison at 30 observed features was repeated on
five dataset seeds never used during development (131, 157, 173, 191, 211). Each seed was
evaluated at two sample sizes, 2,000 and 8,000 events, with new training,
validation and independent test samples and the unchanged protocol. The two
sample sizes share the seed and are not independent replications. Values are
independent-test c-RMSE, mean ± standard deviation over the five seeds;
lower is better.

| Events | Full-space neural refiner | Neural refiner on LHSS subspace | D³-LHSS | Covariance direction + kernel |
|---:|---:|---:|---:|---:|
| 2,000 | 0.298 ± 0.016 | 0.211 ± 0.014 | 0.359 ± 0.016 | 0.161 ± 0.024 |
| 8,000 | 0.177 ± 0.013 | 0.095 ± 0.009 | 0.114 ± 0.029 | 0.080 ± 0.003 |

Paired comparison of the two neural refiners (same architecture, same
training protocol, different input representation):

| Events | Seeds where LHSS subspace < all features | Mean relative change |
|---:|---:|---:|
| 2,000 | 5 of 5 | -29% |
| 8,000 | 5 of 5 | -46% |

This run is separate from the original 35-case study in `results/report.md`.
Its software environment differs from the original one and is recorded in
`run.json` and `requirements-confirm.txt`.

Reproduce with:

```sh
python -m subspace_refinement.benchmark --suite confirm --methods d3lhss nn_lhss nn_full covariance_kernel --output runs/confirm
python -m subspace_refinement.report --confirm runs/confirm/results.csv --output runs/confirm
```
