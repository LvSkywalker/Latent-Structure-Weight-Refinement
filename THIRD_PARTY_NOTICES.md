# Credits and source provenance

## Neural weight refinement

This independent synthetic study is inspired by Benjamin Nachman and Dennis
Noll, **Stay Positive: Neural Refinement of Sample Weights** (2025),
[arXiv:2505.03724](https://arxiv.org/abs/2505.03724).

The Gaussian setup and neural architecture are adapted from
[`Nollde/weight_refiner`](https://github.com/Nollde/weight_refiner), inspected at
commit `4b3a41ea9ff45d1922068387b04c7c694ce1338a`.
Copyright (c) 2025 Dennis Noll. Its MIT license is retained verbatim in
[`third_party/weight_refiner_LICENSE.txt`](third_party/weight_refiner_LICENSE.txt).
The local package implements the generator directly and supplies its own
event-level split, training loop, validation and analysis. It does not need
an upstream checkout. The tested inputs match that snapshot exactly.

## D³-LHSS

The density-ratio method is due to Masashi Sugiyama, Makoto Yamada, Paul von
Bünau, Taiji Suzuki, Takafumi Kanamori and Motoaki Kawanabe:
**Direct Density-ratio Estimation with Dimensionality Reduction via
Least-squares Hetero-distributional Subspace Search**, *Neural Networks* 24
(2011), 183–198. [Paper](https://www.ms.k.u-tokyo.ac.jp/sugi/2011/LHSS.pdf).

This repository supplies a native NumPy/SciPy implementation of the method,
using the numerical protocol of the authors' public reference software.
That software was inspected and used locally for numerical comparison.
The Python implementation is neither an official release nor an endorsement
by the original authors. The original MATLAB source and example arrays are
not included. Their software page requires permission for redistribution:
[reference software and terms](https://www.ms.k.u-tokyo.ac.jp/sugi/software.html).

Reference archive SHA-256:
`64fe900faa1b2e573489ac4772c8595b93cc0ab7aad3303525df906bc6d7391c`.
Numerical parity measurements, containing no reference source or example
arrays, are retained in `results/validation/`.

## Scope

All event samples and model centers are synthetic. This is an independent
study, not affiliated with or endorsed by the authors of the works above.
The repository's MIT license covers its distributed implementation and accompanying original
material; it does not relicense separately obtained reference software.
