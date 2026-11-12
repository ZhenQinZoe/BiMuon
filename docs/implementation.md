# Implementation notes

The retained kernels use the same LL-gradient, Hessian-vector, mixed-derivative,
and truncated-Neumann recurrences as the experiment code. Packaging removes
comparison algorithms, historical orchestration, and machine-specific metadata.

## Spectral transformations

The exact variant computes a supported polar factor. Transformer runs exclude
singular values at or below 1e-6 times the largest singular value; the original
tanh representation engine uses a relative threshold of 1e-14.

Finite NS uses the cubic iteration `O <- 1.5 O - 0.5 O O^T O`. CIFAR-10 and
Transformer runs initialize with `M / ||M||_F`, use 12 iterations, and evaluate
the smaller Gram matrix. Zero momentum maps to zero. Synthetic runs preserve
their historical capped initializer `M / max(1, ||M||_F)` and right-Gram
multiplication order. These choices are intentional and not interchangeable;
finite NS is not asserted to equal the exact polar factor.

`BiMuon.step` keeps one EMA matrix per UL matrix, with no coordinatewise second
moment, weight decay, scheduler, Nesterov correction, or automatic sensitivity
allocation. `relative_stepsizes` is separate: sensitivities must be positive,
progress scores nonnegative, and all-zero scores give a zero update.

## Checks and scope

Tests compare analytical derivatives with autograd, verify rectangular and
rank-deficient spectral cases, check the blockwise budget, and verify optimizer
checkpoint continuation. Such checks do not certify a convergence rate.
Full tuning archives and comparison implementations are outside this code-only
repository. Reusing evaluation seeds is reproduction, not independent
confirmation. No full-scale GPU training was rerun during packaging.

## References and third-party components

The implicit estimator follows stochastic bilevel optimization, including
stocBiO (Ji, Yang, and Liang, *Bilevel Optimization: Convergence Analysis and
Enhanced Design*, ICML 2021). Matrix transformations use Muon-style
orthogonalization. PyTorch provides tensors, autograd, linear algebra, and the
evaluation-only LBFGS solver.

Vision tasks use CIFAR-10/CIFAR-100, ImageNet, ResNet, and DeiT through torchvision
and timm. DeiT is described by Touvron et al., *Training data-efficient image
transformers & distillation through attention* (ICML 2021). Third-party datasets,
weights, and software retain their own attribution and terms; they are not
redistributed in this repository.
