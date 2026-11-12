# BiMuon: A Geometry-Aware Stochastic Bilevel Optimization Approach

[![arXiv](https://img.shields.io/badge/arXiv-2611.10188-b31b1b.svg)](https://arxiv.org/abs/2611.10188)

Official code for **BiMuon: A Geometry-Aware Stochastic Bilevel Optimization Approach**
by Zhen Qin, Zhiyao Zhang, Yingbin Liang, and Jia Liu.

Most stochastic bilevel optimization (BLO) methods treat matrix-valued gradients
as flattened vectors, ignoring the matrix structure of modern neural networks.
In addition, the curvature of the lower-level (LL) problem can cause strong
spectral imbalance in the upper-level (UL) geometry. BiMuon combines implicit
hypergradient estimation with Muon-style orthogonalized updates at the UL.

- **Theory.** BiMuon is the first stochastic BLO method with Muon-style
  orthogonalized matrix updates and an $O(\epsilon^{-4})$ iteration-complexity
  guarantee. This matches the best-known rates of Muon-type methods based on
  matrix parameterizations.
- **Blockwise variant.** For models with many matrix parameters, a blockwise
  diagonal approximation runs the update per block. A relative stepsize rule
  allocates the step across blocks from each block's UL progress and LL
  sensitivity.
- **Experiments.** Representation learning (synthetic, CIFAR-10 with ResNet-18
  features, CIFAR-100 with DeiT-S/16) and blockwise bilevel optimization
  (CIFAR-100 with ResNet-50 features).

## Method

The bilevel problem is

$$
\min_{X}\; \Phi(X) = f\big(X, Y^\*(X)\big), \qquad
Y^\*(X) = \arg\min_{Y} g(X, Y),
$$

where every UL variable $X$ is a matrix and $g(X,\cdot)$ is strongly convex.
Each BiMuon iteration does the following:

1. **LL tracking.** Run $D$ stochastic gradient steps on $g(X_t,\cdot)$, starting
   from the previous $Y_t$.
2. **Hypergradient.** Estimate
   $\nabla\Phi \approx \nabla_X f - \nabla^2_{XY} g\, v$, with
   $v = \eta \sum_{q=0}^{Q} (I - \eta \nabla^2_{YY} g)^q \nabla_Y f$
   (a stochastic truncated Neumann series, with a fresh mini-batch for each
   oracle call).
3. **Momentum.** $M_t = \beta M_{t-1} + (1-\beta)\,\widehat{\nabla\Phi}_t$.
4. **Spectral UL step.** $X_{t+1} = X_t - \gamma\, \mathrm{Orth}(M_t)$, where
   `Orth` is either the exact polar factor $UV^\top$ (`method="exact"`) or $q$
   steps of cubic Newton–Schulz (`method="ns"`).

**Blockwise relative stepsizes.** When the UL matrix is split into blocks with
different LL sensitivities $w_b$, BiMuon scales each block's polar direction by
$\omega_b \propto a_b / w_b$, where $a_b = \langle M_b, \mathrm{Orth}(M_b)\rangle$
is the progress in block $b$. The weights are normalized so that
$\sum_b w_b \omega_b^2 = 1$, which keeps the theoretical LL drift fixed while
moving more in the blocks that are both informative and insensitive.

## Install

Requires Python 3.10 or later. Install a PyTorch build that matches your CPU or
CUDA setup, then run:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[vision]"   # CIFAR and Transformer experiments
python -m pytest -q
```

`requirements-transformer.txt` lists the exact versions used for the Transformer
experiment.

## Usage

```python
from bimuon import BiMuon, estimate

# params: list of matrix-valued UL parameters
optimizer = BiMuon(params, method="ns", lr=1e-3, momentum=0.9, q=12)

for step in range(T):
    hypergrads, y = estimate(features, params, y, streams, cfg)  # steps 1-2
    optimizer.step(hypergrads)                                    # steps 3-4
```

`BiMuon` takes hypergradients, not ordinary gradients. `estimate` implements
steps 1–2 for an L2-regularized linear classifier on top of a differentiable
feature map. `bimuon.relative_stepsizes(a, w)` returns the blockwise weights
$\omega$.

## Experiments

Run every command from the repository root, and give each run a new `--output`
directory.

| Task | Entry point | Config |
|---|---|---|
| Synthetic representation learning | `experiments.synthetic` | `configs/synthetic.json` |
| CIFAR-10, frozen ResNet-18 features | `experiments.cifar10` | `configs/cifar10.json` |
| CIFAR-100 blockwise allocation, frozen ResNet-50 features | `experiments.blockwise` | `configs/cifar100_blockwise.json` |
| CIFAR-100, DeiT-S/16 (last 4 blocks trained) | `experiments.transformer.cli` | `configs/transformer.json` |

Quick CPU check (2 iterations):

```bash
python -m experiments.synthetic --method ns --kappa 10 --seed 10 \
  --steps 2 --output work/synthetic-smoke
```

Drop `--steps` to run the full horizon. [docs/experiments.md](docs/experiments.md)
has data preparation and the full commands for every task, and
[docs/implementation.md](docs/implementation.md) covers numerical details
(polar thresholds, Newton–Schulz initialization, tests).

## Repository layout

```
bimuon/        BiMuon optimizer, hypergradient estimator, blockwise stepsizes
experiments/   task definitions, data preparation, run scripts
configs/       hyperparameters, seeds, and stepsizes used in the paper
tests/         derivative, spectral, allocation, and checkpoint tests
docs/          experiment instructions and implementation notes
```

Datasets and pretrained weights are downloaded from their original sources and
are not redistributed here. Run outputs go to `work/`, which Git ignores.

## Citation

```bibtex
@article{qin2026bimuon,
  title   = {BiMuon: A Geometry-Aware Stochastic Bilevel Optimization Approach},
  author  = {Qin, Zhen and Zhang, Zhiyao and Liang, Yingbin and Liu, Jia},
  journal = {arXiv preprint arXiv:2611.10188},
  year    = {2026}
}
```
