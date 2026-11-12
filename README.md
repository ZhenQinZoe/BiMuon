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

where the UL variable $X$ is a matrix and $g(X,\cdot)$ is $\mu$-strongly convex.
Each outer iteration $k$ of BiMuon (Algorithm 1 in the paper) does the following:

1. **Warm-started LL tracking.** Starting from the previous LL output
   ($Y_k^0 = Y_{k-1}^D$), take $D$ SGD steps
   $Y_k^{t+1} = Y_k^t - \alpha \nabla_Y G(X_k, Y_k^t; \mathcal S_{k,t})$.
2. **Stochastic hypergradient.** Estimate
   $\widehat{\nabla\Phi}(X_k) = \nabla_X F - \nabla^2_{XY} G\, v_Q$ at $(X_k, Y_k^D)$
   with a truncated Neumann series
   $v_Q = \eta \sum_{i=0}^{Q} \prod_{j=i+1}^{Q} (I - \eta \nabla^2_{YY} G)\, \nabla_Y F$,
   which needs $Q$ Hessian–vector products. Every oracle uses an independent
   fresh mini-batch.
3. **EMA momentum.** $M_k = \theta M_{k-1} + (1-\theta)\, \widehat{\nabla\Phi}(X_k)$.
4. **Orthogonalized UL step.** $X_{k+1} = X_k - \beta O_k$, where
   - Option I, **Exact-Polar** (`method="exact"`): $O_k = \mathrm{polar}(M_k) = UV^\top$;
   - Option II, **Finite-NS** (`method="ns"`): $O_k$ is $q$ cubic Newton–Schulz
     iterations $O \leftarrow 1.5\,O - 0.5\,OO^\top O$.

Both variants satisfy $\mathbb E\|\nabla\Phi(X_R)\|_* = O(T^{-1/4})$.

**Blockwise BiMuon.** With $X = (X^{(1)}, \dots, X^{(J)})$, each block is
orthogonalized separately and updated as
$X^{(b)}_{k+1} = X^{(b)}_k - \beta\, \omega_{b,k} O^{(b)}_k$.
The block LL sensitivity is $w_b = L_{XY,b}^2 r_b / \mu^2$ and the predicted UL
progress is $a_{b,k} = \langle M^{(b)}_k, O^{(b)}_k \rangle$. Maximizing
$\sum_b a_{b,k}\omega_{b,k}$ subject to the LL-drift budget
$\sum_b w_b \omega_{b,k}^2 \le 1$ gives

$$
\omega_{b,k} = \frac{a_{b,k} / w_b}{\sqrt{\sum_c a_{c,k}^2 / w_c}},
$$

so blocks with more UL progress move more, and blocks that move the LL target
more are held back. This is `bimuon.relative_stepsizes(a, w)`.

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

`lr` is the UL stepsize $\beta$ and `momentum` is $\theta$. `BiMuon` takes
hypergradients, not ordinary gradients. `estimate` implements steps 1–2 for an
L2-regularized linear classifier on top of a differentiable feature map.
`bimuon.relative_stepsizes(a, w)` returns the blockwise weights $\omega$.

## Experiments

Run every command from the repository root, and give each run a new `--output`
directory. Baselines (stocBiO+EMA, OpNorm, FrobNorm, stocBiO+Adam) are not
included.

| Paper experiment | Entry point | Config |
|---|---|---|
| CIFAR-10 representation learning, frozen ResNet-18 features (main text) | `experiments.cifar10` | `configs/cifar10.json` |
| CIFAR-100 blockwise stepsize allocation, frozen ResNet-50 features, $\rho\in\{1,3,10\}$ (main text) | `experiments.blockwise` | `configs/cifar100_blockwise.json` |
| Synthetic nonlinear representation learning, $\kappa_{\mathrm{in}}\in\{1,10,100,1000\}$ (appendix) | `experiments.synthetic` | `configs/synthetic.json` |
| Larger-scale Transformer: CIFAR-100 with DeiT-S/16, last 4 blocks as UL (appendix) | `experiments.transformer.cli` | `configs/transformer.json` |

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
