# BiMuon

BiMuon combines stochastic lower-level optimization and implicit hypergradient
estimation with matrix-valued upper-level updates. This repository contains the
exact-polar and finite Newton–Schulz variants, the proposed blockwise relative
stepsize rule, and the code needed to run them on the retained experiments.

## Install

Use Python 3.10 or later. Install a PyTorch build appropriate for your CPU or CUDA
runtime, then install the package from this directory:

```bash
python -m pip install -e ".[dev]"
# For CIFAR and Transformer tasks:
python -m pip install -e ".[vision]"
python -m pytest -q
```

The Transformer experiment used the versions in `requirements-transformer.txt`.
No experiment starts during installation or import.

## Algorithm

```python
from bimuon import BiMuon

# params: a list of matrix-valued UL parameters
optimizer = BiMuon(params, method="ns", lr=0.001, momentum=0.9, q=12)
optimizer.step(hypergradients)  # one implicit hypergradient per UL matrix
```

Use `method="exact"` for the supported SVD polar direction.
The optimizer expects hypergradients; it does not turn an ordinary single-level
training loop into a bilevel problem. `bimuon.core.estimate` implements the
warm-start LL solve and stochastic truncated-Neumann estimator for a regularized
linear classifier over differentiable features.

`bimuon.relative_stepsizes(a, w)` returns the proposed nonnegative block weights
under the unit theoretical LL-drift budget, using UL progress `a` and positive
LL sensitivities `w`. Zero progress in every block gives a zero update.

## Experiments

All commands run from the repository root. Each run requires a new output path.

| Task | Entry point | Configuration |
|---|---|---|
| Synthetic representation learning | `experiments.synthetic` | `configs/synthetic.json` |
| CIFAR-10 frozen-feature representation learning | `experiments.cifar10` | `configs/cifar10.json` |
| CIFAR-100 controlled blockwise allocation | `experiments.blockwise` | `configs/cifar100_blockwise.json` |
| CIFAR-100 DeiT-S/16 representation learning | `experiments.transformer.cli` | `configs/transformer.json` |

Start with a small CPU development run:

```bash
python -m experiments.synthetic --method ns --kappa 10 --seed 10 \
  --steps 2 --output work/synthetic-smoke
```

Remove `--steps 2` to use the saved full horizon. This small run checks execution;
it does not reproduce the reported full-horizon result.

See [Experiment instructions](docs/experiments.md) for data preparation and full
commands, and [Implementation notes](docs/implementation.md) for normalization,
support thresholds, evaluation, and the scope of this code-only release.

## Repository contents

- `bimuon/`: spectral UL updates, implicit hypergradient, and proposed allocation.
- `experiments/`: task definitions, data preparation, and explicit run commands.
- `configs/`: retained algorithm settings, evaluation seeds, and frozen stepsizes.
- `tests/`: numerical derivative, spectral, allocation, and checkpoint checks.
- `docs/`: experiment instructions and references.

Datasets and pretrained weights are obtained from their original providers.
Generated runs remain in the ignored `work/` directory. This code release does
not bundle datasets, checkpoints, run logs, historical search records, paper
sources, or implementations of comparison methods.

This is a private preparation repository. Public release and the distribution
license remain pending review; no open-source license is granted by this draft.
