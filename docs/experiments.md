# Experiment instructions

Run from the repository root. Each configuration file stores the evaluation
seeds and the stepsizes used in the paper. Baseline methods are not included.

## Synthetic representation learning

```bash
python -m experiments.synthetic --method exact --kappa 10 --seed 10 \
  --output work/synthetic-polar-s10
python -m experiments.synthetic --method ns --kappa 10 --seed 10 \
  --output work/synthetic-ns-s10
```

Repeat over seeds 10–14 and input anisotropies 1, 10, 100, and 1000. Each setting
has its own saved stepsizes. The horizon is 60 UL iterations in float64. Full LL
refits at logged iterates never replace the warm-started training classifier.

## CIFAR-10 representation learning

```bash
python -m experiments.cifar10 prepare --data-dir work/data \
  --cache work/cifar10_features.npz
python -m experiments.cifar10 run --cache work/cifar10_features.npz \
  --method ns --seed 20 --output work/cifar10-ns-s20
```

Preparation explicitly downloads the official training split and pretrained
ResNet-18 weights through torchvision. The fixed pool has 150 examples per
class. Each seed splits it into 100 LL-training and 50 UL-validation examples
per class and fits one scalar RMS normalization on LL features. The official
test set is not loaded. Repeat both methods for seeds 20–24.

The UL tanh representation is 256-by-512. The LL classifier is L2-regularized.
Optimization uses 60 UL iterations in float64 and an independent full LL refit
at every iteration. `optimized_val_ce` is optimized UL validation cross-entropy,
not a test metric.

## CIFAR-100 controlled blockwise allocation

```bash
python -m experiments.prepare_cifar100 --cache work/cifar100_features \
  --data-dir work/data
python -m experiments.blockwise --cache work/cifar100_features --rho 10 \
  --seed 41001 --output work/blockwise-r10-s41001
```

Preparation uses frozen ImageNet-pretrained ResNet-50 features, a fixed
40,000/10,000 training/validation split, and one LL-training RMS scale. It caches
official test features separately; the validation runner does not load them.
The runner uses the proposed allocation across four classifier blocks,
sensitivity ratios 1, 3, and 10, and the frozen global stepsize 0.1. Repeat over
seeds 41001–41005. CLI `rho` is the sensitivity ratio written as varrho in the
manuscript, not its Hessian-Lipschitz constant.

## Transformer representation learning

The full experiment requires CUDA and sufficient memory. Original runs used one
H100 NVL per run; the frozen-prefix cache alone is about 14 GiB. Select a device
using `CUDA_VISIBLE_DEVICES` before running.

```bash
python -m experiments.transformer.cli prepare --root work/transformer
python -m experiments.transformer.cli run --root work/transformer \
  --method ns --seed 10 --output work/transformer-ns-s10
```

Preparation downloads the non-distilled ImageNet-pretrained DeiT-S/16 checkpoint
and CIFAR-100 training data. Sixteen matrices in the last four Transformer blocks
are trainable (7,077,888 parameters); all other weights stay frozen. Only the
first eight blocks' outputs are cached across iterations. Trainable features
are refreshed after every UL update.

The saved protocol uses D=5, Q=1024 Hessian-vector products (1025 Neumann terms),
mini-batches of 64, and 1,000 UL iterations in FP32. Repeat both methods over
seeds 10–14. Independent float64 LL refits run every 250 iterations. The runner
reports validation metrics only.

## Outputs

Use a new output directory for every run. Numeric configurations, trajectories,
solver residuals, and states are saved there. A run that diverges or whose LL
refit does not converge still writes its outputs and records the status in
`result.json`. Everything under `work/` is ignored by Git.
