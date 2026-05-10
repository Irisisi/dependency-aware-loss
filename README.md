# Dependency-Aware Loss for Extreme Multi-Label EHR Prediction

This repository contains PyTorch implementations of the loss functions used in the CHIL 2026 paper:

**Learning Under Extreme Label Imbalance in EHRs: A Dependency-Aware Loss for Multi-Label Classification**

Included losses:

- Dependency-aware loss
- Weighted binary cross-entropy
- Class-balanced binary cross-entropy
- Focal loss

## Data availability

The paper uses data from the Clinical Practice Research Datalink (CPRD). CPRD data cannot be redistributed due to licensing, governance, and approval requirements.

## Repository structure

```text
losses/
├── __init__.py
├── dependency_aware.py
├── weighted_bce.py
├── class_balanced_bce.py
└── focal.py


## Installation

Clone the repository:

```bash
git clone https://github.com/Irisisi/dependency-aware-loss.git
cd dependency-aware-loss
```

Install the dependency:

```bash
pip install -r requirements.txt
```

The only required package is PyTorch.

The examples below assume that Python is run from the repository root.

## Input format

All losses expect `logits` and `labels` to have the same shape:

```python
logits.shape == labels.shape
```

The final dimension is the label vocabulary size:

```python
logits: [..., V]
labels: [..., V]
```

where `V` is the number of labels.

The label tensor should contain:

```text
 1   positive label
 0   negative / absent label
-1   ignored or padded label
```

Entries with label `-1` are excluded from loss computation.

## Dependency-aware loss example

```python
import torch
from losses import DependencyAwareLoss

B = 8
V = 1538

logits = torch.randn(B, V, requires_grad=True)

# Sparse multi-label targets, roughly 0.5% positives.
labels = (torch.rand(B, V) < 0.005).long()

criterion = DependencyAwareLoss(
    vocab_size=V,
    alpha_neg=25.0,
    lambda_reg=10.0,
    ignore_index=-1,
)

loss = criterion(logits, labels)
loss.backward()

print(loss.item())

```

The dependency-aware loss also accepts higher-dimensional inputs, as long as the final dimension is the vocabulary dimension:

```python
logits = torch.randn(4, 3, V, requires_grad=True)
labels = (torch.rand(4, 3, V) < 0.005).long()

loss = criterion(logits, labels)
loss.backward()

print(loss.item())

```

## Baseline loss examples

```python
import torch
from losses import (
    weighted_bce_with_logits,
    class_balanced_bce_with_logits,
    focal_loss_with_logits,
)

B = 8
V = 1538

logits = torch.randn(B, V, requires_grad=True)
labels = (torch.rand(B, V) < 0.005).long()

loss_bce = weighted_bce_with_logits(logits, labels)

loss_cb = class_balanced_bce_with_logits(
    logits,
    labels,
    beta=0.9999,
)

loss_focal = focal_loss_with_logits(
    logits,
    labels,
    gamma=2.0,
    beta=0.9999,
)

```

## Quick sanity check

Run this from the repository root:

```bash
python - <<'PY'
import torch
from losses import (
    DependencyAwareLoss,
    weighted_bce_with_logits,
    class_balanced_bce_with_logits,
    focal_loss_with_logits,
)

torch.manual_seed(0)

B, V = 4, 64
logits = torch.randn(B, V, requires_grad=True)
labels = (torch.rand(B, V) < 0.02).long()

da = DependencyAwareLoss(vocab_size=V)

loss_da = da(logits, labels)
print("Dependency-aware:", loss_da.item())

print("Weighted BCE:", weighted_bce_with_logits(logits, labels).item())
print("Class-balanced BCE:", class_balanced_bce_with_logits(logits, labels).item())
print("Focal:", focal_loss_with_logits(logits, labels).item())

loss_da.backward()
print("Backward pass: OK")
PY
```

This is only a code sanity check using random tensors. It does not reproduce the CPRD experiments.

## Dependency-aware loss hyperparameters

The dependency-aware loss settings used in the paper were:

```python
alpha_neg = 25.0
lambda_reg = 10.0
```

The implementation computes internally in `float32` for numerical stability, including when model logits are provided in lower precision.

## Inspecting the learned dependency matrix

For the dependency-aware loss, the learnable raw matrix is stored in:

```python
criterion.W_raw
```

The symmetric zero-diagonal matrix used in the forward pass can be obtained by:

```python
W = criterion._sym_zero_diag(criterion.W_raw).detach()
```

The entries of `W` should be interpreted as learned output-space compatibility weights under the training objective. They are not causal effects and should not be interpreted as direct empirical co-occurrence frequencies.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ho2026dependency,
  title     = {Learning Under Extreme Label Imbalance in EHRs: A Dependency-Aware Loss for Multi-Label Classification},
  author    = {Ho, Iris Szu-Szu and Werne, Lars and Rawlik, Konrad and Guthrie, Bruce and Seth, Sohan},
  booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {297},
  pages     = {1--23},
  year      = {2026},
  publisher = {PMLR}
}
```

## License

This code is released under the MIT License. See `LICENSE`.
