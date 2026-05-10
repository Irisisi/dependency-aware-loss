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
