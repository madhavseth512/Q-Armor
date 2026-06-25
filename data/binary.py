"""Binary label utilities for the Phase 5 detect→type cascade.

The quantum models in Phase 5 operate on BINARY labels (benign / attack)
because the fidelity kernel has strong 2-class separability (0.91 AUROC vs
0.35 macro-F1 at 15 classes — Phase 2 result). Multi-class typing is the
Stage-2 classical job.

Usage:
    y_bin = to_binary(y_parent_labels)   # 0 = benign, 1 = attack
"""

from __future__ import annotations

import numpy as np

BENIGN_LABEL: str = "BenignTraffic"

BINARY_CLASS_NAMES: np.ndarray = np.array([0, 1])
"""Canonical class array for binary models: 0 = benign, 1 = attack."""


def to_binary(y: np.ndarray) -> np.ndarray:
    """Map parent-class string labels to binary integers.

    Args:
        y: Array of parent class names (e.g. ``"DDoS"``, ``"BenignTraffic"``).

    Returns:
        Integer array of the same shape: ``0`` for BenignTraffic, ``1`` for any
        attack class.
    """
    return (np.asarray(y) != BENIGN_LABEL).astype(int)
