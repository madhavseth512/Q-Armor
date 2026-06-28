"""Detect->Type two-stage cascade for Q-Armor Phase 8.

Stage 1 — binary detection (attack vs benign):
    Any BaseClassifier (QSVC, RF, SVM-RBF). Uses predict_proba[:, 1] as the
    attack score; returns binary prediction + confidence.

Stage 2 — attack typing (DoS / Injection / Recon / Backdoor):
    Classical RF trained on the 5-class coarse taxonomy. Only runs on samples
    that Stage 1 flagged as attack; benign samples pass through as "Benign".

The cascade is the E8 end goal (Goal G4): quantum-enhanced detection feeds a
classical multi-class typer, giving per-sample attack classification without
requiring the quantum model to solve the multi-class problem directly.

Result convention:
    y_type_pred[i] == "Benign"  when Stage 1 predicted benign (attack_mask[i]=False)
    y_type_pred[i] == <class>   when Stage 1 predicted attack and Stage 2 typed it
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reasoning.base import BaseClassifier


@dataclass
class CascadeResult:
    """Per-sample outputs from a DetectTypeCascade prediction pass."""

    y_bin_scores:  np.ndarray   # Stage 1: P(attack), shape (n,)
    y_bin_pred:    np.ndarray   # Stage 1: hard binary labels (0/1), shape (n,)
    y_bin_conf:    np.ndarray   # Stage 1: max(P(benign), P(attack)), shape (n,)
    y_type_pred:   np.ndarray   # Stage 2: coarse type string, shape (n,)  [object dtype]
    y_type_conf:   np.ndarray   # Stage 2: confidence for typed class, shape (n,)
    attack_mask:   np.ndarray   # True where Stage 1 predicted attack, shape (n,)


class DetectTypeCascade:
    """Two-stage binary-detection -> attack-typing cascade.

    Args:
        binary_model: Fitted Stage 1 binary classifier (BaseClassifier subclass).
                      predict_proba columns: [P(benign=0), P(attack=1)].
        type_model:   Fitted Stage 2 multi-class classifier (BaseClassifier subclass).
                      classes_ must cover the 5-class coarse taxonomy.
    """

    def __init__(
        self,
        binary_model: BaseClassifier,
        type_model:   BaseClassifier,
    ) -> None:
        self.binary_model = binary_model
        self.type_model   = type_model

    def predict(self, X: np.ndarray) -> CascadeResult:
        """Run both stages on an array of preprocessed samples.

        Args:
            X: Scaled feature matrix, shape (n_samples, n_features).

        Returns:
            CascadeResult with per-sample binary and type predictions.
        """
        X = np.asarray(X)

        # -- Stage 1: binary detection ----------------------------------------
        bin_proba    = self.binary_model.predict_proba(X)      # (n, 2)
        y_bin_scores = bin_proba[:, 1]                         # P(attack) — drives AUROC
        # Use predict_labels() instead of argmax so SVM-based models (PegasosQSVC)
        # apply the correct margin boundary rather than a 0.5 probability threshold.
        y_bin_pred   = self.binary_model.predict_labels(X)     # 0 or 1
        y_bin_conf   = bin_proba.max(axis=1)

        # -- Stage 2: attack typing (attack-flagged samples only) -------------
        n = len(X)
        attack_mask = y_bin_pred == 1
        y_type_pred = np.full(n, "Benign", dtype=object)
        y_type_conf = np.zeros(n, dtype=float)

        if attack_mask.any():
            type_proba = self.type_model.predict_proba(X[attack_mask])  # (n_attack, n_classes)
            type_idx   = type_proba.argmax(axis=1)
            y_type_pred[attack_mask] = self.type_model.classes_[type_idx]
            y_type_conf[attack_mask] = type_proba.max(axis=1)

        return CascadeResult(
            y_bin_scores = y_bin_scores,
            y_bin_pred   = y_bin_pred,
            y_bin_conf   = y_bin_conf,
            y_type_pred  = y_type_pred,
            y_type_conf  = y_type_conf,
            attack_mask  = attack_mask,
        )

    @property
    def stage1_name(self) -> str:
        return self.binary_model.name

    @property
    def stage2_name(self) -> str:
        return self.type_model.name
