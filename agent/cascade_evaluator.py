"""Cascade evaluator for Phase 8: metrics for the two-stage Detect->Type pipeline.

Produces a CascadeReport from the raw outputs of DetectTypeCascade.predict() and
the true binary + coarse-type labels. Reports both Stage 1 (binary) and Stage 2
(multi-class typing) metrics, plus end-to-end summary statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from agent import agent_config as config
from reasoning.cascade import CascadeResult


@dataclass
class CascadeReport:
    """Full metrics snapshot for one cascade evaluation pass."""

    # Identification
    dataset:          str
    model_stage1:     str
    model_stage2:     str
    with_retraining:  bool
    n_samples:        int
    timestamp:        str

    # Stage 1 — binary detection
    binary_auroc:     float
    binary_f1:        float
    binary_aupr:      float
    binary_fpr95:     float
    binary_precision: float
    binary_recall:    float

    # Stage 2 — attack typing (on true-attack samples that were correctly detected)
    type_macro_f1:      float
    type_accuracy:      float
    type_per_class_f1:  dict[str, float]
    type_n_typed:       int    # number of samples Stage 2 actually ran on
    type_coverage:      float  # fraction of true-attack samples that Stage 1 flagged

    # End-to-end counts
    n_true_attack:    int
    n_true_benign:    int
    n_pred_attack:    int
    n_pred_benign:    int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_healthy(self) -> bool:
        return self.binary_auroc >= config.AUROC_FLOOR


def _fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray,
                tpr_target: float = 0.95) -> float:
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
    idx = min(np.searchsorted(tpr_arr, tpr_target), len(fpr_arr) - 1)
    return float(fpr_arr[idx])


class CascadeEvaluator:
    """Computes CascadeReport from a CascadeResult and ground-truth labels."""

    def evaluate(
        self,
        result:         CascadeResult,
        y_true_bin:     np.ndarray,    # ground truth binary (0=benign, 1=attack)
        y_true_coarse:  np.ndarray,    # ground truth coarse type strings
        dataset:        str,
        model_stage1:   str,
        model_stage2:   str,
        with_retraining: bool = False,
    ) -> CascadeReport:
        """Compute the full CascadeReport.

        Args:
            result:          Output from DetectTypeCascade.predict().
            y_true_bin:      True binary labels (0 / 1).
            y_true_coarse:   True coarse type labels (e.g. "DoS", "Benign").
            dataset:         Dataset identifier string.
            model_stage1:    Name of the Stage 1 binary model.
            model_stage2:    Name of the Stage 2 typing model.
            with_retraining: Whether Stage 1 was retrained before this eval.

        Returns:
            CascadeReport with all Stage 1, Stage 2, and E2E metrics.
        """
        y_true_bin    = np.asarray(y_true_bin)
        y_true_coarse = np.asarray(y_true_coarse)
        n = len(y_true_bin)
        ts = datetime.now(timezone.utc).isoformat()

        # -- Stage 1 metrics --------------------------------------------------
        auroc     = float(roc_auc_score(y_true_bin, result.y_bin_scores))
        b_f1      = float(f1_score(y_true_bin, result.y_bin_pred, pos_label=1, zero_division=0))
        aupr      = float(average_precision_score(y_true_bin, result.y_bin_scores))
        fpr95     = _fpr_at_tpr(y_true_bin, result.y_bin_scores)
        precision = float(f1_score(y_true_bin, result.y_bin_pred, average=None,
                                   zero_division=0)[1])
        recall    = float(np.sum((result.y_bin_pred == 1) & (y_true_bin == 1)) /
                          max(1, np.sum(y_true_bin == 1)))

        n_true_attack = int(np.sum(y_true_bin == 1))
        n_true_benign = int(np.sum(y_true_bin == 0))
        n_pred_attack = int(np.sum(result.y_bin_pred == 1))
        n_pred_benign = int(np.sum(result.y_bin_pred == 0))

        # -- Stage 2 metrics (evaluate on ALL attack-predicted samples) ------
        # Coverage = fraction of true attacks that Stage 1 caught
        type_coverage = float(
            np.sum(result.attack_mask & (y_true_bin == 1)) / max(1, n_true_attack)
        )

        # For per-class F1: compare predicted types against true coarse types
        # for every sample (including benign; Stage 2 assigns "Benign" to
        # Stage-1-benign samples, so an FP has type "Benign" which is wrong).
        classes     = config.NF_COARSE_CLASSES
        rep         = classification_report(
            y_true_coarse, result.y_type_pred,
            labels=classes, output_dict=True, zero_division=0,
        )
        type_macro_f1 = float(rep["macro avg"]["f1-score"])
        type_accuracy = float(rep.get("accuracy", 0.0))
        type_per_cls  = {c: round(float(rep.get(c, {}).get("f1-score", 0.0)), 4)
                         for c in classes}
        n_typed = int(result.attack_mask.sum())

        return CascadeReport(
            dataset          = dataset,
            model_stage1     = model_stage1,
            model_stage2     = model_stage2,
            with_retraining  = with_retraining,
            n_samples        = n,
            timestamp        = ts,
            binary_auroc     = round(auroc, 4),
            binary_f1        = round(b_f1, 4),
            binary_aupr      = round(aupr, 4),
            binary_fpr95     = round(fpr95, 4),
            binary_precision = round(precision, 4),
            binary_recall    = round(recall, 4),
            type_macro_f1    = round(type_macro_f1, 4),
            type_accuracy    = round(type_accuracy, 4),
            type_per_class_f1 = type_per_cls,
            type_n_typed     = n_typed,
            type_coverage    = round(type_coverage, 4),
            n_true_attack    = n_true_attack,
            n_true_benign    = n_true_benign,
            n_pred_attack    = n_pred_attack,
            n_pred_benign    = n_pred_benign,
        )
