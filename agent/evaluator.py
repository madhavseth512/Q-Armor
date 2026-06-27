"""Episode evaluator for the Reflexion outer loop.

Aggregates per-sample predictions over one episode window and produces a
structured EpisodeReport. Operates at the episode level (not per-sample);
the PlanningModule continues to run per-sample ADWIN and confidence monitoring,
and passes the episode-level drift flag into this evaluator at episode end.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EpisodeReport:
    """Performance snapshot for one episode of EPISODE_SIZE samples."""

    episode_id:      int
    model_used:      str
    dataset:         str
    n_samples:       int
    auroc:           float
    binary_f1:       float
    aupr:            float
    fpr_at_tpr95:    float
    mean_confidence: float
    drift_detected:  bool
    timestamp:       str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_healthy(self) -> bool:
        """True when AUROC is above the floor threshold."""
        from agent import agent_config as config
        return self.auroc >= config.AUROC_FLOOR


def _fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray,
                tpr_target: float = 0.95) -> float:
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
    idx = min(np.searchsorted(tpr_arr, tpr_target), len(fpr_arr) - 1)
    return float(fpr_arr[idx])


class Evaluator:
    """Computes EpisodeReport from episode-level prediction arrays."""

    def evaluate(
        self,
        episode_id:     int,
        model_used:     str,
        dataset:        str,
        y_true:         np.ndarray,
        y_scores:       np.ndarray,
        y_pred:         np.ndarray,
        confidences:    np.ndarray,
        drift_detected: bool,
    ) -> EpisodeReport:
        """Produce a full EpisodeReport for one episode.

        Args:
            episode_id:     Sequential episode counter (0-based).
            model_used:     Name of the model that generated predictions.
            dataset:        Dataset identifier string.
            y_true:         True binary labels (0=benign, 1=attack).
            y_scores:       Attack probability scores (column 1 of predict_proba).
            y_pred:         Hard binary predictions (argmax of predict_proba).
            confidences:    Per-sample max probability (used for mean confidence).
            drift_detected: Whether ADWIN fired during this episode.

        Returns:
            EpisodeReport with all computed metrics.
        """
        y_true   = np.asarray(y_true)
        y_scores = np.asarray(y_scores)
        y_pred   = np.asarray(y_pred)

        auroc    = float(roc_auc_score(y_true, y_scores))
        binary_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        aupr     = float(average_precision_score(y_true, y_scores))
        fpr95    = _fpr_at_tpr(y_true, y_scores)
        mean_conf = float(np.mean(confidences))

        return EpisodeReport(
            episode_id      = episode_id,
            model_used      = model_used,
            dataset         = dataset,
            n_samples       = int(len(y_true)),
            auroc           = round(auroc, 4),
            binary_f1       = round(binary_f1, 4),
            aupr            = round(aupr, 4),
            fpr_at_tpr95    = round(fpr95, 4),
            mean_confidence = round(mean_conf, 4),
            drift_detected  = bool(drift_detected),
            timestamp       = datetime.now(timezone.utc).isoformat(),
        )
