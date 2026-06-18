"""Classical baseline models for Q-Armor: RandomForest and RBF-SVM.

These are active fallback models in the agent (used when noise makes the quantum
models untrustworthy) AND the quantitative baselines the quantum models must beat.
Both implement the shared ``BaseClassifier`` contract.

Design notes:
  * RandomForest uses a capped depth (max_depth=18) but NO leaf floor. A Phase-2
    sweep showed unlimited depth memorises (train F1 1.0, val drops) while a leaf
    floor (min_samples_leaf>1) only lowers validation — the rare-class train-val
    gap was data scarcity, not harmful overfitting. Depth sweep on full-400k:
    14->0.714, 16->0.7345, 18->0.7376 (best), 20->0.728, None->0.715.
  * SVM-RBF uses ``class_weight='balanced'`` instead of SMOTE — kernel methods are
    sensitive to point geometry, so we feed real points and reweight rather than
    inject synthetic interpolations. Confidence comes from sigmoid (Platt)
    calibration via ``CalibratedClassifierCV`` — the sklearn-recommended
    replacement for the deprecated ``SVC(probability=True)``.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from agent import agent_config as config
from reasoning.base import BaseClassifier


class RandomForestModel(BaseClassifier):
    """RandomForest classifier wrapper (regularised baseline)."""

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 18,
        min_samples_leaf: int = 1,
        class_weight: str | dict | None = None,
    ) -> None:
        """Initialise the underlying sklearn RandomForest.

        Args:
            n_estimators: Number of trees.
            max_depth: Max tree depth (capped to limit memorisation).
            min_samples_leaf: Min samples per leaf (regularisation).
            class_weight: Passed through to sklearn; use ``"balanced"`` when
                training on un-SMOTE'd data, ``None`` when data is pre-balanced.
        """
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=config.RANDOM_SEED,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """Train the forest. See :meth:`BaseClassifier.fit`."""
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities from tree vote fractions."""
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        """Fitted class labels."""
        return self._model.classes_


class SVMModel(BaseClassifier):
    """RBF-kernel SVM wrapper (the kernel baseline for QSVC)."""

    name = "svm_rbf"

    def __init__(
        self,
        C: float = 1.0,
        gamma: str | float = "scale",
        class_weight: str | dict | None = "balanced",
    ) -> None:
        """Initialise the underlying sklearn SVC with an RBF kernel.

        Args:
            C: Regularisation strength (set by the CV tuner in Phase 2).
            gamma: RBF kernel width (set by the CV tuner).
            class_weight: ``"balanced"`` reweights rare classes without SMOTE.
        """
        svc = SVC(
            C=C,
            gamma=gamma,
            kernel="rbf",
            class_weight=class_weight,
            random_state=config.RANDOM_SEED,
        )
        # Sigmoid calibration provides predict_proba (the confidence score) without
        # the deprecated SVC(probability=True). ensemble=False keeps a single
        # calibrated model rather than averaging cv folds.
        self._model = CalibratedClassifierCV(svc, method="sigmoid", ensemble=False, cv=3)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMModel":
        """Train the SVM and calibrate its probabilities. See :meth:`BaseClassifier.fit`."""
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated class probabilities (sigmoid/Platt calibration)."""
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        """Fitted class labels."""
        return self._model.classes_
