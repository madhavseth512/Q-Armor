"""Base classifier interface for the Q-Armor reasoning module.

Every model in ``reasoning/`` — the classical baselines now and the quantum
models (PegasosQSVC, VQC, QuantumAutoencoder) in Phase 5 — implements this same
contract so the orchestrator, planning, and action modules can treat them
interchangeably. The load-bearing method is:

    predict(x) -> (label: str, confidence: float, model_name: str)

Subclasses only implement ``fit``, ``predict_proba``, and ``classes_``; the
contract methods (``predict`` / ``predict_batch``) are shared here so confidence
and model-name semantics are identical across every model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import joblib
import numpy as np


class BaseClassifier(ABC):
    """Abstract base for all Q-Armor classifiers.

    Attributes:
        name: Short model identifier returned as the third element of the
            ``predict`` tuple (e.g. ``"random_forest"``, ``"svm_rbf"``).
    """

    name: str = "base"

    # -- to implement in subclasses ----------------------------------------

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        """Train the model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,).

        Returns:
            Self, fitted.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_classes); row i sums to 1.
        """

    @property
    @abstractmethod
    def classes_(self) -> np.ndarray:
        """The class labels, index-aligned with ``predict_proba`` columns."""

    # -- shared contract (do not override) ---------------------------------

    def predict(self, x: np.ndarray) -> tuple[str, float, str]:
        """Predict a single sample under the uniform agent contract.

        Args:
            x: A single feature vector of shape (n_features,).

        Returns:
            Tuple ``(label, confidence, model_name)`` where ``confidence`` is the
            maximum class probability in [0, 1].
        """
        proba = self.predict_proba(np.asarray(x, dtype=float).reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        return str(self.classes_[idx]), float(proba[idx]), self.name

    def predict_batch(self, X: np.ndarray) -> list[tuple[str, float, str]]:
        """Predict a batch of samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            List of ``(label, confidence, model_name)`` tuples, one per row.
        """
        proba = self.predict_proba(np.asarray(X, dtype=float))
        idx = np.argmax(proba, axis=1)
        classes = self.classes_
        return [(str(classes[i]), float(proba[r, i]), self.name) for r, i in enumerate(idx)]

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the fitted model to disk via joblib.

        Args:
            path: Destination ``.joblib`` path.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaseClassifier":
        """Load a model previously saved with :meth:`save`.

        Args:
            path: Path to the ``.joblib`` file.

        Returns:
            The deserialised classifier.
        """
        return joblib.load(path)
