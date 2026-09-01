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

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Return hard integer predictions for a batch, using the model's native
        decision boundary (not a fixed 0.5 probability threshold).

        Default implementation uses argmax of predict_proba, which is correct for
        calibrated classifiers (RF, SVM-RBF). Override in subclasses whose
        probability outputs are not calibrated around 0.5 (e.g. PegasosQSVC).

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Label array of shape (n_samples,), values in classes_. (Prior to
            this fix, the default returned raw argmax column indices instead
            of classes_ values — silently correct only for binary {0,1}
            models where index and value coincide; wrong for any classifier
            with non-0..n-1 classes_, e.g. the string-labelled coarse-type
            models. Caught when predict()/predict_batch() were routed through
            predict_labels() and multiclass tests failed with a lookup
            IndexError.)
        """
        idx = self.predict_proba(np.asarray(X, dtype=float)).argmax(axis=1)
        return self.classes_[idx]

    def predict(self, x: np.ndarray) -> tuple[str, float, str]:
        """Predict a single sample under the uniform agent contract.

        Uses ``predict_labels()`` for the hard decision, not argmax(proba) —
        for models whose probability output is not calibrated around 0.5
        (e.g. PegasosQSVC), the two disagree (see ``predict_labels`` docs).

        Args:
            x: A single feature vector of shape (n_features,).

        Returns:
            Tuple ``(label, confidence, model_name)`` where ``confidence`` is
            the predicted class's probability in [0, 1] (not necessarily the
            row max, since the hard decision need not match argmax(proba)).
        """
        X = np.asarray(x, dtype=float).reshape(1, -1)
        proba = self.predict_proba(X)[0]
        label_val = self.predict_labels(X)[0]
        classes = self.classes_
        idx = int(np.where(classes == label_val)[0][0])
        return str(classes[idx]), float(proba[idx]), self.name

    def predict_batch(self, X: np.ndarray) -> list[tuple[str, float, str]]:
        """Predict a batch of samples.

        Uses ``predict_labels()`` for the hard decision, not argmax(proba) —
        see :meth:`predict`.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            List of ``(label, confidence, model_name)`` tuples, one per row.
        """
        X = np.asarray(X, dtype=float)
        proba = self.predict_proba(X)
        labels = self.predict_labels(X)
        classes = self.classes_
        idx_of = {c: i for i, c in enumerate(classes)}
        return [(str(l), float(proba[r, idx_of[l]]), self.name)
                for r, l in enumerate(labels)]

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
