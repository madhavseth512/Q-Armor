"""Perception module: quantum feature encoding and kernel computation.

Wraps the custom CyberSecurityFeatureMap in a FidelityQuantumKernel and exposes a
clean interface for the reasoning module's quantum models (QSVC, Phase 5). The
kernel similarity is the state overlap K(x1, x2) = |<phi(x1)|phi(x2)>|^2, computed
on a simulator.

An optional MemoryModule provides an LRU cache so repeated single-pair kernel
evaluations are not recomputed (each is a quantum circuit). The trainable-kernel
hooks (get/update_kernel_params) are scaffolded here; the adaptive
TrainableFidelityQuantumKernel is wired in Phase 6.
"""

from __future__ import annotations

import numpy as np
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from perception.feature_map import CyberSecurityFeatureMap


class PerceptionModule:
    """Quantum feature map + fidelity kernel, with optional caching."""

    def __init__(self, feature_map: CyberSecurityFeatureMap | None = None, memory=None) -> None:
        """Initialise the feature map and kernel.

        Args:
            feature_map: Encoding circuit; defaults to a fresh CyberSecurityFeatureMap.
            memory: Optional MemoryModule providing the kernel LRU cache.
        """
        self.feature_map = feature_map or CyberSecurityFeatureMap()
        self.kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        self._memory = memory
        self._X_train: np.ndarray | None = None
        self._kernel_params: np.ndarray | None = None   # scaffold for Phase 6 trainable kernel

    def set_training_data(self, X_train: np.ndarray) -> None:
        """Store training samples so :meth:`encode` can build kernel rows.

        Args:
            X_train: Training feature matrix (n_train, num_features).
        """
        self._X_train = np.asarray(X_train, dtype=float)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode a sample into its kernel row against the stored training set.

        Args:
            x: A single feature vector (num_features,).

        Returns:
            Kernel vector (n_train,) of state overlaps with each training sample.

        Raises:
            RuntimeError: If :meth:`set_training_data` was not called first.
        """
        if self._X_train is None:
            raise RuntimeError("PerceptionModule.set_training_data() must be called first.")
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return self.kernel.evaluate(x, self._X_train)[0]

    def compute_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute the fidelity kernel between two samples, using the cache if set.

        Args:
            x1: First feature vector.
            x2: Second feature vector.

        Returns:
            Kernel value in [0, 1].
        """
        if self._memory is not None:
            cached = self._memory.get_cached_kernel(x1, x2)
            if cached is not None:
                return cached
        x1 = np.asarray(x1, dtype=float).reshape(1, -1)
        x2 = np.asarray(x2, dtype=float).reshape(1, -1)
        value = float(self.kernel.evaluate(x1, x2)[0, 0])
        if self._memory is not None:
            self._memory.cache_kernel(x1.ravel(), x2.ravel(), value)
        return value

    def compute_kernel_matrix(self, X_train: np.ndarray, X_test: np.ndarray | None = None) -> np.ndarray:
        """Compute a full kernel matrix.

        Args:
            X_train: Training feature matrix (n_train, num_features).
            X_test: Optional test matrix (n_test, num_features). If None, returns the
                symmetric training kernel.

        Returns:
            Kernel matrix (n_train, n_train) or (n_test, n_train).
        """
        X_train = np.asarray(X_train, dtype=float)
        if X_test is None:
            return self.kernel.evaluate(X_train)
        return self.kernel.evaluate(np.asarray(X_test, dtype=float), X_train)

    def update_kernel_params(self, params: np.ndarray) -> None:
        """Set trainable kernel parameters (scaffold; trained in Phase 6)."""
        self._kernel_params = np.asarray(params, dtype=float)

    def get_kernel_params(self) -> np.ndarray | None:
        """Return the current trainable kernel parameters (None until trained)."""
        return self._kernel_params
