"""Memory module: agent-state persistence and kernel value cache.

Persists trained models (RandomForest, the fitted preprocessor) and monitor state
(confidence window, ADWIN, noise calibration) to disk so the agent can restore a
previous session without retraining. Also provides an LRU cache of computed kernel
values — scaffolded here and exercised by the quantum kernel in Phase 4.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

import joblib
import numpy as np

from agent import agent_config as config


class MemoryModule:
    """Disk persistence for models/state + an in-memory LRU kernel cache."""

    def __init__(self) -> None:
        """Create the persistence directory and an empty kernel cache."""
        self._dir = config.MEMORY_DIR
        os.makedirs(self._dir, exist_ok=True)
        self._cache: "OrderedDict[tuple[bytes, bytes], float]" = OrderedDict()
        self._max_cache = config.KERNEL_CACHE_MAX_SIZE
        self._hits = 0
        self._misses = 0

    # -- model persistence --------------------------------------------------

    def save_model(self, model: Any, name: str) -> None:
        """Serialise a model/transformer to ``<memory_dir>/<name>.joblib``.

        Args:
            model: Any joblib-serialisable object (classifier, preprocessor).
            name: File stem.
        """
        joblib.dump(model, os.path.join(self._dir, f"{name}.joblib"))

    def load_model(self, name: str) -> Any:
        """Load a model previously saved with :meth:`save_model`.

        Args:
            name: File stem.

        Returns:
            The deserialised object.

        Raises:
            FileNotFoundError: If no such file exists.
        """
        return joblib.load(os.path.join(self._dir, f"{name}.joblib"))

    def has_model(self, name: str) -> bool:
        """Whether a model file ``<name>.joblib`` exists in the memory directory."""
        return os.path.exists(os.path.join(self._dir, f"{name}.joblib"))

    # -- agent state --------------------------------------------------------

    def save_state(self, state: dict[str, Any]) -> None:
        """Persist the full agent state dict (monitor windows, ADWIN, calibration).

        Args:
            state: Serialisable dict (joblib handles river/numpy/deque objects).
        """
        joblib.dump(state, os.path.join(self._dir, "agent_state.joblib"))

    def load_state(self) -> dict[str, Any]:
        """Restore the saved agent state.

        Returns:
            The state dict from :meth:`save_state`, or an empty dict if none saved.
        """
        path = os.path.join(self._dir, "agent_state.joblib")
        return joblib.load(path) if os.path.exists(path) else {}

    # -- kernel cache (LRU) -------------------------------------------------

    @staticmethod
    def _key(x1: np.ndarray, x2: np.ndarray) -> tuple[bytes, bytes]:
        """Order-independent cache key for a symmetric kernel K(x1, x2)=K(x2, x1)."""
        b1 = np.asarray(x1, dtype=np.float64).tobytes()
        b2 = np.asarray(x2, dtype=np.float64).tobytes()
        return (b1, b2) if b1 <= b2 else (b2, b1)

    def cache_kernel(self, x1: np.ndarray, x2: np.ndarray, value: float) -> None:
        """Store a kernel value, evicting the least-recently-used entry if full.

        Args:
            x1: First feature vector.
            x2: Second feature vector.
            value: Kernel value to cache.
        """
        key = self._key(x1, x2)
        self._cache[key] = float(value)
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)

    def get_cached_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float | None:
        """Look up a cached kernel value, updating hit/miss stats.

        Args:
            x1: First feature vector.
            x2: Second feature vector.

        Returns:
            The cached value, or None on a miss.
        """
        key = self._key(x1, x2)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def clear_kernel_cache(self) -> None:
        """Flush all cached kernel values and reset hit/miss counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_cache_stats(self) -> tuple[int, int]:
        """Return ``(hit_count, miss_count)`` since the last clear."""
        return self._hits, self._misses
