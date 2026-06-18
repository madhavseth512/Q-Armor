"""Rule-based model selector for the reasoning module.

Holds a registry of available models (each a BaseClassifier) and picks which one
the agent should use, given the current noise / data / confidence signals. The
preference order follows ``MODEL_HIERARCHY`` (QSVM > VQC > QAE > CLASSICAL).

In Phase 3 only classical models are registered, so ``select`` always returns the
classical model. The full routing — skipping quantum tiers under high noise,
switching on low confidence — is exercised in Phase 5 when the quantum models are
registered, with no change to this interface.
"""

from __future__ import annotations

from agent import agent_config as config
from reasoning.base import BaseClassifier

# Maps a MODEL_HIERARCHY tier to the model name that implements it.
TIER_TO_NAME: dict[str, str] = {
    "QSVM": "pegasos_qsvc",
    "VQC": "vqc",
    "QAE": "quantum_autoencoder",
    "CLASSICAL": "random_forest",
}


class ModelSelector:
    """Registry of models + rule-based selection of which to use per sample."""

    def __init__(self) -> None:
        """Initialise an empty model registry."""
        self._registry: dict[str, BaseClassifier] = {}

    def register(self, model: BaseClassifier) -> None:
        """Add a model to the registry under its ``name``.

        Args:
            model: A fitted BaseClassifier.
        """
        self._registry[model.name] = model

    def get(self, name: str) -> BaseClassifier:
        """Return a registered model by name."""
        return self._registry[name]

    def available(self) -> list[str]:
        """Names of all registered models."""
        return list(self._registry)

    def select(
        self,
        noise_level: float,
        data_size: int,
        confidence: float,
        current_model: str | None = None,
    ) -> str:
        """Choose which registered model to use under the current conditions.

        Walks ``MODEL_HIERARCHY`` and returns the highest-priority *registered*
        model that is permitted: quantum tiers are skipped when noise is at or above
        the critical fallback threshold.

        Args:
            noise_level: Current gate error rate (0-1).
            data_size: Available training-set size (quantum-eligibility context).
            confidence: Recent mean prediction confidence (0-1).
            current_model: Name of the model currently in use, if any.

        Returns:
            The selected model name.

        Raises:
            RuntimeError: If no models are registered.
        """
        if not self._registry:
            raise RuntimeError("ModelSelector has no registered models.")

        quantum_ok = noise_level < config.NOISE_THRESHOLD_FALLBACK
        for tier in config.MODEL_HIERARCHY:
            name = TIER_TO_NAME.get(tier)
            if name not in self._registry:
                continue
            if tier != "CLASSICAL" and not quantum_ok:
                continue  # too noisy to trust a quantum model
            return name
        # nothing matched the hierarchy — fall back to any registered model
        return next(iter(self._registry))
