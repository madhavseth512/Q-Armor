"""Planning module: confidence monitoring, drift detection, noise, mitigation.

The adaptive decision layer of the agent. It watches three health signals —
prediction confidence (sliding window), distribution drift (ADWIN on the
prediction-error stream), and backend noise — and decides which mitigation
strategy to apply (retrain kernel / switch model / apply ZNE / classical fallback
/ none).

In Phase 3 the noise monitor is mocked (no quantum backend yet); its interface is
real and is wired to live backend calibration in Phase 4/6.
"""

from __future__ import annotations

from collections import deque

from river import drift

from agent import agent_config as config


class PlanningModule:
    """Monitors confidence, drift, and noise; selects a mitigation action."""

    def __init__(self) -> None:
        """Initialise the sliding confidence window and the ADWIN drift detector."""
        self._conf_window: deque[float] = deque(maxlen=config.CONFIDENCE_WINDOW_SIZE)
        self._adwin = drift.ADWIN(delta=config.ADWIN_DELTA)

    def update_confidence(self, confidence: float) -> bool:
        """Push a confidence score into the sliding window.

        Args:
            confidence: Confidence of the latest prediction in [0, 1].

        Returns:
            True if the window is full and its mean is below
            ``CONFIDENCE_THRESHOLD`` (a sustained low-confidence signal).
        """
        self._conf_window.append(float(confidence))
        if len(self._conf_window) < self._conf_window.maxlen:
            return False
        mean_conf = sum(self._conf_window) / len(self._conf_window)
        return mean_conf < config.CONFIDENCE_THRESHOLD

    def update_drift(self, prediction_error: float) -> bool:
        """Feed a prediction error into ADWIN and report whether drift fired.

        Args:
            prediction_error: 1.0 if the last prediction was wrong, else 0.0.
                Requires ground truth, so this is only called during evaluation /
                replay (delayed labels), not pure inference.

        Returns:
            True if ADWIN detected a distribution shift.
        """
        self._adwin.update(float(prediction_error))
        return bool(self._adwin.drift_detected)

    def poll_noise(self) -> dict[str, float]:
        """Return current backend gate/readout error rates.

        Phase 3: returns the mock values from config. In Phase 4/6 this polls the
        live IBM backend calibration.

        Returns:
            Dict with ``gate_error`` and ``readout_error`` in [0, 1].
        """
        return {"gate_error": config.MOCK_GATE_ERROR,
                "readout_error": config.MOCK_READOUT_ERROR}

    def decide_mitigation(
        self,
        drift_detected: bool,
        low_confidence: bool,
        noise_metrics: dict[str, float],
    ) -> str:
        """Choose a mitigation action from the current signals.

        Priority order: critical noise > drift > low confidence > high noise > none.

        Args:
            drift_detected: Whether ADWIN fired.
            low_confidence: Whether the confidence monitor flagged a sustained drop.
            noise_metrics: Output of :meth:`poll_noise`.

        Returns:
            One of ``"classical_fallback"``, ``"retrain_kernel"``,
            ``"switch_model"``, ``"apply_zne"``, ``"none"``.
        """
        gate = noise_metrics.get("gate_error", 0.0)
        readout = noise_metrics.get("readout_error", 0.0)

        if gate >= config.NOISE_THRESHOLD_FALLBACK:
            return "classical_fallback"
        if drift_detected:
            return "retrain_kernel"
        if low_confidence:
            return "switch_model"
        if gate >= config.NOISE_THRESHOLD_ZNE or readout >= config.READOUT_ERROR_THRESHOLD:
            return "apply_zne"
        return "none"

    def reset_drift_detector(self) -> None:
        """Reset ADWIN after a retrain has addressed the detected drift."""
        self._adwin = drift.ADWIN(delta=config.ADWIN_DELTA)

    def get_confidence_window(self) -> list[float]:
        """Return the current confidence window contents (oldest first)."""
        return list(self._conf_window)
