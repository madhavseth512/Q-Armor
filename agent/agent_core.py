"""Agent orchestrator: wires the five modules into one decision loop.

Flow per sample:
    Perception -> Memory(load, at init) -> Reasoning(select + predict)
    -> Planning(confidence/drift/noise -> mitigation) -> Action(classify/alert/
    defence/output) -> Memory(update).

Phase 3 runs entirely on the classical RandomForest; the noise monitor is mocked.
Quantum models register into the same ModelSelector in Phase 5 without changing
this loop.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from action.action import ActionModule
from agent import agent_config as config
from data.preprocess import QuantumPreprocessor
from memory.memory import MemoryModule
from planning.planning import PlanningModule
from reasoning.classical import RandomForestModel
from reasoning.selector import ModelSelector

PREPROCESSOR_NAME = "preprocessor"
BASELINE_MODEL_NAME = "random_forest"
# Inference-time data-size context (training pool is large; quantum not skipped for
# data reasons). Real per-run value can be supplied later.
_DATA_SIZE_CONTEXT = 1_000_000


class AgentCore:
    """Orchestrates perception, reasoning, memory, planning, and action."""

    def __init__(self) -> None:
        """Instantiate all modules (models are attached via :meth:`load`)."""
        self.memory = MemoryModule()
        self.planning = PlanningModule()
        self.action = ActionModule()
        self.selector = ModelSelector()
        self._preprocessor: QuantumPreprocessor | None = None
        self._last_confidence = 1.0
        self._current_model = BASELINE_MODEL_NAME

    def load(self) -> None:
        """Restore the fitted preprocessor and classical model from memory.

        Raises:
            FileNotFoundError: If the preprocessor / model were not set up first
                (run ``experiments/phase3_setup.py``).
        """
        self._preprocessor = self.memory.load_model(PREPROCESSOR_NAME)
        self.selector.register(self.memory.load_model(BASELINE_MODEL_NAME))

    def _as_frame(self, sample: Any) -> pd.DataFrame:
        """Coerce a raw sample (dict / Series / 1-row DataFrame) to a DataFrame."""
        if isinstance(sample, pd.DataFrame):
            return sample
        if isinstance(sample, pd.Series):
            return sample.to_frame().T
        return pd.DataFrame([sample])

    def process(
        self,
        sample: Any,
        source_ip: str,
        true_label: str | None = None,
    ) -> dict[str, Any]:
        """Run one raw sample through the full agent loop.

        Args:
            sample: One raw CICIoT row (dict / Series / 1-row DataFrame).
            source_ip: Source IP for alert prioritisation / repeat tracking.
            true_label: Ground-truth parent class, if known (enables drift update).

        Returns:
            The structured result dict (prediction / alert / defence / planning).
        """
        if self._preprocessor is None:
            raise RuntimeError("AgentCore.load() must be called before process().")

        # 1. Perception
        features = self._preprocessor.transform(self._as_frame(sample))[0]

        # 2. Reasoning — select a model, predict
        noise = self.planning.poll_noise()
        model_name = self.selector.select(
            noise_level=noise["gate_error"], data_size=_DATA_SIZE_CONTEXT,
            confidence=self._last_confidence, current_model=self._current_model)
        self._current_model = model_name
        label, confidence, used_model = self.selector.get(model_name).predict(features)
        self._last_confidence = confidence

        # 3. Planning — confidence / drift / noise -> mitigation
        low_conf = self.planning.update_confidence(confidence)
        drift = self.planning.update_drift(0.0 if true_label == label else 1.0) \
            if true_label is not None else False
        mitigation = self.planning.decide_mitigation(drift, low_conf, noise)
        planning_signals = {
            "low_confidence": low_conf, "drift_detected": drift,
            "noise": noise, "mitigation": mitigation,
        }

        # 4. Action — classify, prioritise, recommend, assemble
        attack = self.action.classify_attack(label, confidence)
        priority = self.action.prioritise_alert(label, confidence, source_ip)
        defence = self.action.recommend_defence(label)
        result = self.action.build_output(attack, priority, defence, used_model,
                                          source_ip, planning_signals)
        return result

    def run_stream(
        self,
        samples: pd.DataFrame,
        source_ips: list[str] | None = None,
        true_labels: list[str] | None = None,
        log: bool = True,
        stream: bool = False,
    ) -> list[dict[str, Any]]:
        """Process a batch of raw samples through the agent loop.

        Args:
            samples: DataFrame of raw CICIoT rows.
            source_ips: Per-row source IPs (defaults to a placeholder).
            true_labels: Per-row ground-truth labels, if available.
            log: Append each result to the alert log.
            stream: Print each result to the CLI.

        Returns:
            List of structured result dicts.
        """
        results = []
        for i in range(len(samples)):
            ip = source_ips[i] if source_ips else "0.0.0.0"
            truth = true_labels[i] if true_labels else None
            res = self.process(samples.iloc[[i]], ip, truth)
            if log:
                self.action.log_to_file(res)
            if stream:
                self.action.stream_cli(res)
            results.append(res)
        self.save()
        return results

    def save(self) -> None:
        """Persist monitor state (confidence window, ADWIN) to memory."""
        self.memory.save_state({
            "confidence_window": self.planning.get_confidence_window(),
            "last_confidence": self._last_confidence,
            "current_model": self._current_model,
            "ip_history": self.action.get_ip_history(),
        })
