"""Action module: attack classification, alert prioritisation, defence output.

Turns a reasoning prediction into a structured, actionable response — the final
stage of the agent loop. Inference-only and self-contained (no model dependency):
it consumes a (label, confidence, model_name) prediction plus the planning
signals, and emits a structured result dict suitable for JSON logging and CLI
streaming.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np

from agent import agent_config as config


class ActionModule:
    """Classifies attacks, prioritises alerts, and recommends defences."""

    def __init__(self) -> None:
        """Initialise scoring tables from config and an empty source-IP history."""
        self._severity = config.SEVERITY_WEIGHTS
        self._defence_rules = config.DEFENCE_RULES
        self._repeat_boost = config.REPEAT_IP_BOOST
        self._ip_history: dict[str, int] = {}

    def classify_attack(self, label: str, confidence: float) -> dict[str, Any]:
        """Map a predicted class into a labelled attack record.

        Args:
            label: Predicted parent class.
            confidence: Model confidence in [0, 1].

        Returns:
            Dict with ``attack_type``, ``confidence``, and ``is_attack`` (False for
            BenignTraffic).
        """
        return {
            "attack_type": label,
            "confidence": float(confidence),
            "is_attack": label != "BenignTraffic",
        }

    def prioritise_alert(self, attack_type: str, confidence: float, source_ip: str) -> float:
        """Compute an alert priority and update the source-IP history.

        Priority = confidence * severity, plus ``REPEAT_IP_BOOST`` if the source IP
        has been seen before. BenignTraffic has severity 0 -> priority 0.

        Args:
            attack_type: Predicted class.
            confidence: Model confidence in [0, 1].
            source_ip: Source IP of the sample.

        Returns:
            Priority score (>= 0; may exceed 1.0 for repeat offenders, so the most
            persistent sources rank highest).
        """
        repeat = self._ip_history.get(source_ip, 0) > 0
        self._ip_history[source_ip] = self._ip_history.get(source_ip, 0) + 1
        score = confidence * self._severity.get(attack_type, 0.0)
        if repeat and attack_type != "BenignTraffic":
            score += self._repeat_boost
        return float(score)

    def recommend_defence(self, attack_type: str) -> dict[str, Any]:
        """Look up the rule-based defence recommendation for an attack type.

        Args:
            attack_type: One of the 15 parent classes.

        Returns:
            Dict with ``action``, ``urgency``, and ``description``.
        """
        return dict(self._defence_rules[attack_type])

    def is_repeat_offender(self, source_ip: str) -> bool:
        """Whether ``source_ip`` has produced more than one alert this session."""
        return self._ip_history.get(source_ip, 0) > 1

    def build_output(
        self,
        attack_record: dict[str, Any],
        priority: float,
        defence: dict[str, Any],
        model_name: str,
        source_ip: str,
        planning: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the final structured result for one inference step.

        Args:
            attack_record: Output of :meth:`classify_attack`.
            priority: Output of :meth:`prioritise_alert`.
            defence: Output of :meth:`recommend_defence`.
            model_name: Name of the model that produced the prediction.
            source_ip: Source IP of the sample.
            planning: Planning signals (low_confidence, drift_detected, noise,
                mitigation).

        Returns:
            Structured dict (see module/agent docs for the schema).
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_ip": source_ip,
            "prediction": {
                "attack_type": attack_record["attack_type"],
                "confidence": attack_record["confidence"],
                "is_attack": attack_record["is_attack"],
                "model": model_name,
            },
            "alert": {
                "priority": round(priority, 4),
                "severity": self._severity.get(attack_record["attack_type"], 0.0),
                "repeat_offender": self.is_repeat_offender(source_ip),
            },
            "defence": defence,
            "planning": planning,
        }

    def log_to_file(self, result: dict[str, Any], log_path: str | None = None) -> None:
        """Append a result dict as one JSON line to the alert log.

        Args:
            result: Structured result from :meth:`build_output`.
            log_path: Destination JSONL path; defaults to ``ALERT_LOG_PATH``.
        """
        path = log_path or config.ALERT_LOG_PATH
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    def stream_cli(self, result: dict[str, Any]) -> None:
        """Print a one-line human-readable summary of a result to stdout.

        Args:
            result: Structured result from :meth:`build_output`.
        """
        p, d = result["prediction"], result["defence"]
        flag = "ATTACK " if p["is_attack"] else "benign "
        repeat = " [repeat]" if result["alert"]["repeat_offender"] else ""
        print(f"[{flag}] {p['attack_type']:<20} conf={p['confidence']:.2f} "
              f"prio={result['alert']['priority']:.2f}{repeat} "
              f"({d['urgency']}) -> {d['action']}")

    def get_ip_history(self) -> dict[str, int]:
        """Return per-source-IP alert counts for this session."""
        return dict(self._ip_history)
