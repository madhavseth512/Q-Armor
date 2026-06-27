"""Unit tests for Phase 7 Reflexion components.

Covers:
  - Evaluator: EpisodeReport fields and is_healthy property
  - EpisodicMemory: record, recall, policy mutations for all 4 lesson types
  - SelfReflector: all 4 heuristic rules, plus None (healthy) case
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from agent import agent_config as config
from agent.episodic_memory import (
    BINARY_ONLY,
    REINFORCE,
    SWITCH_MODEL,
    SWITCH_SUBSET,
    EpisodicMemory,
    Lesson,
)
from agent.evaluator import EpisodeReport, Evaluator
from agent.reflector import SelfReflector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_report(
    episode_id: int = 0,
    auroc: float = 0.80,
    binary_f1: float = 0.75,
    drift_detected: bool = False,
    model_used: str = "pegasos_qsvc",
) -> EpisodeReport:
    return EpisodeReport(
        episode_id      = episode_id,
        model_used      = model_used,
        dataset         = "NF-ToN-IoT",
        n_samples       = 100,
        auroc           = auroc,
        binary_f1       = binary_f1,
        aupr            = 0.75,
        fpr_at_tpr95    = 0.10,
        mean_confidence = 0.80,
        drift_detected  = drift_detected,
        timestamp       = "2026-01-01T00:00:00+00:00",
    )


def _memory(tmp_dir: str) -> EpisodicMemory:
    path = os.path.join(tmp_dir, "episodes.jsonl")
    return EpisodicMemory(log_path=path)


def _lesson(memory: EpisodicMemory, action: str, episode_id: int = 0, **params) -> Lesson:
    return Lesson(
        lesson_id  = memory.next_lesson_id(),
        episode_id = episode_id,
        action     = action,
        trigger    = "test trigger",
        params     = params,
        timestamp  = "2026-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

class TestEvaluator:
    def _arrays(self, n: int = 100, attack_frac: float = 0.5):
        rng = np.random.default_rng(42)
        y_true = np.array([1] * int(n * attack_frac) + [0] * (n - int(n * attack_frac)))
        y_scores = np.clip(y_true + rng.normal(0, 0.2, n), 0, 1)
        y_pred = (y_scores > 0.5).astype(int)
        confidences = np.abs(y_scores - 0.5) + 0.5
        return y_true, y_scores, y_pred, confidences

    def test_report_fields(self):
        ev = Evaluator()
        y_true, y_scores, y_pred, conf = self._arrays()
        r = ev.evaluate(0, "pegasos_qsvc", "NF-ToN-IoT", y_true, y_scores, y_pred, conf, False)
        assert isinstance(r, EpisodeReport)
        assert r.episode_id == 0
        assert r.model_used == "pegasos_qsvc"
        assert r.n_samples == 100
        assert 0.0 <= r.auroc <= 1.0
        assert 0.0 <= r.binary_f1 <= 1.0
        assert 0.0 <= r.aupr <= 1.0
        assert 0.0 <= r.fpr_at_tpr95 <= 1.0
        assert 0.0 <= r.mean_confidence <= 1.0
        assert isinstance(r.drift_detected, bool)
        assert r.timestamp.endswith("+00:00")

    def test_is_healthy_above_floor(self):
        r = _make_report(auroc=config.AUROC_FLOOR + 0.05)
        assert r.is_healthy is True

    def test_is_healthy_below_floor(self):
        r = _make_report(auroc=config.AUROC_FLOOR - 0.05)
        assert r.is_healthy is False

    def test_is_healthy_at_floor(self):
        r = _make_report(auroc=config.AUROC_FLOOR)
        assert r.is_healthy is True

    def test_to_dict_roundtrip(self):
        ev = Evaluator()
        y_true, y_scores, y_pred, conf = self._arrays()
        r = ev.evaluate(1, "vqc", "NF-UNSW-NB15", y_true, y_scores, y_pred, conf, True)
        d = r.to_dict()
        r2 = EpisodeReport(**d)
        assert r2.episode_id == r.episode_id
        assert r2.auroc == r.auroc
        assert r2.drift_detected is True


# ---------------------------------------------------------------------------
# EpisodicMemory tests
# ---------------------------------------------------------------------------

class TestEpisodicMemory:
    def test_empty_on_init(self, tmp_path):
        mem = _memory(str(tmp_path))
        assert len(mem) == 0

    def test_record_no_lesson(self, tmp_path):
        mem = _memory(str(tmp_path))
        r = _make_report(auroc=0.85)
        mem.record(r, None)
        assert len(mem) == 1
        records = mem.recall(1)
        assert records[0]["lesson"] == "NONE"

    def test_record_writes_jsonl(self, tmp_path):
        mem = _memory(str(tmp_path))
        mem.record(_make_report(), None)
        log_path = os.path.join(str(tmp_path), "episodes.jsonl")
        lines = open(log_path).readlines()
        assert len(lines) == 1
        import json
        entry = json.loads(lines[0])
        assert entry["record_type"] == "episode"

    def test_recall_last_n(self, tmp_path):
        mem = _memory(str(tmp_path))
        for i in range(5):
            mem.record(_make_report(episode_id=i), None)
        last2 = mem.recall(2)
        assert len(last2) == 2
        assert last2[0]["report"]["episode_id"] == 3
        assert last2[1]["report"]["episode_id"] == 4

    def test_recent_reports(self, tmp_path):
        mem = _memory(str(tmp_path))
        for i in range(4):
            mem.record(_make_report(episode_id=i, auroc=0.80 + i * 0.01), None)
        reps = mem.recent_reports(3)
        assert len(reps) == 3
        assert isinstance(reps[0], EpisodeReport)

    # -- Policy mutations --

    def test_policy_switch_subset(self, tmp_path):
        mem = _memory(str(tmp_path))
        assert mem.get_policy()["subset_reselect"] is False
        lesson = _lesson(mem, SWITCH_SUBSET)
        mem.record(_make_report(), lesson)
        assert mem.get_policy()["subset_reselect"] is True

    def test_acknowledge_reselect(self, tmp_path):
        mem = _memory(str(tmp_path))
        lesson = _lesson(mem, SWITCH_SUBSET)
        mem.record(_make_report(), lesson)
        mem.acknowledge_reselect()
        assert mem.get_policy()["subset_reselect"] is False

    def test_policy_switch_model_demotes(self, tmp_path):
        mem = _memory(str(tmp_path))
        h_before = list(mem.get_policy()["model_hierarchy"])
        lesson = _lesson(mem, SWITCH_MODEL, demoted_model=h_before[0])
        mem.record(_make_report(), lesson)
        h_after = mem.get_policy()["model_hierarchy"]
        # First tier should have moved down one position
        assert h_after[0] == h_before[1]
        assert h_after[1] == h_before[0]

    def test_policy_binary_only(self, tmp_path):
        mem = _memory(str(tmp_path))
        assert mem.get_policy()["binary_only"] is False
        lesson = _lesson(mem, BINARY_ONLY)
        mem.record(_make_report(), lesson)
        assert mem.get_policy()["binary_only"] is True

    def test_policy_reinforce_promotes(self, tmp_path):
        mem = _memory(str(tmp_path))
        # First demote the top model
        h0 = list(mem.get_policy()["model_hierarchy"])
        demote = _lesson(mem, SWITCH_MODEL, demoted_model=h0[0])
        mem.record(_make_report(), demote)
        h_mid = list(mem.get_policy()["model_hierarchy"])
        assert h_mid[0] != h0[0]  # was demoted

        # Now reinforce the original top model
        reinforce = _lesson(mem, REINFORCE, reinforced_model=h0[0])
        mem.record(_make_report(), reinforce)
        h_after = mem.get_policy()["model_hierarchy"]
        assert h_after[0] == h0[0]  # back at top

    def test_next_lesson_id_increments(self, tmp_path):
        mem = _memory(str(tmp_path))
        assert mem.next_lesson_id() == 1
        assert mem.next_lesson_id() == 2
        assert mem.next_lesson_id() == 3


# ---------------------------------------------------------------------------
# SelfReflector tests
# ---------------------------------------------------------------------------

class TestSelfReflector:
    def _run(self, mem, report) -> Lesson | None:
        reflector = SelfReflector()
        policy = mem.get_policy()
        return reflector.reflect(report, mem, policy)

    def test_healthy_returns_none(self, tmp_path):
        mem = _memory(str(tmp_path))
        r = _make_report(auroc=0.85, drift_detected=False)
        assert self._run(mem, r) is None

    def test_switch_subset_fires_on_drift_and_low_auroc(self, tmp_path):
        mem = _memory(str(tmp_path))
        r = _make_report(auroc=config.AUROC_FLOOR - 0.05, drift_detected=True)
        lesson = self._run(mem, r)
        assert lesson is not None
        assert lesson.action == SWITCH_SUBSET

    def test_switch_model_fires_on_low_auroc_no_drift(self, tmp_path):
        mem = _memory(str(tmp_path))
        r = _make_report(auroc=config.AUROC_FLOOR - 0.05, drift_detected=False)
        lesson = self._run(mem, r)
        assert lesson is not None
        assert lesson.action == SWITCH_MODEL
        assert "demoted_model" in lesson.params

    def test_binary_only_fires_on_multiclass_collapse(self, tmp_path):
        mem = _memory(str(tmp_path))
        # binary_f1 is used as macro-F1 proxy when model_used contains 'multiclass'
        r = _make_report(
            auroc=0.85,
            binary_f1=config.MACRO_F1_FLOOR - 0.05,
            model_used="rf_multiclass",
        )
        lesson = self._run(mem, r)
        assert lesson is not None
        assert lesson.action == BINARY_ONLY

    def test_binary_only_not_fired_if_already_set(self, tmp_path):
        mem = _memory(str(tmp_path))
        # Force binary_only into policy
        bl = _lesson(mem, BINARY_ONLY)
        mem.record(_make_report(), bl)
        # Now a multiclass collapse should NOT re-trigger
        r = _make_report(
            auroc=0.85,
            binary_f1=config.MACRO_F1_FLOOR - 0.05,
            model_used="rf_multiclass",
        )
        policy = mem.get_policy()
        assert policy["binary_only"] is True
        lesson = SelfReflector().reflect(r, mem, policy)
        assert lesson is None or lesson.action != BINARY_ONLY

    def test_reinforce_fires_after_n_consecutive_healthy(self, tmp_path):
        mem = _memory(str(tmp_path))
        n = config.REINFORCE_CONSECUTIVE
        # Record n-1 healthy episodes into memory first
        for i in range(n - 1):
            mem.record(_make_report(episode_id=i, auroc=0.85), None)
        # The n-th report (passed into reflect()) completes the streak
        r = _make_report(episode_id=n - 1, auroc=0.85)
        lesson = self._run(mem, r)
        assert lesson is not None
        assert lesson.action == REINFORCE

    def test_reinforce_does_not_fire_early(self, tmp_path):
        mem = _memory(str(tmp_path))
        n = config.REINFORCE_CONSECUTIVE
        # Only n-2 prior healthy records — not enough
        for i in range(n - 2):
            mem.record(_make_report(episode_id=i, auroc=0.85), None)
        r = _make_report(episode_id=n - 2, auroc=0.85)
        lesson = self._run(mem, r)
        # May fire REINFORCE only if exactly n in window — should be None here
        assert lesson is None

    def test_switch_subset_takes_priority_over_switch_model(self, tmp_path):
        """Rule 2 (drift) fires before rule 3 (no drift) when drift=True."""
        mem = _memory(str(tmp_path))
        r = _make_report(auroc=config.AUROC_FLOOR - 0.1, drift_detected=True)
        lesson = self._run(mem, r)
        assert lesson.action == SWITCH_SUBSET
