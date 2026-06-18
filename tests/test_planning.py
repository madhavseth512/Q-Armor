"""Tests for the planning module: confidence, drift, noise, mitigation."""

from __future__ import annotations

from agent import agent_config as config
from planning.planning import PlanningModule


def test_low_confidence_flag():
    p = PlanningModule()
    flag = False
    for _ in range(config.CONFIDENCE_WINDOW_SIZE):
        flag = p.update_confidence(0.30)
    assert flag is True


def test_high_confidence_no_flag():
    p = PlanningModule()
    flag = False
    for _ in range(config.CONFIDENCE_WINDOW_SIZE):
        flag = p.update_confidence(0.95)
    assert flag is False


def test_partial_window_no_flag():
    p = PlanningModule()
    assert p.update_confidence(0.10) is False        # window not yet full


def test_adwin_detects_drift():
    p = PlanningModule()
    for _ in range(400):
        p.update_drift(0.0)
    fired = any(p.update_drift(1.0) for _ in range(400))
    assert fired is True


def test_mitigation_priority_order():
    p = PlanningModule()
    low = {"gate_error": 0.0, "readout_error": 0.0}
    assert p.decide_mitigation(False, False, low) == "none"
    assert p.decide_mitigation(False, True, low) == "switch_model"
    assert p.decide_mitigation(True, True, low) == "retrain_kernel"
    assert p.decide_mitigation(False, False, {"gate_error": 0.02, "readout_error": 0.0}) == "apply_zne"
    assert p.decide_mitigation(True, True, {"gate_error": 0.06, "readout_error": 0.0}) == "classical_fallback"


def test_poll_noise_returns_mock():
    n = PlanningModule().poll_noise()
    assert n["gate_error"] == config.MOCK_GATE_ERROR
    assert n["readout_error"] == config.MOCK_READOUT_ERROR
