"""Tests for the action module: classification, alerting, defence, output."""

from __future__ import annotations

from action.action import ActionModule
from agent import agent_config as config


def test_classify_benign_vs_attack():
    a = ActionModule()
    assert a.classify_attack("BenignTraffic", 0.9)["is_attack"] is False
    assert a.classify_attack("DDoS", 0.9)["is_attack"] is True


def test_priority_math_and_repeat_boost():
    a = ActionModule()
    assert abs(a.prioritise_alert("DDoS", 1.0, "1.1.1.1") - 0.75) < 1e-9        # 1.0*0.75
    assert abs(a.prioritise_alert("DDoS", 1.0, "1.1.1.1") - 0.90) < 1e-9        # +0.15 repeat
    assert a.is_repeat_offender("1.1.1.1")


def test_benign_priority_is_zero():
    a = ActionModule()
    assert a.prioritise_alert("BenignTraffic", 1.0, "2.2.2.2") == 0.0


def test_defence_covers_all_15_classes():
    a = ActionModule()
    for c in config.ATTACK_CLASSES:
        d = a.recommend_defence(c)
        assert {"action", "urgency", "description"} <= set(d)


def test_build_output_schema():
    a = ActionModule()
    rec = a.classify_attack("Mirai", 0.95)
    out = a.build_output(rec, a.prioritise_alert("Mirai", 0.95, "3.3.3.3"),
                         a.recommend_defence("Mirai"), "random_forest", "3.3.3.3",
                         {"low_confidence": False, "drift_detected": False,
                          "noise": {}, "mitigation": "none"})
    assert set(out) == {"timestamp", "source_ip", "prediction", "alert", "defence", "planning"}
    assert out["prediction"]["attack_type"] == "Mirai"
    assert out["prediction"]["model"] == "random_forest"
