"""End-to-end tests for the agent orchestrator (on synthetic data, no disk setup)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agent import agent_config as config
from agent.agent_core import AgentCore
from data.preprocess import QuantumPreprocessor
from reasoning.classical import RandomForestModel
from reasoning.selector import ModelSelector

RAW_COLS = ["Rate", "syn_count", "rst_count", "fin_count", "Header_Length", "AVG",
            "IAT", "ack_count", "urg_count", "Covariance", "Drate"]


def _raw_frame(n_per: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    for cls in config.ATTACK_CLASSES:
        d = {col: rng.uniform(0, 1000, n_per) for col in RAW_COLS}
        for proto in ["TCP", "UDP", "ICMP", "SSH", "DNS", "HTTPS"]:
            d[proto] = rng.integers(0, 2, n_per).astype(float)
        d["Drate"] = np.zeros(n_per)
        parts.append(pd.DataFrame(d))
        labels += [cls] * n_per
    return pd.concat(parts).reset_index(drop=True), pd.Series(labels)


def _agent_with_models(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ALERT_LOG_PATH", str(tmp_path / "alerts.jsonl"))
    X, y = _raw_frame()
    pre = QuantumPreprocessor()
    rf = RandomForestModel().fit(pre.fit_transform(X), y.to_numpy())
    agent = AgentCore()
    agent._preprocessor = pre
    agent.selector.register(rf)
    return agent, X


def test_process_returns_valid_schema(tmp_path, monkeypatch):
    agent, X = _agent_with_models(tmp_path, monkeypatch)
    res = agent.process(X.iloc[[0]], "9.9.9.9")
    assert set(res) == {"timestamp", "source_ip", "prediction", "alert", "defence", "planning"}
    assert res["prediction"]["attack_type"] in config.ATTACK_CLASSES
    assert 0.0 <= res["prediction"]["confidence"] <= 1.0
    assert res["alert"]["priority"] >= 0.0
    assert {"action", "urgency", "description"} <= set(res["defence"])
    assert res["planning"]["mitigation"] in {
        "none", "switch_model", "retrain_kernel", "apply_zne", "classical_fallback"}


def test_run_stream_and_persist(tmp_path, monkeypatch):
    agent, X = _agent_with_models(tmp_path, monkeypatch)
    out = agent.run_stream(X.iloc[:5], source_ips=["a", "b", "c", "a", "a"], log=True)
    assert len(out) == 5
    assert (tmp_path / "alerts.jsonl").exists()
    assert agent.memory.load_state()["ip_history"]["a"] == 3


def test_process_before_load_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path))
    agent = AgentCore()
    with __import__("pytest").raises(RuntimeError):
        agent.process({"Rate": 1.0}, "1.1.1.1")


def test_selector_returns_classical_only():
    sel = ModelSelector()
    sel.register(RandomForestModel())
    assert sel.select(noise_level=0.0, data_size=10**6, confidence=0.9) == "random_forest"
    # even under critical noise, classical is the only option
    assert sel.select(noise_level=0.99, data_size=10**6, confidence=0.9) == "random_forest"
