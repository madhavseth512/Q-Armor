"""Tests for Phase 5 quantum models and binary label utilities.

All quantum circuit evaluations use tiny synthetic data (n <= 10) to keep
each test under a few seconds. The tests verify:
  - correct output shapes and dtypes
  - predict() contract (label, confidence, model_name)
  - predict_proba() column convention: col0=P(benign=0), col1=P(attack=1)
  - predict() agrees with argmax(predict_proba())
  - to_binary() correctness
"""

from __future__ import annotations

import numpy as np
import pytest

from data.binary import BENIGN_LABEL, to_binary
from reasoning.quantum import (
    PegasosQSVCModel,
    QuantumAnomalyDetector,
    VQCModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng_X(n: int, seed: int = 0) -> np.ndarray:
    """Small feature matrix in [0, pi], the expected quantum input range."""
    return np.random.default_rng(seed).uniform(0, np.pi, (n, 8))


def _rng_y(n: int, seed: int = 0) -> np.ndarray:
    """Binary labels (0=benign, 1=attack), roughly balanced."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n)


def _balanced_y(n: int) -> np.ndarray:
    """Exactly balanced binary labels (n//2 each class)."""
    half = n // 2
    return np.array([0] * half + [1] * (n - half))


# ---------------------------------------------------------------------------
# to_binary
# ---------------------------------------------------------------------------

def test_to_binary_benign():
    y = np.array(["BenignTraffic", "DDoS", "Mirai", "BenignTraffic"])
    expected = np.array([0, 1, 1, 0])
    np.testing.assert_array_equal(to_binary(y), expected)


def test_to_binary_all_attack():
    y = np.array(["DDoS", "DoS", "XSS"])
    np.testing.assert_array_equal(to_binary(y), np.array([1, 1, 1]))


def test_to_binary_all_benign():
    y = np.full(5, BENIGN_LABEL)
    np.testing.assert_array_equal(to_binary(y), np.zeros(5, dtype=int))


# ---------------------------------------------------------------------------
# PegasosQSVCModel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_qsvc():
    n = 10
    X, y = _rng_X(n), _balanced_y(n)
    model = PegasosQSVCModel(num_steps=20)
    model.fit(X, y)
    return model, X, y


def test_qsvc_classes(fitted_qsvc):
    model, _, _ = fitted_qsvc
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))


def test_qsvc_predict_proba_shape(fitted_qsvc):
    model, X, _ = fitted_qsvc
    proba = model.predict_proba(X[:3])
    assert proba.shape == (3, 2)


def test_qsvc_predict_proba_sums_to_one(fitted_qsvc):
    model, X, _ = fitted_qsvc
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_qsvc_predict_proba_range(fitted_qsvc):
    model, X, _ = fitted_qsvc
    proba = model.predict_proba(X)
    assert proba.min() >= -1e-9
    assert proba.max() <= 1.0 + 1e-9


def test_qsvc_predict_contract(fitted_qsvc):
    model, X, _ = fitted_qsvc
    label, conf, name = model.predict(X[0])
    assert label in ("0", "1")
    assert 0.0 <= conf <= 1.0
    assert name == "pegasos_qsvc"


def test_qsvc_predict_consistent_with_proba(fitted_qsvc):
    model, X, _ = fitted_qsvc
    for i in range(min(5, len(X))):
        proba = model.predict_proba(X[i : i + 1])[0]
        pred_label, pred_conf, _ = model.predict(X[i])
        expected_idx = int(np.argmax(proba))
        assert pred_label == str(model.classes_[expected_idx])
        assert abs(pred_conf - proba[expected_idx]) < 1e-6


# ---------------------------------------------------------------------------
# VQCModel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_vqc():
    n = 8
    X, y = _rng_X(n, seed=1), _balanced_y(n)
    model = VQCModel(max_iter=5, reps=1)
    model.fit(X, y)
    return model, X, y


def test_vqc_classes(fitted_vqc):
    model, _, _ = fitted_vqc
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))


def test_vqc_predict_proba_shape(fitted_vqc):
    model, X, _ = fitted_vqc
    proba = model.predict_proba(X[:3])
    assert proba.shape == (3, 2)


def test_vqc_predict_proba_sums_to_one(fitted_vqc):
    model, X, _ = fitted_vqc
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-3)


def test_vqc_predict_proba_range(fitted_vqc):
    model, X, _ = fitted_vqc
    proba = model.predict_proba(X)
    assert proba.min() >= -1e-9
    assert proba.max() <= 1.0 + 1e-9


def test_vqc_predict_contract(fitted_vqc):
    model, X, _ = fitted_vqc
    label, conf, name = model.predict(X[0])
    assert label in ("0", "1")
    assert 0.0 <= conf <= 1.0
    assert name == "vqc"


def test_vqc_predict_consistent_with_proba(fitted_vqc):
    model, X, _ = fitted_vqc
    for i in range(min(5, len(X))):
        proba = model.predict_proba(X[i : i + 1])[0]
        pred_label, pred_conf, _ = model.predict(X[i])
        expected_idx = int(np.argmax(proba))
        assert pred_label == str(model.classes_[expected_idx])
        assert abs(pred_conf - proba[expected_idx]) < 1e-3


# ---------------------------------------------------------------------------
# QuantumAnomalyDetector
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_qae():
    n = 10
    X = _rng_X(n, seed=2)
    y = _balanced_y(n)
    model = QuantumAnomalyDetector(max_iter=3, shots=64, seed=42)
    model.fit(X, y)
    return model, X, y


def test_qae_classes(fitted_qae):
    model, _, _ = fitted_qae
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))


def test_qae_anomaly_score_shape(fitted_qae):
    model, X, _ = fitted_qae
    scores = model.anomaly_score(X)
    assert scores.shape == (len(X),)


def test_qae_anomaly_score_range(fitted_qae):
    model, X, _ = fitted_qae
    scores = model.anomaly_score(X)
    assert scores.min() >= -1e-9
    assert scores.max() <= 1.0 + 1e-9


def test_qae_predict_proba_shape(fitted_qae):
    model, X, _ = fitted_qae
    proba = model.predict_proba(X[:3])
    assert proba.shape == (3, 2)


def test_qae_predict_proba_sums_to_one(fitted_qae):
    model, X, _ = fitted_qae
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_qae_predict_proba_col1_is_anomaly_score(fitted_qae):
    model, X, _ = fitted_qae
    scores = model.anomaly_score(X)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba[:, 1], scores, atol=1e-9)


def test_qae_predict_contract(fitted_qae):
    model, X, _ = fitted_qae
    label, conf, name = model.predict(X[0])
    assert label in ("0", "1")
    assert 0.0 <= conf <= 1.0
    assert name == "quantum_autoencoder"


def test_qae_fit_benign_only_raises():
    model = QuantumAnomalyDetector(max_iter=1, shots=64)
    X = _rng_X(5)
    y = np.ones(5, dtype=int)  # all attack — no benign rows
    with pytest.raises(ValueError, match="No benign samples"):
        model.fit(X, y)
