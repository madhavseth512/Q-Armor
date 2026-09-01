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


def test_qsvc_predict_consistent_with_predict_labels(fitted_qsvc):
    """predict() must follow predict_labels() (native margin), not argmax(proba) —
    PegasosQSVC's sigmoid calibration is not centred at 0.5, so the two can
    legitimately disagree. See predict_labels() docstring in reasoning/quantum.py."""
    model, X, _ = fitted_qsvc
    labels = model.predict_labels(X)
    for i in range(min(5, len(X))):
        proba = model.predict_proba(X[i : i + 1])[0]
        pred_label, pred_conf, _ = model.predict(X[i])
        expected_idx = int(np.where(model.classes_ == labels[i])[0][0])
        assert pred_label == str(model.classes_[expected_idx])
        assert abs(pred_conf - proba[expected_idx]) < 1e-6


def test_qsvc_predict_batch_consistent_with_predict_labels(fitted_qsvc):
    model, X, _ = fitted_qsvc
    labels = model.predict_labels(X)
    batch = model.predict_batch(X)
    for i, (pred_label, _, _) in enumerate(batch):
        assert pred_label == str(labels[i])


# ---------------------------------------------------------------------------
# Score-audit protocol (paper Sec. "Score-Audit Protocol" / Table
# "Validation protocol status") -- items 1-4 not yet automated at the time
# the paper was written; verified here.
# ---------------------------------------------------------------------------

def test_qsvc_label_to_internal_pm1_mapping():
    """Item 1: external {0,1} labels map to PegasosQSVC's internal +-1 labels
    exactly as documented in PegasosQSVCModel's docstring (external 0=benign
    -> internal +1 = label_pos; external 1=attack -> internal -1 = label_neg)."""
    n = 20
    X, y = _rng_X(n, seed=10), _balanced_y(n)
    model = PegasosQSVCModel(num_steps=20).fit(X, y)
    assert model._model._label_pos == 0
    assert model._model._label_neg == 1
    assert model._model._label_map[0] == 1
    assert model._model._label_map[1] == -1


def test_qsvc_predict_matches_sign_decision_function(fitted_qsvc):
    """Item 2: predict_labels() (native margin) must agree with
    sign(decision_function) under the label_pos/label_neg convention:
    decision_function > 0 -> benign (label_pos=0), < 0 -> attack (label_neg=1)."""
    model, X, _ = fitted_qsvc
    dec = model._model.decision_function(X)
    assert not np.any(dec == 0), "toy fixture produced an exact-zero margin; regenerate"
    labels = model.predict_labels(X)
    expected = np.where(dec > 0, 0, 1)
    np.testing.assert_array_equal(labels, expected)


def test_qsvc_attack_score_orientation():
    """Item 3: increasing s_atk = predict_proba[:, 1] corresponds to the
    attack class. The raw decision_function's positive direction is benign
    (label_pos), so s_atk must anti-correlate with it: every sample the raw
    margin calls attack (decision_function < 0) must score s_atk >= 0.5, and
    every sample it calls benign must score s_atk < 0.5."""
    n = 30
    X, y = _rng_X(n, seed=11), _balanced_y(n)
    model = PegasosQSVCModel(num_steps=30).fit(X, y)
    dec = model._model.decision_function(X)
    s_atk = model.predict_proba(X)[:, 1]
    attack_mask = dec < 0
    assert np.all(s_atk[attack_mask] >= 0.5)
    assert np.all(s_atk[~attack_mask] < 0.5)


def _manual_pairwise_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Reference AUROC via the Mann-Whitney U definition -- P(score_pos >
    score_neg), ties counted as 0.5 -- independent of sklearn's algorithm."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    wins = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
    return wins / (len(pos) * len(neg))


def test_auroc_pipeline_matches_manual_toy_examples():
    """Item 4: sklearn's roc_auc_score, as used throughout this project for
    every reported AUROC, must agree with an independent pairwise (Mann-
    Whitney) computation on toy examples with an answer known by inspection:
    perfect separation (1.0), perfect reversal (0.0), all-tied scores (0.5),
    and one case with a single ranking violation (checked against the
    independent manual formula, not a hand-typed constant)."""
    from sklearn.metrics import roc_auc_score

    y = np.array([0, 0, 0, 1, 1, 1])

    perfectly_separated = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert roc_auc_score(y, perfectly_separated) == 1.0
    assert _manual_pairwise_auroc(y, perfectly_separated) == 1.0

    perfectly_reversed = np.array([0.7, 0.8, 0.9, 0.1, 0.2, 0.3])
    assert roc_auc_score(y, perfectly_reversed) == 0.0
    assert _manual_pairwise_auroc(y, perfectly_reversed) == 0.0

    all_tied = np.full(6, 0.5)
    assert roc_auc_score(y, all_tied) == 0.5
    assert _manual_pairwise_auroc(y, all_tied) == 0.5

    one_violation = np.array([0.1, 0.2, 0.9, 0.4, 0.8, 0.95])
    manual = _manual_pairwise_auroc(y, one_violation)
    sklearn_auroc = roc_auc_score(y, one_violation)
    assert abs(manual - sklearn_auroc) < 1e-12
    assert 0.0 < manual < 1.0  # sanity: this fixture is not a degenerate case


def test_qsvc_auroc_on_linearly_separable_toy_set():
    """Item 4 (model-facing): the live PegasosQSVCModel's own s_atk =
    predict_proba[:, 1] must yield AUROC = 1.0 on a trivially, maximally
    separable 8-feature toy set (benign near 0, attack near pi) -- the
    simplest possible hand-verifiable case for the actual scoring pipeline
    used to report results."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(42)
    n_per_class = 6
    X_benign = rng.uniform(0, 0.3, (n_per_class, 8))
    X_attack = rng.uniform(np.pi - 0.3, np.pi, (n_per_class, 8))
    X = np.vstack([X_benign, X_attack])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    model = PegasosQSVCModel(num_steps=50).fit(X, y)
    s_atk = model.predict_proba(X)[:, 1]
    assert roc_auc_score(y, s_atk) == 1.0


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
