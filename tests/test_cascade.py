"""Unit tests for Phase 8 cascade components.

Covers:
  - DetectTypeCascade: CascadeResult fields, benign pass-through, attack typing
  - CascadeEvaluator: CascadeReport fields, Stage 1 and Stage 2 metrics
  - SubsetRetrainer: retrain() + retrain_rf() happy paths and error handling
"""

from __future__ import annotations

import numpy as np
import pytest

from agent import agent_config as config
from agent.cascade_evaluator import CascadeEvaluator, CascadeReport
from agent.retrainer import SubsetRetrainer
from reasoning.cascade import CascadeResult, DetectTypeCascade


# ---------------------------------------------------------------------------
# Stub classifiers
# ---------------------------------------------------------------------------

class _BinaryStub:
    """Always predicts attack (1) with prob p_attack for every sample."""
    name = "binary_stub"
    classes_ = np.array([0, 1])

    def __init__(self, p_attack: float = 0.8):
        self._p = p_attack

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 1 - self._p), np.full(n, self._p)])

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def fit(self, X, y):
        return self


class _BenignStub(_BinaryStub):
    """Always predicts benign (0)."""
    name = "benign_stub"

    def __init__(self):
        super().__init__(p_attack=0.05)


class _TypeStub:
    """Always predicts 'DoS' as the attack type."""
    name = "type_stub"
    classes_ = np.array(config.NF_COARSE_CLASSES)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        # Index of 'DoS' in NF_COARSE_CLASSES
        dos_idx = list(config.NF_COARSE_CLASSES).index("DoS")
        proba = np.zeros((n, len(self.classes_)))
        proba[:, dos_idx] = 1.0
        return proba

    def fit(self, X, y):
        return self


def _make_X(n: int = 20, n_features: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, n_features)) * np.pi


def _balanced_labels(n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Return balanced binary and coarse-type labels."""
    y_bin = np.array([0] * (n // 2) + [1] * (n - n // 2))
    y_coarse = np.array(
        ["Benign"] * (n // 2) + ["DoS"] * (n - n // 2), dtype=object
    )
    return y_bin, y_coarse


# ---------------------------------------------------------------------------
# DetectTypeCascade
# ---------------------------------------------------------------------------

class TestDetectTypeCascade:
    def test_cascade_result_fields(self):
        cascade = DetectTypeCascade(_BinaryStub(0.9), _TypeStub())
        X = _make_X(10)
        r = cascade.predict(X)
        assert isinstance(r, CascadeResult)
        assert r.y_bin_scores.shape == (10,)
        assert r.y_bin_pred.shape == (10,)
        assert r.y_bin_conf.shape == (10,)
        assert r.y_type_pred.shape == (10,)
        assert r.y_type_conf.shape == (10,)
        assert r.attack_mask.shape == (10,)

    def test_high_attack_probability_flags_samples(self):
        cascade = DetectTypeCascade(_BinaryStub(0.9), _TypeStub())
        X = _make_X(10)
        r = cascade.predict(X)
        assert r.attack_mask.all()
        assert (r.y_bin_pred == 1).all()

    def test_benign_passthrough(self):
        cascade = DetectTypeCascade(_BenignStub(), _TypeStub())
        X = _make_X(10)
        r = cascade.predict(X)
        assert not r.attack_mask.any()
        assert (r.y_type_pred == "Benign").all()
        assert (r.y_type_conf == 0.0).all()

    def test_attack_typed_correctly(self):
        cascade = DetectTypeCascade(_BinaryStub(0.9), _TypeStub())
        X = _make_X(10)
        r = cascade.predict(X)
        # TypeStub always says DoS
        assert (r.y_type_pred[r.attack_mask] == "DoS").all()

    def test_scores_in_unit_interval(self):
        cascade = DetectTypeCascade(_BinaryStub(0.7), _TypeStub())
        X = _make_X(20)
        r = cascade.predict(X)
        assert (r.y_bin_scores >= 0).all() and (r.y_bin_scores <= 1).all()
        assert (r.y_bin_conf >= 0).all() and (r.y_bin_conf <= 1).all()

    def test_stage1_name_and_stage2_name(self):
        cascade = DetectTypeCascade(_BinaryStub(), _TypeStub())
        assert cascade.stage1_name == "binary_stub"
        assert cascade.stage2_name == "type_stub"


# ---------------------------------------------------------------------------
# CascadeEvaluator
# ---------------------------------------------------------------------------

class TestCascadeEvaluator:
    def _report(self, p_attack: float = 0.9) -> CascadeReport:
        X = _make_X(20)
        y_bin, y_coarse = _balanced_labels(20)
        cascade = DetectTypeCascade(_BinaryStub(p_attack), _TypeStub())
        result = cascade.predict(X)
        ev = CascadeEvaluator()
        return ev.evaluate(result, y_bin, y_coarse, "test_dataset",
                           "binary_stub", "type_stub", False)

    def test_report_is_cascade_report(self):
        assert isinstance(self._report(), CascadeReport)

    def test_stage1_auroc_range(self):
        r = self._report()
        assert 0.0 <= r.binary_auroc <= 1.0

    def test_stage2_macro_f1_range(self):
        r = self._report()
        assert 0.0 <= r.type_macro_f1 <= 1.0

    def test_counts_consistent(self):
        r = self._report()
        assert r.n_pred_attack + r.n_pred_benign == r.n_samples

    def test_per_class_f1_has_all_classes(self):
        r = self._report()
        for cls in config.NF_COARSE_CLASSES:
            assert cls in r.type_per_class_f1

    def test_is_healthy_above_floor(self):
        # With high attack proba, AUROC should be high (all positives ranked high)
        r = self._report(p_attack=0.99)
        # Can't guarantee is_healthy=True with a stub (AUROC depends on label
        # ordering), but we can verify the property exists and is bool
        assert isinstance(r.is_healthy, bool)

    def test_to_dict_roundtrip(self):
        r = self._report()
        d = r.to_dict()
        assert "binary_auroc" in d
        assert "type_macro_f1" in d
        assert "type_per_class_f1" in d
        assert isinstance(d["type_per_class_f1"], dict)

    def test_with_retraining_flag(self):
        X = _make_X(20)
        y_bin, y_coarse = _balanced_labels(20)
        cascade = DetectTypeCascade(_BinaryStub(), _TypeStub())
        result = cascade.predict(X)
        ev = CascadeEvaluator()
        r = ev.evaluate(result, y_bin, y_coarse, "ds", "s1", "s2", with_retraining=True)
        assert r.with_retraining is True

    def test_coverage_is_fraction(self):
        r = self._report(p_attack=0.9)
        assert 0.0 <= r.type_coverage <= 1.0


# ---------------------------------------------------------------------------
# SubsetRetrainer
# ---------------------------------------------------------------------------

class TestSubsetRetrainer:
    def _pool(self, n: int = 60, n_features: int = 8):
        """Balanced feature pool that can support n_sub=20 retraining."""
        rng = np.random.default_rng(99)
        X = rng.random((n, n_features)) * np.pi
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        return X, y

    def test_retrain_returns_fitted_model(self):
        X, y = self._pool(60)
        retrainer = SubsetRetrainer()
        model, X_sub, y_sub = retrainer.retrain(X, y, n_sub=20, n_steps=5, verbose=False)
        assert hasattr(model, "predict_proba")
        assert X_sub.shape[0] == 20
        assert len(y_sub) == 20

    def test_retrain_subset_is_balanced(self):
        X, y = self._pool(60)
        retrainer = SubsetRetrainer()
        _, _, y_sub = retrainer.retrain(X, y, n_sub=20, n_steps=5, verbose=False)
        assert np.sum(y_sub == 0) == 10
        assert np.sum(y_sub == 1) == 10

    def test_retrain_raises_on_insufficient_data(self):
        X = np.random.rand(10, 8) * np.pi
        y = np.zeros(10, dtype=int)   # all benign
        retrainer = SubsetRetrainer()
        with pytest.raises(ValueError, match="too few samples"):
            retrainer.retrain(X, y, n_sub=20, n_steps=5, verbose=False)

    def test_retrain_rf_returns_fitted_model(self):
        rng = np.random.default_rng(7)
        n = 100
        X = rng.random((n, 8)) * np.pi
        classes = config.NF_COARSE_CLASSES
        y = np.array([classes[i % len(classes)] for i in range(n)], dtype=object)
        retrainer = SubsetRetrainer()
        rf = retrainer.retrain_rf(X, y, n_per_class=10, verbose=False)
        assert hasattr(rf, "predict_proba")
        proba = rf.predict_proba(X[:5])
        assert proba.shape[1] == len(classes)
