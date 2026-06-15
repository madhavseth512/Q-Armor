"""End-to-end Phase 1 tests: loader + preprocess + two-sided sampling.

Uses small synthetic CSVs (no dependency on the multi-GB real dataset) so the
suite runs in well under a second. Validates the contracts Phase 2 relies on:
the 15-class assertion, dead-column drop, the smart-8 feature engineering, the
leak-free (0, pi) scaling + clip, and the dynamic-floor sampling helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent import agent_config as config
from data.loader import DatasetError, derive_parent_class, load_dataset
from data.preprocess import (
    QuantumPreprocessor,
    dynamic_smote_strategy,
    majority_undersample_index,
)

# Columns the synthetic frame must carry for loader + smart-8 engineering.
_NUMERIC_COLS = [
    "Rate", "syn_count", "rst_count", "fin_count", "Header_Length", "AVG",
    "IAT", "ack_count", "urg_count", "Covariance",
    "TCP", "UDP", "ICMP", "SSH", "DNS", "HTTPS",
    "Drate",  # degenerate column that the loader must drop
]


def _make_csv(path, classes, rows_per_class: int = 20, seed: int = 0) -> None:
    """Write a synthetic CICIoT2023-shaped CSV with the given parent classes."""
    rng = np.random.default_rng(seed)
    n = len(classes) * rows_per_class
    data = {c: rng.uniform(0, 1000, n) for c in _NUMERIC_COLS}
    data["Drate"] = np.zeros(n)  # degenerate
    for proto in ["TCP", "UDP", "ICMP", "SSH", "DNS", "HTTPS"]:
        data[proto] = rng.integers(0, 2, n).astype(float)
    labels = []
    for cls in classes:
        # mix single-token and Parent-Subtype labels
        sub = cls if cls in {"BenignTraffic", "XSS"} else f"{cls}-Variant"
        labels += [sub] * rows_per_class
    data["label"] = labels
    pd.DataFrame(data).to_csv(path, index=False)


@pytest.fixture
def full_csv(tmp_path):
    p = tmp_path / "train.csv"
    _make_csv(p, config.ATTACK_CLASSES)
    return str(p)


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #

def test_derive_parent_class():
    s = pd.Series(["DDoS-ICMP_Flood", "BenignTraffic", "XSS", "Mirai-greip_flood"])
    assert derive_parent_class(s).tolist() == ["DDoS", "BenignTraffic", "XSS", "Mirai"]


def test_load_dataset_basic(full_csv):
    X, y = load_dataset(full_csv)
    assert "label" not in X.columns
    assert "Drate" not in X.columns          # degenerate column dropped
    assert y.name == "parent"
    assert y.nunique() == config.N_PARENT_CLASSES == 15
    assert set(y.unique()) == set(config.ATTACK_CLASSES)
    assert len(X) == len(y)


def test_load_dataset_asserts_15(tmp_path):
    p = tmp_path / "short.csv"
    _make_csv(p, config.ATTACK_CLASSES[:14])  # one class missing
    with pytest.raises(DatasetError):
        load_dataset(str(p))


def test_load_dataset_missing_label_raises(tmp_path):
    p = tmp_path / "nolabel.csv"
    pd.DataFrame({"Rate": [1.0, 2.0]}).to_csv(p, index=False)
    with pytest.raises(DatasetError):
        load_dataset(str(p))


def test_validate_false_allows_subset(tmp_path):
    p = tmp_path / "subset.csv"
    _make_csv(p, config.ATTACK_CLASSES[:5])
    X, y = load_dataset(str(p), validate=False)   # must not raise
    assert y.nunique() == 5


# --------------------------------------------------------------------------- #
# preprocess: smart-8 engineering + leak-free scaling
# --------------------------------------------------------------------------- #

def test_engineer_features_are_smart8(full_csv):
    X, _ = load_dataset(full_csv)
    feats = QuantumPreprocessor().engineer_features(X)
    assert list(feats.columns) == config.QUANTUM_FEATURE_NAMES
    assert "flow_timing" in feats.columns and "handshake_ratio" in feats.columns
    assert "flow_dispersion" not in feats.columns  # swapped out
    assert not feats.isna().any().any()


def test_fit_transform_in_angle_range(full_csv):
    X, _ = load_dataset(full_csv)
    Z = QuantumPreprocessor().fit_transform(X)
    lo, hi = config.ANGLE_ENCODING_RANGE
    assert Z.shape == (len(X), 8)
    assert not np.isnan(Z).any()
    assert Z.min() >= lo - 1e-9 and Z.max() <= hi + 1e-9


def test_transform_before_fit_raises(full_csv):
    X, _ = load_dataset(full_csv)
    with pytest.raises(RuntimeError):
        QuantumPreprocessor().transform(X)


def test_transform_is_leak_free_and_clips(full_csv):
    """Scalers fit on TRAIN only; a test row beyond train range clips to pi."""
    X, _ = load_dataset(full_csv)
    pre = QuantumPreprocessor().fit(X)
    extreme = X.iloc[[0]].copy()
    for col in ["Rate", "IAT", "Header_Length", "AVG", "ack_count"]:
        extreme[col] = X[col].max() * 1e6  # far beyond anything seen in fit
    Z = pre.transform(extreme)
    lo, hi = config.ANGLE_ENCODING_RANGE
    assert Z.max() <= hi + 1e-9          # clipped, never exceeds pi
    assert Z.min() >= lo - 1e-9


# --------------------------------------------------------------------------- #
# two-sided sampling helpers
# --------------------------------------------------------------------------- #

def test_majority_undersample_caps():
    y = pd.Series(["A"] * 100 + ["B"] * 5)
    idx = majority_undersample_index(y, cap=10, seed=0)
    kept = y.loc[idx]
    assert (kept == "A").sum() == 10   # majority capped
    assert (kept == "B").sum() == 5    # minority untouched


def test_dynamic_smote_strategy_floor():
    y = pd.Series(["A"] * 50_000 + ["B"] * 100 + ["C"] * 4_000)
    strat = dynamic_smote_strategy(y, cap=50_000, multiplier=10)
    assert "A" not in strat                 # majority at cap -> no oversample
    assert strat["B"] == 1_000              # min(50k, 10*100) = dynamic floor
    assert strat["C"] == 40_000             # min(50k, 10*4000)


def test_resample_increases_minorities(full_csv):
    X, y = load_dataset(full_csv)
    Z = QuantumPreprocessor().fit_transform(X)
    strat = dynamic_smote_strategy(y, cap=50_000, multiplier=10)
    Xr, yr = QuantumPreprocessor.resample(Z, y, sampling_strategy=strat)
    counts = pd.Series(yr).value_counts()
    assert len(Xr) == len(yr) > len(Z)
    assert counts.min() >= 20              # every class grown to its floor
