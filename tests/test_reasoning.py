"""Contract tests for the reasoning module (classical + hierarchical models).

Verifies every model honours the BaseClassifier contract — predict returns
(str label, float confidence in [0,1], str model_name), batch works, probabilities
are well-formed, and save/load round-trips — on small synthetic data (fast).
"""

from __future__ import annotations

import numpy as np
import pytest

from agent import agent_config as config
from reasoning.base import BaseClassifier
from reasoning.classical import RandomForestModel, SVMModel
from reasoning.hierarchical import HierarchicalClassifier


def _toy_parents(per_class: int = 12, seed: int = 0):
    """Synthetic (X, y) covering all 15 parent classes, separable-ish in 8-D."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for i, c in enumerate(config.ATTACK_CLASSES):
        centre = rng.uniform(0, np.pi, 8)
        Xs.append(np.clip(centre + rng.normal(0, 0.15, (per_class, 8)), 0, np.pi))
        ys += [c] * per_class
    return np.vstack(Xs), np.array(ys)


@pytest.mark.parametrize("Model", [RandomForestModel, SVMModel])
def test_classical_contract(Model, tmp_path):
    X, y = _toy_parents()
    model = Model().fit(X, y)
    assert isinstance(model, BaseClassifier)

    label, conf, name = model.predict(X[0])
    assert isinstance(label, str) and label in config.ATTACK_CLASSES
    assert isinstance(conf, float) and 0.0 <= conf <= 1.0
    assert name == model.name

    batch = model.predict_batch(X[:5])
    assert len(batch) == 5 and all(len(t) == 3 for t in batch)

    proba = model.predict_proba(X[:10])
    assert proba.shape[0] == 10
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    p = tmp_path / "m.joblib"
    model.save(str(p))
    reloaded = BaseClassifier.load(str(p))
    assert reloaded.predict(X[0])[0] == label


@pytest.mark.parametrize("web_reject", [False, True])
def test_hierarchical_contract(web_reject):
    X, y = _toy_parents(per_class=15)
    hc = HierarchicalClassifier(web_reject=web_reject).fit(X, y)

    proba = hc.predict_proba(X[:20])
    assert proba.shape == (20, 15)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)   # reject preserves total mass

    label, conf, name = hc.predict(X[0])
    assert label in config.ATTACK_CLASSES and 0.0 <= conf <= 1.0
    assert name == "hierarchical_rf"


def test_predict_single_matches_batch():
    X, y = _toy_parents()
    model = RandomForestModel().fit(X, y)
    assert model.predict(X[3]) == model.predict_batch(X[3:4])[0]
