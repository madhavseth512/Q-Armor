"""Tests for the perception module: feature map + fidelity kernel + cache.

Kept to tiny matrices (n <= 12) because each kernel entry is a quantum circuit
evaluation (~ms each).
"""

from __future__ import annotations

import numpy as np

from agent import agent_config as config
from memory.memory import MemoryModule
from perception.feature_map import CyberSecurityFeatureMap
from perception.perception import PerceptionModule


def test_feature_map_structure():
    fm = CyberSecurityFeatureMap()
    assert fm.num_qubits == config.N_QUBITS == 8
    assert fm.num_parameters == 8
    ops = dict(fm.count_ops())
    assert ops["ry"] == 8 * config.FEATURE_MAP_REPS
    assert ops["rz"] == len(config.ENTANGLEMENT_PAIRS) * config.FEATURE_MAP_REPS
    assert ops["cx"] == 2 * len(config.ENTANGLEMENT_PAIRS) * config.FEATURE_MAP_REPS


def test_feature_map_binds_to_concrete_circuit():
    fm = CyberSecurityFeatureMap()
    bound = fm.assign_parameters(np.linspace(0.1, 3.0, 8))
    assert bound.num_parameters == 0


def _toy(n=8, seed=0):
    return np.random.default_rng(seed).uniform(0, np.pi, (n, 8))


def test_kernel_matrix_is_valid():
    X = _toy(10)
    K = PerceptionModule().compute_kernel_matrix(X)
    assert K.shape == (10, 10)
    assert np.allclose(K, K.T, atol=1e-8)               # symmetric
    assert np.allclose(np.diag(K), 1.0, atol=1e-6)      # unit diagonal
    assert K.min() >= -1e-9 and K.max() <= 1 + 1e-9     # range
    assert np.linalg.eigvalsh(K).min() > -1e-6          # PSD


def test_kernel_matrix_train_test_shape():
    pm = PerceptionModule()
    K = pm.compute_kernel_matrix(_toy(6), _toy(4, seed=1))
    assert K.shape == (4, 6)


def test_encode_returns_kernel_row():
    pm = PerceptionModule()
    Xtr = _toy(5)
    pm.set_training_data(Xtr)
    row = pm.encode(Xtr[0])
    assert row.shape == (5,)
    assert abs(row[0] - 1.0) < 1e-6                     # overlap with itself = 1


def test_kernel_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path))
    pm = PerceptionModule(memory=MemoryModule())
    x1, x2 = _toy(1)[0], _toy(1, seed=9)[0]
    v1 = pm.compute_kernel(x1, x2)
    v2 = pm.compute_kernel(x1, x2)                      # second call should hit cache
    assert v1 == v2
    hits, misses = pm._memory.get_cache_stats()
    assert hits >= 1
