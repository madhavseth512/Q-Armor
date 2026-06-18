"""Tests for the memory module: persistence + kernel cache."""

from __future__ import annotations

import numpy as np

from agent import agent_config as config
from memory.memory import MemoryModule


def test_model_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path))
    m = MemoryModule()
    m.save_model({"a": 1}, "thing")
    assert m.has_model("thing")
    assert m.load_model("thing") == {"a": 1}


def test_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path))
    m = MemoryModule()
    assert m.load_state() == {}                       # nothing saved yet
    m.save_state({"window": [1, 2, 3]})
    assert m.load_state()["window"] == [1, 2, 3]


def test_kernel_cache_symmetric_and_stats(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path))
    m = MemoryModule()
    a, b = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    assert m.get_cached_kernel(a, b) is None          # miss
    m.cache_kernel(a, b, 0.5)
    assert m.get_cached_kernel(a, b) == 0.5           # hit
    assert m.get_cached_kernel(b, a) == 0.5           # symmetric hit
    assert m.get_cache_stats() == (2, 1)


def test_kernel_cache_lru_eviction(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path))
    monkeypatch.setattr(config, "KERNEL_CACHE_MAX_SIZE", 2)
    m = MemoryModule()
    for i in range(3):
        m.cache_kernel(np.array([float(i)]), np.array([float(i)]), float(i))
    assert m.get_cached_kernel(np.array([0.0]), np.array([0.0])) is None   # oldest evicted
    assert m.get_cached_kernel(np.array([2.0]), np.array([2.0])) == 2.0
