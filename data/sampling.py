"""Kernel-scale subset selection for Q-Armor.

Kernel methods (RBF-SVM now, QSVC in Phase 5) scale ~O(n^2) and cannot train on
the full dataset, so they train on a small, carefully chosen subset. Naive random
sampling is a poor choice here: our data is heavily redundant (DDoS has millions
of near-identical flood rows) and rare classes would be under-represented.

``select_kernel_subset`` therefore:
  1. allocates the budget across classes by water-filling — every class is
     represented, rare classes contribute all their real rows, and leftover budget
     flows to larger classes (class-balanced, bounded by availability);
  2. selects WITHIN each class either by ``"kmeans"`` representative sampling
     (cluster the class, take the point nearest each centroid — covers the feature
     space, avoids redundant near-duplicates) or by stratified ``"random"``.

It returns positional row indices, so the identical subset can be reused for both
the classical SVM and the quantum QSVC (a fair kernel-vs-kernel comparison).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

from agent import agent_config as config


def allocate_per_class(y: np.ndarray, size: int) -> dict:
    """Water-fill a subset budget across classes, bounded by availability.

    Processing classes from rarest to most common, each receives an equal share
    of the remaining budget capped at how many real rows it has; any shortfall
    from rare classes flows to the larger ones.

    Args:
        y: Label array.
        size: Total subset size to allocate.

    Returns:
        Mapping ``{class_label: n_to_select}``.
    """
    classes, counts = np.unique(y, return_counts=True)
    order = np.argsort(counts)  # rarest first
    alloc: dict = {}
    remaining = size
    n_left = len(classes)
    for i in order:
        share = remaining // n_left
        take = int(min(counts[i], share))
        alloc[classes[i]] = take
        remaining -= take
        n_left -= 1
    return alloc


def _kmeans_representatives(Xc: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Pick ``n`` representative row positions within a single class.

    Clusters the class into ``n`` groups and returns the point nearest each
    centroid, giving feature-space coverage instead of redundant near-duplicates.

    Args:
        Xc: Feature rows for one class, shape (n_class, n_features).
        n: Number of representatives to select (assumed < len(Xc)).
        seed: Random seed.

    Returns:
        Positional indices into ``Xc`` (length ``n``).
    """
    km = MiniBatchKMeans(n_clusters=n, random_state=seed, n_init=3,
                         batch_size=max(256, 3 * n))
    km.fit(Xc)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, Xc)
    sel = np.unique(closest)
    if len(sel) < n:  # two centroids shared a nearest point — top up randomly
        rng = np.random.default_rng(seed)
        pool = np.setdiff1d(np.arange(len(Xc)), sel)
        extra = rng.choice(pool, n - len(sel), replace=False)
        sel = np.concatenate([sel, extra])
    return sel


def select_kernel_subset(
    X: np.ndarray,
    y: np.ndarray,
    size: int,
    method: str = "kmeans",
    seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    """Select a class-balanced, representative subset for kernel training.

    Args:
        X: Scaled feature matrix (the space the kernel operates in), shape
            (n_samples, n_features).
        y: Label array of shape (n_samples,).
        size: Target subset size (actual size may be slightly smaller if rare
            classes cannot supply their full share).
        method: ``"kmeans"`` (representative) or ``"random"`` (stratified).
        seed: Random seed for reproducibility.

    Returns:
        Positional row indices into ``X`` / ``y`` for the selected subset.

    Raises:
        ValueError: If ``method`` is not ``"kmeans"`` or ``"random"``.
    """
    if method not in {"kmeans", "random"}:
        raise ValueError(f"method must be 'kmeans' or 'random', got {method!r}")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    alloc = allocate_per_class(y, size)

    picks: list[np.ndarray] = []
    for cls, n in alloc.items():
        if n <= 0:
            continue
        pos = np.where(y == cls)[0]
        if n >= len(pos):
            picks.append(pos)
        elif method == "random":
            picks.append(rng.choice(pos, n, replace=False))
        else:
            picks.append(pos[_kmeans_representatives(X[pos], n, seed)])

    return np.concatenate(picks)
