"""Phase 4 Step 5: verify the quantum kernel and time its computation.

On a small stratified real subset, builds the CyberSecurityFeatureMap fidelity
kernel and checks it is a VALID kernel (symmetric, unit-diagonal, range [0,1],
positive semi-definite). Also reports a class-separability sanity signal, compares
against the classical RBF kernel, and TIMES the O(n^2) computation at several sizes
so the Phase-5 training subset size can be chosen from data.

Run:  ./venv/Scripts/python.exe -m experiments.phase4_kernel_check
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from agent import agent_config as config
from experiments.phase2_baselines import prepare
from perception.perception import PerceptionModule


def _stratified(X, y, n, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    per = max(1, n // len(np.unique(y)))
    idx = []
    for c in np.unique(y):
        pos = np.where(y == c)[0]
        idx += list(rng.choice(pos, min(per, len(pos)), replace=False))
    return np.array(idx)


def main() -> None:
    X8_real, ytr, _, _ = prepare()
    perception = PerceptionModule()

    print("\n=== kernel computation timing (O(n^2) circuit evaluations) ===")
    for n in (100, 200, 400):
        idx = _stratified(X8_real, ytr, n)
        Xs = X8_real[idx]
        t = time.time()
        K = perception.compute_kernel_matrix(Xs)
        dt = time.time() - t
        print(f"  n={len(Xs):<4} kernel matrix {K.shape} in {dt:6.1f}s "
              f"({dt/len(Xs)**2*1e3:.2f} ms/entry)")

    # full validity check on the largest subset
    idx = _stratified(X8_real, ytr, 400)
    Xs, ys = X8_real[idx], ytr[idx]
    K = perception.compute_kernel_matrix(Xs)

    print("\n=== kernel validity ===")
    print("  symmetric        :", np.allclose(K, K.T, atol=1e-8))
    print("  unit diagonal    :", np.allclose(np.diag(K), 1.0, atol=1e-6))
    print("  range in [0,1]   :", float(K.min()) >= -1e-9 and float(K.max()) <= 1 + 1e-9,
          f"(min {K.min():.4f}, max {K.max():.4f})")
    eig_min = float(np.linalg.eigvalsh(K).min())
    print(f"  PSD (min eigval) : {eig_min:.2e}  ->", "PSD" if eig_min > -1e-6 else "NOT PSD")

    print("\n=== class-separability sanity (mean off-diagonal kernel) ===")
    same = K[(ys[:, None] == ys[None, :]) & ~np.eye(len(ys), dtype=bool)].mean()
    cross = K[ys[:, None] != ys[None, :]].mean()
    print(f"  same-class mean : {same:.4f}")
    print(f"  cross-class mean: {cross:.4f}")
    print(f"  separation      : {same - cross:+.4f}  ({'same > cross (good)' if same > cross else 'no separation'})")

    print("\n=== vs classical RBF kernel (same subset) ===")
    Krbf = rbf_kernel(Xs, gamma="scale" if False else 1.0)
    flat_q, flat_c = K[np.triu_indices(len(Xs), 1)], Krbf[np.triu_indices(len(Xs), 1)]
    print(f"  corr(quantum, RBF) off-diagonal: {np.corrcoef(flat_q, flat_c)[0,1]:.3f}")

    print("\n(test.csv untouched)")


if __name__ == "__main__":
    main()
