"""Phase 14 — Confirmatory Protocol: quantum-specific kernel analysis.

Closes the "kernel-target alignment, eigenvalue spectrum, effective rank,
within/between-class similarity, and classical-quantum geometric difference"
item from docs/paper.tex's Confirmatory Protocol (cites Huang et al. 2021,
already \\cite{huang2021} in the paper).

Computes, on a single n=100 balanced NF-ToN-IoT sample (deliberately small --
a full n x n Gram matrix costs O(n^2) quantum circuit evaluations, and this
item asks for one characterisation, not a sweep):

  - Quantum kernel Gram matrix K_Q via the actual deployed
    CyberSecurityFeatureMap + FidelityQuantumKernel (perception/feature_map.py).
  - Classical comparison kernel K_C: RBF (sklearn default gamma='scale') on
    the SAME points in the SAME [0, pi]-scaled feature space.
  - Kernel-target alignment A(K, y) = <K, yy^T>_F / (||K||_F ||yy^T||_F),
    y in {-1,+1}, for both K_Q and K_C.
  - Eigenvalue spectrum and effective rank (exp of the Shannon entropy of the
    normalised eigenvalue distribution) of K_Q.
  - Within-class vs between-class mean similarity under K_Q.
  - Classical-quantum geometric difference g(K1 || K2), Huang et al. 2021
    Eq. 4-ish definition:
        g(K1 || K2) = sqrt( || sqrt(K2) (K1 + lambda*I)^-1 sqrt(K2) ||_op )
    computed in BOTH directions. lambda = 1e-3 * n (our own regularisation
    choice, not reproduced from the original paper's exact hyperparameter --
    documented here for reproducibility/audit rather than left implicit).
    A large g(K_C || K_Q) means functions the quantum kernel represents
    easily are hard for the classical kernel to represent -- the necessary
    (not sufficient) condition for a quantum-kernel advantage.

Run:  ./venv/Scripts/python.exe -m experiments.phase14_kernel_alignment
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split

from agent import agent_config as config
from data.nf_loader import NFPreprocessor, read_nf
from data.sampling import select_kernel_subset
from perception.feature_map import CyberSecurityFeatureMap

RESULTS_DIR = "results/phase14"
N_SAMPLES = 100
REG_LAMBDA_FACTOR = 1e-3  # lambda = REG_LAMBDA_FACTOR * n


def _kernel_target_alignment(K: np.ndarray, y_pm1: np.ndarray) -> float:
    Y = np.outer(y_pm1, y_pm1)
    num = float(np.sum(K * Y))
    den = float(np.linalg.norm(K, "fro") * np.linalg.norm(Y, "fro"))
    return num / den if den > 0 else float("nan")


def _effective_rank(eigvals: np.ndarray) -> float:
    ev = np.clip(eigvals, 0, None)
    total = ev.sum()
    if total <= 0:
        return 0.0
    p = ev / total
    p = p[p > 1e-15]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def _within_between_similarity(K: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = len(y)
    same = np.zeros((n, n), dtype=bool)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        same[np.ix_(idx, idx)] = True
    off_diag = ~np.eye(n, dtype=bool)
    within = K[same & off_diag].mean()
    between = K[~same & off_diag].mean()
    return float(within), float(between)


def _geometric_difference(K1: np.ndarray, K2: np.ndarray, lam: float) -> float:
    """g(K1 || K2): how hard K1 is for K2 to represent."""
    n = K1.shape[0]
    eigval2, eigvec2 = np.linalg.eigh(K2)
    sqrt_K2 = eigvec2 @ np.diag(np.sqrt(np.clip(eigval2, 0, None))) @ eigvec2.T
    inv_term = np.linalg.inv(K1 + lam * np.eye(n))
    M = sqrt_K2 @ inv_term @ sqrt_K2
    max_eig = float(np.max(np.linalg.eigvalsh(M)))
    return float(np.sqrt(max(max_eig, 0.0)))


def main() -> None:
    print("=" * 70)
    print(f"Phase 14: kernel analysis  n={N_SAMPLES}")
    print("=" * 70)

    X_ton_df, y_ton_bin, _ = read_nf(config.NF_TON_CSV)
    X_ton_tr_df, _, y_ton_bin_tr, _ = train_test_split(
        X_ton_df, y_ton_bin, test_size=0.20, stratify=y_ton_bin,
        random_state=config.RANDOM_SEED)
    pre = NFPreprocessor()
    X8_ton_tr = pre.fit_transform(X_ton_tr_df)

    idx = select_kernel_subset(X8_ton_tr, y_ton_bin_tr, size=N_SAMPLES, method="kmeans")
    X, y = X8_ton_tr[idx], y_ton_bin_tr[idx]
    y_pm1 = np.where(y == 1, 1.0, -1.0)
    print(f"Sample: {X.shape}  labels {np.bincount(y)}")

    fm = CyberSecurityFeatureMap()
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    kernel = FidelityQuantumKernel(feature_map=fm)

    print("\nComputing quantum kernel Gram matrix (n^2 circuit evaluations) ...")
    t0 = time.time()
    K_Q = kernel.evaluate(X)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Computing classical RBF kernel Gram matrix ...")
    K_C = rbf_kernel(X)

    lam = REG_LAMBDA_FACTOR * N_SAMPLES
    eigval_Q = np.linalg.eigvalsh(K_Q)[::-1]  # descending

    kta_q = _kernel_target_alignment(K_Q, y_pm1)
    kta_c = _kernel_target_alignment(K_C, y_pm1)
    eff_rank_q = _effective_rank(eigval_Q)
    within_q, between_q = _within_between_similarity(K_Q, y)
    g_c_given_q = _geometric_difference(K_C, K_Q, lam)  # how hard K_C is for K_Q
    g_q_given_c = _geometric_difference(K_Q, K_C, lam)  # how hard K_Q is for K_C

    print(f"\nKernel-target alignment: quantum={kta_q:.4f}  classical={kta_c:.4f}")
    print(f"Effective rank (quantum): {eff_rank_q:.2f} / {N_SAMPLES}")
    print(f"Within-class similarity (quantum):  {within_q:.4f}")
    print(f"Between-class similarity (quantum): {between_q:.4f}")
    print(f"Geometric difference g(classical||quantum) = {g_c_given_q:.4f}")
    print(f"Geometric difference g(quantum||classical) = {g_q_given_c:.4f}")
    print(f"  (lambda = {lam:.4f} = {REG_LAMBDA_FACTOR} x n, our own regularisation choice)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "experiment": "phase14_kernel_alignment",
        "n_samples": N_SAMPLES,
        "reg_lambda": lam,
        "kernel_target_alignment": {"quantum": kta_q, "classical": kta_c},
        "eigenvalue_spectrum_quantum": eigval_Q.tolist(),
        "effective_rank_quantum": eff_rank_q,
        "within_class_similarity_quantum": within_q,
        "between_class_similarity_quantum": between_q,
        "geometric_difference": {
            "g_classical_given_quantum": g_c_given_q,
            "g_quantum_given_classical": g_q_given_c,
        },
    }
    path = f"{RESULTS_DIR}/phase14_kernel_alignment_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
