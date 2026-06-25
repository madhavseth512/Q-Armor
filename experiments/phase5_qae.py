"""Phase 5 — Experiment E6c: Quantum Autoencoder anomaly detection.

Trains the QuantumAnomalyDetector on the BENIGN subset of the kernel
training pool, then evaluates its anomaly score against the classical
OneClassSVM baseline and the supervised SVM-RBF (as an upper-bound
reference) on the same held-out validation subset.

The QAE contributes to the Phase 5 model zoo as an *unsupervised* anomaly
detector — it learns the structure of normal traffic without seeing attack
labels, which is a realistic deployment scenario. AUROC / AUPR are the
primary metrics (the score is a continuous anomaly score, not a hard label).

Run:  ./venv/Scripts/python.exe -m experiments.phase5_qae
Timing: QAE fit with max_iter=30 at n_benign≈75 ≈ 5–10 min.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.svm import OneClassSVM

from agent import agent_config as config
from data.binary import to_binary
from data.sampling import select_kernel_subset
from experiments.phase2_baselines import prepare
from experiments.phase5_qsvc import _fpr_at_tpr, _eval_binary
from reasoning.classical import SVMModel
from reasoning.quantum import QuantumAnomalyDetector

RESULTS_DIR = "results/phase5"


def _eval_one_class(name: str, y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Metrics for a continuous anomaly score (no decision threshold tuned).

    Args:
        name: Model identifier.
        y_true: True binary labels (0=benign, 1=attack).
        scores: Anomaly scores; higher = more attack-like.

    Returns:
        Metrics dict with AUROC, AUPR, FPR@TPR95.
        F1 uses the 0.5 threshold (informative but not the primary metric).
    """
    y_pred = (scores > 0.5).astype(int)
    return {
        "model": name,
        "binary_f1_at_0.5": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, scores)),
        "aupr": float(average_precision_score(y_true, scores)),
        "fpr_at_tpr95": _fpr_at_tpr(y_true, scores),
    }


def main() -> None:
    print("=== Phase 5 E6c: Quantum Autoencoder anomaly detection ===\n")

    # ── 1. Load + binary labels ───────────────────────────────────────────
    print("Loading CICIoT2023 (real undersampled pool)...")
    X8_real, ytr_parent, X8_va, yva_parent = prepare()
    y_bin_tr = to_binary(ytr_parent)
    y_bin_va = to_binary(yva_parent)
    print(f"  train pool: {X8_real.shape}  binary: {np.bincount(y_bin_tr)}")
    print(f"  val:        {X8_va.shape}    binary: {np.bincount(y_bin_va)}")

    # ── 2. Kernel subset (balanced, same size as E6a) ─────────────────────
    n = config.QSVC_SUBSET_SIZE
    print(f"\nSelecting kernel subset n={n} (k-means, class-balanced)...")
    t0 = time.time()
    idx = select_kernel_subset(X8_real, y_bin_tr, size=n, method="kmeans")
    X_sub, y_sub = X8_real[idx], y_bin_tr[idx]
    n_benign_sub = int((y_sub == 0).sum())
    print(f"  subset: {X_sub.shape}  labels: {np.bincount(y_sub)}"
          f"  (benign n={n_benign_sub})  ({time.time()-t0:.1f}s)")

    # ── 3. Validation subset ──────────────────────────────────────────────
    rng = np.random.default_rng(config.RANDOM_SEED)
    n_va = min(1000, len(X8_va))
    va_idx_b = np.where(y_bin_va == 0)[0]
    va_idx_a = np.where(y_bin_va == 1)[0]
    va_idx = np.concatenate([
        rng.choice(va_idx_b, min(n_va // 2, len(va_idx_b)), replace=False),
        rng.choice(va_idx_a, min(n_va // 2, len(va_idx_a)), replace=False),
    ])
    X_va_sub, y_va_sub = X8_va[va_idx], y_bin_va[va_idx]
    print(f"\nValidation subset: {X_va_sub.shape}  binary: {np.bincount(y_va_sub)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_metrics = []

    # ── 4. Classical OneClassSVM baseline (trained on benign only) ─────────
    print("\n--- OneClassSVM (classical unsupervised baseline, RBF kernel) ---")
    t1 = time.time()
    X_benign_sub = X_sub[y_sub == 0]
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    ocsvm.fit(X_benign_sub)
    # decision_function: lower = more anomalous; flip sign for attack score
    ocsvm_scores = -ocsvm.decision_function(X_va_sub)
    print(f"  fit+predict in {time.time()-t1:.1f}s")
    ocsvm_metrics = _eval_one_class("one_class_svm", y_va_sub, ocsvm_scores)
    print(f"  AUROC={ocsvm_metrics['auroc']:.4f}  AUPR={ocsvm_metrics['aupr']:.4f}"
          f"  FPR@TPR95={ocsvm_metrics['fpr_at_tpr95']:.4f}"
          f"  F1@0.5={ocsvm_metrics['binary_f1_at_0.5']:.4f}")
    all_metrics.append(ocsvm_metrics)

    # ── 5. Supervised SVM-RBF reference (upper bound) ─────────────────────
    print("\n--- SVM-RBF (supervised reference -- upper bound for comparison) ---")
    t1 = time.time()
    svm = SVMModel(class_weight="balanced").fit(X_sub, y_sub)
    svm_proba = svm.predict_proba(X_va_sub)
    print(f"  fit+predict in {time.time()-t1:.1f}s")
    svm_metrics = _eval_one_class(
        "svm_rbf_supervised",
        y_va_sub,
        svm_proba[:, 1],   # attack probability = anomaly score
    )
    print(f"  AUROC={svm_metrics['auroc']:.4f}  AUPR={svm_metrics['aupr']:.4f}"
          f"  FPR@TPR95={svm_metrics['fpr_at_tpr95']:.4f}"
          f"  F1@0.5={svm_metrics['binary_f1_at_0.5']:.4f}")
    all_metrics.append(svm_metrics)

    # ── 6. Quantum Autoencoder ────────────────────────────────────────────
    print("\n--- QuantumAnomalyDetector (quantum autoencoder, benign-only training) ---")
    print(f"  n_trash={config.QAE_N_TRASH}, reps={config.QAE_REPS},"
          f" max_iter={config.QAE_MAX_ITER}, shots={config.QAE_SHOTS}")
    print(f"  training on {n_benign_sub} benign samples...")
    t1 = time.time()
    qae = QuantumAnomalyDetector(
        n_trash=config.QAE_N_TRASH,
        reps=config.QAE_REPS,
        max_iter=config.QAE_MAX_ITER,
        shots=config.QAE_SHOTS,
    ).fit(X_sub, y_sub)
    print(f"  fit in {time.time()-t1:.1f}s")
    t1 = time.time()
    qae_scores = qae.anomaly_score(X_va_sub)
    print(f"  inference in {time.time()-t1:.1f}s")
    qae_metrics = _eval_one_class("quantum_autoencoder", y_va_sub, qae_scores)
    print(f"  AUROC={qae_metrics['auroc']:.4f}  AUPR={qae_metrics['aupr']:.4f}"
          f"  FPR@TPR95={qae_metrics['fpr_at_tpr95']:.4f}"
          f"  F1@0.5={qae_metrics['binary_f1_at_0.5']:.4f}")
    all_metrics.append(qae_metrics)

    # ── 7. Save results ───────────────────────────────────────────────────
    out = {
        "experiment": "phase5_qae",
        "subset_size": int(n),
        "n_benign_train": int(n_benign_sub),
        "val_size": int(len(y_va_sub)),
        "val_attack_frac": float(y_va_sub.mean()),
        "models": all_metrics,
    }
    path = f"{RESULTS_DIR}/phase5_qae_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved -> {path}")

    # ── 8. Summary ────────────────────────────────────────────────────────
    print("\n=== SUMMARY (anomaly detection, val subset) ===")
    print(f"{'Model':<30}  {'AUROC':>7}  {'AUPR':>7}  {'FPR@95':>8}  {'F1@0.5':>7}")
    print("-" * 70)
    for m in all_metrics:
        print(f"  {m['model']:<28}  {m['auroc']:>7.4f}  {m['aupr']:>7.4f}"
              f"  {m['fpr_at_tpr95']:>8.4f}  {m['binary_f1_at_0.5']:>7.4f}")
    print("\n(test.csv untouched)")


if __name__ == "__main__":
    main()
