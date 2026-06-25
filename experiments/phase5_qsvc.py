"""Phase 5 — Experiment E6a: PegasosQSVC vs SVM-RBF on binary CICIoT2023.

Compares the Phase-4 fidelity quantum kernel (via PegasosQSVC) against the
classical RBF-SVM on the IDENTICAL small balanced subset (n = QSVC_SUBSET_SIZE).
This is the fair quantum-vs-classical binary-detection benchmark required by
ROADMAP §5 (E6) and the supervisor metric table.

Metrics: binary F1, AUROC, AUPR, FPR@TPR95.
Subset selector: select_kernel_subset (k-means, class-balanced) — identical
indices reused for both models.

Run:  ./venv/Scripts/python.exe -m experiments.phase5_qsvc
Timing: kernel matrix build for n=150 ≈ 5 min (Phase-4 measured); total
        runtime including QSVC fit ≈ 10–15 min.
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

from agent import agent_config as config
from data.binary import to_binary
from data.sampling import select_kernel_subset
from experiments.phase2_baselines import prepare
from reasoning.classical import SVMModel
from reasoning.quantum import PegasosQSVCModel

RESULTS_DIR = "results/phase5"


def _fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, tpr_target: float = 0.95) -> float:
    """False-positive rate at the threshold that achieves tpr_target recall."""
    from sklearn.metrics import roc_curve
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr_arr, tpr_target)
    idx = min(idx, len(fpr_arr) - 1)
    return float(fpr_arr[idx])


def _eval_binary(name: str, y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Compute the binary detection metrics for a model.

    Args:
        name: Model identifier for the results dict.
        y_true: True binary labels (0=benign, 1=attack).
        proba: predict_proba output (m, 2); column 1 is the attack score.

    Returns:
        Metrics dict with F1, AUROC, AUPR, FPR@TPR95.
    """
    y_score = proba[:, 1]
    y_pred = proba.argmax(axis=1)
    return {
        "model": name,
        "binary_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "aupr": float(average_precision_score(y_true, y_score)),
        "fpr_at_tpr95": _fpr_at_tpr(y_true, y_score),
    }


def main() -> None:
    print("=== Phase 5 E6a: PegasosQSVC vs SVM-RBF (binary) ===\n")

    # ── 1. Load data and prepare binary labels ────────────────────────────
    print("Loading CICIoT2023 (real undersampled pool)...")
    X8_real, ytr_parent, X8_va, yva_parent = prepare()
    y_bin_tr = to_binary(ytr_parent)
    y_bin_va = to_binary(yva_parent)
    print(f"  train pool: {X8_real.shape}  binary: {np.bincount(y_bin_tr)} (benign, attack)")
    print(f"  val:        {X8_va.shape}    binary: {np.bincount(y_bin_va)}")

    # ── 2. Select the IDENTICAL small kernel subset ───────────────────────
    n = config.QSVC_SUBSET_SIZE
    print(f"\nSelecting kernel subset n={n} (k-means, class-balanced)...")
    t0 = time.time()
    idx = select_kernel_subset(X8_real, y_bin_tr, size=n, method="kmeans")
    X_sub, y_sub = X8_real[idx], y_bin_tr[idx]
    print(f"  subset: {X_sub.shape}  labels: {np.bincount(y_sub)} (benign, attack)"
          f"  ({time.time()-t0:.1f}s)")

    # ── 3. Build and evaluate the binary validation set ───────────────────
    # QSVC prediction loops one test sample at a time against all support
    # vectors, so cost = n_val × n_sv × kernel_time. At n_sv≈50 and 7ms/eval
    # this is ~70s per 200 samples (manageable) vs 12 min at 2000. VQC/QAE
    # batch internally so they can afford larger val sets.
    rng = np.random.default_rng(config.RANDOM_SEED)
    n_va = min(200, len(X8_va))
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

    # ── 4. SVM-RBF baseline (same subset) ────────────────────────────────
    print("\n--- SVM-RBF (classical baseline on identical subset) ---")
    t1 = time.time()
    svm = SVMModel(class_weight="balanced").fit(X_sub, y_sub)
    print(f"  fit in {time.time()-t1:.1f}s")
    t1 = time.time()
    svm_proba = svm.predict_proba(X_va_sub)
    print(f"  predict in {time.time()-t1:.1f}s")
    svm_metrics = _eval_binary("svm_rbf_binary", y_va_sub, svm_proba)
    print(f"  F1={svm_metrics['binary_f1']:.4f}  AUROC={svm_metrics['auroc']:.4f}"
          f"  AUPR={svm_metrics['aupr']:.4f}  FPR@TPR95={svm_metrics['fpr_at_tpr95']:.4f}")
    all_metrics.append(svm_metrics)

    # ── 5. PegasosQSVC (quantum) ──────────────────────────────────────────
    print("\n--- PegasosQSVC (quantum fidelity kernel) ---")
    t1 = time.time()
    qsvc = PegasosQSVCModel(num_steps=config.PEGASOS_TAU).fit(X_sub, y_sub)
    print(f"  fit in {time.time()-t1:.1f}s")
    t1 = time.time()
    qsvc_proba = qsvc.predict_proba(X_va_sub)
    print(f"  predict in {time.time()-t1:.1f}s")
    qsvc_metrics = _eval_binary("pegasos_qsvc_binary", y_va_sub, qsvc_proba)
    print(f"  F1={qsvc_metrics['binary_f1']:.4f}  AUROC={qsvc_metrics['auroc']:.4f}"
          f"  AUPR={qsvc_metrics['aupr']:.4f}  FPR@TPR95={qsvc_metrics['fpr_at_tpr95']:.4f}")
    all_metrics.append(qsvc_metrics)

    # ── 6. Save results ───────────────────────────────────────────────────
    out = {
        "experiment": "phase5_qsvc",
        "subset_size": int(n),
        "val_size": int(len(y_va_sub)),
        "val_attack_frac": float(y_va_sub.mean()),
        "models": all_metrics,
    }
    path = f"{RESULTS_DIR}/phase5_qsvc_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved -> {path}")

    # ── 7. Summary ────────────────────────────────────────────────────────
    print("\n=== SUMMARY (binary detection, val subset) ===")
    print(f"{'Model':<28}  {'F1':>6}  {'AUROC':>7}  {'AUPR':>7}  {'FPR@95':>8}")
    print("-" * 64)
    for m in all_metrics:
        print(f"  {m['model']:<26}  {m['binary_f1']:>6.4f}  {m['auroc']:>7.4f}"
              f"  {m['aupr']:>7.4f}  {m['fpr_at_tpr95']:>8.4f}")
    print("\n(test.csv untouched)")


if __name__ == "__main__":
    main()
