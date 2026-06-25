"""Phase 5 — Experiment E6b: VQC binary classifier on CICIoT2023.

Trains a Variational Quantum Circuit (CyberSecurityFeatureMap + RealAmplitudes
ansatz, COBYLA optimiser) on the same kernel subset used in E6a
(phase5_qsvc.py), then evaluates against the same SVM-RBF baseline on the
same validation subset for a consistent three-way comparison.

Imports the subset-selection and baseline code from phase5_qsvc so all three
models (SVM-RBF, QSVC, VQC) share identical train/eval conditions.

Run:  ./venv/Scripts/python.exe -m experiments.phase5_vqc
Timing: VQC training at n=150 with max_iter=100 ≈ 15–30 min.
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
from experiments.phase5_qsvc import _fpr_at_tpr, _eval_binary
from reasoning.classical import SVMModel
from reasoning.quantum import VQCModel

RESULTS_DIR = "results/phase5"


def main() -> None:
    print("=== Phase 5 E6b: VQC binary classifier ===\n")

    # ── 1. Load + binary labels ───────────────────────────────────────────
    print("Loading CICIoT2023 (real undersampled pool)...")
    X8_real, ytr_parent, X8_va, yva_parent = prepare()
    y_bin_tr = to_binary(ytr_parent)
    y_bin_va = to_binary(yva_parent)
    print(f"  train pool: {X8_real.shape}  binary: {np.bincount(y_bin_tr)}")
    print(f"  val:        {X8_va.shape}    binary: {np.bincount(y_bin_va)}")

    # ── 2. Kernel subset (same seed + size as E6a for comparability) ──────
    n = config.QSVC_SUBSET_SIZE
    print(f"\nSelecting kernel subset n={n} (k-means)...")
    t0 = time.time()
    idx = select_kernel_subset(X8_real, y_bin_tr, size=n, method="kmeans")
    X_sub, y_sub = X8_real[idx], y_bin_tr[idx]
    print(f"  subset: {X_sub.shape}  labels: {np.bincount(y_sub)}  ({time.time()-t0:.1f}s)")

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

    # ── 4. SVM-RBF (same-subset baseline for reference) ───────────────────
    print("\n--- SVM-RBF (same-subset baseline) ---")
    t1 = time.time()
    svm = SVMModel(class_weight="balanced").fit(X_sub, y_sub)
    svm_proba = svm.predict_proba(X_va_sub)
    print(f"  fit+predict in {time.time()-t1:.1f}s")
    svm_metrics = _eval_binary("svm_rbf_binary", y_va_sub, svm_proba)
    print(f"  F1={svm_metrics['binary_f1']:.4f}  AUROC={svm_metrics['auroc']:.4f}"
          f"  AUPR={svm_metrics['aupr']:.4f}  FPR@TPR95={svm_metrics['fpr_at_tpr95']:.4f}")
    all_metrics.append(svm_metrics)

    # ── 5. VQC ────────────────────────────────────────────────────────────
    print("\n--- VQC (CyberSecurityFeatureMap + RealAmplitudes, COBYLA) ---")
    print(f"  max_iter={config.VQC_MAX_ITER}, reps={config.VQC_REPS}")
    t1 = time.time()
    vqc = VQCModel(max_iter=config.VQC_MAX_ITER).fit(X_sub, y_sub)
    print(f"  fit in {time.time()-t1:.1f}s")
    t1 = time.time()
    vqc_proba = vqc.predict_proba(X_va_sub)
    print(f"  predict in {time.time()-t1:.1f}s")
    vqc_metrics = _eval_binary("vqc_binary", y_va_sub, vqc_proba)
    print(f"  F1={vqc_metrics['binary_f1']:.4f}  AUROC={vqc_metrics['auroc']:.4f}"
          f"  AUPR={vqc_metrics['aupr']:.4f}  FPR@TPR95={vqc_metrics['fpr_at_tpr95']:.4f}")
    all_metrics.append(vqc_metrics)

    # ── 6. Save results ───────────────────────────────────────────────────
    out = {
        "experiment": "phase5_vqc",
        "subset_size": int(n),
        "val_size": int(len(y_va_sub)),
        "val_attack_frac": float(y_va_sub.mean()),
        "models": all_metrics,
    }
    path = f"{RESULTS_DIR}/phase5_vqc_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved -> {path}")

    print("\n=== SUMMARY ===")
    print(f"{'Model':<28}  {'F1':>6}  {'AUROC':>7}  {'AUPR':>7}  {'FPR@95':>8}")
    print("-" * 64)
    for m in all_metrics:
        print(f"  {m['model']:<26}  {m['binary_f1']:>6.4f}  {m['auroc']:>7.4f}"
              f"  {m['aupr']:>7.4f}  {m['fpr_at_tpr95']:>8.4f}")
    print("\n(test.csv untouched)")


if __name__ == "__main__":
    main()
