"""Phase 6 — Experiment E7a: within-dataset binary detection on NF-ToN-IoT.

Trains PegasosQSVC (quantum fidelity kernel) and SVM-RBF (classical baseline)
on an 80/20 stratified split of NF-ToN-IoT using the 8 features locked by
the Phase 6 EDA (NF_FEATURE_NAMES, NF_SCALE_MODE). This establishes the
within-dataset ceiling before the cross-dataset drift experiment (E7b).

Run:  ./venv/Scripts/python.exe -m experiments.phase6_within
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.model_selection import train_test_split

from agent import agent_config as config
from data.nf_loader import NFPreprocessor, read_nf
from data.sampling import select_kernel_subset
from experiments.phase5_qsvc import _eval_binary, _fpr_at_tpr  # noqa: F401
from reasoning.classical import SVMModel
from reasoning.quantum import PegasosQSVCModel

RESULTS_DIR = "results/phase6"


def main() -> None:
    print("=== Phase 6 E7a: within-dataset binary detection (NF-ToN-IoT) ===\n")

    # ── 1. Load NF-ToN-IoT ────────────────────────────────────────────────
    print(f"Loading {config.NF_TON_CSV} ...")
    t0 = time.time()
    X_df, y_bin, _ = read_nf(config.NF_TON_CSV)
    print(f"  {len(y_bin):,} rows  binary: {np.bincount(y_bin)}  ({time.time()-t0:.1f}s)")

    # ── 2. 80/20 stratified train/val split ───────────────────────────────
    X_tr_df, X_va_df, y_tr, y_va = train_test_split(
        X_df, y_bin,
        test_size=0.20,
        stratify=y_bin,
        random_state=config.RANDOM_SEED,
    )
    print(f"\n  train: {X_tr_df.shape}  binary: {np.bincount(y_tr)}")
    print(f"  val  : {X_va_df.shape}  binary: {np.bincount(y_va)}")

    # ── 3. Scale (fit on train, apply to val) ─────────────────────────────
    pre = NFPreprocessor()
    X8_tr = pre.fit_transform(X_tr_df)
    X8_va = pre.transform(X_va_df)
    print(f"\n  Scaled train: {X8_tr.shape}  range [{X8_tr.min():.3f}, {X8_tr.max():.3f}]")

    # ── 4. Kernel subset (balanced, k-means) ─────────────────────────────
    n = config.QSVC_SUBSET_SIZE
    print(f"\nSelecting kernel subset n={n} (k-means, class-balanced) ...")
    t0 = time.time()
    idx = select_kernel_subset(X8_tr, y_tr, size=n, method="kmeans")
    X_sub, y_sub = X8_tr[idx], y_tr[idx]
    print(f"  subset: {X_sub.shape}  labels: {np.bincount(y_sub)}  ({time.time()-t0:.1f}s)")

    # ── 5. Validation subset (capped for QSVC prediction cost) ───────────
    rng = np.random.default_rng(config.RANDOM_SEED)
    n_va = min(200, len(X8_va))
    va_idx_b = np.where(y_va == 0)[0]
    va_idx_a = np.where(y_va == 1)[0]
    va_idx = np.concatenate([
        rng.choice(va_idx_b, min(n_va // 2, len(va_idx_b)), replace=False),
        rng.choice(va_idx_a, min(n_va // 2, len(va_idx_a)), replace=False),
    ])
    X_va_sub, y_va_sub = X8_va[va_idx], y_va[va_idx]
    print(f"\n  Validation subset: {X_va_sub.shape}  binary: {np.bincount(y_va_sub)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_metrics = []

    # ── 6. SVM-RBF baseline ───────────────────────────────────────────────
    print("\n--- SVM-RBF (classical baseline, identical subset) ---")
    t1 = time.time()
    svm = SVMModel(class_weight="balanced").fit(X_sub, y_sub)
    svm_proba = svm.predict_proba(X_va_sub)
    print(f"  fit+predict in {time.time()-t1:.1f}s")
    svm_m = _eval_binary("svm_rbf_within", y_va_sub, svm_proba)
    print(f"  F1={svm_m['binary_f1']:.4f}  AUROC={svm_m['auroc']:.4f}"
          f"  AUPR={svm_m['aupr']:.4f}  FPR@TPR95={svm_m['fpr_at_tpr95']:.4f}")
    all_metrics.append(svm_m)

    # ── 7. PegasosQSVC (quantum) ──────────────────────────────────────────
    print("\n--- PegasosQSVC (quantum fidelity kernel) ---")
    t1 = time.time()
    qsvc = PegasosQSVCModel(num_steps=config.PEGASOS_TAU).fit(X_sub, y_sub)
    print(f"  fit in {time.time()-t1:.1f}s")
    t1 = time.time()
    qsvc_proba = qsvc.predict_proba(X_va_sub)
    print(f"  predict in {time.time()-t1:.1f}s")
    qsvc_m = _eval_binary("pegasos_qsvc_within", y_va_sub, qsvc_proba)
    print(f"  F1={qsvc_m['binary_f1']:.4f}  AUROC={qsvc_m['auroc']:.4f}"
          f"  AUPR={qsvc_m['aupr']:.4f}  FPR@TPR95={qsvc_m['fpr_at_tpr95']:.4f}")
    all_metrics.append(qsvc_m)

    # ── 8. Save ───────────────────────────────────────────────────────────
    out = {
        "experiment": "phase6_within",
        "dataset": "NF-ToN-IoT",
        "features": config.NF_FEATURE_NAMES,
        "scale_mode": config.NF_SCALE_MODE,
        "subset_size": int(n),
        "val_size": int(len(y_va_sub)),
        "val_attack_frac": float(y_va_sub.mean()),
        "models": all_metrics,
    }
    path = f"{RESULTS_DIR}/phase6_within_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved -> {path}")

    print("\n=== SUMMARY (within-dataset, NF-ToN-IoT val subset) ===")
    print(f"{'Model':<30}  {'F1':>6}  {'AUROC':>7}  {'AUPR':>7}  {'FPR@95':>8}")
    print("-" * 64)
    for m in all_metrics:
        print(f"  {m['model']:<28}  {m['binary_f1']:>6.4f}  {m['auroc']:>7.4f}"
              f"  {m['aupr']:>7.4f}  {m['fpr_at_tpr95']:>8.4f}")
    print("\n(test sets untouched)")


if __name__ == "__main__":
    main()
