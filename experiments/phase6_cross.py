"""Phase 6 — Experiment E7b: cross-dataset binary detection.

Trains PegasosQSVC and SVM-RBF on NF-ToN-IoT (all rows), then evaluates on
NF-UNSW-NB15 — a different network environment with opposite class imbalance
(ToN-IoT: 80% attack; UNSW-NB15: 95% benign). The scaler is fit on
NF-ToN-IoT and applied to NF-UNSW-NB15, mirroring a real deployment where a
model trained on one network is applied to another.

The cross-dataset AUROC drop vs the within-dataset result (E7a) is the
distribution-drift number that Phase 7 (Reflexion loop) is designed to recover.

Run:  ./venv/Scripts/python.exe -m experiments.phase6_cross
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from agent import agent_config as config
from data.nf_loader import NFPreprocessor, read_nf
from data.sampling import select_kernel_subset
from experiments.phase5_qsvc import _eval_binary, _fpr_at_tpr  # noqa: F401
from reasoning.classical import SVMModel
from reasoning.quantum import PegasosQSVCModel

RESULTS_DIR = "results/phase6"


def main() -> None:
    print("=== Phase 6 E7b: cross-dataset binary detection ===")
    print("    Train: NF-ToN-IoT  ->  Eval: NF-UNSW-NB15\n")

    # ── 1. Load both datasets ─────────────────────────────────────────────
    print(f"Loading {config.NF_TON_CSV} ...")
    t0 = time.time()
    X_ton_df, y_ton, _ = read_nf(config.NF_TON_CSV)
    print(f"  {len(y_ton):,} rows  binary: {np.bincount(y_ton)}  ({time.time()-t0:.1f}s)")

    print(f"Loading {config.NF_UNSW_CSV} ...")
    t0 = time.time()
    X_unsw_df, y_unsw, _ = read_nf(config.NF_UNSW_CSV)
    print(f"  {len(y_unsw):,} rows  binary: {np.bincount(y_unsw)}  ({time.time()-t0:.1f}s)")

    # ── 2. Scale: fit on ToN-IoT, apply to UNSW-NB15 ─────────────────────
    print("\nScaling (fit on NF-ToN-IoT, transform NF-UNSW-NB15 with same scaler)...")
    pre = NFPreprocessor()
    X8_ton  = pre.fit_transform(X_ton_df)
    X8_unsw = pre.transform(X_unsw_df)      # out-of-range values clipped to [0, pi]
    print(f"  ToN-IoT  scaled: {X8_ton.shape}   range [{X8_ton.min():.3f}, {X8_ton.max():.3f}]")
    print(f"  UNSW-NB15 scaled: {X8_unsw.shape}  range [{X8_unsw.min():.3f}, {X8_unsw.max():.3f}]")

    # ── 3. Kernel subset from NF-ToN-IoT (same size as E7a and Phase 5) ──
    n = config.QSVC_SUBSET_SIZE
    print(f"\nSelecting kernel subset n={n} from NF-ToN-IoT (k-means, class-balanced)...")
    t0 = time.time()
    idx = select_kernel_subset(X8_ton, y_ton, size=n, method="kmeans")
    X_sub, y_sub = X8_ton[idx], y_ton[idx]
    print(f"  subset: {X_sub.shape}  labels: {np.bincount(y_sub)}  ({time.time()-t0:.1f}s)")

    # ── 4. Validation subset from NF-UNSW-NB15 (balanced, capped for QSVC)
    rng = np.random.default_rng(config.RANDOM_SEED)
    n_va = min(200, len(X8_unsw))
    va_idx_b = np.where(y_unsw == 0)[0]
    va_idx_a = np.where(y_unsw == 1)[0]
    va_idx = np.concatenate([
        rng.choice(va_idx_b, min(n_va // 2, len(va_idx_b)), replace=False),
        rng.choice(va_idx_a, min(n_va // 2, len(va_idx_a)), replace=False),
    ])
    X_va_sub, y_va_sub = X8_unsw[va_idx], y_unsw[va_idx]
    print(f"\n  Cross-dataset val subset: {X_va_sub.shape}  binary: {np.bincount(y_va_sub)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_metrics = []

    # ── 5. SVM-RBF baseline ───────────────────────────────────────────────
    print("\n--- SVM-RBF (classical baseline, trained on NF-ToN-IoT subset) ---")
    t1 = time.time()
    svm = SVMModel(class_weight="balanced").fit(X_sub, y_sub)
    svm_proba = svm.predict_proba(X_va_sub)
    print(f"  fit+predict in {time.time()-t1:.1f}s")
    svm_m = _eval_binary("svm_rbf_cross", y_va_sub, svm_proba)
    print(f"  F1={svm_m['binary_f1']:.4f}  AUROC={svm_m['auroc']:.4f}"
          f"  AUPR={svm_m['aupr']:.4f}  FPR@TPR95={svm_m['fpr_at_tpr95']:.4f}")
    all_metrics.append(svm_m)

    # ── 6. PegasosQSVC (quantum) ──────────────────────────────────────────
    print("\n--- PegasosQSVC (quantum fidelity kernel, trained on NF-ToN-IoT subset) ---")
    t1 = time.time()
    qsvc = PegasosQSVCModel(num_steps=config.PEGASOS_TAU).fit(X_sub, y_sub)
    print(f"  fit in {time.time()-t1:.1f}s")
    t1 = time.time()
    qsvc_proba = qsvc.predict_proba(X_va_sub)
    print(f"  predict in {time.time()-t1:.1f}s")
    qsvc_m = _eval_binary("pegasos_qsvc_cross", y_va_sub, qsvc_proba)
    print(f"  F1={qsvc_m['binary_f1']:.4f}  AUROC={qsvc_m['auroc']:.4f}"
          f"  AUPR={qsvc_m['aupr']:.4f}  FPR@TPR95={qsvc_m['fpr_at_tpr95']:.4f}")
    all_metrics.append(qsvc_m)

    # ── 7. Save ───────────────────────────────────────────────────────────
    out = {
        "experiment": "phase6_cross",
        "train_dataset": "NF-ToN-IoT",
        "eval_dataset": "NF-UNSW-NB15",
        "features": config.NF_FEATURE_NAMES,
        "scale_mode": config.NF_SCALE_MODE,
        "subset_size": int(n),
        "val_size": int(len(y_va_sub)),
        "val_attack_frac": float(y_va_sub.mean()),
        "models": all_metrics,
    }
    path = f"{RESULTS_DIR}/phase6_cross_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved -> {path}")

    print("\n=== SUMMARY (cross-dataset: NF-ToN-IoT -> NF-UNSW-NB15) ===")
    print(f"{'Model':<30}  {'F1':>6}  {'AUROC':>7}  {'AUPR':>7}  {'FPR@95':>8}")
    print("-" * 64)
    for m in all_metrics:
        print(f"  {m['model']:<28}  {m['binary_f1']:>6.4f}  {m['auroc']:>7.4f}"
              f"  {m['aupr']:>7.4f}  {m['fpr_at_tpr95']:>8.4f}")
    print("\n(NF-UNSW-NB15 used as cross-dataset eval; no test set touched)")


if __name__ == "__main__":
    main()
