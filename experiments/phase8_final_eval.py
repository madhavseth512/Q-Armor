"""Phase 8 — Experiment E8c: sealed test set final evaluation.

Unlocks the held-out test splits that have NEVER been touched in any Phase 6/7
experiment. These are the publication-quality final numbers.

Evaluations on test splits:
  T1 — Within NF-ToN-IoT       : cascade QSVC + RF typer (ceiling)
  T2 — Cross-dataset baseline  : NF-ToN-IoT model -> NF-UNSW-NB15 (no retrain)
  T3 — Cross-dataset retrained : SWITCH_SUBSET + retrained QSVC

WARNING: Run this experiment ONLY ONCE when development is complete.
The test sets are sealed after Phase 5 EDA. Any tuning done after viewing test
results violates the experiment protocol. This script explicitly flags when it
is about to access test data so the researcher can confirm readiness.

Run:  ./venv/Scripts/python.exe -m experiments.phase8_final_eval
      ./venv/Scripts/python.exe -m experiments.phase8_final_eval --confirm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

from agent import agent_config as config
from agent.cascade_evaluator import CascadeEvaluator
from agent.retrainer import SubsetRetrainer
from data.nf_loader import NFPreprocessor, read_nf
from data.sampling import select_kernel_subset
from data.taxonomy import apply_taxonomy
from reasoning.cascade import DetectTypeCascade
from reasoning.classical import RandomForestModel
from reasoning.quantum import PegasosQSVCModel

RESULTS_DIR = "results/phase8"


def _balanced_sample(X, y, n_per_class, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    parts = []
    for cls in np.unique(y):
        pos = np.where(y == cls)[0]
        parts.append(rng.choice(pos, min(n_per_class, len(pos)), replace=False))
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _print_report(label, r) -> None:
    print(f"\n  {label}")
    print(f"    Stage1  AUROC={r.binary_auroc:.4f}  F1={r.binary_f1:.4f}  "
          f"AUPR={r.binary_aupr:.4f}  FPR@95={r.binary_fpr95:.4f}")
    print(f"    Stage2  macro-F1={r.type_macro_f1:.4f}  acc={r.type_accuracy:.4f}  "
          f"coverage={r.type_coverage:.4f}")
    print(f"    Per-class F1: " +
          "  ".join(f"{c}={v:.3f}" for c, v in r.type_per_class_f1.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true",
                        help="Confirm you are ready to evaluate on sealed test sets.")
    args = parser.parse_args()

    if not args.confirm:
        print("=" * 70)
        print("Phase 8 E8c: SEALED TEST SET FINAL EVALUATION")
        print("=" * 70)
        print("""
  WARNING: This experiment accesses the SEALED TEST SPLITS that have
  never been seen by any model during training or validation.

  Only run this when:
    1. All hyperparameter choices and model architectures are FINAL.
    2. Phase 8 E8a (cascade) and E8b (IBM) experiments are COMPLETE.
    3. No further tuning will occur after seeing these numbers.

  To proceed, re-run with:
    ./venv/Scripts/python.exe -m experiments.phase8_final_eval --confirm
""")
        sys.exit(0)

    print("=" * 70)
    print("Phase 8 E8c: Final evaluation on SEALED TEST SETS")
    print("  *** TEST SPLIT ACCESSED — results are FINAL ***")
    print("=" * 70)

    # ── 1. Load datasets ──────────────────────────────────────────────────────
    print(f"\nLoading {config.NF_TON_CSV} ...")
    t0 = time.time()
    X_ton_df, y_ton_bin, y_ton_att = read_nf(config.NF_TON_CSV)
    print(f"  {len(y_ton_bin):,} rows  ({time.time()-t0:.1f}s)")

    print(f"Loading {config.NF_UNSW_CSV} ...")
    t0 = time.time()
    X_unsw_df, y_unsw_bin, y_unsw_att = read_nf(config.NF_UNSW_CSV)
    print(f"  {len(y_unsw_bin):,} rows  ({time.time()-t0:.1f}s)")

    y_ton_coarse  = apply_taxonomy(y_ton_att)
    y_unsw_coarse = apply_taxonomy(y_unsw_att)

    # ── 2. Fixed 80/20 split (same seed as all prior experiments) ─────────────
    (X_ton_tr_df, X_ton_te_df,
     y_ton_bin_tr, y_ton_bin_te,
     y_ton_crs_tr, y_ton_crs_te) = train_test_split(
        X_ton_df, y_ton_bin, y_ton_coarse,
        test_size=0.20, stratify=y_ton_bin, random_state=config.RANDOM_SEED,
    )
    (X_unsw_tr_df, X_unsw_te_df,
     y_unsw_bin_tr, y_unsw_bin_te,
     y_unsw_crs_tr, y_unsw_crs_te) = train_test_split(
        X_unsw_df, y_unsw_bin, y_unsw_coarse,
        test_size=0.20, stratify=y_unsw_bin, random_state=config.RANDOM_SEED,
    )

    print(f"\n  ToN-IoT  train {X_ton_tr_df.shape}  TEST {X_ton_te_df.shape}")
    print(f"  UNSW-NB15 train {X_unsw_tr_df.shape}  TEST {X_unsw_te_df.shape}")

    # ── 3. Scale ──────────────────────────────────────────────────────────────
    pre = NFPreprocessor()
    X8_ton_tr = pre.fit_transform(X_ton_tr_df)
    X8_ton_te = pre.transform(X_ton_te_df)
    X8_unsw_te = pre.transform(X_unsw_te_df)
    X8_unsw_tr = pre.transform(X_unsw_tr_df)

    # ── 4. Full test sets (no capping — this is the final eval) ──────────────
    # For QSVC: evaluation on the full test set is very expensive.
    # Cap at 1000 samples (500 benign + 500 attack) for QSVC predict cost.
    # RF has no such constraint and runs on the full test set.
    QSVC_CAP = 1000
    rng = np.random.default_rng(config.RANDOM_SEED + 900)

    def _capped_subset(X, y_bin, y_crs, cap):
        b = np.where(y_bin == 0)[0]
        a = np.where(y_bin == 1)[0]
        nb = min(cap // 2, len(b))
        na = min(cap // 2, len(a))
        idx = np.concatenate([rng.choice(b, nb, replace=False),
                               rng.choice(a, na, replace=False)])
        rng.shuffle(idx)
        return X[idx], y_bin[idx], y_crs[idx]

    X_ton_qsvc_te, y_ton_qsvc_bin, y_ton_qsvc_crs = _capped_subset(
        X8_ton_te, y_ton_bin_te, y_ton_crs_te, QSVC_CAP)
    X_unsw_qsvc_te, y_unsw_qsvc_bin, y_unsw_qsvc_crs = _capped_subset(
        X8_unsw_te, y_unsw_bin_te, y_unsw_crs_te, QSVC_CAP)

    print(f"\n  QSVC test subset ToN-IoT  : {X_ton_qsvc_te.shape}  "
          f"binary {np.bincount(y_ton_qsvc_bin)}")
    print(f"  QSVC test subset UNSW-NB15: {X_unsw_qsvc_te.shape}  "
          f"binary {np.bincount(y_unsw_qsvc_bin)}")

    # ── 5. Train models (using TRAIN splits, not test) ─────────────────────────
    n_sub = config.QSVC_SUBSET_SIZE
    print(f"\nTraining Stage 1 QSVC (n={n_sub}) ...")
    t0 = time.time()
    idx = select_kernel_subset(X8_ton_tr, y_ton_bin_tr, size=n_sub, method="kmeans")
    qsvc = PegasosQSVCModel(num_steps=config.PEGASOS_TAU).fit(
        X8_ton_tr[idx], y_ton_bin_tr[idx])
    print(f"  done in {time.time()-t0:.1f}s")

    print("\nTraining Stage 2 RF typer ...")
    t0 = time.time()
    X_s2, y_s2 = _balanced_sample(X8_ton_tr, y_ton_crs_tr, config.CASCADE_STAGE2_N_PER_CLASS)
    rf_s2 = RandomForestModel(class_weight=None).fit(X_s2, y_s2)
    print(f"  done in {time.time()-t0:.2f}s")

    ev = CascadeEvaluator()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_reports = []

    # ── T1: Within NF-ToN-IoT TEST set ────────────────────────────────────────
    print("\n" + "-" * 60)
    print("T1 — Within NF-ToN-IoT [SEALED TEST SET]")
    t0 = time.time()
    t1_cascade = DetectTypeCascade(qsvc, rf_s2)
    t1_result  = t1_cascade.predict(X_ton_qsvc_te)
    t1_report  = ev.evaluate(t1_result, y_ton_qsvc_bin, y_ton_qsvc_crs,
                             "NF-ToN-IoT_TEST", "pegasos_qsvc", "random_forest")
    print(f"  predict time: {time.time()-t0:.1f}s")
    _print_report("T1 (within, QSVC, TEST)", t1_report)
    all_reports.append(t1_report.to_dict())

    # ── T2: Cross-dataset TEST set, no retraining ─────────────────────────────
    print("\n" + "-" * 60)
    print("T2 — Cross-dataset NF-ToN-IoT -> NF-UNSW-NB15 [SEALED TEST SET, no retrain]")
    t0 = time.time()
    t2_cascade = DetectTypeCascade(qsvc, rf_s2)
    t2_result  = t2_cascade.predict(X_unsw_qsvc_te)
    t2_report  = ev.evaluate(t2_result, y_unsw_qsvc_bin, y_unsw_qsvc_crs,
                             "NF-UNSW-NB15_TEST", "pegasos_qsvc", "random_forest",
                             with_retraining=False)
    print(f"  predict time: {time.time()-t0:.1f}s")
    _print_report("T2 (cross, no retrain, TEST)", t2_report)
    all_reports.append(t2_report.to_dict())

    # ── T3: Cross-dataset TEST set WITH SWITCH_SUBSET retraining ─────────────
    print("\n" + "-" * 60)
    print("T3 — Cross-dataset SWITCH_SUBSET retrained [SEALED TEST SET]")
    n_cross = config.SWITCH_SUBSET_N_CROSS
    b_pool = np.where(y_unsw_bin_tr == 0)[0]
    a_pool = np.where(y_unsw_bin_tr == 1)[0]
    rng2   = np.random.default_rng(config.RANDOM_SEED + 901)
    pool_idx = np.concatenate([
        rng2.choice(b_pool, min(n_cross // 2, len(b_pool)), replace=False),
        rng2.choice(a_pool, min(n_cross // 2, len(a_pool)), replace=False),
    ])
    X_cross_pool = X8_unsw_tr[pool_idx]
    y_cross_pool = y_unsw_bin_tr[pool_idx]
    print(f"  Retraining pool: {X_cross_pool.shape}  labels {np.bincount(y_cross_pool)}")

    retrainer = SubsetRetrainer()
    qsvc_retrained, _, _ = retrainer.retrain(X_cross_pool, y_cross_pool, verbose=True)

    t0 = time.time()
    t3_cascade = DetectTypeCascade(qsvc_retrained, rf_s2)
    t3_result  = t3_cascade.predict(X_unsw_qsvc_te)
    t3_report  = ev.evaluate(t3_result, y_unsw_qsvc_bin, y_unsw_qsvc_crs,
                             "NF-UNSW-NB15_TEST", "pegasos_qsvc_retrained", "random_forest",
                             with_retraining=True)
    print(f"  predict time: {time.time()-t0:.1f}s")
    _print_report("T3 (cross, retrained, TEST)", t3_report)
    all_reports.append(t3_report.to_dict())

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "experiment":    "phase8_final_eval",
        "split":         "SEALED_TEST",
        "warning":       "These are final test set numbers. Do not tune after viewing.",
        "qsvc_n_sub":    n_sub,
        "qsvc_test_cap": QSVC_CAP,
        "retrain_pool":  n_cross,
        "results":       all_reports,
    }
    path = f"{RESULTS_DIR}/phase8_final_eval_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved -> {path}")

    # ── Final summary table ───────────────────────────────────────────────────
    labels  = ["T1 (within, QSVC)",
               "T2 (cross, no retrain)",
               "T3 (cross, retrained)"]
    reports = [t1_report, t2_report, t3_report]

    print("\n" + "=" * 70)
    print("FINAL TEST SET RESULTS")
    print("=" * 70)
    print(f"  {'Exp':<24}  {'S1 AUROC':>8}  {'S1 F1':>6}  "
          f"{'S2 macro-F1':>11}  {'FPR@95':>7}")
    print("  " + "-" * 62)
    for lbl, r in zip(labels, reports):
        print(f"  {lbl:<24}  {r.binary_auroc:>8.4f}  {r.binary_f1:>6.4f}  "
              f"{r.type_macro_f1:>11.4f}  {r.binary_fpr95:>7.4f}")

    retrain_gain = t3_report.binary_auroc - t2_report.binary_auroc
    type_gain    = t3_report.type_macro_f1 - t2_report.type_macro_f1
    print(f"\n  SWITCH_SUBSET gain: AUROC {retrain_gain:+.4f}  "
          f"macro-F1 {type_gain:+.4f}")


if __name__ == "__main__":
    main()
