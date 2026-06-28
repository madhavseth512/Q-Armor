"""Phase 8 — Experiment E8a: Detect->Type cascade + SWITCH_SUBSET retraining.

Four evaluations:
  C1 — Within NF-ToN-IoT       : cascade without retraining (ceiling)
  C2 — Cross-dataset baseline  : NF-ToN-IoT model -> NF-UNSW-NB15 (no retrain)
  C3 — Cross-dataset retrained : SWITCH_SUBSET fires, Stage 1 retrained on UNSW samples
  C4 — Cross-dataset RF only   : Stage 1 = RF (classical fallback comparison)

Stage 1 models: PegasosQSVC (QSVM tier) + RandomForest (CLASSICAL fallback)
Stage 2 model:  RandomForest trained on NF-ToN-IoT coarse 5-class taxonomy
                (same model used for C1-C4 — typing model not retrained)

Run:  ./venv/Scripts/python.exe -m experiments.phase8_cascade
"""

from __future__ import annotations

import json
import os
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


def _print_report(label: str, r) -> None:
    print(f"\n  {label}")
    print(f"    Stage1  AUROC={r.binary_auroc:.4f}  F1={r.binary_f1:.4f}  "
          f"AUPR={r.binary_aupr:.4f}  FPR@95={r.binary_fpr95:.4f}")
    print(f"    Stage2  macro-F1={r.type_macro_f1:.4f}  acc={r.type_accuracy:.4f}  "
          f"coverage={r.type_coverage:.4f}  n_typed={r.type_n_typed}")
    print(f"    Per-class F1: " +
          "  ".join(f"{c}={v:.3f}" for c, v in r.type_per_class_f1.items()))


def main() -> None:
    print("=" * 70)
    print("Phase 8 E8a: Detect->Type cascade + SWITCH_SUBSET retraining")
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

    # Coarse taxonomy
    y_ton_coarse  = apply_taxonomy(y_ton_att)
    y_unsw_coarse = apply_taxonomy(y_unsw_att)

    # ── 2. Train/test split ───────────────────────────────────────────────────
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
    print(f"\n  ToN-IoT  train {X_ton_tr_df.shape}  test {X_ton_te_df.shape}")
    print(f"  UNSW-NB15 train {X_unsw_tr_df.shape}  test {X_unsw_te_df.shape}")

    # ── 3. Scale ─────────────────────────────────────────────────────────────
    print("\nScaling (fit on NF-ToN-IoT train) ...")
    pre = NFPreprocessor()
    X8_ton_tr = pre.fit_transform(X_ton_tr_df)
    X8_ton_te = pre.transform(X_ton_te_df)
    X8_unsw_te = pre.transform(X_unsw_te_df)
    # UNSW train pool for retraining
    X8_unsw_tr = pre.transform(X_unsw_tr_df)
    print(f"  ToN-IoT train {X8_ton_tr.shape}  range [{X8_ton_tr.min():.3f}, {X8_ton_tr.max():.3f}]")
    print(f"  UNSW-NB15 test  {X8_unsw_te.shape}")

    # ── 4. Validation subsets ─────────────────────────────────────────────────
    # For QSVC: cap at 500 samples (predict cost)
    rng = np.random.default_rng(config.RANDOM_SEED)
    N_VAL = 500

    def _val_subset(X, y_bin, y_coarse, n=N_VAL):
        b_idx = np.where(y_bin == 0)[0]
        a_idx = np.where(y_bin == 1)[0]
        chosen = np.concatenate([
            rng.choice(b_idx, min(n // 2, len(b_idx)), replace=False),
            rng.choice(a_idx, min(n // 2, len(a_idx)), replace=False),
        ])
        rng.shuffle(chosen)
        return X[chosen], y_bin[chosen], y_coarse[chosen]

    X_ton_val, y_ton_val_bin, y_ton_val_crs = _val_subset(X8_ton_te, y_ton_bin_te, y_ton_crs_te)
    X_unsw_val, y_unsw_val_bin, y_unsw_val_crs = _val_subset(X8_unsw_te, y_unsw_bin_te, y_unsw_crs_te)
    print(f"\n  Val subset ToN-IoT  : {X_ton_val.shape}  binary {np.bincount(y_ton_val_bin)}")
    print(f"  Val subset UNSW-NB15: {X_unsw_val.shape}  binary {np.bincount(y_unsw_val_bin)}")

    # ── 5. Train Stage 1 — QSVC on NF-ToN-IoT ────────────────────────────────
    n_sub = config.QSVC_SUBSET_SIZE
    print(f"\nTraining Stage 1 QSVC (n={n_sub}, k-means) ...")
    t0 = time.time()
    idx = select_kernel_subset(X8_ton_tr, y_ton_bin_tr, size=n_sub, method="kmeans")
    X_sub, y_sub = X8_ton_tr[idx], y_ton_bin_tr[idx]
    qsvc = PegasosQSVCModel(num_steps=config.PEGASOS_TAU).fit(X_sub, y_sub)
    print(f"  done in {time.time()-t0:.1f}s  subset {X_sub.shape}  labels {np.bincount(y_sub)}")

    # ── 6. Train Stage 1 — RF fallback (on 2000-sample balanced subset) ───────
    print("\nTraining Stage 1 RF (classical fallback) ...")
    t0 = time.time()
    X_rf_tr, y_rf_tr = _balanced_sample(X8_ton_tr, y_ton_bin_tr, 1000)
    rf_s1 = RandomForestModel(class_weight=None).fit(X_rf_tr, y_rf_tr)
    print(f"  done in {time.time()-t0:.2f}s  subset {X_rf_tr.shape}")

    # ── 7. Train Stage 2 — RF multi-class typer ───────────────────────────────
    print(f"\nTraining Stage 2 RF typer (n_per_class={config.CASCADE_STAGE2_N_PER_CLASS}) ...")
    t0 = time.time()
    X_s2_tr, y_s2_tr = _balanced_sample(X8_ton_tr, y_ton_crs_tr, config.CASCADE_STAGE2_N_PER_CLASS)
    rf_s2 = RandomForestModel(class_weight=None).fit(X_s2_tr, y_s2_tr)
    print(f"  done in {time.time()-t0:.2f}s  subset {X_s2_tr.shape}")
    print(f"  classes: {dict(zip(*np.unique(y_s2_tr, return_counts=True)))}")

    ev = CascadeEvaluator()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_reports = []

    # ── C1: Within NF-ToN-IoT (QSVC ceiling) ─────────────────────────────────
    print("\n" + "-" * 60)
    print("C1 — Within NF-ToN-IoT (QSVC + RF typer)")
    t0 = time.time()
    c1_cascade = DetectTypeCascade(qsvc, rf_s2)
    c1_result  = c1_cascade.predict(X_ton_val)
    c1_report  = ev.evaluate(c1_result, y_ton_val_bin, y_ton_val_crs,
                             "NF-ToN-IoT", "pegasos_qsvc", "random_forest")
    print(f"  predict time: {time.time()-t0:.1f}s")
    _print_report("C1 (within, QSVC)", c1_report)
    all_reports.append(c1_report.to_dict())

    # ── C2: Cross-dataset, no retraining (baseline) ───────────────────────────
    print("\n" + "-" * 60)
    print("C2 — Cross-dataset (NF-ToN-IoT model -> NF-UNSW-NB15), no retraining")
    t0 = time.time()
    c2_cascade = DetectTypeCascade(qsvc, rf_s2)
    c2_result  = c2_cascade.predict(X_unsw_val)
    c2_report  = ev.evaluate(c2_result, y_unsw_val_bin, y_unsw_val_crs,
                             "NF-UNSW-NB15", "pegasos_qsvc", "random_forest",
                             with_retraining=False)
    print(f"  predict time: {time.time()-t0:.1f}s")
    _print_report("C2 (cross, no retrain)", c2_report)
    all_reports.append(c2_report.to_dict())

    # ── C3: Cross-dataset WITH SWITCH_SUBSET retraining ───────────────────────
    print("\n" + "-" * 60)
    print("C3 — Cross-dataset WITH SWITCH_SUBSET: retrain Stage 1 on UNSW-NB15 samples")
    n_cross = config.SWITCH_SUBSET_N_CROSS
    b_pool = np.where(y_unsw_bin_tr == 0)[0]
    a_pool = np.where(y_unsw_bin_tr == 1)[0]
    cross_pool_idx = np.concatenate([
        rng.choice(b_pool, min(n_cross // 2, len(b_pool)), replace=False),
        rng.choice(a_pool, min(n_cross // 2, len(a_pool)), replace=False),
    ])
    X_cross_pool = X8_unsw_tr[cross_pool_idx]
    y_cross_pool = y_unsw_bin_tr[cross_pool_idx]
    print(f"  Cross-domain retraining pool: {X_cross_pool.shape}  "
          f"labels {np.bincount(y_cross_pool)}")

    retrainer = SubsetRetrainer()
    qsvc_retrained, _, _ = retrainer.retrain(X_cross_pool, y_cross_pool, verbose=True)

    t0 = time.time()
    c3_cascade = DetectTypeCascade(qsvc_retrained, rf_s2)
    c3_result  = c3_cascade.predict(X_unsw_val)
    c3_report  = ev.evaluate(c3_result, y_unsw_val_bin, y_unsw_val_crs,
                             "NF-UNSW-NB15", "pegasos_qsvc_retrained", "random_forest",
                             with_retraining=True)
    print(f"  predict time: {time.time()-t0:.1f}s")
    _print_report("C3 (cross, SWITCH_SUBSET retrained)", c3_report)
    all_reports.append(c3_report.to_dict())

    # ── C4: Cross-dataset, RF Stage 1 fallback ────────────────────────────────
    print("\n" + "-" * 60)
    print("C4 — Cross-dataset, classical RF Stage 1 fallback")
    t0 = time.time()
    c4_cascade = DetectTypeCascade(rf_s1, rf_s2)
    c4_result  = c4_cascade.predict(X_unsw_val)
    c4_report  = ev.evaluate(c4_result, y_unsw_val_bin, y_unsw_val_crs,
                             "NF-UNSW-NB15", "random_forest", "random_forest",
                             with_retraining=False)
    print(f"  predict time: {time.time()-t0:.2f}s")
    _print_report("C4 (cross, RF fallback)", c4_report)
    all_reports.append(c4_report.to_dict())

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "experiment":    "phase8_cascade",
        "features":      config.NF_FEATURE_NAMES,
        "qsvc_n_sub":    n_sub,
        "val_n":         N_VAL,
        "retrain_pool":  n_cross,
        "results":       all_reports,
    }
    path = f"{RESULTS_DIR}/phase8_cascade_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved -> {path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    labels = ["C1 (within, QSVC)", "C2 (cross, no retrain)",
              "C3 (cross, retrained)", "C4 (cross, RF)"]
    reports = [c1_report, c2_report, c3_report, c4_report]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Exp':<24}  {'S1 AUROC':>8}  {'S1 F1':>6}  "
          f"{'S2 macro-F1':>11}  {'FPR@95':>7}")
    print("  " + "-" * 62)
    for lbl, r in zip(labels, reports):
        print(f"  {lbl:<24}  {r.binary_auroc:>8.4f}  {r.binary_f1:>6.4f}  "
              f"{r.type_macro_f1:>11.4f}  {r.binary_fpr95:>7.4f}")
    print("\n(test sets untouched — these are val subset results)")


if __name__ == "__main__":
    main()
