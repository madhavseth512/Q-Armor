"""Phase 11 — Confirmatory Protocol: natural class-prevalence evaluation.

Closes the "natural class prevalence" item from docs/paper.tex's Confirmatory
Protocol: every other reported result in this project uses a 50/50 balanced
evaluation sample. This script evaluates the SAME three conditions
(T1 within-domain, T2 cross-domain no-adapt, T3 cross-domain SWITCH_SUBSET-
adapted) on an UNBALANCED, natural-prevalence sample instead, and reports the
extra metrics the protocol requires: precision, MCC, balanced accuracy, TPR
at a fixed low FPR (0.01), expected calibration error (ECE, 10-bin), and
alerts per 10,000 flows.

Model training mirrors experiments/phase8_final_eval.py exactly (same source
n=150 k-means subset, same SWITCH_SUBSET retrain via SubsetRetrainer) so
these numbers are directly comparable to the paper's headline T1/T2/T3 row --
only the evaluation SAMPLING changes (natural prevalence instead of balanced),
not the models.

Eval-set size is capped at 1000 (same QSVC_CAP convention as
phase8_final_eval.py / phase10_confirmatory.py) but drawn WITHOUT class
rebalancing, so NF-UNSW-NB15's ~4.5% attack rate is preserved.

Run:  ./venv/Scripts/python.exe -m experiments.phase11_natural_prevalence --confirm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from agent import agent_config as config
from agent.cascade_evaluator import _fpr_at_tpr
from agent.retrainer import SubsetRetrainer
from data.nf_loader import NFPreprocessor, read_nf
from data.sampling import select_kernel_subset
from reasoning.classical import GBTModel, LinearSVMModel, RandomForestModel, SVMModel
from reasoning.quantum import PegasosQSVCModel

RESULTS_DIR = "results/phase11"
NATURAL_CAP = 1000
FPR_TARGET = 0.01  # "fixed low FPR" operating point for TPR-at-low-FPR

MODEL_REGISTRY = {
    "pegasos_qsvc":  lambda: PegasosQSVCModel(num_steps=config.PEGASOS_TAU),
    "random_forest": lambda: RandomForestModel(class_weight=None),
    "svm_rbf":       lambda: SVMModel(),
    "svm_linear":    lambda: LinearSVMModel(),
    "gbt":           lambda: GBTModel(),
}


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float) -> float:
    """TPR at the smallest achieved FPR <= fpr_target (0 if none achieve it)."""
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
    ok = fpr_arr <= fpr_target
    return float(tpr_arr[ok].max()) if ok.any() else 0.0


def _expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """10-bin ECE: sum over bins of (bin weight) x |mean confidence - accuracy|.

    ``proba`` here is the model's confidence in its OWN hard decision (not
    just P(attack)), i.e. max(P(benign), P(attack)) per sample -- the
    standard top-label calibration definition.
    """
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def _natural_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    proba = model.predict_proba(X_test)
    s_atk = proba[:, 1]
    y_pred = model.predict_labels(X_test)
    return {
        "n":                 int(len(y_test)),
        "attack_prevalence": float(y_test.mean()),
        "auroc":             float(roc_auc_score(y_test, s_atk)),
        "f1":                float(f1_score(y_test, y_pred, zero_division=0)),
        "precision":         float(precision_score(y_test, y_pred, zero_division=0)),
        "mcc":               float(matthews_corrcoef(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "fpr95":             _fpr_at_tpr(y_test, s_atk),
        f"tpr_at_fpr{FPR_TARGET}": _tpr_at_fpr(y_test, s_atk, FPR_TARGET),
        "ece_10bin":         _expected_calibration_error(y_test, proba),
        "alerts_per_10k":    float(y_pred.mean() * 10_000),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--models", type=str, default=",".join(MODEL_REGISTRY.keys()))
    parser.add_argument("--eval-cap", type=int, default=NATURAL_CAP,
                        help="Natural-prevalence eval set size. QSVC predict cost "
                             "scales with n_train x n_test -- shrink this for the "
                             "quantum arm (see experiments/phase10_confirmatory.py, "
                             "which hit the same cost and added the same flag).")
    args = parser.parse_args()
    models = args.models.split(",")
    eval_cap = args.eval_cap

    if not args.confirm:
        print("Phase 11: natural class-prevalence evaluation.")
        print(f"  models: {models}")
        print("  Re-run with --confirm to proceed.")
        sys.exit(0)

    print("=" * 70)
    print(f"Phase 11: natural-prevalence eval  models={models}")
    print("=" * 70)

    X_ton_df, y_ton_bin, _ = read_nf(config.NF_TON_CSV)
    X_unsw_df, y_unsw_bin, _ = read_nf(config.NF_UNSW_CSV)

    X_ton_tr_df, X_ton_te_df, y_ton_bin_tr, y_ton_bin_te = train_test_split(
        X_ton_df, y_ton_bin, test_size=0.20, stratify=y_ton_bin,
        random_state=config.RANDOM_SEED)
    X_unsw_tr_df, X_unsw_te_df, y_unsw_bin_tr, y_unsw_bin_te = train_test_split(
        X_unsw_df, y_unsw_bin, test_size=0.20, stratify=y_unsw_bin,
        random_state=config.RANDOM_SEED)

    pre = NFPreprocessor()
    X8_ton_tr = pre.fit_transform(X_ton_tr_df)
    X8_ton_te = pre.transform(X_ton_te_df)
    X8_unsw_tr = pre.transform(X_unsw_tr_df)
    X8_unsw_te = pre.transform(X_unsw_te_df)

    # Natural-prevalence eval draws: NO class rebalancing, just a capped
    # uniform random sample from the test partition (preserves true prevalence).
    def _natural_sample(X, y, cap, seed_offset):
        rng = np.random.default_rng(config.RANDOM_SEED + seed_offset)
        n = min(cap, len(y))
        idx = rng.choice(len(y), n, replace=False)
        return X[idx], y[idx]

    X_ton_nat, y_ton_nat = _natural_sample(X8_ton_te, y_ton_bin_te, eval_cap, 910)
    X_unsw_nat, y_unsw_nat = _natural_sample(X8_unsw_te, y_unsw_bin_te, eval_cap, 911)
    print(f"\nNatural-prevalence eval sets:")
    print(f"  ToN-IoT  : n={len(y_ton_nat)}  attack%={y_ton_nat.mean():.4f}")
    print(f"  UNSW-NB15: n={len(y_unsw_nat)}  attack%={y_unsw_nat.mean():.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    for model_name in models:
        print(f"\n{'=' * 60}\nMODEL: {model_name}\n{'=' * 60}")

        # -- Source model (n=150 k-means subset of ToN-IoT train) --------------
        t0 = time.time()
        src_idx = select_kernel_subset(X8_ton_tr, y_ton_bin_tr, size=config.QSVC_SUBSET_SIZE,
                                        method="kmeans") if model_name == "pegasos_qsvc" \
            else np.random.default_rng(config.RANDOM_SEED).choice(
                len(y_ton_bin_tr), config.QSVC_SUBSET_SIZE, replace=False)
        m_source = MODEL_REGISTRY[model_name]().fit(X8_ton_tr[src_idx], y_ton_bin_tr[src_idx])
        print(f"  [source model trained in {time.time() - t0:.1f}s]")

        t1 = _natural_metrics(m_source, X_ton_nat, y_ton_nat)
        print(f"  T1 (within, natural prevalence)      : AUROC={t1['auroc']:.4f}  "
              f"F1={t1['f1']:.4f}  MCC={t1['mcc']:.4f}  balAcc={t1['balanced_accuracy']:.4f}")

        t2 = _natural_metrics(m_source, X_unsw_nat, y_unsw_nat)
        print(f"  T2 (cross, no-adapt, natural prev.)  : AUROC={t2['auroc']:.4f}  "
              f"F1={t2['f1']:.4f}  MCC={t2['mcc']:.4f}  balAcc={t2['balanced_accuracy']:.4f}")

        # -- SWITCH_SUBSET-adapted model (same pool convention as phase8) -----
        t0 = time.time()
        n_cross = config.SWITCH_SUBSET_N_CROSS
        b_pool = np.where(y_unsw_bin_tr == 0)[0]
        a_pool = np.where(y_unsw_bin_tr == 1)[0]
        rng2 = np.random.default_rng(config.RANDOM_SEED + 901)
        pool_idx = np.concatenate([
            rng2.choice(b_pool, min(n_cross // 2, len(b_pool)), replace=False),
            rng2.choice(a_pool, min(n_cross // 2, len(a_pool)), replace=False),
        ])
        X_pool, y_pool = X8_unsw_tr[pool_idx], y_unsw_bin_tr[pool_idx]
        if model_name == "pegasos_qsvc":
            retrainer = SubsetRetrainer()
            m_adapted, _, _ = retrainer.retrain(X_pool, y_pool, verbose=False)
        else:
            adapt_idx = select_kernel_subset(X_pool, y_pool, size=config.QSVC_SUBSET_SIZE,
                                              method="kmeans")
            m_adapted = MODEL_REGISTRY[model_name]().fit(X_pool[adapt_idx], y_pool[adapt_idx])
        print(f"  [adapted model trained in {time.time() - t0:.1f}s]")

        t3 = _natural_metrics(m_adapted, X_unsw_nat, y_unsw_nat)
        print(f"  T3 (cross, SWITCH_SUBSET, natural prev.): AUROC={t3['auroc']:.4f}  "
              f"F1={t3['f1']:.4f}  MCC={t3['mcc']:.4f}  balAcc={t3['balanced_accuracy']:.4f}")
        print(f"  T3 alerts/10k={t3['alerts_per_10k']:.1f}  ECE={t3['ece_10bin']:.4f}  "
              f"TPR@FPR{FPR_TARGET}={t3[f'tpr_at_fpr{FPR_TARGET}']:.4f}")

        results[model_name] = {"T1": t1, "T2": t2, "T3": t3, "eval_cap": eval_cap}

        # Merge with any existing file rather than overwrite -- different
        # models may have been run in separate invocations at different
        # eval_cap sizes (QSVC predict cost forced a smaller cap than the
        # classical models used; see --eval-cap).
        out_path = f"{RESULTS_DIR}/phase11_natural_prevalence_metrics.json"
        existing_results = {}
        if os.path.exists(out_path):
            with open(out_path) as f:
                existing_results = json.load(f).get("results", {})
        existing_results.update(results)

        with open(out_path, "w") as f:
            json.dump({
                "experiment": "phase11_natural_prevalence",
                "fpr_target": FPR_TARGET,
                "eval_prevalence": {"ton_iot": float(y_ton_nat.mean()),
                                     "unsw_nb15": float(y_unsw_nat.mean())},
                "results": existing_results,
            }, f, indent=2)
        print(f"  [saved -> {out_path}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
