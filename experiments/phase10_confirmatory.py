"""Phase 10 — Confirmatory Protocol: equal-budget classical baselines +
target-label budget sweep + multi-seed repetition with bootstrap CI.

Closes two Confirmatory Protocol items from docs/paper.tex (Sec. "Confirmatory
Protocol" / Table "Validation protocol status"):
  - Equal-budget classical baselines: linear SVM, RBF-SVM, Random Forest,
    gradient-boosted trees, compared against PegasosQSVC under the SAME
    target-label budget.
  - Budget sweep B in {0, 25, 50, 100, 150, 300} (Eq. "budgets").
  - Multi-seed repetition (>=10 seeds) with mean, std, and 95% bootstrap CI.

Design notes on test-set reuse (read before extending this script):
  The evaluation split (NF-UNSW-NB15 80/20, random_state=RANDOM_SEED,
  capped to QSVC_CAP=1000) is the SAME split experiments/phase8_final_eval.py
  already unsealed and reported as T2/T3. This script does NOT draw a fresh
  test partition. That is deliberate, not an oversight: the paper already
  frames E8c's split as a "held-out evaluation subset", not a pristine sealed
  set (see Sec. "Prototype Protocol"), precisely because Phase 6/7 touched
  pieces of the same source data during development. Reusing the already-
  reported split for further honest characterisation (more seeds, more
  budgets, more baselines) is what the Confirmatory Protocol section itself
  asks for; it is not "tuning after viewing" because nothing here feeds back
  into a hyperparameter or design choice -- every (model, B, seed) result is
  recorded and reported, including any that look worse.

  B=0 ("source-only") is a single fixed row per model type: trained once on
  a 150-sample k-means subset of the NF-ToN-IoT TRAIN pool (matching
  QSVC_SUBSET_SIZE, so it is an equal-budget baseline in its own right) and
  evaluated cross-domain with no target-domain adaptation. It does not vary
  by seed because no target-domain randomness is involved.

  B>0 rows are "target-adapted": a fresh model of each type is trained from
  scratch on ONLY B target-domain (NF-UNSW-NB15) samples drawn from the
  TRAIN pool for that seed (balanced where possible), then evaluated on the
  shared test split above. For PegasosQSVC this mirrors SWITCH_SUBSET's own
  methodology (k-means selection down to size B from a larger random pool,
  via agent.retrainer.SubsetRetrainer); classical baselines use a plain
  balanced random draw of size B (they don't need a kernel-training subset).

Run (full 10-seed, all 6 budgets, all 5 models -- expensive, run in background):
  ./venv/Scripts/python.exe -m experiments.phase10_confirmatory --confirm

Run a cheaper pilot first:
  ./venv/Scripts/python.exe -m experiments.phase10_confirmatory --confirm \
      --seeds 3 --budgets 0,50,150,300 --models random_forest,svm_rbf,svm_linear,gbt

Skip the expensive quantum model entirely (classical-only, fast):
  ./venv/Scripts/python.exe -m experiments.phase10_confirmatory --confirm \
      --models random_forest,svm_rbf,svm_linear,gbt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from agent import agent_config as config
from agent.cascade_evaluator import _fpr_at_tpr
from agent.retrainer import SubsetRetrainer
from data.nf_loader import NFPreprocessor, read_nf
from data.sampling import select_kernel_subset
from reasoning.classical import GBTModel, LinearSVMModel, RandomForestModel, SVMModel
from reasoning.quantum import PegasosQSVCModel

RESULTS_DIR = "results/phase10"

MODEL_REGISTRY = {
    "pegasos_qsvc":  lambda: PegasosQSVCModel(num_steps=config.PEGASOS_TAU),
    "random_forest": lambda: RandomForestModel(class_weight=None),
    "svm_rbf":       lambda: SVMModel(),
    "svm_linear":    lambda: LinearSVMModel(),
    "gbt":           lambda: GBTModel(),
}
DEFAULT_BUDGETS = [0, 25, 50, 100, 150, 300]
DEFAULT_MODELS = list(MODEL_REGISTRY.keys())
SOURCE_TRAIN_N = config.QSVC_SUBSET_SIZE  # 150 -- equal-budget source baseline


def _balanced_draw(X, y, n, rng):
    """Draw ~n samples, balanced across the two binary classes where possible."""
    b_idx = np.where(y == 0)[0]
    a_idx = np.where(y == 1)[0]
    half = n // 2
    chosen = np.concatenate([
        rng.choice(b_idx, min(half, len(b_idx)), replace=False),
        rng.choice(a_idx, min(n - half, len(a_idx)), replace=False),
    ])
    rng.shuffle(chosen)
    return X[chosen], y[chosen]


def _fit_model(name: str, X: np.ndarray, y: np.ndarray):
    return MODEL_REGISTRY[name]().fit(X, y)


def _evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict_labels(X_test)
    return {
        "auroc":  float(roc_auc_score(y_test, proba)),
        "f1":     float(f1_score(y_test, y_pred)),
        "fpr95":  _fpr_at_tpr(y_test, proba),
    }


def _bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI over a list of per-seed metric values."""
    arr = np.asarray(values)
    if len(arr) < 2:
        return (float(arr[0]), float(arr[0])) if len(arr) else (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _summarise(rows: list[dict]) -> dict:
    """Mean / std / 95% bootstrap CI across seeds for one (model, budget) cell."""
    out = {}
    for metric in ("auroc", "f1", "fpr95"):
        vals = [r[metric] for r in rows]
        lo, hi = _bootstrap_ci(vals)
        out[metric] = {
            "mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "ci95_lo": lo, "ci95_hi": hi, "n_seeds": len(vals),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true",
                        help="Confirm you understand this reuses the E8c evaluation split.")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of independent seeds per (model, budget>0) cell.")
    parser.add_argument("--budgets", type=str, default=",".join(map(str, DEFAULT_BUDGETS)),
                        help="Comma-separated target-label budgets.")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                        help="Comma-separated model names from MODEL_REGISTRY.")
    parser.add_argument("--eval-cap", type=int, default=1000,
                        help="Shared eval-subset size (balanced). QSVC predict cost "
                             "scales with n_train x n_test -- shrink this for a fast "
                             "pilot before committing to the full 1000-sample cap.")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    models = args.models.split(",")
    for m in models:
        if m not in MODEL_REGISTRY:
            sys.exit(f"Unknown model '{m}'. Choices: {list(MODEL_REGISTRY)}")

    if not args.confirm:
        print("=" * 70)
        print("Phase 10: Confirmatory Protocol -- budget sweep + multi-seed CI")
        print("=" * 70)
        print(f"""
  This reuses the SAME NF-UNSW-NB15 evaluation split that
  experiments/phase8_final_eval.py already unsealed and reported as T2/T3
  (see the module docstring for why that is methodologically sound here).

  Seeds:   {args.seeds}
  Budgets: {budgets}
  Models:  {models}

  To proceed:
    ./venv/Scripts/python.exe -m experiments.phase10_confirmatory --confirm \\
        --seeds {args.seeds} --budgets {args.budgets} --models {args.models}
""")
        sys.exit(0)

    print("=" * 70)
    print(f"Phase 10: seeds={args.seeds}  budgets={budgets}  models={models}")
    print("=" * 70)

    # ── 1. Load + split (identical to phase8_final_eval.py) ───────────────────
    print(f"\nLoading {config.NF_TON_CSV} ...")
    X_ton_df, y_ton_bin, _ = read_nf(config.NF_TON_CSV)
    print(f"Loading {config.NF_UNSW_CSV} ...")
    X_unsw_df, y_unsw_bin, _ = read_nf(config.NF_UNSW_CSV)

    X_ton_tr_df, _, y_ton_bin_tr, _ = train_test_split(
        X_ton_df, y_ton_bin, test_size=0.20, stratify=y_ton_bin,
        random_state=config.RANDOM_SEED,
    )
    X_unsw_tr_df, X_unsw_te_df, y_unsw_bin_tr, y_unsw_bin_te = train_test_split(
        X_unsw_df, y_unsw_bin, test_size=0.20, stratify=y_unsw_bin,
        random_state=config.RANDOM_SEED,
    )

    pre = NFPreprocessor()
    X8_ton_tr = pre.fit_transform(X_ton_tr_df)
    X8_unsw_tr = pre.transform(X_unsw_tr_df)
    X8_unsw_te = pre.transform(X_unsw_te_df)

    # Capped test subset (same seed offset as phase8_final_eval.py; size is
    # configurable here since QSVC predict cost scales with n_train x n_test).
    QSVC_CAP = args.eval_cap
    rng900 = np.random.default_rng(config.RANDOM_SEED + 900)
    b = np.where(y_unsw_bin_te == 0)[0]
    a = np.where(y_unsw_bin_te == 1)[0]
    te_idx = np.concatenate([
        rng900.choice(b, min(QSVC_CAP // 2, len(b)), replace=False),
        rng900.choice(a, min(QSVC_CAP // 2, len(a)), replace=False),
    ])
    rng900.shuffle(te_idx)
    X_test, y_test = X8_unsw_te[te_idx], y_unsw_bin_te[te_idx]
    print(f"Shared eval subset: {X_test.shape}  binary {np.bincount(y_test)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results: dict[str, dict[int, dict]] = {}
    path = f"{RESULTS_DIR}/phase10_confirmatory_metrics.json"

    def _persist():
        # Merge with any existing file rather than overwrite -- models and
        # budgets are often run in separate invocations (QSVC is far slower
        # than the classical baselines, and budgets get filled in
        # incrementally), so a fresh in-memory `results` dict here must not
        # clobber results a prior invocation already saved for this model at
        # other budgets, or for other models entirely.
        existing: dict[str, dict[str, dict]] = {}
        if os.path.exists(path):
            with open(path) as ef:
                existing = json.load(ef).get("results", {})
        for m, bd in results.items():
            merged_budgets = existing.get(m, {})
            merged_budgets.update({str(b): v for b, v in bd.items()})
            existing[m] = merged_budgets

        out = {
            "experiment": "phase10_confirmatory",
            "n_seeds": args.seeds,
            "budgets": budgets,
            "models_completed": list(existing.keys()),
            "source_train_n": SOURCE_TRAIN_N,
            "results": existing,
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  [saved -> {path}]")

    for model_name in models:
        print(f"\n{'=' * 60}\nMODEL: {model_name}\n{'=' * 60}")
        results[model_name] = {}

        # -- B=0: source-only, fixed, no target-domain data ---------------------
        t0 = time.time()
        src_idx = select_kernel_subset(X8_ton_tr, y_ton_bin_tr, size=SOURCE_TRAIN_N,
                                        method="kmeans") if model_name == "pegasos_qsvc" \
            else np.random.default_rng(config.RANDOM_SEED).choice(
                len(y_ton_bin_tr), SOURCE_TRAIN_N, replace=False)
        m0 = _fit_model(model_name, X8_ton_tr[src_idx], y_ton_bin_tr[src_idx])
        row0 = _evaluate(m0, X_test, y_test)
        print(f"  B=0    (source-only, n={SOURCE_TRAIN_N})  "
              f"AUROC={row0['auroc']:.4f}  F1={row0['f1']:.4f}  "
              f"FPR95={row0['fpr95']:.4f}  ({time.time() - t0:.1f}s)")
        results[model_name][0] = _summarise([row0])
        results[model_name][0]["raw"] = [row0]
        _persist()

        # -- B>0: target-adapted, multi-seed ------------------------------------
        for B in budgets:
            if B == 0:
                continue
            rows = []
            t0 = time.time()
            for s in range(args.seeds):
                seed = config.RANDOM_SEED + 2000 + s
                rng = np.random.default_rng(seed)
                if model_name == "pegasos_qsvc":
                    # Balanced pool draw, not plain uniform random -- UNSW-NB15's
                    # train pool is ~4.5% attack, so a uniform draw can starve the
                    # minority class below what SubsetRetrainer needs for a
                    # balanced n_sub-sized kmeans subset (hit in practice: a 300
                    # -row uniform pool drew only 13 attack rows against 25 needed).
                    pool_half = max(B, config.SWITCH_SUBSET_N_CROSS // 2)
                    X_pool, y_pool = _balanced_draw(
                        X8_unsw_tr, y_unsw_bin_tr, pool_half * 2, rng)
                    retrainer = SubsetRetrainer()
                    m, _, _ = retrainer.retrain(
                        X_pool, y_pool,
                        n_sub=B, n_steps=config.PEGASOS_TAU, verbose=False)
                else:
                    X_b, y_b = _balanced_draw(X8_unsw_tr, y_unsw_bin_tr, B, rng)
                    m = _fit_model(model_name, X_b, y_b)
                rows.append(_evaluate(m, X_test, y_test))
            elapsed = time.time() - t0
            summary = _summarise(rows)
            print(f"  B={B:<4} ({args.seeds} seeds)  "
                  f"AUROC={summary['auroc']['mean']:.4f}+-{summary['auroc']['std']:.4f}  "
                  f"F1={summary['f1']['mean']:.4f}+-{summary['f1']['std']:.4f}  "
                  f"FPR95={summary['fpr95']['mean']:.4f}+-{summary['fpr95']['std']:.4f}  "
                  f"({elapsed:.1f}s)")
            summary["raw"] = rows
            results[model_name][B] = summary
            _persist()

    print(f"\nDone. Full results -> {path}")


if __name__ == "__main__":
    main()
