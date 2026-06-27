"""Phase 6 EDA — NF-ToN-IoT and NF-UNSW-NB15 feature analysis and classical baseline.

Four parts run in sequence:
  Part 1 — Data quality   : NaN counts, dtypes, row counts, schema match.
  Part 2 — Distributions  : percentile ranges for all 10 numeric features;
                             cross-dataset shift; heavy-tail detection.
  Part 3 — Importance     : RandomForest feature importances on both datasets;
                             average rank -> top-8 candidate (NF_FEATURE_NAMES).
  Part 4 — Baseline       : SVM-RBF + RF with all-10 vs top-8, linear vs log
                             scaling, within-dataset and cross-dataset AUROC.
                             Locks NF_FEATURE_NAMES and NF_SCALE_MODE.

Run:    ./venv/Scripts/python.exe -m experiments.phase6_eda
Output: results/phase6/phase6_eda.json
"""

from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from agent import agent_config as config

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TON_CSV  = "data/NF-ToN-IoT/NF-ToN-IoT.csv"
UNSW_CSV = "data/NF-UNSW-NB15/NF-UNSW-NB15.csv"
RESULTS_DIR = "results/phase6"

NF_NUMERIC_COLS: list[str] = [
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
]
NF_LABEL_COL  = "Label"   # binary: 0=benign, 1=attack
NF_ATTACK_COL = "Attack"  # multi-class string

# Features likely to be heavy-tailed -> candidates for log1p before MinMax
LOG_COLS: list[str] = [
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS",
]

TRAIN_N_PER_CLASS = 2500   # balanced training subset size (per class)
VAL_N_PER_CLASS   = 1000   # balanced val subset size (per class)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _balanced_sample(df: pd.DataFrame, n_per_class: int,
                     seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for label in [0, 1]:
        pool = df[df[NF_LABEL_COL] == label]
        n = min(n_per_class, len(pool))
        idx = rng.choice(len(pool), n, replace=False)
        parts.append(pool.iloc[idx])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def _scale(X: np.ndarray, cols: list[str], mode: str,
           scaler: MinMaxScaler | None = None) -> tuple[np.ndarray, MinMaxScaler]:
    """Scale X to [0, pi]. mode='linear' applies MinMax directly; mode='log'
    applies log1p to LOG_COLS first, then MinMax. Returns (X_scaled, scaler)."""
    X = X.copy().astype(float)
    if mode == "log":
        log_idx = [cols.index(c) for c in LOG_COLS if c in cols]
        X[:, log_idx] = np.log1p(X[:, log_idx])
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)))
        return scaler.fit_transform(X), scaler
    return scaler.transform(X), scaler


def _scores(clf, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def _metrics(clf, X: np.ndarray, y: np.ndarray) -> dict:
    s = _scores(clf, X)
    return {
        "auroc": float(roc_auc_score(y, s)),
        "aupr":  float(average_precision_score(y, s)),
    }


# ---------------------------------------------------------------------------
# Part 1 — Data quality
# ---------------------------------------------------------------------------

def part1_quality(ton: pd.DataFrame, unsw: pd.DataFrame) -> dict:
    print("\n" + "=" * 64)
    print("PART 1 -- Data quality")
    print("=" * 64)
    result: dict = {}

    for name, df in [("NF-ToN-IoT", ton), ("NF-UNSW-NB15", unsw)]:
        nan_counts   = df[NF_NUMERIC_COLS].isnull().sum().to_dict()
        label_dist   = df[NF_LABEL_COL].value_counts().to_dict()
        attack_dist  = df[NF_ATTACK_COL].value_counts().to_dict()
        total_nan    = sum(nan_counts.values())
        benign_pct   = 100.0 * label_dist.get(0, 0) / len(df)
        attack_pct   = 100.0 * label_dist.get(1, 0) / len(df)

        result[name] = {
            "rows": len(df),
            "cols": list(df.columns),
            "total_nan": int(total_nan),
            "nan_per_col": {k: int(v) for k, v in nan_counts.items()},
            "label_dist":  {int(k): int(v) for k, v in label_dist.items()},
            "attack_classes": {str(k): int(v) for k, v in attack_dist.items()},
        }
        print(f"\n  {name}")
        print(f"    rows        : {len(df):,}")
        print(f"    NaN total   : {total_nan}")
        print(f"    benign      : {label_dist.get(0,0):,}  ({benign_pct:.1f}%)")
        print(f"    attack      : {label_dist.get(1,0):,}  ({attack_pct:.1f}%)")
        print(f"    attack types: {list(attack_dist.keys())}")

    schemas_match = ton.columns.tolist() == unsw.columns.tolist()
    result["schemas_match"] = schemas_match
    print(f"\n  Schemas identical: {schemas_match}")
    return result


# ---------------------------------------------------------------------------
# Part 2 — Feature distributions
# ---------------------------------------------------------------------------

def part2_distributions(ton: pd.DataFrame, unsw: pd.DataFrame) -> dict:
    print("\n" + "=" * 64)
    print("PART 2 -- Feature distributions")
    print("=" * 64)

    pct_labels = [1, 25, 50, 75, 99]
    result: dict = {}
    heavy_tailed: list[str] = []

    hdr = f"  {'Feature':<30} {'Dataset':<15} {'p1':>10} {'p50':>12} {'p99':>14} {'max':>16} {'tail?':>6}"
    print(f"\n{hdr}")
    print("  " + "-" * 90)

    for col in NF_NUMERIC_COLS:
        col_result: dict = {}
        for name, df in [("NF-ToN-IoT", ton), ("NF-UNSW-NB15", unsw)]:
            vals = df[col].dropna().values
            pcts = np.percentile(vals, pct_labels)
            vmax = float(vals.max())
            p50  = float(pcts[2])
            p99  = float(pcts[4])
            tail = (p50 > 0 and (p99 / (p50 + 1e-9)) > 100)
            col_result[name] = {
                "p1": float(pcts[0]), "p25": float(pcts[1]), "p50": p50,
                "p75": float(pcts[3]), "p99": p99, "max": vmax,
                "mean": float(vals.mean()), "std": float(vals.std()),
                "heavy_tailed": tail,
            }
            if tail:
                heavy_tailed.append(col)
            flag = "YES" if tail else ""
            print(f"  {col:<30} {name:<15} {pcts[0]:>10.1f} {p50:>12.1f} {p99:>14.1f} {vmax:>16.1f} {flag:>6}")
        result[col] = col_result

    unique_heavy = sorted(set(heavy_tailed))
    result["heavy_tailed"] = unique_heavy
    print(f"\n  Heavy-tailed (p99/p50 > 100): {unique_heavy}")
    print(f"  Proposed log1p candidates   : {LOG_COLS}")
    overlap = [c for c in unique_heavy if c in LOG_COLS]
    missed  = [c for c in unique_heavy if c not in LOG_COLS]
    if missed:
        print(f"  WARNING: heavy-tailed but not in LOG_COLS: {missed}")
    return result


# ---------------------------------------------------------------------------
# Part 3 — Feature importance
# ---------------------------------------------------------------------------

def part3_importance(ton: pd.DataFrame, unsw: pd.DataFrame) -> dict:
    print("\n" + "=" * 64)
    print("PART 3 -- Feature importance (RandomForest)")
    print("=" * 64)

    importances: dict[str, dict[str, float]] = {}
    result: dict = {}

    for name, df in [("NF-ToN-IoT", ton), ("NF-UNSW-NB15", unsw)]:
        n_tr = TRAIN_N_PER_CLASS
        print(f"\n  {name}: balanced sample {n_tr*2} rows ...")
        sub = _balanced_sample(df, n_tr, seed=config.RANDOM_SEED)
        X   = sub[NF_NUMERIC_COLS].values.astype(float)
        y   = sub[NF_LABEL_COL].values

        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=config.RANDOM_SEED, n_jobs=-1,
        )
        rf.fit(X, y)
        print(f"  fit in {time.time()-t0:.1f}s")

        imp = dict(zip(NF_NUMERIC_COLS, rf.feature_importances_))
        importances[name] = imp
        ranked = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
        result[name] = {col: float(v) for col, v in ranked}

        print(f"  Importance ranking:")
        for rank, (col, val) in enumerate(ranked, 1):
            bar = "#" * max(1, int(val * 80))
            print(f"    {rank:2d}. {col:<32} {val:.4f}  {bar}")

    # Average across both datasets -> pick top 8
    avg_imp = {
        col: (importances["NF-ToN-IoT"][col] + importances["NF-UNSW-NB15"][col]) / 2
        for col in NF_NUMERIC_COLS
    }
    ranked_avg = sorted(avg_imp.items(), key=lambda kv: kv[1], reverse=True)
    top8   = [col for col, _ in ranked_avg[:8]]
    bottom2 = [col for col, _ in ranked_avg[8:]]

    result["avg_importance"] = {col: float(v) for col, v in ranked_avg}
    result["top8_candidate"] = top8
    result["bottom2_dropped"] = bottom2

    print(f"\n  Average importance (both datasets):")
    for rank, (col, val) in enumerate(ranked_avg, 1):
        tag = "  <- DROP" if col in bottom2 else ""
        print(f"    {rank:2d}. {col:<32} {val:.4f}{tag}")
    print(f"\n  Proposed NF_FEATURE_NAMES: {top8}")
    print(f"  Features to drop         : {bottom2}")
    return result


# ---------------------------------------------------------------------------
# Part 4 — Classical baseline
# ---------------------------------------------------------------------------

def part4_baseline(ton: pd.DataFrame, unsw: pd.DataFrame, top8: list[str]) -> dict:
    print("\n" + "=" * 64)
    print("PART 4 -- Classical baseline (within-dataset + cross-dataset)")
    print("=" * 64)

    # Build balanced train / val sets
    ton_train  = _balanced_sample(ton,  TRAIN_N_PER_CLASS, seed=config.RANDOM_SEED)
    ton_val    = _balanced_sample(ton,  VAL_N_PER_CLASS,   seed=config.RANDOM_SEED + 1)
    unsw_val   = _balanced_sample(unsw, VAL_N_PER_CLASS,   seed=config.RANDOM_SEED)
    print(f"\n  ton_train : {ton_train.shape}")
    print(f"  ton_val   : {ton_val.shape}    (within-dataset eval)")
    print(f"  unsw_val  : {unsw_val.shape}    (cross-dataset eval)")

    y_ton_val  = ton_val[NF_LABEL_COL].values
    y_unsw_val = unsw_val[NF_LABEL_COL].values

    # Four feature/scale configs
    configs = [
        ("all10_linear", NF_NUMERIC_COLS, "linear"),
        ("all10_log",    NF_NUMERIC_COLS, "log"),
        ("top8_linear",  top8,            "linear"),
        ("top8_log",     top8,            "log"),
    ]

    result: dict = {}

    print(f"\n  {'Config':<16} {'Model':<10} {'within_AUROC':>13} {'cross_AUROC':>12} {'drift_gap':>10} {'time':>6}")
    print("  " + "-" * 68)

    for cfg_name, feat_cols, scale_mode in configs:
        X_tr   = ton_train[feat_cols].values.astype(float)
        y_tr   = ton_train[NF_LABEL_COL].values
        X_tval = ton_val[feat_cols].values.astype(float)
        X_uval = unsw_val[feat_cols].values.astype(float)

        X_tr_s,   scaler = _scale(X_tr,   feat_cols, scale_mode)
        X_tval_s, _      = _scale(X_tval, feat_cols, scale_mode, scaler)
        X_uval_s, _      = _scale(X_uval, feat_cols, scale_mode, scaler)

        cfg_result: dict = {}

        for clf_name, clf in [
            ("RF", RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=config.RANDOM_SEED, n_jobs=-1)),
            ("SVM-RBF", SVC(
                kernel="rbf", class_weight="balanced",
                probability=True, random_state=config.RANDOM_SEED)),
        ]:
            t0 = time.time()
            clf.fit(X_tr_s, y_tr)
            elapsed = time.time() - t0

            within = _metrics(clf, X_tval_s, y_ton_val)
            cross  = _metrics(clf, X_uval_s, y_unsw_val)
            gap    = within["auroc"] - cross["auroc"]

            cfg_result[clf_name] = {
                "within_auroc": within["auroc"],
                "within_aupr":  within["aupr"],
                "cross_auroc":  cross["auroc"],
                "cross_aupr":   cross["aupr"],
                "drift_gap":    float(gap),
                "fit_time_s":   round(elapsed, 2),
            }
            print(f"  {cfg_name:<16} {clf_name:<10} {within['auroc']:>13.4f} {cross['auroc']:>12.4f} {gap:>+10.4f} {elapsed:>5.1f}s")

        result[cfg_name] = cfg_result

    # Determine best config by cross-dataset RF AUROC (most important for Phase 6)
    best_cfg = max(
        [c for c in configs],
        key=lambda c: result[c[0]]["RF"]["cross_auroc"],
    )
    best_name, best_feats, best_scale = best_cfg
    result["best_config"] = {
        "name":       best_name,
        "features":   best_feats,
        "scale_mode": best_scale,
        "cross_auroc_rf": result[best_name]["RF"]["cross_auroc"],
    }
    print(f"\n  Best config by cross-dataset RF AUROC: {best_name}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("Phase 6 EDA -- NF-ToN-IoT + NF-UNSW-NB15")
    print("=" * 64)

    # Load both datasets
    print("\nLoading NF-ToN-IoT ...")
    t0 = time.time()
    ton = pd.read_csv(TON_CSV)
    print(f"  loaded {len(ton):,} rows in {time.time()-t0:.1f}s")

    print("Loading NF-UNSW-NB15 ...")
    t0 = time.time()
    unsw = pd.read_csv(UNSW_CSV)
    print(f"  loaded {len(unsw):,} rows in {time.time()-t0:.1f}s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output: dict = {}

    output["part1_quality"]       = part1_quality(ton, unsw)
    output["part2_distributions"] = part2_distributions(ton, unsw)
    output["part3_importance"]    = part3_importance(ton, unsw)

    top8 = output["part3_importance"]["top8_candidate"]
    output["part4_baseline"]      = part4_baseline(ton, unsw, top8)

    # Save JSON
    out_path = f"{RESULTS_DIR}/phase6_eda.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── Decisions locked ────────────────────────────────────────────────────
    best  = output["part4_baseline"]["best_config"]
    heavy = output["part2_distributions"]["heavy_tailed"]

    print("\n" + "=" * 64)
    print("DECISIONS LOCKED")
    print("=" * 64)
    print(f"  NF_FEATURE_NAMES  : {top8}")
    print(f"  NF_SCALE_MODE     : {best['scale_mode']}")
    print(f"  Heavy-tailed cols : {heavy}")
    print(f"  Best config       : {best['name']}  (cross AUROC RF={best['cross_auroc_rf']:.4f})")
    print(f"\n  -> Copy NF_FEATURE_NAMES and NF_SCALE_MODE into agent_config.py")
    print(f"\nResults saved -> {out_path}")
    print("(test sets untouched)")


if __name__ == "__main__":
    main()
