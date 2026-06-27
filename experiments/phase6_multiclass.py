"""Phase 6 — Experiment E3: multi-class attack-type classification on NF datasets.

Three parts:
  E3a — Within NF-ToN-IoT   : RF on native 9-class fine-grained labels.
  E3b — Within NF-UNSW-NB15 : RF on native 9-class fine-grained labels.
  E3c — Cross-dataset        : train on NF-ToN-IoT (coarse 5-class taxonomy),
                               eval on NF-UNSW-NB15 (same coarse taxonomy).

The coarse taxonomy (Benign/DoS/Injection/Recon/Backdoor) creates a common
label space across both datasets, enabling the cross-dataset multi-class
experiment that binary detection cannot provide.

Metrics: macro-F1, per-class F1, accuracy.

Run:  ./venv/Scripts/python.exe -m experiments.phase6_multiclass
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from agent import agent_config as config
from data.nf_loader import NFPreprocessor, read_nf
from data.taxonomy import apply_taxonomy
from reasoning.classical import RandomForestModel

RESULTS_DIR = "results/phase6"
TRAIN_N_PER_CLASS = 1000   # balanced per-class cap for training subsets
VAL_N_PER_CLASS   = 400    # balanced per-class cap for val subsets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _balanced_sample(X: np.ndarray, y: np.ndarray, n_per_class: int,
                     seed: int = config.RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return a class-balanced subsample capped at n_per_class per class."""
    rng = np.random.default_rng(seed)
    idx_parts = []
    for cls in np.unique(y):
        pos = np.where(y == cls)[0]
        n = min(n_per_class, len(pos))
        idx_parts.append(rng.choice(pos, n, replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _report(name: str, y_true: np.ndarray, y_pred: np.ndarray,
            classes: list[str]) -> dict:
    """Compute and print a classification report; return metrics dict."""
    rep = classification_report(y_true, y_pred, labels=classes,
                                output_dict=True, zero_division=0)
    macro_f1 = rep["macro avg"]["f1-score"]
    acc      = rep["accuracy"]
    per_cls  = {c: round(rep.get(c, {}).get("f1-score", 0.0), 4) for c in classes}

    print(f"\n  {name}  macro-F1={macro_f1:.4f}  acc={acc:.4f}")
    print(f"  {'Class':<16} {'F1':>7}  {'Support':>8}")
    print("  " + "-" * 36)
    for cls in classes:
        row = rep.get(cls, {})
        print(f"  {cls:<16} {row.get('f1-score', 0.0):>7.4f}  {int(row.get('support', 0)):>8}")
    return {"model": name, "macro_f1": round(macro_f1, 4),
            "accuracy": round(acc, 4), "per_class_f1": per_cls}


# ---------------------------------------------------------------------------
# E3a — Within NF-ToN-IoT (native fine-grained labels)
# ---------------------------------------------------------------------------

def e3a_within_ton(X8_tr: np.ndarray, y_tr: np.ndarray,
                   X8_va: np.ndarray, y_va: np.ndarray) -> dict:
    print("\n" + "=" * 60)
    print("E3a — Within NF-ToN-IoT (native fine-grained labels)")
    print("=" * 60)

    classes = sorted(set(np.unique(y_tr)) | set(np.unique(y_va)))
    X_tr_s, y_tr_s = _balanced_sample(X8_tr, y_tr, TRAIN_N_PER_CLASS)
    X_va_s, y_va_s = _balanced_sample(X8_va, y_va, VAL_N_PER_CLASS)
    print(f"  train subset: {X_tr_s.shape}  classes: {dict(zip(*np.unique(y_tr_s, return_counts=True)))}")
    print(f"  val subset  : {X_va_s.shape}")

    t0 = time.time()
    rf = RandomForestModel(class_weight="balanced").fit(X_tr_s, y_tr_s)
    print(f"  RF fit in {time.time()-t0:.1f}s")

    y_pred = rf.classes_[rf.predict_proba(X_va_s).argmax(axis=1)]
    return _report("rf_ton_finegrained", y_va_s, y_pred, classes)


# ---------------------------------------------------------------------------
# E3b — Within NF-UNSW-NB15 (native fine-grained labels)
# ---------------------------------------------------------------------------

def e3b_within_unsw(X8_tr: np.ndarray, y_tr: np.ndarray,
                    X8_va: np.ndarray, y_va: np.ndarray) -> dict:
    print("\n" + "=" * 60)
    print("E3b — Within NF-UNSW-NB15 (native fine-grained labels)")
    print("=" * 60)

    classes = sorted(set(np.unique(y_tr)) | set(np.unique(y_va)))
    X_tr_s, y_tr_s = _balanced_sample(X8_tr, y_tr, TRAIN_N_PER_CLASS)
    X_va_s, y_va_s = _balanced_sample(X8_va, y_va, VAL_N_PER_CLASS)
    print(f"  train subset: {X_tr_s.shape}  classes: {dict(zip(*np.unique(y_tr_s, return_counts=True)))}")
    print(f"  val subset  : {X_va_s.shape}")

    t0 = time.time()
    rf = RandomForestModel(class_weight="balanced").fit(X_tr_s, y_tr_s)
    print(f"  RF fit in {time.time()-t0:.1f}s")

    y_pred = rf.classes_[rf.predict_proba(X_va_s).argmax(axis=1)]
    return _report("rf_unsw_finegrained", y_va_s, y_pred, classes)


# ---------------------------------------------------------------------------
# E3c — Cross-dataset coarse taxonomy
# ---------------------------------------------------------------------------

def e3c_cross_coarse(X8_ton: np.ndarray, y_ton_coarse: np.ndarray,
                     X8_unsw: np.ndarray, y_unsw_coarse: np.ndarray) -> dict:
    print("\n" + "=" * 60)
    print("E3c — Cross-dataset coarse taxonomy (NF-ToN-IoT -> NF-UNSW-NB15)")
    print("=" * 60)

    classes = config.NF_COARSE_CLASSES
    X_tr_s, y_tr_s = _balanced_sample(X8_ton,  y_ton_coarse,  TRAIN_N_PER_CLASS)
    X_va_s, y_va_s = _balanced_sample(X8_unsw, y_unsw_coarse, VAL_N_PER_CLASS)
    print(f"  train subset (ToN-IoT)    : {X_tr_s.shape}  {dict(zip(*np.unique(y_tr_s, return_counts=True)))}")
    print(f"  val subset   (UNSW-NB15)  : {X_va_s.shape}  {dict(zip(*np.unique(y_va_s, return_counts=True)))}")

    t0 = time.time()
    rf = RandomForestModel(class_weight="balanced").fit(X_tr_s, y_tr_s)
    print(f"  RF fit in {time.time()-t0:.1f}s")

    y_pred = rf.classes_[rf.predict_proba(X_va_s).argmax(axis=1)]
    return _report("rf_cross_coarse", y_va_s, y_pred, classes)


# ---------------------------------------------------------------------------
# Also run within-dataset coarse for fair comparison with E3c
# ---------------------------------------------------------------------------

def e3d_within_coarse(name: str, X8_tr: np.ndarray, y_tr_coarse: np.ndarray,
                      X8_va: np.ndarray, y_va_coarse: np.ndarray) -> dict:
    print("\n" + "=" * 60)
    print(f"E3d — Within {name} (coarse taxonomy, for drift comparison)")
    print("=" * 60)

    classes = config.NF_COARSE_CLASSES
    X_tr_s, y_tr_s = _balanced_sample(X8_tr, y_tr_coarse, TRAIN_N_PER_CLASS)
    X_va_s, y_va_s = _balanced_sample(X8_va, y_va_coarse, VAL_N_PER_CLASS)
    print(f"  train subset: {X_tr_s.shape}")
    print(f"  val subset  : {X_va_s.shape}")

    t0 = time.time()
    rf = RandomForestModel(class_weight="balanced").fit(X_tr_s, y_tr_s)
    print(f"  RF fit in {time.time()-t0:.1f}s")

    y_pred = rf.classes_[rf.predict_proba(X_va_s).argmax(axis=1)]
    tag = f"rf_{name.lower().replace('-', '_')}_coarse"
    return _report(tag, y_va_s, y_pred, classes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 6 E3: Multi-class attack-type classification")
    print("=" * 60)

    # ── Load NF-ToN-IoT ───────────────────────────────────────────────────
    print(f"\nLoading {config.NF_TON_CSV} ...")
    t0 = time.time()
    X_ton_df, y_ton_bin, y_ton_att = read_nf(config.NF_TON_CSV)
    print(f"  {len(y_ton_bin):,} rows  ({time.time()-t0:.1f}s)")

    # ── Load NF-UNSW-NB15 ─────────────────────────────────────────────────
    print(f"Loading {config.NF_UNSW_CSV} ...")
    t0 = time.time()
    X_unsw_df, y_unsw_bin, y_unsw_att = read_nf(config.NF_UNSW_CSV)
    print(f"  {len(y_unsw_bin):,} rows  ({time.time()-t0:.1f}s)")

    # ── Apply coarse taxonomy ─────────────────────────────────────────────
    y_ton_coarse  = apply_taxonomy(y_ton_att)
    y_unsw_coarse = apply_taxonomy(y_unsw_att)
    print("\nCoarse class counts after taxonomy mapping:")
    for name, y in [("NF-ToN-IoT", y_ton_coarse), ("NF-UNSW-NB15", y_unsw_coarse)]:
        cls, cnt = np.unique(y, return_counts=True)
        print(f"  {name}: { {c: int(n) for c, n in zip(cls, cnt)} }")

    # ── Scale: fit on NF-ToN-IoT train split ─────────────────────────────
    (X_ton_tr_df, X_ton_va_df,
     y_ton_att_tr, y_ton_att_va,
     y_ton_coarse_tr, y_ton_coarse_va) = train_test_split(
        X_ton_df, y_ton_att, y_ton_coarse,
        test_size=0.20, stratify=y_ton_coarse,
        random_state=config.RANDOM_SEED,
    )
    (X_unsw_tr_df, X_unsw_va_df,
     y_unsw_att_tr, y_unsw_att_va,
     y_unsw_coarse_tr, y_unsw_coarse_va) = train_test_split(
        X_unsw_df, y_unsw_att, y_unsw_coarse,
        test_size=0.20, stratify=y_unsw_coarse,
        random_state=config.RANDOM_SEED,
    )

    pre_ton  = NFPreprocessor()
    pre_unsw = NFPreprocessor()

    print("\nScaling ...")
    X8_ton_tr  = pre_ton.fit_transform(X_ton_tr_df)
    X8_ton_va  = pre_ton.transform(X_ton_va_df)
    X8_unsw_tr = pre_unsw.fit_transform(X_unsw_tr_df)
    X8_unsw_va = pre_unsw.transform(X_unsw_va_df)

    # Cross-dataset: fit scaler on ALL of ToN-IoT, apply to ALL of UNSW
    pre_cross = NFPreprocessor()
    X8_ton_all  = pre_cross.fit_transform(X_ton_df)
    X8_unsw_all = pre_cross.transform(X_unsw_df)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    # ── E3a: within NF-ToN-IoT (native labels) ───────────────────────────
    r = e3a_within_ton(X8_ton_tr, y_ton_att_tr, X8_ton_va, y_ton_att_va)
    all_results.append(r)

    # ── E3b: within NF-UNSW-NB15 (native labels) ─────────────────────────
    r = e3b_within_unsw(X8_unsw_tr, y_unsw_att_tr, X8_unsw_va, y_unsw_att_va)
    all_results.append(r)

    # ── E3d: within each dataset (coarse) — drift comparison baseline ─────
    r = e3d_within_coarse("NF-ToN-IoT", X8_ton_tr, y_ton_coarse_tr,
                           X8_ton_va, y_ton_coarse_va)
    all_results.append(r)

    r = e3d_within_coarse("NF-UNSW-NB15", X8_unsw_tr, y_unsw_coarse_tr,
                           X8_unsw_va, y_unsw_coarse_va)
    all_results.append(r)

    # ── E3c: cross-dataset (coarse) ───────────────────────────────────────
    r = e3c_cross_coarse(X8_ton_all, y_ton_coarse, X8_unsw_all, y_unsw_coarse)
    all_results.append(r)

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "experiment": "phase6_multiclass",
        "features": config.NF_FEATURE_NAMES,
        "coarse_taxonomy": config.NF_COARSE_TAXONOMY,
        "results": all_results,
    }
    path = f"{RESULTS_DIR}/phase6_multiclass_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved -> {path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — macro-F1 across all experiments")
    print("=" * 60)
    print(f"  {'Experiment':<40} {'macro-F1':>9}")
    print("  " + "-" * 52)
    for r in all_results:
        print(f"  {r['model']:<40} {r['macro_f1']:>9.4f}")
    print("\n(test sets untouched)")


if __name__ == "__main__":
    main()
