"""Phase 1 validation gates for Q-Armor.

Builds the two-sided-sampled training set (undersample majorities to a cap, SMOTE
minorities to a dynamic floor) on the TRAINING split only, then runs:

  * Gate B - diagnostic baseline: RandomForest on all raw features vs. on the 8
    engineered features. The 8-feature macro-F1 must land within ~5-7% of the
    raw-feature macro-F1, else the engineering lost signal.
  * Gate A - SHAP: every one of the 8 features must matter for >=1 of 15 classes.

Validation set is used for evaluation; the test set is left untouched for Phase 6.
Run:  ./venv/Scripts/python.exe -m experiments.phase1_gates
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from agent import agent_config as config
from data.loader import load_dataset
from data.preprocess import (
    QuantumPreprocessor,
    dynamic_smote_strategy,
    majority_undersample_index,
)

RF_KWARGS = dict(n_estimators=150, max_depth=16, n_jobs=-1, random_state=config.RANDOM_SEED)
SHAP_SAMPLE_PER_CLASS = 60   # stratified explanation subsample


def _hr(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def macro_f1(model, X, y) -> float:
    return f1_score(y, model.predict(X), average="macro", labels=config.ATTACK_CLASSES)


def main() -> None:
    t0 = time.time()

    # ---- load (train full; validation for eval; test untouched) -------------
    Xtr_full, ytr_full = load_dataset(config.TRAIN_CSV)
    Xva, yva = load_dataset(config.VALIDATION_CSV, validate=False)
    print(f"train {Xtr_full.shape} | val {Xva.shape} | loaded in {time.time()-t0:.1f}s")

    # ---- two-sided sampling on TRAIN only -----------------------------------
    keep = majority_undersample_index(ytr_full)
    Xtr, ytr = Xtr_full.loc[keep], ytr_full.loc[keep]
    del Xtr_full, ytr_full
    strategy = dynamic_smote_strategy(ytr)

    _hr("SAMPLING PLAN (train only)")
    plan = pd.DataFrame({"after_undersample": ytr.value_counts()})
    plan["smote_target"] = plan.index.map(lambda c: strategy.get(c, plan.loc[c, "after_undersample"]))
    plan["synthetic_%"] = (100 * (1 - plan["after_undersample"] / plan["smote_target"])).round(1)
    print(plan.sort_values("smote_target", ascending=False).to_string())

    # ---- Gate B path 1: RandomForest on ALL RAW features --------------------
    _hr("GATE B - baseline comparison")
    raw_cols = Xtr.columns.tolist()
    raw_scaler = StandardScaler().fit(Xtr[raw_cols])
    Xtr_raw = raw_scaler.transform(Xtr[raw_cols])
    Xtr_raw_res, y_raw_res = SMOTE(random_state=config.RANDOM_SEED,
                                   sampling_strategy=strategy).fit_resample(Xtr_raw, ytr)
    rf_raw = RandomForestClassifier(**RF_KWARGS).fit(Xtr_raw_res, y_raw_res)
    f1_raw = macro_f1(rf_raw, raw_scaler.transform(Xva[raw_cols]), yva)
    print(f"RAW   features ({len(raw_cols)}): val macro-F1 = {f1_raw:.4f}")

    # ---- Gate B path 2 + Gate A model: RF on the 8 ENGINEERED features -------
    pre = QuantumPreprocessor()
    Xtr8 = pre.fit_transform(Xtr)
    Xtr8_res, y8_res = SMOTE(random_state=config.RANDOM_SEED,
                             sampling_strategy=strategy).fit_resample(Xtr8, ytr)
    rf8 = RandomForestClassifier(**RF_KWARGS).fit(Xtr8_res, y8_res)
    Xva8 = pre.transform(Xva)
    f1_8 = macro_f1(rf8, Xva8, yva)
    print(f"8 eng features:            val macro-F1 = {f1_8:.4f}")

    gap = f1_raw - f1_8
    verdict = "PASS" if gap <= 0.07 else "REVISIT"
    print(f"\nGap (raw - 8eng) = {gap:+.4f}  ->  Gate B: {verdict} (threshold 0.07)")
    print("\nPer-class F1 (8 engineered features, on validation):")
    print(classification_report(yva, rf8.predict(Xva8),
                                labels=config.ATTACK_CLASSES, zero_division=0, digits=3))

    # ---- Gate A: SHAP on the 8-feature model --------------------------------
    _hr("GATE A - SHAP feature importance (8 features x 15 classes)")
    import shap

    parts = []
    va8_df = pd.DataFrame(Xva8, columns=pre.feature_names); va8_df["y"] = yva.values
    for cls in config.ATTACK_CLASSES:
        g = va8_df[va8_df["y"] == cls]
        if len(g):
            parts.append(g.sample(min(len(g), SHAP_SAMPLE_PER_CLASS),
                                  random_state=config.RANDOM_SEED))
    expl_df = pd.concat(parts)
    Xexpl = expl_df[pre.feature_names].to_numpy()

    sv = shap.TreeExplainer(rf8).shap_values(Xexpl)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)  # (n, feat, class)
    imp = np.abs(sv).mean(axis=0)                                           # (feat, class)
    imp_df = pd.DataFrame(imp, index=pre.feature_names, columns=rf8.classes_)

    print("Mean |SHAP| per feature per class:")
    print(imp_df.round(4).to_string())
    print("\nPer-feature: top class and its importance, share of that feature's row:")
    for f in pre.feature_names:
        row = imp_df.loc[f]
        top = row.idxmax()
        glob = row.sum()
        flag = "  <-- WEAK" if row.max() < 0.10 * imp_df.max(axis=1).max() else ""
        print(f"  {f:<18} max={row.max():.4f} @ {top:<20} total={glob:.4f}{flag}")

    weak = [f for f in pre.feature_names
            if imp_df.loc[f].max() < 0.10 * imp_df.max(axis=1).max()]
    print(f"\nGate A: {'PASS - every feature matters for >=1 class' if not weak else 'WEAK FEATURES: ' + str(weak)}")

    print(f"\nTotal runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
