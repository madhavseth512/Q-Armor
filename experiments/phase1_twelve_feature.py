"""Phase 1 measurement: does adding 4 diagnostic-backed features help?

Compares macro-F1 of RandomForest on:
  *  8 engineered features            (Gate B result: 0.4268)
  * 12 features = 8 + IAT, flow_duration, ack_count, Variance
  * 45 raw features                  (Gate B ceiling: 0.6754)

The 4 extra features are log1p'd then MinMax-scaled to (0, pi) (fit on train),
so all 12 share the angle-encoding scale and SMOTE distances stay balanced.
Same two-sided sampling; evaluated on the pristine validation set.

Run:  ./venv/Scripts/python.exe -m experiments.phase1_twelve_feature
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MinMaxScaler

from agent import agent_config as config
from data.loader import load_dataset
from data.preprocess import (
    QuantumPreprocessor,
    dynamic_smote_strategy,
    majority_undersample_index,
)

RF_KWARGS = dict(n_estimators=150, max_depth=16, n_jobs=-1, random_state=config.RANDOM_SEED)
EXTRA_COLS = ["IAT", "flow_duration", "ack_count", "Variance"]


def macro_f1(model, X, y) -> float:
    return f1_score(y, model.predict(X), average="macro", labels=config.ATTACK_CLASSES)


def main() -> None:
    t0 = time.time()
    Xtr_full, ytr_full = load_dataset(config.TRAIN_CSV)
    Xva, yva = load_dataset(config.VALIDATION_CSV, validate=False)
    keep = majority_undersample_index(ytr_full)
    Xtr, ytr = Xtr_full.loc[keep], ytr_full.loc[keep]
    del Xtr_full, ytr_full
    strategy = dynamic_smote_strategy(ytr)
    print(f"setup done in {time.time()-t0:.1f}s")

    # 8 engineered features (fit on train)
    pre = QuantumPreprocessor()
    X8_tr = pre.fit_transform(Xtr)
    X8_va = pre.transform(Xva)

    # 4 extra features: log1p -> MinMax(0, pi), fit on train, clip on val
    lo, hi = config.ANGLE_ENCODING_RANGE
    ex_scaler = MinMaxScaler(feature_range=config.ANGLE_ENCODING_RANGE)
    Xex_tr = ex_scaler.fit_transform(np.log1p(Xtr[EXTRA_COLS]))
    Xex_va = np.clip(ex_scaler.transform(np.log1p(Xva[EXTRA_COLS])), lo, hi)

    X12_tr = np.hstack([X8_tr, Xex_tr])
    X12_va = np.hstack([X8_va, Xex_va])

    # SMOTE (train only) + RF on the 12 features
    X12_res, y_res = SMOTE(random_state=config.RANDOM_SEED,
                           sampling_strategy=strategy).fit_resample(X12_tr, ytr)
    rf12 = RandomForestClassifier(**RF_KWARGS).fit(X12_res, y_res)
    f1_12 = macro_f1(rf12, X12_va, yva)

    print("\n" + "=" * 70)
    print("MACRO-F1 COMPARISON (validation)")
    print("=" * 70)
    print(f"   8 engineered features : 0.4268   (Gate B)")
    print(f"  12 features (8 + extras): {f1_12:.4f}   (this run)")
    print(f"  45 raw features         : 0.6754   (Gate B ceiling)")
    print(f"\n  recovered vs 8-feat: {f1_12 - 0.4268:+.4f}   |   "
          f"remaining gap to 45-raw: {0.6754 - f1_12:+.4f}")

    print("\nPer-class F1 (12 features):")
    print(classification_report(yva, rf12.predict(X12_va),
                                labels=config.ATTACK_CLASSES, zero_division=0, digits=3))
    print(f"Total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
