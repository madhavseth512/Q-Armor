"""Phase 1 overfitting check — TRAIN vs VALIDATION only (test.csv untouched).

Trains the locked smart-8 RandomForest (same two-sided sampling as the gates),
then measures macro-F1 on the REAL (non-SMOTE) training rows vs the held-out
validation set. A small train-val gap = healthy generalization; a large gap =
the forest is memorising. Also reports per-class to see if any rare class is
memorised (high train recall, low val recall = SMOTE-artifact overfitting).

Run:  ./venv/Scripts/python.exe -m experiments.phase1_overfit_check
"""

from __future__ import annotations

import time

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

from agent import agent_config as config
from data.loader import load_dataset
from data.preprocess import (
    QuantumPreprocessor,
    dynamic_smote_strategy,
    majority_undersample_index,
)

RF_KWARGS = dict(n_estimators=150, max_depth=16, n_jobs=-1, random_state=config.RANDOM_SEED)


def mf1(model, X, y) -> float:
    return f1_score(y, model.predict(X), average="macro", labels=config.ATTACK_CLASSES)


def main() -> None:
    t0 = time.time()
    Xtr_full, ytr_full = load_dataset(config.TRAIN_CSV)
    Xva, yva = load_dataset(config.VALIDATION_CSV, validate=False)
    keep = majority_undersample_index(ytr_full)
    Xtr, ytr = Xtr_full.loc[keep], ytr_full.loc[keep]
    del Xtr_full, ytr_full
    strategy = dynamic_smote_strategy(ytr)

    pre = QuantumPreprocessor()
    X8_tr = pre.fit_transform(Xtr)     # REAL training rows (pre-SMOTE)
    X8_va = pre.transform(Xva)
    Xr, yr = SMOTE(random_state=config.RANDOM_SEED, sampling_strategy=strategy).fit_resample(X8_tr, ytr)
    rf = RandomForestClassifier(**RF_KWARGS).fit(Xr, yr)
    print(f"trained in {time.time()-t0:.0f}s")

    f1_train = mf1(rf, X8_tr, ytr)     # on real (imbalanced) training rows
    f1_val = mf1(rf, X8_va, yva)
    print("\n" + "=" * 60)
    print("OVERFITTING CHECK (train vs validation)")
    print("=" * 60)
    print(f"  macro-F1  REAL train : {f1_train:.4f}")
    print(f"  macro-F1  validation : {f1_val:.4f}")
    print(f"  gap (train - val)    : {f1_train - f1_val:+.4f}")

    # per-class recall side by side: spot rare-class memorisation
    rep_tr = classification_report(ytr, rf.predict(X8_tr), labels=config.ATTACK_CLASSES,
                                   output_dict=True, zero_division=0)
    rep_va = classification_report(yva, rf.predict(X8_va), labels=config.ATTACK_CLASSES,
                                   output_dict=True, zero_division=0)
    print("\n  per-class F1:           train    val     gap")
    for c in config.ATTACK_CLASSES:
        ft, fv = rep_tr[c]["f1-score"], rep_va[c]["f1-score"]
        print(f"    {c:<22} {ft:.3f}   {fv:.3f}   {ft-fv:+.3f}")

    print(f"\nTotal runtime {time.time()-t0:.1f}s   (test.csv NOT touched)")


if __name__ == "__main__":
    main()
