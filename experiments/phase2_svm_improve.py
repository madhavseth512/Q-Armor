"""Phase 2 follow-up: improve the SVM-RBF-10k baseline (diagnostics).

Targets the two measured causes of the weak 0.3525 kernel:
  1. Balanced-train / imbalanced-eval MISMATCH -> prior correction. Adjust the
     SVM's class probabilities by (pi_real / pi_train)^alpha and sweep alpha.
  2. SCALING -> the features are scaled to (0,pi) for quantum angle encoding, which
     may handicap a classical RBF. Compare against StandardScaler'd features.

Reference benchmark is RF-10k (0.4712) — what a strong model gets at this scale —
not RF-full (0.7376). Reports macro-F1 on a 200k stratified validation subsample.
Run:  ./venv/Scripts/python.exe -u -m experiments.phase2_svm_improve
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from agent import agent_config as config
from data.loader import load_dataset
from data.preprocess import (
    LOG1P_FEATURES,
    QuantumPreprocessor,
    majority_undersample_index,
)
from data.sampling import select_kernel_subset

SIZE = 10_000
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def macro(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=config.ATTACK_CLASSES)


def standard_features(pre, X_raw):
    f = pre.engineer_features(X_raw).copy()
    f[LOG1P_FEATURES] = np.log1p(f[LOG1P_FEATURES])
    return f


def prior_corrected_pred(proba, classes, pi_train, pi_real, alpha):
    w = (pi_real / pi_train) ** alpha
    adj = proba * w[None, :]
    return classes[adj.argmax(axis=1)]


def evaluate(name, model, Xva, yva, pi_train, pi_real):
    proba = model.predict_proba(Xva)
    classes = model.classes_
    print(f"\n[{name}]  prior-correction strength sweep:")
    best = (-1.0, 0.0)
    for a in ALPHAS:
        yp = prior_corrected_pred(proba, classes, pi_train, pi_real, a)
        f = macro(yva, yp)
        tag = "  (baseline)" if a == 0.0 else ""
        print(f"    alpha={a:.2f} -> macro-F1 {f:.4f}{tag}")
        if f > best[0]:
            best = (f, a)
    print(f"  -> best: macro-F1 {best[0]:.4f} at alpha={best[1]:.2f}")
    return best


def main() -> None:
    t0 = time.time()
    X_raw, y = load_dataset(config.TRAIN_CSV)
    Xva_raw, yva = load_dataset(config.VALIDATION_CSV, validate=False)
    keep = majority_undersample_index(y)
    Xtr_raw, ytr = X_raw.loc[keep], y.loc[keep]

    # real class priors from the FULL train distribution (the true deployment prior)
    pi_real = y.value_counts(normalize=True)

    pre = QuantumPreprocessor()
    X_angle_tr = pre.fit_transform(Xtr_raw)
    X_angle_va = pre.transform(Xva_raw)
    ss = StandardScaler().fit(standard_features(pre, Xtr_raw))
    X_std_tr = ss.transform(standard_features(pre, Xtr_raw))
    X_std_va = ss.transform(standard_features(pre, Xva_raw))

    idx = select_kernel_subset(X_angle_tr, np.asarray(ytr), SIZE, method="kmeans")
    ysub = np.asarray(ytr)[idx]
    print(f"setup done in {time.time()-t0:.0f}s | subset {len(idx)}")

    # disjoint 200k validation subsample for speed
    _, Xa_va, _, yv = train_test_split(X_angle_va, yva, test_size=200_000,
                                       stratify=yva, random_state=config.RANDOM_SEED)
    _, Xs_va, _, _ = train_test_split(X_std_va, yva, test_size=200_000,
                                      stratify=yva, random_state=config.RANDOM_SEED)

    pi_tr = pd.Series(ysub).value_counts(normalize=True)

    configs = [
        ("angle (0,pi), gamma=1",    X_angle_tr, Xa_va, dict(gamma=1, C=10)),
        ("angle (0,pi), gamma=scale", X_angle_tr, Xa_va, dict(gamma="scale", C=10)),
        ("StandardScaler, gamma=scale", X_std_tr, Xs_va, dict(gamma="scale", C=10)),
    ]
    results = []
    for name, Xtr_f, Xva_f, kw in configs:
        ts = time.time()
        svc = SVC(kernel="rbf", class_weight=None, random_state=config.RANDOM_SEED, **kw)
        model = CalibratedClassifierCV(svc, method="sigmoid", ensemble=False, cv=3).fit(Xtr_f[idx], ysub)
        order = model.classes_
        pit = np.array([pi_tr.get(c, 1e-9) for c in order])
        pir = np.array([pi_real.get(c, 1e-9) for c in order])
        best_f, best_a = evaluate(name, model, Xva_f, yv, pit, pir)
        results.append((name, best_f, best_a, time.time()-ts))

    print("\n" + "=" * 64 + "\nSUMMARY  (reference: SVM-10k 0.3525, RF-10k 0.4712)\n" + "=" * 64)
    for name, f, a, dt in results:
        print(f"  {name:<30} best macro-F1 {f:.4f} (alpha={a:.2f}) [{dt:.0f}s]")
    print(f"\nTotal {time.time()-t0:.0f}s  (test.csv untouched)")


if __name__ == "__main__":
    main()
