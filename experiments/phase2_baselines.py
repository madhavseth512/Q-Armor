"""Phase 2 classical baselines — orchestrator.

Stages (pass as argv[1], default 'rf_full'):
  rf_full   - Exp 1: RandomForest on the full ~400k two-sided-sampled set (ceiling).
  subset    - Exp 2+3: build the 10k kernel subset (random vs k-means ablation),
              CV-tune + train SVM-RBF, train RF on the same 10k comparator.
  spotcheck - Exp 4: SVM + RF at 20k to test the learning-curve slope (lock size).
  all       - run everything in sequence.

Every model: macro-F1 / weighted-F1 / accuracy / per-class report (JSON) +
row-normalised confusion matrix (PNG) -> results/phase2/. Models -> models/.
Evaluated on validation.csv; test.csv is never touched.

Run:  ./venv/Scripts/python.exe -m experiments.phase2_baselines [stage]
"""

from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

from agent import agent_config as config
from data.loader import load_dataset
from data.preprocess import (
    QuantumPreprocessor,
    dynamic_smote_strategy,
    majority_undersample_index,
)
from data.sampling import select_kernel_subset
from reasoning.classical import RandomForestModel, SVMModel

RESULTS_DIR = "results/phase2"
MODELS_DIR = "models"
SVM_GRID = {"C": [1, 10, 100], "gamma": ["scale", 0.1, 1]}


def _labels(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    return model.classes_[proba.argmax(axis=1)]


def evaluate(name: str, model, X_va: np.ndarray, y_va: np.ndarray) -> float:
    """Score a model on validation, save metrics JSON + confusion-matrix PNG."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    y_pred = _labels(model, X_va)
    rep = classification_report(y_va, y_pred, labels=config.ATTACK_CLASSES,
                                output_dict=True, zero_division=0)
    macro = rep["macro avg"]["f1-score"]
    summary = {
        "model": name,
        "macro_f1": macro,
        "weighted_f1": rep["weighted avg"]["f1-score"],
        "accuracy": rep["accuracy"],
        "per_class": {c: rep[c] for c in config.ATTACK_CLASSES},
    }
    with open(f"{RESULTS_DIR}/{name}_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    cm = confusion_matrix(y_va, y_pred, labels=config.ATTACK_CLASSES, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 10))
    ConfusionMatrixDisplay(cm, display_labels=config.ATTACK_CLASSES).plot(
        ax=ax, cmap="Blues", xticks_rotation="vertical", values_format=".2f", colorbar=False)
    ax.set_title(f"{name} — row-normalised confusion (val macro-F1 {macro:.4f})")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/{name}_confusion.png", dpi=120)
    plt.close(fig)
    print(f"  [{name}] val macro-F1 = {macro:.4f}  (saved metrics + confusion matrix)")
    return macro


def prepare():
    """Load data; fit preprocessor on the real undersampled training pool."""
    t0 = time.time()
    Xtr_full, ytr_full = load_dataset(config.TRAIN_CSV)
    Xva, yva = load_dataset(config.VALIDATION_CSV, validate=False)
    keep = majority_undersample_index(ytr_full)
    Xtr, ytr = Xtr_full.loc[keep], ytr_full.loc[keep]
    del Xtr_full, ytr_full
    pre = QuantumPreprocessor()
    X8_real = pre.fit_transform(Xtr)        # real undersampled pool (no SMOTE)
    X8_va = pre.transform(Xva)
    print(f"prepared in {time.time()-t0:.0f}s | real pool {X8_real.shape} | val {X8_va.shape}")
    return X8_real, np.asarray(ytr), X8_va, np.asarray(yva)


def run_rf_full(X8_real, ytr, X8_va, yva) -> None:
    print("\n=== Exp 1: RandomForest on full ~400k (SMOTE) ===")
    strategy = dynamic_smote_strategy(ytr)
    Xr, yr = SMOTE(random_state=config.RANDOM_SEED, sampling_strategy=strategy).fit_resample(X8_real, ytr)
    rf = RandomForestModel(class_weight=None).fit(Xr, yr)
    evaluate("rf_full400k", rf, X8_va, yva)
    os.makedirs(MODELS_DIR, exist_ok=True)
    rf.save(f"{MODELS_DIR}/rf_full400k.joblib")


def _tune_svm(X, y) -> dict:
    gs = GridSearchCV(SVC(kernel="rbf", class_weight="balanced", random_state=config.RANDOM_SEED),
                      SVM_GRID, scoring="f1_macro", cv=3, n_jobs=-1)
    gs.fit(X, y)
    print(f"    best SVM params: {gs.best_params_}  (cv f1_macro {gs.best_score_:.4f})")
    return gs.best_params_


def _val_subsample(X_va, yva, n: int = 200_000):
    """Stratified validation subsample to keep slow SVM evaluation tractable."""
    if len(yva) <= n:
        return X_va, yva
    Xs, _, ys, _ = train_test_split(X_va, yva, train_size=n, stratify=yva,
                                    random_state=config.RANDOM_SEED)
    return Xs, ys


def run_subset(X8_real, ytr, X8_va, yva, size: int = 10_000) -> None:
    print(f"\n=== Exp 2+3: kernel subset @ {size} ===")
    X8_va, yva = _val_subsample(X8_va, yva)
    print(f"  (evaluating on stratified {len(yva):,}-row validation subsample)")
    # Exp 2: selection ablation (random vs kmeans), scored by SVM macro-F1
    best_method, best_f1 = None, -1.0
    for method in ("random", "kmeans"):
        idx = select_kernel_subset(X8_real, ytr, size, method=method)
        Xs, ys = X8_real[idx], ytr[idx]
        params = _tune_svm(Xs, ys)
        svm = SVMModel(**params).fit(Xs, ys)
        f1 = evaluate(f"svm_{method}{size//1000}k", svm, X8_va, yva)
        if f1 > best_f1:
            best_method, best_f1, best_idx, best_svm = method, f1, idx, svm
    print(f"  -> selection winner: {best_method} ({best_f1:.4f})")

    # Exp 3: RF comparator on the same winning subset; persist both + the subset
    os.makedirs(MODELS_DIR, exist_ok=True)
    Xs, ys = X8_real[best_idx], ytr[best_idx]
    rf = RandomForestModel(class_weight="balanced").fit(Xs, ys)
    evaluate(f"rf_{size//1000}k", rf, X8_va, yva)
    best_svm.save(f"{MODELS_DIR}/svm_{size//1000}k.joblib")
    rf.save(f"{MODELS_DIR}/rf_{size//1000}k.joblib")
    np.save(f"{MODELS_DIR}/kernel_subset_{size//1000}k_idx.npy", best_idx)


def run_spotcheck(X8_real, ytr, X8_va, yva) -> None:
    print("\n=== Exp 4: 20k spot-check (learning-curve slope) ===")
    X8_va, yva = _val_subsample(X8_va, yva)
    idx = select_kernel_subset(X8_real, ytr, 20_000, method="kmeans")
    Xs, ys = X8_real[idx], ytr[idx]
    params = _tune_svm(Xs, ys)
    SVMModel(**params).fit(Xs, ys)
    evaluate("svm_kmeans20k", SVMModel(**params).fit(Xs, ys), X8_va, yva)
    RandomForestModel(class_weight="balanced").fit(Xs, ys)
    evaluate("rf_20k", RandomForestModel(class_weight="balanced").fit(Xs, ys), X8_va, yva)


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "rf_full"
    data = prepare()
    if stage in ("rf_full", "all"):
        run_rf_full(*data)
    if stage in ("subset", "all"):
        run_subset(*data)
    if stage in ("spotcheck", "all"):
        run_spotcheck(*data)
    print("\nDone.  (test.csv untouched)")


if __name__ == "__main__":
    main()
