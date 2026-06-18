"""Regenerate the tuned SVM-10k per-class breakdown (winning config).

Reproduces the exact report slice from phase2_svm_tune (same split params) and
the winning config (k-means subset, gamma=1, C=10, class_weight=None), then saves
per-class metrics + confusion matrix. Shows exactly where the 0.3525 is lost.

Run:  ./venv/Scripts/python.exe -m experiments.phase2_svm_perclass
"""

from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from agent import agent_config as config
from data.sampling import select_kernel_subset
from experiments.phase2_baselines import prepare


def main() -> None:
    X8_real, ytr, X8_va, yva = prepare()
    _, X_rep, _, y_rep = train_test_split(
        X8_va, yva, train_size=8_000, test_size=200_000,
        stratify=yva, random_state=config.RANDOM_SEED)
    idx = select_kernel_subset(X8_real, ytr, 10_000, method="kmeans")

    t = time.time()
    svc = SVC(kernel="rbf", gamma=1, C=10, class_weight=None,
              cache_size=1000, max_iter=20_000, random_state=config.RANDOM_SEED)
    svc.fit(X8_real[idx], ytr[idx])
    yp = svc.predict(X_rep)
    print(f"fit+predict in {time.time()-t:.0f}s | nSV={svc.n_support_.sum()}")

    rep = classification_report(y_rep, yp, labels=config.ATTACK_CLASSES,
                                output_dict=True, zero_division=0)
    print(f"\nmacro-F1 {rep['macro avg']['f1-score']:.4f} | "
          f"weighted-F1 {rep['weighted avg']['f1-score']:.4f} | acc {rep['accuracy']:.4f}")
    print(f"{'class':<22}{'prec':>7}{'rec':>7}{'f1':>7}{'support':>10}")
    for c in config.ATTACK_CLASSES:
        v = rep[c]
        print(f"{c:<22}{v['precision']:>7.2f}{v['recall']:>7.2f}{v['f1-score']:>7.2f}{int(v['support']):>10}")

    os.makedirs("results/phase2", exist_ok=True)
    json.dump({"model": "svm_10k_tuned", "config": {"gamma": 1, "C": 10, "class_weight": None},
               "macro_f1": rep["macro avg"]["f1-score"],
               "weighted_f1": rep["weighted avg"]["f1-score"], "accuracy": rep["accuracy"],
               "per_class": {c: rep[c] for c in config.ATTACK_CLASSES}},
              open("results/phase2/svm_10k_tuned_metrics.json", "w"), indent=2)
    cm = confusion_matrix(y_rep, yp, labels=config.ATTACK_CLASSES, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 10))
    ConfusionMatrixDisplay(cm, display_labels=config.ATTACK_CLASSES).plot(
        ax=ax, cmap="Blues", xticks_rotation="vertical", values_format=".2f", colorbar=False)
    ax.set_title(f"svm_10k_tuned (gamma=1, C=10) macro-F1 {rep['macro avg']['f1-score']:.4f}")
    fig.tight_layout()
    fig.savefig("results/phase2/svm_10k_tuned_confusion.png", dpi=120)
    plt.close(fig)
    print("saved metrics + confusion matrix")


if __name__ == "__main__":
    main()
