"""Phase 2 hierarchical (Option B): measure the Stage-1 router.

The cascade's payoff hinges on whether Stage 1 routes each sample to the right
coarse group. This trains the 4-group router (Benign / Volumetric / Network / Web)
on the smart-8 features and reports, on the full validation set:
  * per-group precision / recall / F1  (recall = routing accuracy for that group)
  * the 4x4 group confusion matrix      (where mis-routes actually go)

Critically, a sample mis-routed at Stage 1 can never be recovered by Stage 2, so
per-group recall here is an upper bound on what the cascade can achieve per group.
test.csv untouched. Run:
    ./venv/Scripts/python.exe -m experiments.phase2_stage1_router
"""

from __future__ import annotations

import os

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

from agent import agent_config as config
from data.preprocess import dynamic_smote_strategy
from experiments.phase2_baselines import prepare
from reasoning.classical import RandomForestModel


def to_group(parent_labels: np.ndarray) -> np.ndarray:
    return np.array([config.GROUP_OF[p] for p in parent_labels])


def main() -> None:
    X8_real, ytr, X8_va, yva = prepare()

    # Train router on the parent-level two-sided sample, mapped to groups.
    strat = dynamic_smote_strategy(ytr)
    Xr, yr_parent = SMOTE(random_state=config.RANDOM_SEED, sampling_strategy=strat).fit_resample(X8_real, ytr)
    gr, gva = to_group(yr_parent), to_group(yva)

    print("group composition (train, post-SMOTE):")
    for g in config.GROUP_NAMES:
        print(f"  {g:<11} {(gr==g).sum():>8}")

    router = RandomForestModel(class_weight="balanced").fit(Xr, gr)
    g_pred = router.classes_[router.predict_proba(X8_va).argmax(1)]

    print("\n=== Stage-1 router: per-group routing on full validation ===")
    rep = classification_report(gva, g_pred, labels=config.GROUP_NAMES,
                                output_dict=True, zero_division=0)
    print(f"{'group':<12}{'precision':>11}{'recall':>9}{'f1':>8}{'support':>11}")
    for g in config.GROUP_NAMES:
        v = rep[g]
        print(f"{g:<12}{v['precision']:>11.3f}{v['recall']:>9.3f}{v['f1-score']:>8.3f}{int(v['support']):>11}")
    print(f"\nrouter macro-F1 {rep['macro avg']['f1-score']:.4f} | accuracy {rep['accuracy']:.4f}")

    cm = confusion_matrix(gva, g_pred, labels=config.GROUP_NAMES, normalize="true")
    print("\n4x4 group confusion (row=true, normalised — where each group is routed):")
    print(f"{'':<12}" + "".join(f"{g:>12}" for g in config.GROUP_NAMES))
    for i, g in enumerate(config.GROUP_NAMES):
        print(f"{g:<12}" + "".join(f"{cm[i,j]:>12.3f}" for j in range(len(config.GROUP_NAMES))))

    os.makedirs("results/phase2", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=config.GROUP_NAMES).plot(
        ax=ax, cmap="Blues", values_format=".3f", colorbar=False)
    ax.set_title(f"Stage-1 router (4 groups) macro-F1 {rep['macro avg']['f1-score']:.3f}")
    fig.tight_layout()
    fig.savefig("results/phase2/stage1_router_confusion.png", dpi=120)
    plt.close(fig)
    print("\nsaved router confusion matrix -> results/phase2/stage1_router_confusion.png")
    print("(test.csv untouched)")


if __name__ == "__main__":
    main()
