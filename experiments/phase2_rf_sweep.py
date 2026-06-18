"""Phase 2 helper: pick the RandomForest config for the full-400k baseline.

Trains RF at several (max_depth, min_samples_leaf) settings on the same SMOTE'd
full training set, scores each on validation (and on real train, to show the
gap), and reports the best. Locks the baseline by evidence rather than a guess.

Run:  ./venv/Scripts/python.exe -m experiments.phase2_rf_sweep
"""

from __future__ import annotations

import time

from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score

from agent import agent_config as config
from data.preprocess import dynamic_smote_strategy
from experiments.phase2_baselines import prepare
from reasoning.classical import RandomForestModel

CONFIGS = [(14, 1), (14, 5), (20, 1), (20, 5), (None, 1), (None, 5)]


def mf1(model, X, y) -> float:
    return f1_score(y, model.classes_[model.predict_proba(X).argmax(1)],
                    average="macro", labels=config.ATTACK_CLASSES)


def main() -> None:
    X8_real, ytr, X8_va, yva = prepare()
    strategy = dynamic_smote_strategy(ytr)
    Xr, yr = SMOTE(random_state=config.RANDOM_SEED, sampling_strategy=strategy).fit_resample(X8_real, ytr)

    print(f"\n{'max_depth':>10} {'leaf':>5} {'train':>8} {'val':>8} {'gap':>8}")
    results = []
    for depth, leaf in CONFIGS:
        t = time.time()
        rf = RandomForestModel(max_depth=depth, min_samples_leaf=leaf, class_weight=None).fit(Xr, yr)
        f_tr, f_va = mf1(rf, X8_real, ytr), mf1(rf, X8_va, yva)
        results.append((depth, leaf, f_va))
        print(f"{str(depth):>10} {leaf:>5} {f_tr:>8.4f} {f_va:>8.4f} {f_tr-f_va:>+8.4f}   ({time.time()-t:.0f}s)")

    best = max(results, key=lambda r: r[2])
    print(f"\nBest by validation macro-F1: max_depth={best[0]}, min_samples_leaf={best[1]}  ->  {best[2]:.4f}")


if __name__ == "__main__":
    main()
