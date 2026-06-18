"""Phase 2: measure the full Option-B cascade end-to-end vs flat RF.

Trains the HierarchicalClassifier (4-group router + Volumetric/Network/Web
specialists) and evaluates the final 15-class macro-F1 on the full validation set,
head-to-head with the flat RandomForest baseline (0.7376). test.csv untouched.

Run:  ./venv/Scripts/python.exe -m experiments.phase2_hierarchical
"""

from __future__ import annotations

import json
import os
import time

from sklearn.metrics import classification_report

from agent import agent_config as config
from experiments.phase2_baselines import evaluate, prepare
from reasoning.hierarchical import HierarchicalClassifier


def main() -> None:
    t0 = time.time()
    X8_real, ytr, X8_va, yva = prepare()

    hc = HierarchicalClassifier(web_reject=True).fit(X8_real, ytr)
    print(f"hierarchical (web_reject) fitted in {time.time()-t0:.0f}s")

    macro = evaluate("hierarchical_reject", hc, X8_va, yva)

    # per-class detail + head-to-head
    flat = json.load(open("results/phase2/rf_full400k_metrics.json"))["macro_f1"]
    y_pred = hc.classes_[hc.predict_proba(X8_va).argmax(1)]
    rep = classification_report(yva, y_pred, labels=config.ATTACK_CLASSES,
                                output_dict=True, zero_division=0)
    print(f"\n{'class':<22}{'prec':>7}{'rec':>7}{'f1':>7}")
    for c in config.ATTACK_CLASSES:
        v = rep[c]
        print(f"{c:<22}{v['precision']:>7.2f}{v['recall']:>7.2f}{v['f1-score']:>7.2f}")

    print(f"\n=== head-to-head (full validation macro-F1) ===")
    print(f"  flat RandomForest        : {flat:.4f}")
    print(f"  hierarchical no-reject    : 0.6180")
    print(f"  hierarchical + web_reject : {macro:.4f}")
    print(f"  delta vs flat             : {macro - flat:+.4f}")
    print(f"  delta vs no-reject        : {macro - 0.6180:+.4f}")

    os.makedirs("models", exist_ok=True)
    hc.save("models/hierarchical_reject.joblib")
    print(f"\nTotal runtime {time.time()-t0:.0f}s   (test.csv untouched)")


if __name__ == "__main__":
    main()
