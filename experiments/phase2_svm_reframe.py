"""Phase 2 follow-up: does the SVM-RBF kernel succeed on REDUCED-class tasks?

The 15-class kernel is weak (~0.35). This tests whether reframing to fewer classes
rescues it — the regime kernels are designed for, and what QSVC will run on:
  * Binary: BenignTraffic vs Attack (everything else).
  * Coarse: 4 groups (Benign / Volumetric / Network / Web).

Same 10k kmeans subset pipeline (balanced for each task's labels). Small grid tuned
on an 8k validation slice, reported on a disjoint 200k slice. test.csv untouched.
Run:  ./venv/Scripts/python.exe -u -m experiments.phase2_svm_reframe
"""

from __future__ import annotations

import itertools
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from agent import agent_config as config
from data.sampling import select_kernel_subset
from experiments.phase2_baselines import prepare

GAMMAS = ["scale", 1]
CS = [1, 10]
CWS = [None, "balanced"]
SVC_KW = dict(kernel="rbf", cache_size=1000, max_iter=50_000, random_state=config.RANDOM_SEED)


def to_binary(y):
    return np.where(np.asarray(y) == "BenignTraffic", "Benign", "Attack")


def to_group(y):
    return np.array([config.GROUP_OF[c] for c in np.asarray(y)])


def run_task(name, relabel, label_order, X8_real, ytr, Xtune, ytune15, Xrep, yrep15):
    ysub = relabel(ytr)
    idx = select_kernel_subset(X8_real, ysub, 10_000, method="kmeans")
    Xs, ys = X8_real[idx], ysub[idx]
    ytune, yrep = relabel(ytune15), relabel(yrep15)

    best = (-1.0, None, None)
    for g, C, cw in itertools.product(GAMMAS, CS, CWS):
        svc = SVC(gamma=g, C=C, class_weight=cw, **SVC_KW).fit(Xs, ys)
        f = f1_score(ytune, svc.predict(Xtune), average="macro", labels=label_order)
        if f > best[0]:
            best = (f, dict(gamma=g, C=C, class_weight=cw), svc)

    _, params, model = best
    yp = model.predict(Xrep)
    macro = f1_score(yrep, yp, average="macro", labels=label_order)
    print(f"\n=== {name}: best macro-F1 {macro:.4f}  cfg={params} ===")
    print(classification_report(yrep, yp, labels=label_order, digits=3, zero_division=0))
    cm = confusion_matrix(yrep, yp, labels=label_order)
    print("confusion (rows=true):")
    print("        " + "".join(f"{l[:10]:>12}" for l in label_order))
    for i, l in enumerate(label_order):
        print(f"{l[:8]:<8}" + "".join(f"{cm[i,j]:>12d}" for j in range(len(label_order))))

    if name == "BINARY":
        ai, bi = label_order.index("Attack"), label_order.index("Benign")
        det = cm[ai, ai] / cm[ai].sum()                       # detection rate (attack recall)
        fa = cm[bi, ai] / cm[bi].sum()                        # false-alarm rate
        print(f"  detection rate (attack recall): {det:.4f}")
        print(f"  false-alarm rate (benign->attack): {fa:.4f}")
    return macro


def main() -> None:
    t0 = time.time()
    X8_real, ytr, X8_va, yva = prepare()
    Xtmp, Xrep, ytmp, yrep15 = train_test_split(X8_va, yva, test_size=200_000,
                                                stratify=yva, random_state=config.RANDOM_SEED)
    Xtune, _, ytune15, _ = train_test_split(Xtmp, ytmp, train_size=8_000,
                                            stratify=ytmp, random_state=config.RANDOM_SEED)
    ytr = np.asarray(ytr)
    print(f"setup done in {time.time()-t0:.0f}s | tune {len(ytune15)} report {len(yrep15)}")

    mb = run_task("BINARY", to_binary, ["Attack", "Benign"],
                  X8_real, ytr, Xtune, np.asarray(ytune15), Xrep, np.asarray(yrep15))
    mc = run_task("COARSE", to_group, config.GROUP_NAMES,
                  X8_real, ytr, Xtune, np.asarray(ytune15), Xrep, np.asarray(yrep15))

    print("\n" + "=" * 60)
    print("SUMMARY (SVM-RBF-10k on reduced tasks; 15-class was 0.3525)")
    print(f"  binary (attack/benign) macro-F1 : {mb:.4f}")
    print(f"  coarse (4 groups)      macro-F1 : {mc:.4f}")
    print(f"\nTotal {time.time()-t0:.0f}s  (test.csv untouched)")


if __name__ == "__main__":
    main()
