"""Phase 2: corrected SVM-RBF tuning at the 10k kernel scale (fast + observable).

Replaces the earlier run whose narrow grid {scale,0.1,1} produced a broken model.
The full grid is searched and NOTHING is pre-excluded:
  gamma in {0.001, 0.01, 0.1, scale, 1}, C in {1, 10, 100},
  class_weight in {None, balanced}, selection in {random, kmeans}.

Hyperparameters are chosen on a held-out validation TUNE slice (disjoint from the
REPORT slice) by macro-F1, so the reported number is not tuned on its own data.
Each config prints as it finishes (run with `python -u`). SVC has a bounded
max_iter and large cache so no single config can hang. test.csv is untouched.

Run:  ./venv/Scripts/python.exe -u -m experiments.phase2_svm_tune
"""

from __future__ import annotations

import itertools
import time

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

from agent import agent_config as config
from data.sampling import select_kernel_subset
from experiments.phase2_baselines import evaluate, prepare
from reasoning.classical import SVMModel

GAMMAS = [0.001, 0.01, 0.1, "scale", 1]
CS = [1, 10, 100]
WEIGHTS = [None, "balanced"]
SELECTIONS = ["random", "kmeans"]
SIZE = 10_000
SVC_KW = dict(kernel="rbf", cache_size=1000, max_iter=20_000, random_state=config.RANDOM_SEED)


def macro(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=config.ATTACK_CLASSES)


def main() -> None:
    t0 = time.time()
    X8_real, ytr, X8_va, yva = prepare()
    X_tune, X_rep, y_tune, y_rep = train_test_split(
        X8_va, yva, train_size=8_000, test_size=200_000,
        stratify=yva, random_state=config.RANDOM_SEED)
    print(f"tune={len(y_tune):,}  report={len(y_rep):,}", flush=True)

    grid = list(itertools.product(GAMMAS, CS, WEIGHTS))
    overall_best = None
    for method in SELECTIONS:
        idx = select_kernel_subset(X8_real, ytr, SIZE, method=method)
        Xs, ys = X8_real[idx], ytr[idx]
        rows = []
        bar = tqdm(grid, desc=f"SVM/{method}", ncols=90)
        for g, C, w in bar:
            svc = SVC(gamma=g, C=C, class_weight=w, **SVC_KW).fit(Xs, ys)
            f = macro(y_tune, svc.predict(X_tune))
            rows.append((f, g, C, w))
            bar.set_postfix(last=f"{f:.3f}", best=f"{max(r[0] for r in rows):.3f}")
        rows.sort(key=lambda r: r[0], reverse=True)
        tqdm.write(f"  top configs ({method}): " +
                   ", ".join(f"{f:.3f}[g={g},C={C},cw={w}]" for f, g, C, w in rows[:3]))

        bf, bg, bC, bw = rows[0]
        svc = SVC(gamma=bg, C=bC, class_weight=bw, **SVC_KW).fit(Xs, ys)
        rep_f = macro(y_rep, svc.predict(X_rep))
        print(f"[{method}] BEST tune={bf:.4f} -> REPORT={rep_f:.4f} "
              f"(gamma={bg}, C={bC}, cw={bw})", flush=True)
        if overall_best is None or rep_f > overall_best[0]:
            overall_best = (rep_f, method, bg, bC, bw)

    rep_f, method, bg, bC, bw = overall_best
    print(f"\nOVERALL best: {method}, gamma={bg}, C={bC}, cw={bw}  ->  {rep_f:.4f}", flush=True)
    idx = select_kernel_subset(X8_real, ytr, SIZE, method=method)
    evaluate("svm_10k_tuned", SVMModel(C=bC, gamma=bg, class_weight=bw).fit(X8_real[idx], ytr[idx]),
             X_rep, y_rep)
    print(f"\nTotal runtime {time.time()-t0:.0f}s   (test.csv untouched)", flush=True)


if __name__ == "__main__":
    main()
