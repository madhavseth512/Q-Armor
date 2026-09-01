"""Phase 15 — Confirmatory Protocol: finite-shot sweep S in {128,...,4096}.

Closes the "finite-shot settings" item from docs/paper.tex's Confirmatory
Protocol. Every other AerSimulator result in this project uses
FidelityQuantumKernel's DEFAULT fidelity object, which is EXACT (statevector,
zero shot noise) -- verified directly before writing this script: calling
.evaluate() twice on identical inputs returns bit-identical kernel values.
That is legitimate and matches the paper's "AerSimulator (ideal, noiseless)"
framing for every other result, but it means no result anywhere in this
project has yet exercised genuine finite-shot sampling noise -- including
the number currently labelled the "finite-shot pilot" (AUROC=0.800,
experiments/phase8_ibm_hardware.py's AerSimulator row), which is ALSO the
exact/default kernel, not a shot-sampled one. This script is the first one
that actually is finite-shot.

Wires an explicit qiskit_aer.primitives.SamplerV2(default_shots=S) into a
ComputeUncompute fidelity object (see reasoning/quantum.py's
PegasosQSVCModel for why the default wrapper can't do this: it hard-codes
FidelityQuantumKernel(feature_map=fm) with no fidelity/sampler override).

Uses the SAME tiny n_train=20/n_test=10 configuration as
experiments/phase8_ibm_hardware.py (already proven feasible and the
convention this project uses for hardware-adjacent quantum pilots), for
S in {128, 256, 512, 1024, 4096}, plus the exact/default kernel as the
S=infinity reference point.

Run:  ./venv/Scripts/python.exe -m experiments.phase15_finiteshot_sweep
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from sklearn.metrics import roc_auc_score

from agent import agent_config as config
from data.nf_loader import NFPreprocessor, read_nf
from perception.feature_map import CyberSecurityFeatureMap

RESULTS_DIR = "results/phase15"
SHOT_SETTINGS = [128, 256, 512, 1024, 4096]
N_TRAIN = config.IBM_HARDWARE_N_TRAIN  # 20
N_TEST = config.IBM_HARDWARE_N_TEST    # 10


def _build_kernel(shots: int | None, seed: int):
    fm = CyberSecurityFeatureMap()
    if shots is None:
        return FidelityQuantumKernel(feature_map=fm)  # exact/default, S=infinity reference
    sampler = SamplerV2(default_shots=shots, seed=seed)
    fidelity = ComputeUncompute(sampler=sampler)
    return FidelityQuantumKernel(feature_map=fm, fidelity=fidelity)


def _train_and_eval(X_tr, y_tr, X_te, y_te, shots, seed) -> dict:
    kernel = _build_kernel(shots, seed)
    model = PegasosQSVC(quantum_kernel=kernel, C=1.0, num_steps=config.PEGASOS_TAU)

    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    t0 = time.time()
    scores = model.decision_function(X_te)
    predict_time = time.time() - t0
    # PegasosQSVC's raw decision_function is oriented benign-positive (see the
    # score-audit tests in tests/test_quantum.py) -- flip for an attack-scored AUROC.
    auroc = float(roc_auc_score(y_te, -scores))

    return {
        "shots": shots if shots is not None else "exact",
        "auroc": round(auroc, 4),
        "train_time_s": round(train_time, 1),
        "predict_time_s": round(predict_time, 1),
    }


def main() -> None:
    print("=" * 70)
    print(f"Phase 15: finite-shot sweep S={SHOT_SETTINGS}  n_train={N_TRAIN}  n_test={N_TEST}")
    print("=" * 70)

    X_df, y_bin, _ = read_nf(config.NF_TON_CSV)
    pre = NFPreprocessor()
    X8 = pre.fit_transform(X_df)

    rng = np.random.default_rng(config.RANDOM_SEED + 800)  # same seed as phase8_ibm_hardware
    b_idx = np.where(y_bin == 0)[0]
    a_idx = np.where(y_bin == 1)[0]
    chosen_b = rng.choice(b_idx, N_TRAIN // 2 + N_TEST // 2, replace=False)
    chosen_a = rng.choice(a_idx, N_TRAIN // 2 + N_TEST // 2, replace=False)
    X_b, y_b = X8[chosen_b], y_bin[chosen_b]
    X_a, y_a = X8[chosen_a], y_bin[chosen_a]
    X_tr = np.vstack([X_b[:N_TRAIN // 2], X_a[:N_TRAIN // 2]])
    y_tr = np.concatenate([y_b[:N_TRAIN // 2], y_a[:N_TRAIN // 2]])
    X_te = np.vstack([X_b[N_TRAIN // 2:], X_a[N_TRAIN // 2:]])
    y_te = np.concatenate([y_b[N_TRAIN // 2:], y_a[N_TRAIN // 2:]])
    print(f"Train {X_tr.shape}  labels {np.bincount(y_tr)}   "
          f"Test {X_te.shape}  labels {np.bincount(y_te)}")

    results = []

    print("\n--- Exact / default kernel (S=infinity reference) ---")
    r_exact = _train_and_eval(X_tr, y_tr, X_te, y_te, shots=None, seed=config.RANDOM_SEED)
    print(f"  AUROC={r_exact['auroc']:.4f}  train={r_exact['train_time_s']}s  "
          f"predict={r_exact['predict_time_s']}s")
    results.append(r_exact)

    for S in SHOT_SETTINGS:
        print(f"\n--- shots={S} ---")
        r = _train_and_eval(X_tr, y_tr, X_te, y_te, shots=S, seed=config.RANDOM_SEED)
        print(f"  AUROC={r['auroc']:.4f}  train={r['train_time_s']}s  "
              f"predict={r['predict_time_s']}s")
        results.append(r)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "experiment": "phase15_finiteshot_sweep",
        "n_train": N_TRAIN, "n_test": N_TEST,
        "shot_settings": SHOT_SETTINGS,
        "results": results,
    }
    path = f"{RESULTS_DIR}/phase15_finiteshot_sweep_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
