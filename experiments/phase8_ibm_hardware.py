"""Phase 8 — Experiment E8b: IBM real-hardware quantum validation.

Compares PegasosQSVC performance on AerSimulator (noiseless) vs a real IBM
quantum backend. Both models are trained on the SAME tiny balanced subset
(IBM_HARDWARE_N_TRAIN samples = 10 benign + 10 attack) and evaluated on the
SAME test set (IBM_HARDWARE_N_TEST samples = 5 benign + 5 attack).

The IBM run requires an IBM Quantum account and API token:
  - Set env var IBM_TOKEN=<your_token> before running, OR
  - Pass --token <token> as a CLI argument.
  - If neither is provided, the experiment runs in AER-ONLY mode and prints
    the IBM setup instructions.

Why tiny n:
  Kernel circuits = n_test * n_train = 10 * 20 = 200 circuits per evaluation.
  On IBM free tier this is manageable (1-2 jobs, ~10-20 min queue).
  Statistical validity is limited at n=20 training / n=10 test — but the goal
  is to confirm the quantum circuit runs on real hardware and measure any
  noise-induced performance gap, not to match Phase 6 AUROC numbers.

Run:  ./venv/Scripts/python.exe -m experiments.phase8_ibm_hardware
      ./venv/Scripts/python.exe -m experiments.phase8_ibm_hardware --token TOKEN
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from agent import agent_config as config
from data.nf_loader import NFPreprocessor, read_nf

RESULTS_DIR = "results/phase8"

# ---------------------------------------------------------------------------
# AerSimulator-based QSVC (standard training path)
# ---------------------------------------------------------------------------

def _train_and_eval_aer(X_tr, y_tr, X_te, y_te, verbose=True):
    """Train + eval QSVC using AerSimulator (our standard path)."""
    from reasoning.quantum import PegasosQSVCModel
    if verbose:
        print("\n  [AerSimulator] Training PegasosQSVC ...")
    t0 = time.time()
    model = PegasosQSVCModel(num_steps=config.PEGASOS_TAU).fit(X_tr, y_tr)
    train_time = time.time() - t0
    if verbose:
        print(f"  [AerSimulator] fit in {train_time:.1f}s")

    t0 = time.time()
    proba = model.predict_proba(X_te)
    predict_time = time.time() - t0
    auroc = float(roc_auc_score(y_te, proba[:, 1]))
    if verbose:
        print(f"  [AerSimulator] predict in {predict_time:.1f}s  AUROC={auroc:.4f}")
    return {"backend": "AerSimulator", "auroc": round(auroc, 4),
            "train_time_s": round(train_time, 1),
            "predict_time_s": round(predict_time, 1),
            "n_train": len(y_tr), "n_test": len(y_te)}


# ---------------------------------------------------------------------------
# IBM real-hardware path
# ---------------------------------------------------------------------------

def _build_ibm_qsvc(token: str, n_qubits: int = 8):
    """Build a PegasosQSVC that uses IBM Quantum runtime for kernel evaluation."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSamplerV2
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
        from qiskit_machine_learning.algorithms import PegasosQSVC
        from perception.feature_map import CyberSecurityFeatureMap
    except ImportError as e:
        raise RuntimeError(f"IBM runtime import failed: {e}") from e

    print("\n  [IBM QPU] Connecting to IBM Quantum service ...")
    # channel="ibm_quantum" was retired in IBM's 2026-03 Open Plan migration
    # (see data/CONFIG_PROVENANCE.md); ibm_quantum_platform is the current
    # channel, and instance= now takes the account's real CRN, not the legacy
    # hub/group/project string.
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=config.IBM_INSTANCE,
    )
    # least_busy() no longer takes operational=/simulator= kwargs directly in
    # qiskit-ibm-runtime 0.47 -- IBMBackend has no .operational/.simulator
    # attributes any more. Use the documented filters= callable instead.
    def _is_real_and_operational(b) -> bool:
        try:
            return bool(b.status().operational) and not bool(b.configuration().simulator)
        except Exception:
            return False

    backend = service.least_busy(min_num_qubits=n_qubits, filters=_is_real_and_operational)
    print(f"  [IBM QPU] Selected backend: {backend.name}")

    # FidelityQuantumKernel takes fidelity=, not sampler= (verified against
    # the installed qiskit-machine-learning signature -- passing sampler=
    # directly raises a TypeError on construction, before any API call).
    # IBMSamplerV2 also needs options={"default_shots": ...} to actually
    # apply config.IBM_SHOTS; its constructor has no shots= argument.
    sampler = IBMSamplerV2(backend, options={"default_shots": config.IBM_SHOTS})
    fidelity = ComputeUncompute(sampler=sampler)
    feature_map = CyberSecurityFeatureMap(n_qubits=n_qubits)
    kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    model = PegasosQSVC(quantum_kernel=kernel, C=1.0, num_steps=config.PEGASOS_TAU)
    return model, backend.name


def _train_and_eval_ibm(X_tr, y_tr, X_te, y_te, token: str, verbose=True):
    """Train + eval QSVC using real IBM QPU backend."""
    try:
        model, backend_name = _build_ibm_qsvc(token, n_qubits=len(config.NF_FEATURE_NAMES))
    except Exception as e:
        return {"backend": "IBM_QPU", "error": str(e), "auroc": None,
                "train_time_s": None, "predict_time_s": None,
                "n_train": len(y_tr), "n_test": len(y_te)}

    if verbose:
        print(f"  [IBM QPU] Training on {backend_name} "
              f"(n_train={len(y_tr)}, n_steps={config.PEGASOS_TAU}) ...")
        print(f"  [IBM QPU] Kernel circuits at training: {len(y_tr)**2}")
        print(f"  [IBM QPU] Kernel circuits at predict : {len(y_te) * len(y_tr)}")
        print("  [IBM QPU] Submitting jobs — queue times may vary ...")

    t0 = time.time()
    try:
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0
        if verbose:
            print(f"  [IBM QPU] fit done in {train_time:.1f}s")
    except Exception as e:
        return {"backend": f"IBM_QPU/{backend_name}", "error": f"fit failed: {e}",
                "auroc": None, "train_time_s": None, "predict_time_s": None,
                "n_train": len(y_tr), "n_test": len(y_te)}

    t0 = time.time()
    try:
        # PegasosQSVC.predict returns hard labels; use decision_function for AUROC
        scores = model.decision_function(X_te)
        predict_time = time.time() - t0
        auroc = float(roc_auc_score(y_te, scores))
        if verbose:
            print(f"  [IBM QPU] predict in {predict_time:.1f}s  AUROC={auroc:.4f}")
        return {"backend": f"IBM_QPU/{backend_name}", "auroc": round(auroc, 4),
                "train_time_s": round(train_time, 1),
                "predict_time_s": round(predict_time, 1),
                "n_train": len(y_tr), "n_test": len(y_te)}
    except Exception as e:
        return {"backend": f"IBM_QPU/{backend_name}", "error": f"predict failed: {e}",
                "auroc": None, "train_time_s": time.time() - t0, "predict_time_s": None,
                "n_train": len(y_tr), "n_test": len(y_te)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("IBM_TOKEN", ""),
                        help="IBM Quantum API token (or set env var IBM_TOKEN)")
    parser.add_argument("--aer-only", action="store_true",
                        help="Force AerSimulator-only mode (skip IBM QPU)")
    args = parser.parse_args()

    ibm_token = args.token.strip()
    run_ibm   = bool(ibm_token) and not args.aer_only

    print("=" * 70)
    print("Phase 8 E8b: IBM real-hardware quantum validation")
    print(f"  Mode: {'AerSimulator + IBM QPU' if run_ibm else 'AerSimulator ONLY'}")
    if not run_ibm:
        print("\n  To run on real IBM hardware, set IBM_TOKEN env var or pass --token.")
        print("  Get a token at: https://quantum.ibm.com  (free account)")
    print("=" * 70)

    # ── Load a small slice of NF-ToN-IoT ────────────────────────────────────
    print(f"\nLoading {config.NF_TON_CSV} ...")
    t0 = time.time()
    X_df, y_bin, _ = read_nf(config.NF_TON_CSV)
    print(f"  {len(y_bin):,} rows  ({time.time()-t0:.1f}s)")

    pre = NFPreprocessor()
    X8  = pre.fit_transform(X_df)

    # Balanced tiny subset: n_train + n_test per class
    N_TR = config.IBM_HARDWARE_N_TRAIN    # 20
    N_TE = config.IBM_HARDWARE_N_TEST     # 10
    rng  = np.random.default_rng(config.RANDOM_SEED + 800)

    b_idx = np.where(y_bin == 0)[0]
    a_idx = np.where(y_bin == 1)[0]
    chosen_b = rng.choice(b_idx, N_TR // 2 + N_TE // 2, replace=False)
    chosen_a = rng.choice(a_idx, N_TR // 2 + N_TE // 2, replace=False)

    X_b, y_b = X8[chosen_b], y_bin[chosen_b]
    X_a, y_a = X8[chosen_a], y_bin[chosen_a]

    # First N_TR//2 per class = train; remaining = test
    X_tr = np.vstack([X_b[:N_TR // 2], X_a[:N_TR // 2]])
    y_tr = np.concatenate([y_b[:N_TR // 2], y_a[:N_TR // 2]])
    X_te = np.vstack([X_b[N_TR // 2:], X_a[N_TR // 2:]])
    y_te = np.concatenate([y_b[N_TR // 2:], y_a[N_TR // 2:]])

    print(f"\n  Train: {X_tr.shape}  labels {np.bincount(y_tr)}")
    print(f"  Test : {X_te.shape}  labels {np.bincount(y_te)}")
    print(f"  Kernel circuits (predict): {len(y_te)} x {len(y_tr)} = {len(y_te)*len(y_tr)}")

    results = []

    # ── AerSimulator run ─────────────────────────────────────────────────────
    print("\n--- AerSimulator run ---")
    aer_r = _train_and_eval_aer(X_tr, y_tr, X_te, y_te)
    results.append(aer_r)

    # ── IBM QPU run (optional) ────────────────────────────────────────────────
    if run_ibm:
        print("\n--- IBM QPU run ---")
        ibm_r = _train_and_eval_ibm(X_tr, y_tr, X_te, y_te, ibm_token)
        results.append(ibm_r)

        if ibm_r.get("auroc") is not None and aer_r.get("auroc") is not None:
            gap = ibm_r["auroc"] - aer_r["auroc"]
            print(f"\n  Hardware noise gap: {gap:+.4f} AUROC "
                  f"(IBM={ibm_r['auroc']:.4f} vs Aer={aer_r['auroc']:.4f})")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "experiment":  "phase8_ibm_hardware",
        "n_qubits":    len(config.NF_FEATURE_NAMES),
        "n_train":     N_TR,
        "n_test":      N_TE,
        "pegasos_tau": config.PEGASOS_TAU,
        "ibm_shots":   config.IBM_SHOTS,
        "results":     results,
    }
    path = f"{RESULTS_DIR}/phase8_ibm_hardware_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved -> {path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Backend':<35}  {'AUROC':>7}  {'Train (s)':>10}  {'Predict (s)':>12}")
    print("  " + "-" * 68)
    for r in results:
        auroc_str = f"{r['auroc']:.4f}" if r.get("auroc") is not None else "N/A"
        tr_str    = f"{r['train_time_s']:.1f}"  if r.get("train_time_s") else "N/A"
        pr_str    = f"{r['predict_time_s']:.1f}" if r.get("predict_time_s") else "N/A"
        err       = f"  [ERROR: {r['error']}]" if r.get("error") else ""
        print(f"  {r['backend']:<35}  {auroc_str:>7}  {tr_str:>10}  {pr_str:>12}{err}")


if __name__ == "__main__":
    main()
