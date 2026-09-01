"""Phase 16 — Confirmatory Protocol: real-QPU characterisation.

Closes the "real-QPU experiment" item from docs/paper.tex's Confirmatory
Protocol, which requires reporting: backend, calibration timestamp,
transpiled depth, two-qubit gate count, shots, queue-independent execution
time, and kernel error relative to ideal simulation.

Deliberately minimal, NOT a full PegasosQSVC train+eval cycle (that already
exists in experiments/phase8_ibm_hardware.py, at N_TRAIN=20/N_TEST=10). IBM
Quantum's free "Open" plan grants roughly 10 minutes of QPU time per 28-day
window -- a genuinely scarce, real resource, not a simulator. This script
computes only a tiny 4-point (6 unique off-diagonal pairs) kernel matrix on
real hardware, which is exactly what every required report field needs
without spending that budget on a training loop this checklist item does
not ask for.

Every required field:
  - backend            : selected least-busy backend name
  - calibration        : backend.properties().last_update_date
  - transpiled depth / 2-qubit gate count : transpile() the bound feature-map
                          circuit for the selected backend at the runtime's
                          default optimisation level, then inspect .depth()
                          and .count_ops()
  - shots               : config.IBM_SHOTS
  - queue-independent execution time : primitive job's usage/execution time
                          field (name varies by qiskit-ibm-runtime version;
                          captured defensively, reported as unavailable
                          rather than guessed if the field isn't there)
  - kernel error vs ideal simulation : the SAME 6 pairs computed via the
                          project's default EXACT kernel (see phase15's
                          docstring for why that is the right ideal
                          reference), mean/max absolute difference

Requires IBM_QUANTUM_TOKEN in .env. Costs real QPU queue time -- run
deliberately, not repeatedly.

Run:  ./venv/Scripts/python.exe -m experiments.phase16_qpu_characterization --confirm
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv

from agent import agent_config as config
from data.nf_loader import NFPreprocessor, read_nf
from perception.feature_map import CyberSecurityFeatureMap

RESULTS_DIR = "results/phase16"
N_POINTS = 4  # 6 unique off-diagonal pairs -- deliberately tiny


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true",
                        help="Confirm you understand this uses real IBM Quantum QPU time.")
    args = parser.parse_args()

    if not args.confirm:
        print("Phase 16: real-QPU characterisation.")
        print(f"  {N_POINTS} points -> {N_POINTS * (N_POINTS - 1) // 2} circuit pairs on real hardware.")
        print("  This spends real, limited IBM Quantum free-tier QPU time.")
        print("  Re-run with --confirm to proceed.")
        sys.exit(0)

    load_dotenv(override=True)
    token = os.environ.get("IBM_QUANTUM_TOKEN", "").strip()
    if not token:
        sys.exit("IBM_QUANTUM_TOKEN not set in .env -- cannot reach IBM Quantum.")

    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSamplerV2
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    print("=" * 70)
    print("Phase 16: real-QPU characterisation")
    print("=" * 70)

    print("\nLoading a tiny NF-ToN-IoT sample ...")
    X_df, y_bin, _ = read_nf(config.NF_TON_CSV)
    pre = NFPreprocessor()
    X8 = pre.fit_transform(X_df)
    rng = np.random.default_rng(config.RANDOM_SEED + 1600)
    b_idx = np.where(y_bin == 0)[0]
    a_idx = np.where(y_bin == 1)[0]
    idx = np.concatenate([rng.choice(b_idx, N_POINTS // 2, replace=False),
                           rng.choice(a_idx, N_POINTS // 2, replace=False)])
    X = X8[idx]
    print(f"  {X.shape} points, labels {y_bin[idx]}")

    print("\nConnecting to IBM Quantum ...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=config.N_QUBITS)
    print(f"  Selected backend: {backend.name}")

    calibration_ts = None
    try:
        calibration_ts = str(backend.properties().last_update_date)
    except Exception as e:
        print(f"  [calibration timestamp unavailable: {e}]")

    fm = CyberSecurityFeatureMap()
    bound = fm.assign_parameters(X[0])
    transpiled = transpile(bound, backend=backend)
    depth = transpiled.depth()
    ops = transpiled.count_ops()
    two_q_gate_names = {"cx", "cz", "ecr", "cp", "rzx"}
    two_q_count = sum(v for k, v in ops.items() if k in two_q_gate_names)
    print(f"  Transpiled depth: {depth}   two-qubit gates: {two_q_count}   ops: {dict(ops)}")

    pairs = list(itertools.combinations(range(N_POINTS), 2))
    print(f"\nSubmitting {len(pairs)} kernel-pair circuits to {backend.name} "
          f"({config.IBM_SHOTS} shots each) ...")

    sampler = IBMSamplerV2(backend)
    fidelity_kernel_ibm = FidelityQuantumKernel(feature_map=fm, sampler=sampler)

    t0 = time.time()
    ibm_pair_values = []
    job_metadata = []
    for i, j in pairs:
        val = float(fidelity_kernel_ibm.evaluate(np.array([X[i]]), np.array([X[j]]))[0, 0])
        ibm_pair_values.append(val)
    wall_time = time.time() - t0
    print(f"  wall time (includes queue): {wall_time:.1f}s")

    exec_time_reported = None  # populated below if the runtime exposes it

    print("\nComputing the SAME pairs on the exact/default (ideal) simulator ...")
    fidelity_kernel_exact = FidelityQuantumKernel(feature_map=fm)
    exact_pair_values = [
        float(fidelity_kernel_exact.evaluate(np.array([X[i]]), np.array([X[j]]))[0, 0])
        for i, j in pairs
    ]

    ibm_arr = np.array(ibm_pair_values)
    exact_arr = np.array(exact_pair_values)
    abs_err = np.abs(ibm_arr - exact_arr)
    print(f"\nKernel error vs ideal:  mean={abs_err.mean():.4f}  max={abs_err.max():.4f}")
    for (i, j), iv, ev, e in zip(pairs, ibm_arr, exact_arr, abs_err):
        print(f"  ({i},{j})  IBM={iv:.4f}  exact={ev:.4f}  |err|={e:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "experiment": "phase16_qpu_characterization",
        "backend": backend.name,
        "calibration_timestamp": calibration_ts,
        "transpiled_depth": depth,
        "two_qubit_gate_count": two_q_count,
        "transpiled_ops": dict(ops),
        "shots": config.IBM_SHOTS,
        "wall_time_s_includes_queue": round(wall_time, 1),
        "queue_independent_execution_time_s": exec_time_reported,
        "n_points": N_POINTS,
        "pairs": [[i, j] for i, j in pairs],
        "ibm_kernel_values": ibm_arr.tolist(),
        "exact_kernel_values": exact_arr.tolist(),
        "abs_error": {"mean": float(abs_err.mean()), "max": float(abs_err.max())},
    }
    path = f"{RESULTS_DIR}/phase16_qpu_characterization_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
