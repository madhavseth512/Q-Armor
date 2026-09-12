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
# 24 points -> 276 unique off-diagonal pairs, ~185 s of QPU time at the
# 0.67 s/pair rate measured by the original 4-point run. Large enough to put a
# confidence interval on the kernel-error distribution (the 4-point run gave
# only 6 error samples), still well inside the Open plan's 600 s/28-day budget.
DEFAULT_N_POINTS = 24


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true",
                        help="Confirm you understand this uses real IBM Quantum QPU time.")
    parser.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS,
                        help="Balanced sample size. The run submits n(n-1)/2 kernel "
                             "circuits at roughly 0.67 s of QPU time each.")
    args = parser.parse_args()
    n_points = args.n_points
    n_pairs = n_points * (n_points - 1) // 2

    if not args.confirm:
        print("Phase 16: real-QPU characterisation.")
        print(f"  {n_points} points -> {n_pairs} circuit pairs on real hardware.")
        print(f"  Estimated cost: ~{n_pairs * 0.67:.0f} s of the 600 s / 28-day allowance.")
        print("  This spends real, limited IBM Quantum free-tier QPU time.")
        print("  Re-run with --confirm to proceed.")
        sys.exit(0)

    load_dotenv(override=True)
    token = os.environ.get("IBM_QUANTUM_TOKEN", "").strip()
    if not token:
        sys.exit("IBM_QUANTUM_TOKEN not set in .env -- cannot reach IBM Quantum.")

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSamplerV2
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

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
    idx = np.concatenate([rng.choice(b_idx, n_points // 2, replace=False),
                           rng.choice(a_idx, n_points // 2, replace=False)])
    X = X8[idx]
    print(f"  {X.shape} points, labels {y_bin[idx]}")

    print("\nConnecting to IBM Quantum ...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token,
                                    instance=config.IBM_INSTANCE)
    # least_busy() no longer takes operational=/simulator= kwargs directly in
    # qiskit-ibm-runtime 0.47 -- IBMBackend has no .operational/.simulator
    # attributes any more (verified against the installed version before
    # spending real QPU time). Use the documented filters= callable instead:
    # status().operational is the current way to check backend availability,
    # and configuration().simulator still reports whether a backend is a
    # cloud simulator rather than real hardware.
    def _is_real_and_operational(b) -> bool:
        try:
            return bool(b.status().operational) and not bool(b.configuration().simulator)
        except Exception:
            return False

    backend = service.least_busy(min_num_qubits=config.N_QUBITS, filters=_is_real_and_operational)
    print(f"  Selected backend: {backend.name}")

    calibration_ts = None
    try:
        calibration_ts = str(backend.properties().last_update_date)
    except Exception as e:
        print(f"  [calibration timestamp unavailable: {e}]")

    # IBM's runtime primitives reject circuits that aren't already expressed
    # in the backend's native ISA gate set (confirmed live: submitting the
    # feature map's raw ry gates raised IBMInputValueError, rejected before
    # any job was created -- IBM has required pre-transpiled ("ISA") circuits
    # since March 2024). generate_preset_pass_manager(backend=...) builds the
    # right transpilation target; reused for both the reported depth/gate
    # count AND the actual circuits ComputeUncompute submits, so the reported
    # numbers describe exactly what ran.
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

    fm = CyberSecurityFeatureMap()
    bound = fm.assign_parameters(X[0])
    isa_circuit = pm.run(bound)
    depth = isa_circuit.depth()
    ops = isa_circuit.count_ops()
    two_q_gate_names = {"cx", "cz", "ecr", "cp", "rzx"}
    two_q_count = sum(v for k, v in ops.items() if k in two_q_gate_names)
    print(f"  Transpiled depth: {depth}   two-qubit gates: {two_q_count}   ops: {dict(ops)}")

    pairs = list(itertools.combinations(range(n_points), 2))
    print(f"\nSubmitting {len(pairs)} kernel-pair circuits to {backend.name} "
          f"({config.IBM_SHOTS} shots each) ...")

    # FidelityQuantumKernel takes fidelity=, not sampler= directly (verified
    # against the installed qiskit-machine-learning signature before spending
    # real QPU time -- passing sampler= would have raised a TypeError on
    # construction). Wrap the sampler in a ComputeUncompute fidelity object,
    # same pattern as experiments/phase15_finiteshot_sweep.py. IBMSamplerV2
    # also has no shots= constructor arg -- default_shots must be set via
    # options=, or config.IBM_SHOTS is silently never applied. pass_manager=pm
    # is what makes ComputeUncompute submit ISA-transpiled circuits instead of
    # the raw feature map.
    sampler = IBMSamplerV2(backend, options={"default_shots": config.IBM_SHOTS})
    fidelity_ibm = ComputeUncompute(sampler=sampler, pass_manager=pm)
    fidelity_kernel_ibm = FidelityQuantumKernel(feature_map=fm, fidelity=fidelity_ibm)

    # One batched evaluate(X) call over the whole n x n matrix, not one call
    # per pair -- FidelityQuantumKernel batches the circuits it needs into a
    # single Sampler job internally, whereas per-pair calls would submit one
    # job per pair, each paying its own IBM queue wait. That queue
    # time doesn't count against the QPU-time quota, but it does turn a
    # sub-minute job into a potentially much longer wall-clock wait for no
    # benefit -- same batching already used in phase14_kernel_alignment.py.
    t0 = time.time()
    K_ibm = fidelity_kernel_ibm.evaluate(X)
    ibm_pair_values = [float(K_ibm[i, j]) for i, j in pairs]
    wall_time = time.time() - t0
    print(f"  wall time (includes queue): {wall_time:.1f}s")

    # FidelityQuantumKernel.evaluate() doesn't return the underlying job, so
    # the queue-independent execution time (the number that actually counts
    # against the free-tier quota, distinct from wall-clock queue wait) has
    # to be recovered after the fact via the account's own job history --
    # confirmed live: job.usage() / job.metrics()['usage']['quantum_seconds']
    # is that number (this run: 4 seconds of quantum time for the whole
    # 6-pair job, out of the 600s/28-day Open plan quota).
    exec_time_reported = None
    try:
        recent_job = next(iter(service.jobs(limit=1)))
        exec_time_reported = recent_job.usage()
        print(f"  actual QPU quantum-time used: {exec_time_reported}s "
              f"(job {recent_job.job_id()})")
    except Exception as e:
        print(f"  [queue-independent execution time unavailable: {e}]")

    print("\nComputing the SAME pairs on the exact/default (ideal) simulator ...")
    fidelity_kernel_exact = FidelityQuantumKernel(feature_map=fm)
    K_exact = fidelity_kernel_exact.evaluate(X)
    exact_pair_values = [float(K_exact[i, j]) for i, j in pairs]

    ibm_arr = np.array(ibm_pair_values)
    exact_arr = np.array(exact_pair_values)
    abs_err = np.abs(ibm_arr - exact_arr)

    # With n(n-1)/2 error samples rather than 6, the mean deviation is worth a
    # confidence interval -- that is the whole point of running more than the
    # original 4-point characterisation. Percentile bootstrap, same approach
    # as experiments/phase10_confirmatory.py.
    boot_rng = np.random.default_rng(config.RANDOM_SEED)
    boot_means = [boot_rng.choice(abs_err, size=len(abs_err), replace=True).mean()
                  for _ in range(2000)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    err_stats = {
        "n_samples": int(len(abs_err)),
        "mean": float(abs_err.mean()),
        "std": float(abs_err.std(ddof=1)),
        "median": float(np.median(abs_err)),
        "p95": float(np.percentile(abs_err, 95)),
        "max": float(abs_err.max()),
        "mean_ci95_lo": float(ci_lo),
        "mean_ci95_hi": float(ci_hi),
    }

    print(f"\nKernel error vs ideal ({err_stats['n_samples']} matrix entries):")
    print(f"  mean={err_stats['mean']:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  std={err_stats['std']:.4f}  median={err_stats['median']:.4f}  "
          f"p95={err_stats['p95']:.4f}  max={err_stats['max']:.4f}")
    worst = np.argsort(abs_err)[::-1][:5]
    print("  five largest deviations:")
    for k in worst:
        i, j = pairs[k]
        print(f"    ({i},{j})  IBM={ibm_arr[k]:.4f}  exact={exact_arr[k]:.4f}  "
              f"|err|={abs_err[k]:.4f}")

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
        "n_points": n_points,
        "pairs": [[i, j] for i, j in pairs],
        "ibm_kernel_values": ibm_arr.tolist(),
        "exact_kernel_values": exact_arr.tolist(),
        "abs_error": err_stats,
    }
    path = f"{RESULTS_DIR}/phase16_qpu_characterization_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
