# Phase 5 Results — Quantum Models on CICIoT2023 Binary Detection

> **Status:** Complete. All three experiments run; results locked.
> This is the evidence trail for the Phase 5 quantum model zoo.

---

## 0. Experimental setup (fixed across all runs)

| Parameter | Value | Source |
|---|---|---|
| Dataset | CICIoT2023 (train/val splits, test.csv **sealed**) | Phase 1 |
| Feature set | smart-8 (8 features, [0, pi]) | `QUANTUM_FEATURE_NAMES` |
| Feature map | `CyberSecurityFeatureMap` (8-qubit, reps=2) | Phase 4 |
| Kernel | Fidelity quantum kernel (verified PSD, Phase 4) | `FidelityQuantumKernel` |
| Task | **Binary detection** (0=BenignTraffic, 1=attack) | `data/binary.py` |
| Subset selector | `select_kernel_subset` (k-means, class-balanced) | `data/sampling.py` |
| Train subset size n | 150 (75 benign + 75 attack) | `QSVC_SUBSET_SIZE` |
| Simulator | `qiskit_aer.primitives.SamplerV2` (local AerSimulator) | `USE_REAL_HARDWARE=False` |
| Seed | 42 | `RANDOM_SEED` |

---

## 1. E6a — PegasosQSVC vs SVM-RBF (identical subset)

**Val subset:** 200 samples (100 benign, 100 attack) — capped due to QSVC prediction cost
(n_val x n_sv kernel evals; 200 x ~75 = 15,000 circuit evaluations at ~15ms each).

### Parameters

| Parameter | Value |
|---|---|
| PegasosQSVC num_steps | 100 (`PEGASOS_TAU`) |
| PegasosQSVC C | 1.0 |
| SVM-RBF | `balanced` class_weight, gamma=`scale` |

### Results

| Model | Binary F1 | AUROC | AUPR | FPR@TPR95 |
|---|---|---|---|---|
| SVM-RBF (classical baseline) | **0.9524** | **0.9970** | **0.9967** | **0.0100** |
| PegasosQSVC (quantum kernel) | 0.9388 | 0.9811 | 0.9834 | 0.0900 |

### Timing

| Stage | Time |
|---|---|
| Data load + preprocess | 61s |
| Subset selection (k-means n=150) | 6.4s |
| SVM-RBF fit + predict | <1s |
| PegasosQSVC fit (100 steps) | 64.7s |
| PegasosQSVC predict (200 samples) | 219.8s |

### Interpretation

The quantum fidelity kernel (QSVC) achieves **AUROC 0.981** vs **0.997** for SVM-RBF — a gap
of 1.6 points, which is competitive given the n=150 constraint. Both models are strong binary
detectors. The main weakness is FPR@TPR95: at the 95% recall operating point, QSVC lets through
9% false alarms vs 1% for SVM-RBF. This reflects the kernel's slightly wider decision boundary
at high sensitivity thresholds.

**Key finding (D-P5.3):** The quantum fidelity kernel is competitive with the classical RBF
kernel at n=150 binary. The contribution is the agentic use of the kernel, not a raw accuracy
win (consistent with ROADMAP non-goal).

---

## 2. E6b — VQC binary classifier

**Val subset:** 1000 samples (500 benign, 500 attack).

### Parameters

| Parameter | Value |
|---|---|
| Feature map | `CyberSecurityFeatureMap` (same as QSVC) |
| Ansatz | `RealAmplitudes(8, reps=2)` — **24 parameters** |
| Optimizer | COBYLA |
| Max iterations | 500 (`VQC_MAX_ITER`) — increased from 100 after confirming 100 did not converge |

### Results

| Model | Binary F1 | AUROC | AUPR | FPR@TPR95 |
|---|---|---|---|---|
| SVM-RBF (reference) | **0.9527** | **0.9961** | **0.9969** | **0.0000** |
| VQC | 0.8345 | 0.8890 | 0.8750 | 0.5460 |

### Timing

| Stage | Time |
|---|---|
| Data load + preprocess | 102s |
| VQC fit (500 COBYLA iters) | 248.7s (~0.50s/iter) |
| VQC predict (1000 samples) | 7.7s |

### Convergence note (D-P5.1)

At 100 iterations VQC gave AUROC 0.569 (near-random). At 500 iterations it improved to
0.889, demonstrating the model IS learning but needs extensive optimization. Still not
fully converged: AUROC 0.889 vs 0.996 for SVM-RBF, and FPR@TPR95 = 0.546 indicates
the model's probabilities are bunched near 0.5 (indecisive). A fully converged VQC
would likely need 1000+ COBYLA iterations (~8 min) or a gradient-based optimizer.

**Key finding (D-P5.1):** VQC convergence at n=150 is slow. Kernel methods (QSVC) dominate
variational methods at small n — consistent with theory. For this project's claim that
the quantum contribution is agentic/architectural (not accuracy), the QSVC result is the
headline quantum result; VQC is a complementary model in the zoo that falls back to when
QSVC is unavailable or noise makes it untrustworthy.

---

## 3. E6c — Quantum Autoencoder anomaly detection

**Val subset:** 1000 samples (500 benign, 500 attack).

### Parameters

| Parameter | Value |
|---|---|
| Architecture | `CyberSecurityFeatureMap` + `RealAmplitudes(8, reps=1)` — **16 parameters** |
| Trash qubits | 4 (qubits 0–3); latent qubits: 4 (qubits 4–7) |
| Optimizer | COBYLA (scipy minimize) |
| Max iterations | 200 (`QAE_MAX_ITER`) — increased from 30 after first run |
| Shots | 1024 (`QAE_SHOTS`) — increased from 256 after first run |
| Training data | 75 benign samples only (n=75 from the 150-row subset) |

### Results

| Model | AUROC | AUPR | FPR@TPR95 | F1@0.5 |
|---|---|---|---|---|
| OneClassSVM (classical unsupervised) | **0.9859** | **0.9899** | 0.0280 | 0.6684 |
| SVM-RBF (supervised reference) | 0.9961 | 0.9969 | **0.0000** | **0.9527** |
| QuantumAnomalyDetector (QAE) | 0.5594 | 0.5741 | 0.9420 | 0.6406 |

### Timing

| Stage | Time |
|---|---|
| Data load + preprocess | 50s |
| QAE fit (200 iters, 1024 shots, 75 benign) | 71.7s (~0.36s/iter) |
| QAE inference (1000 samples) | 5.1s |

### Convergence and architecture analysis (D-P5.2)

Two runs were performed:

| Run | max_iter | shots | AUROC |
|---|---|---|---|
| Run 1 | 30 | 256 | 0.5875 |
| Run 2 | 200 | 1024 | 0.5594 |

AUROC barely changed between runs — this is **not a convergence failure** but a
**model limitation**. Three contributing factors:

1. **Shot noise floor:** P(trash=|0000>) for 4 random qubits is 1/16 ≈ 6.25%. At 1024
   shots, the std on this estimate is ~sqrt(64)/1024 ≈ 0.8%. The objective landscape is
   shallow (all fidelities cluster near 6.25%), giving COBYLA almost no gradient signal
   to follow regardless of iterations.

2. **Training set diversity:** 75 benign samples span diverse traffic (DNS, HTTP, ICMP,
   SSH, etc.). The QAE must compress this diverse manifold into 4 latent qubits with
   only 16 free parameters — insufficient expressibility for the task.

3. **Basis misalignment:** The loss targets the computational basis state |0000> on
   trash qubits. There is no prior reason this particular state is the natural compression
   target for network traffic; the circuit may need deeper initialisation or a swap-test
   loss to find a meaningful encoding.

**Key finding (D-P5.2):** The fidelity-based QAE does not learn a useful anomaly signal
at n=75, reps=1. AUROC 0.56 ≈ random. The classical OneClassSVM on the same benign
training set achieves AUROC 0.986 — the data is separable, the quantum encoder just
isn't learning it. This is an honest null result that motivates exploring the QAE as
a future direction (deeper ansatz, more data, swap-test loss) rather than a current
deliverable.

---

## 4. Consolidated three-model comparison

All quantum models vs the SVM-RBF classical baseline on the same training subset (n=150).

| Model | Type | Binary F1 | AUROC | AUPR | FPR@TPR95 |
|---|---|---|---|---|---|
| SVM-RBF | Classical (baseline) | **0.9524** | **0.9970** | **0.9967** | **0.0100** |
| PegasosQSVC | Quantum kernel | 0.9388 | 0.9811 | 0.9834 | 0.0900 |
| VQC | Variational quantum | 0.8345 | 0.8890 | 0.8750 | 0.5460 |
| QuantumAnomalyDetector | Quantum anomaly (unsupervised) | — | 0.5594 | 0.5741 | 0.9420 |

*QAE F1 omitted from comparison: it operates as an anomaly scorer (no inherent threshold),
not a calibrated classifier. AUROC is the correct primary metric.*

---

## 5. Decisions locked (Phase 5)

| Decision | Resolution |
|---|---|
| **D-P5.1 Convergence** | VQC did NOT converge at 100 iter (AUROC 0.57). At 500 iter: 0.889. Not fully converged. `VQC_MAX_ITER` set to 500. For future work: gradient-based optimizer or >1000 iterations. |
| **D-P5.2 QAE fidelity gap** | QAE does NOT achieve meaningful benign/attack fidelity gap. AUROC 0.56 ≈ random at 30 and 200 iterations. Architecture limitation, not convergence. QAE remains in the model zoo for the agent hierarchy but is not a strong detector at current scale. |
| **D-P5.3 Quantum vs classical** | QSVC ≈ SVM-RBF at binary detection (AUROC 0.981 vs 0.997). VQC weaker at 0.889. **Confirms non-goal: quantum advantage on accuracy is not expected; the contribution is the agentic quantum system.** |
| **D-P5.4 Timing** | QSVC predict is the bottleneck (219s for 200 samples = 1.1s/sample). VQC predict is fast (7.7ms/sample). Data load dominates wall-clock (~50-100s per run). QSVC val capped at 200; others at 1000. |

---

## 6. Next steps — Phase 6

Phase 6 ports the quantum pipeline to the NetFlow datasets (NF-ToN-IoT,
NF-UNSW-NB15). The 8-feature NetFlow-v1 schema maps directly to the 8-qubit feature
map — no re-engineering needed. Cross-dataset binary detection (train on NF-ToN-IoT,
eval on NF-UNSW-NB15) is the primary E5 experiment, providing real distribution drift
for the Reflexion loop.

The **QSVC** (AUROC 0.981 on CICIoT2023) is the quantum model to carry forward.
VQC carries forward as a secondary model; QAE remains in the zoo but is not the
headline claim.
