# Q-Armor — Phase 4 Results (Quantum Perception)

> Built and **verified** the quantum kernel machinery: a custom 8-qubit feature
> map producing a mathematically valid fidelity kernel on the simulator. This is
> the realization of project goal 2 (quantum-enhanced perception). All steps are
> task-independent — they hold regardless of which classification task QSVC
> trains on in Phase 5.

---

## 1. Entanglement pairs — derived from the smart-8 correlation matrix

The brief's pairs were chosen from the old feature set (its pair C, AVG↔Covariance,
is invalid — Covariance was dropped). Recomputed on 200k scaled samples; chosen the
4 strongest couplings (≤2 pairs/qubit, including the top feature `flow_timing`):

| Pair | Qubits | Features | correlation |
|---|---|---|---|
| A | q2–q3 | teardown_activity ↔ header_overhead | **+0.70** |
| B | q3–q7 | header_overhead ↔ protocol_profile | −0.50 |
| C | q1–q7 | syn_activity ↔ protocol_profile | −0.44 |
| D | q2–q5 | teardown_activity ↔ flow_timing | −0.42 |

`handshake_ratio` (q6) is ~independent of all features (|r| ≤ 0.11) → single-qubit
encoding (correct — no real coupling to entangle). The brief's `Rate↔protocol`
pair is only −0.07 in smart-8 (another invalidated assumption).
Recorded in `agent_config.ENTANGLEMENT_PAIRS`.

## 2. `CyberSecurityFeatureMap`

Custom 8-qubit circuit (NOT ZZ/Pauli — supervisor directive), `reps=2`:
- Layer 1: `Ry(x_i)` angle encoding per qubit (features pre-scaled to (0, π)).
- Layer 2: for each pair (i,j): `CX(i,j) · Rz(x_i·x_j, j) · CX(i,j)`.
- Gate counts: 16 Ry, 16 CX, 8 Rz.

## 3. Quantum kernel — VALID (a genuine Mercer kernel)

`FidelityQuantumKernel` over the feature map, `K(x₁,x₂)=|⟨φ(x₁)|φ(x₂)⟩|²`, on the
statevector simulator. Verified on a 390-sample stratified real subset:

| Property | Result |
|---|---|
| Symmetric | ✅ |
| Unit diagonal | ✅ |
| Range [0, 1] | ✅ (0.0 – 1.0) |
| **Positive semi-definite** | ✅ (min eigenvalue −1.2e-15 ≈ 0) |

**Class-separability sanity:** same-class mean kernel **0.229** vs cross-class
**0.153** → separation **+0.076** (the kernel captures class structure).
**Correlation with classical RBF = 0.42** — related but distinct.

## 4. Timing — sizes the Phase-5 QSVC subset (the decisive result)

| n (training samples) | kernel-matrix build time |
|---|---|
| 90 | 1.0 min |
| 195 | 5.0 min |
| 390 | 38.6 min |

Per-entry cost rises with n (7.5 → 15 ms), worse than pure O(n²). Extrapolated:
n=500 ≈ 1.5 hr, n=1000 ≈ 6+ hr. **Conclusion: the Phase-5 QSVC training subset must
be small — n ≈ 100–200** (plus a small validation subsample for prediction).

This is the measured, hardware-realistic confirmation of the earlier analysis:
**"same points as the classical SVM, but a much smaller size"** — the classical
SVM-RBF must be re-baselined at this small n for a fair quantum-vs-classical
comparison in Phase 5.

## 5. PerceptionModule + tests

`PerceptionModule` exposes `encode`, `compute_kernel` (with the memory LRU cache),
`compute_kernel_matrix`, and scaffolds the trainable-kernel hooks
(`get/update_kernel_params`) for Phase 6. `tests/test_perception.py` (6 tests)
covers feature-map structure, kernel validity, train/test shapes, encode rows, and
the cache hit. Full suite: 42 tests pass.

## 6. Deferred to Phase 5 (needs the supervisor task decision)
- Which task QSVC trains on (full 15-class vs binary attack/benign vs coarse groups).
- Training + measuring QSVC at n ≈ 100–200 against a re-baselined classical SVM-RBF
  on the **identical** points.
- Wiring the quantum model into `ModelSelector` / `AgentCore`.
