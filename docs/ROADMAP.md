# Q-Armor — Forward Roadmap (Supervisor-Expanded Scope)

> **Authoritative forward plan.** Written 2026-06 after supervisor direction expanded
> the project. Read this together with `docs/PROJECT_CHARTER.md` (sole source of
> truth for decisions), `docs/PHASE2_RESULTS.md`, `docs/PHASE4_RESULTS.md`, and
> `data/FEATURE_ANALYSIS.md`. **Goals in §2 are FIXED — do not drift from them.**
> Local-only `docs/SUPERVISOR_BRIEFING.md` holds the meeting-facing summary.

---

## 0. Status snapshot — what is already built (Phases 0–4, committed)

| Phase | Done | Key result |
|---|---|---|
| 0 Foundation | ✅ | central config + provenance |
| 1 Data pipeline | ✅ | **smart-8 features** (CICIoT2023); Gate A+B passed |
| 2 Classical baselines | ✅ | **flat RF = 0.7376** (15-class, locked); kernel SVM weak at 15-class (0.35) but **0.91 binary / 0.63 coarse**; hierarchical cascade tested (superseded) |
| 3 Agent skeleton | ✅ | end-to-end agent on RF; `ModelSelector`, planning (ADWIN drift, confidence, mocked noise, mitigation), action (alerts/defence); 36 tests |
| 4 Quantum perception | ✅ | custom `CyberSecurityFeatureMap` + **verified valid PSD fidelity kernel**; timing → quantum kernel feasible only at **n≈100–200** |

**Datasets so far:** CICIoT2023 only (5.49M train, 15 parent classes, severe imbalance ~28,560:1, **flow-only features — no payload**).

**Key learned facts (do not re-litigate):**
- `IAT` is the single most important feature (the brief said to drop it — it was wrong).
- The kernel weakness is the **task** (15-class), not the method: kernels excel at **binary detection**.
- The quantum fidelity kernel is **O(n²) circuit-cost → small subsets only**.
- The **web/application-layer attacks** (XSS/SQLi/etc.) are a **data ceiling** — flow features lack payload signal. Confirmed 5+ ways.
- Prior-correction and scaling do **not** fix the 15-class kernel (tested).

---

## 1. The expanded scope (supervisor direction)

1. **Four new datasets, AUGMENTING CICIoT2023 (not replacing):** `ToN-IoT`,
   `NF-ToN-IoT`, `UNSW-NB15`, `NF-UNSW-NB15`. The `NF-*` are NetFlow-standardised
   (common ~8-feature schema → cross-dataset capable + 8-qubit-friendly).
2. **Reflexion-style two-loop agent:**
   - **Outer loop (Reflexion):** reflects on the inner loop's performance over
     trials, **learns the model-selection + mitigation policy** (upgrading planning
     from rule-based to reflective-learning), storing lessons in **episodic memory**.
   - **Inner loop (model zoo):** the **QML + classical** models that actually detect.
3. **End goal is DEEP multi-class attack typing** (DDoS, DoS, Mirai, …), **not** just
   attack/benign. Binary detection is a *stage*, not the deliverable.

---

## 2. FIXED GOALS (locked — the project is judged on these)

- **G1 — Agentic architecture:** 5 modules (perception, reasoning, memory, planning,
  action) + the **Reflexion outer loop** (new). ✅ skeleton built; outer loop = new work.
- **G2 — Quantum-enhanced perception:** custom quantum feature map + fidelity/
  trainable kernel. ✅ kernel built & verified.
- **G3 — Adaptive decision layer → reflective-learning:** monitor uncertainty,
  confidence, drift, noise → **learn** mitigation (retrain / kernel-adapt / switch /
  fallback) from experience. *(This is the headline upgrade.)*
- **G4 — Cybersecurity response with DEEP attack typing:** classify the specific
  attack class + prioritise alerts + recommend defence. **Multi-class is the goal.**
- **G5 (NEW) — Cross-dataset generalisation + drift adaptation:** train on one
  dataset, adapt to another; real drift the agent reflects on.

**Non-goals / honest boundaries (do not chase):**
- Beating classical models on raw accuracy via quantum (not expected on tabular data).
- Recovering payload-less web-attack signal (a data limitation; sidestep via binary
  detection + anomaly detection, don't pretend to solve it).

---

## 3. Target architecture (two-loop)

```
                ┌──────────────── OUTER LOOP (Reflexion) ─────────────────┐
                │  Evaluator  →  Self-Reflection  →  Episodic Memory       │
                │  (scores)      (writes lessons)    (stores lessons)      │
                │        ▲                                  │              │
                │        │ outcomes                         ▼ policy       │
                └────────┼──────────────────────────────────┼─────────────┘
                         │                                  ▼
       ┌──────────────── INNER LOOP (model zoo, per sample/batch) ─────────┐
   x → │ Perception (quantum kernel)  →  ModelSelector → {QSVC,VQC,QAE,RF,  │ → detect→type
       │                                                  SVM}  → predict   │
       └───────────────────────────────────────────────────────────────────┘
```

- **Detect→Type cascade (resolves binary-vs-multiclass):**
  - **Stage 1 — Detect (binary attack/benign):** quantum-friendly + cross-dataset-
    friendly (common label space across datasets). QML models demonstrate here.
  - **Stage 2 — Type (multi-class attack class):** the END GOAL. Per-dataset
    multi-class (DDoS/DoS/…). Classical models (RF) lead here; quantum optional on a
    coarse mapped taxonomy.
- **Outer loop maps to Reflexion roles:** Actor = inner loop; Evaluator = planning
  monitors; Self-Reflection = NEW **structured/heuristic** lesson generator (rule-derived
  policy updates, NOT an LLM); Memory = episodic store; Policy = learned ModelSelector +
  mitigation.
- **Episode = a batch/window of `EPISODE_SIZE` samples (default ~2,000)**; the agent
  reflects after each episode and updates its policy; a cross-dataset switch is the drift
  it adapts to.

---

## 4. Dataset analysis plan (verify everything — no assumptions)

For **each** dataset, replicate the CICIoT2023 rigor (`experiments/eda_verification.py`):
1. Acquire + locate the raw files; confirm row counts, columns, label fields.
2. **NetFlow schema check:** confirm whether `NF-*` is v1 (~8 features) or v2 (~43).
   v1 → 8 qubits directly; v2 → select 8 features.
3. Class taxonomy per dataset (they DIFFER — UNSW-NB15 ~9 categories, ToN-IoT ~9–10).
4. Imbalance, NaN/inf, dead/constant columns.
5. Feature correlations → re-derive entanglement pairs for the NetFlow feature map.
6. Document each in a `data/<dataset>_ANALYSIS.md`.

**Dataset plan (ALL 4 in scope, decided):**
- **NF-2 (NF-ToN-IoT, NF-UNSW-NB15) = cross-dataset core:** common ~8-feature NetFlow
  schema → ONE pipeline, cross-dataset drift, 8-qubit fit. The quantum + drift work
  centres here.
- **Raw-2 (ToN-IoT, UNSW-NB15) = per-dataset multi-class benchmarks:** richer features,
  own feature engineering (incompatible schemas, so not used for cross-dataset directly).
- **CICIoT2023 retained** (smart-8 results stand).
- **Common coarse taxonomy** (DoS/Recon/Injection/Backdoor/Benign…): build a per-dataset
  → taxonomy mapping so cross-dataset can go beyond binary into **coarse multi-class**.

---

## 5. Experiment plan (ordered, data-driven, with gates)

> Principle (locked): report only measured results; do not predict; do not discard an
> option without a data-verified result. Keep test splits sealed until final.

- **E1 — Acquire + EDA-verify all datasets** (§4). Gate: schemas + taxonomies known.
- **E2 — NetFlow pipeline:** loader + preprocessing (8 features → (0,π)) + entanglement
  re-derivation. Gate: clean `(0,π)` arrays per NF dataset.
- **E3 — Classical MULTI-CLASS baselines (RF) per dataset** (the G4 end goal). Report
  macro-F1 + per-class + confusion. Gate: per-dataset multi-class baseline locked.
- **E4 — Binary DETECTION baselines (RF + SVM-RBF) per dataset + the quantum-feasible
  small-n version.** Report detection rate / false-alarm / AUROC (match the supervisor's
  metric table: F1, AUROC, AUPR, FPR@TPR95).
- **E5 — Cross-dataset experiment:** train on NF-ToN-IoT, evaluate on NF-UNSW-NB15 (and
  reverse). Binary first (common labels). Measures generalisation + provides REAL drift.
- **E6 — Quantum models (Phase 5 work) on binary at n≈100–200:** PegasosQSVC (uses the
  built kernel), then VQC, then QuantumAutoencoder (anomaly/benign). Compare against the
  classical SVM/RF on the IDENTICAL small subset.
- **E7 — Reflexion outer loop prototype:** Evaluator + Self-Reflection + episodic memory;
  the agent learns model-selection/mitigation across the cross-dataset drift stream.
  Compare reflective policy vs the current rule-based policy.
- **E8 — Detect→type cascade end-to-end** per dataset + cross-dataset; final metrics.

---

## 6. How this addresses past problems (honest)

| Past problem | Status under new scope |
|---|---|
| Kernel weak at 15-class | **Resolved by scoping** — quantum runs on binary detection (0.91) |
| Single-dataset overfit | **Resolved** — cross-dataset generalisation (E5) |
| Adaptive layer under-evidenced (drift synthetic, noise mocked) | **Resolved** — real cross-dataset drift + Reflexion learning (E7) |
| Quantum scale limit n≈100–200 | **Worked around** (not removed) — binary + NetFlow-8 fits it |
| Web/app-layer data ceiling | **Sidestepped** at detection; fine typing still limited where payload absent |
| Quantum accuracy advantage | **Reframed** — contribution is the agentic/reflective system, not raw quantum accuracy |
| Severity weights / thresholds placeholders | Unchanged — Phase-8 calibration |

---

## 7. Phase mapping (extends the charter)

| Phase | Scope |
|---|---|
| **5** | Quantum models on CICIoT2023 **binary/coarse** (QSVC → VQC → QAE) — finish the quantum zoo on the known dataset |
| **6** | **Multi-dataset integration** — NetFlow loader, per-dataset EDA (E1–E2), multi-class + binary baselines (E3–E4), cross-dataset (E5) |
| **7** | **Reflexion outer loop** — Evaluator/Reflection/episodic memory; learned policy; validated on cross-dataset drift (E7) |
| **8** | **Detect→type cascade + calibration + final evaluation** — thresholds, CVSS severity, sealed test sets (E8) |

*(Sequencing of Phase 5 vs 6 is an OPEN DECISION — see §8.)*

---

## 8. Decisions log (RESOLVED 2026-06 with supervisor)

| Decision | Resolution |
|---|---|
| **Reflector mechanism** | **Structured/heuristic** reflector (rule-derived policy updates) — NOT an LLM. |
| **GNN baseline table** | **Context only** (which datasets to use). NOT a benchmark to match/beat; unrelated to our metrics. |
| **Episode/trial** | An **episode = a batch/window of `EPISODE_SIZE` samples (default ~2,000)**. Reflect after each episode; aggregate accuracy/confidence/drift/noise → rule-derived lesson → policy update. A **cross-dataset switch = the drift** the agent adapts to. |
| **Sequencing** | **Phase 5 (quantum models on CICIoT2023) FIRST**, then Phase 6 datasets. Kernel is ready; small/fast; completes the quantum zoo before porting to NetFlow. |
| **Quantum benchmark task** | **BINARY detection** — QSVC *binary* vs SVM-RBF *binary* on the IDENTICAL small subset (fair, feasible). Coarse 4-group optional secondary. **NOT** full multi-class for the quantum comparison (data-starved at small-n). Multi-class attack typing is the **classical RF (Stage 2)** job. |
| **Datasets** | **ALL 4** (ToN-IoT, NF-ToN-IoT, UNSW-NB15, NF-UNSW-NB15) + CICIoT2023 retained. NF-2 are the cross-dataset core (common schema); raw-2 are per-dataset multi-class benchmarks. |
| **Cross-dataset multi-class** | **YES — map all datasets to a COMMON COARSE TAXONOMY** (e.g. DoS/Recon/Injection/Backdoor/Benign…) so cross-dataset goes beyond binary into coarse multi-class. (Requires a manual per-dataset → taxonomy mapping, built in E1/E5.) |

**Remaining to verify from data (E1 — not assumptions):** are the 4 datasets downloaded
and where; NF version v1 (~8 features) vs v2 (~43).
