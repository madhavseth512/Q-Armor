# Q-Armor — Project Charter (Source of Truth)

> **Status:** living document. This is the authoritative description of what
> Q-Armor is and how it is built. It supersedes the original
> `PROJECT_REVAMP_BRIEF.md` (see [§6](#6-how-we-treat-the-revamp-brief)).
> Where this charter and the brief disagree, **this charter wins.**
> Every quantitative claim here is either marked ✅ *verified on the full
> dataset* or 🔶 *to be verified*. Nothing is assumed true because the brief
> said so.

---

## 1. Vision — what we are building

Q-Armor is an **agentic quantum-AI system for network intrusion detection**,
built as a research internship project at **Deakin University under
Dr. Shiva Pokhrel**. It treats intrusion detection not as a single classifier
but as a **closed decision loop** that:

1. **Perceives** network traffic by encoding it into quantum states,
2. **Reasons** over it with whichever model is most trustworthy right now —
   quantum (QSVM, VQC, Quantum Autoencoder) or classical (SVM-RBF, Random
   Forest),
3. **Remembers** what it has learned and computed across sessions,
4. **Plans** by monitoring its own confidence, data drift, and hardware noise,
   and deciding when to adapt, and
5. **Acts** by classifying the attack, prioritising the alert, and recommending
   a defence.

The research thesis: **a quantum ML system that is *self-aware* — that knows
when its quantum models are degraded by noise or drift and routes around them —
is more robust than any single fixed model.**

---

## 2. Core principles (non-negotiable)

- **Five loosely-coupled modules + an orchestrator.** Each is independently
  importable and testable; no circular imports; a uniform
  `predict(x) -> (label, confidence, model_name)` contract.
- **Classical-first development.** The entire agent must work end-to-end on
  classical models before any quantum component is introduced. Quantum slots
  into proven interfaces, never the other way round.
- **Simulator-first execution.** All development runs on a local `AerSimulator`
  with `FakeBackend` noise. Real IBM hardware is gated and only used after full
  simulator verification.
- **Evidence over assumption.** Feature choices, dropped columns, and thresholds
  must be justified by analysis of the *full* dataset, not by intuition or by a
  brief written from a 50-row sample. Validation gates (SHAP, baseline F1) are
  mandatory.
- **Honest provenance.** Every constant is tagged `grounded` / `anchored` /
  `fixed` / `[VALIDATE]`. A placeholder never silently becomes a "result."
  (See `data/CONFIG_PROVENANCE.md`.)

---

## 3. System architecture (5 modules + orchestrator)

| Module | Responsibility |
|---|---|
| `perception/` | Custom `CyberSecurityFeatureMap` (8-qubit) + FidelityQuantumKernel |
| `reasoning/` | PegasosQSVC, VQC, QuantumAutoencoder, classical SVM/RF; rule-based model selector |
| `memory/` | Disk persistence of params/weights/ADWIN state; kernel value cache |
| `planning/` | Confidence monitor, ADWIN drift detector, noise monitor, mitigation decider |
| `action/` | 15-class attack classifier, alert prioritiser, defence recommender, JSON/CLI output |
| `agent/` | `AgentCore` orchestrator + `agent_config.py` (single source of truth for values) |

Orchestration order: **Perception → Memory(load) → Reasoning → Planning →
Action → Memory(update).**

---

## 4. Dataset — CICIoT2023 (✅ verified on full `train.csv`)

| Fact | Value | Status |
|---|---|---|
| Source | CIC, Univ. of New Brunswick | ✅ |
| Train rows | 5,491,971 | ✅ measured |
| Raw columns | 47 (→ 41 after dropping 5, + label) | ✅ measured |
| Subtype labels | 34, `Parent-Subtype` pattern | ✅ |
| **Parent classes** | **exactly 15** (via `label.split('-')[0]`) | ✅ asserted |
| Class balance | DDoS 72.8% (3,998,500) … Uploading_Attack 140 | ✅ measured |
| **True imbalance ratio** | **≈ 28,560 : 1** (not the brief's ~6,000:1) | ✅ corrected |
| Provided splits | `train.csv`, `validation.csv`, `test.csv` exist | ✅ |

The 15 parent classes: `DDoS, DoS, Mirai, BenignTraffic, Recon, MITM,
DNS_Spoofing, Backdoor_Malware, VulnerabilityScan, BrowserHijacking,
DictionaryBruteForce, XSS, SqlInjection, CommandInjection, Uploading_Attack`.

**Classification target:** full 15-class multiclass from day one. Tail-class
policy (detect-all vs. collapse-rare) is a deferred **config switch**
(`COLLAPSE_RARE_CLASSES`), to be decided from SHAP + model evidence — made more
pressing by the true ~28,560:1 ratio.

---

## 5. Confirmed architectural decisions (from supervisor)

- **Dataset:** NSL-KDD → **CICIoT2023**. ✅
- **No PCA.** Engineer interpretable features instead. ✅ (supervisor directive)
- **Custom feature map**, not ZZFeatureMap/PauliFeatureMap (those may appear only
  as named baselines). ✅ (supervisor directive)
- **8 qubits**, fixed — a *design choice* for simulability / interpretability /
  low noise, **not** an IBM free-tier limit (the free plan offers 127–156
  qubits). ✅ corrected
- **Models:** PegasosQSVC (QSVM), VQC (RealAmplitudes `reps=2` + COBYLA),
  QuantumAutoencoder (trained on BenignTraffic only), classical SVM-RBF + RF as
  active fallbacks. ✅

---

## 6. The revamp brief — RETIRED

> **Status: RETIRED as of Phase 1 completion.** Every actionable claim has been
> verified or corrected against the full dataset and folded into this charter and
> `data/FEATURE_ANALYSIS.md`. The brief is no longer a reference for this project
> and was never a file in the repo — it lived only in chat history. Going forward,
> **this charter is the sole source of truth.** The audit below is kept only as a
> record of what the brief got right vs. wrong.

`PROJECT_REVAMP_BRIEF.md` was produced via an online Claude chat from a **~50-row
manual paste** of `train.csv`. Its *direction* was largely correct (CICIoT2023,
no PCA, custom feature map — all supervisor-confirmed), but its *numbers* were
unreliable, and verifying them every step was essential — `IAT`, which the brief
told us to delete, turned out to be our single most important feature.

### Brief audit

| Brief claim | Verdict |
|---|---|
| 15 parent classes; `split('-')[0]` extraction; 34 subtypes | ✅ verified true |
| Column names incl. typo `Magnitue`, spaces (`Protocol Type`, `Tot sum`/`size`) | ✅ verified true |
| "8 qubits = IBM free tier" | ❌ wrong — free plan is 127–156 qubits |
| "Imbalance ~6,000:1" | ❌ wrong — actually ≈ 28,560:1 |
| Scale **before** train/test split (CHANGE 3 order) | ❌ leaks test stats — reorder |
| `Drate` all-zero; `Number`=9.5 const; `Weight`=141.55 const | 🔶 from 50 rows — re-verify on full data |
| `IAT` near-constant across classes | 🔶 re-verify |
| `Std` ↔ `AVG` correlation 0.76 (basis for dropping `Std`) | 🔶 re-verify |
| Per-feature skew / discriminativeness (e.g. `rst_count` top discriminator) | 🔶 re-verify via real EDA + SHAP |
| Entanglement-pair correlations (`rst_count`↔`Header_Length`=0.75, etc.) | 🔶 re-verify on full data |
| `protocol_profile` weights (2.0×Telnet, 1.5×ICMP, …) | 🔶 `[VALIDATE]` via SHAP |

**Consequence:** Phase 1 now *begins* with a real full-dataset EDA that confirms
or corrects the 🔶 rows above, **before** any feature/column decision is locked.

---

## 7. Development phases

Classical-first, quantum-last. Each phase has an explicit exit gate.

| # | Phase | Key deliverables | Exit gate | Status |
|---|---|---|---|---|
| **0** | Foundation & Config | repo scaffold, `agent_config.py`, provenance, README, `.gitignore` | imports clean; values sourced | ✅ **done** |
| **1** | Data Pipeline | full-dataset EDA → `loader.py` → `preprocess.py` (smart-8 features, scaling, two-sided sampling) → SHAP gate → diagnostic-baseline gate → `FEATURE_ANALYSIS.md` | **both gates PASS** (Gate B 0.733 > raw 0.675; Gate A all 8 matter) | ✅ **done** |
| **2** | Classical Baselines | RandomForest + SVM-RBF + hierarchical cascade; `predict` contract; metrics | **done** — flat RF locked at **0.7376**; kernel + cascade tested & documented | ✅ **done** |
| **3** | Full Agent Skeleton (classical only) | action, planning (mock noise), memory, `ModelSelector`, `agent_core.py`, pytest | **done** — end-to-end agent runs on RF; 36 tests pass | ✅ **done** |
| **4** | Perception (quantum kernel) | `CyberSecurityFeatureMap` (4 pairs), FidelityQuantumKernel, live cache | kernel matrix computes on simulator | ⬜ |
| **5** | Quantum Models | PegasosQSVC → VQC → QAE; live ModelSelector | all 4 model types selectable | ⬜ |
| **6** | Experiments & Calibration | threshold calibration, CVSS severity, ZNE, ablations, (optional) real HW | `[VALIDATE]` values replaced with evidence-based ones | ⬜ |

**Decision gates needing you/your supervisor:** SHAP feature results (Phase 1),
`SEVERITY_WEIGHTS` CVSS mapping (Phase 6), threshold calibration (Phase 6).

---

## 8. Open decisions (decision log)

| # | Decision | Options | Resolution | Status |
|---|---|---|---|---|
| D1 | Scaling vs. split order | (a) split-first/leak-free  (b) brief's scale-first | **(a)** fit scalers on train only; `transform` val/test | ✅ decided (a) |
| D2 | Train/test/val splits | (a) use provided CSVs  (b) re-split from train | **(a)** use provided; fit on train.csv, transform val/test | ✅ decided (a) |
| D3 | Tail-class policy | detect-all vs. collapse-rare | **keep all 15** — smart-8 makes every class viable (web attacks .38–.72) | ✅ decided: keep 15 |
| D4 | Qubit count | stay at 8 vs. 12+ | **stay at 8** — smart-8 gets 0.733 at 8q; 8→12 buys only +0.02–0.03 | ✅ decided: 8 |
| D5 | Drop list | drop brief's 5 vs. only degenerate | **only `Drate`** — Number/Weight/IAT/Std kept (EDA: NOT constant). `IAT` later became the top feature | ✅ decided |
| D6 | `protocol_profile` formula | brief's (Telnet/ARP weighted) vs. EDA-revised | **EDA-revised** — drop dead Telnet/ARP; `is_gre`-anchored (Mirai 65.7%) + ICMP/SSH/UDP/DNS −HTTPS; weights `[VALIDATE]` | ✅ decided |
| D7 | `urg_count` (feature 6) | keep vs. swap | **swapped out** — Gate A showed it weak; smart-8 replaces it with `handshake_ratio` | ✅ superseded by D9 |
| D8 | SMOTE strategy at scale | full-balance vs. undersample+target | **two-sided**: undersample majorities → 50k cap; SMOTE minorities to dynamic floor `min(cap, 10×n_real)` | ✅ decided |
| D9 | Feature sufficiency (Gate B fail) | collapse classes / more qubits / re-engineer 8 | **re-engineer the 8** ("smart-8": swap volume-only `flow_dispersion`+`urgent_activity` → `flow_timing`(IAT)+`handshake_ratio`). 0.43→0.73 at 8q; both gates pass | ✅ decided |

| D10 | Classical baseline model | flat RF vs kernel SVM vs hierarchical cascade | **flat RandomForest (0.7376)** — kernel SVM-10k (0.353) and cascade+reject (0.6425) both measured lower; full evidence in `docs/PHASE2_RESULTS.md` | ✅ decided |

> **D9 resolution:** Gate B initially failed (8 volume-only features = 0.43 macro-F1;
> application-layer classes collapsed). A per-class SHAP diagnostic + an A/B/C
> feature-design experiment showed the fix was feature *design*, not count:
> covering timing + behaviour families lifts the same 8 qubits to 0.73 (beats the
> 45-raw baseline). All 15 classes retained; 8-qubit design preserved. Full
> evidence trail in `data/FEATURE_ANALYSIS.md` §6–10.

*(Resolved decisions stay logged here for traceability.)*

---

## 9. Working agreement

- I act as the **technical guide**: I propose, verify against real data, and
  flag anything in the brief (or my own prior steps) that doesn't hold up.
- **You make the calls.** I will not lock a methodology decision without your
  explicit go-ahead, and I will not write implementation code before approval.
- We move **one phase at a time**, committing when you're satisfied.
