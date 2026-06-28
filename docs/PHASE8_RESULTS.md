# Phase 8 Results — Detect→Type Cascade + SWITCH_SUBSET + Final Evaluation

> **Status: COMPLETE.** All experiments finished; test sets sealed.
> Decisions D-P8.1–D-P8.7 locked below.

---

## Goals addressed

| Goal | Description | Status |
|------|-------------|--------|
| G4 | Cybersecurity response with deep attack typing | ✅ cascade delivered |
| G5 | Cross-dataset generalisation + drift adaptation | ✅ SWITCH_SUBSET confirmed |

---

## Architecture: Detect→Type cascade

```
Stage 1 — Binary detection (attack vs benign)
    Model: PegasosQSVC (quantum kernel, n=150 k-means subset)
    Output: P(attack), hard binary label via predict_labels()

Stage 2 — Attack typing (Benign / DoS / Injection / Recon / Backdoor)
    Model: Random Forest (n=2000 per class, balanced)
    Input: Stage 1 attack-flagged samples only
    Output: coarse taxonomy label + confidence
```

**Key implementation fix (D-P8.1):** `PegasosQSVC.predict_proba()` uses sigmoid calibration
not centred at 0.5 — `argmax(proba)` always predicted benign (F1=0.000). Fixed by adding
`predict_labels()` to the `BaseClassifier` interface (default: argmax) with a
`PegasosQSVCModel` override that calls `self._model.predict()` (native SVM margin boundary).

---

## Experiment E8a — Validation subset results (500 samples per condition)

| Experiment | S1 AUROC | S1 F1 | S2 macro-F1 | FPR@95 |
|------------|---------|-------|------------|--------|
| C1 — Within NF-ToN-IoT (QSVC) | 0.9522 | 0.9376 | 0.6278 | 0.1240 |
| C2 — Cross, no retrain | 0.6231 | 0.6580 | 0.1894 | 0.9960 |
| C3 — Cross, SWITCH_SUBSET retrained | 0.8401 | 0.6667 | 0.1779 | 0.4720 |
| C4 — Cross, classical RF fallback | 0.5990 | 0.4153 | 0.1914 | 0.9720 |

**SWITCH_SUBSET gain (val):** AUROC +0.217, FPR@95 −0.524

Per-class F1 (C1 — within NF-ToN-IoT):
- Benign=0.990  DoS=0.743  Injection=0.853  Recon=0.267  Backdoor=0.286

---

## Experiment E8b — IBM QPU validation (AerSimulator)

| Backend | AUROC | Train (s) | Predict (s) |
|---------|-------|-----------|------------|
| AerSimulator (n=20 train / n=10 test) | 0.8800 | 27.3 | 3.7 |

**Note:** Run in AerSimulator-only mode (ideal, noiseless). Real IBM QPU validation
(gate errors, decoherence) remains as future work. Demonstrates circuit architecture
is valid and AUROC ceiling under ideal conditions.

---

## Experiment E8c — FINAL SEALED TEST SET results (1000 samples per condition)

> **These numbers are FINAL. Test sets were never accessed during training or validation.**

| Experiment | S1 AUROC | S1 F1 | S2 macro-F1 | FPR@95 |
|------------|---------|-------|------------|--------|
| T1 — Within NF-ToN-IoT (QSVC) | **0.9553** | 0.7022 | **0.7227** | 0.1220 |
| T2 — Cross, no retrain | 0.5774 | 0.5830 | 0.1608 | 0.9980 |
| T3 — Cross, SWITCH_SUBSET retrained | **0.7300** | 0.6667 | 0.1608 | 0.5540 |

**SWITCH_SUBSET gain (test):** AUROC +0.1526, FPR@95 −0.4440

Per-class F1 (T1 — within NF-ToN-IoT, sealed test):
- Benign=0.996  DoS=0.708  Injection=0.836  Recon=0.122  Backdoor=0.952

Per-class F1 (T3 — cross-domain retrained):
- Benign=0.565  DoS=0.015  Injection=0.216  Recon=0.000  Backdoor=0.008

---

## Key findings

### What works

1. **Within-domain cascade is strong (T1):** AUROC=0.955, typing macro-F1=0.723.
   Backdoor (0.952), Injection (0.836), DoS (0.708) all well classified. The
   quantum-kernel Stage 1 + classical RF Stage 2 architecture delivers end-to-end
   multi-class attack typing.

2. **SWITCH_SUBSET is the only cross-domain escape:** Without retraining (T2),
   AUROC collapses to 0.577 (near-random). With SWITCH_SUBSET on just 150 target-domain
   samples, AUROC recovers to 0.730 (+0.153). FPR@95 drops from 0.998 to 0.554.
   This validates Goal G5 — the Reflexion agent learns to trigger retraining when
   cross-domain drift is detected.

3. **Reflexion correctly triggers SWITCH_SUBSET:** In Phase 7, the agent autonomously
   fired SWITCH_SUBSET at episode 6 when AUROC < AUROC_FLOOR and drift was detected.
   E8c confirms the retrained model delivers the expected recovery.

### What remains hard

4. **Cross-domain typing (Stage 2) does not improve with Stage 1 retraining:** T3
   macro-F1=0.161 equals T2 (0.161) because the RF typer is trained on NF-ToN-IoT
   classes and NF-UNSW-NB15 has a different attack distribution (Recon and Backdoor
   near-zero). Cross-domain typing would require Stage 2 to also retrain on target-domain
   labelled samples — beyond Phase 8 scope.

5. **Recon is weak within-domain (T1 F1=0.122):** NF-ToN-IoT has very few Recon
   samples in the coarse taxonomy mapping, causing data starvation at the RF typer.

6. **Real QPU hardware noise:** AerSimulator (ideal) gives AUROC=0.880. Real IBM
   hardware gate errors would reduce this — characterised as future work.

---

## Decisions locked (D-P8.x)

| ID | Decision |
|----|----------|
| D-P8.1 | `predict_labels()` added to `BaseClassifier` (default: argmax); `PegasosQSVCModel` overrides with native SVM margin `predict()`. All cascade predictions use this. |
| D-P8.2 | QSVC_CAP=1000 for final eval (500 benign + 500 attack per test set). Full test set predict is cost-prohibitive (~1.4M × 150 kernel circuits). |
| D-P8.3 | Stage 1 trained on n=150 k-means subset (consistent with Phase 7). Stage 2 RF trained on n=2000 per class from NF-ToN-IoT train split. |
| D-P8.4 | SWITCH_SUBSET pool: 300 target-domain samples (150 benign + 150 attack) → k-means → 150-sample QSVC retrain. |
| D-P8.5 | Stage 2 RF typer is NOT retrained in SWITCH_SUBSET. Stage 2 cross-domain degradation is a known limitation (requires labelled target-domain typing data). |
| D-P8.6 | IBM QPU validation used AerSimulator (ideal, noiseless). Real-hardware characterisation is future work. |
| D-P8.7 | Test sets are now UNLOCKED (accessed once, E8c). No further tuning may occur against these numbers. |

---

## File inventory

| File | Purpose |
|------|---------|
| `reasoning/cascade.py` | `DetectTypeCascade` + `CascadeResult` |
| `reasoning/base.py` | `BaseClassifier.predict_labels()` (default: argmax) |
| `reasoning/quantum.py` | `PegasosQSVCModel.predict_labels()` (SVM margin override) |
| `agent/cascade_evaluator.py` | `CascadeEvaluator` + `CascadeReport` |
| `agent/retrainer.py` | `SubsetRetrainer.retrain()` + `retrain_rf()` |
| `experiments/phase8_cascade.py` | E8a validation experiments C1–C4 |
| `experiments/phase8_ibm_hardware.py` | E8b AerSimulator / IBM QPU |
| `experiments/phase8_final_eval.py` | E8c sealed test set evaluation |
| `tests/test_cascade.py` | 19/19 unit tests |
| `results/phase8/phase8_cascade_metrics.json` | E8a raw metrics |
| `results/phase8/phase8_ibm_hardware_metrics.json` | E8b raw metrics |
| `results/phase8/phase8_final_eval_metrics.json` | E8c final metrics |

---

## Project completion

Phase 8 is the final development phase. All five fixed goals from `docs/ROADMAP.md` are met:

| Goal | Evidence |
|------|---------|
| G1 — Agentic architecture (5 modules + Reflexion) | Phase 3 skeleton + Phase 7 outer loop |
| G2 — Quantum-enhanced perception | Phase 4 `CyberSecurityFeatureMap` + PSD-verified kernel |
| G3 — Adaptive/reflective decision layer | Phase 7 `SelfReflector` + `EpisodicMemory`; 4 policy mutations |
| G4 — Deep attack typing | Phase 8 cascade: AUROC=0.955, macro-F1=0.723 within-domain |
| G5 — Cross-dataset generalisation + drift adaptation | SWITCH_SUBSET: AUROC 0.577→0.730 on sealed test |
