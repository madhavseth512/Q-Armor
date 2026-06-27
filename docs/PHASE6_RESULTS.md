# Phase 6 Results — Multi-Dataset Binary Detection (NF-ToN-IoT / NF-UNSW-NB15)

> **Status:** Complete. E7a and E7b run; results locked.

---

## 0. Experimental setup

| Parameter | Value | Source |
|---|---|---|
| Train dataset | NF-ToN-IoT (1,379,274 rows, 80.4% attack) | `data/NF-ToN-IoT/NF-ToN-IoT.csv` |
| Eval dataset (within) | NF-ToN-IoT holdout (20% stratified split) | E7a only |
| Eval dataset (cross) | NF-UNSW-NB15 (1,623,118 rows, 4.5% attack) | E7b only |
| Feature set | NF_FEATURE_NAMES (8 features, locked by EDA) | `agent_config.NF_FEATURE_NAMES` |
| Dropped features | PROTOCOL, L7_PROTO (lowest avg RF importance: 0.021, 0.031) | `experiments/phase6_eda.py` |
| Scale mode | log (log1p on NF_LOG_COLS, then MinMax → [0, π]) | `agent_config.NF_SCALE_MODE` |
| Subset selector | k-means class-balanced, n=150 | `QSVC_SUBSET_SIZE` |
| Quantum kernel | FidelityQuantumKernel (CyberSecurityFeatureMap) | Phase 4 |
| Val subset | 200 samples (100 benign + 100 attack) | QSVC predict cost cap |
| Seed | 42 | `RANDOM_SEED` |

---

## 1. E7a — Within-dataset (NF-ToN-IoT train → NF-ToN-IoT val)

### Results

| Model | Binary F1 | AUROC | AUPR | FPR@TPR95 |
|---|---|---|---|---|
| SVM-RBF (classical baseline) | **0.9340** | **0.9709** | **0.9706** | 0.1300 |
| PegasosQSVC (quantum kernel) | 0.9238 | 0.9444 | 0.9328 | **0.1200** |

### Timing

| Stage | Time |
|---|---|
| Data load + split | ~2s |
| k-means subset selection | 3.3s |
| SVM-RBF fit + predict | <1s |
| PegasosQSVC fit (100 steps) | 80.5s |
| PegasosQSVC predict (200 samples) | 283.8s |

### Interpretation

The quantum fidelity kernel achieves **AUROC 0.944** vs **0.971** for SVM-RBF on
within-dataset detection — a gap of 2.7 points, consistent with the Phase 5 result
on CICIoT2023 (gap was 1.6 points there). QSVC is competitive at n=150.

Notably QSVC achieves a **lower FPR@TPR95 (0.12 vs 0.13)** — at high sensitivity
it lets through slightly fewer false alarms than SVM-RBF. This reflects the quantum
kernel's tighter decision boundary at the 95% recall threshold.

---

## 2. E7b — Cross-dataset (NF-ToN-IoT train → NF-UNSW-NB15 eval)

Same scaler (fit on NF-ToN-IoT), same n=150 subset, same 200-sample eval — only
the evaluation environment changes.

### Results

| Model | Binary F1 | AUROC | AUPR | FPR@TPR95 |
|---|---|---|---|---|
| SVM-RBF | 0.6575 | 0.6442 | 0.6391 | 0.6700 |
| PegasosQSVC | **0.6923** | 0.6383 | **0.6442** | 0.9800 |

### Timing

| Stage | Time |
|---|---|
| Data load (both datasets) | ~4s |
| Scaling (fit on ToN, transform UNSW) | included |
| k-means subset selection | 14.7s |
| SVM-RBF fit + predict | <1s |
| PegasosQSVC fit | 79.8s |
| PegasosQSVC predict | 321.2s |

### Interpretation

Both models collapse under distribution shift. AUROC drops from ~0.97/0.94 to
~0.64/0.64 — only slightly above chance (0.5). The cross-dataset environment
is fundamentally different:

| Property | NF-ToN-IoT (train) | NF-UNSW-NB15 (eval) |
|---|---|---|
| Class balance | 80% attack | 95% benign |
| Attack classes | injection, ddos, password, xss, scanning, dos, backdoor, mitm, ransomware | Exploits, Fuzzers, Reconnaissance, Generic, DoS, Analysis, Backdoor, Shellcode, Worms |
| Attack overlap | — | Only DoS / Backdoor share category names |

The attack classes are almost entirely different — the model learns to flag IoT
attacks (port-targeted floods, credential attacks) but UNSW-NB15 contains
enterprise exploits and fuzzing, which have different traffic signatures.

**QSVC FPR@TPR95 = 0.98:** At 95% recall the quantum model flags 98% of benign
flows as attacks. The kernel is over-generalising — anything sufficiently
"different from the training benign manifold" scores as attack, which is
indiscriminate on a shifted benign distribution.

---

## 3. Drift gap summary

| Model | Within AUROC | Cross AUROC | Drift gap |
|---|---|---|---|
| SVM-RBF | 0.9709 | 0.6442 | **-0.327** |
| PegasosQSVC | 0.9444 | 0.6383 | **-0.306** |

The drift gap is the quantified target for Phase 7 (Reflexion loop): any
recovery of cross-dataset AUROC above the ~0.64 baseline is a measurable
contribution from the agentic adaptation system.

---

## 4. Decisions locked (Phase 6)

| Decision | Resolution |
|---|---|
| **D-P6.1 Feature selection** | PROTOCOL and L7_PROTO dropped (avg RF importance 0.021 / 0.031). L4_DST_PORT kept — highest importance (0.25), contrary to initial assumption. |
| **D-P6.2 Scale mode** | log — log1p on NF_LOG_COLS before MinMax→[0,π]. EDA Part 4: SVM-RBF cross AUROC 0.743 (log) vs 0.694 (linear). |
| **D-P6.3 Within-dataset** | QSVC AUROC 0.944 on NF-ToN-IoT — quantum kernel transfers to NF features with same CyberSecurityFeatureMap. Competitive with SVM-RBF. |
| **D-P6.4 Distribution drift** | Cross-dataset AUROC ~0.64 for both models. Drift gap -0.31 to -0.33. This is the Phase 7 Reflexion target baseline. |
| **D-P6.5 Attack class mismatch** | NF-ToN-IoT and NF-UNSW-NB15 attack classes are almost entirely non-overlapping. Binary detection is the correct framing for cross-dataset; multi-class cascade (Stage 2) must be within-dataset. |

---

## 5. Next steps — Phase 7

Phase 7 implements the Reflexion two-loop agent. It detects the cross-dataset
AUROC drop (D-P6.4 baseline: ~0.64) and autonomously adapts — either by
re-selecting a more domain-agnostic training subset, applying domain adaptation,
or switching to a model that generalises better under shift.

The within-dataset result (E7a: AUROC 0.944) is the upper-bound target.
Any Reflexion-driven recovery between 0.64 and 0.94 is a measurable contribution.
