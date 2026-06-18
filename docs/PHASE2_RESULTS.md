# Q-Armor — Phase 2 Results (Classical Baselines)

> Evidence record for Phase 2. All macro-F1 figures are on the **validation**
> split; `test.csv` is sealed for Phase 6. The locked classical baseline is the
> **flat RandomForest at 0.7376 macro-F1.** Every number below is measured, not
> estimated.

---

## 1. Locked baseline — flat RandomForest

| Setting | Value |
|---|---|
| Features | smart-8 (Phase 1) |
| Training | full ~400k two-sided sample (undersample majorities to 50k + SMOTE minorities to dynamic floor) |
| Model | `RandomForestClassifier`, `n_estimators=200`, **`max_depth=18`**, `min_samples_leaf=1` |
| **Validation macro-F1** | **0.7376** |

**Depth sweep (full-400k, leaf=1):** 14 → 0.714, 16 → 0.7345, **18 → 0.7376**, 20 → 0.728, None → 0.715.
Unlimited depth memorises (train F1 1.0, val drops); a leaf floor (`min_samples_leaf>1`) only lowered validation. So: cap depth at 18, no leaf floor.

Per-class: volumetric (DDoS/DoS/Mirai) ~0.99; mid tier (Benign 0.91, MITM 0.86, Recon 0.80, VulnScan 0.84, DNS_Spoofing 0.72); web/rare tier 0.29–0.67 (the data-ceiling classes).

---

## 2. Kernel SVM-RBF investigation (the kernel-scale wall)

Kernel methods scale ~O(n²) and cannot train on the full dataset, so they use a
**10k subset** (this also previews the quantum QSVC's constraint in Phase 5).

### Findings
- **A 10k subset is required, and it is costly.** Even a well-behaved RF drops from
  0.7376 (full) to **0.471 (RF-10k)** — the rare/web classes need more data than a
  10k kernel subset can hold.
- **First SVM run was misconfigured** (narrow grid `{scale,0.1,1}`) → broken 0.32.
- **Corrected, fully-searched grid** (`gamma ∈ {0.001,0.01,0.1,scale,1}`,
  `C ∈ {1,10,100}`, `class_weight ∈ {None,balanced}`, selection ∈ {random,kmeans};
  tuned on a held-out validation slice, reported on a disjoint slice):
  **best SVM-10k = 0.3525** (k-means subset, `gamma=1, C=10, class_weight=None`).
- **k-means subset selection beat random** (0.3525 vs 0.3257) — verified, after an
  earlier broken run had suggested the opposite.

### Root cause of the SVM's weakness (data-verified)
The tuned SVM degraded **even on DDoS** (recall 0.70; accuracy 0.74). Cause: the
**balanced-train / imbalanced-eval mismatch.** The 10k subset must be class-balanced
(5.7:1) to represent rare classes at all, but evaluation is on the real
28,560:1 distribution — so the SVM over-predicts rare classes (precision 0.01–0.42)
and that cannibalises even the dominant classes. RF tolerates this mismatch far
better (0.471 vs 0.353). **This is the kernel baseline QSVC will be measured
against in Phase 5** — no claim is made here about whether QSVC beats it.

---

## 3. Two-stage hierarchical cascade (Option B) — tested, does not beat flat RF

Design: Stage-1 router → 4 groups (Benign / Volumetric / Network / Web); Stage-2
specialists classify within each group. Soft combination `P(class)=P(group)·P(class|group)`.

### Stage-1 router routing (full validation)
| Group | Recall (routing accuracy) | Precision |
|---|---|---|
| Volumetric | **0.998** | 1.000 |
| Benign | 0.922 | 0.929 |
| Network | 0.825 | 0.841 |
| Web | 0.836 | **0.198** |

Routing **recall** is strong for every group (cascade doesn't *lose* samples), but
Web routing **precision is 0.198** — benign+network traffic floods the Web group.

### Full cascade vs flat RF
| Model | Val macro-F1 |
|---|---|
| Flat RandomForest | **0.7376** |
| Hierarchical, no reject | 0.6180 |
| Hierarchical + Web reject path | 0.6425 |

- **The Web reject path worked as designed:** giving the Web specialist a `not_web`
  class (trained on benign+network negatives) and redistributing rejected mass to
  non-Web branches lifted web-class precision and added **+0.0245** over no-reject.
- **But the cascade still trails flat RF by 0.095.** Two measured reasons:
  (1) errors compound across stages; (2) specialists over-predict within their
  group (e.g. Network specialist drives VulnerabilityScan to recall 1.0 /
  precision 0.34, F1 0.50, vs flat RF's 0.84).

### Verdict
Flat RF makes better global precision/recall tradeoffs than a two-stage cascade on
this data. The cascade is retained in the codebase (it honours the model contract
and could be a routing target), but **flat RF is the locked baseline.** The reject
result is a genuine positive finding documented for the record.

---

## 4. The two walls (carried forward)

- **Wall A — web/application-layer data ceiling.** XSS/SQLi/CmdInjection/Uploading/
  Backdoor are limited by *missing payload signal* (CICIoT2023 ships flow
  aggregates, no payload). Confirmed across Gate B, SHAP diagnostic, RF and SVM
  confusion, and the cascade. Not fixable by modelling alone.
- **Wall B — kernel-scale.** Kernel methods (SVM-RBF; QSVC later) are forced onto a
  ~10k subset where macro-F1 collapses (0.47/0.35 vs flat 0.74) via the
  balanced/imbalanced mismatch.

**Open future angles (untested, for later phases):** QuantumAutoencoder anomaly
detection for the web tail (detect-not-classify, needs no web-class data); payload
features via raw PCAP + DPI (a scope decision for the supervisor).

---

## 5. Artifacts & reproducibility

Models persist to `models/*.joblib` (git-ignored). Metrics/confusion matrices
regenerate to `results/phase2/` (git-ignored) via:
`phase2_baselines.py` (RF baselines), `phase2_rf_sweep.py` (depth sweep),
`phase2_svm_tune.py` + `phase2_svm_perclass.py` (kernel SVM), `phase2_stage1_router.py`
(router), `phase2_hierarchical.py` (cascade). Contract tested in `tests/test_reasoning.py`.
