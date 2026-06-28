# Q-Armor — Research Prototype to Production Tool

> **Purpose:** This document maps everything required to turn the Phase 0–8
> research prototype into a fully operational network intrusion detection system.
> It covers performance gaps, architectural changes, methodology improvements,
> and a prioritised experiment backlog. Read alongside `docs/ROADMAP.md`
> (research goals) and `docs/PHASE8_RESULTS.md` (final research numbers).

---

## 1. Current State vs. Production Requirements

### 1.1 What the prototype does

The prototype is a complete research pipeline across two separate subsystems:

**Subsystem A — CICIoT2023 (Phase 0–3)**
- Raw packet-level features → `QuantumPreprocessor` → `ModelSelector` → `RandomForest`
- Full agent loop: perception → reasoning → planning → action → memory
- 15-class attack classification; alert prioritisation; defence recommendations
- `AgentCore.process()` and `run_stream()` are callable on CICIoT2023 rows

**Subsystem B — NetFlow / NF-* (Phase 5–8)**
- NetFlow 8-feature input → `NFPreprocessor` → `DetectTypeCascade`
- Stage 1: PegasosQSVC binary detection (quantum kernel)
- Stage 2: Random Forest coarse typing (5 classes: Benign/DoS/Injection/Recon/Backdoor)
- Reflexion outer loop: `Evaluator` → `SelfReflector` → `EpisodicMemory` → policy update
- Validated on NF-ToN-IoT and NF-UNSW-NB15

### 1.2 The critical gap

**These two subsystems are not connected.** The cascade (`reasoning/cascade.py`) and
the agent loop (`agent/agent_core.py`) are independent experiment scripts. Nothing wires
them together. Additionally, neither subsystem accepts live network traffic — both require
pre-extracted CSV rows.

```
TODAY (research):

  CSV rows → NFPreprocessor → DetectTypeCascade → CascadeReport
                                                        (dead end)

  CSV rows → QuantumPreprocessor → AgentCore → ActionModule → alerts
                                                        (dead end)

GOAL (production):

  Live traffic → PacketCapture → FlowExtractor → NFPreprocessor
              → DetectTypeCascade (QSVC Stage 1 + RF Stage 2)
              → ActionModule (alerts, defence, priority)
              → EpisodicMemory / SelfReflector (Reflexion)
              → Policy update → back to cascade
```

---

## 2. Performance Gaps

All numbers below are from the sealed test set (E8c) unless noted.

### 2.1 Metrics that need improvement

| Metric | Current | Production Target | Gap | Root Cause |
|--------|---------|------------------|-----|------------|
| Within-domain AUROC | 0.9553 | ≥ 0.97 | −0.015 | Small kernel subset (n=150) |
| Within-domain macro-F1 | 0.7227 | ≥ 0.85 | −0.127 | Recon class data starvation |
| Cross+retrained AUROC | 0.7300 | ≥ 0.85 | −0.120 | Pool size (n=300) too small |
| Cross+retrained macro-F1 | 0.1608 | ≥ 0.50 | −0.339 | Stage 2 RF never adapts cross-domain |
| FPR@95 (cross+retrain) | 0.5540 | ≤ 0.20 | −0.354 | Domain shift not fully corrected |
| Recon F1 (within-domain) | 0.1220 | ≥ 0.50 | −0.378 | Too few Recon flows in NF-ToN-IoT |
| Backdoor F1 (cross) | 0.0080 | ≥ 0.40 | −0.392 | UNSW-NB15 Backdoor is different distribution |
| QSVC predict speed | ~3s/sample | ≤ 50ms | ×60 too slow | O(n_test × n_sv) kernel cost |
| PegasosQSVC confidence | uncalibrated | calibrated | broken | Sigmoid not centred at 0.5 |

### 2.2 Why cross-domain typing is so poor

The Stage 2 RF typer is trained on NF-ToN-IoT attack types (ddos, dos, injection,
scanning, backdoor, etc.) and never retrained when the agent switches to NF-UNSW-NB15.
UNSW-NB15 has a completely different distribution: Fuzzers, Exploits, Shellcode,
Analysis dominate — none of which map cleanly to what the RF has seen. The taxonomy
mapping (NF_COARSE_TAXONOMY in agent_config.py) handles the label translation but the
feature-space distribution of those classes differs between datasets.

`SWITCH_SUBSET` currently retrains only Stage 1 (the QSVC binary detector). Stage 2
retraining is implemented in `SubsetRetrainer.retrain_rf()` but is never triggered.

---

## 3. Architectural Changes Required

### 3.1 Unify the two subsystems into one agent loop

**Current:** Two disconnected pipelines.
**Required:** One loop where the cascade output feeds the agent's action and memory modules.

Files to change:
- `agent/agent_core.py` — replace `QuantumPreprocessor` + `ModelSelector` inner loop
  with `NFPreprocessor` + `DetectTypeCascade`; wire `CascadeReport` to `ActionModule`
- `action/action.py` — extend `classify_attack()` and `build_output()` to accept
  a `CascadeResult` instead of a single `(label, confidence, model_name)` tuple;
  include both binary and type predictions in the output dict
- `agent/evaluator.py` — replace `EpisodeReport` construction from simulated arrays
  with construction from live `CascadeReport` objects
- `agent/agent_config.py` — add the unified pipeline entry point; deprecate the
  CICIoT2023-specific config keys (or keep both behind a `DATASET_MODE` switch)

**New flow per inference window (episode):**
```python
# Every EPISODE_SIZE flows:
results   = cascade.predict(X_window)           # CascadeResult
report    = evaluator.evaluate(results, y_bin)  # EpisodeReport
lesson    = reflector.reflect(report, memory, policy)
memory.record(report, lesson)
if lesson and lesson.action == "SWITCH_SUBSET":
    new_qsvc, _, _ = retrainer.retrain(X_target_pool, y_target_pool)
    new_rf         = retrainer.retrain_rf(X_target_pool, y_target_pool)
    cascade.binary_model = new_qsvc
    cascade.type_model   = new_rf
policy = memory.get_policy()
```

### 3.2 Build the packet-to-feature pipeline

There is currently no code that goes from live network traffic to the 8 NF features
the model expects. This is the single most important missing piece for production.

**Required new module:** `perception/flow_extractor.py`

```
Live interface / pcap file
        ↓
  PacketCapture (scapy / pyshark / dpkt)
        ↓
  FlowAggregator  — groups packets into flows by (src_ip, dst_ip, src_port, dst_port, proto)
        ↓
  FeatureExtractor — computes:
        L4_DST_PORT              (destination port)
        L4_SRC_PORT              (source port)
        IN_BYTES / OUT_BYTES     (byte counts per direction)
        IN_PKTS  / OUT_PKTS      (packet counts)
        FLOW_DURATION_MILLISECONDS
        TCP_FLAGS                (flag bitmask OR across flow)
        ↓
  NFPreprocessor.transform()    (log1p + MinMax → [0, π])
        ↓
  DetectTypeCascade.predict()
```

Dependencies to add: `scapy` or `pyshark` for live capture; `dpkt` or `scapy` for
pcap file replay. All are pure Python and pip-installable.

### 3.3 Fix the Reflexion loop from simulation to live

Phase 7 ran 10 scripted episodes in `experiments/phase7_reflexion.py`. The Reflexion
components (`EpisodicMemory`, `SelfReflector`, `Evaluator`) are correct and all tests
pass, but they are called from a simulation script, not from within the agent's
inference path.

**Required:** A production inference runner that:
1. Accepts a stream of flows (live or from a pcap replay)
2. Buffers them into windows of `EPISODE_SIZE` samples
3. Calls the cascade on each window
4. Calls `Evaluator.evaluate()` → `SelfReflector.reflect()` → `EpisodicMemory.record()`
5. Acts on the returned lesson (retrain / switch / reinforce)
6. Continues to the next window

This can be added as `experiments/run_agent.py` or promoted to a CLI entry point.

### 3.4 Activate VQC and QAE in the model hierarchy

VQC and QAE are in the model hierarchy `['QSVM', 'VQC', 'QAE', 'CLASSICAL']` but are
never registered in `ModelSelector` because online training cost is too high. Every
`SWITCH_MODEL` lesson cascades through them as unregistered stubs until reaching QSVC.

**Two options — pick one before production:**

**Option A — Train offline, load at startup:**
Train `VQCModel` and `QuantumAnomalyDetector` once on NF-ToN-IoT binary data, serialize
with `model.save()`, and load them at agent startup via `selector.register()`. The
hierarchy then has real models at every tier. VQC training takes ~30 min (500 COBYLA
iterations on n=150); QAE ~20 min.

**Option B — Remove from hierarchy:**
Document that the production model zoo is QSVC + RF only. Remove VQC and QAE from
`MODEL_HIERARCHY` in config. This is honest — they were never benchmarked on NF data
and their performance is unknown.

Option A is the research-correct choice (validates the full model zoo claim). Option B
is the production-pragmatic choice.

### 3.5 Persist cascade models across restarts

Currently the cascade retrains from scratch each run. A production system must:
- Save the fitted QSVC and RF typer after training / retraining
- Load them at startup if checkpoints exist
- Tag each checkpoint with the dataset it was trained on and the episode number

`BaseClassifier.save()` and `.load()` already exist via joblib. The checkpoint
directory `agent_state/checkpoints/` is defined in config. What is missing is the
call sites: save after `SubsetRetrainer.retrain()` and load at `AgentCore.__init__()`.

---

## 4. Methodology Improvements

### 4.1 Calibrate PegasosQSVC probabilities

**Problem:** `PegasosQSVC.predict_proba()` uses an internal sigmoid that is not
centred at 0.5. The confidence field in every alert (`prediction.confidence`) is
therefore unreliable — it does not reflect the true probability of attack.
`predict_labels()` was added to fix hard predictions (and it works), but the soft
scores are still wrong.

**Fix:** Add a post-hoc calibration layer using Platt scaling (logistic regression
on a held-out calibration set) or isotonic regression. This wraps `predict_proba()`
and maps its output to calibrated probabilities.

```python
from sklearn.calibration import CalibratedClassifierCV
# or manually: fit LogisticRegression on (decision_scores, y_cal)
```

Once calibrated, the `confidence` scores can be used directly for alert priority
scoring (`ActionModule.prioritise_alert`) and AUROC thresholding.

### 4.2 Trigger Stage 2 retraining in SWITCH_SUBSET

`SubsetRetrainer.retrain_rf()` already exists and is tested. It is never called
in `SWITCH_SUBSET`. When the agent detects cross-domain drift and retrains Stage 1,
it must also retrain Stage 2 on target-domain labelled samples.

**Condition for Stage 2 retrain:** At least `n_per_class` labelled flows available
from the target domain for each coarse class. If a class has fewer than
`CASCADE_STAGE2_N_PER_CLASS` samples, fall back to the existing RF for that class
and flag it as low-confidence.

**Expected impact:** Cross-domain macro-F1 should climb from 0.161 toward 0.40+
once Stage 2 has seen even a small sample of the target distribution.

### 4.3 SWITCH_SUBSET pool size ablation

The current pool is 300 samples (150 benign + 150 attack) → k-means → 150 kernel
subset. This is the minimum viable configuration. The recovery gain on the sealed
test was AUROC +0.153 (0.577 → 0.730).

More target-domain data should push this higher, up to a plateau. The experiment
(E-P9 below) measures exactly where that plateau is. Expected ceiling: AUROC ~0.85
at pool ~1200 samples, based on the Phase 6 cross-dataset SVM-RBF baseline (0.742).

### 4.4 Recon class fix

Recon F1 = 0.122 within NF-ToN-IoT. The coarse taxonomy maps only `scanning` to
Recon, and NF-ToN-IoT has relatively few scanning flows. Two fixes:

1. **Taxonomy expansion:** Check if any currently unmapped NF-ToN-IoT label should
   map to Recon (review `data/taxonomy.py` against raw label distribution).
2. **Upsample Recon at Stage 2 training:** Raise the Recon `n_per_class` above 2000,
   or apply SMOTE specifically to Recon during RF training.
3. **Use UNSW-NB15 Recon to supplement:** UNSW-NB15 has `Fuzzers`, `Reconnaissance`,
   and `Analysis` classes — all map to Recon. Pool them in Stage 2 training even
   before a full SWITCH_SUBSET event.

### 4.5 CVSS severity calibration

All `SEVERITY_WEIGHTS` in `agent_config.py` are tagged `[VALIDATE]`. They were
initialised from approximate CVSS v3.1 base scores but never empirically calibrated.

For a production tool, severity weights directly determine which alerts the analyst
sees first. They need to be:
1. Reviewed with a domain expert / SOC policy
2. Tied to specific CVE examples per class
3. Validated against a labelled alert dataset (if available)

### 4.6 Confidence threshold calibration

`CONFIDENCE_THRESHOLD = 0.65` is tagged `[VALIDATE]` — it was set as a placeholder
and never calibrated. Until PegasosQSVC probabilities are fixed (§4.1), this
threshold is meaningless because the confidence scores are unreliable.

After calibration (§4.1), sweep `CONFIDENCE_THRESHOLD` over the validation set and
pick the value that maximises F1 or minimises (FPR + miss-rate) for the use case.

---

## 5. Experiment Backlog

Ordered by expected impact on turning this into a working tool.

---

### E-P9 — SWITCH_SUBSET pool size ablation

**What:** Retrain QSVC Stage 1 on target pools of 150 / 300 / 600 / 1200 samples
(currently only 300 tested). Measure AUROC and FPR@95 on the NF-UNSW-NB15 test set.

**Why:** The +0.153 AUROC gain at pool=300 may plateau or grow. This tells us the
minimum data collection requirement for production cross-domain adaptation.

**Expected cost:** 4 × ~90s retrain + 4 × ~3000s predict ≈ 3.5 hours.

**Success condition:** AUROC ≥ 0.85 on UNSW-NB15 at some pool size.

**Files:** `experiments/phase9_pool_ablation.py` (new)

---

### E-P10 — Stage 2 joint retraining on SWITCH_SUBSET

**What:** When `SWITCH_SUBSET` fires, also call `SubsetRetrainer.retrain_rf()` on
target-domain labelled flows. Measure cross-domain macro-F1 before and after.

**Why:** Stage 2 macro-F1 = 0.161 cross-domain is the worst metric in the project.
This is the highest-leverage single fix available.

**Dependency:** Needs labelled target-domain flows with coarse taxonomy labels.
NF-UNSW-NB15 train split already provides these.

**Expected outcome:** Cross macro-F1 0.161 → 0.40–0.60 (matching within-domain
baseline degraded by class distribution shift).

**Files:** Modify `agent/retrainer.py`; add `experiments/phase9_joint_retrain.py`

---

### E-P11 — PegasosQSVC probability calibration

**What:** Fit a Platt calibration layer (logistic regression on held-out calibration
split) on top of `PegasosQSVC.predict_proba()`. Compare calibrated vs uncalibrated
reliability diagrams and Brier scores.

**Why:** All alert confidence scores are currently wrong. Fixing this unlocks
meaningful threshold-based decisions and accurate alert prioritisation.

**Files:** Add `reasoning/calibration.py`; modify `PegasosQSVCModel.predict_proba()`
to optionally apply the calibration layer.

---

### E-P12 — VQC benchmark on NF binary data

**What:** Train `VQCModel` on NF-ToN-IoT binary (n=150 k-means subset, same as
QSVC). Compare AUROC, F1, and training time against PegasosQSVC on the same subset.

**Why:** VQC has never been evaluated on NetFlow data. The model zoo claim (QSVC
outperforms VQC at binary on small n) has only been verified on CICIoT2023. This
experiment confirms or refutes it for the NF setting.

**Expected cost:** ~30 min training (500 COBYLA iterations) + ~5 min predict.

**Files:** `experiments/phase9_vqc_nf_benchmark.py` (new)

---

### E-P13 — QAE benchmark on NF binary data

**What:** Train `QuantumAnomalyDetector` on NF-ToN-IoT benign-only subset. Measure
AUROC as an anomaly detector vs the supervised QSVC on the same test set.

**Why:** QAE is the only unsupervised option — it does not need attack labels during
training, which matters in environments where labelled attack data is scarce. It
has never been evaluated on NF data.

**Files:** `experiments/phase9_qae_nf_benchmark.py` (new)

---

### E-P14 — Real IBM QPU run

**What:** Re-run `experiments/phase8_ibm_hardware.py` without `--aer-only` using
a real IBM Quantum account (free tier). Compare AerSimulator AUROC=0.880 vs real
hardware AUROC under gate errors and decoherence.

**Why:** The current IBM result is AerSimulator only (ideal, noiseless). Real
hardware noise will reduce AUROC. Quantifying this gap is the missing validation
for the quantum contribution claim.

**Setup needed:**
1. Create a free account at quantum.ibm.com
2. `export IBM_TOKEN=<your_token>`
3. `python -m experiments.phase8_ibm_hardware` (no `--aer-only`)

**Expected cost:** ~2–8 hours queue time on IBM free tier. Actual compute is fast
(200 circuits × 1024 shots).

**Files:** `experiments/phase8_ibm_hardware.py` already exists; no code changes needed.

---

### E-P15 — Pcap-to-cascade end-to-end test

**What:** Build `perception/flow_extractor.py` that reads a `.pcap` file, extracts
the 8 NF features per flow, and pipes them into `DetectTypeCascade.predict()`.
Validate on a publicly available labelled pcap (e.g. CIC-IDS-2018 raw files).

**Why:** This is the gateway to real deployment. Without it the system only works
on pre-extracted CSV rows.

**Dependencies:** `pip install scapy` or `pip install dpkt`

**Files:**
- `perception/flow_extractor.py` (new) — `FlowAggregator` + `FeatureExtractor`
- `experiments/phase9_pcap_test.py` (new) — end-to-end pcap → alerts test

---

### E-P16 — Live Reflexion loop integration

**What:** Connect the Reflexion components (Evaluator, SelfReflector, EpisodicMemory)
to the live cascade inference path. Build `experiments/run_agent.py` — a production
runner that processes a pcap or network interface in sliding windows of `EPISODE_SIZE`
flows, reflects after each window, and updates the model policy.

**Why:** The Reflexion loop is currently a simulation script (Phase 7). This
experiment validates that it works correctly in a continuous, real-data stream.

**Files:**
- `experiments/run_agent.py` (new) — the unified production runner
- `agent/agent_core.py` (modify) — integrate cascade + Reflexion

---

### E-P17 — Multi-dataset Stage 2 pre-training

**What:** Train the Stage 2 RF typer on a pooled dataset (NF-ToN-IoT + NF-UNSW-NB15
combined, balanced per coarse class). This gives Stage 2 exposure to both attack
distributions before any deployment.

**Why:** The current Stage 2 RF sees only NF-ToN-IoT. A pre-trained joint model
would degrade less severely on cross-domain deployment, even without SWITCH_SUBSET.

**Expected outcome:** Cross macro-F1 0.161 → 0.30–0.45 without any retraining.

**Files:** `experiments/phase9_joint_stage2.py` (new)

---

## 6. Prioritised Action Plan

Grouped by what each achieves for the transition to a working tool.

### Phase 9-A — Fix what's broken in production (1–2 weeks)

| Task | File(s) | Impact |
|------|---------|--------|
| Calibrate QSVC probabilities (E-P11) | `reasoning/calibration.py` | Fixes broken confidence scores |
| Trigger Stage 2 retrain in SWITCH_SUBSET (E-P10) | `agent/retrainer.py` | Cross macro-F1 0.16 → 0.40+ |
| Pool size ablation (E-P9) | `experiments/phase9_pool_ablation.py` | Cross AUROC 0.73 → 0.85 |
| Save/load cascade checkpoints | `agent/agent_core.py` | Survives restarts |

### Phase 9-B — Complete the model zoo (2–3 weeks)

| Task | File(s) | Impact |
|------|---------|--------|
| VQC benchmark on NF data (E-P12) | `experiments/phase9_vqc_nf_benchmark.py` | Validates model zoo |
| QAE benchmark on NF data (E-P13) | `experiments/phase9_qae_nf_benchmark.py` | Unsupervised option |
| Train VQC + QAE offline, register in selector | `agent/agent_core.py` | Full hierarchy active |
| Real IBM QPU run (E-P14) | `experiments/phase8_ibm_hardware.py` | Hardware validation |

### Phase 9-C — Turn it into a tool (3–6 weeks)

| Task | File(s) | Impact |
|------|---------|--------|
| Pcap → features adapter (E-P15) | `perception/flow_extractor.py` | Works on real traffic |
| Unify agent loop with cascade | `agent/agent_core.py` | Single coherent pipeline |
| Live Reflexion runner (E-P16) | `experiments/run_agent.py` | Production inference |
| Multi-dataset Stage 2 pre-training (E-P17) | `experiments/phase9_joint_stage2.py` | Better cross-domain baseline |
| CVSS severity calibration | `agent/agent_config.py` | Accurate alert priorities |
| Confidence threshold calibration | `agent/agent_config.py` | Reliable model switching |

---

## 7. Honest Limitations That Cannot Be Fully Fixed

These are structural limits of the approach, not implementation bugs. A production
tool must document them clearly.

**Quantum speed:** QSVC takes ~3s per sample. Quantum speedup on the kernel
is not achievable with current hardware/algorithms at production scale. The correct
production role for the quantum model is: periodic offline retraining of the kernel
subset, not real-time per-packet classification. RF handles real-time; QSVC
provides the kernel-adapted binary boundary after each SWITCH_SUBSET event.

**No payload signal:** All features are flow-level (byte counts, packet counts,
port numbers, TCP flags). Web/application-layer attacks (XSS, SQL injection) that
leave no flow-level signature cannot be detected from these features regardless
of the model. A full IDS needs deep packet inspection for those classes.

**Real IBM QPU noise:** Even with calibration, real quantum hardware introduces
gate errors (~0.1–1% per 2-qubit gate) that degrade AUROC below the AerSimulator
ceiling of 0.880. The quantum contribution on current NISQ hardware is
demonstrably close to classical SVM-RBF, not dramatically better. The research
value is the agentic / reflective framework, not raw quantum accuracy advantage.

**Cross-domain typing:** Even with Stage 2 retraining, Recon and Backdoor
cross-domain F1 will remain low until NF-UNSW-NB15 provides enough labelled
examples of those classes in the target feature space. This is a data limitation.

**Labelled feedback requirement:** SWITCH_SUBSET retraining needs labelled
target-domain flows. In a real deployment, labels may not be available immediately.
An unlabelled adaptation strategy (clustering, anomaly score, pseudo-labelling)
would be needed for fully unsupervised drift adaptation.

---

## 8. File Map for Phase 9 Work

```
Q-Armor/
├── perception/
│   └── flow_extractor.py          ← NEW: pcap → 8 NF features
├── reasoning/
│   ├── base.py                    ← add: calibration support
│   ├── quantum.py                 ← modify: calibrated predict_proba
│   ├── cascade.py                 ← modify: accept calibrated model
│   └── calibration.py             ← NEW: Platt / isotonic calibration layer
├── agent/
│   ├── agent_core.py              ← MAJOR MODIFY: unified cascade + Reflexion loop
│   ├── retrainer.py               ← modify: trigger retrain_rf in SWITCH_SUBSET
│   └── agent_config.py            ← modify: DATASET_MODE switch, calibrated thresholds
├── experiments/
│   ├── run_agent.py               ← NEW: production inference runner (live / pcap)
│   ├── phase9_pool_ablation.py    ← NEW: E-P9 pool size sweep
│   ├── phase9_joint_retrain.py    ← NEW: E-P10 Stage 2 retraining
│   ├── phase9_vqc_nf_benchmark.py ← NEW: E-P12 VQC on NF
│   ├── phase9_qae_nf_benchmark.py ← NEW: E-P13 QAE on NF
│   └── phase9_joint_stage2.py     ← NEW: E-P17 pooled Stage 2 training
└── tests/
    ├── test_flow_extractor.py      ← NEW: pcap extraction tests
    └── test_calibration.py         ← NEW: calibration layer tests
```

---

## 9. Minimum Viable Production Checklist

Before calling this a working tool, every item below must be checked:

- [ ] `perception/flow_extractor.py` extracts all 8 NF features from a real pcap
- [ ] `NFPreprocessor` + `DetectTypeCascade` wired into `AgentCore`
- [ ] PegasosQSVC `predict_proba()` is calibrated (Brier score < 0.15)
- [ ] `SWITCH_SUBSET` triggers both Stage 1 and Stage 2 retraining
- [ ] Cascade models (QSVC + RF typer) persist across restarts via checkpoints
- [ ] VQC and QAE either registered in `ModelSelector` or removed from hierarchy
- [ ] `experiments/run_agent.py` processes a 10-minute pcap without crashing
- [ ] Reflexion loop fires at least one lesson in a 30-minute live test
- [ ] Alert `confidence` scores are calibrated (verified on held-out data)
- [ ] `SEVERITY_WEIGHTS` reviewed with a domain expert
- [ ] Real IBM QPU run completed and noise degradation documented
- [ ] All [VALIDATE] tags in `agent_config.py` replaced with calibrated values

---

*Generated 2026-06-29. Update this document each time a Phase 9 experiment
completes. Cross-reference with `docs/ROADMAP.md` for research goals and
`docs/PHASE8_RESULTS.md` for the sealed baseline numbers.*
