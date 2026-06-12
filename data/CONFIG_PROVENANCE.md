# Q-Armor — Configuration Provenance

This document records the source and justification for every non-obvious value in
[`agent/agent_config.py`](../agent/agent_config.py). It exists so that the origin
of each number is auditable, and so that placeholders are never mistaken for
empirically proven results.

Each value is tagged with a **trust tier**:

| Tier | Meaning |
|---|---|
| 🟢 **grounded** | A library / literature standard default. |
| 🟡 **anchored** | Tied to measured real-hardware characteristics (IBM devices). Order-of-magnitude, varies by device/day. |
| ⚪ **fixed** | Structural to the dataset or architecture. Not tunable. |
| 🔴 **[VALIDATE]** | Heuristic placeholder. MUST be calibrated empirically before being presented as a result. |

---

## 1. Quantum circuit parameters

### `N_QUBITS = 8` — ⚪ fixed (design choice)
**Not** a free-tier hardware limit. The IBM Quantum **Open (free) plan exposes
real devices up to 127 qubits (Eagle) and 156 qubits (Heron r2, `ibm_kingston`)**,
so 8 is well within reach. It is fixed at 8 by design because:

1. **Classical simulability** — the state vector has `2^n` amplitudes. 8 qubits =
   256, trivial on a laptop. Quantum-kernel methods evaluate `O(N^2)` circuits to
   build the kernel matrix, so a low qubit count keeps simulation tractable.
2. **Interpretability** — 8 qubits map one-to-one onto the 8 engineered features.
3. **Noise economy** — fewer qubits → shallower circuits → less noise on real
   hardware, and minimal QPU time against the ~10-minute monthly budget.

**Sources:**
- IBM Quantum — Plans overview: https://quantum.cloud.ibm.com/docs/en/guides/plans-overview
- IBM Quantum — Processor types: https://quantum.cloud.ibm.com/docs/en/guides/processor-types
- Open Plan hardware update (2026-03-16): https://quantum.cloud.ibm.com/announcements/en/product-updates/2026-03-16-open-plan-news

### `N_QUANTUM_FEATURES = 8`, `FEATURE_MAP_REPS = 2`, `VQC_REPS = 2` — ⚪ fixed
Structural to the architecture (CHANGE 3 / CHANGE 4 of the revamp brief):
one engineered feature per qubit; 2 repetitions for both the custom feature map
and the RealAmplitudes ansatz.

---

## 2. Hardware & execution policy

### `USE_REAL_HARDWARE = False` — ⚪ fixed (safety gate)
All development runs on `AerSimulator` + `FakeBackend` noise models. Real hardware
is only enabled after 100% simulator verification (CHANGE 5).

### IBM Open (free) plan limits — context for the above
- **Runtime:** **10 minutes of QPU time per rolling 28-day window** (commonly
  summarised as "10 min/month"). As of **2026-03-16**, active users who consume
  20 min within a 12-month period may opt into a one-time **180 minutes for 12
  months** promotion.
- **Qubits:** up to **127 (Eagle)**; the **156-qubit Heron r2 (`ibm_kingston`)**
  is now also available on the Open plan.

**Sources:**
- Plans overview: https://quantum.cloud.ibm.com/docs/en/guides/plans-overview
- Doubling down on open-access quantum computing: https://www.ibm.com/quantum/blog/open-plan-updates
- Open Plan news (2026-03-16): https://quantum.cloud.ibm.com/announcements/en/product-updates/2026-03-16-open-plan-news

---

## 3. Classification target

### `ATTACK_CLASSES` (15), `N_PARENT_CLASSES = 15` — ⚪ fixed (dataset)
The 15 parent classes of CICIoT2023, derived from the `label` column via
`label.split('-')[0]`. Confirmed by EDA on the full `train.csv`. The loader
asserts exactly 15 unique parents and raises otherwise.

**Source:** CICIoT2023 dataset, Canadian Institute for Cybersecurity —
https://www.unb.ca/cic/datasets/iotdataset-2023.html

### `COLLAPSE_RARE_CLASSES = False` — 🔴 [VALIDATE]
Tail-class strategy is deliberately deferred (CHANGE 0). Imbalance is ~6000:1.
Whether to detect all 15 classes or collapse rare ones into "other" is decided
later from SHAP and model results — kept as a config switch, not a hardcoded policy.

---

## 4. Alert severity weights — 🔴 [VALIDATE]

### `SEVERITY_WEIGHTS` (15 values)
**Methodology:** map each class to a representative **CVSS v3.1 base score**
(0–10 scale, maintained by FIRST.org), normalised to `[0, 1]` by dividing by 10.
Alert priority = `confidence * severity_weight`.

**Caveat:** CVSS scores *individual vulnerabilities (CVEs)*, not attack
*categories*. Collapsing a whole category to one score is a judgment call, so
these values are reasonable **initialisations, not final**. They are to be
replaced with per-class CVSS-grounded values pending supervisor (Dr. Pokhrel)
input. CVSS severity bands for reference: None 0.0 · Low 0.1–3.9 · Medium
4.0–6.9 · High 7.0–8.9 · Critical 9.0–10.0.

| Class | Weight | ≈ CVSS | Rationale |
|---|---|---|---|
| Mirai | 0.90 | 9.0 Critical | IoT botnet / remote takeover |
| Backdoor_Malware | 0.95 | 9.5 Critical | full host compromise |
| CommandInjection | 0.90 | 9.0 Critical | remote code execution |
| MITM | 0.85 | 8.5 High | confidentiality + integrity loss |
| SqlInjection | 0.85 | 8.5 High | data exfiltration |
| Uploading_Attack | 0.85 | 8.5 High | malicious file upload |
| BrowserHijacking | 0.80 | 8.0 High | client compromise |
| DDoS | 0.75 | 7.5 High | availability impact |
| DictionaryBruteForce | 0.75 | 7.5 High | credential attack |
| DoS | 0.70 | 7.0 High | availability impact |
| DNS_Spoofing | 0.70 | 7.0 High | redirection / integrity |
| XSS | 0.60 | 6.0 Medium | client-side script injection |
| Recon | 0.40 | 4.0 Medium | information gathering |
| VulnerabilityScan | 0.40 | 4.0 Medium | pre-attack probing |
| BenignTraffic | 0.0 | — | non-attack class (fixed) |

**Source:** CVSS v3.1 Specification, FIRST.org — https://www.first.org/cvss/v3.1/specification-document

---

## 5. Decision thresholds

### `ADWIN_DELTA = 0.002` — 🟢 grounded
ADWIN's confidence parameter: an upper bound on the **false-positive rate** of
drift detection (~0.2%). Smaller → fewer false alarms, slower to detect real
drift; larger → more sensitive, noisier. This is the **default in the `river`
library's `ADWIN` class**.

**Sources:**
- river `ADWIN` API (default `delta=0.002`): https://riverml.xyz/latest/api/drift/ADWIN/
- Bifet & Gavaldà, *Learning from Time-Changing Data with Adaptive Windowing*,
  SIAM SDM 2007: https://doi.org/10.1137/1.9781611972771.42

### `NOISE_THRESHOLD_ZNE = 0.01` — 🟡 anchored
2-qubit gate error rate (1%) above which Zero-Noise Extrapolation is applied.
Anchored to the **typical median 2-qubit (ECR/CNOT) gate error on current IBM
superconducting devices** (Eagle/Heron-class), available live via
`backend.properties()`. Varies by device and day → anchored, not proven.

**Source:** IBM Quantum processor calibration data —
https://quantum.cloud.ibm.com/docs/en/guides/processor-types

### `READOUT_ERROR_THRESHOLD = 0.03` — 🟡 anchored
Readout (measurement) error rate (3%) above which readout is considered degraded
— the chance a measured qubit is reported as the wrong value (0↔1). Anchored to
**typical IBM readout/assignment error (~1–3%)** on current devices.

**Source:** IBM Quantum processor calibration data —
https://quantum.cloud.ibm.com/docs/en/guides/processor-types

### `NOISE_THRESHOLD_FALLBACK = 0.05` — 🔴 [VALIDATE]
Noise level (5%) above which the agent abandons quantum models entirely for
classical. Set at ~5× the ZNE trigger to mark "too noisy to trust even with
mitigation." **No authoritative source** — a deliberate, conservative starting
point. Must be calibrated against the live backend's median error in Phase 6.

### `CONFIDENCE_THRESHOLD = 0.65` — 🔴 [VALIDATE]
Sliding-window mean prediction confidence (65%) below which the agent switches
model. Chosen as a moderate placeholder: clearly above the ~0.067 chance level
for a 15-class problem, yet below a "confident" >0.8 regime. **No empirical
basis yet** — must come from a precision/recall sweep on the validation set in
Phase 6.

### `CONFIDENCE_WINDOW_SIZE = 20` — 🔴 [VALIDATE]
Number of recent predictions averaged by the confidence monitor. Tuned together
with `CONFIDENCE_THRESHOLD` in Phase 6.

---

## 6. Data pipeline

### `DROP_COLUMNS` — ⚪ fixed (EDA evidence, CHANGE 2)
`Drate` (all-zero), `Number` (constant 9.5), `Weight` (constant 141.55), `IAT`
(near-constant across all classes), `Std` (0.76-correlated with `AVG`). Dropped
to remove degenerate/redundant signal and avoid wasting a qubit on
multicollinearity.

### `APPLY_SMOTE = True` — ⚪ fixed (policy)
SMOTE applied to the **training split only**, never the test/validation sets, to
address the ~6000:1 imbalance without leaking into evaluation.

### `ANGLE_ENCODING_RANGE = (0, π)` — ⚪ fixed
Target range for the final `MinMaxScaler`, matching qubit rotation-angle encoding.

### `RANDOM_SEED = 42` — ⚪ fixed (reproducibility)
Single seed for splits, SMOTE, and model initialisation so runs are reproducible.

---

## Revision policy

When any 🔴 [VALIDATE] value is calibrated, update **both** this file and the
inline comment in `agent_config.py`, recording the new value, the date, and the
experiment that produced it. A placeholder must never silently become a "result."
