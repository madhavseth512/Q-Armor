# Q-Armor: Agentic Quantum AI for Cybersecurity Intrusion Detection

Q-Armor is a research prototype built at Deakin University under Dr. Shiva Pokhrel that frames network intrusion detection as an agentic decision loop. The system encodes CICIoT2023 traffic features into quantum states via a custom entangling feature map, dynamically selects the most appropriate model (QSVM, VQC, Quantum Autoencoder, or classical SVM / Random Forest) based on real-time confidence, noise, and distribution-drift signals, and closes the loop by emitting structured attack-type classifications, prioritised alerts, and rule-based defence recommendations across all 15 attack classes.

---

## Module Overview

| Module | Responsibility |
|---|---|
| `perception/` | Custom `CyberSecurityFeatureMap` (8-qubit, 4 entanglement pairs) + FidelityQuantumKernel |
| `reasoning/` | PegasosQSVC, VQC, QuantumAutoencoder, classical SVM/RF; rule-based model selector |
| `memory/` | Disk persistence of kernel params, model weights, ADWIN window; kernel value cache |
| `planning/` | Confidence monitor, ADWIN drift detector, noise monitor, mitigation decider |
| `action/` | 15-class attack classifier, alert prioritiser, defence recommender, JSON/CLI output |
| `agent/` | `AgentCore` orchestrator + `AgentConfig` (single source of truth for all thresholds) |

---

## Repository Structure

```
Q-Armor/
├── agent/          # Orchestrator + centralised config
├── perception/     # Quantum feature encoding & kernel
├── reasoning/      # Model zoo & selector
├── memory/         # State persistence & kernel cache
├── planning/       # Monitoring, drift detection & mitigation
├── action/         # Attack classification, alerts & defence
├── data/           # loader, preprocess, CICIoT2023 data (not committed) + FEATURE_ANALYSIS.md
├── docs/           # PROJECT_CHARTER.md — source of truth (vision, decisions, phases)
├── experiments/    # EDA, validation gates & feature-design studies
├── notebooks/      # Exploratory Jupyter notebooks
├── results/        # Plots and evaluation outputs
└── tests/          # Pytest unit and smoke tests
```

---

## Project Status

Development is **classical-first, quantum-last** across 7 phases (see
[`docs/PROJECT_CHARTER.md`](docs/PROJECT_CHARTER.md) for the full plan, decision
log, and per-phase exit gates).

| Phase | Description | Status |
|---|---|---|
| 0 | Foundation & config | ✅ done |
| 1 | Data pipeline (smart-8 features, two-sided sampling, validation gates) | ✅ done |
| 2 | Classical baselines (RandomForest 0.7376; SVM-RBF + cascade tested) | ✅ done |
| **3** | **Full agent skeleton (perception→reasoning→memory→planning→action)** | ✅ **done** |
| 4 | Perception — custom quantum feature map & kernel | ⬜ next |
| 5 | Quantum models (PegasosQSVC, VQC, QAE) | ⬜ |
| 6 | Experiments & threshold calibration | ⬜ |

**Phase 1 result:** the smart-8 feature set reaches **0.733 validation macro-F1**
(RandomForest), beating the 45-raw-feature baseline (0.675) at just 8 qubits, with
all 15 classes retained. Both validation gates (SHAP feature importance,
diagnostic baseline) pass.

---

## Dataset: CICIoT2023

- **Source:** Canadian Institute for Cybersecurity — https://www.unb.ca/cic/datasets/iotdataset-2023.html
- **Task:** full 15-class multiclass classification (no binary-first, no super-category grouping)
- **Labels:** 34 subtype strings in a `label` column; the 15 **parent** classes are derived via `label.split('-')[0]`:
  `DDoS, DoS, Mirai, BenignTraffic, Recon, MITM, DNS_Spoofing, Backdoor_Malware, VulnerabilityScan, BrowserHijacking, DictionaryBruteForce, XSS, SqlInjection, CommandInjection, Uploading_Attack`
- **Scale:** 5,491,971 train rows; separate `validation.csv` and `test.csv` (1,176,851 rows each).
- **Imbalance:** severe (**≈28,560:1**, from DDoS at 72.8% / 3,998,500 rows down to `Uploading_Attack` at 140 rows). Handled by a two-sided strategy on the **training split only**: undersample majorities to a cap, then SMOTE minorities to a dynamic floor `min(cap, 10×n_real)`.
- **Features:** 8 interpretable, EDA-grounded engineered features ("smart-8", one per qubit) — **no PCA**. Each spans one signal family — volume, size, timing, behaviour, protocol:

  | q | Feature | Source | Family |
  |---|---|---|---|
  | 0 | `traffic_rate` | `Rate` | volume |
  | 1 | `syn_activity` | `syn_count` | volume |
  | 2 | `teardown_activity` | `rst_count + fin_count` | volume |
  | 3 | `header_overhead` | `Header_Length` | volume |
  | 4 | `avg_packet_size` | `AVG` | size |
  | 5 | `flow_timing` | `IAT` | timing |
  | 6 | `handshake_ratio` | `ack_count / (syn_count + 1)` | behaviour |
  | 7 | `protocol_profile` | weighted protocol score (`is_gre`-anchored) | protocol |

  Full evidence trail (EDA, validation gates, design experiments) in [`data/FEATURE_ANALYSIS.md`](data/FEATURE_ANALYSIS.md).

Expected local layout (git-ignored, multi-GB — never committed):

```
data/CICIOT23/
├── train/train.csv
├── test/test.csv
└── validation/validation.csv
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/madhavseth512/Q-Armor.git
cd Q-Armor
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure IBM Quantum credentials (optional)

```bash
cp .env.example .env
# Edit .env and paste your IBM Quantum API token
```

> All development and testing run on a local `AerSimulator` with `FakeBackend` noise models. Real IBM hardware is gated behind `USE_REAL_HARDWARE = False` in `agent/agent_config.py` and is only used after full simulator verification.

### 4. Provide the CICIoT2023 dataset

Place the dataset under `data/CICIOT23/` using the layout shown above.

### 5. Run smoke tests

```bash
pytest tests/ -v
```

---

## Research Context

- **Supervisor:** Dr. Shiva Pokhrel, Deakin University
- **Dataset:** CICIoT2023 (15 parent attack classes; 8 engineered quantum features)
- **Quantum stack:** custom `CyberSecurityFeatureMap`, FidelityQuantumKernel, PegasosQSVC, VQC (RealAmplitudes + COBYLA), QuantumAutoencoder — all on `AerSimulator`
- **Key techniques:** quantum kernel methods, noise-aware model selection, ADWIN drift detection, ZNE error mitigation
