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
├── data/           # CICIoT2023 data (not committed) + feature analysis
├── experiments/    # Training scripts, EDA & ablation studies
├── notebooks/      # Exploratory Jupyter notebooks
├── results/        # Plots and evaluation outputs
└── tests/          # Pytest unit and smoke tests
```

---

## Dataset: CICIoT2023

- **Source:** Canadian Institute for Cybersecurity — https://www.unb.ca/cic/datasets/iotdataset-2023.html
- **Task:** full 15-class multiclass classification (no binary-first, no super-category grouping)
- **Labels:** 34 subtype strings in a `label` column; the 15 **parent** classes are derived via `label.split('-')[0]`:
  `DDoS, DoS, Mirai, BenignTraffic, Recon, MITM, DNS_Spoofing, Backdoor_Malware, VulnerabilityScan, BrowserHijacking, DictionaryBruteForce, XSS, SqlInjection, CommandInjection, Uploading_Attack`
- **Imbalance:** severe (~6000:1, from DDoS at ~72.8% down to `Uploading_Attack` at ~140 rows). Handled with SMOTE on the training split only.
- **Features:** 8 interpretable, EDA-grounded engineered features (one per qubit) — **no PCA**. See `data/FEATURE_ANALYSIS.md` once generated.

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
