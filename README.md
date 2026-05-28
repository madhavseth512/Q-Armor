# Q-Armor: Agentic Quantum AI for Cybersecurity Intrusion Detection

Q-Armor is a research prototype built at Deakin University under Dr. Shiva Pokhrel that frames network intrusion detection as an agentic decision loop. The system encodes NSL-KDD traffic features into quantum states, dynamically selects the most appropriate model (QSVM, VQC, Quantum Autoencoder, or classical SVM / Random Forest) based on real-time confidence, noise, and distribution-drift signals, and closes the loop by emitting structured attack-type classifications, prioritised alerts, and rule-based defence recommendations.

---

## Module Overview

| Module | Responsibility |
|---|---|
| `perception/` | ZZFeatureMap encoding, FidelityQuantumKernel + TrainableFidelityQuantumKernel |
| `reasoning/` | PegasosQSVC, VQC, QuantumAutoencoder, classical SVM/RF; rule-based model selector |
| `memory/` | Disk persistence of kernel params, model weights, ADWIN window; kernel value cache |
| `planning/` | Confidence monitor, ADWIN drift detector, IBM noise poller, mitigation decider |
| `action/` | Multiclass attack classifier, alert prioritiser, defence recommender, JSON/CLI output |
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
├── data/           # Raw & processed NSL-KDD data (not committed)
├── experiments/    # Training scripts & ablation studies
├── notebooks/      # Exploratory Jupyter notebooks
├── results/        # Plots and evaluation outputs
└── tests/          # Pytest unit and smoke tests
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

### 3. Configure IBM Quantum credentials

```bash
cp .env.example .env
# Edit .env and paste your IBM Quantum API token
```

### 4. Download NSL-KDD dataset

Place `KDDTrain+.arff` and `KDDTest+.arff` inside `data/raw/`.  
Download from: https://www.unb.ca/cic/datasets/nsl.html

### 5. Run smoke tests

```bash
pytest tests/ -v
```

---

## Research Context

- **Supervisor:** Dr. Shiva Pokhrel, Deakin University
- **Dataset:** NSL-KDD (41 features, 5 classes: normal / DoS / Probe / R2L / U2R)
- **Key techniques:** Quantum kernel methods, VQC, noise-aware model selection, ADWIN drift detection, ZNE error mitigation
