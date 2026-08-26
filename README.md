# Q-Armor: Agentic Quantum-Enhanced Network Intrusion Detection

**Q-Armor** is a research prototype conceived and led at **Deakin University by Dr. Shiva Raj Pokhrel, Chief Investigator (CI)**, with **Madhav Seth contributing as a Research Intern under the CI's research supervision**.

Q-Armor investigates adaptive cyber defence by formulating network intrusion detection as a two-loop agentic architecture. The **inner loop** processes each network flow through five modules:

**Perception → Reasoning → Planning → Action → Memory**

An **outer Reflexion loop** evaluates episode-level behaviour, identifies failure modes, retrieves prior experience from episodic memory, and autonomously adapts the model-selection policy.

The research integrates an **8-qubit quantum fidelity kernel** (`CyberSecurityFeatureMap`) with a **Detect→Type cascade**, ADWIN-based concept-drift detection, episodic Reflexion, and a **`SWITCH_SUBSET` cross-domain adaptation mechanism**. The prototype is validated using two large-scale public NetFlow cybersecurity datasets with strictly separated training and sealed test partitions.

---

## Project Leadership and Contributions

### Chief Investigator

**Dr. Shiva Raj Pokhrel**  
School of Information Technology, Deakin University, Australia

Chief Investigator and research lead responsible for:

- project conception and scientific direction;
- Q-Armor research architecture;
- agentic quantum-cybersecurity research methodology;
- integration of quantum learning, adaptive networking, and autonomous decision mechanisms;
- research questions and experimental strategy;
- cross-domain adaptation framework;
- scientific interpretation and validation methodology;
- research supervision; and
- overall project leadership.

### Research Intern

**Madhav Seth**  
Indian Institute of Technology Kharagpur, India

Research Intern working under the supervision of Dr. Shiva Raj Pokhrel, contributing to:

- software implementation;
- experimental pipeline development;
- quantum-kernel implementation;
- dataset preprocessing;
- model training and benchmarking;
- Reflexion-loop implementation;
- experiment execution;
- testing and result generation; and
- technical documentation.

---

## Key Results

| Metric | Value |
|--------|-------|
| Within-domain S1 AUROC (sealed test) | **0.973** |
| Within-domain S2 typing macro-F1 | **0.496** |
| Cross-domain AUROC before retraining (T2) | 0.606 |
| Cross-domain AUROC after `SWITCH_SUBSET` (T3) | **0.913** (+0.306) |
| FPR@95 after `SWITCH_SUBSET` | **0.198** (↓ from 0.974) |
| AerSimulator quantum-kernel validation AUROC | **0.800** |
| Reflexion episodes simulated | 10 |
| `SWITCH_SUBSET` autonomously triggered | Episode 6 |
| Drift confirmation | ADWIN |

---

## System Architecture

### Inner Loop — Per Flow

| Module | Responsibility |
|--------|----------------|
| `perception/` | `CyberSecurityFeatureMap`: 8-qubit angle encoding with four correlation-derived entanglement pairs; `FidelityQuantumKernel`; log1p + MinMax scaling to [0, π] |
| `reasoning/` | `PegasosQSVC` binary classifier; `DetectTypeCascade`; Stage 1 binary detection + Stage 2 RF attack typing |
| `planning/` | ADWIN drift detection (δ=0.002); confidence monitoring; `SubsetRetrainer` for k-means subset reselection and QSVC retraining |
| `action/` | Alert generation, severity assessment, defence recommendations, and JSON/CLI outputs |
| `memory/` | Model persistence, quantum-kernel cache, and append-only `EpisodicMemory` |

### Outer Reflexion Loop — Per Episode

For episodes containing \(N_{\mathrm{ep}}=100\) flows:

| Component | Function |
|-----------|----------|
| `CascadeEvaluator` | Produces `EpisodeReport`: AUROC, F1, macro-F1, AUPR, FPR@95, and drift state |
| `SelfReflector` | Evaluates failure conditions and generates `Lesson`: `BINARY_ONLY`, `SWITCH_SUBSET`, `SWITCH_MODEL`, or `REINFORCE` |
| `EpisodicMemory` | Stores `(EpisodeReport, Lesson)` pairs and maintains experience used by subsequent policy adaptation |

The architecture separates **fast flow-level inference** from **slower episode-level self-evaluation and adaptation**.

---

## Datasets

Q-Armor currently uses two large-scale public NetFlow datasets sharing an eight-feature schema, permitting controlled cross-domain experimentation.

| Dataset | Rows | Attack % | Train | Test |
|---------|------|----------|-------|------|
| NF-ToN-IoT | 1,379,274 | 80.4% | 1,103,419 | 275,855 |
| NF-UNSW-NB15 | 1,623,118 | 4.5% | 1,298,494 | 324,624 |

### Input Features

1. `L4_DST_PORT`
2. `OUT_BYTES`
3. `L4_SRC_PORT`
4. `FLOW_DURATION_MILLISECONDS`
5. `IN_BYTES`
6. `TCP_FLAGS`
7. `OUT_PKTS`
8. `IN_PKTS`

Features undergo `log1p` transformation followed by MinMax scaling to \([0,\pi]\).

### Coarse Attack Taxonomy

**Benign / DoS / Injection / Recon / Backdoor**

The common taxonomy maps the native label spaces of both datasets into a consistent cross-domain evaluation space.

Expected local dataset structure:

```text
data/
├── NF-ToN-IoT/NF-ToN-IoT.csv
└── NF-UNSW-NB15/NF-UNSW-NB15.csv
```

Large datasets are excluded from Git tracking.

---

## Quantum Fidelity Kernel

The `CyberSecurityFeatureMap` maps eight NetFlow features onto **eight qubits** using parameterised \(R_y\) angle encoding.

Entanglement is introduced using four CNOT-connected feature pairs selected from empirical correlations in NF-ToN-IoT:

| Pair | Correlation | Features |
|------|-------------|----------|
| (q1, q6) | +0.926 | OUT_BYTES ↔ OUT_PKTS |
| (q4, q7) | +0.911 | IN_BYTES ↔ IN_PKTS |
| (q6, q7) | +0.814 | OUT_PKTS ↔ IN_PKTS |
| (q4, q6) | +0.774 | IN_BYTES ↔ OUT_PKTS |

The circuit uses:

```text
FEATURE_MAP_REPS = 2
```

Kernel evaluation is performed using Qiskit's `FidelityQuantumKernel`.

Positive-semidefinite behaviour is numerically checked through the resulting Gram matrix.

Q-Armor treats the quantum kernel as an **experimental representation mechanism**. Current results should not be interpreted as evidence of computational quantum advantage.

---

## Cross-Domain Adaptation

A major capability investigated in Q-Armor is autonomous adaptation under distribution shift.

When

\[
\mathrm{AUROC}<0.70
\]

and ADWIN independently indicates concept drift, the outer agent can invoke:

```text
SWITCH_SUBSET
```

The adaptation process performs target-domain subset reselection followed by QSVC retraining.

In the reported experiment:

\[
0.606 \rightarrow 0.913
\]

cross-domain AUROC was observed following `SWITCH_SUBSET`, while:

\[
\mathrm{FPR@95}: 0.974 \rightarrow 0.198.
\]

These results represent performance of the current experimental configuration and should not be interpreted as universal performance guarantees.

---

## Reflexion Policy

The `SelfReflector` currently implements four ordered adaptation rules.

1. **`BINARY_ONLY`**  
   If a multiclass model produces macro-F1 < 0.30, attack typing can be disabled.

2. **`SWITCH_SUBSET`**  
   If AUROC < 0.70 and ADWIN detects drift, the system triggers target-domain subset reselection and QSVC retraining.

3. **`SWITCH_MODEL`**  
   If AUROC < 0.70 without confirmed drift, the active model tier is changed according to:

   ```text
   QSVM → VQC → QAE → CLASSICAL
   ```

4. **`REINFORCE`**  
   If AUROC ≥ 0.70 for three consecutive episodes, the current policy is reinforced.

During the 10-episode experimental simulation:

```text
REINFORCE      × 3
SWITCH_SUBSET  × 1
SWITCH_MODEL   × 4
```

`SWITCH_SUBSET` was autonomously activated at **Episode 6** following ADWIN-confirmed drift.

---

## Repository Structure

```text
Q-Armor/
├── agent/
│   ├── AgentConfig
│   ├── AgentCore
│   ├── CascadeEvaluator
│   ├── SubsetRetrainer
│   └── SelfReflector
│
├── perception/
│   ├── CyberSecurityFeatureMap
│   └── FidelityQuantumKernel
│
├── reasoning/
│   ├── BaseClassifier
│   ├── PegasosQSVCModel
│   └── DetectTypeCascade
│
├── planning/
│   ├── ADWIN drift detector
│   └── confidence monitor
│
├── action/
│   ├── alert generation
│   ├── severity assessment
│   └── defence recommendations
│
├── memory/
│   ├── model persistence
│   ├── kernel cache
│   └── EpisodicMemory
│
├── data/
├── experiments/
├── results/
│   ├── phase7/
│   └── phase8/
├── agent_state/
│   └── episodes.jsonl
├── docs/
│   ├── PROJECT_CHARTER.md
│   ├── ROADMAP.md
│   ├── PHASE7_RESULTS.md
│   ├── PHASE8_RESULTS.md
│   ├── PRODUCTION_ROADMAP.md
│   └── paper.tex
└── tests/
```

---

## Development Phases

| Phase | Description | Status |
|-------|-------------|:------:|
| 0 | Foundation, configuration, project structure | ✅ |
| 1 | NetFlow data pipeline, stratified splits, validation gates | ✅ |
| 2 | Classical baselines and cross-dataset evaluation | ✅ |
| 3 | Five-module agentic inner loop | ✅ |
| 4 | `CyberSecurityFeatureMap` and quantum fidelity kernel | ✅ |
| 5 | Quantum model zoo: PegasosQSVC, VQC, QAE | ✅ |
| 6 | NF-ToN-IoT + NF-UNSW-NB15 cross-domain integration | ✅ |
| 7 | Reflexion, evaluator, `SelfReflector`, episodic memory | ✅ |
| 8 | Detect→Type cascade, `SWITCH_SUBSET`, sealed evaluation | ✅ |

---

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/madhavseth512/Q-Armor.git
cd Q-Armor
```

### 2. Create Environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. IBM Quantum Configuration — Optional

```bash
cp .env.example .env
```

Add IBM Quantum credentials to `.env` when required.

The current experiments can be executed locally using `AerSimulator`. Access to IBM hardware is controlled through:

```python
IBM_HARDWARE_MODE = False
```

in `agent/agent_config.py`.

### 5. Place Datasets

Download NF-ToN-IoT and NF-UNSW-NB15 and place them as shown in the dataset section above.

### 6. Run Tests

```bash
pytest tests/ -v
```

---

## Research Paper

The associated manuscript is maintained in:

```text
docs/paper.tex
```

**Working title:**  
*Q-Armor: Agentic Quantum-Enhanced Network Intrusion Detection with Reflexion-Based Adaptive Policy Learning*

Publication authorship and author ordering should follow the actual intellectual and technical contributions to the resulting manuscript.

---

## Research Context

**Project:** Q-Armor — Agentic Quantum-Enhanced Network Intrusion Detection  
**Chief Investigator:** Dr. Shiva Raj Pokhrel, Deakin University  
**Research Intern:** Madhav Seth, IIT Kharagpur  
**Lead Institution:** Deakin University, Australia  
**Research Areas:** Quantum Machine Learning, Agentic AI, Cybersecurity, Network Intrusion Detection, Concept Drift, Cross-Domain Learning  
**Quantum Stack:** Qiskit, `FidelityQuantumKernel`, `SamplerV2`, `PegasosQSVC`, `AerSimulator`

Q-Armor is an **academic research prototype**. It is not currently intended to constitute a production intrusion-detection or operational cyber-defence system.

---

## Citation

If this repository contributes to academic work, please cite the corresponding peer-reviewed publication when available.

Until publication, the project may be referenced as:

```bibtex
@misc{qarmor2026,
  title        = {Q-Armor: Agentic Quantum-Enhanced Network Intrusion Detection},
  author       = { Seth, Madhav and Pokhrel, Shiva Raj},
  year         = {2026},
  institution  = {Deakin University},
  note         = {Research prototype}
}
```

---

## Disclaimer

Q-Armor is developed for **academic research, experimentation, and defensive cybersecurity purposes**. Evaluation is performed using public cybersecurity datasets and controlled experimental environments.

## License

License information will be provided with the public release of the repository.
