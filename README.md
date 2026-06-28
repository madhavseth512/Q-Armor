# Q-Armor: Agentic Quantum-Enhanced Network Intrusion Detection

Q-Armor is a research prototype developed at Deakin University under the supervision of Dr. Shiva Raj Pokhrel. It frames network intrusion detection as a two-loop agentic system: an **inner loop** that processes each network flow through five modules (Perception → Reasoning → Planning → Action → Memory), and an **outer Reflexion loop** that evaluates episode-level performance, reflects on failure modes, and autonomously mutates the model-selection policy using episodic memory.

The system combines an **8-qubit quantum fidelity kernel** (CyberSecurityFeatureMap) with a **Detect→Type cascade** and a **SWITCH_SUBSET** cross-domain adaptation mechanism. It is validated on two large-scale public NetFlow datasets across sealed test splits.

---

## Key Results

| Metric | Value |
|--------|-------|
| Within-domain S1 AUROC (sealed test) | **0.955** |
| Within-domain S2 typing macro-F1 | **0.723** |
| Cross-domain AUROC before retraining (T2) | 0.577 |
| Cross-domain AUROC after SWITCH_SUBSET (T3) | **0.730** (+0.153) |
| FPR@95 after SWITCH_SUBSET | 0.554 (↓ from 0.998) |
| AerSimulator QPU validation AUROC | 0.880 |
| Reflexion episodes simulated | 10 (Block A: within, Block B: cross-domain) |
| SWITCH_SUBSET triggered autonomously at | Episode 6 (drift confirmed by ADWIN) |

---

## System Architecture

### Inner Loop (per-flow)

| Module | Responsibility |
|--------|----------------|
| `perception/` | `CyberSecurityFeatureMap` — 8-qubit angle encoding with 4 correlation-derived entanglement pairs; `FidelityQuantumKernel` (Qiskit SamplerV2); log1p + MinMax scaling to [0, π] |
| `reasoning/` | `PegasosQSVC` binary classifier (n=150 k-means subset, τ=100, C=1.0); `DetectTypeCascade` — Stage 1 binary + Stage 2 RF attack typer; `predict_labels()` interface for margin-based hard prediction |
| `planning/` | ADWIN drift detector (δ=0.002); confidence monitor; `SubsetRetrainer` — k-means subset reselection + QSVC retraining (~85 s) |
| `action/` | Alert generation, severity scoring, defence recommendations, JSON/CLI output |
| `memory/` | Model persistence, kernel cache, `EpisodicMemory` — append-only JSONL episode log |

### Outer Reflexion Loop (per-episode, N_ep = 100 flows)

| Component | Role |
|-----------|------|
| `CascadeEvaluator` | Aggregates predictions into `EpisodeReport` (AUROC, F1, macro-F1, AUPR, FPR@95, drift flag) |
| `SelfReflector` | Applies 4 ordered heuristic rules → `Lesson` (BINARY_ONLY / SWITCH_SUBSET / SWITCH_MODEL / REINFORCE) |
| `EpisodicMemory` | Appends `(EpisodeReport, Lesson)` to `agent_state/episodes.jsonl`; mutates model-selection policy |

---

## Datasets

Both datasets share an 8-feature NetFlow schema enabling direct cross-dataset evaluation.

| Dataset | Rows | Attack % | Train | Test |
|---------|------|----------|-------|------|
| NF-ToN-IoT | 1,379,274 | 80.4% | 1,103,419 | 275,855 |
| NF-UNSW-NB15 | 1,623,118 | 4.5% | 1,298,494 | 324,624 |

**Features (8):** `L4_DST_PORT`, `OUT_BYTES`, `L4_SRC_PORT`, `FLOW_DURATION_MILLISECONDS`, `IN_BYTES`, `TCP_FLAGS`, `OUT_PKTS`, `IN_PKTS` — log1p transformed then MinMax scaled to [0, π].

**5-class coarse taxonomy:** Benign / DoS / Injection / Recon / Backdoor — spanning both datasets' native label spaces.

Expected local layout (git-ignored, multi-GB):

```
data/
├── NF-ToN-IoT/NF-ToN-IoT.csv
└── NF-UNSW-NB15/NF-UNSW-NB15.csv
```

---

## Quantum Kernel

The **CyberSecurityFeatureMap** encodes 8 flow features into an 8-qubit parameterised circuit using angle encoding (R_y gates). Entanglement is applied via CNOT gates on four pairs derived from the NF-ToN-IoT feature correlation matrix:

| Pair | Correlation |
|------|-------------|
| (q2, q3) | +0.70 (teardown–header) |
| (q3, q7) | −0.50 (header–protocol) |
| (q1, q7) | −0.44 (SYN–protocol) |
| (q2, q5) | −0.42 (teardown–flow_timing) |

Circuit repeated r=2 times (FEATURE_MAP_REPS). Kernel evaluated via `FidelityQuantumKernel` + `SamplerV2`. PSD verified (min eigenvalue > 0).

---

## Repository Structure

```
Q-Armor/
├── agent/                  # AgentConfig, AgentCore, CascadeEvaluator, SubsetRetrainer, SelfReflector
├── perception/             # CyberSecurityFeatureMap, FidelityQuantumKernel
├── reasoning/              # BaseClassifier, PegasosQSVCModel, DetectTypeCascade
├── memory/                 # State persistence, kernel cache, EpisodicMemory
├── planning/               # ADWIN drift detector, confidence monitor
├── action/                 # Alert, severity, defence recommendation
├── data/                   # Dataset loaders and preprocessing
├── experiments/            # Phase experiment scripts (phase4_*.py … phase8_*.py)
├── results/                # Per-phase JSON metrics and plots
│   ├── phase7/             # Reflexion simulation outputs
│   └── phase8/             # Cascade validation (E8a), QPU (E8b), sealed eval (E8c)
├── agent_state/            # episodes.jsonl — Reflexion episode log
├── docs/
│   ├── PROJECT_CHARTER.md  # Source of truth: vision, decisions, phases
│   ├── ROADMAP.md          # Supervisor-expanded scope (Reflexion, multi-dataset)
│   ├── PHASE7_RESULTS.md   # Reflexion simulation results
│   ├── PHASE8_RESULTS.md   # Final sealed evaluation results
│   ├── PRODUCTION_ROADMAP.md # Prototype-to-production transition plan
│   └── paper.tex           # IEEE conference paper (LaTeX)
└── tests/                  # Pytest unit tests (19 cascade tests, 24 Reflexion tests)
```

---

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Foundation, config, project structure | ✅ |
| 1 | Data pipeline — NetFlow features, stratified splits, validation gates | ✅ |
| 2 | Classical baselines (RF, SVM-RBF); cross-dataset evaluation | ✅ |
| 3 | Full agent skeleton (5 modules, inner loop) | ✅ |
| 4 | Quantum perception — CyberSecurityFeatureMap, PSD-verified fidelity kernel | ✅ |
| 5 | Quantum model zoo — PegasosQSVC, VQC, QAE on binary CICIoT2023 | ✅ |
| 6 | Multi-dataset integration — NF-ToN-IoT + NF-UNSW-NB15 cross-domain eval | ✅ |
| 7 | Reflexion outer loop — Evaluator, SelfReflector, EpisodicMemory (10-episode sim) | ✅ |
| 8 | Detect→Type cascade, SWITCH_SUBSET retraining, sealed final evaluation | ✅ |

---

## Reflexion Policy Rules

The `SelfReflector` applies four ordered rules each episode:

1. **BINARY_ONLY** — if model name contains "multiclass" and macro-F1 < 0.30, disable typing
2. **SWITCH_SUBSET** — if AUROC < 0.70 **and** ADWIN fired: trigger k-means reselection + QSVC retrain on target-domain samples
3. **SWITCH_MODEL** — if AUROC < 0.70 and no drift: demote model tier in [QSVM → VQC → QAE → CLASSICAL]
4. **REINFORCE** — if AUROC ≥ 0.70 for 3 consecutive episodes: lock current policy tier

In the 10-episode simulation: REINFORCE ×3 (episodes 2–4), SWITCH_SUBSET ×1 (episode 6), SWITCH_MODEL ×4 (episodes 5, 7, 8, 9).

---

## Setup

### 1. Clone and create environment

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

All experiments run on local `AerSimulator` (ideal, noiseless). Real IBM hardware is gated behind `IBM_HARDWARE_MODE = False` in `agent/agent_config.py`.

### 4. Place datasets

Download NF-ToN-IoT and NF-UNSW-NB15 from the NetFlow repository and place them as shown in the dataset section above.

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Research Paper

An IEEE conference paper documenting this work is available at [`docs/paper.tex`](docs/paper.tex).

**Title:** Q-Armor: Agentic Quantum-Enhanced Network Intrusion Detection with Reflexion-Based Adaptive Policy Learning

**Authors:** Madhav Seth (IIT Kharagpur) and Shiva Raj Pokhrel (Deakin University)

---

## Research Context

- **Supervisor:** Dr. Shiva Raj Pokhrel, School of Information Technology, Deakin University
- **Institution:** Indian Institute of Technology, Kharagpur (Madhav Seth) / Deakin University (Dr. Pokhrel)
- **Quantum stack:** Qiskit — `FidelityQuantumKernel`, `SamplerV2`, `PegasosQSVC`, `AerSimulator`
- **Key techniques:** quantum fidelity kernel, ADWIN drift detection, Reflexion episodic memory, k-means subset selection, Detect→Type cascade
