# Q-Armor

**A Reflexive Quantum-Kernel Intrusion Detection Approach**

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format-B31B1B?style=for-the-badge)](docs/paper.tex)
[![Datasets](https://img.shields.io/badge/Datasets-NF--ToN--IoT%20%7C%20NF--UNSW--NB15-1F6FEB?style=for-the-badge)](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
[![Hardware](https://img.shields.io/badge/Validated%20on-IBM%20Quantum-6929C4?style=for-the-badge)](experiments/phase16_qpu_characterization.py)
[![License](https://img.shields.io/badge/License-MIT-2EA043?style=for-the-badge)](LICENSE)

## Overview

Q-Armor couples an 8-qubit fidelity quantum kernel with a deterministic, stateful episodic controller to detect and adapt to network intrusions across heterogeneous NetFlow domains, without retraining on the full target dataset. On a sealed held-out split it reaches **0.973 AUROC** within-domain and recovers cross-domain AUROC from **0.606 to 0.913** after adapting on 150 target-domain samples.

An equal-budget confirmatory study (10 seeds, matched target-label budgets from 0 to 300 samples) shows this recovery is **not** evidence of quantum advantage — classical baselines match or exceed it at the same budget, with lower variance. That finding, alongside a poisoning study, a natural-class-prevalence evaluation, and a live IBM Quantum run on `ibm_fez`, is reported in full in the paper.

## Architecture

<p align="center">
  <img src="docs/figures/fig1_overview.png" width="760" alt="Q-Armor architecture: inner detection pipeline and outer reflexive adaptation loop">
</p>

| Plane | Modules | Role |
|:--|:--|:--|
| **Data** | `perception/` `reasoning/` `action/` | 8-qubit `CyberSecurityFeatureMap` → `FidelityQuantumKernel` → `PegasosQSVC`, feeding a classical Detect→Type cascade |
| **Control** | `agent/` `planning/` `memory/` | Scores every 100-flow episode and fires one typed intervention |

Interventions are fixed rules, never a learned policy: `REINFORCE` · `SWITCH_MODEL` · `SWITCH_SUBSET` · `BINARY_ONLY`

## Reproduce

```bash
git clone https://github.com/madhavseth512/Q-Armor.git && cd Q-Armor
python -m venv venv && venv\Scripts\activate   # source venv/bin/activate on macOS/Linux
pip install -r requirements.txt

pytest tests/ -v                                    # full test suite
python -m experiments.phase8_final_eval --confirm   # reproduce the sealed results
python docs/generate_figures.py                     # regenerate every figure from current results
```

Datasets are not bundled (multi-GB, git-ignored) — download NF-ToN-IoT and NF-UNSW-NB15 and place them under `data/`.

## Authors

**Shiva Raj Pokhrel** (Deakin University, Chief Investigator) and **Madhav Seth** (IIT Kharagpur, Research Intern).

Academic research prototype — not intended for production deployment.
