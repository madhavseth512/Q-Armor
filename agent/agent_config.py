"""Centralised configuration for the Q-Armor agent system.

This module is the single source of truth for every tunable value in the
project: qubit counts, class definitions, severity weights, and all decision
thresholds. No numeric constant or class list may be hardcoded anywhere else.

Provenance convention — every value carries a trust tag in its inline comment:
    - grounded   : a library / literature standard default.
    - anchored   : tied to measured real-hardware characteristics (IBM devices).
    - fixed      : structural to the dataset or architecture (not tunable).
    - [VALIDATE] : a heuristic placeholder that MUST be calibrated empirically
                   before it is presented as a result.

See data/CONFIG_PROVENANCE.md for the full sourcing of every value below.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Quantum circuit  (fixed — structural)
# ---------------------------------------------------------------------------

# 8 qubits is a DESIGN CHOICE, not a free-tier limit. The IBM Open (free) plan
# exposes real devices up to 127 qubits (Eagle) and 156 (Heron r2), so 8 is not
# a hardware cap. It is fixed at 8 because: (1) classical simulation cost is 2^n
# = 256 amplitudes — trivial on a laptop, and quantum-kernel methods need O(N^2)
# circuit evaluations; (2) it maps one-to-one onto the 8 engineered features
# for interpretability; (3) fewer qubits => shallower, lower-noise circuits.
N_QUBITS: int = 8                      # fixed — see rationale above
N_QUANTUM_FEATURES: int = 8            # fixed — one engineered feature per qubit
FEATURE_MAP_REPS: int = 2              # fixed — CyberSecurityFeatureMap repetitions (CHANGE 4)

# CyberSecurityFeatureMap entanglement pairs (qubit indices), DERIVED from the
# smart-8 correlation matrix (experiments/phase4_entanglement.py) — the 4 strongest
# couplings, <=2 pairs per qubit, including the top feature flow_timing (q5):
#   q2-q3 teardown<->header (+0.70) · q3-q7 header<->protocol (-0.50)
#   q1-q7 syn<->protocol (-0.44)    · q2-q5 teardown<->flow_timing (-0.42)
# handshake_ratio (q6) is ~independent of all features (|r|<=0.11) -> single-qubit.
# Supersedes the brief's pairs (its pair C AVG<->Covariance is invalid; Covariance
# was dropped, and its pair D Rate<->protocol is only -0.07 in smart-8).
ENTANGLEMENT_PAIRS: list[tuple[int, int]] = [(2, 3), (3, 7), (1, 7), (2, 5)]
VQC_REPS: int = 2                      # fixed — RealAmplitudes ansatz repetitions
SHOTS: int = 1024                      # measurement shots per circuit evaluation

# ---------------------------------------------------------------------------
# Hardware & execution policy  (CHANGE 5)
# ---------------------------------------------------------------------------

# All development/testing runs on the local AerSimulator + FakeBackend noise
# models. Real IBM hardware (free Open plan: ~10 min QPU time per 28-day window)
# is gated behind this flag and only used after full simulator verification.
USE_REAL_HARDWARE: bool = False        # fixed OFF until 100% simulator-verified
IBM_BACKEND_NAME: str = "ibm_kingston" # Heron r2, available on the free Open plan

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

# Preference order the rule-based ModelSelector falls back through.
MODEL_HIERARCHY: list[str] = ["QSVM", "VQC", "QAE", "CLASSICAL"]

# ---------------------------------------------------------------------------
# Classification target  (fixed — CICIoT2023 dataset parameters, CHANGE 0)
# ---------------------------------------------------------------------------

# The 15 parent classes, derived from the `label` column via label.split('-')[0].
# The data loader MUST assert exactly these 15 are present and raise otherwise.
ATTACK_CLASSES: list[str] = [
    "DDoS",
    "DoS",
    "Mirai",
    "BenignTraffic",
    "Recon",
    "MITM",
    "DNS_Spoofing",
    "Backdoor_Malware",
    "VulnerabilityScan",
    "BrowserHijacking",
    "DictionaryBruteForce",
    "XSS",
    "SqlInjection",
    "CommandInjection",
    "Uploading_Attack",
]
N_PARENT_CLASSES: int = 15             # fixed — loader asserts len(unique parents) == this

# Tail-class strategy is deferred (CHANGE 0): kept as a config switch, not a
# structural assumption. False = keep all 15 classes; True = collapse rare
# classes into an "other" bucket. Decided later from SHAP / model results.
COLLAPSE_RARE_CLASSES: bool = False    # [VALIDATE] — revisit after SHAP (Phase 1)

# Hierarchical grouping (Option B) for the two-stage cascade classifier. The
# Stage-1 router predicts one of these 4 coarse groups; a Stage-2 specialist then
# classifies within the group. Grouping follows the attack taxonomy + the
# observed separability tiers (volumetric = easy, web = hard / data-limited).
ATTACK_GROUPS: dict[str, list[str]] = {
    "Benign":     ["BenignTraffic"],
    "Volumetric": ["DDoS", "DoS", "Mirai"],
    "Network":    ["Recon", "MITM", "DNS_Spoofing", "VulnerabilityScan"],
    "Web":        ["Backdoor_Malware", "BrowserHijacking", "DictionaryBruteForce",
                   "XSS", "SqlInjection", "CommandInjection", "Uploading_Attack"],
}
GROUP_NAMES: list[str] = list(ATTACK_GROUPS)                 # router output classes
GROUP_OF: dict[str, str] = {c: g for g, m in ATTACK_GROUPS.items() for c in m}

# ---------------------------------------------------------------------------
# Alert severity weights  (CHANGE 6)
# ---------------------------------------------------------------------------

# METHODOLOGY: each class maps to a representative CVSS v3.1 base score (0-10
# scale, FIRST.org) normalised to [0, 1] by dividing by 10. Alert priority is
# computed as confidence * severity_weight.
#
# CAVEAT: CVSS scores individual vulnerabilities (CVEs), not attack *categories*,
# so mapping a whole category to one score is itself a judgment call. The values
# below are reasonable initialisations, NOT final.
# [VALIDATE] — pending per-class CVSS mapping / supervisor (Dr. Pokhrel) input.
SEVERITY_WEIGHTS: dict[str, float] = {
    "DDoS":                 0.75,   # [VALIDATE] ~CVSS 7.5 (High) — availability impact
    "DoS":                  0.70,   # [VALIDATE] ~CVSS 7.0 (High)
    "Mirai":                0.90,   # [VALIDATE] ~CVSS 9.0 (Critical) — botnet/RCE
    "BenignTraffic":        0.0,    # fixed — non-attack class, no severity
    "Recon":                0.40,   # [VALIDATE] ~CVSS 4.0 (Medium) — info gathering
    "MITM":                 0.85,   # [VALIDATE] ~CVSS 8.5 (High) — confidentiality+integrity
    "DNS_Spoofing":         0.70,   # [VALIDATE] ~CVSS 7.0 (High)
    "Backdoor_Malware":     0.95,   # [VALIDATE] ~CVSS 9.5 (Critical) — full compromise
    "VulnerabilityScan":    0.40,   # [VALIDATE] ~CVSS 4.0 (Medium) — pre-attack probing
    "BrowserHijacking":     0.80,   # [VALIDATE] ~CVSS 8.0 (High)
    "DictionaryBruteForce": 0.75,   # [VALIDATE] ~CVSS 7.5 (High) — credential attack
    "XSS":                  0.60,   # [VALIDATE] ~CVSS 6.0 (Medium)
    "SqlInjection":         0.85,   # [VALIDATE] ~CVSS 8.5 (High) — data exfiltration
    "CommandInjection":     0.90,   # [VALIDATE] ~CVSS 9.0 (Critical) — RCE
    "Uploading_Attack":     0.85,   # [VALIDATE] ~CVSS 8.5 (High) — malicious upload
}

# Additive boost to an alert's priority when its source IP is a repeat offender.
REPEAT_IP_BOOST: float = 0.15          # [VALIDATE] — heuristic; tune in Phase 6
ALERT_LOG_PATH: str = "results/alerts.jsonl"   # structured JSON-lines alert log

# Rule-based defence recommendations, one per parent class (editable strings in
# one place). action = response steps, urgency = analyst priority, description =
# rationale. [VALIDATE] — playbook to be reviewed with the supervisor / SOC policy.
DEFENCE_RULES: dict[str, dict[str, str]] = {
    "DDoS":                 {"action": "Enable rate-limiting and upstream traffic scrubbing; alert NOC", "urgency": "high",     "description": "Volumetric flood degrading availability."},
    "DoS":                  {"action": "Rate-limit and block the offending source; alert NOC",           "urgency": "high",     "description": "Single-source denial of service."},
    "Mirai":                {"action": "Isolate the IoT device; force firmware re-image and audit",      "urgency": "critical", "description": "IoT botnet infection / takeover."},
    "BenignTraffic":        {"action": "No action — normal traffic",                                     "urgency": "none",     "description": "Legitimate traffic."},
    "Recon":                {"action": "Flag source; raise logging verbosity; monitor",                  "urgency": "low",      "description": "Reconnaissance / information gathering."},
    "MITM":                 {"action": "Run ARP/DNS integrity checks; isolate the affected segment",     "urgency": "high",     "description": "Man-in-the-middle interception."},
    "DNS_Spoofing":         {"action": "Validate DNS responses; flush resolver cache; enforce DNSSEC",   "urgency": "high",     "description": "DNS cache poisoning / redirection."},
    "Backdoor_Malware":     {"action": "Isolate host; capture forensics; escalate to SOC",               "urgency": "critical", "description": "Persistent backdoor / full compromise."},
    "VulnerabilityScan":    {"action": "Flag source; raise logging; review exposed services",            "urgency": "low",      "description": "Pre-attack vulnerability probing."},
    "BrowserHijacking":     {"action": "Isolate host; reset sessions and credentials",                   "urgency": "high",     "description": "Client/browser session compromise."},
    "DictionaryBruteForce": {"action": "Block source IP; enforce lockout and re-auth / MFA",             "urgency": "high",     "description": "Credential brute-force attempt."},
    "XSS":                  {"action": "Apply WAF rule; sanitise and output-encode inputs",              "urgency": "medium",   "description": "Cross-site scripting injection."},
    "SqlInjection":         {"action": "Apply WAF rule; parameterise queries; sanitise inputs",          "urgency": "high",     "description": "SQL injection / data exfiltration."},
    "CommandInjection":     {"action": "Apply WAF rule; isolate host; sanitise inputs",                  "urgency": "critical", "description": "OS command injection / RCE."},
    "Uploading_Attack":     {"action": "Quarantine uploaded artifacts; isolate host; escalate to SOC",   "urgency": "high",     "description": "Malicious file upload."},
}

# ---------------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------------

# Drift detection sensitivity: ADWIN's confidence parameter, an upper bound on
# the false-positive rate of drift detection. 0.002 = ~0.2% false-alarm bound.
# grounded — this is the default in river's ADWIN class, which implements
# Bifet & Gavalda, "Learning from Time-Changing Data with Adaptive Windowing"
# (SIAM SDM 2007).
ADWIN_DELTA: float = 0.002             # grounded (river / literature default)

# 2-qubit gate error rate above which Zero-Noise Extrapolation (ZNE) is applied.
# 0.01 = 1% error per 2-qubit gate. anchored — typical median 2-qubit (ECR/CNOT)
# gate error on current IBM superconducting devices (Eagle/Heron-class), exposed
# live via backend.properties(). Varies by device/day, so anchored not proven.
NOISE_THRESHOLD_ZNE: float = 0.01      # anchored (IBM device 2-qubit gate error)

# Readout (measurement) error rate above which readout is considered degraded.
# 0.03 = 3% chance a measured qubit is reported as the wrong value (0<->1).
# anchored — typical IBM readout/assignment error (~1-3% on current devices).
READOUT_ERROR_THRESHOLD: float = 0.03  # anchored (IBM device readout error)

# Noise level above which the agent abandons quantum models entirely and falls
# back to classical. 0.05 = 5%, set at ~5x the ZNE trigger to mark "too noisy to
# trust the quantum kernel even with mitigation."
# [VALIDATE] — heuristic; must be calibrated against the live backend's median
# error in Phase 6. The 5% value is a deliberate, conservative starting point,
# not a measured limit.
NOISE_THRESHOLD_FALLBACK: float = 0.05  # [VALIDATE] — calibrate vs backend (Phase 6)

# Mock backend noise for Phase 3 (no quantum backend yet; USE_REAL_HARDWARE=False).
# The PlanningModule.poll_noise() interface is real; these stub values are below
# the thresholds above, so on the classical-only path mitigation stays "none".
# Replaced by live backend.properties() polling in Phase 4/6.
MOCK_GATE_ERROR: float = 0.008         # mock 2-qubit gate error (< NOISE_THRESHOLD_ZNE)
MOCK_READOUT_ERROR: float = 0.02       # mock readout error (< READOUT_ERROR_THRESHOLD)

# Sliding-window mean prediction confidence below which the agent switches model.
# 0.65 = if average confidence drops under 65%, the current model is deemed
# unreliable. 0.65 chosen as a moderate placeholder — clearly above random for a
# 15-class problem (~0.067 chance level) yet below a "confident" >0.8 regime.
# [VALIDATE] — must come from a precision/recall sweep on the validation set in
# Phase 6; not yet empirically justified.
CONFIDENCE_THRESHOLD: float = 0.65     # [VALIDATE] — calibrate via P/R sweep (Phase 6)

# Number of recent predictions kept in the confidence monitor's sliding window.
CONFIDENCE_WINDOW_SIZE: int = 20       # [VALIDATE] — tune with CONFIDENCE_THRESHOLD

# ---------------------------------------------------------------------------
# Data pipeline  (fixed — dataset layout & preprocessing policy)
# ---------------------------------------------------------------------------

DATA_DIR: str = "data/CICIOT23"
TRAIN_CSV: str = f"{DATA_DIR}/train/train.csv"
TEST_CSV: str = f"{DATA_DIR}/test/test.csv"
VALIDATION_CSV: str = f"{DATA_DIR}/validation/validation.csv"

# Columns dropped at load time. Full-dataset EDA (data/FEATURE_ANALYSIS.md)
# REVISED the brief's 5-column drop list down to just the one genuinely
# degenerate column. The brief's other 4 targets are NOT constant and are kept
# as raw columns so the Phase-1 diagnostic baseline (RandomForest on all raw
# features) is a fair, tough benchmark:
#   Number — 96 distinct values (range 1-15), NOT constant
#   Weight — 102 distinct values (range 1-244), NOT constant
#   IAT    — huge within-row variance; only a WEAK class separator (not constant)
#   Std    — 0.76-correlated with AVG; simply not chosen among the 8 quantum
#            features (a feature-selection decision, not a global drop)
#   Drate  — mean ~3e-6 across 5.49M rows, effectively all-zero (degenerate)
DROP_COLUMNS: list[str] = ["Drate"]

# The 8 engineered quantum features, in qubit order (q0..q7). This is the
# "smart 8" set (data/FEATURE_ANALYSIS.md §9): the brief's volume-only 8 scored
# only 0.43 macro-F1 because all 8 measured one signal family. Swapping the 2
# weakest (flow_dispersion/Covariance, urgent_activity/urg_count) for a TIMING
# feature (flow_timing/IAT) and an engineered BEHAVIOUR feature (handshake_ratio)
# lifted it to 0.73 at the SAME 8 qubits — covering volume+timing+behaviour+protocol.
QUANTUM_FEATURE_NAMES: list[str] = [
    "traffic_rate",       # q0  Rate                       (volume)
    "syn_activity",       # q1  syn_count                  (volume)
    "teardown_activity",  # q2  rst_count + fin_count      (volume)
    "header_overhead",    # q3  Header_Length              (volume)
    "avg_packet_size",    # q4  AVG                        (size)
    "flow_timing",        # q5  IAT                        (timing)
    "handshake_ratio",    # q6  ack_count / (syn_count+1)  (behaviour)
    "protocol_profile",   # q7  weighted protocol score    (protocol)
]

# Weights for the engineered protocol_profile feature (q7). EDA-REVISED from the
# brief: Telnet and ARP are all-zero/near-zero in CICIoT2023 and were removed
# (their original 2.0 weights contributed nothing). is_gre is the dominant
# signal (Mirai 65.7% vs <1.5% elsewhere). [VALIDATE] — confirm/tune via SHAP.
PROTOCOL_PROFILE_WEIGHTS: dict[str, float] = {
    "is_gre": 3.0,    # [VALIDATE] Mirai signature: TCP+UDP+ICMP all zero (GRE)
    "ICMP":   1.5,    # [VALIDATE] DDoS — ICMP floods (0.225 vs ~0 elsewhere)
    "SSH":    1.0,    # [VALIDATE] DictionaryBruteForce (0.162 vs 0 elsewhere)
    "UDP":    1.0,    # [VALIDATE] UDP floods
    "DNS":    1.0,    # [VALIDATE] DNS spoofing
    "HTTPS": -1.0,    # [VALIDATE] legitimate-leaning (BenignTraffic 0.71)
}

APPLY_SMOTE: bool = True                # SMOTE on the TRAIN split only — never test
ANGLE_ENCODING_RANGE: tuple[float, float] = (0.0, float(np.pi))  # MinMax target for qubit angles
RANDOM_SEED: int = 42                   # reproducibility across splits / SMOTE / models

# Two-sided training-set sampling to tame the ~28,560:1 imbalance (D8), applied
# to the TRAINING split ONLY (validation/test stay pristine): undersample majority
# classes to MAJORITY_CAP, then SMOTE each minority up to a DYNAMIC floor =
# min(cap, SMOTE_MULTIPLIER * n_real). The multiplier bounds synthetic dominance
# so ultra-rare classes (e.g. Uploading_Attack, 140 rows) are not inflated ~70x
# into mostly-synthetic noise. [VALIDATE] — tune for the final model in Phase 2/6.
MAJORITY_CAP: int = 50_000              # [VALIDATE] max real rows kept per majority class
SMOTE_MULTIPLIER: int = 10              # [VALIDATE] minority target = min(cap, mult * n_real)

# ---------------------------------------------------------------------------
# Memory / persistence
# ---------------------------------------------------------------------------

MEMORY_DIR: str = "agent_state"
KERNEL_CACHE_MAX_SIZE: int = 1000       # max (x1, x2) pairs kept in the kernel cache

# ---------------------------------------------------------------------------
# Phase 6: NF-ToN-IoT and NF-UNSW-NB15 dataset configuration
# ---------------------------------------------------------------------------

NF_TON_CSV:  str = "data/NF-ToN-IoT/NF-ToN-IoT.csv"
NF_UNSW_CSV: str = "data/NF-UNSW-NB15/NF-UNSW-NB15.csv"

# The 8 NF features selected for the 8-qubit quantum model. Derived from
# RandomForest importance averaged across both datasets (experiments/phase6_eda.py).
# PROTOCOL and L7_PROTO dropped — lowest average importance (0.021 and 0.031).
# L4_DST_PORT is the highest-importance feature (0.25) contrary to the initial
# assumption; port numbers carry strong attack-type signal in IoT traffic.
NF_FEATURE_NAMES: list[str] = [
    "L4_DST_PORT",
    "OUT_BYTES",
    "L4_SRC_PORT",
    "FLOW_DURATION_MILLISECONDS",
    "IN_BYTES",
    "TCP_FLAGS",
    "OUT_PKTS",
    "IN_PKTS",
]

# Log-scale these columns before MinMax -> [0, pi]. All span multiple orders of
# magnitude; log1p prevents outliers from compressing 99% of the data near 0.
# L4_DST_PORT and L4_SRC_PORT added after EDA confirmed heavy-tail in NF-ToN-IoT
# (L4_DST_PORT p50=80, p99=59170, ratio 739x).
NF_LOG_COLS: list[str] = [
    "L4_DST_PORT",
    "L4_SRC_PORT",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
]

# Scaling mode for NF features: 'log' applies log1p to NF_LOG_COLS first,
# then MinMax all features to [0, pi]. EDA Part 4 confirmed log > linear:
# SVM-RBF cross-dataset AUROC 0.7426 (log) vs 0.6939 (linear).
NF_SCALE_MODE: str = "log"   # fixed — locked by experiments/phase6_eda.py

# NF binary label column (0=benign, 1=attack) and multi-class attack column.
NF_LABEL_COL:  str = "Label"
NF_ATTACK_COL: str = "Attack"

# Five-class coarse taxonomy shared across NF-ToN-IoT and NF-UNSW-NB15.
# Maps each dataset's native attack-type strings to a common label space so
# cross-dataset multi-class classification is possible (E3 / Phase 6).
# Derivation rationale per class:
#   DoS        — volumetric/availability attacks (ddos, dos, DoS, Generic*)
#   Injection  — exploit/credential/code injection (injection, xss, password,
#                Exploits, Generic*, Shellcode)
#   Recon      — probing/enumeration (scanning, Fuzzers, Reconnaissance, Analysis)
#   Backdoor   — persistence/exfil/lateral (backdoor, mitm, ransomware,
#                Backdoor, Worms)
#   Benign     — normal traffic
# *Generic (UNSW-NB15): generic block-cipher attacks — mapped to DoS (availability
#  impact; not an injection or recon primitive).
NF_COARSE_TAXONOMY: dict[str, str] = {
    # NF-ToN-IoT native labels
    "Benign":        "Benign",
    "injection":     "Injection",
    "ddos":          "DoS",
    "password":      "Injection",
    "xss":           "Injection",
    "scanning":      "Recon",
    "dos":           "DoS",
    "backdoor":      "Backdoor",
    "mitm":          "Backdoor",
    "ransomware":    "Backdoor",
    # NF-UNSW-NB15 native labels
    "Exploits":      "Injection",
    "Fuzzers":       "Recon",
    "Reconnaissance":"Recon",
    "Generic":       "DoS",
    "DoS":           "DoS",
    "Analysis":      "Recon",
    "Backdoor":      "Backdoor",
    "Shellcode":     "Injection",
    "Worms":         "Backdoor",
}

NF_COARSE_CLASSES: list[str] = ["Benign", "DoS", "Injection", "Recon", "Backdoor"]

# ---------------------------------------------------------------------------
# Phase 5: quantum model training parameters
# ---------------------------------------------------------------------------

# Kernel-subset size for binary QSVC + SVM-RBF comparison (E6).
# Phase-4 timing: n=90→1min, n=195→5min. 150 is a comfortable training budget.
QSVC_SUBSET_SIZE: int = 150            # fixed — calibrated from Phase-4 timing

# PegasosQSVC stochastic update steps. Default in qiskit_ml; adequate at n=150.
PEGASOS_TAU: int = 100                 # grounded (qiskit_ml default)

# VQC COBYLA max iterations. Typically converges in 50–80 for small binary sets.
VQC_MAX_ITER: int = 500                # [VALIDATE] — 100 did not converge (24 params); 500 = ~20x params

# QAE ansatz: RealAmplitudes repetitions (8 qubits, reps=1 → 16 params).
QAE_REPS: int = 1                      # fixed — depth/parameter tradeoff at 8 qubits

# QAE trash qubits: the first QAE_N_TRASH qubits are the reconstruction target.
QAE_N_TRASH: int = 4                   # fixed — half of N_QUBITS = 8

# QAE COBYLA max iterations for unsupervised reconstruction loss optimisation.
QAE_MAX_ITER: int = 200                # [VALIDATE] — 30 did not converge; 200 = ~12x params (16)

# QAE shots per circuit evaluation (fidelity estimation; higher = more accurate).
# 256 shots introduced too much objective noise for COBYLA; 1024 gives ~3% std.
QAE_SHOTS: int = 1024                  # [VALIDATE] — balance speed vs fidelity variance
