# Q-Armor — Feature Analysis (Phase 1 EDA)

> Evidence trail for the data pipeline. All figures below are from
> `experiments/eda_verification.py` run on the **full** `train.csv`
> (**5,491,971 rows**, 0 NaN, 0 inf). This document verifies — and where
> necessary corrects — the claims of the original revamp brief, which were
> derived from only a ~50-row sample. SHAP results (Gate A) are appended later.

---

## 1. Drop-column verification

| Column | Brief claim | Full-data reality | Decision |
|---|---|---|---|
| `Drate` | zero everywhere | 303 distinct, but mean 3e-6 — effectively all-zero | **drop** (degenerate) |
| `Number` | constant 9.5 | 96 distinct, range 1–15, std 0.82 — **not constant** | **keep** for baseline |
| `Weight` | constant 141.55 | 102 distinct, range 1–244, std 21 — **not constant** | **keep** for baseline |
| `IAT` | near-constant, no signal | per-class means 79.8M–93.1M (tight) → weak separator | **keep** for baseline |
| `Std` | 0.76-corr with AVG | corr = **0.7639** ✅ | **keep** raw; not a quantum feature |

**Outcome:** `DROP_COLUMNS = ["Drate"]`. Only the genuinely degenerate column is
dropped. The other four are retained as raw columns so the Gate-B diagnostic
baseline (RandomForest on all surviving raw features) is a fair, hard benchmark.
`Std` is excluded from the 8 *quantum* features (to avoid spending a qubit on its
0.76 redundancy with `AVG`) — a feature-selection choice, not a global drop.

---

## 2. The 8 engineered quantum features

| q | Feature | Source | Scaling | EDA evidence |
|---|---|---|---|---|
| 0 | `traffic_rate` | `Rate` | log1p → MinMax(0,π) | skew **23.3** (max 8.39M ≫ 99th pct 123k); separates DoS/DDoS/Mirai (8k–11k) from others |
| 1 | `syn_activity` | `syn_count` | log1p → MinMax | skew 2.2; weak at class-mean level (all <1.5) — **SHAP watch** |
| 2 | `teardown_activity` | `rst_count + fin_count` | log1p → MinMax | `rst_count` is a **top discriminator** (MITM 1404, Benign 1073 vs DDoS 0.76); `fin_count` ~0 (rst dominates) |
| 3 | `header_overhead` | `Header_Length` | log1p → MinMax | skew 9.7; **strong** separator (MITM 2.0M, Benign 1.0M vs DDoS 7k) |
| 4 | `avg_packet_size` | `AVG` | StandardScaler → MinMax | separates MITM 851 / Benign 620 vs DDoS 78 / DoS 70 |
| 5 | `flow_dispersion` | `Covariance` | log1p → MinMax | skew **53.0**; Benign 641k / MITM 585k vs DDoS 10k / DoS 5k |
| 6 | `urgent_activity` | `urg_count` | log1p → MinMax | separates MITM 362; **0.70 corr with `rst_count`** — kept, SHAP to judge (swap → `Radius`) |
| 7 | `protocol_profile` | weighted protocol score | MinMax | see §4 |

**Scaling pipeline (leak-free):** engineer → `log1p` on {0,1,2,3,5,6} →
`StandardScaler` on {4} (fit on train) → `MinMaxScaler(0, π)` on all 8 (fit on
train) → **clip to (0, π)** on val/test. SMOTE on train only. Scalers + SMOTE are
fitted on `train.csv` only and `transform`-applied to `validation.csv` / `test.csv`.

---

## 3. Entanglement-pair correlations (feature-map design, Phase 4)

All confirmed on full data — the brief's pair choices hold:

| Pair | Columns | Brief | Full data | Verdict |
|---|---|---|---|---|
| A | `syn_count` ↔ `rst_count` | decorrelated | 0.05 | ✅ independent signals |
| B | `rst_count` ↔ `Header_Length` | 0.75 | **0.75** | ✅ exact |
| C | `AVG` ↔ `Covariance` | 0.50 | **0.50** | ✅ exact |
| D | `Rate` ↔ protocol | intensity×identity | `Rate` ~0 corr w/ all | ✅ independent intensity axis |

*New observation:* `urg_count` ↔ `rst_count` = 0.70 (not flagged by the brief) —
tracked as the reason feature 6 is a SHAP watch item.

---

## 4. `protocol_profile` — EDA-revised formula

The brief's formula leaned on `Telnet` (2.0) and `ARP` (2.0). **Both are dead in
CICIoT2023:** `Telnet` = 0.000 in every class; `ARP` ≤ 0.003. Their terms
contributed nothing. Revised, evidence-based formula (weights `[VALIDATE]`, in
`agent_config.PROTOCOL_PROFILE_WEIGHTS`):

```
protocol_profile = 3.0*is_gre + 1.5*ICMP + 1.0*SSH + 1.0*UDP + 1.0*DNS - 1.0*HTTPS
where is_gre = 1 if (TCP + UDP + ICMP == 0) else 0
```

Evidence per term (per-class means / fractions):

| Term | Signal | Evidence |
|---|---|---|
| `is_gre` | **Mirai** | 65.7% of Mirai rows vs DDoS 1.4%, DoS 0.04% — strongest single protocol signal |
| `ICMP` | DDoS | 0.225 for DDoS vs ~0 elsewhere (ICMP floods) |
| `SSH` | DictionaryBruteForce | 0.162 vs 0 for every other class |
| `UDP` / `DNS` | flood / spoof context | UDP elevated for DoS 0.41, DNS_Spoofing 0.36 |
| `HTTPS` (−) | legitimate-leaning | BenignTraffic 0.71, MITM 0.66 — penalised |

`Telnet`, `ARP` removed (dead). SHAP will confirm or tune the remaining weights.

---

## 5. Open item — SMOTE at scale (D8)

True imbalance is **≈ 28,560:1** (DDoS 3,998,500 vs Uploading_Attack 140).
Naively SMOTE-balancing all 15 classes to DDoS's level ≈ 60M synthetic rows —
infeasible. The realistic strategy is *undersample majorities + SMOTE minorities
to a target ratio*, with the target to be decided before any full-scale run.
Feature engineering and scaling are independent of this and proceed now.

---

## 6. Validation gates — RESULTS

Sampling (train only): undersample majorities to 50,000; SMOTE minorities to a
dynamic floor `min(50k, 10×n_real)`. Ultra-rare classes capped at 90% synthetic
(10×) instead of being inflated ~70×. RandomForest (150 trees, depth 16);
evaluated on the **pristine validation set** (1,176,851 rows). Test untouched.

### Gate A — SHAP: ✅ PASS
Every one of the 8 features is important for ≥1 of the 15 classes (mean |SHAP|
per feature's strongest class, with that class):

| Feature | max mean\|SHAP\| | strongest class | row total |
|---|---|---|---|
| `teardown_activity` | 0.0564 | DoS | 0.3427 |
| `avg_packet_size` | 0.0650 | Mirai | 0.3084 |
| `header_overhead` | 0.0446 | MITM | 0.2865 |
| `syn_activity` | 0.0490 | Recon | 0.2619 (watch item — **earns its qubit**) |
| `protocol_profile` | 0.0435 | MITM | 0.2560 |
| `traffic_rate` | 0.0338 | MITM | 0.2184 |
| `urgent_activity` | 0.0275 | BenignTraffic | 0.1906 (weakest-but-valid) |
| `flow_dispersion` | 0.0261 | VulnerabilityScan | 0.1655 (weakest) |

`syn_activity` (a watch item) is clearly useful (Recon/VulnerabilityScan).
`urgent_activity` and `flow_dispersion` are the two weakest but stay above the
10%-of-top threshold — consistent with D7's `urg_count`↔`rst_count` redundancy note.

### Gate B — diagnostic baseline: ❌ REVISIT (large miss)

| Model | Features | Val macro-F1 |
|---|---|---|
| RandomForest | all 45 raw | **0.6754** |
| RandomForest | 8 engineered | **0.4268** |
| | **Gap** | **0.2486** (threshold 0.07) |

RandomForest is essentially invariant to monotonic scaling, so this gap is
**genuine information loss from 45 → 8 features**, not an artifact of the
angle-encoding pipeline.

**Where the signal is lost** — per-class F1 (8 features):

| Class group | Classes | 8-feat F1 | Read |
|---|---|---|---|
| Volumetric | DDoS .889, Mirai .994, DoS .655, Benign .826 | strong | the 8 flow features separate these well |
| Network mid | MITM .680, DNS_Spoofing .450, Recon .543, VulnScan .357 | partial | |
| **Application-layer** | **XSS .000, Uploading .000, SqlInjection .086, Backdoor .115, DictBruteForce .195, BrowserHijack .299, CmdInjection .314** | **collapse** | the 8 flow features are **blind** to web/payload attacks |

**Interpretation:** all 8 engineered features are *volumetric/flow* signals. They
excel at flood-type attacks but cannot distinguish application-layer attacks
(XSS, SQLi, CommandInjection, Uploading, Backdoor), which have similar flow
profiles and differ in payload/protocol features absent from the 8. The raw set
recovers ~25 macro-F1 points from columns the engineering discarded.

**Consequence (per charter rule):** features must be revisited before quantum
work, OR the scope/target re-framed (D3 tail-class policy). Options under review:
(a) collapse ultra-rare application-layer classes into "other" and report
Q-Armor as a volumetric-attack detector; (b) redesign 1–2 of the 8 features
(swap the weakest — `flow_dispersion`/`urgent_activity`) using raw-feature
importances to recover web-attack signal within the 8-qubit budget; (c) revisit
the 8-feature constraint. Decision pending (see PROJECT_CHARTER §8).

## 7. Raw per-class SHAP diagnostic (web-attack classes)

Per-class SHAP on the 45-feature RF, for the 5 collapsed classes. Aggregate
|SHAP| of **new** raw columns (not already among the 8 feature sources):

| New raw column | Summed \|SHAP\| over 5 web classes |
|---|---|
| **`IAT`** | **0.0583** (clear #1) |
| `flow_duration` | 0.0195 |
| `ack_count` | 0.0125 |
| `Weight` | 0.0069 |
| `Number` | 0.0067 |
| `HTTP` | 0.0026 |
| `Variance` | 0.0022 |

**Two honest readings:**

1. **`IAT` is the standout web-attack signal — vindicating D5.** The brief wanted
   to drop `IAT` as "near-constant, no signal"; in fact it is the single most
   useful column for XSS, SqlInjection, CommandInjection, Uploading, Backdoor.
   Keeping it (D5) was correct. `flow_duration` and `ack_count` follow.

2. **But the signal is weak and smeared, not concentrated.** The *absolute*
   magnitudes are small — `IAT`'s best per-class |SHAP| is ~0.017
   (CommandInjection), roughly a quarter of the volumetric features' importances
   (`avg_packet_size` 0.065, `teardown` 0.056). No column or pair cleanly
   separates the web attacks; even the full 45-feature model only reached 0.675
   macro-F1. These classes are intrinsically hard from flow features.

**Implication for option (b):** closing Gate B (gap 0.25 → ≤0.07) would require
the 8-feature macro-F1 to rise ~0.18. Swapping the 2 weakest features for
`IAT`/`flow_duration`/`ack_count`-based features can realistically add only a few
points (small SHAP magnitudes), so **(b) alone will not pass Gate B.** The
evidence supports **(a): collapse the ultra-rare application-layer classes...**

> ⚠️ **This prediction was WRONG — see §8.** The measurement overturned it.

## 8. 12-feature measurement — Gate B PASSES, all 15 classes viable

Adding 4 features (`IAT`, `flow_duration`, `ack_count`, `Variance`) to the 8
engineered features, each log1p → MinMax(0, π). Same sampling, RF, pristine val:

| Feature set | Val macro-F1 |
|---|---|
| 8 engineered | 0.4268 |
| **12 (8 + 4 extras)** | **0.7593** |
| 45 raw (prior "ceiling") | 0.6754 |

**12 curated features beat both the 8-feature set (+0.33) AND the 45-raw
baseline (+0.08).** Gate B passes outright (engineered ≥ raw). Per-class, the
collapse is gone — every class is now viable:

| Tier | Classes (12-feat F1) |
|---|---|
| Volumetric | DDoS .999, DoS .999, Mirai .998, Benign .927 |
| Network mid | Recon .823, MITM .862, VulnScan .840, DNS_Spoofing .732 |
| Application-layer | CmdInjection .711, Backdoor .699, BrowserHijack .705, SqlInjection .591, DictBruteForce .568, XSS .553, Uploading .383 |

**Why 12 curated > 45 raw:** RF feature-subsampling dilutes good features among
45 mostly-redundant columns; the 8 engineered features also inject domain
knowledge raw columns lack (log-scaling, `teardown = rst+fin`, `is_gre`,
weighted `protocol_profile`). Curated low-D beats noisy high-D — a known effect.

**`IAT` is the linchpin** — the column the brief wanted to drop is the single
biggest contributor to recovering the application-layer classes. Strongest
vindication yet of "verify the brief, don't trust it."

**Consequence:** the data supports **keeping all 15 classes** via a ~12-feature /
12-qubit design (reopens D4), NOT collapsing. 12 qubits remains simulable
(2^12 = 4,096 amplitudes) with only mild quantum-side cost. Open follow-up: an
ablation to find the *minimal* feature/qubit count (9–12) that retains most of
0.759, to add the fewest qubits necessary. NOTE: 0.759 is the **RandomForest**
ceiling for 12 features; the quantum models are measured against it and may match
or trail it (no expected quantum F1 advantage on tabular flow data).

## 9. Feature-design A/B/C — the problem was the *wrong* 8, not too few

The root cause was that the brief's 8 features were all **one signal family
(volume/size)**. Replacing the 2 weakest (`flow_dispersion`, `urgent_activity`)
with a **timing** feature (`IAT`) and an **engineered behavioural** feature
(`handshake_ratio = ack_count/(syn_count+1)`) gives signal-family coverage —
volume + timing + behaviour + protocol.

| Experiment | Qubits | Macro-F1 |
|---|---|---|
| brief 8 (volume-only) | 8 | 0.4268 |
| **A — smart 8** | **8** | **0.7333** |
| B — engineered 12 | 12 | 0.7503 |
| C — raw 12 | 12 | 0.7593 |
| raw 45 (reference) | — | 0.6754 |

**Decisive findings:**
1. **The smart 8 jumps +0.31 (0.427 → 0.733) at the SAME 8 qubits**, and beats
   the raw-45 baseline. The fix was feature *design*, not feature *count*.
2. **8 → 12 qubits buys only +0.02–0.03** (0.733 → 0.759). Not worth the
   quantum-side cost (kernel concentration, barren plateaus, feature-map redesign).
   **D4 can stay at 8 qubits.**
3. **Engineered combos did NOT beat raw** (B 0.750 < C 0.759): `handshake_ratio`
   is valuable *replacing a weak feature*, but `norm_dispersion` ≤ raw `Variance`.
   Confirms: engineering is not inherently better — **signal per qubit** is what
   matters, raw or engineered.

**Recommended feature set — "smart 8" (8 qubits, all 15 classes):**

| q | Feature | Source | Family |
|---|---|---|---|
| 0 | `traffic_rate` | `Rate` | volume |
| 1 | `syn_activity` | `syn_count` | volume |
| 2 | `teardown_activity` | `rst_count + fin_count` | volume |
| 3 | `header_overhead` | `Header_Length` | volume |
| 4 | `avg_packet_size` | `AVG` | size |
| 5 | `flow_timing` | `IAT` | **timing (NEW)** |
| 6 | `handshake_ratio` | `ack_count/(syn_count+1)` | **behaviour (NEW)** |
| 7 | `protocol_profile` | weighted protocol score | protocol |

Replaces `flow_dispersion` (`Covariance`) and `urgent_activity` (`urg_count`).
Note: this **invalidates entanglement pair C** (`AVG↔Covariance`) — pairs must be
re-derived for the smart-8 set in Phase 4. 0.733 is the RandomForest ceiling;
quantum models are measured against it.

## 10. FINAL Phase-1 gates on smart-8 — BOTH PASS ✅

Re-ran both gates with the adopted smart-8 feature set (config + preprocess.py).

### Gate B — PASS
| Model | Val macro-F1 |
|---|---|
| RandomForest, 45 raw | 0.6754 |
| RandomForest, **smart-8** | **0.7333** |
| Gap (raw − 8eng) | **−0.0579** (smart-8 *beats* raw; threshold 0.07) |

### Gate A — PASS (every feature matters for ≥1 class)
| Feature | max mean\|SHAP\| | strongest class | row total |
|---|---|---|---|
| **`flow_timing`** (IAT, NEW) | **0.0800** | DoS | **0.6822** (now the strongest feature) |
| `teardown_activity` | 0.0476 | DDoS | 0.3129 |
| `header_overhead` | 0.0355 | MITM | 0.2574 |
| `avg_packet_size` | 0.0498 | DDoS | 0.2408 |
| `protocol_profile` | 0.0308 | Mirai | 0.1851 |
| `syn_activity` | 0.0389 | Recon | 0.1774 |
| `handshake_ratio` (NEW) | 0.0503 | VulnerabilityScan | 0.1390 |
| `traffic_rate` | 0.0271 | MITM | 0.1378 |

`flow_timing` (the new timing feature) is now the **single most important
feature** — fully vindicating the swap. `handshake_ratio` earns its qubit via
VulnerabilityScan/SqlInjection. All 8 clear the threshold.

**Phase 1 outcome:** smart-8 (8 qubits) detects all 15 classes; both validation
gates pass. Feature pipeline is locked. Decisions resolved — D4: stay at 8
qubits; D3: keep all 15 classes (no collapse); D8: two-sided dynamic-floor
sampling; D9: feature sufficiency achieved via design, not count. Remaining
`[VALIDATE]` items (protocol weights, thresholds) deferred to later phases.
