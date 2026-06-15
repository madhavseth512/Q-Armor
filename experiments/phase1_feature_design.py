"""Phase 1 feature-design A/B/C experiment.

Tests whether *better engineering* (not just more raw columns) is the answer:

  A - "smart 8" : brief's 8, but the 2 weakest volumetric features
                  (flow_dispersion, urgent_activity) are swapped for a TIMING
                  feature (IAT) and an engineered BEHAVIOURAL feature
                  (handshake_ratio = ack_count / (syn_count + 1)).
                  -> Can a better-designed 8 capture web attacks WITHOUT more qubits?
  B - "engineered 12" : the 12, but ack_count/Variance replaced by engineered
                  combos (handshake_ratio, norm_dispersion = Variance/(AVG^2+1)).
                  -> Do engineered combos beat raw-picked features at 12 qubits?
  C - "raw 12"  : 8 + raw IAT, flow_duration, ack_count, Variance (= 0.759).

Same two-sided sampling; evaluated on the pristine validation set.
Run:  ./venv/Scripts/python.exe -m experiments.phase1_feature_design
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from agent import agent_config as config
from data.loader import load_dataset
from data.preprocess import (
    QuantumPreprocessor,
    dynamic_smote_strategy,
    majority_undersample_index,
)

RF_KWARGS = dict(n_estimators=150, max_depth=16, n_jobs=-1, random_state=config.RANDOM_SEED)
_PRE = QuantumPreprocessor()

# feature builders: raw df -> Series
F = {
    "traffic_rate":      lambda d: d["Rate"],
    "syn_activity":      lambda d: d["syn_count"],
    "teardown_activity": lambda d: d["rst_count"] + d["fin_count"],
    "header_overhead":   lambda d: d["Header_Length"],
    "avg_packet_size":   lambda d: d["AVG"],
    "flow_dispersion":   lambda d: d["Covariance"],
    "urgent_activity":   lambda d: d["urg_count"],
    "protocol_profile":  lambda d: _PRE._protocol_profile(d),
    "iat":               lambda d: d["IAT"],
    "flow_duration":     lambda d: d["flow_duration"],
    "ack_count":         lambda d: d["ack_count"],
    "variance":          lambda d: d["Variance"],
    "handshake_ratio":   lambda d: d["ack_count"] / (d["syn_count"] + 1),
    "norm_dispersion":   lambda d: d["Variance"] / (d["AVG"] ** 2 + 1),
}
# scaling per feature: 'standard' for AVG, 'none' for protocol (can be negative), else 'log1p'
SCALE = {"avg_packet_size": "standard", "protocol_profile": "none"}

BASE8 = ["traffic_rate", "syn_activity", "teardown_activity", "header_overhead",
         "avg_packet_size", "flow_dispersion", "urgent_activity", "protocol_profile"]
SMART8 = ["traffic_rate", "syn_activity", "teardown_activity", "header_overhead",
          "avg_packet_size", "iat", "handshake_ratio", "protocol_profile"]
ENG12 = BASE8 + ["iat", "flow_duration", "handshake_ratio", "norm_dispersion"]
RAW12 = BASE8 + ["iat", "flow_duration", "ack_count", "variance"]


def build(names, Xtr, Xva):
    Rtr = pd.DataFrame({n: F[n](Xtr) for n in names})
    Rva = pd.DataFrame({n: F[n](Xva) for n in names})
    log_cols = [n for n in names if SCALE.get(n, "log1p") == "log1p"]
    std_cols = [n for n in names if SCALE.get(n) == "standard"]
    Rtr[log_cols] = np.log1p(Rtr[log_cols]); Rva[log_cols] = np.log1p(Rva[log_cols])
    if std_cols:
        ss = StandardScaler().fit(Rtr[std_cols])
        Rtr[std_cols] = ss.transform(Rtr[std_cols]); Rva[std_cols] = ss.transform(Rva[std_cols])
    lo, hi = config.ANGLE_ENCODING_RANGE
    mm = MinMaxScaler(feature_range=config.ANGLE_ENCODING_RANGE).fit(Rtr[names])
    return mm.transform(Rtr[names]), np.clip(mm.transform(Rva[names]), lo, hi)


def run(names, Xtr, ytr, Xva, yva, strategy):
    Xtr_s, Xva_s = build(names, Xtr, Xva)
    Xr, yr = SMOTE(random_state=config.RANDOM_SEED, sampling_strategy=strategy).fit_resample(Xtr_s, ytr)
    rf = RandomForestClassifier(**RF_KWARGS).fit(Xr, yr)
    f1 = f1_score(yva, rf.predict(Xva_s), average="macro", labels=config.ATTACK_CLASSES)
    return f1, rf, Xva_s


def main() -> None:
    t0 = time.time()
    Xtr_full, ytr_full = load_dataset(config.TRAIN_CSV)
    Xva, yva = load_dataset(config.VALIDATION_CSV, validate=False)
    keep = majority_undersample_index(ytr_full)
    Xtr, ytr = Xtr_full.loc[keep], ytr_full.loc[keep]
    del Xtr_full, ytr_full
    strategy = dynamic_smote_strategy(ytr)
    print(f"setup done in {time.time()-t0:.1f}s")

    results = {}
    for tag, names in [("A smart-8 (8q)", SMART8), ("B engineered-12 (12q)", ENG12),
                       ("C raw-12 (12q)", RAW12)]:
        f1, rf, Xva_s = run(names, Xtr, ytr, Xva, yva, strategy)
        results[tag] = (f1, rf, Xva_s)
        print(f"  {tag:<24} macro-F1 = {f1:.4f}   ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 64)
    print("SUMMARY (validation macro-F1)")
    print("=" * 64)
    print(f"  brief 8 (8q)            0.4268   [reference]")
    for tag, (f1, _, _) in results.items():
        print(f"  {tag:<24} {f1:.4f}")
    print(f"  raw 45 (reference)      0.6754")

    best = max(results, key=lambda k: results[k][0])
    f1, rf, Xva_s = results[best]
    print(f"\nPer-class F1 for best: {best}")
    print(classification_report(yva, rf.predict(Xva_s),
                                labels=config.ATTACK_CLASSES, zero_division=0, digits=3))
    print(f"Total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
