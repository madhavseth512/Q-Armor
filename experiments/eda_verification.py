"""Full-dataset EDA verification for Q-Armor Phase 1.

Re-checks every quantitative claim the revamp brief made from a ~50-row sample,
against the FULL CICIoT2023 train.csv. Prints a section per claim so each can be
marked verified / corrected. Run:

    ./venv/Scripts/python.exe experiments/eda_verification.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agent import agent_config as config

# Only load the columns the EDA needs — keeps 5.5M rows light/fast.
DROP_CHECK = ["Drate", "Number", "Weight", "IAT", "Std"]
FEATURE_SRC = [
    "Rate", "syn_count", "rst_count", "fin_count",
    "Header_Length", "AVG", "Covariance", "urg_count",
]
PROTOCOL = ["Telnet", "ARP", "ICMP", "UDP", "DNS", "HTTPS", "SSH", "TCP", "HTTP"]
SWAP_CANDIDATES = ["ack_count", "Radius", "Magnitue", "Tot size"]
USECOLS = sorted(set(DROP_CHECK + FEATURE_SRC + PROTOCOL + SWAP_CANDIDATES + ["label"]))


def _hr(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main() -> None:
    import time
    t0 = time.time()
    df = pd.read_csv(config.TRAIN_CSV, usecols=USECOLS)
    df["parent"] = df["label"].str.split("-").str[0]
    print(f"Loaded {len(df):,} rows x {df.shape[1]} cols in {time.time()-t0:.1f}s")

    # ---- 1. Are the 5 'dead' drop columns actually dead? ---------------------
    _hr("1. DROP-COLUMN VERIFICATION  (brief: Drate=0, Number=9.5, Weight=141.55, IAT~const, Std redundant)")
    summary = df[DROP_CHECK].agg(["nunique", "min", "max", "mean", "std"]).T
    print(summary.to_string())
    print("\nPer-class mean of IAT (brief: near-constant across ALL classes):")
    print(df.groupby("parent")["IAT"].mean().round(2).sort_values().to_string())

    # ---- 2. Std vs AVG correlation (brief: 0.76 -> justifies dropping Std) ----
    _hr("2. Std <-> AVG CORRELATION  (brief: 0.76)")
    print(f"corr(Std, AVG) = {df['Std'].corr(df['AVG']):.4f}")

    # ---- 3. Per-class means of the 8 feature sources -------------------------
    _hr("3. PER-CLASS MEANS OF FEATURE SOURCES  (do they separate the classes?)")
    teardown = df[["rst_count", "fin_count"]].sum(axis=1).rename("teardown(rst+fin)")
    means = df.groupby("parent")[FEATURE_SRC].mean()
    means["teardown(rst+fin)"] = teardown.groupby(df["parent"]).mean()
    print(means.round(2).T.to_string())

    # ---- 4. Skew / range for log1p decisions ---------------------------------
    _hr("4. SKEW & RANGE  (log1p justified where skew is extreme)")
    skew_cols = FEATURE_SRC + ["ack_count", "Radius", "Magnitue", "Tot size"]
    desc = df[skew_cols].describe(percentiles=[.5, .95, .99]).T[["mean", "50%", "95%", "99%", "max"]]
    desc["skew"] = df[skew_cols].skew()
    print(desc.round(2).to_string())

    # ---- 5. Entanglement-pair correlations (brief's CHANGE 4 basis) ----------
    _hr("5. CORRELATIONS FOR ENTANGLEMENT PAIRS  (brief: rst<->Header=0.75, AVG<->Cov=0.50)")
    pair_cols = ["Rate", "syn_count", "rst_count", "fin_count",
                 "Header_Length", "AVG", "Covariance", "urg_count"]
    print(df[pair_cols].corr().round(2).to_string())

    # ---- 6. Protocol profile signals -----------------------------------------
    _hr("6. PROTOCOL SIGNALS  (per-class means + is_gre fraction)")
    print(df.groupby("parent")[PROTOCOL].mean().round(3).T.to_string())
    is_gre = (df[["TCP", "UDP", "ICMP"]].sum(axis=1) == 0).astype(int)
    print("\nFraction of rows with is_gre==1 (TCP+UDP+ICMP all zero), per class:")
    print(is_gre.groupby(df["parent"]).mean().round(4).sort_values(ascending=False).to_string())

    # ---- 7. Data hygiene -----------------------------------------------------
    _hr("7. DATA HYGIENE  (NaN / inf in feature sources)")
    check = df[FEATURE_SRC + PROTOCOL]
    print(f"NaN count total: {int(check.isna().sum().sum())}")
    print(f"Inf count total: {int(np.isinf(check.to_numpy(dtype=float)).sum())}")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
