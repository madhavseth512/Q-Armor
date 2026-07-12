"""Phase 4b: derive correct entanglement pairs from NF-ToN-IoT features.

The original ENTANGLEMENT_PAIRS in agent_config.py were derived from
CICIoT2023 smart-8 features (teardown, syn, header, protocol) — not from
the NF-ToN-IoT raw NetFlow features actually used in Phase 6–8 experiments.
This script re-derives the pairs from the correct dataset.

Run:  ./venv/Scripts/python.exe -m experiments.phase4b_nf_entanglement
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd
from scipy import stats

from agent import agent_config as config
from data.nf_loader import NFPreprocessor, read_nf


def main() -> None:
    print(f"Loading NF-ToN-IoT from {config.NF_TON_CSV} ...")
    X_df, y_bin, _ = read_nf(config.NF_TON_CSV)
    print(f"  Loaded {len(X_df):,} rows, {X_df.shape[1]} features")

    prep = NFPreprocessor()
    X_scaled = prep.fit_transform(X_df)

    rng = np.random.default_rng(config.RANDOM_SEED)
    n = min(200_000, len(X_scaled))
    idx = rng.choice(len(X_scaled), size=n, replace=False)
    X_sub = X_scaled[idx]
    print(f"  Subsampled {n:,} rows for correlation estimate")

    names = config.NF_FEATURE_NAMES
    df_sub = pd.DataFrame(X_sub, columns=names)

    corr_pearson = df_sub.corr(method="pearson")
    corr_spearman = df_sub.corr(method="spearman")

    print("\n=== NF-ToN-IoT Pearson correlation matrix (scaled [0, pi] features) ===")
    print(corr_pearson.round(3).to_string())

    print("\n=== Candidate entanglement pairs ranked by |Pearson r| ===")
    print(f"{'Pair':<8}  {'|r|':>6}  {'r':>7}  {'ρ(Spearman)':>12}  Features")
    print("-" * 70)
    pairs: list[tuple[float, float, float, str, str, int, int]] = []
    for i, j in itertools.combinations(range(len(names)), 2):
        r   = corr_pearson.iloc[i, j]
        rho = corr_spearman.iloc[i, j]
        pairs.append((abs(r), r, rho, names[i], names[j], i, j))
    pairs.sort(reverse=True)
    for absr, r, rho, a, b, i, j in pairs:
        print(f"  q{i}-q{j}   {absr:>6.3f}  {r:>+7.3f}  {rho:>+12.3f}  {a} <-> {b}")

    current = config.ENTANGLEMENT_PAIRS
    top4 = [(i, j) for _, _, _, _, _, i, j in pairs[:4]]

    print("\n=== Comparison ===")
    print(f"  Current  (CICIoT2023-derived): {current}")
    print(f"  Suggested (NF-ToN-IoT top-4) : {top4}")

    changed = set(map(tuple, current)) != set(map(tuple, top4))
    if changed:
        print("\n  PAIRS DIFFER — Phase 8 re-run required.")
        print("\n  New ENTANGLEMENT_PAIRS for agent_config.py:")
        print(f"  ENTANGLEMENT_PAIRS: list[tuple[int, int]] = {top4}")
        print("\n  Comment lines:")
        for rank, (absr, r, rho, a, b, i, j) in enumerate(pairs[:4], 1):
            print(f"  # q{i}-q{j}  {a} <-> {b}  r={r:+.3f}  (rank {rank})")
    else:
        print("\n  Pairs are identical — no re-run needed.")


if __name__ == "__main__":
    main()
