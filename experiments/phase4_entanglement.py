"""Phase 4 Step 1: derive the feature-map entanglement pairs from smart-8 data.

The original brief's entanglement pairs were chosen from the OLD feature set and
included pair C (AVG <-> Covariance) — invalid now, since smart-8 dropped Covariance
(replaced by flow_timing/IAT and handshake_ratio). This recomputes the correlation
matrix of the actual scaled smart-8 features and ranks candidate entanglement pairs
by |correlation| so the 4 entangled pairs are chosen from measured couplings.

Run:  ./venv/Scripts/python.exe -m experiments.phase4_entanglement
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from agent import agent_config as config
from experiments.phase2_baselines import prepare


def main() -> None:
    X8_real, _, _, _ = prepare()
    # subsample for a fast, stable correlation estimate
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.choice(len(X8_real), size=min(200_000, len(X8_real)), replace=False)
    df = pd.DataFrame(X8_real[idx], columns=config.QUANTUM_FEATURE_NAMES)

    corr = df.corr()
    print("\n=== smart-8 feature correlation matrix (scaled (0,pi) features) ===")
    print(corr.round(2).to_string())

    print("\n=== candidate entanglement pairs, ranked by |correlation| ===")
    names = config.QUANTUM_FEATURE_NAMES
    pairs = []
    for i, j in itertools.combinations(range(8), 2):
        r = corr.iloc[i, j]
        pairs.append((abs(r), r, names[i], names[j], i, j))
    pairs.sort(reverse=True)
    for absr, r, a, b, i, j in pairs:
        print(f"  q{i}-q{j}  |r|={absr:.3f}  r={r:+.3f}  {a} <-> {b}")


if __name__ == "__main__":
    main()
