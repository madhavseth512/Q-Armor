"""NF-dataset loader and preprocessor for Phase 6+.

Handles both NF-ToN-IoT and NF-UNSW-NB15 (identical 14-column schema).
The two IP address columns are dropped; the remaining 8 features are selected
from NF_FEATURE_NAMES (locked by experiments/phase6_eda.py).

Scaling pipeline (leak-free — fit on train, apply to val/test):
  1. Select NF_FEATURE_NAMES columns
  2. log1p on NF_LOG_COLS  (heavy-tailed; stateless)
  3. MinMaxScaler -> [0, pi]  (fit on train only)
  4. clip to [0, pi] on val/test  (out-of-range guard for unseen extremes)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from agent import agent_config as config

# Positional indices of NF_LOG_COLS within NF_FEATURE_NAMES (precomputed once).
_LOG_IDX: list[int] = [
    config.NF_FEATURE_NAMES.index(c)
    for c in config.NF_LOG_COLS
    if c in config.NF_FEATURE_NAMES
]


def read_nf(csv_path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Read an NF CSV and return raw features + both label arrays.

    Drops the two IP address columns (IPV4_SRC_ADDR, IPV4_DST_ADDR) and the
    label columns, returning only the numeric feature block. No scaling is
    applied — call NFPreprocessor for that.

    Args:
        csv_path: Path to NF-ToN-IoT.csv or NF-UNSW-NB15.csv.

    Returns:
        Tuple of:
          X_df   -- DataFrame with NF_FEATURE_NAMES columns (unscaled).
          y_bin  -- int array (0=benign, 1=attack) from the Label column.
          y_att  -- str array of attack-type names from the Attack column.
    """
    df = pd.read_csv(csv_path)
    X_df  = df[config.NF_FEATURE_NAMES].copy()
    y_bin = df[config.NF_LABEL_COL].values.astype(int)
    y_att = df[config.NF_ATTACK_COL].values.astype(str)
    return X_df, y_bin, y_att


class NFPreprocessor:
    """Stateful log + MinMax scaler for NF NetFlow features.

    Fit on the training split, then apply to validation/test via transform.
    Mirrors the QuantumPreprocessor API so it can be used interchangeably
    in experiment scripts.
    """

    def __init__(self) -> None:
        self._minmax = MinMaxScaler(feature_range=(0.0, float(np.pi)))
        self._fitted = False

    def _log_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply log1p in-place to the heavy-tailed columns."""
        X = X.copy()
        X[:, _LOG_IDX] = np.log1p(X[:, _LOG_IDX])
        return X

    def fit_transform(self, X_df: pd.DataFrame) -> np.ndarray:
        """Fit scalers on training data and return scaled array.

        Args:
            X_df: DataFrame with NF_FEATURE_NAMES columns (from read_nf).

        Returns:
            Array of shape (n, 8) with values in [0, pi].
        """
        X = X_df[config.NF_FEATURE_NAMES].values.astype(float)
        X = self._log_transform(X)
        out = self._minmax.fit_transform(X)
        self._fitted = True
        return out

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        """Scale validation/test data using the fitted scalers.

        Clips to [0, pi] to guard against out-of-range values from unseen
        extremes in a different dataset (expected in cross-dataset evaluation).

        Args:
            X_df: DataFrame with NF_FEATURE_NAMES columns (from read_nf).

        Returns:
            Array of shape (n, 8) with values clipped to [0, pi].

        Raises:
            RuntimeError: If called before fit_transform.
        """
        if not self._fitted:
            raise RuntimeError("NFPreprocessor.transform called before fit_transform.")
        X = X_df[config.NF_FEATURE_NAMES].values.astype(float)
        X = self._log_transform(X)
        scaled = self._minmax.transform(X)
        return np.clip(scaled, 0.0, float(np.pi))
