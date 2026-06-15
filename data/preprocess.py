"""CICIoT2023 feature engineering and leak-free preprocessing.

Turns raw CICIoT2023 columns into the 8 interpretable quantum features (one per
qubit), scaled into the qubit angle range (0, pi). All learned transforms
(StandardScaler, MinMaxScaler) and SMOTE are fitted on the TRAINING split only
and applied to validation/test via ``transform`` — see data/FEATURE_ANALYSIS.md
and the D1/D2 decisions in docs/PROJECT_CHARTER.md.

Scaling pipeline (strict order):
    1. engineer the 8 features
    2. log1p on the skewed features {0,1,2,3,5,6}            (stateless)
    3. StandardScaler on feature 4 (avg_packet_size)         (fit on train)
    4. MinMaxScaler(0, pi) on all 8                           (fit on train)
    5. clip to (0, pi) on val/test                           (out-of-range guard)
SMOTE is applied separately, on the training split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from agent import agent_config as config

# Maps clean internal names -> the dataset's literal column headers. Centralises
# every name correction, including the dataset's own typo (`Magnitue`) and the
# headers that contain spaces. Keep exact spelling.
COLUMN_MAP: dict[str, str] = {
    "rate": "Rate",
    "header_length": "Header_Length",
    "syn_count": "syn_count",
    "rst_count": "rst_count",
    "fin_count": "fin_count",
    "urg_count": "urg_count",
    "ack_count": "ack_count",
    "avg": "AVG",
    "iat": "IAT",
    "covariance": "Covariance",
    "radius": "Radius",
    "magnitude": "Magnitue",          # dataset typo — keep exact
    "tot_size": "Tot size",           # has a space
    # protocol indicators used by protocol_profile
    "tcp": "TCP", "udp": "UDP", "icmp": "ICMP", "ssh": "SSH",
    "dns": "DNS", "https": "HTTPS",
    "label": "label",
}

# Features that receive log1p before scaling (q0,q1,q2,q3,q5,q6). Extreme right
# skew on the full data justifies each (Rate 23, rst 13, Header 10, IAT/ack heavy).
LOG1P_FEATURES: list[str] = [
    "traffic_rate", "syn_activity", "teardown_activity",
    "header_overhead", "flow_timing", "handshake_ratio",
]
# Feature standardised (mean 0 / std 1) before the global MinMax (q4).
STANDARD_FEATURE: str = "avg_packet_size"


# -- two-sided training-set sampling (D8; train only) -----------------------

def majority_undersample_index(
    y: pd.Series,
    cap: int = config.MAJORITY_CAP,
    seed: int = config.RANDOM_SEED,
) -> pd.Index:
    """Select row indices that undersample majority classes down to ``cap``.

    Classes with at most ``cap`` rows are kept whole; larger classes are randomly
    subsampled to ``cap``. Operates on labels only (no feature access), so it can
    be applied to the raw dataframe before any scaling.

    Args:
        y: Label Series for the training split.
        cap: Maximum rows kept per class.
        seed: Random seed for reproducible subsampling.

    Returns:
        Index of rows to keep (unordered union across classes).
    """
    parts: list[np.ndarray] = []
    for cls in y.unique():
        idx = y.index[y == cls]
        if len(idx) > cap:
            idx = pd.Series(idx).sample(cap, random_state=seed).to_numpy()
        parts.append(np.asarray(idx))
    return pd.Index(np.concatenate(parts))


def dynamic_smote_strategy(
    y: pd.Series,
    cap: int = config.MAJORITY_CAP,
    multiplier: int = config.SMOTE_MULTIPLIER,
) -> dict:
    """Build a SMOTE ``sampling_strategy`` with a per-class dynamic floor.

    Each class target is ``min(cap, multiplier * n_real)`` so synthesis is bounded
    relative to how much real data a class actually has — preventing ultra-rare
    classes from being inflated into mostly-synthetic noise. Only classes whose
    target exceeds their current count are included (SMOTE oversamples only).

    Args:
        y: Label Series AFTER majority undersampling.
        cap: Absolute ceiling per class.
        multiplier: Max oversampling factor relative to real count.

    Returns:
        Mapping ``{class_label: target_count}`` for ``imblearn.SMOTE``.
    """
    strategy: dict = {}
    for cls, n in y.value_counts().items():
        target = min(cap, multiplier * int(n))
        if target > n:
            strategy[cls] = int(target)
    return strategy


class QuantumPreprocessor:
    """Engineers and scales the 8 quantum features with leak-free fit/transform.

    Fit the learned scalers on the training split (``fit`` / ``fit_transform``),
    then apply them to validation/test with ``transform``. SMOTE is exposed
    separately via :meth:`resample` and must only ever be called on training data.
    """

    def __init__(self) -> None:
        """Initialise (unfitted) scalers and feature ordering from config."""
        self.feature_names: list[str] = config.QUANTUM_FEATURE_NAMES
        self._standard = StandardScaler()
        self._minmax = MinMaxScaler(feature_range=config.ANGLE_ENCODING_RANGE)
        self._fitted = False

    # -- feature engineering (stateless) ------------------------------------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the 8 engineered features from raw CICIoT2023 columns.

        Args:
            df: Raw feature dataframe (as returned by ``data.loader.load_dataset``).

        Returns:
            Dataframe with exactly the 8 columns in ``self.feature_names``.
        """
        c = COLUMN_MAP
        out = pd.DataFrame(index=df.index)
        out["traffic_rate"] = df[c["rate"]]
        out["syn_activity"] = df[c["syn_count"]]
        out["teardown_activity"] = df[c["rst_count"]] + df[c["fin_count"]]
        out["header_overhead"] = df[c["header_length"]]
        out["avg_packet_size"] = df[c["avg"]]
        out["flow_timing"] = df[c["iat"]]                                  # timing
        out["handshake_ratio"] = df[c["ack_count"]] / (df[c["syn_count"]] + 1)  # behaviour
        out["protocol_profile"] = self._protocol_profile(df)
        return out[self.feature_names]

    def _protocol_profile(self, df: pd.DataFrame) -> pd.Series:
        """Compute the weighted protocol_profile score (feature q7).

        Uses ``agent_config.PROTOCOL_PROFILE_WEIGHTS`` (EDA-revised: Telnet/ARP
        removed as dead; is_gre-anchored). ``is_gre`` fires when no listed
        protocol matches (TCP+UDP+ICMP all zero) — the Mirai-GRE signature.

        Args:
            df: Raw feature dataframe.

        Returns:
            Series of protocol_profile scores, index-aligned with ``df``.
        """
        c = COLUMN_MAP
        w = config.PROTOCOL_PROFILE_WEIGHTS
        is_gre = ((df[c["tcp"]] + df[c["udp"]] + df[c["icmp"]]) == 0).astype(float)
        return (
            w["is_gre"] * is_gre
            + w["ICMP"] * df[c["icmp"]]
            + w["SSH"] * df[c["ssh"]]
            + w["UDP"] * df[c["udp"]]
            + w["DNS"] * df[c["dns"]]
            + w["HTTPS"] * df[c["https"]]
        )

    # -- scaling (learned; fit on train only) -------------------------------

    def _apply_pre_minmax(self, feats: pd.DataFrame, *, fit: bool) -> np.ndarray:
        """Apply log1p and the StandardScaler stage, before the global MinMax.

        Args:
            feats: The 8 engineered features.
            fit: If True, fit the StandardScaler; otherwise apply the fitted one.

        Returns:
            The 8-column matrix ready for the MinMax stage.
        """
        m = feats.copy()
        m[LOG1P_FEATURES] = np.log1p(m[LOG1P_FEATURES])
        col = m[[STANDARD_FEATURE]].to_numpy()
        m[STANDARD_FEATURE] = (
            self._standard.fit_transform(col) if fit else self._standard.transform(col)
        ).ravel()
        return m[self.feature_names].to_numpy()

    def fit(self, X_raw: pd.DataFrame) -> "QuantumPreprocessor":
        """Fit the scalers on the training split.

        Args:
            X_raw: Raw training feature dataframe.

        Returns:
            Self, fitted.
        """
        feats = self.engineer_features(X_raw)
        m = self._apply_pre_minmax(feats, fit=True)
        self._minmax.fit(m)
        self._fitted = True
        return self

    def transform(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Engineer + scale features using the already-fitted scalers.

        Applies the explicit clip to (0, pi): validation/test rows may exceed the
        training min/max, and angle encoding requires values within the rotation
        range. The clip bounds the output without re-fitting (no leakage).

        Args:
            X_raw: Raw feature dataframe (validation or test).

        Returns:
            Array of shape (n_samples, 8) with all values in (0, pi).

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError("QuantumPreprocessor.transform called before fit().")
        feats = self.engineer_features(X_raw)
        m = self._apply_pre_minmax(feats, fit=False)
        scaled = self._minmax.transform(m)
        low, high = config.ANGLE_ENCODING_RANGE
        return np.clip(scaled, low, high)

    def fit_transform(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Fit on training data and return its scaled features.

        Args:
            X_raw: Raw training feature dataframe.

        Returns:
            Scaled training array of shape (n_samples, 8).
        """
        return self.fit(X_raw).transform(X_raw)

    # -- resampling (train only) --------------------------------------------

    @staticmethod
    def resample(
        X: np.ndarray,
        y: pd.Series,
        sampling_strategy: str | dict = "auto",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to balance the TRAINING split. Never call on val/test.

        Args:
            X: Scaled training features, shape (n_samples, 8).
            y: Training labels.
            sampling_strategy: Passed to ``imblearn.SMOTE``. Default "auto" fully
                balances and is NOT appropriate at the dataset's ~28,560:1 ratio
                — pass a target dict (undersample majorities first). See D8.

        Returns:
            Tuple ``(X_resampled, y_resampled)``.
        """
        smote = SMOTE(random_state=config.RANDOM_SEED, sampling_strategy=sampling_strategy)
        return smote.fit_resample(X, y)
