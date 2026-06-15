"""CICIoT2023 dataset loader.

Reads the raw CICIoT2023 CSVs, derives the 15 parent attack classes from the
``label`` column, asserts that exactly the expected 15 classes are present, and
drops the dead/constant/redundant columns identified by EDA. This module does no
feature engineering or scaling — that is the responsibility of ``preprocess.py``.
"""

from __future__ import annotations

import pandas as pd

from agent import agent_config as config


class DatasetError(RuntimeError):
    """Raised when the loaded dataset violates an expected invariant."""


def derive_parent_class(labels: pd.Series) -> pd.Series:
    """Derive the parent attack class from full subtype labels.

    The CICIoT2023 ``label`` column holds 34 subtype strings following a
    ``Parent-Subtype`` pattern (e.g. ``"DDoS-ICMP_Flood"`` -> ``"DDoS"``).
    Single-token labels (e.g. ``"BenignTraffic"``, ``"XSS"``) are returned
    unchanged, which is the intended behaviour.

    Args:
        labels: Series of full subtype label strings.

    Returns:
        Series of parent-class strings, index-aligned with ``labels``.
    """
    return labels.str.split("-").str[0]


def _assert_expected_classes(parents: pd.Series) -> None:
    """Validate that the derived parent classes are exactly the expected 15.

    Args:
        parents: Series of derived parent-class strings.

    Raises:
        DatasetError: If the number or names of unique parent classes do not
            match ``agent_config.ATTACK_CLASSES``.
    """
    found = set(parents.unique())
    expected = set(config.ATTACK_CLASSES)

    if len(found) != config.N_PARENT_CLASSES:
        raise DatasetError(
            f"Expected {config.N_PARENT_CLASSES} parent classes, "
            f"found {len(found)}: {sorted(found)}"
        )
    if found != expected:
        missing = expected - found
        unexpected = found - expected
        raise DatasetError(
            "Parent classes do not match the configured ATTACK_CLASSES. "
            f"Missing: {sorted(missing)} | Unexpected: {sorted(unexpected)}"
        )


def _drop_dead_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the EDA-identified dead/constant/redundant columns.

    Columns are taken from ``agent_config.DROP_COLUMNS``. Columns that are
    already absent are ignored so the loader is robust to minor schema drift.

    Args:
        df: The raw dataframe.

    Returns:
        A copy of ``df`` without the dropped columns.
    """
    present = [c for c in config.DROP_COLUMNS if c in df.columns]
    return df.drop(columns=present)


def load_dataset(
    csv_path: str | None = None,
    *,
    validate: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CICIoT2023 CSV into features and parent-class labels.

    Pipeline: read CSV -> derive parent class -> assert 15 classes -> drop dead
    columns -> split into (features, labels).

    Args:
        csv_path: Path to the CSV to load. Defaults to
            ``agent_config.TRAIN_CSV`` when ``None``.
        validate: If True, assert exactly the expected 15 parent classes are
            present and raise ``DatasetError`` otherwise. Set False only for the
            test/validation splits if a class is legitimately absent.

    Returns:
        A tuple ``(X, y)`` where ``X`` is the feature dataframe (label and
        derived parent column removed, dead columns dropped) and ``y`` is the
        parent-class label Series.

    Raises:
        DatasetError: If ``validate`` is True and the parent classes do not
            match the configured 15, or if the ``label`` column is missing.
    """
    path = csv_path if csv_path is not None else config.TRAIN_CSV

    df = pd.read_csv(path)

    if "label" not in df.columns:
        raise DatasetError(f"'label' column not found in {path}")

    parents = derive_parent_class(df["label"])

    if validate:
        _assert_expected_classes(parents)

    df = _drop_dead_columns(df)
    X = df.drop(columns=["label"])
    y = parents.rename("parent")

    return X, y
