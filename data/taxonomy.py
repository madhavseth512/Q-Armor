"""Common coarse taxonomy for cross-dataset multi-class classification.

Maps per-dataset native attack-type strings to the shared 5-class coarse
taxonomy (Benign / DoS / Injection / Recon / Backdoor) defined in
agent_config.NF_COARSE_TAXONOMY. Any unmapped label raises KeyError so that
schema drift is caught immediately rather than silently producing wrong labels.
"""

from __future__ import annotations

import numpy as np

from agent import agent_config as config


def apply_taxonomy(y_attack: np.ndarray) -> np.ndarray:
    """Map native attack-type strings to the 5-class coarse taxonomy.

    Args:
        y_attack: Array of native attack-type strings (from NF_ATTACK_COL).

    Returns:
        Array of coarse class strings aligned with NF_COARSE_CLASSES.

    Raises:
        KeyError: If any label is not present in NF_COARSE_TAXONOMY.
    """
    mapping = config.NF_COARSE_TAXONOMY
    unknown = set(y_attack) - set(mapping)
    if unknown:
        raise KeyError(f"Unknown attack labels not in NF_COARSE_TAXONOMY: {unknown}")
    return np.array([mapping[label] for label in y_attack])
