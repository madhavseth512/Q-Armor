"""Phase 3 setup: fit and persist the agent's perception + reasoning models.

Trains the smart-8 preprocessor and the locked flat RandomForest on the full
training pool, then saves both to the memory directory so AgentCore.load() boots
without retraining. Run once before using the agent.

Run:  ./venv/Scripts/python.exe -m experiments.phase3_setup
"""

from __future__ import annotations

import time

from imblearn.over_sampling import SMOTE

from agent import agent_config as config
from data.loader import load_dataset
from data.preprocess import (
    QuantumPreprocessor,
    dynamic_smote_strategy,
    majority_undersample_index,
)
from memory.memory import MemoryModule
from reasoning.classical import RandomForestModel


def main() -> None:
    t0 = time.time()
    X, y = load_dataset(config.TRAIN_CSV)
    keep = majority_undersample_index(y)
    Xtr, ytr = X.loc[keep], y.loc[keep]

    pre = QuantumPreprocessor()
    X8 = pre.fit_transform(Xtr)
    Xr, yr = SMOTE(random_state=config.RANDOM_SEED,
                   sampling_strategy=dynamic_smote_strategy(ytr)).fit_resample(X8, ytr)
    rf = RandomForestModel().fit(Xr, yr)

    mem = MemoryModule()
    mem.save_model(pre, "preprocessor")
    mem.save_model(rf, "random_forest")
    print(f"saved preprocessor + random_forest to '{config.MEMORY_DIR}/' "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
