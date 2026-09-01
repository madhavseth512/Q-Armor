"""Phase 12 — Confirmatory Protocol: recurring A->B->A domain-shift study.

Closes the "recurring A->B->A drift study" item from docs/paper.tex's
Confirmatory Protocol: "recurring A->B->A streams are especially important
because they test whether episodic memory provides measurable benefit beyond
a memoryless trigger."

Three 5-episode blocks (15 episodes total): A1 (NF-ToN-IoT, within) -> B
(NF-UNSW-NB15, cross) -> A2 (NF-ToN-IoT again, within). Runs the SAME
SelfReflector/rules TWICE under two conditions that differ in exactly one
mechanism -- what "episodic memory" means in this codebase, the episode
history EpisodicMemory.recent_reports() feeds into the REINFORCE rule's
consecutive-streak check (agent/reflector.py, Rule 4):

  - "persistent": episode history accumulates normally across all 15
    episodes (default agent behaviour, identical to experiments/
    phase7_reflexion.py).
  - "memoryless": episode history is wiped at each block transition (after
    episode 4 and after episode 9), so REINFORCE's 3-consecutive-healthy-
    episode streak can never carry across a domain change. The POLICY
    (model_hierarchy, thresholds -- what the controller currently believes)
    is NOT wiped, only the raw history; this isolates the one mechanism the
    paper calls "episodic memory" from the persistent policy state every
    reflexive controller has by construction.

Model setup mirrors experiments/phase7_reflexion.py exactly (PegasosQSVC on
the QSVM tier, RandomForest on the CLASSICAL tier, via the real
MODEL_HIERARCHY / TIER_TO_NAME machinery) -- an earlier draft of this script
tried to substitute two classical models as fake "tiers" and silently broke
SWITCH_MODEL/REINFORCE bookkeeping, since reflector._actual_tier() only
recognises the four real tier names. Both QSVC and RF are trained ONCE
up front and reused read-only across both 15-episode conditions.

The comparison of interest: in block A2 (episodes 10-14, the second time the
stream returns to the domain of block A1), does "persistent" reach a healthy
REINFORCE state faster than "memoryless", given it has already seen 5 healthy
A-domain episodes before? Both conditions use the same sampled episodes and
the same fitted models, so only the reflector's memory differs.

Run:  ./venv/Scripts/python.exe -m experiments.phase12_recurring_drift
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.model_selection import train_test_split

from agent import agent_config as config
from agent.episodic_memory import REINFORCE, SWITCH_SUBSET, EpisodicMemory
from agent.evaluator import Evaluator
from agent.reflector import SelfReflector
from data.nf_loader import NFPreprocessor, read_nf
from data.sampling import select_kernel_subset
from planning.planning import PlanningModule
from reasoning.classical import RandomForestModel
from reasoning.quantum import PegasosQSVCModel
from reasoning.selector import TIER_TO_NAME, ModelSelector

RESULTS_DIR = "results/phase12"
EPISODE_SIZE = config.EPISODE_SIZE
BLOCK_SIZE = 5
N_EPISODES = BLOCK_SIZE * 3  # A1, B, A2
RF_TRAIN_N = 2000


def _balanced_subset(X, y, n_per_class, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for cls in np.unique(y):
        pos = np.where(y == cls)[0]
        parts.append(rng.choice(pos, min(n_per_class, len(pos)), replace=False))
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _sample_episode(X, y, episode_id, n=EPISODE_SIZE):
    rng = np.random.default_rng(config.RANDOM_SEED + 1200 + episode_id)
    b_idx = np.where(y == 0)[0]
    a_idx = np.where(y == 1)[0]
    n_b = min(n // 2, len(b_idx))
    n_a = min(n - n_b, len(a_idx))
    chosen = np.concatenate([
        rng.choice(b_idx, n_b, replace=False),
        rng.choice(a_idx, n_a, replace=False),
    ])
    rng.shuffle(chosen)
    return X[chosen], y[chosen]


def _pick_model(policy: dict, selector: ModelSelector) -> str:
    available = set(selector.available())
    for tier in policy["model_hierarchy"]:
        name = TIER_TO_NAME.get(tier)
        if name in available:
            return name
    return next(iter(available))


def _run_episode(model, X_ep, y_ep, planner):
    proba = model.predict_proba(X_ep)
    y_scores = proba[:, 1]
    y_pred = model.predict_labels(X_ep)
    confidences = proba.max(axis=1)
    drift_seen = False
    for i in range(len(y_ep)):
        error = float(y_pred[i] != y_ep[i])
        if planner.update_drift(error):
            drift_seen = True
    return y_scores, y_pred, confidences, drift_seen


def _run_condition(
    label: str,
    selector: ModelSelector,
    X_ton: np.ndarray, y_ton: np.ndarray,
    X_unsw: np.ndarray, y_unsw: np.ndarray,
    wipe_history_at_transitions: bool,
) -> list[dict]:
    print(f"\n{'=' * 60}\nCONDITION: {label}\n{'=' * 60}")

    log_path = f"{RESULTS_DIR}/{label}_episodes.jsonl"
    if os.path.exists(log_path):
        os.remove(log_path)
    memory = EpisodicMemory(log_path=log_path)
    evaluator = Evaluator()
    reflector = SelfReflector()
    planner = PlanningModule()
    policy = memory.get_policy()

    summaries = []

    for ep_id in range(N_EPISODES):
        block = ep_id // BLOCK_SIZE  # 0=A1, 1=B, 2=A2
        is_cross = block == 1
        dataset_name = "NF-UNSW-NB15" if is_cross else "NF-ToN-IoT"
        X_pool, y_pool = (X_unsw, y_unsw) if is_cross else (X_ton, y_ton)

        # Wipe episode HISTORY (not policy) at each block transition -- the
        # one line that differentiates the two conditions.
        if wipe_history_at_transitions and ep_id in (BLOCK_SIZE, 2 * BLOCK_SIZE):
            memory._records = []

        model_name = _pick_model(policy, selector)
        model = selector.get(model_name)

        X_ep, y_ep = _sample_episode(X_pool, y_pool, ep_id)
        y_scores, y_pred, confidences, drift = _run_episode(model, X_ep, y_ep, planner)

        report = evaluator.evaluate(
            episode_id=ep_id, model_used=model_name, dataset=dataset_name,
            y_true=y_ep, y_scores=y_scores, y_pred=y_pred,
            confidences=confidences, drift_detected=drift,
        )
        lesson = reflector.reflect(report, memory, policy)
        memory.record(report, lesson)
        policy = memory.get_policy()

        if lesson is not None and lesson.action == SWITCH_SUBSET:
            planner.reset_drift_detector()
            memory.acknowledge_reselect()

        block_name = ["A1", "B", "A2"][block]
        lesson_str = lesson.action if lesson else "-"
        print(f"  ep{ep_id:2d} [{block_name}] model={model_name:<13} "
              f"AUROC={report.auroc:.4f}  drift={drift}  lesson={lesson_str}")

        summaries.append({
            "episode_id": ep_id, "block": block_name, "dataset": dataset_name,
            "model_used": model_name, "auroc": report.auroc,
            "binary_f1": report.binary_f1, "drift": drift,
            "lesson_action": lesson.action if lesson else None,
        })

    return summaries


def main() -> None:
    print("=" * 70)
    print("Phase 12: Recurring A->B->A domain-shift study")
    print("=" * 70)

    X_ton_df, y_ton, _ = read_nf(config.NF_TON_CSV)
    X_unsw_df, y_unsw, _ = read_nf(config.NF_UNSW_CSV)

    X_ton_tr_df, X_ton_te_df, y_ton_tr, y_ton_te = train_test_split(
        X_ton_df, y_ton, test_size=0.20, stratify=y_ton, random_state=config.RANDOM_SEED)

    pre = NFPreprocessor()
    X8_ton_tr = pre.fit_transform(X_ton_tr_df)
    X8_ton_te = pre.transform(X_ton_te_df)
    X8_unsw = pre.transform(X_unsw_df)

    n_sub = config.QSVC_SUBSET_SIZE
    print(f"\nSelecting QSVC kernel subset n={n_sub} ...")
    idx = select_kernel_subset(X8_ton_tr, y_ton_tr, size=n_sub, method="kmeans")
    t0 = time.time()
    print("Training PegasosQSVC ...")
    qsvc = PegasosQSVCModel(num_steps=config.PEGASOS_TAU).fit(X8_ton_tr[idx], y_ton_tr[idx])
    print(f"  done in {time.time() - t0:.1f}s")

    t0 = time.time()
    print(f"Training RandomForest (n_per_class={RF_TRAIN_N // 2}) ...")
    X_rf, y_rf = _balanced_subset(X8_ton_tr, y_ton_tr, RF_TRAIN_N // 2, config.RANDOM_SEED)
    rf = RandomForestModel(class_weight=None).fit(X_rf, y_rf)
    print(f"  done in {time.time() - t0:.1f}s")

    selector = ModelSelector()
    selector.register(qsvc)
    selector.register(rf)
    print(f"Registered models: {selector.available()}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    persistent = _run_condition("persistent", selector, X8_ton_te, y_ton_te,
                                 X8_unsw, y_unsw, wipe_history_at_transitions=False)
    memoryless = _run_condition("memoryless", selector, X8_ton_te, y_ton_te,
                                 X8_unsw, y_unsw, wipe_history_at_transitions=True)

    def _first_reinforce_in_block(summaries, block_name):
        for s in summaries:
            if s["block"] == block_name and s["lesson_action"] == REINFORCE:
                return s["episode_id"]
        return None

    p_a2 = _first_reinforce_in_block(persistent, "A2")
    m_a2 = _first_reinforce_in_block(memoryless, "A2")

    out = {
        "experiment": "phase12_recurring_drift",
        "n_episodes": N_EPISODES, "block_size": BLOCK_SIZE,
        "persistent": persistent, "memoryless": memoryless,
        "first_reinforce_in_A2": {"persistent": p_a2, "memoryless": m_a2},
    }
    path = f"{RESULTS_DIR}/phase12_recurring_drift_summary.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPARISON: first REINFORCE episode-id within block A2 (10-14)")
    print("=" * 70)
    print(f"  persistent : {p_a2}")
    print(f"  memoryless : {m_a2}")
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
