"""Phase 7 — Reflexion drift simulation (Experiment E8).

10-episode simulation in two blocks:
  Block A (episodes 0-4): sample from NF-ToN-IoT test split — within-dataset.
  Block B (episodes 5-9): sample from NF-UNSW-NB15 — cross-dataset (drift).

Outer Reflexion loop after each episode:
    Evaluator → SelfReflector → EpisodicMemory (policy update)

Model registry:
  QSVM tier  → PegasosQSVCModel  (150-sample k-means subset; quantum fidelity kernel)
  CLASSICAL  → RandomForestModel  (2000-sample balanced subset; fast classical fallback)
  VQC / QAE  → NOT registered here (training cost prohibitive for 10-episode sim;
                                    exercised in Phase 8 with pre-fitted checkpoints)

Expected trajectory:
  Episode 0-1: within, AUROC ~0.94, no lesson (streak too short)
  Episode 2:   within, AUROC ~0.94, REINFORCE fires (3 consecutive)
  Episode 3-4: within, AUROC ~0.94, REINFORCE repeats (idempotent)
  Episode 5:   cross, AUROC ~0.64, ADWIN detects drift → SWITCH_SUBSET
  Episode 6:   cross, AUROC ~0.64, no drift yet (ADWIN reset) → SWITCH_MODEL → RF
  Episode 7-9: cross with RF, AUROC ~0.74, healthy; REINFORCE at episode 9

Run:  ./venv/Scripts/python.exe -m experiments.phase7_reflexion
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.model_selection import train_test_split

from agent import agent_config as config
from agent.episodic_memory import SWITCH_SUBSET, EpisodicMemory
from agent.evaluator import Evaluator
from agent.reflector import SelfReflector
from data.nf_loader import NFPreprocessor, read_nf
from data.sampling import select_kernel_subset
from planning.planning import PlanningModule
from reasoning.classical import RandomForestModel
from reasoning.quantum import PegasosQSVCModel
from reasoning.selector import TIER_TO_NAME, ModelSelector

RESULTS_DIR  = "results/phase7"
STATE_DIR    = "agent_state"
N_WITHIN     = 5
N_CROSS      = 5
N_EPISODES   = N_WITHIN + N_CROSS
EPISODE_SIZE = config.EPISODE_SIZE   # 100
RF_TRAIN_N   = 2000                  # balanced subset for RF (larger than QSVC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_episode(
    X: np.ndarray,
    y: np.ndarray,
    episode_id: int,
    n: int = EPISODE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a balanced n-sample episode from (X, y)."""
    rng = np.random.default_rng(config.RANDOM_SEED + 700 + episode_id)
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


def _balanced_subset(
    X: np.ndarray,
    y: np.ndarray,
    n_per_class: int,
    seed: int = config.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a class-balanced subsample capped at n_per_class per class."""
    rng = np.random.default_rng(seed)
    parts = []
    for cls in np.unique(y):
        pos = np.where(y == cls)[0]
        parts.append(rng.choice(pos, min(n_per_class, len(pos)), replace=False))
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _pick_model(policy: dict, selector: ModelSelector) -> str:
    """Return the name of the highest-priority registered model per Reflexion policy."""
    available = set(selector.available())
    for tier in policy["model_hierarchy"]:
        name = TIER_TO_NAME.get(tier)
        if name in available:
            return name
    return next(iter(available))


def _run_episode(
    model,
    X_ep: np.ndarray,
    y_ep: np.ndarray,
    planner: PlanningModule,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Run one episode: get predictions, stream errors through ADWIN."""
    proba        = model.predict_proba(X_ep)
    y_scores     = proba[:, 1]
    y_pred       = proba.argmax(axis=1)
    confidences  = proba.max(axis=1)
    drift_seen   = False
    for i in range(len(y_ep)):
        error = float(y_pred[i] != y_ep[i])
        if planner.update_drift(error):
            drift_seen = True
    return y_scores, y_pred, confidences, drift_seen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Phase 7 — Reflexion drift simulation (E8)")
    print(f"  {N_WITHIN} within-dataset episodes then {N_CROSS} cross-dataset episodes")
    print("=" * 70)

    # ── 1. Load datasets ─────────────────────────────────────────────────────
    print(f"\nLoading {config.NF_TON_CSV} ...")
    t0 = time.time()
    X_ton_df, y_ton, _ = read_nf(config.NF_TON_CSV)
    print(f"  {len(y_ton):,} rows  binary: {np.bincount(y_ton)}  ({time.time()-t0:.1f}s)")

    print(f"Loading {config.NF_UNSW_CSV} ...")
    t0 = time.time()
    X_unsw_df, y_unsw, _ = read_nf(config.NF_UNSW_CSV)
    print(f"  {len(y_unsw):,} rows  binary: {np.bincount(y_unsw)}  ({time.time()-t0:.1f}s)")

    # ── 2. Train/test split on NF-ToN-IoT ────────────────────────────────────
    X_ton_tr_df, X_ton_te_df, y_ton_tr, y_ton_te = train_test_split(
        X_ton_df, y_ton,
        test_size=0.20,
        stratify=y_ton,
        random_state=config.RANDOM_SEED,
    )
    print(f"\n  ToN-IoT train: {X_ton_tr_df.shape}  test: {X_ton_te_df.shape}")

    # ── 3. Scale ─────────────────────────────────────────────────────────────
    print("Scaling (fit on NF-ToN-IoT train) ...")
    pre = NFPreprocessor()
    X8_ton_tr = pre.fit_transform(X_ton_tr_df)
    X8_ton_te = pre.transform(X_ton_te_df)
    X8_unsw   = pre.transform(X_unsw_df)      # apply train-fit scaler to UNSW
    print(f"  ToN-IoT train  {X8_ton_tr.shape}  "
          f"range [{X8_ton_tr.min():.3f}, {X8_ton_tr.max():.3f}]")
    print(f"  UNSW-NB15      {X8_unsw.shape}  "
          f"range [{X8_unsw.min():.3f}, {X8_unsw.max():.3f}]")

    # ── 4. QSVC kernel subset ─────────────────────────────────────────────────
    n_sub = config.QSVC_SUBSET_SIZE
    print(f"\nSelecting QSVC kernel subset n={n_sub} (k-means, balanced) ...")
    t0 = time.time()
    idx = select_kernel_subset(X8_ton_tr, y_ton_tr, size=n_sub, method="kmeans")
    X_sub, y_sub = X8_ton_tr[idx], y_ton_tr[idx]
    print(f"  subset {X_sub.shape}  labels {np.bincount(y_sub)}  ({time.time()-t0:.1f}s)")

    # ── 5. Train QSVC ─────────────────────────────────────────────────────────
    print("\nTraining PegasosQSVC (quantum fidelity kernel) ...")
    t0 = time.time()
    qsvc = PegasosQSVCModel(num_steps=config.PEGASOS_TAU).fit(X_sub, y_sub)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── 6. Train RF on larger balanced subset ────────────────────────────────
    print(f"\nTraining RandomForest (balanced subset n_per_class={RF_TRAIN_N // 2}) ...")
    t0 = time.time()
    X_rf_tr, y_rf_tr = _balanced_subset(X8_ton_tr, y_ton_tr, RF_TRAIN_N // 2)
    rf = RandomForestModel(class_weight=None).fit(X_rf_tr, y_rf_tr)
    print(f"  done in {time.time()-t0:.1f}s  subset {X_rf_tr.shape}")

    # ── 7. Build selector + Reflexion components ──────────────────────────────
    selector = ModelSelector()
    selector.register(qsvc)
    selector.register(rf)
    print(f"\nRegistered models: {selector.available()}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)

    memory   = EpisodicMemory()
    evaluator = Evaluator()
    reflector = SelfReflector()
    planner  = PlanningModule()
    policy   = memory.get_policy()

    episode_summaries = []

    print("\n" + "=" * 70)
    print("Starting Reflexion loop ...")
    print("=" * 70)

    for ep_id in range(N_EPISODES):
        is_cross = ep_id >= N_WITHIN
        dataset_name = "NF-UNSW-NB15" if is_cross else "NF-ToN-IoT"
        X_pool = X8_unsw    if is_cross else X8_ton_te
        y_pool = y_unsw     if is_cross else y_ton_te

        block_label = "CROSS" if is_cross else "WITHIN"
        model_name = _pick_model(policy, selector)
        model = selector.get(model_name)

        print(f"\n--- Episode {ep_id:2d} [{block_label}] "
              f"dataset={dataset_name}  model={model_name} ---")

        # Sample episode
        X_ep, y_ep = _sample_episode(X_pool, y_pool, ep_id)

        # Run predictions + ADWIN stream
        t0 = time.time()
        y_scores, y_pred, confidences, drift = _run_episode(model, X_ep, y_ep, planner)
        elapsed = time.time() - t0

        # Evaluate
        report = evaluator.evaluate(
            episode_id     = ep_id,
            model_used     = model_name,
            dataset        = dataset_name,
            y_true         = y_ep,
            y_scores       = y_scores,
            y_pred         = y_pred,
            confidences    = confidences,
            drift_detected = drift,
        )

        # Reflect
        lesson = reflector.reflect(report, memory, policy)

        # Record (updates policy in-place)
        memory.record(report, lesson)
        policy = memory.get_policy()

        # Handle post-lesson side effects
        if lesson is not None and lesson.action == SWITCH_SUBSET:
            # In production: re-run k-means and retrain here.
            # Simulation: reset ADWIN so the next episode gets a clean slate.
            planner.reset_drift_detector()
            print("  [SIM] SWITCH_SUBSET: ADWIN reset; subset retraining would happen here.")
            memory.acknowledge_reselect()

        # Print episode summary
        print(f"  AUROC={report.auroc:.4f}  F1={report.binary_f1:.4f}  "
              f"AUPR={report.aupr:.4f}  FPR@95={report.fpr_at_tpr95:.4f}")
        print(f"  drift={report.drift_detected}  conf={report.mean_confidence:.4f}  "
              f"predict_time={elapsed:.2f}s")
        if lesson:
            print(f"  >> LESSON: {lesson.action} — {lesson.trigger}")
            print(f"     params: {lesson.params}")
        else:
            print("  >> No lesson (episode healthy, no rule triggered)")
        print(f"  Policy after: model_hierarchy={policy['model_hierarchy'][:2]}... "
              f"subset_reselect={policy['subset_reselect']}  "
              f"binary_only={policy['binary_only']}")

        episode_summaries.append({
            "episode_id":    ep_id,
            "block":         block_label,
            "dataset":       dataset_name,
            "model_used":    model_name,
            "auroc":         report.auroc,
            "binary_f1":     report.binary_f1,
            "drift":         report.drift_detected,
            "lesson_action": lesson.action if lesson else None,
        })

    # ── 8. Save summary ───────────────────────────────────────────────────────
    out = {
        "experiment":       "phase7_reflexion",
        "n_episodes":       N_EPISODES,
        "n_within":         N_WITHIN,
        "n_cross":          N_CROSS,
        "episode_size":     EPISODE_SIZE,
        "auroc_floor":      config.AUROC_FLOOR,
        "reinforce_consec": config.REINFORCE_CONSECUTIVE,
        "models_registered": selector.available(),
        "episodes":         episode_summaries,
        "final_policy":     policy,
    }
    out_path = f"{RESULTS_DIR}/phase7_reflexion_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSummary saved -> {out_path}")
    print(f"Episode log   -> {config.REFLEXION_LOG_PATH}")

    # ── 9. Print table ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EPISODE SUMMARY")
    print("=" * 70)
    print(f"  {'Ep':>3}  {'Block':>7}  {'Model':>18}  {'AUROC':>7}  "
          f"{'F1':>6}  {'Drift':>5}  {'Lesson'}")
    print("  " + "-" * 68)
    for s in episode_summaries:
        drift_flag = "YES" if s["drift"] else "no"
        lesson_str = s["lesson_action"] or "—"
        print(f"  {s['episode_id']:>3}  {s['block']:>7}  {s['model_used']:>18}  "
              f"{s['auroc']:>7.4f}  {s['binary_f1']:>6.4f}  "
              f"{drift_flag:>5}  {lesson_str}")

    print(f"\nFinal policy hierarchy: {policy['model_hierarchy']}")
    print(f"Total episodes logged:  {len(memory)}")


if __name__ == "__main__":
    main()
