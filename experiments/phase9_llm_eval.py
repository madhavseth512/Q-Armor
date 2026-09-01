"""Phase 9 — LLM Reflexion agent evaluation.

Compares three reflection strategies on the same 10-episode simulation
(5 within-domain NF-ToN-IoT + 5 cross-domain NF-UNSW-NB15):

  A) LLM agent      — LLMDiagnoser + LLMPlanner + Verifier + LLMReflector
  B) Rule-based     — existing heuristic SelfReflector (Phase 7 baseline)
  C) ADWIN-only     — SWITCH_SUBSET whenever drift fires; no other reflection

To isolate reflector quality from stochastic training noise, the same
QSVC and RF models are used for all three agents. Episode-level predictions
are re-generated per agent run (deterministic — same seeds), but model
weights are fixed across agents.

Run:
    # Without LLM (rule-based and ADWIN-only only):
    ./venv/Scripts/python.exe -m experiments.phase9_llm_eval

    # With LLM agent (requires GROQ_API_KEY env var):
    set GROQ_API_KEY=gsk_...
    ./venv/Scripts/python.exe -m experiments.phase9_llm_eval --llm
"""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy

from dotenv import load_dotenv
load_dotenv(override=True)

import numpy as np
from sklearn.model_selection import train_test_split

from agent import agent_config as config
from agent.episodic_memory import EpisodicMemory, SWITCH_SUBSET
from agent.evaluator import Evaluator
from agent.llm_agent import LLMReflexionAgent
from agent.reflector import SelfReflector
from data.nf_loader import NFPreprocessor, read_nf
from data.sampling import select_kernel_subset
from memory.semantic_memory import SemanticMemory
from planning.planning import PlanningModule
from reasoning.classical import RandomForestModel
from reasoning.quantum import PegasosQSVCModel
from reasoning.selector import TIER_TO_NAME, ModelSelector

RESULTS_DIR  = "results/phase9"
STATE_DIR    = "agent_state"
N_WITHIN     = 5
N_CROSS      = 5
N_EPISODES   = N_WITHIN + N_CROSS
EPISODE_SIZE = config.EPISODE_SIZE
RF_TRAIN_N   = 2000


# ---------------------------------------------------------------------------
# Helpers (shared with phase7_reflexion.py)
# ---------------------------------------------------------------------------

def _sample_episode(X, y, episode_id, n=EPISODE_SIZE):
    rng   = np.random.default_rng(config.RANDOM_SEED + 700 + episode_id)
    b_idx = np.where(y == 0)[0]
    a_idx = np.where(y == 1)[0]
    n_b   = min(n // 2, len(b_idx))
    n_a   = min(n - n_b, len(a_idx))
    idx   = np.concatenate([rng.choice(b_idx, n_b, replace=False),
                             rng.choice(a_idx, n_a, replace=False)])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _balanced_subset(X, y, n_per_class, seed=config.RANDOM_SEED):
    rng  = np.random.default_rng(seed)
    parts = []
    for cls in np.unique(y):
        pos = np.where(y == cls)[0]
        parts.append(rng.choice(pos, min(n_per_class, len(pos)), replace=False))
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _pick_model(policy, selector):
    available = set(selector.available())
    for tier in policy["model_hierarchy"]:
        name = TIER_TO_NAME.get(tier)
        if name in available:
            return name
    return next(iter(available))


def _run_episode(model, X_ep, y_ep, planner):
    """Run one episode. Uses predict_labels() rather than argmax(predict_proba())
    for the hard decision — see the matching fix and rationale in
    experiments/phase7_reflexion.py (D-P8.1: PegasosQSVC's sigmoid calibration
    is not centred at 0.5, so raw argmax can degenerate to a single-class
    prediction regardless of AUROC)."""
    proba       = model.predict_proba(X_ep)
    y_scores    = proba[:, 1]
    y_pred      = model.predict_labels(X_ep)
    confidences = proba.max(axis=1)
    drift_seen  = False
    for i in range(len(y_ep)):
        if planner.update_drift(float(y_pred[i] != y_ep[i])):
            drift_seen = True
    return y_scores, y_pred, confidences, drift_seen


# ---------------------------------------------------------------------------
# ADWIN-only baseline — triggers SWITCH_SUBSET on drift, nothing else
# ---------------------------------------------------------------------------

class ADWINOnlyReflector:
    """Minimal baseline: only fires SWITCH_SUBSET when ADWIN detects drift."""

    def reflect(self, report, memory, policy):
        if report.drift_detected:
            from datetime import datetime, timezone
            from agent.episodic_memory import Lesson
            return Lesson(
                lesson_id     = memory.next_lesson_id(),
                episode_id    = report.episode_id,
                action        = SWITCH_SUBSET,
                trigger       = "ADWIN drift detected",
                params        = {"pool_size": config.SWITCH_SUBSET_N_CROSS},
                timestamp     = datetime.now(timezone.utc).isoformat(),
                verbal_lesson = "",
                source        = "adwin_only",
            )
        return None


# ---------------------------------------------------------------------------
# Single-agent simulation run
# ---------------------------------------------------------------------------

def _run_simulation(
    agent_name: str,
    reflector,
    selector: ModelSelector,
    X8_ton_te, y_ton_te,
    X8_unsw, y_unsw,
    verbose: bool = True,
) -> list[dict]:
    """Run the 10-episode simulation with the given reflector."""
    # Fresh memory + planner per agent so policies don't bleed across agents
    sem_mem  = SemanticMemory(
        path=f"agent_state/semantic_memory_{agent_name.lower().replace(' ', '_')}.jsonl"
    )
    if isinstance(reflector, LLMReflexionAgent):
        reflector._sem_mem = sem_mem

    memory   = EpisodicMemory(
        log_path=f"agent_state/episodes_{agent_name.lower().replace(' ', '_')}.jsonl"
    )
    evaluator = Evaluator()
    planner   = PlanningModule()
    policy    = memory.get_policy()
    summaries = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Agent: {agent_name}")
        print(f"{'=' * 60}")

    for ep_id in range(N_EPISODES):
        is_cross     = ep_id >= N_WITHIN
        dataset_name = "NF-UNSW-NB15" if is_cross else "NF-ToN-IoT"
        X_pool       = X8_unsw if is_cross else X8_ton_te
        y_pool       = y_unsw  if is_cross else y_ton_te

        model_name = _pick_model(policy, selector)
        model      = selector.get(model_name)

        X_ep, y_ep = _sample_episode(X_pool, y_pool, ep_id)

        t0 = time.time()
        y_scores, y_pred, confidences, drift = _run_episode(model, X_ep, y_ep, planner)
        elapsed = time.time() - t0

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

        lesson = reflector.reflect(report, memory, policy)
        memory.record(report, lesson)
        policy = memory.get_policy()

        if lesson is not None and lesson.action == SWITCH_SUBSET:
            planner.reset_drift_detector()
            memory.acknowledge_reselect()

        if verbose:
            block = "CROSS" if is_cross else "WITHIN"
            print(f"  Ep {ep_id:2d} [{block}] {model_name:>18}  "
                  f"AUROC={report.auroc:.4f}  F1={report.binary_f1:.4f}  "
                  f"drift={str(drift):>5}  "
                  f"lesson={lesson.action if lesson else '—'}")
            if lesson and getattr(lesson, "verbal_lesson", ""):
                print(f"         verbal: {lesson.verbal_lesson[:70]}...")

        summaries.append({
            "episode_id":    ep_id,
            "block":         "CROSS" if is_cross else "WITHIN",
            "dataset":       dataset_name,
            "model_used":    model_name,
            "auroc":         report.auroc,
            "binary_f1":     report.binary_f1,
            "fpr_at_tpr95":  report.fpr_at_tpr95,
            "drift":         report.drift_detected,
            "lesson_action": lesson.action if lesson else None,
            "lesson_source": getattr(lesson, "source", None) if lesson else None,
            "verbal_lesson": getattr(lesson, "verbal_lesson", "") if lesson else "",
        })

    return summaries


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _print_comparison(all_results: dict[str, list[dict]]) -> None:
    agents = list(all_results.keys())
    n_ep   = len(next(iter(all_results.values())))

    print("\n" + "=" * 80)
    print("EPISODE-LEVEL COMPARISON")
    print("=" * 80)

    header = f"{'Ep':>3}  {'Block':>7}"
    for a in agents:
        header += f"  {a[:18]:>18}(AUROC/lesson)"
    print(header)
    print("-" * 80)

    for ep_id in range(n_ep):
        row = f"{ep_id:>3}  {all_results[agents[0]][ep_id]['block']:>7}"
        for a in agents:
            s = all_results[a][ep_id]
            lesson_str = (s["lesson_action"] or "—")[:12]
            row += f"  {s['auroc']:>6.4f}/{lesson_str:<12}"
        print(row)

    print("\n" + "=" * 80)
    print("AGENT SUMMARY")
    print("=" * 80)
    for a, summaries in all_results.items():
        auroc_vals = [s["auroc"] for s in summaries]
        cross_vals = [s["auroc"] for s in summaries if s["block"] == "CROSS"]
        lessons    = [s["lesson_action"] for s in summaries if s["lesson_action"]]
        from collections import Counter
        lcount = Counter(lessons)
        print(f"\n  {a}:")
        print(f"    Mean AUROC (all):   {np.mean(auroc_vals):.4f}")
        print(f"    Mean AUROC (cross): {np.mean(cross_vals):.4f}" if cross_vals else "")
        print(f"    Lessons fired:      {dict(lcount)}")
        verbal = [s["verbal_lesson"] for s in summaries if s.get("verbal_lesson")]
        if verbal:
            print(f"    Verbal lessons:     {len(verbal)} written")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true",
                        help="Include LLM agent (requires GROQ_API_KEY)")
    parser.add_argument("--no-rule", action="store_true",
                        help="Skip rule-based agent (faster if only testing LLM)")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 9 — LLM Reflexion agent evaluation")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print(f"\nLoading {config.NF_TON_CSV} ...")
    X_ton_df, y_ton, _ = read_nf(config.NF_TON_CSV)
    print(f"  {len(y_ton):,} rows")

    print(f"Loading {config.NF_UNSW_CSV} ...")
    X_unsw_df, y_unsw, _ = read_nf(config.NF_UNSW_CSV)
    print(f"  {len(y_unsw):,} rows")

    # ── 2. Split + scale ──────────────────────────────────────────────────────
    X_ton_tr_df, X_ton_te_df, y_ton_tr, y_ton_te = train_test_split(
        X_ton_df, y_ton, test_size=0.20, stratify=y_ton,
        random_state=config.RANDOM_SEED,
    )
    pre       = NFPreprocessor()
    X8_ton_tr = pre.fit_transform(X_ton_tr_df)
    X8_ton_te = pre.transform(X_ton_te_df)
    X8_unsw   = pre.transform(X_unsw_df)
    print(f"\n  ToN-IoT train {X8_ton_tr.shape}  test {X8_ton_te.shape}")
    print(f"  UNSW-NB15    {X8_unsw.shape}")

    # ── 3. Train models once (shared across all agents) ───────────────────────
    n_sub = config.QSVC_SUBSET_SIZE
    print(f"\nTraining QSVC (n={n_sub}) ...")
    t0  = time.time()
    idx = select_kernel_subset(X8_ton_tr, y_ton_tr, size=n_sub, method="kmeans")
    qsvc = PegasosQSVCModel(num_steps=config.PEGASOS_TAU).fit(
        X8_ton_tr[idx], y_ton_tr[idx])
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"Training RF ...")
    t0 = time.time()
    X_rf_tr, y_rf_tr = _balanced_subset(X8_ton_tr, y_ton_tr, RF_TRAIN_N // 2)
    rf = RandomForestModel(class_weight=None).fit(X_rf_tr, y_rf_tr)
    print(f"  done in {time.time()-t0:.2f}s")

    # Shared selector
    selector = ModelSelector()
    selector.register(qsvc)
    selector.register(rf)
    print(f"\nRegistered: {selector.available()}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)

    # ── 4. Run agents ─────────────────────────────────────────────────────────
    all_results: dict[str, list[dict]] = {}

    # A) LLM agent
    if args.llm:
        if not os.environ.get("GROQ_API_KEY"):
            print("\nWARNING: --llm specified but GROQ_API_KEY not set.")
            print("LLM agent will use heuristic fallback for all calls.")
        llm_agent = LLMReflexionAgent(verbose=True)
        all_results["LLM Agent"] = _run_simulation(
            "LLM Agent", llm_agent, selector,
            X8_ton_te, y_ton_te, X8_unsw, y_unsw,
        )

    # B) Rule-based
    if not args.no_rule:
        all_results["Rule-based"] = _run_simulation(
            "Rule-based", SelfReflector(), selector,
            X8_ton_te, y_ton_te, X8_unsw, y_unsw,
        )

    # C) ADWIN-only
    all_results["ADWIN-only"] = _run_simulation(
        "ADWIN-only", ADWINOnlyReflector(), selector,
        X8_ton_te, y_ton_te, X8_unsw, y_unsw,
    )

    # ── 5. Print comparison ───────────────────────────────────────────────────
    _print_comparison(all_results)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    out = {
        "experiment":  "phase9_llm_eval",
        "n_episodes":  N_EPISODES,
        "agents":      list(all_results.keys()),
        "results":     all_results,
    }
    path = f"{RESULTS_DIR}/phase9_comparison.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved -> {path}")


if __name__ == "__main__":
    main()
