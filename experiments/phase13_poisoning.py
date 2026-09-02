"""Phase 13 — Confirmatory Protocol: adaptation-pool poisoning study.

Closes the "adaptation-pool poisoning study" item from docs/paper.tex's
Confirmatory Protocol: "A poisoning study should contaminate the adaptation
pool at controlled rates and measure whether the persistent controller
amplifies malicious adaptation."

Threat model (matches the paper's existing Threat Model section, which
already states the system does not defend against adaptation-pool
poisoning -- this experiment quantifies that stated limitation rather than
introducing a new claim): an attacker who can inject a controlled fraction
of the SWITCH_SUBSET target-domain adaptation pool flips ATTACK-labelled
rows to BENIGN before retraining -- the realistic "hide my traffic from the
next retrain" attack, not generic random label noise.

For poison_rate in {0, 0.05, 0.1, 0.2, 0.3, 0.5}:
  1. Draw the SAME 300-sample balanced UNSW-NB15 adaptation pool convention
     as experiments/phase8_final_eval.py's T3 (SWITCH_SUBSET_N_CROSS).
  2. Flip poison_rate of the pool's ATTACK rows to BENIGN.
  3. Retrain (PegasosQSVC via agent.retrainer.SubsetRetrainer; classical
     baselines via a matching k-means subset) on the poisoned pool.
  4. Evaluate on the CLEAN capped UNSW-NB15 test split (never poisoned).
  5. Feed the retrained model through 3 synthetic post-adaptation episodes
     (Evaluator -> SelfReflector -> EpisodicMemory, same components as
     experiments/phase7_reflexion.py) to check whether REINFORCE fires and
     locks in the poisoned model -- i.e. whether the persistent controller
     has any built-in defence (expected: no, by design; this measures that).

Run:  ./venv/Scripts/python.exe -m experiments.phase13_poisoning --confirm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from agent import agent_config as config
from agent.cascade_evaluator import _fpr_at_tpr
from agent.episodic_memory import REINFORCE, EpisodicMemory
from agent.evaluator import Evaluator
from agent.reflector import SelfReflector
from agent.retrainer import SubsetRetrainer
from data.nf_loader import NFPreprocessor, read_nf
from data.sampling import select_kernel_subset
from reasoning.classical import GBTModel, RandomForestModel
from reasoning.quantum import PegasosQSVCModel

RESULTS_DIR = "results/phase13"
POISON_RATES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
EPISODE_SIZE = config.EPISODE_SIZE

MODEL_REGISTRY = {
    "pegasos_qsvc":  lambda: PegasosQSVCModel(num_steps=config.PEGASOS_TAU),
    "random_forest": lambda: RandomForestModel(class_weight=None),
    "gbt":           lambda: GBTModel(),
}


def _poison_pool(X_pool, y_pool, rate, seed):
    """Flip `rate` fraction of ATTACK rows to BENIGN (evasion-style poisoning)."""
    y_poisoned = y_pool.copy()
    attack_idx = np.where(y_pool == 1)[0]
    n_flip = int(round(rate * len(attack_idx)))
    if n_flip > 0:
        rng = np.random.default_rng(seed)
        flip_idx = rng.choice(attack_idx, n_flip, replace=False)
        y_poisoned[flip_idx] = 0
    return y_poisoned, n_flip


def _post_adapt_episodes(model, X_pool, y_pool, seed_base):
    """3 synthetic post-adaptation episodes; returns whether REINFORCE fired
    and the AUROC trajectory, using the exact same reflexive-control loop as
    experiments/phase7_reflexion.py."""
    memory = EpisodicMemory(log_path=f"{RESULTS_DIR}/_scratch_episodes.jsonl")
    if os.path.exists(memory._log_path):
        os.remove(memory._log_path)
    memory = EpisodicMemory(log_path=f"{RESULTS_DIR}/_scratch_episodes.jsonl")
    evaluator = Evaluator()
    reflector = SelfReflector()
    policy = memory.get_policy()

    aurocs = []
    reinforced = False
    for i in range(3):
        rng = np.random.default_rng(seed_base + i)
        b_idx = np.where(y_pool == 0)[0]
        a_idx = np.where(y_pool == 1)[0]
        n_b = min(EPISODE_SIZE // 2, len(b_idx))
        n_a = min(EPISODE_SIZE - n_b, len(a_idx))
        idx = np.concatenate([rng.choice(b_idx, n_b, replace=False),
                               rng.choice(a_idx, n_a, replace=False)])
        X_ep, y_ep = X_pool[idx], y_pool[idx]

        proba = model.predict_proba(X_ep)
        y_pred = model.predict_labels(X_ep)
        report = evaluator.evaluate(
            episode_id=i, model_used=model.name, dataset="post_adapt_probe",
            y_true=y_ep, y_scores=proba[:, 1], y_pred=y_pred,
            confidences=proba.max(axis=1), drift_detected=False,
        )
        lesson = reflector.reflect(report, memory, policy)
        memory.record(report, lesson)
        policy = memory.get_policy()
        aurocs.append(report.auroc)
        if lesson is not None and lesson.action == REINFORCE:
            reinforced = True

    return reinforced, aurocs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--models", type=str, default="pegasos_qsvc,random_forest,gbt")
    parser.add_argument("--eval-cap", type=int, default=1000,
                        help="Clean eval-set size. QSVC predict cost scales with "
                             "n_train x n_test -- shrink this for the quantum arm "
                             "(6 poison-rate retrains x a 1000-sample predict each "
                             "is prohibitively slow; see phase10/phase11's --eval-cap).")
    args = parser.parse_args()
    models = args.models.split(",")

    if not args.confirm:
        print("Phase 13: adaptation-pool poisoning study.")
        print(f"  poison_rates={POISON_RATES}  models={models}")
        print("  Re-run with --confirm to proceed.")
        sys.exit(0)

    print("=" * 70)
    print(f"Phase 13: poisoning study  rates={POISON_RATES}  models={models}")
    print("=" * 70)

    X_ton_df, y_ton_bin, _ = read_nf(config.NF_TON_CSV)
    X_unsw_df, y_unsw_bin, _ = read_nf(config.NF_UNSW_CSV)
    X_unsw_tr_df, X_unsw_te_df, y_unsw_bin_tr, y_unsw_bin_te = train_test_split(
        X_unsw_df, y_unsw_bin, test_size=0.20, stratify=y_unsw_bin,
        random_state=config.RANDOM_SEED)

    pre = NFPreprocessor()
    X_ton_tr_df, _, y_ton_bin_tr, _ = train_test_split(
        X_ton_df, y_ton_bin, test_size=0.20, stratify=y_ton_bin,
        random_state=config.RANDOM_SEED)
    pre.fit_transform(X_ton_tr_df)  # fit scaler on source, exactly as elsewhere
    X8_unsw_tr = pre.transform(X_unsw_tr_df)
    X8_unsw_te = pre.transform(X_unsw_te_df)

    # Same capped clean test subset as phase8_final_eval.py / phase10.
    QSVC_CAP = args.eval_cap
    rng900 = np.random.default_rng(config.RANDOM_SEED + 900)
    b = np.where(y_unsw_bin_te == 0)[0]
    a = np.where(y_unsw_bin_te == 1)[0]
    te_idx = np.concatenate([
        rng900.choice(b, min(QSVC_CAP // 2, len(b)), replace=False),
        rng900.choice(a, min(QSVC_CAP // 2, len(a)), replace=False),
    ])
    X_test, y_test = X8_unsw_te[te_idx], y_unsw_bin_te[te_idx]
    print(f"Clean eval subset: {X_test.shape}  binary {np.bincount(y_test)}")

    # Same 300-sample balanced adaptation pool convention as phase8's T3.
    n_cross = config.SWITCH_SUBSET_N_CROSS
    b_pool = np.where(y_unsw_bin_tr == 0)[0]
    a_pool = np.where(y_unsw_bin_tr == 1)[0]
    rng2 = np.random.default_rng(config.RANDOM_SEED + 901)
    pool_idx = np.concatenate([
        rng2.choice(b_pool, min(n_cross // 2, len(b_pool)), replace=False),
        rng2.choice(a_pool, min(n_cross // 2, len(a_pool)), replace=False),
    ])
    X_pool, y_pool_clean = X8_unsw_tr[pool_idx], y_unsw_bin_tr[pool_idx]
    print(f"Adaptation pool: {X_pool.shape}  labels {np.bincount(y_pool_clean)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    for model_name in models:
        print(f"\n{'=' * 60}\nMODEL: {model_name}\n{'=' * 60}")
        results[model_name] = []

        for rate in POISON_RATES:
            y_poisoned, n_flip = _poison_pool(X_pool, y_pool_clean, rate,
                                               seed=config.RANDOM_SEED + 3000)
            t0 = time.time()
            if model_name == "pegasos_qsvc":
                retrainer = SubsetRetrainer()
                m, _, _ = retrainer.retrain(X_pool, y_poisoned,
                                             n_steps=config.PEGASOS_TAU, verbose=False)
            else:
                sub_idx = select_kernel_subset(X_pool, y_poisoned,
                                                size=config.QSVC_SUBSET_SIZE, method="kmeans")
                m = MODEL_REGISTRY[model_name]().fit(X_pool[sub_idx], y_poisoned[sub_idx])
            elapsed = time.time() - t0

            proba = m.predict_proba(X_test)
            y_pred = m.predict_labels(X_test)
            auroc = float(roc_auc_score(y_test, proba[:, 1]))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            fpr95 = _fpr_at_tpr(y_test, proba[:, 1])

            reinforced, probe_aurocs = _post_adapt_episodes(
                m, X8_unsw_tr, y_unsw_bin_tr, seed_base=config.RANDOM_SEED + 4000)

            row = {
                "poison_rate": rate, "n_flipped": n_flip,
                "auroc": auroc, "f1": f1, "fpr95": fpr95,
                "reinforce_fired_post_adapt": reinforced,
                "post_adapt_probe_aurocs": probe_aurocs,
                "train_time_s": round(elapsed, 1),
            }
            results[model_name].append(row)
            print(f"  poison={rate:.2f} (n_flip={n_flip:3d})  AUROC={auroc:.4f}  "
                  f"F1={f1:.4f}  FPR95={fpr95:.4f}  "
                  f"REINFORCE_fires={reinforced}  ({elapsed:.1f}s)")

            # Merge with any existing file rather than overwrite -- models are
            # often run in separate invocations (QSVC is much slower than the
            # classical baselines), and this checkpoints after every poison
            # rate, not just at the end of a model, so a slow QSVC run doesn't
            # lose progress if interrupted.
            out_path = f"{RESULTS_DIR}/phase13_poisoning_metrics.json"
            existing_results = {}
            if os.path.exists(out_path):
                with open(out_path) as ef:
                    existing_results = json.load(ef).get("results", {})
            existing_results[model_name] = results[model_name]
            with open(out_path, "w") as f:
                json.dump({"experiment": "phase13_poisoning",
                           "poison_rates": POISON_RATES, "eval_cap": QSVC_CAP,
                           "pool_size": n_cross, "results": existing_results}, f, indent=2)

    scratch = f"{RESULTS_DIR}/_scratch_episodes.jsonl"
    if os.path.exists(scratch):
        os.remove(scratch)

    print(f"\nDone -> {RESULTS_DIR}/phase13_poisoning_metrics.json")


if __name__ == "__main__":
    main()
