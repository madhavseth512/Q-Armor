"""Phase 1 follow-up: which RAW columns carry the web-attack signal the 8
engineered features miss?

Rebuilds the 45-feature RandomForest (same two-sided sampling as the gates) and
runs per-class SHAP, then reports the top raw columns for the 5 application-layer
classes that collapsed under the 8 features (XSS, SqlInjection, CommandInjection,
Uploading_Attack, Backdoor_Malware). These columns are the candidate sources for
redesigned features (option b) — or, if the signal is smeared, evidence for
collapsing the rare classes (option a).

Run:  ./venv/Scripts/python.exe -m experiments.phase1_raw_diagnostic
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from agent import agent_config as config
from data.loader import load_dataset
from data.preprocess import dynamic_smote_strategy, majority_undersample_index

RF_KWARGS = dict(n_estimators=150, max_depth=16, n_jobs=-1, random_state=config.RANDOM_SEED)
WEB_CLASSES = ["XSS", "SqlInjection", "CommandInjection", "Uploading_Attack", "Backdoor_Malware"]
SHAP_SAMPLE_PER_CLASS = 60
TOP_N = 8


def _hr(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main() -> None:
    t0 = time.time()
    Xtr_full, ytr_full = load_dataset(config.TRAIN_CSV)
    Xva, yva = load_dataset(config.VALIDATION_CSV, validate=False)
    print(f"loaded in {time.time()-t0:.1f}s")

    keep = majority_undersample_index(ytr_full)
    Xtr, ytr = Xtr_full.loc[keep], ytr_full.loc[keep]
    del Xtr_full, ytr_full
    strategy = dynamic_smote_strategy(ytr)

    raw_cols = Xtr.columns.tolist()
    scaler = StandardScaler().fit(Xtr[raw_cols])
    Xtr_s, y_s = SMOTE(random_state=config.RANDOM_SEED, sampling_strategy=strategy).fit_resample(
        scaler.transform(Xtr[raw_cols]), ytr
    )
    rf = RandomForestClassifier(**RF_KWARGS).fit(Xtr_s, y_s)
    print(f"raw RF trained ({len(raw_cols)} features) in {time.time()-t0:.1f}s")

    # explanation subsample from the pristine validation set
    import shap

    Xva_s = pd.DataFrame(scaler.transform(Xva[raw_cols]), columns=raw_cols)
    Xva_s["y"] = yva.values
    parts = [
        Xva_s[Xva_s["y"] == c].sample(min((Xva_s["y"] == c).sum(), SHAP_SAMPLE_PER_CLASS),
                                      random_state=config.RANDOM_SEED)
        for c in config.ATTACK_CLASSES if (Xva_s["y"] == c).any()
    ]
    expl = pd.concat(parts)
    Xexpl = expl[raw_cols].to_numpy()

    sv = shap.TreeExplainer(rf).shap_values(Xexpl)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)  # (n, feat, class)
    imp = np.abs(sv).mean(axis=0)                                           # (feat, class)
    imp_df = pd.DataFrame(imp, index=raw_cols, columns=rf.classes_)

    eng_sources = {"Rate", "syn_count", "rst_count", "fin_count", "Header_Length",
                   "AVG", "Covariance", "urg_count", "TCP", "UDP", "ICMP", "SSH",
                   "DNS", "HTTPS"}

    _hr("TOP RAW COLUMNS PER COLLAPSED WEB-ATTACK CLASS")
    candidates: dict[str, float] = {}
    for cls in WEB_CLASSES:
        col = imp_df[cls].sort_values(ascending=False).head(TOP_N)
        print(f"\n{cls}:")
        for name, val in col.items():
            used = "  (already a feature source)" if name in eng_sources else ""
            candidates[name] = candidates.get(name, 0.0) + (val if name not in eng_sources else 0.0)
            print(f"    {name:<18} {val:.4f}{used}")

    _hr("AGGREGATE: NEW raw columns most useful across the 5 web classes")
    rank = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    for name, score in rank[:12]:
        if score > 0:
            print(f"    {name:<18} summed |SHAP| = {score:.4f}")

    print(f"\nTotal runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
