"""Two-stage hierarchical (cascade) classifier for Q-Armor — Option B.

A Stage-1 router predicts one of 4 coarse groups (Benign / Volumetric / Network /
Web); a Stage-2 specialist then classifies within the group. Groups with a single
member (Benign) need no specialist.

Final class probability is computed softly:
    P(class) = P(group) * P(class | group)
so the model emits a proper 15-class distribution and plugs into the agent via
the same BaseClassifier contract as any other model.

Optional Web reject path (``web_reject=True``): the Web specialist gains a
``"not_web"`` class trained on benign+network negatives. At inference the rejected
probability mass P(Web) * P(not_web | Web) is NOT assigned to a web label — it is
redistributed across the non-Web branches (proportional to their group
probability), so router contamination that lands in the Web group can be sent back
rather than forced into a false web positive.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE

from agent import agent_config as config
from data.preprocess import dynamic_smote_strategy
from reasoning.base import BaseClassifier
from reasoning.classical import RandomForestModel

NOT_WEB = "not_web"


class HierarchicalClassifier(BaseClassifier):
    """Stage-1 group router + per-group Stage-2 specialists (RandomForest)."""

    name = "hierarchical_rf"

    def __init__(self, web_reject: bool = False) -> None:
        """Initialise empty stages.

        Args:
            web_reject: If True, the Web specialist learns a ``not_web`` reject
                class and rejected mass is redistributed to non-Web branches.
        """
        self._web_reject = web_reject
        self._router: RandomForestModel | None = None
        self._specialists: dict[str, RandomForestModel] = {}
        self._single: dict[str, str] = {}
        self._classes = np.array(config.ATTACK_CLASSES)
        self._idx = {c: i for i, c in enumerate(config.ATTACK_CLASSES)}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HierarchicalClassifier":
        """Train the router (on groups) and one specialist per multi-class group.

        Args:
            X: Feature matrix (n_samples, n_features) of real training rows.
            y: Parent-class labels (n_samples,).

        Returns:
            Self, fitted.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Stage 1 — router: parent-level SMOTE, mapped to groups, group-balanced.
        strat = dynamic_smote_strategy(y)
        Xr, yr = SMOTE(random_state=config.RANDOM_SEED, sampling_strategy=strat).fit_resample(X, y)
        gr = np.array([config.GROUP_OF[p] for p in yr])
        self._router = RandomForestModel(class_weight="balanced").fit(Xr, gr)

        # Stage 2 — one specialist per group (balanced within the group).
        for group, members in config.ATTACK_GROUPS.items():
            if len(members) == 1:
                self._single[group] = members[0]
                continue
            mask = np.isin(y, members)
            Xg, yg = X[mask], y[mask]
            Xg2, yg2 = SMOTE(random_state=config.RANDOM_SEED,
                             sampling_strategy=dynamic_smote_strategy(yg)).fit_resample(Xg, yg)

            if group == "Web" and self._web_reject:
                Xg2, yg2 = self._add_not_web_negatives(X, y, Xg2, yg2)

            self._specialists[group] = RandomForestModel(class_weight="balanced").fit(Xg2, yg2)
        return self

    def _add_not_web_negatives(self, X, y, Xweb, yweb):
        """Append benign+network rows (labelled ``not_web``) to the Web training set."""
        neg_members = config.ATTACK_GROUPS["Benign"] + config.ATTACK_GROUPS["Network"]
        neg_idx = np.where(np.isin(y, neg_members))[0]
        n_neg = min(len(neg_idx), len(yweb))           # ~balance not_web vs all web
        sel = np.random.default_rng(config.RANDOM_SEED).choice(neg_idx, n_neg, replace=False)
        Xout = np.vstack([Xweb, X[sel]])
        yout = np.concatenate([yweb, np.full(n_neg, NOT_WEB)])
        return Xout, yout

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft cascade probabilities, with optional Web reject redistribution.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Array (n_samples, 15) aligned to ATTACK_CLASSES.
        """
        X = np.asarray(X, dtype=float)
        n = len(X)
        proba = np.zeros((n, len(self._classes)))
        gp = self._router.predict_proba(X)
        router_classes = list(self._router.classes_)
        web_gi = router_classes.index("Web")

        # --- Web branch (with optional reject) ---
        wspec = self._specialists["Web"]
        wp = wspec.predict_proba(X)
        wclasses = list(wspec.classes_)
        reject_mass = np.zeros(n)
        for k, cls in enumerate(wclasses):
            if cls == NOT_WEB:
                reject_mass = gp[:, web_gi] * wp[:, k]
            else:
                proba[:, self._idx[cls]] += gp[:, web_gi] * wp[:, k]

        # --- non-Web branches, absorbing the rejected mass proportionally ---
        nonweb = [gi for gi in range(len(router_classes)) if gi != web_gi]
        nonweb_total = gp[:, nonweb].sum(axis=1) + 1e-12
        for gi in nonweb:
            group = router_classes[gi]
            eff = gp[:, gi] + reject_mass * gp[:, gi] / nonweb_total
            if group in self._single:
                proba[:, self._idx[self._single[group]]] += eff
            else:
                sp = self._specialists[group].predict_proba(X)
                for k, cls in enumerate(self._specialists[group].classes_):
                    proba[:, self._idx[cls]] += eff * sp[:, k]
        return proba

    @property
    def classes_(self) -> np.ndarray:
        """The 15 parent classes, in ATTACK_CLASSES order."""
        return self._classes
