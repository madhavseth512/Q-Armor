"""SubsetRetrainer: implements the SWITCH_SUBSET lesson from the Reflexion loop.

When EpisodicMemory fires SWITCH_SUBSET (low AUROC + drift detected), the agent
needs to adapt Stage 1 (binary detection) to the new data distribution. This
module does the actual work:

  1. Takes a pool of samples from the new distribution (cross-domain data).
  2. Runs k-means balanced subset selection to pick QSVC_SUBSET_SIZE support
     vectors from the new distribution.
  3. Retrains PegasosQSVC on the new subset.
  4. Returns the retrained model (caller updates ModelSelector registry).

The Stage 2 typing model is NOT retrained here — it uses the coarse taxonomy
which is distribution-agnostic (label names are the same across datasets).
"""

from __future__ import annotations

import os
import time

import numpy as np

from agent import agent_config as config
from data.sampling import select_kernel_subset
from reasoning.base import BaseClassifier
from reasoning.quantum import PegasosQSVCModel


class SubsetRetrainer:
    """Retrains Stage 1 (binary QSVC) on a new data distribution.

    Stateless — one retrainer instance can be called multiple times as new
    cross-domain batches arrive. Each call produces a fresh fitted model;
    caller is responsible for updating ModelSelector.
    """

    def retrain(
        self,
        X_new:     np.ndarray,
        y_new:     np.ndarray,
        n_sub:     int = config.QSVC_SUBSET_SIZE,
        n_steps:   int = config.PEGASOS_TAU,
        verbose:   bool = True,
    ) -> tuple[PegasosQSVCModel, np.ndarray, np.ndarray]:
        """Select a kernel subset from cross-domain data and retrain QSVC.

        Args:
            X_new:   Preprocessed cross-domain feature matrix (n_samples, n_features).
                     Must already be scaled to [0, pi] using the same NFPreprocessor
                     that was fit on the original training dataset.
            y_new:   Binary labels (0=benign, 1=attack) for X_new.
            n_sub:   Target kernel subset size (default QSVC_SUBSET_SIZE=150).
            n_steps: PegasosQSVC training steps (default PEGASOS_TAU=100).
            verbose: Print timing and subset statistics.

        Returns:
            Tuple of (retrained PegasosQSVCModel, X_subset, y_subset).

        Raises:
            ValueError: If either class has fewer than n_sub//2 samples in X_new.
        """
        n_per_class = n_sub // 2
        n_b = int(np.sum(y_new == 0))
        n_a = int(np.sum(y_new == 1))
        if n_b < n_per_class or n_a < n_per_class:
            raise ValueError(
                f"X_new has too few samples per class for n_sub={n_sub}: "
                f"benign={n_b} (need {n_per_class}), attack={n_a} (need {n_per_class}). "
                f"Pass a larger X_new pool (SWITCH_SUBSET_N_CROSS={config.SWITCH_SUBSET_N_CROSS})."
            )

        if verbose:
            print(f"  [Retrainer] k-means subset selection: pool={len(X_new)}, n_sub={n_sub} ...")
        t0 = time.time()
        idx = select_kernel_subset(X_new, y_new, size=n_sub, method="kmeans")
        X_sub, y_sub = X_new[idx], y_new[idx]
        if verbose:
            print(f"  [Retrainer] subset {X_sub.shape}  labels {np.bincount(y_sub)}  ({time.time()-t0:.1f}s)")

        if verbose:
            print(f"  [Retrainer] training PegasosQSVC (n_steps={n_steps}) ...")
        t0 = time.time()
        model = PegasosQSVCModel(num_steps=n_steps).fit(X_sub, y_sub)
        if verbose:
            print(f"  [Retrainer] done in {time.time()-t0:.1f}s")

        return model, X_sub, y_sub

    def retrain_rf(
        self,
        X_new:  np.ndarray,
        y_new:  np.ndarray,
        n_per_class: int = config.CASCADE_STAGE2_N_PER_CLASS,
        verbose: bool = True,
    ) -> "BaseClassifier":
        """Retrain the Stage 2 RF typer on a cross-domain distribution.

        Args:
            X_new:       Scaled feature matrix (cross-domain).
            y_new:       Coarse type label strings (matching NF_COARSE_CLASSES).
            n_per_class: Max samples per class for balanced RF training.
            verbose:     Print progress.

        Returns:
            Fitted RandomForestModel (Stage 2).
        """
        from reasoning.classical import RandomForestModel

        rng = np.random.default_rng(config.RANDOM_SEED)
        parts = []
        for cls in np.unique(y_new):
            pos = np.where(y_new == cls)[0]
            parts.append(rng.choice(pos, min(n_per_class, len(pos)), replace=False))
        idx = np.concatenate(parts)
        rng.shuffle(idx)
        X_sub, y_sub = X_new[idx], y_new[idx]

        if verbose:
            cls_counts = {c: int(np.sum(y_sub == c)) for c in np.unique(y_sub)}
            print(f"  [Retrainer] RF Stage 2 training: {X_sub.shape}  {cls_counts}")
        t0 = time.time()
        rf = RandomForestModel(class_weight=None).fit(X_sub, y_sub)
        if verbose:
            print(f"  [Retrainer] RF done in {time.time()-t0:.1f}s")
        return rf
