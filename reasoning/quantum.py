"""Quantum classifier wrappers for Phase 5 of Q-Armor.

Three models, each implementing the ``BaseClassifier`` contract so they plug
into ``ModelSelector`` and the rest of the agent without change:

* ``PegasosQSVCModel`` — binary SVM trained with the stochastic Pegasos
  algorithm on the Phase-4 fidelity quantum kernel.
* ``VQCModel`` — variational quantum circuit classifier; uses the Phase-4
  ``CyberSecurityFeatureMap`` as the data-encoding layer and a
  ``RealAmplitudes`` ansatz, optimised with COBYLA.
* ``QuantumAnomalyDetector`` — quantum autoencoder trained on benign traffic
  only; reconstruction fidelity on trash qubits becomes the anomaly score.

All three are BINARY (0 = benign, 1 = attack). Multi-class attack typing is
the classical Stage-2 job (Phase 8).

Column convention shared by all three models:
    predict_proba[:, 0] = P(benign = 0)
    predict_proba[:, 1] = P(attack  = 1)
    classes_             = np.array([0, 1])
"""

from __future__ import annotations

import numpy as np
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import PegasosQSVC, VQC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import real_amplitudes

from agent import agent_config as config
from perception.feature_map import CyberSecurityFeatureMap
from reasoning.base import BaseClassifier


# ---------------------------------------------------------------------------
# PegasosQSVC wrapper
# ---------------------------------------------------------------------------

class PegasosQSVCModel(BaseClassifier):
    """Binary SVM using the Phase-4 fidelity quantum kernel (Pegasos algorithm).

    PegasosQSVC maps labels to {+1, -1} internally. With integer labels
    {0, 1}, it assigns label_pos = 0 (benign) and label_neg = 1 (attack),
    so its raw predict_proba columns are [P(attack), P(benign)]. This wrapper
    flips them to match the shared [P(benign), P(attack)] convention.
    """

    name = "pegasos_qsvc"

    def __init__(
        self,
        num_steps: int = config.PEGASOS_TAU,
        C: float = 1.0,
        feature_map: CyberSecurityFeatureMap | None = None,
    ) -> None:
        """Initialise.

        Args:
            num_steps: Pegasos stochastic gradient steps.
            C: SVM regularisation parameter.
            feature_map: Encoding circuit; defaults to ``CyberSecurityFeatureMap()``.
        """
        fm = feature_map or CyberSecurityFeatureMap()
        self._kernel = FidelityQuantumKernel(feature_map=fm)
        self._num_steps = num_steps
        self._C = C
        self._model: PegasosQSVC | None = None
        self._classes = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PegasosQSVCModel":
        """Train PegasosQSVC on binary labels {0, 1}.

        Args:
            X: Feature matrix (n, 8), values in [0, pi].
            y: Binary label array {0=benign, 1=attack}.

        Returns:
            Self, fitted.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._model = PegasosQSVC(
            quantum_kernel=self._kernel,
            C=self._C,
            num_steps=self._num_steps,
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities from sigmoid-transformed decision function.

        PegasosQSVC with {0,1} labels sets label_pos=0 (benign), so its raw
        output is [P(attack), P(benign)]; we flip to [P(benign), P(attack)].

        Args:
            X: Feature matrix (m, 8).

        Returns:
            Array (m, 2): column 0 = P(benign=0), column 1 = P(attack=1).
        """
        raw = self._model.predict_proba(np.asarray(X, dtype=float))
        return raw[:, ::-1].copy()

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Hard binary predictions using the PegasosQSVC native decision boundary.

        PegasosQSVC's predict_proba uses sigmoid calibration that is NOT centred
        at 0.5 — argmax on probabilities would always pick benign (class 0).
        Using the native SVM predict() instead applies the correct margin boundary.

        Returns:
            Integer array of shape (n_samples,) with values 0 (benign) or 1 (attack).
        """
        return np.asarray(self._model.predict(np.asarray(X, dtype=float)), dtype=int)

    @property
    def classes_(self) -> np.ndarray:
        """Binary class labels [0, 1] (benign, attack)."""
        return self._classes


# ---------------------------------------------------------------------------
# VQC wrapper
# ---------------------------------------------------------------------------

class VQCModel(BaseClassifier):
    """Variational Quantum Circuit binary classifier.

    Uses the Phase-4 ``CyberSecurityFeatureMap`` as the data-encoding layer
    and a ``RealAmplitudes`` ansatz with ``config.VQC_REPS`` repetitions.
    Trained with COBYLA (gradient-free, robust to shot noise).

    VQC with integer {0, 1} labels uses one-hot encoding (class 0 → column 0,
    class 1 → column 1), so predict_proba already follows the shared
    [P(benign), P(attack)] convention without any column flip.
    """

    name = "vqc"

    def __init__(
        self,
        max_iter: int = config.VQC_MAX_ITER,
        reps: int = config.VQC_REPS,
        feature_map: CyberSecurityFeatureMap | None = None,
        seed: int = config.RANDOM_SEED,
    ) -> None:
        """Initialise.

        Args:
            max_iter: COBYLA maximum function evaluations.
            reps: RealAmplitudes ansatz repetitions.
            feature_map: Encoding circuit; defaults to ``CyberSecurityFeatureMap()``.
            seed: Random seed for the Aer simulator sampler.
        """
        self._feature_map = feature_map or CyberSecurityFeatureMap()
        self._ansatz = real_amplitudes(config.N_QUBITS, reps=reps)
        self._max_iter = max_iter
        self._seed = seed
        self._model: VQC | None = None
        self._classes = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQCModel":
        """Train VQC on binary labels {0, 1}.

        Args:
            X: Feature matrix (n, 8), values in [0, pi].
            y: Binary label array {0=benign, 1=attack}.

        Returns:
            Self, fitted.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        sampler = SamplerV2(seed=self._seed)
        optimizer = COBYLA(maxiter=self._max_iter)
        self._model = VQC(
            feature_map=self._feature_map,
            ansatz=self._ansatz,
            optimizer=optimizer,
            sampler=sampler,
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax class probabilities from the trained VQC neural network.

        Args:
            X: Feature matrix (m, 8).

        Returns:
            Array (m, 2): column 0 = P(benign=0), column 1 = P(attack=1).
        """
        return self._model.predict_proba(np.asarray(X, dtype=float))

    @property
    def classes_(self) -> np.ndarray:
        """Binary class labels [0, 1] (benign, attack)."""
        return self._classes


# ---------------------------------------------------------------------------
# Quantum Anomaly Detector (QAE)
# ---------------------------------------------------------------------------

class QuantumAnomalyDetector(BaseClassifier):
    """Quantum Autoencoder anomaly detector.

    Architecture
    ------------
    A parameterised quantum circuit is appended after the Phase-4 feature map::

        |φ(x)⟩ → U(θ) → measure trash qubits 0..(N_TRASH-1)

    Training (benign samples only): maximise P(trash qubits = |0…0⟩) via
    COBYLA — this forces U(θ) to compress benign traffic into the latent
    qubits, leaving the trash qubits unentangled.

    Anomaly score: 1 − P(trash = 0…0).  High score → attack-like.

    At inference, the anomaly score is mapped to class probabilities::

        predict_proba[:, 0] = 1 − anomaly_score  (P benign)
        predict_proba[:, 1] = anomaly_score       (P attack)

    The QAE is trained in an UNSUPERVISED manner on benign data; it receives
    y to extract the benign subset but never uses attack labels.
    """

    name = "quantum_autoencoder"

    def __init__(
        self,
        n_trash: int = config.QAE_N_TRASH,
        reps: int = config.QAE_REPS,
        max_iter: int = config.QAE_MAX_ITER,
        shots: int = config.QAE_SHOTS,
        seed: int = config.RANDOM_SEED,
        feature_map: CyberSecurityFeatureMap | None = None,
    ) -> None:
        """Initialise.

        Args:
            n_trash: Number of qubits used for the reconstruction target.
            reps: ``RealAmplitudes`` ansatz repetitions.
            max_iter: COBYLA maximum function evaluations.
            shots: Measurement shots per circuit evaluation.
            seed: Random seed for reproducibility.
            feature_map: Encoding circuit; defaults to ``CyberSecurityFeatureMap()``.
        """
        self._n_trash = n_trash
        self._reps = reps
        self._max_iter = max_iter
        self._shots = shots
        self._seed = seed
        self._fm = feature_map or CyberSecurityFeatureMap()
        self._n_qubits = self._fm.num_qubits
        self._theta: np.ndarray | None = None
        self._classes = np.array([0, 1])

        self._circuit = self._build_circuit()
        # Sorted parameter list: x[0..7] first (feature map), then ansatz params
        self._sorted_params = sorted(self._circuit.parameters, key=lambda p: p.name)
        self._n_data_params = self._fm.num_parameters          # 8
        self._n_ansatz_params = len(self._sorted_params) - self._n_data_params

    def _build_circuit(self):
        """Compose feature map + RealAmplitudes ansatz with measurement."""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(self._n_qubits)
        qc.compose(self._fm, inplace=True)
        qc.compose(real_amplitudes(self._n_qubits, reps=self._reps), inplace=True)
        qc.measure_all()
        return qc

    def _fidelity_batch(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute P(trash = 0…0) for each sample in X with fixed ansatz theta.

        Args:
            theta: Ansatz parameters, shape (n_ansatz_params,).
            X: Feature matrix (n, n_data_params), values in [0, pi].

        Returns:
            Fidelity array (n,), each value in [0, 1].
        """
        n = len(X)
        # Build parameter matrix: rows are [x_i (8), theta (16)] but the
        # sorted parameter order is x[0..7] then ansatz params, so column
        # ordering matches sorted_params.
        param_matrix = np.zeros((n, len(self._sorted_params)))
        param_matrix[:, : self._n_data_params] = X
        param_matrix[:, self._n_data_params :] = theta

        sampler = SamplerV2(seed=self._seed)
        job = sampler.run([(self._circuit, param_matrix)], shots=self._shots)
        pub_result = job.result()[0]

        trash_mask = "0" * self._n_trash   # qubits 0..(n_trash-1) = last n_trash bits in bitstring

        fidelities = np.zeros(n)
        for i in range(n):
            counts = pub_result.data.meas.get_counts(i)
            total = sum(counts.values())
            zeros = sum(cnt for bits, cnt in counts.items() if bits[-self._n_trash :] == trash_mask)
            fidelities[i] = zeros / total if total > 0 else 0.0
        return fidelities

    def _loss(self, theta: np.ndarray, X_benign: np.ndarray) -> float:
        """Negative mean reconstruction fidelity (minimise = maximise fidelity).

        Args:
            theta: Ansatz parameters.
            X_benign: Benign feature matrix, shape (n_benign, 8).

        Returns:
            Scalar loss (negative mean fidelity).
        """
        return -float(np.mean(self._fidelity_batch(theta, X_benign)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumAnomalyDetector":
        """Optimise ansatz parameters on the benign subset of X.

        Args:
            X: Feature matrix (n, 8), values in [0, pi].
            y: Binary label array; only y==0 (benign) rows are used.

        Returns:
            Self, fitted.
        """
        from scipy.optimize import minimize

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        X_benign = X[y == 0]
        if len(X_benign) == 0:
            raise ValueError("No benign samples (y==0) found in training data.")

        rng = np.random.default_rng(self._seed)
        theta0 = rng.uniform(0, 2 * np.pi, self._n_ansatz_params)

        result = minimize(
            self._loss,
            theta0,
            args=(X_benign,),
            method="COBYLA",
            options={"maxiter": self._max_iter, "rhobeg": 0.5},
        )
        self._theta = result.x
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Compute the anomaly score (high = attack-like) for each row in X.

        Args:
            X: Feature matrix (n, 8), values in [0, pi].

        Returns:
            Anomaly scores in [0, 1] (1 − reconstruction fidelity).
        """
        if self._theta is None:
            raise RuntimeError("QuantumAnomalyDetector must be fitted before inference.")
        fidelities = self._fidelity_batch(self._theta, np.asarray(X, dtype=float))
        return 1.0 - fidelities

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Convert anomaly score to class probabilities.

        Args:
            X: Feature matrix (m, 8).

        Returns:
            Array (m, 2): column 0 = P(benign=0), column 1 = P(attack=1).
        """
        scores = self.anomaly_score(np.asarray(X, dtype=float))
        return np.column_stack([1.0 - scores, scores])

    @property
    def classes_(self) -> np.ndarray:
        """Binary class labels [0, 1] (benign, attack)."""
        return self._classes
