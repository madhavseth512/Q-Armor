"""Custom quantum feature map for Q-Armor perception.

`CyberSecurityFeatureMap` encodes the 8 smart-8 features into an 8-qubit state.
It is a hand-designed circuit (NOT ZZFeatureMap / PauliFeatureMap — those appear
only as named baselines), per the supervisor directive:

    Layer 1 (per rep): Ry(x_i) angle encoding on each qubit.
    Layer 2 (per rep): for each entanglement pair (i, j):  CX(i,j) · Rz(x_i*x_j, j) · CX(i,j)

The entanglement pairs come from the measured smart-8 correlation matrix
(agent_config.ENTANGLEMENT_PAIRS), so the map encodes the feature interactions that
actually exist in the data. Features are pre-scaled to (0, pi) (Phase 1), matching
the rotation-angle range.
"""

from __future__ import annotations

from qiskit.circuit import ParameterVector, QuantumCircuit

from agent import agent_config as config


class CyberSecurityFeatureMap(QuantumCircuit):
    """8-qubit data-encoding circuit with EDA-derived entanglement pairs."""

    def __init__(
        self,
        num_qubits: int = config.N_QUBITS,
        reps: int = config.FEATURE_MAP_REPS,
        entanglement_pairs: list[tuple[int, int]] | None = None,
        name: str = "CyberSecurityFeatureMap",
    ) -> None:
        """Build the parameterised feature-map circuit.

        Args:
            num_qubits: Number of qubits (= number of features).
            reps: Number of encode+entangle repetitions.
            entanglement_pairs: Qubit-index pairs to entangle; defaults to
                ``agent_config.ENTANGLEMENT_PAIRS``.
            name: Circuit name.
        """
        super().__init__(num_qubits, name=name)
        self.num_features = num_qubits
        self.reps = reps
        self.entanglement_pairs = entanglement_pairs or config.ENTANGLEMENT_PAIRS

        x = ParameterVector("x", num_qubits)
        self._x = x
        for _ in range(reps):
            for i in range(num_qubits):                       # Layer 1: angle encoding
                self.ry(x[i], i)
            for i, j in self.entanglement_pairs:              # Layer 2: pairwise entanglement
                self.cx(i, j)
                self.rz(x[i] * x[j], j)
                self.cx(i, j)

    @property
    def feature_parameters(self) -> ParameterVector:
        """The data-encoding parameters (one per feature)."""
        return self._x
