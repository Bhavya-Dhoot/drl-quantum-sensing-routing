"""
GHZ state assembly and fidelity computation.

Star-topology GHZ assembly: a hub node holds one qubit of each of N
entangled pairs. Applying CNOT gates and measuring produces an N-qubit
GHZ state with fidelity determined by the individual pair fidelities.
"""

from typing import Sequence


def ghz_parameter(fidelities: Sequence[float]) -> float:
    """Effective depolarisation parameter for N-qubit GHZ from N Werner pairs.

    p_GHZ = ∏_i (4F_i - 1) / 3

    Args:
        fidelities: Per-pair fidelities delivered to the hub.

    Returns:
        Effective depolarisation parameter p_GHZ.
    """
    p_ghz = 1.0
    for F in fidelities:
        p_ghz *= (4.0 * F - 1.0) / 3.0
    return p_ghz


def ghz_fidelity(fidelities: Sequence[float]) -> float:
    """GHZ fidelity from star-topology assembly.

    F_GHZ = p_GHZ · (1 - 1/2^N) + 1/2^N

    where p_GHZ = ∏_i (4F_i - 1)/3 and N = len(fidelities).

    Args:
        fidelities: Per-pair fidelities delivered to the hub.

    Returns:
        Fidelity of the assembled GHZ state.
    """
    N = len(fidelities)
    if N == 0:
        return 0.0
    p = ghz_parameter(fidelities)
    return p * (1.0 - 1.0 / 2**N) + 1.0 / 2**N


def ghz_fidelity_from_p(p_ghz: float, N: int) -> float:
    """GHZ fidelity given pre-computed p_GHZ and number of qubits.

    Args:
        p_ghz: Effective depolarisation parameter.
        N: Number of qubits.

    Returns:
        GHZ fidelity.
    """
    return p_ghz * (1.0 - 1.0 / 2**N) + 1.0 / 2**N
