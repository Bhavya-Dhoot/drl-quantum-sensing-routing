"""
Quantum Fisher Information (QFI) computation for depolarised GHZ states.

QFI quantifies the ultimate precision of parameter estimation achievable
with a quantum state. For a depolarised N-qubit GHZ state under J_z
collective phase encoding:

    F_Q = (2F_GHZ - 1) · N²    when F_GHZ ≥ 0.5
    F_Q = 0                      when F_GHZ < 0.5

The Heisenberg limit is F_Q = N² and the standard quantum limit is F_Q = N.
"""

import numpy as np


def qfi_depolarised_ghz(p_ghz: float, N: int) -> float:
    """QFI lower bound for depolarised GHZ under J_z encoding.

    Args:
        p_ghz: Effective depolarisation parameter of the GHZ state.
        N: Number of sensor qubits.

    Returns:
        Quantum Fisher Information. Returns 0 if no quantum advantage.
    """
    F_ghz = p_ghz * (1.0 - 1.0 / 2**N) + 1.0 / 2**N
    if F_ghz < 0.5:
        return 0.0
    return (2.0 * F_ghz - 1.0) * N**2


def qfi_from_fidelity(F_ghz: float, N: int) -> float:
    """QFI directly from GHZ fidelity.

    Args:
        F_ghz: Fidelity of the GHZ state.
        N: Number of sensor qubits.

    Returns:
        Quantum Fisher Information.
    """
    if F_ghz < 0.5:
        return 0.0
    return (2.0 * F_ghz - 1.0) * N**2


def epsilon(F: float, N: int) -> float:
    """QFI degradation factor.

    ε(F, N) = 1 - p^N where p = (4F - 1)/3

    Measures how much QFI is lost relative to the ideal case.

    Args:
        F: Per-pair fidelity.
        N: Number of sensors.

    Returns:
        Degradation factor ∈ [0, 1].
    """
    p = (4.0 * F - 1.0) / 3.0
    return 1.0 - p**N


def f_min(N: int, delta: float) -> float:
    """Minimum per-pair fidelity for degradation ≤ delta.

    F_min(N, δ) = (3·(1-δ)^(1/N) + 1) / 4

    Args:
        N: Number of sensors.
        delta: Maximum allowed degradation.

    Returns:
        Minimum per-pair fidelity.
    """
    return (3.0 * (1.0 - delta) ** (1.0 / N) + 1.0) / 4.0


def qcrb(qfi: float) -> float:
    """Quantum Cramér-Rao Bound.

    QCRB = 1 / F_Q (single-shot, single parameter).

    Args:
        qfi: Quantum Fisher Information.

    Returns:
        QCRB value. Returns infinity if QFI = 0.
    """
    if qfi <= 0.0:
        return float('inf')
    return 1.0 / qfi


def heisenberg_scaling_exponent(qfi_values: list, N_values: list) -> float:
    """Estimate scaling exponent α̂ from log-log fit of QFI vs N.

    F_Q ∝ N^α̂  →  log(F_Q) = α̂ · log(N) + const

    Args:
        qfi_values: List of measured QFI values.
        N_values: Corresponding N values.

    Returns:
        Estimated scaling exponent.
    """
    log_n = np.log(np.array(N_values, dtype=np.float64))
    log_qfi = np.log(np.array(qfi_values, dtype=np.float64) + 1e-30)

    # Linear regression: log_qfi = alpha * log_n + beta
    coeffs = np.polyfit(log_n, log_qfi, 1)
    return float(coeffs[0])
