"""
Quantum memory decoherence model.

Entangled pairs stored in quantum memories undergo decoherence,
modelled as exponential decay of the depolarisation parameter
with coherence time T₂.
"""

import numpy as np


def decohere(F_0: float, t: float, T2: float) -> float:
    """Fidelity after time t in memory with coherence time T₂.

    F(t) = ((4F₀ - 1) · exp(-t/T₂) + 1) / 4

    As t → ∞, F → 1/4 (maximally mixed state).
    At t = 0, F = F₀.

    Args:
        F_0: Initial fidelity at time of storage.
        t: Time elapsed (same units as T2).
        T2: Coherence time of the quantum memory.

    Returns:
        Fidelity after decoherence.
    """
    return ((4.0 * F_0 - 1.0) * np.exp(-t / T2) + 1.0) / 4.0


def decohere_batch(F_0: np.ndarray, t: np.ndarray, T2: float) -> np.ndarray:
    """Vectorised decoherence for arrays of fidelities and times.

    Args:
        F_0: Array of initial fidelities.
        t: Array of elapsed times.
        T2: Coherence time.

    Returns:
        Array of fidelities after decoherence.
    """
    return ((4.0 * F_0 - 1.0) * np.exp(-t / T2) + 1.0) / 4.0


def time_to_threshold(F_0: float, F_threshold: float, T2: float) -> float:
    """Time until fidelity drops to given threshold.

    Solving F(t) = F_threshold for t:
        t = -T₂ · ln((4·F_threshold - 1) / (4·F₀ - 1))

    Args:
        F_0: Initial fidelity (must be > F_threshold).
        F_threshold: Target fidelity threshold.
        T2: Coherence time.

    Returns:
        Time to reach threshold. Returns 0 if already below, inf if unreachable.
    """
    if F_0 <= F_threshold:
        return 0.0
    if F_threshold <= 0.25:
        return float('inf')

    p_0 = (4.0 * F_0 - 1.0)
    p_thresh = (4.0 * F_threshold - 1.0)

    if p_0 <= 0 or p_thresh <= 0:
        return float('inf')

    return -T2 * np.log(p_thresh / p_0)
