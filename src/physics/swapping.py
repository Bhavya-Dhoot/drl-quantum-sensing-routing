"""
Entanglement swapping for quantum routing.

BSM-based entanglement swapping of two Werner states produces a new
Werner state with degraded fidelity.
"""

from typing import List, Sequence


def swap_fidelity(F_A: float, F_B: float) -> float:
    """Fidelity after BSM-based entanglement swapping of two Werner states.

    Given two Werner states with fidelities F_A and F_B, a Bell-state
    measurement at the intermediate node produces a new Werner state with:
        F_out = ((4F_A - 1)(4F_B - 1) + 3) / 12

    Args:
        F_A: Fidelity of the first link.
        F_B: Fidelity of the second link.

    Returns:
        Output fidelity after swapping.
    """
    return ((4.0 * F_A - 1.0) * (4.0 * F_B - 1.0) + 3.0) / 12.0


def multihop_fidelity(link_fidelities: Sequence[float]) -> float:
    """Fidelity after sequential swapping along a multi-hop path.

    For a path with K links, the total depolarisation parameter is:
        p_total = ∏_i p_i = ∏_i (4F_i - 1)/3

    And the delivered fidelity is:
        F_delivered = (3 · p_total + 1) / 4

    Args:
        link_fidelities: List of per-link fidelities along the path.

    Returns:
        End-to-end delivered fidelity.
    """
    if len(link_fidelities) == 0:
        return 0.25  # maximally mixed

    p_total = 1.0
    for F in link_fidelities:
        p_total *= (4.0 * F - 1.0) / 3.0

    return (3.0 * p_total + 1.0) / 4.0


def swap_success_probability(eta: float = 0.5) -> float:
    """Probability of successful Bell-state measurement.

    With linear optics, BSM succeeds with probability ≤ 0.5.
    With auxiliary photons, higher rates are possible.

    Args:
        eta: BSM success probability (default 0.5 for linear optics).

    Returns:
        Success probability.
    """
    return eta
