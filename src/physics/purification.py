"""
DEJMPS entanglement purification protocol.

The DEJMPS (Deutsch-Ekert-Jozsa-Macchiavello-Popescu-Sanpera) protocol
consumes two Werner pairs of the same fidelity and produces one pair
of higher fidelity (probabilistically).
"""


def dejmps_purify(F: float) -> float:
    """Output fidelity of DEJMPS purification consuming 2 pairs.

    F_out = (F² + (1-F)²/9) / (F² + 2F(1-F)/3 + 5(1-F)²/9)

    Purification is beneficial when F > 0.5. The protocol succeeds
    with probability equal to the denominator.

    Args:
        F: Input fidelity of each pair (must be equal).

    Returns:
        Output fidelity after successful purification.
    """
    num = F**2 + (1.0 - F)**2 / 9.0
    den = F**2 + 2.0 * F * (1.0 - F) / 3.0 + 5.0 * (1.0 - F)**2 / 9.0
    return num / den


def dejmps_success_probability(F: float) -> float:
    """Probability of successful DEJMPS purification.

    P_success = F² + 2F(1-F)/3 + 5(1-F)²/9

    Args:
        F: Input fidelity.

    Returns:
        Success probability.
    """
    return F**2 + 2.0 * F * (1.0 - F) / 3.0 + 5.0 * (1.0 - F)**2 / 9.0


def purification_rounds_needed(F_initial: float, F_target: float,
                                max_rounds: int = 20) -> int:
    """Number of purification rounds to reach target fidelity.

    Each round consumes 2^round pairs from the original supply.

    Args:
        F_initial: Starting fidelity.
        F_target: Target fidelity.
        max_rounds: Maximum allowed rounds.

    Returns:
        Number of rounds needed, or -1 if unreachable.
    """
    if F_initial >= F_target:
        return 0
    if F_initial <= 0.5:
        return -1  # Cannot purify below threshold

    F = F_initial
    for r in range(1, max_rounds + 1):
        F = dejmps_purify(F)
        if F >= F_target:
            return r
    return -1
