"""
Werner state model for quantum entanglement routing.

A Werner state is a mixture of a maximally entangled Bell state |Φ⁺⟩
and the maximally mixed state I/4:

    ρ_W(p) = p |Φ⁺⟩⟨Φ⁺| + (1-p)/4 · I₄

where p ∈ [0, 1] is the depolarisation parameter and the fidelity
F = ⟨Φ⁺|ρ_W|Φ⁺⟩ = (3p + 1)/4.
"""

import numpy as np


def werner_state_fidelity_to_p(F: float) -> float:
    """Convert fidelity to depolarisation parameter.

    Args:
        F: Fidelity ∈ [0.25, 1.0].

    Returns:
        p: Depolarisation parameter ∈ [0, 1].
    """
    return (4.0 * F - 1.0) / 3.0


def werner_state_p_to_fidelity(p: float) -> float:
    """Convert depolarisation parameter to fidelity.

    Args:
        p: Depolarisation parameter ∈ [0, 1].

    Returns:
        F: Fidelity ∈ [0.25, 1.0].
    """
    return (3.0 * p + 1.0) / 4.0


def werner_density_matrix(F: float) -> np.ndarray:
    """Construct 4×4 density matrix of a Werner state with fidelity F.

    Args:
        F: Fidelity ∈ [0.25, 1.0].

    Returns:
        rho: 4×4 complex density matrix.
    """
    # |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
    phi_plus = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    proj = np.outer(phi_plus, phi_plus.conj())
    identity = np.eye(4, dtype=np.complex128)
    rho = F * proj + (1.0 - F) / 3.0 * (identity - proj)
    return rho


def is_entangled(F: float) -> bool:
    """Check if Werner state with fidelity F is entangled.

    A Werner state is entangled iff F > 0.5 (equivalently p > 1/3).

    Args:
        F: Fidelity.

    Returns:
        True if the state is entangled.
    """
    return F > 0.5
