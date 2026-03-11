"""Quantum physics primitives for entanglement routing simulation."""

from .werner import (
    werner_state_fidelity_to_p,
    werner_state_p_to_fidelity,
)
from .qfi import (
    qfi_depolarised_ghz,
    epsilon as qfi_epsilon,
    f_min,
)
from .ghz import ghz_parameter, ghz_fidelity
from .swapping import swap_fidelity, multihop_fidelity
from .purification import dejmps_purify
from .decoherence import decohere
