"""
Evaluation metrics for quantum sensing routing.

Includes:
- Scaling exponent estimation
- QCRB computation
- Satisfaction ratio
- Statistical aggregation utilities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


def estimate_scaling_exponent(
    N_values: List[int],
    qfi_values: List[float],
) -> Tuple[float, float, float]:
    """Estimate QFI scaling exponent via log-log linear regression.

    F_Q ∝ N^α̂  →  log(F_Q) = α̂ · log(N) + const

    Args:
        N_values: List of sensor counts.
        qfi_values: Corresponding QFI values.

    Returns:
        (alpha_hat, alpha_std, r_squared): estimated exponent, std error, R².
    """
    log_n = np.log(np.array(N_values, dtype=np.float64))
    log_qfi = np.log(np.array(qfi_values, dtype=np.float64) + 1e-30)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_qfi)

    return float(slope), float(std_err), float(r_value ** 2)


def compute_qcrb(qfi: float) -> float:
    """Compute Quantum Cramér-Rao Bound.

    Args:
        qfi: Quantum Fisher Information.

    Returns:
        QCRB = 1/F_Q.
    """
    if qfi <= 0:
        return float('inf')
    return 1.0 / qfi


def satisfaction_ratio(
    qfi_values: List[float],
    N_values: List[int],
    threshold_fraction: float = 0.5,
) -> float:
    """Fraction of demands where QFI exceeds threshold.

    Threshold = threshold_fraction × N² (fraction of Heisenberg limit).

    Args:
        qfi_values: List of achieved QFI values.
        N_values: Corresponding sensor counts.
        threshold_fraction: Fraction of Heisenberg limit.

    Returns:
        Satisfaction ratio ∈ [0, 1].
    """
    if not qfi_values:
        return 0.0

    satisfied = sum(
        1 for qfi, n in zip(qfi_values, N_values)
        if qfi >= threshold_fraction * n**2
    )
    return satisfied / len(qfi_values)


def aggregate_seeds(
    seed_results: List[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    """Aggregate results across seeds.

    Args:
        seed_results: List of result dicts, one per seed.

    Returns:
        Dict mapping metric name → (mean, 95% CI half-width).
    """
    if not seed_results:
        return {}

    keys = seed_results[0].keys()
    aggregated = {}

    for key in keys:
        values = [r[key] for r in seed_results if isinstance(r.get(key), (int, float))]
        if values:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=min(1, len(values)-1)))
            ci_95 = 1.96 * std / np.sqrt(max(len(values), 1))
            aggregated[key] = (mean, ci_95)

    return aggregated


def compute_heisenberg_ratio(qfi: float, N: int) -> float:
    """Ratio of achieved QFI to Heisenberg limit.

    Args:
        qfi: Achieved QFI.
        N: Number of sensors.

    Returns:
        Ratio QFI / N².
    """
    return qfi / max(N**2, 1)


def compute_sql_ratio(qfi: float, N: int) -> float:
    """Ratio of achieved QFI to Standard Quantum Limit.

    Args:
        qfi: Achieved QFI.
        N: Number of sensors.

    Returns:
        Ratio QFI / N.
    """
    return qfi / max(N, 1)
