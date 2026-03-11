"""
QFI-based reward function for quantum sensing-aware routing.

Multi-objective reward combining:
- QFI improvement (ΔQFI)
- QCRB improvement over SQL baseline
- Latency penalty
- Memory usage penalty
- Heisenberg scaling bonus
"""


def compute_reward(
    delta_qfi: float,
    qcrb_improvement: float,
    latency_norm: float,
    memory_usage: float,
    heisenberg_est: float,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma_r: float = 0.1,
    delta_w: float = 0.05,
    epsilon_r: float = 0.2,
) -> float:
    """Sensing-performance-aware reward.

    R = α · ΔQFI + β · QCRB_improvement - γ · latency - δ · memory + ε · heis_bonus

    Args:
        delta_qfi: Change in total QFI from previous evaluation.
        qcrb_improvement: Improvement in QCRB relative to SQL.
        latency_norm: Normalised routing latency ∈ [0, 1].
        memory_usage: Normalised memory usage ∈ [0, 1].
        heisenberg_est: Estimated Heisenberg scaling ratio.
        alpha: Weight for QFI improvement.
        beta: Weight for QCRB improvement.
        gamma_r: Weight for latency penalty.
        delta_w: Weight for memory penalty.
        epsilon_r: Weight for Heisenberg bonus.

    Returns:
        Scalar reward.
    """
    heis_bonus = max(0.0, heisenberg_est - 1.5)
    return (alpha * delta_qfi
            + beta * qcrb_improvement
            - gamma_r * latency_norm
            - delta_w * memory_usage
            + epsilon_r * heis_bonus)


def compute_communication_reward(
    throughput: float,
    latency_norm: float,
    memory_usage: float,
    alpha: float = 1.0,
    gamma_r: float = 0.1,
    delta_w: float = 0.05,
) -> float:
    """Communication-oriented reward (for ablation study).

    Replaces QFI-based components with throughput maximisation.

    Args:
        throughput: Number of successfully delivered pairs per unit time.
        latency_norm: Normalised routing latency.
        memory_usage: Normalised memory usage.
        alpha: Weight for throughput.
        gamma_r: Weight for latency penalty.
        delta_w: Weight for memory penalty.

    Returns:
        Scalar reward.
    """
    return (alpha * throughput
            - gamma_r * latency_norm
            - delta_w * memory_usage)
