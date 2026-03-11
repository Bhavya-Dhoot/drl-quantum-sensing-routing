"""
Two-timescale VQS co-optimization for joint routing and sensing.

The Variational Quantum Sensing (VQS) parameters control the optimal
measurement basis for the GHZ state. The routing policy and VQS
parameters are updated on two different timescales:
    - Routing policy: slow timescale (lr = 3e-4)
    - VQS parameters: fast timescale (lr = 1e-3)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class VQSOptimizer:
    """Variational Quantum Sensing parameter optimizer.

    Parameterises the measurement basis angles for each sensor
    to maximise the Quantum Fisher Information. Uses gradient-based
    optimisation on a differentiable surrogate of the QFI.
    """

    def __init__(
        self,
        n_sensors: int,
        n_layers: int = 2,
        lr: float = 1e-3,
        device: str = 'cpu',
    ):
        """
        Args:
            n_sensors: Number of sensor nodes.
            n_layers: Number of VQS circuit layers.
            lr: Learning rate (fast timescale).
            device: Computation device.
        """
        self.n_sensors = n_sensors
        self.n_layers = n_layers
        self.device = torch.device(device)

        # VQS parameters: rotation angles for each sensor and layer
        # Shape: [n_layers, n_sensors, 3] for (Rx, Ry, Rz)
        self.params = nn.Parameter(
            torch.randn(n_layers, n_sensors, 3, device=self.device) * 0.1
        )

        self.optimizer = torch.optim.Adam([self.params], lr=lr)

    def compute_measurement_quality(self, fidelities: torch.Tensor) -> torch.Tensor:
        """Compute measurement quality factor from VQS parameters.

        The VQS circuit adjusts the measurement basis to maximise
        the Fisher information. We model this as a multiplicative
        factor on the QFI.

        Args:
            fidelities: Per-sensor delivered fidelities [N_sensors].

        Returns:
            Quality factor ∈ (0, 1].
        """
        # VQS quality: product of cos² terms from rotation angles
        cos_sq = torch.cos(self.params) ** 2  # [L, N, 3]
        # Average over layers and angles
        quality_per_sensor = cos_sq.mean(dim=(0, 2))  # [N]
        # Weight by fidelities
        weighted = quality_per_sensor * fidelities
        return weighted.mean()

    def compute_enhanced_qfi(
        self,
        base_qfi: float,
        fidelities: np.ndarray,
    ) -> Tuple[float, torch.Tensor]:
        """Compute VQS-enhanced QFI.

        Args:
            base_qfi: Base QFI from routing.
            fidelities: Per-sensor delivered fidelities.

        Returns:
            (enhanced_qfi, differentiable_qfi_tensor)
        """
        fid_tensor = torch.FloatTensor(fidelities).to(self.device)
        quality = self.compute_measurement_quality(fid_tensor)

        # Enhanced QFI: base_qfi × quality_factor
        enhanced = base_qfi * quality

        return enhanced.item(), enhanced

    def update(self, base_qfi: float, fidelities: np.ndarray) -> float:
        """Update VQS parameters to maximise QFI.

        Args:
            base_qfi: Base QFI from routing.
            fidelities: Per-sensor delivered fidelities.

        Returns:
            Loss value.
        """
        _, qfi_tensor = self.compute_enhanced_qfi(base_qfi, fidelities)

        # Maximise QFI = minimise -QFI
        loss = -qfi_tensor

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_params_dict(self) -> Dict[str, np.ndarray]:
        """Get current VQS parameters as numpy arrays."""
        return {
            'params': self.params.detach().cpu().numpy(),
        }

    def reset(self):
        """Reinitialise VQS parameters."""
        with torch.no_grad():
            self.params.data = torch.randn_like(self.params) * 0.1


class TwoTimescaleTrainer:
    """Coordinates two-timescale optimisation of routing and VQS.

    Slow timescale: Routing policy (PPO, lr = 3e-4)
    Fast timescale: VQS parameters (Adam, lr = 1e-3)

    The VQS parameters are updated more frequently (every step),
    while the routing policy is updated after collecting a rollout.
    """

    def __init__(
        self,
        ppo_agent,
        vqs_optimizer: VQSOptimizer,
        vqs_update_freq: int = 1,
        coupling_weight: float = 0.1,
    ):
        """
        Args:
            ppo_agent: PPO agent for routing.
            vqs_optimizer: VQS parameter optimizer.
            vqs_update_freq: VQS updates per environment step.
            coupling_weight: Weight for VQS contribution to reward.
        """
        self.ppo_agent = ppo_agent
        self.vqs_optimizer = vqs_optimizer
        self.vqs_update_freq = vqs_update_freq
        self.coupling_weight = coupling_weight

    def update_vqs(self, base_qfi: float, fidelities: np.ndarray) -> float:
        """Update VQS parameters (fast timescale).

        Args:
            base_qfi: Base QFI from routing.
            fidelities: Per-sensor fidelities.

        Returns:
            VQS loss.
        """
        total_loss = 0.0
        for _ in range(self.vqs_update_freq):
            loss = self.vqs_optimizer.update(base_qfi, fidelities)
            total_loss += loss
        return total_loss / max(self.vqs_update_freq, 1)

    def compute_coupled_reward(
        self,
        base_reward: float,
        base_qfi: float,
        fidelities: np.ndarray,
    ) -> float:
        """Compute reward with VQS coupling.

        Args:
            base_reward: Routing reward.
            base_qfi: QFI from routing alone.
            fidelities: Per-sensor fidelities.

        Returns:
            Enhanced reward.
        """
        enhanced_qfi, _ = self.vqs_optimizer.compute_enhanced_qfi(
            base_qfi, fidelities
        )
        vqs_bonus = self.coupling_weight * (enhanced_qfi - base_qfi)
        return base_reward + vqs_bonus
