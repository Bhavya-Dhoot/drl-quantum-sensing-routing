"""
Demand generation for quantum sensing requests.

Generates sensing demands as a Poisson process, where each demand
specifies a set of sensor nodes that should be entangled via GHZ
state at the hub.
"""

import numpy as np
from typing import List, Optional, Tuple


class DemandGenerator:
    """Poisson-based sensing demand generator.

    Each demand specifies a subset of sensor nodes that must establish
    entangled pairs with the hub to form a GHZ state.
    """

    def __init__(
        self,
        sensor_nodes: List[int],
        hub_node: int,
        arrival_rate: float = 5.0,
        min_sensors: int = 2,
        max_sensors: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Args:
            sensor_nodes: List of sensor node IDs.
            hub_node: Hub node ID.
            arrival_rate: Mean number of demands per time step (Poisson λ).
            min_sensors: Minimum sensors per demand.
            max_sensors: Maximum sensors per demand.
            seed: Random seed.
        """
        self.sensor_nodes = list(sensor_nodes)
        self.hub_node = hub_node
        self.arrival_rate = arrival_rate
        self.min_sensors = min_sensors
        self.max_sensors = max_sensors or len(sensor_nodes)
        self.rng = np.random.RandomState(seed)
        self._demand_id = 0

    def generate(self, time_step: int = 0) -> List[dict]:
        """Generate demands for a single time step.

        Args:
            time_step: Current simulation time step.

        Returns:
            List of demand dicts, each with:
                - id: unique demand identifier
                - sensors: list of sensor node IDs
                - hub: hub node ID
                - arrival_time: time step of arrival
                - N: number of sensors
        """
        n_demands = self.rng.poisson(self.arrival_rate)
        demands = []

        for _ in range(n_demands):
            # Random number of sensors for this demand
            n_sensors = self.rng.randint(
                self.min_sensors,
                min(self.max_sensors, len(self.sensor_nodes)) + 1
            )
            # Random subset of sensor nodes
            selected = self.rng.choice(
                self.sensor_nodes, size=n_sensors, replace=False
            ).tolist()

            self._demand_id += 1
            demands.append({
                'id': self._demand_id,
                'sensors': selected,
                'hub': self.hub_node,
                'arrival_time': time_step,
                'N': n_sensors,
            })

        return demands

    def generate_fixed(self, n_sensors: Optional[int] = None) -> dict:
        """Generate a single demand with fixed number of sensors.

        Useful for evaluation with controlled N.

        Args:
            n_sensors: Number of sensors (default: all sensors).

        Returns:
            Demand dict.
        """
        n = n_sensors or len(self.sensor_nodes)
        n = min(n, len(self.sensor_nodes))
        selected = self.rng.choice(
            self.sensor_nodes, size=n, replace=False
        ).tolist()

        self._demand_id += 1
        return {
            'id': self._demand_id,
            'sensors': selected,
            'hub': self.hub_node,
            'arrival_time': 0,
            'N': n,
        }

    def reset(self, seed: Optional[int] = None):
        """Reset demand generator state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._demand_id = 0
