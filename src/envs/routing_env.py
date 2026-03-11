"""
Gymnasium-compatible RL environment for quantum sensing-aware routing.

The agent receives graph observations (node and edge features) and
selects next-hop routing decisions to maximize QFI of assembled GHZ states.
"""

import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

from ..network.topology import create_topology, get_sensor_nodes, get_hub_node
from ..network.quantum_network import QuantumNetwork
from ..network.demand import DemandGenerator
from ..physics.qfi import qfi_depolarised_ghz, f_min
from ..physics.ghz import ghz_parameter, ghz_fidelity
from ..physics.swapping import multihop_fidelity


class QuantumRoutingEnv(gym.Env):
    """Quantum sensing-aware entanglement routing environment.

    Observation space: Dict with node_features, edge_features, edge_index, action_mask
    Action space: Discrete (next-hop selection)
    Reward: QFI-based multi-objective reward
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Environment configuration dict.
        """
        super().__init__()
        self.config = config or {}

        # Topology setup
        topo_name = self.config.get('topology', 'nsfnet')
        topo_config = self.config.get('topology_config', {})
        self.topology = create_topology(topo_name, topo_config)
        self.n_nodes = self.topology.number_of_nodes()
        self.n_edges = self.topology.number_of_edges()

        # Network simulator
        self.network = QuantumNetwork(self.topology, self.config)

        # Demand generator
        self.sensors = get_sensor_nodes(self.topology)
        self.hub = get_hub_node(self.topology)
        self.demand_gen = DemandGenerator(
            sensor_nodes=self.sensors,
            hub_node=self.hub,
            arrival_rate=self.config.get('arrival_rate', 1.0),
            min_sensors=self.config.get('min_sensors', 2),
            max_sensors=self.config.get('max_sensors', None),
            seed=self.config.get('seed', 42),
        )

        # Node feature dimension: [type, m, T2, n_q, occupied, local_qfi]
        self.node_feat_dim = 6
        # Edge feature dimension: [F_ij, age, p_ij, d_ij]
        self.edge_feat_dim = 4

        # Action space: select next hop from neighbors
        self.max_degree = max(dict(self.topology.degree()).values())
        self.action_space = spaces.Discrete(self.max_degree)

        # Observation space (flat for compatibility, actual is graph-structured)
        self.observation_space = spaces.Dict({
            'node_features': spaces.Box(
                low=-1.0, high=10.0,
                shape=(self.n_nodes, self.node_feat_dim),
                dtype=np.float32
            ),
            'edge_index': spaces.Box(
                low=0, high=self.n_nodes,
                shape=(2, 2 * self.n_edges),
                dtype=np.int64
            ),
            'edge_features': spaces.Box(
                low=-1.0, high=10.0,
                shape=(2 * self.n_edges, self.edge_feat_dim),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1,
                shape=(self.max_degree,),
                dtype=np.float32
            ),
            'current_node': spaces.Discrete(self.n_nodes),
            'target_node': spaces.Discrete(self.n_nodes),
        })

        # Episode state
        self.current_demand: Optional[dict] = None
        self.current_sensor_idx: int = 0
        self.current_node: int = 0
        self.target_node: int = self.hub
        self.path: List[int] = []
        self.visited: set = set()
        self.step_count: int = 0
        self.max_steps: int = self.config.get('max_steps', 50)
        self.delivered_fidelities: Dict[int, float] = {}
        self.prev_qfi: float = 0.0
        self.episode_qfi: float = 0.0

        # Reward weights
        self.alpha = self.config.get('alpha', 1.0)
        self.beta = self.config.get('beta', 0.3)
        self.gamma_r = self.config.get('gamma_r', 0.1)
        self.delta_w = self.config.get('delta_w', 0.05)
        self.epsilon_r = self.config.get('epsilon_r', 0.2)

        # Precompute edge index (bidirectional)
        self._edge_index = self._build_edge_index()
        self._neighbor_map = self._build_neighbor_map()

        # Steps for network evolution
        self.network_steps_per_action = self.config.get('network_steps_per_action', 5)

    def _build_edge_index(self) -> np.ndarray:
        """Build edge index array for GNN (COO format, bidirectional)."""
        edges = list(self.topology.edges())
        src = [u for u, v in edges] + [v for u, v in edges]
        dst = [v for u, v in edges] + [u for u, v in edges]
        return np.array([src, dst], dtype=np.int64)

    def _build_neighbor_map(self) -> Dict[int, List[int]]:
        """Build neighbor list for each node."""
        return {n: sorted(list(self.topology.neighbors(n)))
                for n in self.topology.nodes()}

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment for a new episode.

        Args:
            seed: Random seed.
            options: Optional reset options. Can include 'n_sensors' for fixed N.

        Returns:
            (observation, info) tuple.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.demand_gen.reset(seed)

        self.network.reset(seed=seed)
        self.step_count = 0
        self.delivered_fidelities = {}
        self.prev_qfi = 0.0
        self.episode_qfi = 0.0

        # Generate network state (let some pairs form)
        for _ in range(20):
            self.network.step()

        # Generate demand
        options = options or {}
        n_sensors = options.get('n_sensors', None)
        if n_sensors:
            self.current_demand = self.demand_gen.generate_fixed(n_sensors)
        else:
            demands = self.demand_gen.generate(0)
            if demands:
                self.current_demand = demands[0]
            else:
                self.current_demand = self.demand_gen.generate_fixed()

        # Start routing for the first sensor
        self.current_sensor_idx = 0
        self._start_sensor_routing()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _start_sensor_routing(self) -> None:
        """Begin routing for the current sensor in the demand."""
        sensors = self.current_demand['sensors']
        if self.current_sensor_idx < len(sensors):
            self.current_node = sensors[self.current_sensor_idx]
            self.target_node = self.hub
            self.path = [self.current_node]
            self.visited = {self.current_node}
        else:
            self.current_node = self.hub
            self.target_node = self.hub

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one routing step.

        Args:
            action: Index into the neighbor list of the current node.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        self.step_count += 1

        # Advance network
        for _ in range(self.network_steps_per_action):
            self.network.step()

        # Get valid neighbors
        neighbors = self._neighbor_map[self.current_node]
        action_mask = self._get_action_mask()

        # Clamp action
        if action >= len(neighbors):
            action = 0
        if not action_mask[action]:
            # Select first valid action
            valid = np.where(action_mask[:len(neighbors)])[0]
            if len(valid) > 0:
                action = valid[0]
            else:
                # No valid actions, terminate
                return self._get_obs(), -1.0, True, False, self._get_info()

        next_node = neighbors[action]

        # Update path
        self.path.append(next_node)
        self.visited.add(next_node)
        self.current_node = next_node

        reward = 0.0
        terminated = False
        truncated = False

        # Check if we reached the hub
        if self.current_node == self.target_node:
            # Route this path and compute delivered fidelity
            delivered_F = self._route_current_path()
            sensor_id = self.current_demand['sensors'][self.current_sensor_idx]
            self.delivered_fidelities[sensor_id] = delivered_F

            # Move to next sensor
            self.current_sensor_idx += 1

            if self.current_sensor_idx >= len(self.current_demand['sensors']):
                # All sensors routed — assemble GHZ and compute final reward
                F_ghz, qfi = self.network.assemble_ghz(
                    self.hub, self.delivered_fidelities
                )
                self.episode_qfi = qfi
                reward = self._compute_reward(qfi, F_ghz)
                terminated = True
            else:
                # Small reward for reaching hub for this sensor
                reward = 0.1 * delivered_F
                self._start_sensor_routing()

        else:
            # Step penalty
            reward = -0.01

        # Truncate if too many steps
        if self.step_count >= self.max_steps:
            truncated = True
            if not terminated:
                # Partial reward based on what was delivered
                if self.delivered_fidelities:
                    F_ghz, qfi = self.network.assemble_ghz(
                        self.hub, self.delivered_fidelities
                    )
                    reward = self._compute_reward(qfi, F_ghz) * 0.5
                else:
                    reward = -1.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _route_current_path(self) -> float:
        """Compute delivered fidelity along current path."""
        if len(self.path) < 2:
            return 0.25

        link_fidelities = []
        for i in range(len(self.path) - 1):
            u, v = self.path[i], self.path[i+1]
            pair = self.network.memories[u].get_best_pair(v)
            if pair is not None:
                link_fidelities.append(pair.fidelity)
            else:
                # No pair available, use base fidelity with degradation
                edge = self.topology.edges.get((u, v), {})
                F0 = edge.get('F0', 0.7)
                link_fidelities.append(F0 * 0.5)

        return multihop_fidelity(link_fidelities)

    def _compute_reward(self, qfi: float, F_ghz: float) -> float:
        """Compute multi-objective reward.

        Args:
            qfi: Achieved QFI.
            F_ghz: GHZ state fidelity.

        Returns:
            Scalar reward.
        """
        N = len(self.delivered_fidelities)
        if N == 0:
            return 0.0

        delta_qfi = qfi - self.prev_qfi
        self.prev_qfi = qfi

        # QCRB improvement (relative to SQL)
        sql_qfi = float(N)
        qcrb_improvement = max(0, qfi - sql_qfi) / max(sql_qfi, 1.0)

        # Latency (normalised step count)
        latency_norm = self.step_count / self.max_steps

        # Memory usage
        memory_usage = self.network.get_memory_usage()

        # Heisenberg estimate
        heisenberg_limit = float(N ** 2)
        heisenberg_est = qfi / max(heisenberg_limit, 1.0) * 2.0

        heis_bonus = max(0.0, heisenberg_est - 1.5)

        reward = (self.alpha * delta_qfi / max(heisenberg_limit, 1.0)
                  + self.beta * qcrb_improvement
                  - self.gamma_r * latency_norm
                  - self.delta_w * memory_usage
                  + self.epsilon_r * heis_bonus)

        return float(reward)

    def _get_action_mask(self) -> np.ndarray:
        """Get action mask (1 = valid, 0 = invalid).

        Ensures at least one action is always valid to prevent NaN in softmax.
        Falls back to allowing all neighbors (backtracking) if all are visited.
        """
        neighbors = self._neighbor_map[self.current_node]
        mask = np.zeros(self.max_degree, dtype=np.float32)
        for i, n in enumerate(neighbors):
            if n not in self.visited or n == self.target_node:
                mask[i] = 1.0

        # Fallback: if all actions masked, allow all neighbors (backtracking)
        if mask.sum() == 0 and len(neighbors) > 0:
            for i in range(len(neighbors)):
                mask[i] = 1.0

        return mask

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Build observation dict."""
        # Node features
        node_features = np.zeros(
            (self.n_nodes, self.node_feat_dim), dtype=np.float32
        )
        for n in self.topology.nodes():
            node_features[n] = self.network.get_node_features(n)

        # Edge features
        n_dir_edges = 2 * self.n_edges
        edge_features = np.zeros(
            (n_dir_edges, self.edge_feat_dim), dtype=np.float32
        )
        edges = list(self.topology.edges())
        for i, (u, v) in enumerate(edges):
            edge_features[i] = self.network.get_edge_features(u, v)
            edge_features[i + self.n_edges] = self.network.get_edge_features(v, u)

        return {
            'node_features': node_features,
            'edge_index': self._edge_index.copy(),
            'edge_features': edge_features,
            'action_mask': self._get_action_mask(),
            'current_node': np.int64(self.current_node),
            'target_node': np.int64(self.target_node),
        }

    def _get_info(self) -> Dict[str, Any]:
        """Build info dict."""
        return {
            'step_count': self.step_count,
            'current_node': self.current_node,
            'target_node': self.target_node,
            'path': list(self.path),
            'sensor_idx': self.current_sensor_idx,
            'delivered_fidelities': dict(self.delivered_fidelities),
            'episode_qfi': self.episode_qfi,
            'memory_usage': self.network.get_memory_usage(),
            'network_stats': dict(self.network.stats),
        }


def make_env(config: Optional[Dict] = None, seed: int = 0) -> QuantumRoutingEnv:
    """Factory function for creating the routing environment.

    Args:
        config: Environment configuration.
        seed: Random seed.

    Returns:
        QuantumRoutingEnv instance.
    """
    cfg = dict(config or {})
    cfg['seed'] = seed
    return QuantumRoutingEnv(cfg)
