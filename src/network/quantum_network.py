"""
Discrete-event quantum network simulator.

Implements the core quantum network simulation including:
- Entangled pair generation on links
- Quantum memory management with decoherence
- Entanglement swapping at intermediate nodes
- Purification operations
- GHZ state assembly at the hub

This replaces NetSquid with a pure Python/NumPy implementation
that models the same physics.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from ..physics.werner import werner_state_fidelity_to_p, werner_state_p_to_fidelity
from ..physics.swapping import swap_fidelity, multihop_fidelity
from ..physics.decoherence import decohere
from ..physics.purification import dejmps_purify, dejmps_success_probability
from ..physics.ghz import ghz_parameter, ghz_fidelity
from ..physics.qfi import qfi_depolarised_ghz


@dataclass
class EntangledPair:
    """Represents an entangled pair stored in quantum memories."""
    node_a: int
    node_b: int
    fidelity: float
    creation_time: float
    pair_id: int = 0

    def age(self, current_time: float) -> float:
        """Time since creation."""
        return current_time - self.creation_time


@dataclass
class QuantumMemory:
    """Quantum memory at a node."""
    node_id: int
    capacity: int
    T2: float
    pairs: List[EntangledPair] = field(default_factory=list)

    @property
    def occupied(self) -> int:
        return len(self.pairs)

    @property
    def available(self) -> int:
        return max(0, self.capacity - self.occupied)

    def store(self, pair: EntangledPair) -> bool:
        """Store a pair if capacity allows."""
        if self.occupied < self.capacity:
            self.pairs.append(pair)
            return True
        return False

    def remove(self, pair: EntangledPair) -> None:
        """Remove a pair from memory."""
        if pair in self.pairs:
            self.pairs.remove(pair)

    def get_best_pair(self, partner_node: int) -> Optional[EntangledPair]:
        """Get highest fidelity pair shared with partner_node."""
        relevant = [p for p in self.pairs
                    if p.node_a == partner_node or p.node_b == partner_node]
        if not relevant:
            return None
        return max(relevant, key=lambda p: p.fidelity)

    def get_pairs_with(self, partner_node: int) -> List[EntangledPair]:
        """Get all pairs shared with partner_node."""
        return [p for p in self.pairs
                if p.node_a == partner_node or p.node_b == partner_node]

    def apply_decoherence(self, current_time: float) -> None:
        """Apply decoherence to all stored pairs."""
        for pair in self.pairs:
            age = pair.age(current_time)
            if age > 0:
                pair.fidelity = decohere(pair.fidelity, age, self.T2)
                pair.creation_time = current_time  # reset to avoid double decay

    def evict_low_fidelity(self, threshold: float = 0.25) -> int:
        """Remove pairs below fidelity threshold."""
        before = len(self.pairs)
        self.pairs = [p for p in self.pairs if p.fidelity > threshold]
        return before - len(self.pairs)

    def clear(self) -> None:
        """Remove all stored pairs."""
        self.pairs.clear()


class QuantumNetwork:
    """Discrete-event quantum network simulator.

    Maintains the full network state including entangled pair inventories,
    memory occupancies, and link statistics.
    """

    def __init__(
        self,
        topology: nx.Graph,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            topology: NetworkX graph with quantum attributes.
            config: Optional configuration overrides.
        """
        self.topology = topology.copy()
        self.config = config or {}
        self.current_time: float = 0.0
        self._pair_counter: int = 0

        # Default parameters
        self.dt: float = self.config.get('dt', 0.01)  # time step duration (s)
        self.default_T2: float = self.config.get('T2', 1.0)
        self.default_F0: float = self.config.get('F0', 0.95)
        self.default_p_gen: float = self.config.get('p_gen', 0.5)
        self.swap_success_prob: float = self.config.get('swap_prob', 0.5)
        self.purify_enabled: bool = self.config.get('purify', True)

        # Initialize quantum memories
        self.memories: Dict[int, QuantumMemory] = {}
        for node in self.topology.nodes():
            nd = self.topology.nodes[node]
            self.memories[node] = QuantumMemory(
                node_id=node,
                capacity=nd.get('n_memories', 10),
                T2=nd.get('T2', self.default_T2),
            )

        # Link state tracking
        self.link_pairs: Dict[Tuple[int, int], List[EntangledPair]] = {}
        for u, v in self.topology.edges():
            self.link_pairs[(u, v)] = []
            self.link_pairs[(v, u)] = []

        # Statistics
        self.stats = {
            'pairs_generated': 0,
            'pairs_consumed': 0,
            'swaps_performed': 0,
            'swaps_failed': 0,
            'purifications': 0,
            'ghz_assembled': 0,
            'total_qfi': 0.0,
        }

        self._rng = np.random.RandomState(
            self.config.get('seed', 42)
        )

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset network state."""
        self.current_time = 0.0
        self._pair_counter = 0

        if seed is not None:
            self._rng = np.random.RandomState(seed)

        for mem in self.memories.values():
            mem.clear()

        for key in self.link_pairs:
            self.link_pairs[key] = []

        for key in self.stats:
            self.stats[key] = 0 if isinstance(self.stats[key], int) else 0.0

    def step(self) -> None:
        """Advance simulation by one time step.

        1. Apply decoherence to all memories
        2. Attempt entangled pair generation on each link
        3. Evict low-fidelity pairs
        """
        self.current_time += self.dt

        # 1. Apply decoherence
        for mem in self.memories.values():
            mem.apply_decoherence(self.current_time)

        # 2. Generate entangled pairs on links
        for u, v in self.topology.edges():
            edge = self.topology.edges[u, v]
            p_gen = edge.get('p_gen', self.default_p_gen)
            F0 = edge.get('F0', self.default_F0)

            if self._rng.random() < p_gen:
                if (self.memories[u].available > 0 and
                        self.memories[v].available > 0):
                    self._pair_counter += 1
                    pair = EntangledPair(
                        node_a=u,
                        node_b=v,
                        fidelity=F0,
                        creation_time=self.current_time,
                        pair_id=self._pair_counter,
                    )
                    self.memories[u].store(pair)
                    self.memories[v].store(pair)
                    self.link_pairs[(u, v)].append(pair)
                    self.link_pairs[(v, u)].append(pair)
                    self.stats['pairs_generated'] += 1

        # 3. Evict low-fidelity pairs
        for mem in self.memories.values():
            evicted = mem.evict_low_fidelity(0.25)
            self.stats['pairs_consumed'] += evicted

    def perform_swap(self, node: int, left_partner: int,
                     right_partner: int) -> Optional[EntangledPair]:
        """Perform entanglement swapping at an intermediate node.

        Args:
            node: Intermediate node performing BSM.
            left_partner: Left-side partner node.
            right_partner: Right-side partner node.

        Returns:
            New entangled pair between left_partner and right_partner,
            or None if swap failed.
        """
        pair_left = self.memories[node].get_best_pair(left_partner)
        pair_right = self.memories[node].get_best_pair(right_partner)

        if pair_left is None or pair_right is None:
            return None

        # BSM success probability
        if self._rng.random() > self.swap_success_prob:
            # Swap failed — consume both pairs
            self._consume_pair(pair_left, node, left_partner)
            self._consume_pair(pair_right, node, right_partner)
            self.stats['swaps_failed'] += 1
            return None

        # Compute output fidelity
        F_out = swap_fidelity(pair_left.fidelity, pair_right.fidelity)

        # Consume input pairs
        self._consume_pair(pair_left, node, left_partner)
        self._consume_pair(pair_right, node, right_partner)

        # Create new pair
        self._pair_counter += 1
        new_pair = EntangledPair(
            node_a=left_partner,
            node_b=right_partner,
            fidelity=F_out,
            creation_time=self.current_time,
            pair_id=self._pair_counter,
        )

        # Store in endpoint memories
        if (self.memories[left_partner].available > 0 and
                self.memories[right_partner].available > 0):
            self.memories[left_partner].store(new_pair)
            self.memories[right_partner].store(new_pair)
            self.stats['swaps_performed'] += 1
            return new_pair

        self.stats['swaps_failed'] += 1
        return None

    def perform_purification(self, node_a: int, node_b: int) -> Optional[EntangledPair]:
        """Perform DEJMPS purification on two pairs between node_a and node_b.

        Consumes 2 pairs, probabilistically produces 1 pair of higher fidelity.

        Args:
            node_a: First node.
            node_b: Second node.

        Returns:
            Purified pair or None if failed/insufficient pairs.
        """
        pairs = self.memories[node_a].get_pairs_with(node_b)
        if len(pairs) < 2:
            return None

        # Take the two worst pairs for purification
        pairs_sorted = sorted(pairs, key=lambda p: p.fidelity)
        p1, p2 = pairs_sorted[0], pairs_sorted[1]

        avg_F = (p1.fidelity + p2.fidelity) / 2.0

        # Success probability
        p_success = dejmps_success_probability(avg_F)

        # Consume both pairs
        self._consume_pair(p1, node_a, node_b)
        self._consume_pair(p2, node_a, node_b)
        self.stats['purifications'] += 1

        if self._rng.random() < p_success:
            F_out = dejmps_purify(avg_F)
            self._pair_counter += 1
            new_pair = EntangledPair(
                node_a=node_a,
                node_b=node_b,
                fidelity=F_out,
                creation_time=self.current_time,
                pair_id=self._pair_counter,
            )
            self.memories[node_a].store(new_pair)
            self.memories[node_b].store(new_pair)
            return new_pair

        return None

    def route_path(self, path: List[int]) -> Optional[float]:
        """Route entanglement along a path via sequential swapping.

        Args:
            path: Ordered list of node IDs from source to destination.

        Returns:
            End-to-end fidelity, or None if routing failed.
        """
        if len(path) < 2:
            return None

        # Check that entangled pairs exist on each link
        link_fidelities = []
        for i in range(len(path) - 1):
            pair = self.memories[path[i]].get_best_pair(path[i+1])
            if pair is None:
                return None
            link_fidelities.append(pair.fidelity)

        # Perform sequential swapping at intermediate nodes
        for i in range(1, len(path) - 1):
            result = self.perform_swap(path[i], path[i-1], path[i+1])
            if result is None:
                return None

        # Delivered fidelity
        return multihop_fidelity(link_fidelities)

    def assemble_ghz(self, hub: int, sensor_fidelities: Dict[int, float]) -> Tuple[float, float]:
        """Assemble GHZ state at hub from delivered pairs.

        Args:
            hub: Hub node ID.
            sensor_fidelities: Dict mapping sensor_id → delivered fidelity.

        Returns:
            (ghz_fidelity, qfi) tuple.
        """
        if not sensor_fidelities:
            return 0.0, 0.0

        fids = list(sensor_fidelities.values())
        N = len(fids)
        p_ghz = ghz_parameter(fids)
        F_ghz = ghz_fidelity(fids)
        qfi = qfi_depolarised_ghz(p_ghz, N)

        self.stats['ghz_assembled'] += 1
        self.stats['total_qfi'] += qfi

        return F_ghz, qfi

    def get_node_features(self, node: int) -> np.ndarray:
        """Get feature vector for a node.

        Features: [type_id, n_memories, T2, n_qubits, occupied_ratio, local_qfi]

        Args:
            node: Node ID.

        Returns:
            Feature vector of shape (6,).
        """
        nd = self.topology.nodes[node]
        mem = self.memories[node]
        return np.array([
            nd.get('type_id', 0.0),
            mem.capacity / 20.0,         # normalised
            nd.get('T2', 1.0),
            nd.get('n_qubits', 10) / 20.0,  # normalised
            mem.occupied / max(mem.capacity, 1),  # occupancy ratio
            nd.get('local_qfi', 0.0),
        ], dtype=np.float32)

    def get_edge_features(self, u: int, v: int) -> np.ndarray:
        """Get feature vector for an edge.

        Features: [fidelity, age, p_gen, distance_normalised]

        Args:
            u, v: Edge endpoints.

        Returns:
            Feature vector of shape (4,).
        """
        edge = self.topology.edges[u, v]
        pair = self.memories[u].get_best_pair(v)
        fidelity = pair.fidelity if pair else 0.0
        age = pair.age(self.current_time) if pair else 0.0

        return np.array([
            fidelity,
            min(age, 10.0) / 10.0,  # normalised, capped
            edge.get('p_gen', 0.5),
            min(edge.get('distance', 50.0), 3000.0) / 3000.0,  # normalised
        ], dtype=np.float32)

    def get_memory_usage(self) -> float:
        """Get total memory occupancy ratio across all nodes."""
        total_occ = sum(m.occupied for m in self.memories.values())
        total_cap = sum(m.capacity for m in self.memories.values())
        return total_occ / max(total_cap, 1)

    def _consume_pair(self, pair: EntangledPair, *nodes: int) -> None:
        """Remove a pair from all relevant memories and link tracking."""
        for n in nodes:
            self.memories[n].remove(pair)

        # Clean from link pairs
        key1 = (pair.node_a, pair.node_b)
        key2 = (pair.node_b, pair.node_a)
        for key in [key1, key2]:
            if key in self.link_pairs and pair in self.link_pairs[key]:
                self.link_pairs[key].remove(pair)

        self.stats['pairs_consumed'] += 1
