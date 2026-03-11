"""
Random router baseline.

Routes each sensor to the hub via a random walk, avoiding loops.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Set
from ..physics.swapping import multihop_fidelity
from ..physics.ghz import ghz_parameter, ghz_fidelity
from ..physics.qfi import qfi_depolarised_ghz


class RandomRouter:
    """Routes using random walk (avoiding revisits)."""

    def __init__(self, topology: nx.Graph, config: Optional[Dict] = None,
                 seed: int = 42):
        self.topology = topology
        self.config = config or {}
        self.hub = self.topology.graph.get('hub', 0)
        self.sensors = self.topology.graph.get('sensors', [])
        self.rng = np.random.RandomState(seed)

    def _random_path(self, source: int, target: int) -> List[int]:
        path = [source]
        visited: Set[int] = {source}
        current = source
        max_hops = 2 * self.topology.number_of_nodes()

        for _ in range(max_hops):
            if current == target:
                break

            neighbors = [n for n in self.topology.neighbors(current)
                        if n not in visited or n == target]

            if not neighbors:
                # Dead end - try to recover with shortest path
                try:
                    remaining = nx.shortest_path(self.topology, current, target)
                    path.extend(remaining[1:])
                except nx.NetworkXNoPath:
                    pass
                break

            next_node = self.rng.choice(neighbors)
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    def route(self, demand: Dict) -> Dict:
        sensor_fidelities = {}
        paths = {}
        total_latency = 0

        for sensor_id in demand['sensors']:
            path = self._random_path(sensor_id, self.hub)
            if not path or path[-1] != self.hub:
                sensor_fidelities[sensor_id] = 0.25
                paths[sensor_id] = []
                continue

            link_fids = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = self.topology.edges.get((u, v), {})
                link_fids.append(edge.get('F0', 0.9))

            delivered_F = multihop_fidelity(link_fids)
            sensor_fidelities[sensor_id] = delivered_F
            paths[sensor_id] = path
            total_latency += len(path) - 1

        fids = list(sensor_fidelities.values())
        N = len(fids)
        if N > 0 and all(f > 0.25 for f in fids):
            p_ghz = ghz_parameter(fids)
            F_ghz = ghz_fidelity(fids)
            qfi = qfi_depolarised_ghz(p_ghz, N)
        else:
            F_ghz, qfi = 0.0, 0.0

        return {
            'paths': paths,
            'sensor_fidelities': sensor_fidelities,
            'ghz_fidelity': F_ghz,
            'qfi': qfi,
            'latency': total_latency,
            'method': 'RANDOM',
        }

    def evaluate(self, demands: List[Dict]) -> Dict:
        qfi_values = []
        latencies = []
        for d in demands:
            result = self.route(d)
            qfi_values.append(result['qfi'])
            latencies.append(result['latency'])
        return {
            'mean_qfi': float(np.mean(qfi_values)),
            'std_qfi': float(np.std(qfi_values)),
            'mean_latency': float(np.mean(latencies)),
            'method': 'RANDOM',
        }
