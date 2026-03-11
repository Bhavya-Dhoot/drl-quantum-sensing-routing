"""
Entanglement Rate (ER) baseline (Caleffi).

Routes to maximise entanglement generation rate, using path weights
proportional to link generation probabilities. Based on Caleffi's
routing metric for quantum networks.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional
from ..physics.swapping import multihop_fidelity
from ..physics.ghz import ghz_parameter, ghz_fidelity
from ..physics.qfi import qfi_depolarised_ghz


class EntanglementRateRouter:
    """Routes maximising entanglement generation rate."""

    def __init__(self, topology: nx.Graph, config: Optional[Dict] = None):
        self.topology = topology
        self.config = config or {}
        self.hub = self.topology.graph.get('hub', 0)
        self.sensors = self.topology.graph.get('sensors', [])
        self.swap_prob = config.get('swap_prob', 0.5) if config else 0.5

        # Set rate-based edge weights: -log(p_gen * p_swap)
        self._weighted_graph = topology.copy()
        for u, v in self._weighted_graph.edges():
            p_gen = self._weighted_graph.edges[u, v].get('p_gen', 0.5)
            rate = p_gen * self.swap_prob
            rate = max(rate, 1e-10)
            self._weighted_graph.edges[u, v]['rate_weight'] = -np.log(rate)

    def _best_rate_path(self, source: int, target: int) -> List[int]:
        try:
            return nx.dijkstra_path(
                self._weighted_graph, source, target,
                weight='rate_weight'
            )
        except nx.NetworkXNoPath:
            return []

    def route(self, demand: Dict) -> Dict:
        sensor_fidelities = {}
        paths = {}
        total_latency = 0

        for sensor_id in demand['sensors']:
            path = self._best_rate_path(sensor_id, self.hub)
            if not path:
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
            'method': 'ER',
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
            'method': 'ER',
        }
