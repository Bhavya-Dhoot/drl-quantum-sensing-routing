"""
Fidelity-First (FF) baseline router.

Routes along the path that maximises the delivered fidelity,
using a modified Dijkstra with edge weights = -log(p_i)
where p_i = (4F_i - 1)/3.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional
from ..physics.swapping import multihop_fidelity
from ..physics.ghz import ghz_parameter, ghz_fidelity
from ..physics.qfi import qfi_depolarised_ghz


class FidelityFirstRouter:
    """Routes using maximum-fidelity paths."""

    def __init__(self, topology: nx.Graph, config: Optional[Dict] = None):
        self.topology = topology
        self.config = config or {}
        self.hub = self.topology.graph.get('hub', 0)
        self.sensors = self.topology.graph.get('sensors', [])

        # Set fidelity-based edge weights
        self._weighted_graph = topology.copy()
        for u, v in self._weighted_graph.edges():
            F = self._weighted_graph.edges[u, v].get('F0', 0.9)
            p = max((4 * F - 1) / 3, 1e-10)
            self._weighted_graph.edges[u, v]['fid_weight'] = -np.log(p)

    def _best_fidelity_path(self, source: int, target: int) -> List[int]:
        """Find max-fidelity path using Dijkstra on -log(p) weights."""
        try:
            return nx.dijkstra_path(
                self._weighted_graph, source, target,
                weight='fid_weight'
            )
        except nx.NetworkXNoPath:
            return []

    def route(self, demand: Dict) -> Dict:
        sensor_fidelities = {}
        paths = {}
        total_latency = 0

        for sensor_id in demand['sensors']:
            path = self._best_fidelity_path(sensor_id, self.hub)
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
            'method': 'FF',
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
            'method': 'FF',
        }
