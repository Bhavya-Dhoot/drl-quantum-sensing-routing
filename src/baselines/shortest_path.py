"""
Shortest Path (SP) baseline router.

Routes entanglement along the shortest (hop-count) path from each
sensor to the hub. Does not consider fidelity or QFI.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..physics.swapping import multihop_fidelity
from ..physics.ghz import ghz_parameter, ghz_fidelity
from ..physics.qfi import qfi_depolarised_ghz


class ShortestPathRouter:
    """Routes using shortest hop-count path."""

    def __init__(self, topology: nx.Graph, config: Optional[Dict] = None):
        self.topology = topology
        self.config = config or {}
        self.hub = self.topology.graph.get('hub', 0)
        self.sensors = self.topology.graph.get('sensors', [])

        # Precompute shortest paths
        self._paths: Dict[int, List[int]] = {}
        for s in self.sensors:
            try:
                self._paths[s] = nx.shortest_path(topology, s, self.hub)
            except nx.NetworkXNoPath:
                self._paths[s] = []

    def route(self, demand: Dict) -> Dict:
        """Route a sensing demand.

        Args:
            demand: Demand dict with 'sensors', 'hub', 'N'.

        Returns:
            Result dict with paths, fidelities, QFI.
        """
        sensor_fidelities = {}
        paths = {}
        total_latency = 0

        for sensor_id in demand['sensors']:
            path = self._paths.get(sensor_id, [])
            if not path:
                sensor_fidelities[sensor_id] = 0.25
                paths[sensor_id] = []
                continue

            # Get link fidelities along path
            link_fids = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = self.topology.edges.get((u, v), {})
                F0 = edge.get('F0', edge.get('fidelity', 0.9))
                link_fids.append(F0)

            delivered_F = multihop_fidelity(link_fids)
            sensor_fidelities[sensor_id] = delivered_F
            paths[sensor_id] = path
            total_latency += len(path) - 1

        # Assemble GHZ
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
            'method': 'SP',
        }

    def evaluate(self, demands: List[Dict]) -> Dict:
        """Evaluate on multiple demands."""
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
            'method': 'SP',
        }
