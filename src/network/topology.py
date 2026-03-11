"""
Network topology definitions for quantum routing experiments.

Provides standard quantum networking benchmark topologies:
- NSFNET (14 nodes, 21 links)
- Linear chain (configurable length)
- Grid (configurable dimensions)
- SURFnet (50 nodes, 68 links)
- Erdős-Rényi random graphs
"""

import numpy as np
import networkx as nx
from typing import Optional, Dict, Any, Tuple, List


# ── NSFNET Topology ──────────────────────────────────────────────────────────

NSFNET_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
    (4, 5), (4, 6), (5, 9), (5, 12), (6, 7), (7, 8), (8, 9), (8, 11),
    (8, 13), (9, 12), (10, 11), (10, 13), (11, 13),
]

NSFNET_SENSOR_NODES: List[int] = [0, 3, 5, 7, 9, 11, 12, 13]
NSFNET_REPEATER_NODES: List[int] = [1, 2, 4, 6, 8, 10]
NSFNET_HUB: int = 8

# Approximate fibre distances (km) for NSFNET edges
NSFNET_DISTANCES: Dict[Tuple[int, int], float] = {
    (0, 1): 1100, (0, 2): 1600, (0, 3): 2800, (1, 2): 600, (1, 7): 2000,
    (2, 5): 2000, (3, 4): 700, (3, 10): 900, (4, 5): 600, (4, 6): 1100,
    (5, 9): 800, (5, 12): 500, (6, 7): 700, (7, 8): 700, (8, 9): 900,
    (8, 11): 500, (8, 13): 300, (9, 12): 400, (10, 11): 800,
    (10, 13): 300, (11, 13): 300,
}

# ── SURFnet Topology ─────────────────────────────────────────────────────────

SURFNET_EDGES: List[Tuple[int, int]] = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),
    (10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),
    (18,19),(19,20),(20,21),(21,22),(22,23),(23,24),(24,25),(25,26),
    (26,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,33),(33,34),
    (34,35),(35,36),(36,37),(37,38),(38,39),(39,40),(40,41),(41,42),
    (42,43),(43,44),(44,45),(45,46),(46,47),(47,48),(48,49),
    # Additional cross-links for SURFnet ring + mesh
    (0,5),(2,8),(4,12),(6,15),(9,18),(11,20),(14,23),(16,26),
    (19,28),(21,31),(24,34),(27,37),(30,40),(33,43),(36,46),(39,49),
    (1,10),(3,13),
]


def _set_default_node_attrs(G: nx.Graph, config: Optional[Dict] = None) -> None:
    """Set default quantum node attributes."""
    cfg = config or {}
    default_T2 = cfg.get('T2', 1.0)           # coherence time in seconds
    default_n_memories = cfg.get('n_memories', 10)
    default_gen_rate = cfg.get('gen_rate', 100)  # pairs per second

    for n in G.nodes():
        node_data = G.nodes[n]
        node_data.setdefault('T2', default_T2)
        node_data.setdefault('n_memories', default_n_memories)
        node_data.setdefault('gen_rate', default_gen_rate)
        node_data.setdefault('n_qubits', default_n_memories)
        node_data.setdefault('occupied', 0)
        node_data.setdefault('local_qfi', 0.0)


def _set_default_edge_attrs(G: nx.Graph, config: Optional[Dict] = None) -> None:
    """Set default quantum edge attributes."""
    cfg = config or {}
    default_F0 = cfg.get('F0', 0.95)      # initial fidelity
    default_p_gen = cfg.get('p_gen', 0.5)  # generation probability per attempt
    L_att = cfg.get('L_att', 22.0)         # attenuation length (km)

    for u, v in G.edges():
        edge_data = G.edges[u, v]
        d = edge_data.get('distance', 50.0)
        # Fidelity can depend on distance
        F = default_F0
        edge_data.setdefault('F0', F)
        edge_data.setdefault('fidelity', F)
        edge_data.setdefault('age', 0.0)
        edge_data.setdefault('p_gen', default_p_gen * np.exp(-d / L_att))
        edge_data.setdefault('distance', d)


def create_nsfnet(config: Optional[Dict] = None) -> nx.Graph:
    """Create NSFNET topology (14 nodes, 21 links).

    Args:
        config: Optional configuration overrides.

    Returns:
        NetworkX Graph with quantum attributes.
    """
    G = nx.Graph()
    G.add_nodes_from(range(14))

    for u, v in NSFNET_EDGES:
        d = NSFNET_DISTANCES.get((u, v), NSFNET_DISTANCES.get((v, u), 500.0))
        G.add_edge(u, v, distance=d)

    # Set node types
    for n in G.nodes():
        if n in NSFNET_SENSOR_NODES:
            G.nodes[n]['type'] = 'sensor'
            G.nodes[n]['type_id'] = 1.0
        else:
            G.nodes[n]['type'] = 'repeater'
            G.nodes[n]['type_id'] = 0.0

    G.nodes[NSFNET_HUB]['type'] = 'hub'
    G.nodes[NSFNET_HUB]['type_id'] = 2.0

    _set_default_node_attrs(G, config)
    _set_default_edge_attrs(G, config)
    G.graph['name'] = 'NSFNET'
    G.graph['hub'] = NSFNET_HUB
    G.graph['sensors'] = NSFNET_SENSOR_NODES
    return G


def create_linear(n_nodes: int = 10, config: Optional[Dict] = None) -> nx.Graph:
    """Create linear chain topology.

    Args:
        n_nodes: Number of nodes in the chain.
        config: Optional configuration overrides.

    Returns:
        NetworkX Graph.
    """
    G = nx.path_graph(n_nodes)
    for u, v in G.edges():
        G.edges[u, v]['distance'] = 50.0

    # First and last are sensors, middle is hub
    hub = n_nodes // 2
    sensors = [i for i in range(n_nodes) if i != hub]
    for n in G.nodes():
        if n == hub:
            G.nodes[n]['type'] = 'hub'
            G.nodes[n]['type_id'] = 2.0
        else:
            G.nodes[n]['type'] = 'sensor'
            G.nodes[n]['type_id'] = 1.0

    _set_default_node_attrs(G, config)
    _set_default_edge_attrs(G, config)
    G.graph['name'] = 'Linear'
    G.graph['hub'] = hub
    G.graph['sensors'] = sensors
    return G


def create_grid(rows: int = 4, cols: int = 4,
                config: Optional[Dict] = None) -> nx.Graph:
    """Create grid topology.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        config: Optional configuration overrides.

    Returns:
        NetworkX Graph.
    """
    G = nx.grid_2d_graph(rows, cols)
    # Relabel to integer nodes
    mapping = {(r, c): r * cols + c for r, c in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    for u, v in G.edges():
        G.edges[u, v]['distance'] = 50.0

    n_total = rows * cols
    hub = n_total // 2
    # Corner and edge nodes are sensors
    sensors = []
    for n in G.nodes():
        r, c = n // cols, n % cols
        if r in (0, rows-1) or c in (0, cols-1):
            G.nodes[n]['type'] = 'sensor'
            G.nodes[n]['type_id'] = 1.0
            sensors.append(n)
        elif n == hub:
            G.nodes[n]['type'] = 'hub'
            G.nodes[n]['type_id'] = 2.0
        else:
            G.nodes[n]['type'] = 'repeater'
            G.nodes[n]['type_id'] = 0.0

    _set_default_node_attrs(G, config)
    _set_default_edge_attrs(G, config)
    G.graph['name'] = 'Grid'
    G.graph['hub'] = hub
    G.graph['sensors'] = sensors
    return G


def create_surfnet(config: Optional[Dict] = None) -> nx.Graph:
    """Create SURFnet topology (50 nodes, 68 links).

    Args:
        config: Optional configuration overrides.

    Returns:
        NetworkX Graph.
    """
    G = nx.Graph()
    G.add_nodes_from(range(50))

    for u, v in SURFNET_EDGES:
        G.add_edge(u, v, distance=80.0)

    hub = 25
    sensors = list(range(0, 50, 3))  # Every 3rd node is a sensor
    if hub in sensors:
        sensors.remove(hub)

    for n in G.nodes():
        if n == hub:
            G.nodes[n]['type'] = 'hub'
            G.nodes[n]['type_id'] = 2.0
        elif n in sensors:
            G.nodes[n]['type'] = 'sensor'
            G.nodes[n]['type_id'] = 1.0
        else:
            G.nodes[n]['type'] = 'repeater'
            G.nodes[n]['type_id'] = 0.0

    _set_default_node_attrs(G, config)
    _set_default_edge_attrs(G, config)
    G.graph['name'] = 'SURFnet'
    G.graph['hub'] = hub
    G.graph['sensors'] = sensors
    return G


def create_erdos_renyi(n_nodes: int = 20, p_edge: float = 0.2,
                       seed: int = 42,
                       config: Optional[Dict] = None) -> nx.Graph:
    """Create Erdős-Rényi random graph topology.

    Ensures the graph is connected by using connected_watts_strogatz as
    fallback if the ER graph is disconnected.

    Args:
        n_nodes: Number of nodes.
        p_edge: Edge probability.
        seed: Random seed for reproducibility.
        config: Optional configuration overrides.

    Returns:
        NetworkX Graph.
    """
    rng = np.random.RandomState(seed)

    # Keep generating until we get a connected graph
    for attempt in range(100):
        G = nx.erdos_renyi_graph(n_nodes, p_edge, seed=seed + attempt)
        if nx.is_connected(G):
            break
    else:
        # Fallback: make a connected random graph
        G = nx.connected_watts_strogatz_graph(n_nodes, 4, p_edge, seed=seed)

    for u, v in G.edges():
        G.edges[u, v]['distance'] = rng.uniform(30, 200)

    hub = 0
    sensors = list(range(1, n_nodes, 2))

    for n in G.nodes():
        if n == hub:
            G.nodes[n]['type'] = 'hub'
            G.nodes[n]['type_id'] = 2.0
        elif n in sensors:
            G.nodes[n]['type'] = 'sensor'
            G.nodes[n]['type_id'] = 1.0
        else:
            G.nodes[n]['type'] = 'repeater'
            G.nodes[n]['type_id'] = 0.0

    _set_default_node_attrs(G, config)
    _set_default_edge_attrs(G, config)
    G.graph['name'] = 'ErdosRenyi'
    G.graph['hub'] = hub
    G.graph['sensors'] = sensors
    return G


def create_topology(name: str, config: Optional[Dict] = None) -> nx.Graph:
    """Factory function to create a topology by name.

    Args:
        name: Topology name (nsfnet, linear, grid, surfnet, random).
        config: Optional configuration overrides.

    Returns:
        NetworkX Graph with quantum attributes.
    """
    creators = {
        'nsfnet': create_nsfnet,
        'linear': create_linear,
        'grid': create_grid,
        'surfnet': create_surfnet,
        'random': create_erdos_renyi,
        'erdos_renyi': create_erdos_renyi,
    }
    name_lower = name.lower()
    if name_lower not in creators:
        raise ValueError(f"Unknown topology: {name}. "
                         f"Available: {list(creators.keys())}")
    return creators[name_lower](config=config)


def get_sensor_nodes(G: nx.Graph) -> List[int]:
    """Get sensor node IDs from a topology graph."""
    return G.graph.get('sensors', [])


def get_hub_node(G: nx.Graph) -> int:
    """Get hub node ID from a topology graph."""
    return G.graph.get('hub', 0)


def get_num_sensors(G: nx.Graph) -> int:
    """Get number of sensor nodes."""
    return len(get_sensor_nodes(G))
