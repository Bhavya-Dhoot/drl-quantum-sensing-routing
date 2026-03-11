"""
Run all baseline routing algorithms.

Usage:
    python scripts/run_baselines.py --topology nsfnet --seeds 5
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.network.topology import create_topology, get_sensor_nodes, get_hub_node
from src.network.demand import DemandGenerator
from src.baselines.shortest_path import ShortestPathRouter
from src.baselines.fidelity_first import FidelityFirstRouter
from src.baselines.greedy_qfi import GreedyQFIRouter
from src.baselines.random_router import RandomRouter
from src.baselines.entanglement_rate import EntanglementRateRouter
from src.utils.seed import set_all_seeds
from src.utils.metrics import estimate_scaling_exponent


def parse_args():
    parser = argparse.ArgumentParser(description='Run baseline routers')
    parser.add_argument('--topology', type=str, default='nsfnet')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--n_episodes', type=int, default=500)
    parser.add_argument('--output_dir', type=str, default='results')
    return parser.parse_args()


def run_baselines(args):
    """Run all baselines across seeds."""
    print(f"Running baselines on {args.topology} ({args.seeds} seeds, "
          f"{args.n_episodes} episodes each)")

    all_results = {}

    baselines = {
        'SP': ShortestPathRouter,
        'FF': FidelityFirstRouter,
        'GREEDY-QFI': GreedyQFIRouter,
        'RANDOM': None,  # needs seed
        'ER': EntanglementRateRouter,
    }

    for method_name, router_class in baselines.items():
        print(f"\n{'='*60}")
        print(f"Running {method_name}...")
        seed_results = []

        for seed in range(args.seeds):
            set_all_seeds(seed)
            topology = create_topology(args.topology)
            sensors = get_sensor_nodes(topology)
            hub = get_hub_node(topology)

            if method_name == 'RANDOM':
                router = RandomRouter(topology, seed=seed)
            else:
                router = router_class(topology)

            demand_gen = DemandGenerator(
                sensor_nodes=sensors,
                hub_node=hub,
                seed=seed,
            )

            qfi_values = []
            latencies = []

            for ep in range(args.n_episodes):
                demand = demand_gen.generate_fixed()
                result = router.route(demand)
                qfi_values.append(result['qfi'])
                latencies.append(result['latency'])

            seed_results.append({
                'mean_qfi': float(np.mean(qfi_values)),
                'std_qfi': float(np.std(qfi_values)),
                'mean_latency': float(np.mean(latencies)),
                'qfi_values': qfi_values,
            })

            print(f"  Seed {seed}: QFI = {np.mean(qfi_values):.2f} ± "
                  f"{np.std(qfi_values):.2f}, Latency = {np.mean(latencies):.1f}")

        # Aggregate across seeds
        mean_qfis = [r['mean_qfi'] for r in seed_results]
        mean_lats = [r['mean_latency'] for r in seed_results]
        all_results[method_name] = {
            'mean_qfi': float(np.mean(mean_qfis)),
            'std_qfi': float(np.std(mean_qfis)),
            'ci_95_qfi': float(1.96 * np.std(mean_qfis) / np.sqrt(args.seeds)),
            'mean_latency': float(np.mean(mean_lats)),
            'seed_results': seed_results,
        }

        print(f"  Overall: QFI = {np.mean(mean_qfis):.2f} ± "
              f"{1.96 * np.std(mean_qfis) / np.sqrt(args.seeds):.2f}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Method':<15} {'QFI (mean)':<15} {'95% CI':<15} {'Latency':<10}")
    print(f"{'-'*55}")
    for name, res in all_results.items():
        print(f"{name:<15} {res['mean_qfi']:<15.2f} "
              f"±{res['ci_95_qfi']:<14.2f} {res['mean_latency']:<10.1f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'baseline_results.json')
    # Remove non-serialisable data
    saveable = {}
    for k, v in all_results.items():
        saveable[k] = {kk: vv for kk, vv in v.items()
                       if kk != 'seed_results'}
        saveable[k]['per_seed_qfi'] = [r['mean_qfi'] for r in v['seed_results']]
    saveable['topology'] = args.topology
    saveable['n_seeds'] = args.seeds
    saveable['n_episodes'] = args.n_episodes

    with open(output_path, 'w') as f:
        json.dump(saveable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == '__main__':
    args = parse_args()
    run_baselines(args)
