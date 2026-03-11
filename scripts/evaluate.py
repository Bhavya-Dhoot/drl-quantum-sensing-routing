"""
Evaluation script for trained agents and scaling/robustness experiments.

Usage:
    python scripts/evaluate.py --checkpoint results/checkpoints/model.pt --config configs/nsfnet.yaml
    python scripts/evaluate.py --scaling --N_values 4,6,8,12,16,24 --topology nsfnet
    python scripts/evaluate.py --robustness --T2_values 0.05,0.1,0.2,0.5,1,2,5,10
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.routing_env import QuantumRoutingEnv
from src.agent.ppo import PPOAgent
from src.network.topology import create_topology, get_sensor_nodes, get_hub_node
from src.network.demand import DemandGenerator
from src.baselines.shortest_path import ShortestPathRouter
from src.baselines.fidelity_first import FidelityFirstRouter
from src.baselines.greedy_qfi import GreedyQFIRouter
from src.baselines.random_router import RandomRouter
from src.baselines.entanglement_rate import EntanglementRateRouter
from src.utils.seed import set_all_seeds
from src.utils.metrics import estimate_scaling_exponent, compute_qcrb, satisfaction_ratio
from src.physics.qfi import qfi_depolarised_ghz, f_min
from src.physics.ghz import ghz_parameter, ghz_fidelity


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate routing agents')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/nsfnet.yaml')
    parser.add_argument('--topology', type=str, default='nsfnet')
    parser.add_argument('--n_eval_episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')

    # Scaling experiment
    parser.add_argument('--scaling', action='store_true')
    parser.add_argument('--N_values', type=str, default='4,6,8,12,16,24')

    # Robustness experiment
    parser.add_argument('--robustness', action='store_true')
    parser.add_argument('--T2_values', type=str, default='0.05,0.1,0.2,0.5,1,2,5,10')

    return parser.parse_args()


def evaluate_agent(agent, env, n_episodes=200, seed=42):
    """Evaluate a trained agent."""
    set_all_seeds(seed)
    qfi_values = []
    latencies = []
    fidelities = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_steps = 0

        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_steps += 1

        qfi_values.append(info.get('episode_qfi', 0.0))
        latencies.append(ep_steps)
        fids = list(info.get('delivered_fidelities', {}).values())
        if fids:
            fidelities.append(np.mean(fids))

    return {
        'mean_qfi': float(np.mean(qfi_values)),
        'std_qfi': float(np.std(qfi_values)),
        'mean_latency': float(np.mean(latencies)),
        'mean_fidelity': float(np.mean(fidelities)) if fidelities else 0.0,
        'qfi_values': qfi_values,
    }


def evaluate_baseline(router_class, topology, n_episodes=200, seed=42, **kwargs):
    """Evaluate a baseline router."""
    set_all_seeds(seed)
    router = router_class(topology, **kwargs)
    demand_gen = DemandGenerator(
        sensor_nodes=get_sensor_nodes(topology),
        hub_node=get_hub_node(topology),
        seed=seed,
    )

    qfi_values = []
    latencies = []

    for _ in range(n_episodes):
        demand = demand_gen.generate_fixed()
        result = router.route(demand)
        qfi_values.append(result['qfi'])
        latencies.append(result['latency'])

    return {
        'mean_qfi': float(np.mean(qfi_values)),
        'std_qfi': float(np.std(qfi_values)),
        'mean_latency': float(np.mean(latencies)),
        'qfi_values': qfi_values,
    }


def run_scaling_experiment(args):
    """Run QFI scaling experiment for varying N."""
    N_values = [int(x) for x in args.N_values.split(',')]
    print(f"Running scaling experiment: N = {N_values}")

    results = {'N_values': N_values}

    # Create topology
    topology = create_topology(args.topology)
    all_sensors = get_sensor_nodes(topology)
    hub = get_hub_node(topology)

    methods = {
        'SP': ShortestPathRouter,
        'FF': FidelityFirstRouter,
        'GREEDY-QFI': GreedyQFIRouter,
        'RANDOM': lambda t, **kw: RandomRouter(t, seed=args.seed, **kw),
        'ER': EntanglementRateRouter,
    }

    for method_name, router_class in methods.items():
        print(f"  Evaluating {method_name}...")
        method_qfi = []

        for N in N_values:
            if N > len(all_sensors):
                method_qfi.append(0.0)
                continue

            demand_gen = DemandGenerator(
                sensor_nodes=all_sensors, hub_node=hub, seed=args.seed
            )

            qfi_vals = []
            for ep in range(args.n_eval_episodes):
                demand = demand_gen.generate_fixed(n_sensors=N)
                if callable(router_class) and not isinstance(router_class, type):
                    router = router_class(topology)
                else:
                    router = router_class(topology)
                result = router.route(demand)
                qfi_vals.append(result['qfi'])

            method_qfi.append(float(np.mean(qfi_vals)))

        # Estimate scaling exponent
        valid_idx = [i for i, q in enumerate(method_qfi) if q > 0]
        if len(valid_idx) >= 2:
            valid_N = [N_values[i] for i in valid_idx]
            valid_Q = [method_qfi[i] for i in valid_idx]
            alpha, alpha_std, r2 = estimate_scaling_exponent(valid_N, valid_Q)
        else:
            alpha, alpha_std, r2 = 0.0, 0.0, 0.0

        results[method_name] = {
            'qfi': method_qfi,
            'alpha': alpha,
            'alpha_std': alpha_std,
            'r_squared': r2,
        }
        print(f"    α̂ = {alpha:.3f} ± {alpha_std:.3f} (R² = {r2:.3f})")

    # Also evaluate trained agent if checkpoint available
    if args.checkpoint and os.path.exists(args.checkpoint):
        print("  Evaluating trained agent...")
        agent_qfi = []
        for N in N_values:
            if N > len(all_sensors):
                agent_qfi.append(0.0)
                continue

            env_config = {
                'topology': args.topology,
                'seed': args.seed,
            }
            env = QuantumRoutingEnv(env_config)
            agent = PPOAgent(
                max_action_dim=env.max_degree,
                device='cpu',
            )
            agent.load(args.checkpoint)

            qfi_vals = []
            for ep in range(args.n_eval_episodes):
                obs, info = env.reset(
                    seed=args.seed + ep,
                    options={'n_sensors': N}
                )
                done = False
                while not done:
                    action, _, _ = agent.select_action(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                qfi_vals.append(info.get('episode_qfi', 0.0))

            agent_qfi.append(float(np.mean(qfi_vals)))

        valid_idx = [i for i, q in enumerate(agent_qfi) if q > 0]
        if len(valid_idx) >= 2:
            valid_N = [N_values[i] for i in valid_idx]
            valid_Q = [agent_qfi[i] for i in valid_idx]
            alpha, alpha_std, r2 = estimate_scaling_exponent(valid_N, valid_Q)
        else:
            alpha, alpha_std, r2 = 0.0, 0.0, 0.0

        results['Ours'] = {
            'qfi': agent_qfi,
            'alpha': alpha,
            'alpha_std': alpha_std,
            'r_squared': r2,
        }
        print(f"    α̂ = {alpha:.3f} ± {alpha_std:.3f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'scaling_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nScaling results saved to {output_path}")

    return results


def run_robustness_experiment(args):
    """Run robustness experiment sweeping T₂ coherence time."""
    T2_values = [float(x) for x in args.T2_values.split(',')]
    print(f"Running robustness experiment: T₂ = {T2_values}")

    results = {'T2_values': T2_values}

    methods = {
        'SP': ShortestPathRouter,
        'FF': FidelityFirstRouter,
        'GREEDY-QFI': GreedyQFIRouter,
    }

    for method_name, router_class in methods.items():
        print(f"  Evaluating {method_name}...")
        method_qfi = []

        for T2 in T2_values:
            topo_config = {'T2': T2}
            topology = create_topology(args.topology, topo_config)
            router = router_class(topology)

            demand_gen = DemandGenerator(
                sensor_nodes=get_sensor_nodes(topology),
                hub_node=get_hub_node(topology),
                seed=args.seed,
            )

            qfi_vals = []
            for _ in range(args.n_eval_episodes):
                demand = demand_gen.generate_fixed()
                result = router.route(demand)
                qfi_vals.append(result['qfi'])

            N = len(get_sensor_nodes(topology))
            mean_qfi = float(np.mean(qfi_vals))
            normalised = mean_qfi / max(N**2, 1)
            method_qfi.append(normalised)

        results[method_name] = method_qfi

    # Trained agent
    if args.checkpoint and os.path.exists(args.checkpoint):
        print("  Evaluating trained agent...")
        agent_qfi_norm = []

        for T2 in T2_values:
            env_config = {
                'topology': args.topology,
                'seed': args.seed,
                'T2': T2,
                'topology_config': {'T2': T2},
            }
            env = QuantumRoutingEnv(env_config)
            agent = PPOAgent(
                max_action_dim=env.max_degree,
                device='cpu',
            )
            agent.load(args.checkpoint)

            qfi_vals = []
            for ep in range(args.n_eval_episodes):
                obs, info = env.reset(seed=args.seed + ep)
                done = False
                while not done:
                    action, _, _ = agent.select_action(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                qfi_vals.append(info.get('episode_qfi', 0.0))

            N = len(get_sensor_nodes(env.topology))
            mean_qfi = float(np.mean(qfi_vals))
            agent_qfi_norm.append(mean_qfi / max(N**2, 1))

        results['Ours'] = agent_qfi_norm

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'robustness_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRobustness results saved to {output_path}")

    return results


if __name__ == '__main__':
    args = parse_args()

    if args.scaling:
        run_scaling_experiment(args)
    elif args.robustness:
        run_robustness_experiment(args)
    else:
        # Standard evaluation
        if not args.checkpoint:
            print("Error: --checkpoint required for standard evaluation")
            sys.exit(1)

        topology = create_topology(args.topology)
        env = QuantumRoutingEnv({'topology': args.topology, 'seed': args.seed})
        agent = PPOAgent(max_action_dim=env.max_degree, device='cpu')
        agent.load(args.checkpoint)

        results = evaluate_agent(agent, env, args.n_eval_episodes, args.seed)
        print(f"\nEvaluation Results:")
        print(f"  Mean QFI: {results['mean_qfi']:.2f} ± {results['std_qfi']:.2f}")
        print(f"  Mean Latency: {results['mean_latency']:.1f}")
        print(f"  Mean Fidelity: {results['mean_fidelity']:.4f}")

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'eval_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        print(f"Results saved to {output_path}")
