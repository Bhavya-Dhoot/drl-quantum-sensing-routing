"""
Main training script for GNN-PPO quantum routing agent.

Usage:
    python scripts/train.py --config configs/nsfnet.yaml --episodes 10000 --seed 42 --gpu 0
"""

import os
import sys
import argparse
import yaml
import json
import time
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.routing_env import QuantumRoutingEnv
from src.agent.ppo import PPOAgent
from src.agent.vqs_coopt import VQSOptimizer, TwoTimescaleTrainer
from src.utils.seed import set_all_seeds, get_device
from src.utils.logging import TrainingLogger
from src.network.topology import get_num_sensors


def parse_args():
    parser = argparse.ArgumentParser(description='Train GNN-PPO agent')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config YAML')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Override number of training episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--log_dir', type=str, default='results/logs',
                        help='Logging directory')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--save_freq', type=int, default=500,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Evaluate every N episodes')
    parser.add_argument('--use_vqs', type=bool, default=True,
                        help='Enable VQS co-optimization')
    parser.add_argument('--use_mlp', action='store_true',
                        help='Use MLP instead of GAT (ablation)')
    parser.add_argument('--comm_reward', action='store_true',
                        help='Use communication reward instead of QFI (ablation)')
    parser.add_argument('--no_curriculum', action='store_true',
                        help='Disable curriculum learning (ablation)')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and merge configuration."""
    # Load default config first
    default_path = os.path.join(os.path.dirname(config_path), 'default.yaml')
    config = {}
    if os.path.exists(default_path) and config_path != default_path:
        with open(default_path) as f:
            config = yaml.safe_load(f) or {}

    # Override with specific config
    if os.path.exists(config_path):
        with open(config_path) as f:
            specific = yaml.safe_load(f) or {}
        config.update(specific)

    return config


def train(args):
    """Main training loop."""
    config = load_config(args.config)

    # Override config with CLI args
    if args.episodes is not None:
        config['total_episodes'] = args.episodes
    config['seed'] = args.seed

    total_episodes = config.get('total_episodes', 10000)
    rollout_length = config.get('rollout_length', 256)

    # Reproducibility
    set_all_seeds(args.seed)

    # Device
    if args.gpu >= 0:
        device = get_device(args.gpu)
    else:
        device = torch.device('cpu')

    # Environment
    env_config = {
        'topology': config.get('topology', 'nsfnet'),
        'topology_config': config.get('topology_config', {}),
        'seed': args.seed,
        'max_steps': config.get('max_steps', 50),
        'arrival_rate': config.get('arrival_rate', 1.0),
        'network_steps_per_action': config.get('network_steps_per_action', 5),
        'T2': config.get('T2', 1.0),
        'F0': config.get('F0', 0.95),
    }
    env = QuantumRoutingEnv(env_config)

    # Agent
    n_sensors = get_num_sensors(env.topology)
    agent = PPOAgent(
        node_feat_dim=6,
        edge_feat_dim=4,
        hidden_dim=config.get('hidden_dim', 128),
        max_action_dim=env.max_degree,
        lr=config.get('lr_routing', 3e-4),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_epsilon=config.get('clip_epsilon', 0.2),
        entropy_coeff=config.get('entropy_coeff', 0.01),
        value_loss_coeff=config.get('value_loss_coeff', 0.5),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        num_epochs=config.get('num_epochs_per_update', 4),
        batch_size=config.get('batch_size', 512),
        rollout_length=rollout_length,
        use_mlp=args.use_mlp,
        device=str(device),
    )

    # VQS co-optimization
    vqs_trainer = None
    if args.use_vqs:
        vqs_opt = VQSOptimizer(
            n_sensors=n_sensors,
            lr=config.get('lr_vqs', 1e-3),
            device=str(device),
        )
        vqs_trainer = TwoTimescaleTrainer(
            ppo_agent=agent,
            vqs_optimizer=vqs_opt,
        )

    # Logger
    topo_name = config.get('topology', 'nsfnet')
    exp_name = f'{topo_name}_seed{args.seed}'
    if args.use_mlp:
        exp_name += '_mlp'
    if args.comm_reward:
        exp_name += '_comm'
    if args.no_curriculum:
        exp_name += '_nocurr'

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = TrainingLogger(args.log_dir, exp_name)
    logger.save_config(config)

    # Curriculum stages
    curriculum_stages = config.get('curriculum_stages', 3) if not args.no_curriculum else 1
    episodes_per_stage = total_episodes // curriculum_stages
    curriculum_n_sensors = np.linspace(2, n_sensors, curriculum_stages, dtype=int)

    print(f"Training config: {topo_name}, {total_episodes} episodes, "
          f"seed={args.seed}, device={device}")
    print(f"Curriculum stages: {curriculum_stages}, sensors per stage: {curriculum_n_sensors}")

    # Training metrics storage for later analysis
    training_metrics = {
        'episodes': [],
        'qfi': [],
        'reward': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
    }

    start_time = time.time()
    best_qfi = 0.0

    for episode in range(total_episodes):
        # Determine curriculum stage
        stage = min(episode // episodes_per_stage, curriculum_stages - 1)
        current_n_sensors = int(curriculum_n_sensors[stage])

        # Reset environment
        obs, info = env.reset(
            seed=args.seed + episode,
            options={'n_sensors': current_n_sensors}
        )

        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done:
            # Select action
            action, log_prob, value = agent.select_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(obs, action, log_prob, reward, value, done)
            obs = next_obs
            episode_reward += reward
            episode_steps += 1

            # Update policy when buffer is full enough
            if agent.buffer.size >= rollout_length:
                _, _, last_value = agent.select_action(obs)
                update_stats = agent.update(last_value if not done else 0.0)

        # Get episode QFI
        episode_qfi = info.get('episode_qfi', 0.0)

        # VQS update
        if vqs_trainer and episode_qfi > 0:
            fids = np.array(list(info.get('delivered_fidelities', {}).values()))
            if len(fids) > 0:
                vqs_trainer.update_vqs(episode_qfi, fids)

        # Log metrics
        metrics = {
            'reward': episode_reward,
            'qfi': episode_qfi,
            'steps': episode_steps,
            'n_sensors': current_n_sensors,
            'stage': stage,
        }
        logger.log_episode(episode, metrics)

        training_metrics['episodes'].append(episode)
        training_metrics['qfi'].append(episode_qfi)
        training_metrics['reward'].append(episode_reward)

        # Print progress
        if episode % 100 == 0:
            elapsed = time.time() - start_time
            avg_qfi = np.mean(training_metrics['qfi'][-100:]) if training_metrics['qfi'] else 0
            print(f"Episode {episode}/{total_episodes} | "
                  f"QFI: {episode_qfi:.2f} (avg100: {avg_qfi:.2f}) | "
                  f"Reward: {episode_reward:.3f} | "
                  f"Steps: {episode_steps} | "
                  f"N={current_n_sensors} | "
                  f"Time: {elapsed:.0f}s")

        # Save checkpoint
        if episode % args.save_freq == 0 and episode > 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                f'{exp_name}_ep{episode}.pt'
            )
            agent.save(ckpt_path)

        # Track best
        if episode_qfi > best_qfi:
            best_qfi = episode_qfi
            best_path = os.path.join(
                args.checkpoint_dir,
                f'{exp_name}_best.pt'
            )
            agent.save(best_path)

    # Final save
    final_path = os.path.join(args.checkpoint_dir, f'{exp_name}_final.pt')
    agent.save(final_path)

    # Save training metrics
    metrics_path = os.path.join(args.log_dir, f'{exp_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f)

    elapsed = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Best QFI: {best_qfi:.2f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Logs saved to: {args.log_dir}")

    logger.close()
    return training_metrics


if __name__ == '__main__':
    args = parse_args()
    train(args)
