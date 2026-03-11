"""
Run all experiments: training, baselines, scaling, robustness.

Usage:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --topology nsfnet --seeds 5 --episodes 10000
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='Run full experiment suite')
    parser.add_argument('--topology', type=str, default='nsfnet')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer episodes and seeds')
    return parser.parse_args()


def run_cmd(cmd: str, desc: str) -> int:
    """Run a command and print its output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {desc}")
    print(f"CMD: {cmd}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    status = "✓ PASSED" if result.returncode == 0 else "✗ FAILED"
    print(f"{status} ({elapsed:.0f}s)\n")
    return result.returncode


def main():
    args = parse_args()

    if args.quick:
        args.episodes = 2000
        args.seeds = 2
        print("Quick mode: 2000 episodes, 2 seeds")

    scripts_dir = Path(__file__).parent
    project_root = scripts_dir.parent
    python = sys.executable

    total_start = time.time()
    results_dir = args.output_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'checkpoints'), exist_ok=True)

    # Experiment 1: Unit Tests
    ret = run_cmd(
        f'{python} -m pytest tests/ -v --tb=short',
        'Experiment 1: Unit Tests'
    )
    if ret != 0:
        print("Unit tests failed! Fix issues before proceeding.")
        print("Continuing anyway for demonstration...")

    # Experiment 2: Training (multiple seeds)
    for seed in range(args.seeds):
        config_path = os.path.join(project_root, 'configs', f'{args.topology}.yaml')
        if not os.path.exists(config_path):
            config_path = os.path.join(project_root, 'configs', 'default.yaml')

        run_cmd(
            f'{python} {scripts_dir / "train.py"} '
            f'--config {config_path} '
            f'--episodes {args.episodes} '
            f'--seed {seed} '
            f'--gpu {args.gpu} '
            f'--log_dir {results_dir}/logs '
            f'--checkpoint_dir {results_dir}/checkpoints',
            f'Experiment 2: Training seed {seed}'
        )

    # Experiment 3: Ablation - MLP encoder
    run_cmd(
        f'{python} {scripts_dir / "train.py"} '
        f'--config {os.path.join(project_root, "configs", "default.yaml")} '
        f'--episodes {args.episodes // 2} '
        f'--seed 0 --use_mlp '
        f'--log_dir {results_dir}/logs '
        f'--checkpoint_dir {results_dir}/checkpoints',
        'Experiment 3a: Ablation - MLP encoder'
    )

    # Experiment 3b: Ablation - Communication reward
    run_cmd(
        f'{python} {scripts_dir / "train.py"} '
        f'--config {os.path.join(project_root, "configs", "default.yaml")} '
        f'--episodes {args.episodes // 2} '
        f'--seed 0 --comm_reward '
        f'--log_dir {results_dir}/logs '
        f'--checkpoint_dir {results_dir}/checkpoints',
        'Experiment 3b: Ablation - Communication reward'
    )

    # Experiment 3c: Ablation - No curriculum
    run_cmd(
        f'{python} {scripts_dir / "train.py"} '
        f'--config {os.path.join(project_root, "configs", "default.yaml")} '
        f'--episodes {args.episodes // 2} '
        f'--seed 0 --no_curriculum '
        f'--log_dir {results_dir}/logs '
        f'--checkpoint_dir {results_dir}/checkpoints',
        'Experiment 3c: Ablation - No curriculum'
    )

    # Experiment 4: Run baselines
    run_cmd(
        f'{python} {scripts_dir / "run_baselines.py"} '
        f'--topology {args.topology} '
        f'--seeds {args.seeds} '
        f'--output_dir {results_dir}',
        'Experiment 4: Baselines'
    )

    # Experiment 5: Scaling
    best_ckpt = os.path.join(results_dir, 'checkpoints',
                             f'{args.topology}_seed0_best.pt')
    ckpt_flag = f'--checkpoint {best_ckpt}' if os.path.exists(best_ckpt) else ''
    run_cmd(
        f'{python} {scripts_dir / "evaluate.py"} '
        f'--scaling --N_values 4,6,8,12,16,24 '
        f'--topology {args.topology} '
        f'--output_dir {results_dir} '
        f'{ckpt_flag}',
        'Experiment 5: QFI Scaling'
    )

    # Experiment 6: Robustness
    run_cmd(
        f'{python} {scripts_dir / "evaluate.py"} '
        f'--robustness --T2_values 0.05,0.1,0.2,0.5,1,2,5,10 '
        f'--topology {args.topology} '
        f'--output_dir {results_dir} '
        f'{ckpt_flag}',
        'Experiment 6: Robustness (T₂ sweep)'
    )

    # Experiment 7: Generate figures
    run_cmd(
        f'{python} {scripts_dir / "generate_figures.py"} '
        f'--results_dir {results_dir} '
        f'--output_dir {results_dir}/figures',
        'Experiment 7: Generate Figures'
    )

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    print(f"Results: {results_dir}")
    print(f"Figures: {results_dir}/figures")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
