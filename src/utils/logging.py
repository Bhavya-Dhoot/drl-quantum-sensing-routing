"""
Logging utilities for training and evaluation.

Supports TensorBoard logging and CSV export of training metrics.
"""

import os
import csv
import json
import time
from typing import Any, Dict, List, Optional
from datetime import datetime


class TrainingLogger:
    """Logger for training metrics with TensorBoard and CSV support."""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = 'default',
        use_tensorboard: bool = True,
    ):
        """
        Args:
            log_dir: Base log directory.
            experiment_name: Name for this experiment run.
            use_tensorboard: Whether to use TensorBoard.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(log_dir, f'{experiment_name}_{timestamp}')
        os.makedirs(self.log_dir, exist_ok=True)

        # CSV logging
        self.csv_path = os.path.join(self.log_dir, 'metrics.csv')
        self._csv_writer = None
        self._csv_file = None
        self._csv_fields: Optional[List[str]] = None

        # TensorBoard
        self.use_tensorboard = use_tensorboard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir)
            except ImportError:
                print("TensorBoard not available. Logging to CSV only.")
                self.use_tensorboard = False

        self.step_count = 0
        self.start_time = time.time()

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a scalar value.

        Args:
            tag: Metric name.
            value: Metric value.
            step: Training step (auto-incremented if None).
        """
        if step is None:
            step = self.step_count

        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple scalar values.

        Args:
            metrics: Dict of metric name → value.
            step: Training step.
        """
        if step is None:
            step = self.step_count

        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

        # CSV logging
        row = {'step': step, 'time': time.time() - self.start_time}
        row.update(metrics)
        self._write_csv_row(row)

    def log_episode(self, episode: int, metrics: Dict[str, float]):
        """Log end-of-episode metrics.

        Args:
            episode: Episode number.
            metrics: Episode metrics.
        """
        self.step_count = episode
        self.log_scalars(metrics, step=episode)

    def _write_csv_row(self, row: Dict[str, Any]):
        """Write a row to the CSV file."""
        if self._csv_writer is None:
            self._csv_fields = list(row.keys())
            self._csv_file = open(self.csv_path, 'w', newline='')
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_fields,
                extrasaction='ignore'
            )
            self._csv_writer.writeheader()

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def save_config(self, config: Dict):
        """Save experiment configuration."""
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def close(self):
        """Close all logging resources."""
        if self.writer:
            self.writer.close()
        if self._csv_file:
            self._csv_file.close()

    def __del__(self):
        self.close()
