"""
Reproducibility utilities.

Sets all random seeds for Python, NumPy, PyTorch, and CUDA
to ensure deterministic results.
"""

import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(gpu_id: int = 0) -> torch.device:
    """Get computation device with CUDA fallback to CPU.

    Args:
        gpu_id: GPU device ID.

    Returns:
        torch.device for computation.
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        # Check VRAM
        total_mem = torch.cuda.get_device_properties(device).total_mem
        if total_mem < 2 * 1024**3:  # Less than 2GB
            print(f"Warning: GPU has only {total_mem / 1024**3:.1f}GB VRAM. "
                  "Consider reducing batch size.")
        return device
    else:
        print("CUDA not available. Using CPU.")
        return torch.device('cpu')
