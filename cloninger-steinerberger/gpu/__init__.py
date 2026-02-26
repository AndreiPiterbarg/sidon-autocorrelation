"""GPU acceleration for Cloninger-Steinerberger branch-and-prune.

Usage:
    from gpu import is_available, gpu_find_best_bound_direct, gpu_run_single_level

    if is_available():
        bound = gpu_find_best_bound_direct(n_half=2, m=100)
"""
from .wrapper import (is_available, get_device_name, get_free_memory,
                      max_survivors_for_dim, load_survivors_chunk)
from .solvers import gpu_find_best_bound_direct, gpu_run_single_level

__all__ = [
    'is_available',
    'get_device_name',
    'gpu_find_best_bound_direct',
    'gpu_run_single_level',
]
