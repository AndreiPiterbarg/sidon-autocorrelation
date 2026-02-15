"""GPU acceleration for Cloninger-Steinerberger branch-and-prune.

Usage:
    from gpu import is_available, gpu_find_best_bound_direct, gpu_run_single_level

    if is_available():
        bound = gpu_find_best_bound_direct(n_half=2, m=100)
"""
from gpu.wrapper import is_available, get_device_name
from gpu.solvers import gpu_find_best_bound_direct, gpu_run_single_level

__all__ = [
    'is_available',
    'get_device_name',
    'gpu_find_best_bound_direct',
    'gpu_run_single_level',
]
