"""GPU acceleration for Cloninger-Steinerberger branch-and-prune.

Usage:
    from gpu import is_available, gpu_find_best_bound_direct, gpu_run_single_level

    if is_available():
        bound = gpu_find_best_bound_direct(n_half=2, m=100)

    # Hierarchical multi-level refinement:
    from gpu import hierarchical_prove
    result = hierarchical_prove(c_target=1.20, n_base=3, m=50)
"""
from gpu.wrapper import (is_available, get_device_name, get_free_memory,
                         max_survivors_for_dim, load_survivors_chunk)
from gpu.solvers import gpu_find_best_bound_direct, gpu_run_single_level
from gpu.multilevel import hierarchical_prove

__all__ = [
    'is_available',
    'get_device_name',
    'gpu_find_best_bound_direct',
    'gpu_run_single_level',
    'hierarchical_prove',
]
