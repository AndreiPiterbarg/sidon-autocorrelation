"""New mathematical pruning bounds for Level 0 survivor reduction.

Provides:
- bounds: All bound functions (block-sum, two-max, cross-term, sum-of-squares)
- validate: Exhaustive validation against ground-truth test values
- benchmark: Pruning power measurement for each bound
"""
from .bounds import (
    block_sum_bound,
    all_block_sum_bounds,
    new_block_sum_bounds,
    two_max_bound,
    cross_term_bound,
    cross_term_c0c5_bound,
    sum_of_squares_bound,
    central_conv_ell2_bound,
    adjacent_conv_ell2_bounds,
    ell3_central_bounds,
    partial_ell4_center_bound,
    best_new_bound,
    true_test_value,
)
