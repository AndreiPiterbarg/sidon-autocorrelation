"""Core routines for Cloninger-Steinerberger branch-and-prune algorithm.

Proves lower bounds on C_{1a} = inf ||f*f||_inf / ||f||_1^2
for nonneg f supported on [-1/4, 1/4].

Reference: Cloninger & Steinerberger, "On Suprema of Autoconvolutions
with an Application to Sidon Sets", Proc. AMS 145(8), 2017.

This module re-exports everything from the sub-modules for backward compat.
"""
from pruning import correction, asymmetry_threshold, count_compositions, asymmetry_prune_mask, _canonical_mask
from compositions import (
    generate_compositions_batched,
    generate_canonical_compositions_batched,
)
from test_values import compute_test_values_batch, compute_test_value_single, _test_values_jit
from solvers import (
    run_single_level, find_best_bound, find_best_bound_direct,
    _find_min_eff_d4, _find_min_eff_d6, _build_interleaved_order,
)
