"""Exhaustive validation of pruning bounds.

Enumerates ALL canonical compositions and verifies that each bound
is a valid lower bound on the true test value. ZERO violations allowed.
"""
import numpy as np
import time
import sys
import os

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from cpu.compositions import generate_canonical_compositions_batched
from test_values import compute_test_value_single
from gpu.pruning.bounds import (
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
    existing_max_element_bound,
    existing_half_sum_bound,
)


# Tolerance for floating-point comparison: bound <= true_val + eps
EPS = 1e-12


def _validate_single_bound(bound_fn, bound_name, c, n_half, m, true_val):
    """Check that bound_fn(c, n_half, m) <= true_val + EPS.

    Returns (passed: bool, bound_val: float, violation_gap: float).
    """
    bound_val = bound_fn(c, n_half, m)
    gap = bound_val - true_val
    passed = gap <= EPS
    return passed, bound_val, gap


def validate_bound_exhaustive(bound_fn, bound_name, n_half, m, batch_size=100000):
    """Validate a single bound over ALL canonical compositions.

    Parameters
    ----------
    bound_fn : callable(c, n_half, m) -> float
    bound_name : str
    n_half : int
    m : int

    Returns
    -------
    dict with keys: passed, total, violations, max_gap, worst_config, time_sec
    """
    d = 2 * n_half
    S = 4 * n_half * m

    total = 0
    violations = 0
    max_gap = -np.inf
    worst_config = None
    worst_bound_val = 0.0
    worst_true_val = 0.0

    t0 = time.time()

    for batch in generate_canonical_compositions_batched(d, S, batch_size):
        for row_idx in range(len(batch)):
            c = batch[row_idx]
            a = c.astype(np.float64) / m
            true_val = compute_test_value_single(a, n_half)

            passed, bound_val, gap = _validate_single_bound(
                bound_fn, bound_name, c, n_half, m, true_val
            )
            total += 1

            if not passed:
                violations += 1
                if gap > max_gap:
                    max_gap = gap
                    worst_config = c.copy()
                    worst_bound_val = bound_val
                    worst_true_val = true_val

        # Progress
        if total % 500000 == 0:
            elapsed = time.time() - t0
            print(f"  [{bound_name}] n={n_half} m={m}: {total:,} checked, "
                  f"{violations} violations, {elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0

    result = {
        'passed': violations == 0,
        'total': total,
        'violations': violations,
        'max_gap': max_gap if violations > 0 else 0.0,
        'worst_config': worst_config,
        'worst_bound_val': worst_bound_val,
        'worst_true_val': worst_true_val,
        'time_sec': elapsed,
    }

    status = "PASS" if result['passed'] else "FAIL"
    print(f"  [{bound_name}] n={n_half} m={m}: {status} "
          f"({total:,} configs, {violations} violations, {elapsed:.1f}s)")
    if violations > 0:
        print(f"    Worst violation: bound={worst_bound_val:.10f} > "
              f"true={worst_true_val:.10f} (gap={max_gap:.2e})")
        print(f"    Config: {worst_config}")

    return result


def validate_all_bounds():
    """Run exhaustive validation for all bounds at multiple parameter sets.

    D=4: n=2, m in {5, 10, 20}
    D=6: n=3, m=5

    Prints results and returns True if all passed.
    """
    # Define all bounds to validate
    bounds = [
        # (name, function)
        ("block_sum_all", lambda c, n, m: all_block_sum_bounds(c, n, m)),
        ("new_block_sum", lambda c, n, m: new_block_sum_bounds(c, n, m)),
        ("two_max", lambda c, n, m: two_max_bound(c, n, m)),
        ("sum_of_squares", lambda c, n, m: sum_of_squares_bound(c, n, m)),
        ("central_conv_ell2", lambda c, n, m: central_conv_ell2_bound(c, n, m)),
        ("adjacent_conv_ell2", lambda c, n, m: adjacent_conv_ell2_bounds(c, n, m)),
        ("ell3_central", lambda c, n, m: ell3_central_bounds(c, n, m)),
        ("partial_ell4_ctr", lambda c, n, m: partial_ell4_center_bound(c, n, m)),
        ("existing_max_elem", lambda c, n, m: existing_max_element_bound(c, n, m)),
        ("existing_half_sum", lambda c, n, m: existing_half_sum_bound(c, n, m)),
    ]

    # D=6-specific bounds
    bounds_d6 = [
        ("cross_c0c5", lambda c, n, m: cross_term_c0c5_bound(c, n, m)),
    ]

    # Parameter sets: (n_half, m_values)
    param_sets = [
        (2, [5, 10, 20]),   # D=4
        (3, [5]),            # D=6
    ]

    all_passed = True
    results = []

    for n_half, m_values in param_sets:
        d = 2 * n_half
        for m in m_values:
            print(f"\n{'='*60}")
            print(f"Validating D={d}, n={n_half}, m={m} (S={4*n_half*m})")
            print(f"{'='*60}")

            active_bounds = bounds[:]
            if d == 6:
                active_bounds.extend(bounds_d6)

            for name, fn in active_bounds:
                result = validate_bound_exhaustive(fn, name, n_half, m)
                results.append((d, m, name, result))
                if not result['passed']:
                    all_passed = False

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    for d, m, name, result in results:
        status = "PASS" if result['passed'] else "***FAIL***"
        print(f"  D={d} m={m:3d} {name:20s}: {status} "
              f"({result['total']:>10,} configs, {result['time_sec']:.1f}s)")

    print(f"\nOverall: {'ALL PASSED' if all_passed else 'FAILURES DETECTED'}")
    return all_passed


if __name__ == '__main__':
    success = validate_all_bounds()
    sys.exit(0 if success else 1)
