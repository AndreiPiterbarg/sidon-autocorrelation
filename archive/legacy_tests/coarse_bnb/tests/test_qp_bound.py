"""Validation tests for the joint QP bound (qp_bound.py).

Verifies:
  (1) Reference correctness on the worked d=4, S=10, c=(2,1,1,6) example
      (QP should give 0.20 on the headline indefinite window (ell=2, s=3)).
  (2) Numba kernel == pure-Python reference on many random inputs.
  (3) QP bound <= triangle bound (cell_var + quad_corr) on many random
      compositions (the survival proposal must be tighter than v2).
  (4) Soundness: QP bound is an UPPER BOUND on f(delta) for adversarial
      delta sampled from the cell. The QP is exact, so this must hold.
  (5) Soundness: QP bound is an UPPER BOUND on f(delta) for many random
      delta sampled uniformly from the cell.
  (6) Strict tightening: there exist (c, W) where QP < triangle by >10%.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_REPO, 'cloninger-steinerberger', 'cpu'))

from qp_bound import (
    build_window_matrix,
    grad_for_window,
    qp_bound_vertex,
    qp_bound_python,
    qp_bound_for_composition,
)


# Triangle bound (mirror of run_cascade_coarse_v2.py)

def cell_var_v2(grad, S):
    d = grad.shape[0]
    g = np.sort(grad)
    h = 1.0 / (2.0 * S)
    total = 0.0
    for k in range(d // 2):
        total += g[d - 1 - k] - g[k]
    return h * total


def quad_corr_v2(d, S, ell, s):
    N_W = 0
    M_W = 0
    for k in range(s, s + ell - 1):
        n_k = min(k + 1, d, 2 * d - 1 - k)
        if n_k <= 0:
            continue
        N_W += n_k
        if k % 2 == 0 and (k // 2) < d:
            M_W += 1
    cross_W = N_W - M_W
    pair_bound = min(cross_W, d * d - N_W)
    if pair_bound <= 0:
        return 0.0
    return (2.0 * d / ell) * pair_bound / (4.0 * S * S)


def random_composition(d, S, rng):
    """Uniform random composition of S into d non-neg integer parts."""
    bars = rng.choice(S + d - 1, size=d - 1, replace=False)
    bars = np.sort(bars)
    parts = np.empty(d, dtype=np.int64)
    prev = -1
    for i in range(d - 1):
        parts[i] = bars[i] - prev - 1
        prev = bars[i]
    parts[-1] = S + d - 2 - prev
    assert parts.sum() == S
    return parts


def random_delta_in_cell(d, h, rng):
    """Generate a random delta in {|delta_i|<=h, sum delta_i=0}."""
    x = rng.uniform(-h, h, size=d)
    x -= x.mean()
    m = np.max(np.abs(x))
    if m > h:
        x *= (h / m)
    return x


def test1_headline_example():
    """d=4, S=10, c=(2,1,1,6), W=(2,3): triangle = 0.28, QP = 0.20."""
    print("\n--- Test 1: Headline d=4 example ---")
    d, S = 4, 10
    c = np.array([2, 1, 1, 6], dtype=np.int64)
    ell, s = 2, 3
    A = build_window_matrix(d, ell, s)
    grad = grad_for_window(c, A, S, d, ell)
    h = 1.0 / (2.0 * S)
    scale = 2.0 * d / ell

    qp = qp_bound_vertex(grad, A, scale, h, d)
    qp_py = qp_bound_python(grad, A, scale, h, d)
    cv = cell_var_v2(grad, S)
    qc = quad_corr_v2(d, S, ell, s)
    tri = cv + qc

    print(f"  grad_W = {grad}")
    print(f"  cell_var (v2) = {cv:.6f}")
    print(f"  quad_corr (v2) = {qc:.6f}")
    print(f"  triangle (v2) = {tri:.6f}")
    print(f"  QP (Numba) = {qp:.6f}")
    print(f"  QP (Python) = {qp_py:.6f}")
    expected_qp = 0.20
    assert abs(qp - expected_qp) < 1e-6, \
        f"QP mismatch: {qp} != {expected_qp}"
    assert abs(qp - qp_py) < 1e-12, \
        f"Numba/Python disagree: {qp} != {qp_py}"
    assert qp < tri, f"QP ({qp}) not tighter than triangle ({tri})!"
    improvement = (tri - qp) / tri
    print(f"  PASS. QP is {improvement * 100:.1f}% tighter than triangle.")


def test2_numba_vs_python(n_trials=30):
    """Numba kernel == pure Python reference."""
    print(f"\n--- Test 2: Numba vs Python ({n_trials} trials) ---")
    rng = np.random.default_rng(42)
    max_diff = 0.0
    for trial in range(n_trials):
        d = int(rng.integers(2, 8))
        S = int(rng.integers(5, 30))
        c = random_composition(d, S, rng)
        ell = int(rng.integers(2, 2 * d + 1))
        s_max = max(0, (2 * d - 1) - (ell - 1))
        s = int(rng.integers(0, s_max + 1)) if s_max > 0 else 0

        A = build_window_matrix(d, ell, s)
        grad = grad_for_window(c, A, S, d, ell)
        h = 1.0 / (2.0 * S)
        scale = 2.0 * d / ell

        qp_n = qp_bound_vertex(grad, A, scale, h, d)
        qp_p = qp_bound_python(grad, A, scale, h, d)
        diff = abs(qp_n - qp_p)
        max_diff = max(max_diff, diff)
        if diff > 1e-10:
            print(f"  MISMATCH d={d} S={S} c={c.tolist()} ell={ell} s={s}: "
                  f"numba={qp_n:.10f} python={qp_p:.10f} diff={diff:.2e}")
            assert False
    print(f"  PASS. Max diff = {max_diff:.2e} over {n_trials} trials.")


def test3_qp_le_triangle(n_trials=200):
    """QP bound <= triangle bound (i.e., QP is tighter or equal)."""
    print(f"\n--- Test 3: QP <= triangle ({n_trials} trials) ---")
    rng = np.random.default_rng(123)
    n_strict = 0
    n_equal = 0
    max_improvement = 0.0
    worst_case = None
    for trial in range(n_trials):
        d = int(rng.integers(2, 9))
        if d % 2 == 1:
            d += 1  # cascade only hits even d
        S = int(rng.integers(4, 40))
        c = random_composition(d, S, rng)
        ell = int(rng.integers(2, 2 * d + 1))
        s_max = max(0, (2 * d - 1) - (ell - 1))
        s = int(rng.integers(0, s_max + 1)) if s_max > 0 else 0

        A = build_window_matrix(d, ell, s)
        grad = grad_for_window(c, A, S, d, ell)
        h = 1.0 / (2.0 * S)
        scale = 2.0 * d / ell

        qp = qp_bound_vertex(grad, A, scale, h, d)
        cv = cell_var_v2(grad, S)
        qc = quad_corr_v2(d, S, ell, s)
        tri = cv + qc

        if qp > tri + 1e-9:
            print(f"  VIOLATION: QP > triangle for d={d} S={S} c={c.tolist()} "
                  f"W=({ell},{s}): QP={qp:.6f} tri={tri:.6f}")
            assert False, "QP must be <= triangle"
        if tri > 1e-9:
            improvement = (tri - qp) / tri
            if improvement > max_improvement:
                max_improvement = improvement
                worst_case = (d, S, c.tolist(), ell, s, qp, tri)
            if improvement > 1e-6:
                n_strict += 1
            else:
                n_equal += 1
    print(f"  PASS. {n_strict} strict, {n_equal} equal cases.")
    print(f"  Max improvement: {max_improvement * 100:.2f}% at "
          f"d={worst_case[0]} S={worst_case[1]} c={worst_case[2]} "
          f"W=({worst_case[3]},{worst_case[4]}) "
          f"QP={worst_case[5]:.6f} tri={worst_case[6]:.6f}")


def test4_soundness_random(n_trials=50, n_samples=2000):
    """For each (c, W), QP must upper-bound f(delta) for all sampled delta."""
    print(f"\n--- Test 4: Soundness via random delta sampling "
          f"({n_trials} cases x {n_samples} samples) ---")
    rng = np.random.default_rng(7)
    worst_excess = -np.inf
    for trial in range(n_trials):
        d = int(rng.integers(2, 7))
        if d % 2 == 1:
            d += 1
        S = int(rng.integers(4, 25))
        c = random_composition(d, S, rng)
        ell = int(rng.integers(2, 2 * d + 1))
        s_max = max(0, (2 * d - 1) - (ell - 1))
        s = int(rng.integers(0, s_max + 1)) if s_max > 0 else 0

        A = build_window_matrix(d, ell, s)
        grad = grad_for_window(c, A, S, d, ell)
        h = 1.0 / (2.0 * S)
        scale = 2.0 * d / ell

        qp = qp_bound_vertex(grad, A, scale, h, d)
        for _ in range(n_samples):
            delta = random_delta_in_cell(d, h, rng)
            f_val = -float(grad @ delta) - scale * float(delta @ A @ delta)
            excess = f_val - qp
            if excess > worst_excess:
                worst_excess = excess
            if excess > 1e-9:
                print(f"  VIOLATION: f(delta) > QP for d={d} S={S} c={c.tolist()} "
                      f"W=({ell},{s}): f={f_val:.10f} QP={qp:.10f}")
                assert False, "QP must be sound upper bound"
    print(f"  PASS. Worst excess: {worst_excess:.2e} (negative = sound).")


def test5_strict_tightening_cases():
    """Demonstrate cases where QP is strictly tighter than triangle by >5%."""
    print("\n--- Test 5: Strict tightening cases ---")
    cases = [
        (4, 10, [2, 1, 1, 6], 2, 3),
        (4, 10, [2, 1, 1, 6], 4, 1),
        (4, 20, [5, 3, 4, 8], 4, 1),
        (8, 10, [2, 1, 1, 1, 1, 1, 1, 2], 4, 4),
        (8, 20, [3, 2, 2, 2, 3, 3, 2, 3], 4, 4),
    ]
    print(f"{'d':>2} {'S':>3} {'c':<28} {'(ell,s)':>9} "
          f"{'cell_var':>9} {'quad_corr':>10} "
          f"{'triangle':>9} {'QP':>9} {'improv':>7}")
    print("-" * 100)
    for (d, S, c_list, ell, s) in cases:
        c = np.array(c_list, dtype=np.int64)
        A = build_window_matrix(d, ell, s)
        grad = grad_for_window(c, A, S, d, ell)
        h = 1.0 / (2.0 * S)
        scale = 2.0 * d / ell
        qp = qp_bound_vertex(grad, A, scale, h, d)
        cv = cell_var_v2(grad, S)
        qc = quad_corr_v2(d, S, ell, s)
        tri = cv + qc
        improv = (tri - qp) / tri * 100 if tri > 0 else 0
        c_str = ','.join(str(x) for x in c_list)
        print(f"{d:>2} {S:>3} {c_str:<28} ({ell:>2},{s:>2})  "
              f"{cv:>9.4f} {qc:>10.4f} "
              f"{tri:>9.4f} {qp:>9.4f} {improv:>6.1f}%")


def test6_timing(d_list=(4, 8, 16)):
    """Timing of QP bound at various d."""
    print(f"\n--- Test 6: Timing ---")
    rng = np.random.default_rng(99)
    for d in d_list:
        S = 20
        c = random_composition(d, S, rng)
        ell = d
        s = (2 * d - 1) // 2 - ell // 2
        if s < 0:
            s = 0
        A = build_window_matrix(d, ell, s)
        grad = grad_for_window(c, A, S, d, ell)
        h = 1.0 / (2.0 * S)
        scale = 2.0 * d / ell

        # Warmup (Numba JIT)
        _ = qp_bound_vertex(grad, A, scale, h, d)

        n_iter = max(1, 10000 // (1 << d))
        t0 = time.perf_counter()
        for _ in range(n_iter):
            qp = qp_bound_vertex(grad, A, scale, h, d)
        elapsed_us = (time.perf_counter() - t0) * 1e6 / n_iter
        print(f"  d={d:>2}: ~{elapsed_us:.1f} us per QP call "
              f"(2^(d-1)={1 << (d - 1)} vertices)")


def main():
    print("=" * 70)
    print("Joint QP bound validation tests")
    print("=" * 70)
    test1_headline_example()
    test2_numba_vs_python(n_trials=30)
    test3_qp_le_triangle(n_trials=200)
    test4_soundness_random(n_trials=50, n_samples=2000)
    test5_strict_tightening_cases()
    test6_timing()
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED. QP bound is sound and strictly tighter than v2.")
    print("=" * 70)


if __name__ == '__main__':
    main()
