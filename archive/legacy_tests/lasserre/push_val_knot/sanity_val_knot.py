"""Sanity check: take a candidate mu and compute ||f_step * f_step||_inf
two ways:
(a) closed form via cross-term lemma: max_s 4n MC[s]
(b) explicit numerical convolution on a fine grid

These must agree (within quadrature error). Confirms our 'upper bound on C_{1a}'
is actually the EXACT inf-norm of an admissible step function.
"""
from __future__ import annotations

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasserre.fourier_xterm import (
    BinGeometry, step_autoconv_inf_norm,
)


def explicit_step_autoconv_inf_norm(mu, n_grid: int = 5000):
    """Build f_step on a dense grid, compute (f*f) by FFT convolution,
    return max."""
    d = len(mu)
    n = d // 2
    h = 1.0 / (2.0 * d)
    pts_per_bin = max(10, n_grid // d)
    total_pts = pts_per_bin * d

    # f_step on grid
    f_step = np.zeros(total_pts)
    for i in range(d):
        f_step[i * pts_per_bin:(i + 1) * pts_per_bin] = 4.0 * n * mu[i]

    # Step size for the discretization of [-1/4, 1/4]
    dx = 0.5 / total_pts

    # Discrete convolution
    ff = np.convolve(f_step, f_step) * dx
    # ff has length 2 * total_pts - 1, supports [-1/2, 1/2]
    return float(ff.max())


def check_admissible(mu, atol: float = 1e-6):
    """f_step is admissible iff mu >= 0 and sum mu = 1."""
    assert mu.min() >= -atol, f"mu has negative entry: {mu.min()}"
    assert abs(mu.sum() - 1.0) < atol, f"sum mu = {mu.sum()}"
    # Renormalize for safety
    return mu / mu.sum()


def main():
    # The d=8 best mu from previous run (val_knot = 1.5796)
    test_cases = [
        # d=4
        np.array([0.26066549, 0.39429492, 0.09608037, 0.24895923]),
        # uniform d=4
        np.ones(4) / 4,
        # d=8 from earlier results (1.5796)
        np.array([0.11763589, 0.13404930, 0.16041973, 0.23681846,
                  0.04031483, 0.04899280, 0.07043925, 0.19132973]),
    ]

    print("=" * 78)
    print("Sanity check: closed-form max_s 4n MC[s]  vs  numeric (FFT) inf-norm")
    print("=" * 78)

    for case_idx, mu in enumerate(test_cases):
        d = len(mu)
        mu = check_admissible(mu)
        print(f"\n  Case {case_idx + 1}: d={d}")
        print(f"    mu = {mu}")
        print(f"    sum mu = {mu.sum():.10f}")

        v_closed = step_autoconv_inf_norm(mu)
        v_numeric = explicit_step_autoconv_inf_norm(mu, n_grid=5000)

        diff = abs(v_closed - v_numeric)
        rel_err = diff / max(v_closed, 1e-10)
        print(f"    closed-form (cross-term lemma):  {v_closed:.6f}")
        print(f"    numeric FFT convolution:         {v_numeric:.6f}")
        print(f"    diff:                            {diff:.6f}  (rel err {rel_err:.4f})")

        if rel_err > 0.01:
            print(f"    *** WARNING: discrepancy > 1% ***")
        else:
            print(f"    PASS: agree within numeric tolerance")
        print(f"    ==> This mu gives a valid UPPER bound on C_{{1a}} of {v_closed:.5f}")


if __name__ == "__main__":
    main()
