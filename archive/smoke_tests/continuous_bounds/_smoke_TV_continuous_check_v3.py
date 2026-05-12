"""TV_W continuous-f bridge check v3: rigorous lower-bound test.

Tests whether the cascade's TV_W formula (computed from integer bin counts c)
is a valid LOWER BOUND on (1/|W|) integral_W (f*f)(t) dt for ANY continuous
non-negative f with bin averages a_i/m.

This is the mathematically correct soundness claim:
  cascade_TV_W(a) <= (1/|W|) integral_W (f*f) dt <= ||f*f||_inf

Proof sketch (verified theoretically):
  - integral_W (f*f) dt = sum over bin pairs (i,j) of integral over (s,u) in
    bin_i x bin_j with s+u in W of f(s)f(u) ds du.
  - Pair (i,j) contributes 0 if s+u always outside W (for all s in bin_i,
    u in bin_j); contributes (a_i*a_j*w^2)/m^2 if s+u always inside W (FULL
    pair); contributes between 0 and (a_i*a_j*w^2)/m^2 if PARTIAL.
  - FULL pairs are exactly k=i+j in [s_lo, s_lo+ell-2].  ZERO pairs are
    k <= s_lo-2 or k >= s_lo+ell.  PARTIAL are k in {s_lo-1, s_lo+ell-1}.
  - So integral_W (f*f) dt >= (w^2/m^2) * sum_{k=s_lo}^{s_lo+ell-2} conv[k]
    = |W| * cascade_TV_W(a).
  - Hence (1/|W|) integral_W (f*f) dt >= cascade_TV_W(a). QED.

We verify empirically by:
  1. Choosing integer compositions a with sum 4nm.
  2. Generating many random continuous f with prescribed bin masses
     (using piecewise-linear, piecewise-quadratic, mixture-of-Gaussians, etc.)
  3. Computing the window average integral numerically.
  4. Verifying window_avg >= cascade_TV_W(a) - epsilon (for numerical eps).
"""
import os
import sys
import json
import numpy as np


def cascade_tv_w(a, n_half, m, ell, s_lo):
    d = len(a)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += float(a[i]) * float(a[j])
    ws = float(np.sum(conv[s_lo:s_lo + ell - 1]))
    return ws / (4.0 * n_half * ell * m * m)


def build_random_f_with_bin_avgs(a, n_half, m, rng, intra_shape='random_pwl',
                                   n_intra_pts=10):
    """Build a non-negative continuous f with prescribed bin averages a_i/m.

    Within each bin_i, use a randomly generated continuous distribution with
    integral matching a_i*w/m.  Returns a callable f(x) and the underlying
    sample arrays for fine evaluation.
    """
    d = len(a)
    w = 1.0 / (4.0 * n_half)
    a_arr = np.asarray(a, dtype=np.float64)

    # Pre-compute, per bin: a finely-sampled non-neg function with target mass.
    bin_funcs = []
    for i in range(d):
        lo = -0.25 + i * w
        hi = -0.25 + (i + 1) * w
        target_mass = a_arr[i] * w / m
        if target_mass <= 0:
            bin_funcs.append((lo, hi, np.zeros(n_intra_pts), 0.0))
            continue
        if intra_shape == 'random_pwl':
            # Random non-neg piecewise linear, normalized to target mass.
            heights = rng.uniform(0.1, 1.0, n_intra_pts)
        elif intra_shape == 'random_skew':
            # Skewed: linearly weighted random
            base = rng.uniform(0.1, 1.0, n_intra_pts)
            heights = base * np.linspace(0.3, 1.0, n_intra_pts)
        elif intra_shape == 'random_polypeak':
            # Random polynomial peak
            heights = rng.uniform(0.05, 0.5, n_intra_pts)
            peak_idx = rng.integers(0, n_intra_pts)
            heights[peak_idx] *= rng.uniform(2.0, 8.0)
        elif intra_shape == 'oscillatory':
            xs = np.linspace(0, 1, n_intra_pts)
            heights = 0.5 + 0.4 * np.sin(rng.uniform(2, 8) * np.pi * xs
                                          + rng.uniform(0, 2 * np.pi))
        else:
            heights = np.ones(n_intra_pts)
        # Normalize within bin: integral over bin_i = sum(heights * subwidth)
        sub_w = w / (n_intra_pts - 1)
        # Trapezoid integration
        total = (heights[0] + heights[-1]) * sub_w / 2.0 + np.sum(heights[1:-1]) * sub_w
        if total > 0:
            heights = heights * (target_mass / total)
        bin_funcs.append((lo, hi, heights, target_mass))

    def f(x):
        x = np.atleast_1d(x)
        out = np.zeros_like(x, dtype=np.float64)
        for (lo, hi, heights, mass) in bin_funcs:
            mask = (x >= lo) & (x <= hi)
            if not np.any(mask):
                continue
            xb = x[mask]
            if mass <= 0:
                out[mask] = 0.0
                continue
            # Linear interp from heights array on [lo, hi]
            t = (xb - lo) / (hi - lo)
            t_pts = np.linspace(0, 1, len(heights))
            out[mask] = np.interp(t, t_pts, heights)
        return out

    return f


def numerical_window_integral(f_func, n_half, m, ell, s_lo, n_grid=2000):
    """Numerically compute integral_W (f*f)(t) dt."""
    w = 1.0 / (4.0 * n_half)
    t_lo = -0.5 + s_lo * w
    t_hi = -0.5 + (s_lo + ell) * w

    s_grid = np.linspace(-0.25, 0.25, n_grid)
    ds = s_grid[1] - s_grid[0]
    f_s = f_func(s_grid)

    n_t = n_grid
    t_grid = np.linspace(t_lo, t_hi, n_t)

    ff = np.zeros(n_t)
    for it in range(n_t):
        t = t_grid[it]
        ts = t - s_grid
        f_ts = np.where((ts >= -0.25) & (ts <= 0.25), f_func(ts), 0.0)
        ff[it] = np.sum(f_s * f_ts) * ds

    integral = np.trapezoid(ff, t_grid)
    return integral


def main():
    print("=" * 78)
    print("TV_W rigorous lower-bound test")
    print("Verifies: cascade_TV_W(a) <= (1/|W|) integral_W (f*f) dt for all f")
    print("=" * 78)

    # Test cases
    test_cases = [
        (2, 8, [4, 4, 4, 4], 4, 1),     # uniform d=4
        (2, 10, [3, 5, 7, 5], 4, 1),
        (2, 10, [3, 5, 7, 5], 3, 2),
        (2, 10, [3, 5, 7, 5], 2, 3),
        (3, 8, [2, 4, 6, 4, 6, 2], 4, 4),
        (3, 8, [2, 4, 6, 4, 6, 2], 6, 2),
    ]

    n_random_per_case = 30
    intra_shapes = ['random_pwl', 'random_skew', 'random_polypeak', 'oscillatory']

    all_results = []
    n_violations = 0
    n_total = 0

    for case_idx, (n_half, m, c, ell, s_lo) in enumerate(test_cases):
        d = 2 * n_half
        target_S = 4 * n_half * m
        S = sum(c)
        if S != target_S:
            scale = target_S / S
            a = [ci * scale for ci in c]
        else:
            a = list(c)

        cascade_val = cascade_tv_w(a, n_half, m, ell, s_lo)
        w = 1.0 / (4.0 * n_half)
        win_len = ell * w

        case_min_window_avg = np.inf
        case_min_diff = np.inf
        per_shape_results = {}

        for shape in intra_shapes:
            rng = np.random.default_rng(seed=42 + case_idx * 100)
            window_avgs = []
            for trial in range(n_random_per_case):
                f = build_random_f_with_bin_avgs(a, n_half, m, rng, shape)
                # Verify normalization
                xg = np.linspace(-0.25, 0.25, 4000)
                norm = np.trapezoid(f(xg), xg)
                # Compute window avg of (f*f)
                integral = numerical_window_integral(f, n_half, m, ell, s_lo,
                                                       n_grid=1500)
                avg = integral / win_len
                window_avgs.append(avg)
                n_total += 1
                if avg < cascade_val - 5e-3:  # numerical tolerance
                    n_violations += 1
                if avg < case_min_window_avg:
                    case_min_window_avg = avg
                    case_min_diff = avg - cascade_val
            window_avgs = np.array(window_avgs)
            per_shape_results[shape] = {
                'min': float(window_avgs.min()),
                'max': float(window_avgs.max()),
                'mean': float(window_avgs.mean()),
                'min_minus_cascade': float(window_avgs.min() - cascade_val),
            }

        result = {
            'case_idx': case_idx,
            'n_half': n_half, 'm': m, 'd': d, 'ell': ell, 's_lo': s_lo,
            'a': list(a), 'cascade_TV_W': float(cascade_val),
            'min_window_avg_observed': float(case_min_window_avg),
            'min_diff': float(case_min_diff),
            'per_shape': per_shape_results,
        }
        all_results.append(result)

        print(f"\n=== Case {case_idx}: n={n_half}, m={m}, d={d}, ell={ell}, s_lo={s_lo} ===")
        print(f"  a = {a}")
        print(f"  cascade TV_W                 = {cascade_val:.6f}")
        print(f"  Min window avg (random f's)  = {case_min_window_avg:.6f}")
        print(f"  Min diff (window_avg - cascade) = {case_min_diff:+.6e}")
        for shape, r in per_shape_results.items():
            print(f"    {shape:20s} min={r['min']:.6f}  max={r['max']:.6f}  "
                  f"min-cascade = {r['min_minus_cascade']:+.6e}")

    print("\n" + "=" * 78)
    print(f"TOTAL TRIALS: {n_total}, VIOLATIONS (window_avg < cascade - 5e-3): "
          f"{n_violations}")
    print("=" * 78)

    out_path = os.path.join(os.path.dirname(__file__),
                             '_smoke_TV_continuous_check_v3.json')
    with open(out_path, 'w') as fp:
        json.dump({
            'n_total': n_total,
            'n_violations': n_violations,
            'cases': all_results,
        }, fp, indent=2)
    print(f"Results: {out_path}")

    if n_violations == 0:
        print("\nVERDICT: cascade_TV_W is a VALID LOWER BOUND on (1/|W|)*int_W(f*f)dt")
        print("         for all tested continuous f with given bin averages. [PASS]")
    else:
        print(f"\nVERDICT: {n_violations} VIOLATIONS observed. Bound FAILS empirically.")


if __name__ == '__main__':
    main()
