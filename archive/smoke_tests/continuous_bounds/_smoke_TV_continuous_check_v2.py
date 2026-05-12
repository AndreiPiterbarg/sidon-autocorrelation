"""TV_W continuous-f bridge check v2: search for adversarial f.

Question: Is the cascade's TV_W formula a valid LOWER BOUND on
(1/|W|) integral_W (f*f) dt for ANY continuous f >= 0 with bin averages a_i/m?

If yes, then cascade TV_W <= window avg <= ||f*f||_inf.

If NO, then cascade TV_W cannot be used directly with a continuous correction
that ignores intra-bin shape; the correction must account for adversarial
intra-bin distributions.

We search for adversarial intra-bin shapes that minimize window_avg of (f*f)
subject to the bin-average constraint, and check whether window_avg can fall
below cascade_TV_W.

The key observation: for two functions f, g with same bin averages, the
window average difference depends on intra-bin moments.  So we can construct
'extremal' functions: e.g., delta-functions at one point in each bin, or
delta-functions at the bin boundary.  These are LIMITS of valid continuous
non-negative f, so any rigorous bound must hold for them.

In particular, consider DELTA functions: place all mass of bin_i as a Dirac
at position x_i within bin_i.  Then (f*f) is a sum of Diracs.  The window
average becomes a sum over Dirac pairs i,j whose sum-position x_i + x_j lies
in W.

For DIRAC f: window_avg behaves discretely.  As we shift x_i within bin_i,
the contribution to the window jumps when x_i + x_j crosses W boundary.

This is the most extreme case where intra-bin shape matters.
"""
import numpy as np


def cascade_tv_w(c, n_half, m, ell, s_lo):
    d = len(c)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += float(c[i]) * float(c[j])
    ws = float(np.sum(conv[s_lo:s_lo + ell - 1]))
    return ws / (4.0 * n_half * ell * m * m)


def closed_form_step_window_avg(a, n_half, m, ell, s_lo):
    """Window avg for step f_a = cascade_TV_W + edge corrections."""
    d = len(a)
    w = 1.0 / (4.0 * n_half)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += float(a[i]) * float(a[j])

    integral = 0.0
    for k in range(s_lo, s_lo + ell - 1):
        if 0 <= k < conv_len:
            integral += conv[k] * w * w
    k_left = s_lo - 1
    if 0 <= k_left < conv_len:
        integral += conv[k_left] * w * w / 2.0
    k_right = s_lo + ell - 1
    if 0 <= k_right < conv_len:
        integral += conv[k_right] * w * w / 2.0

    integral /= (m * m)
    return integral / (ell * w)


def adversarial_dirac_window_avg(a, n_half, m, ell, s_lo, positions):
    """For DIRAC mass distribution within each bin.

    f = sum_i (mass_i) * delta(x - x_i),  mass_i = a_i * w / m.
    f*f = sum_{i,j} mass_i * mass_j * delta(x - (x_i + x_j)).

    integral_W (f*f) dt = sum_{i,j: x_i+x_j in W} mass_i * mass_j.

    positions: array of x_i values (one per bin), each in bin_i.
    """
    d = len(a)
    w = 1.0 / (4.0 * n_half)
    win_lo = -0.5 + s_lo * w
    win_hi = -0.5 + (s_lo + ell) * w

    # mass_i = a_i * w / m
    masses = np.array([ai * w / m for ai in a])

    integral = 0.0
    for i in range(d):
        for j in range(d):
            sum_pos = positions[i] + positions[j]
            if win_lo <= sum_pos <= win_hi:
                integral += masses[i] * masses[j]
    return integral / (ell * w)


def search_min_dirac_window_avg(a, n_half, m, ell, s_lo, n_samples=10000):
    """Sample random Dirac positions, find minimum window avg."""
    d = len(a)
    w = 1.0 / (4.0 * n_half)
    rng = np.random.default_rng(42)
    best_min = np.inf
    best_positions = None

    for trial in range(n_samples):
        positions = np.array([
            -0.25 + i * w + rng.uniform(0, w) for i in range(d)
        ])
        avg = adversarial_dirac_window_avg(a, n_half, m, ell, s_lo, positions)
        if avg < best_min:
            best_min = avg
            best_positions = positions.copy()

    return best_min, best_positions


def main():
    print("=" * 78)
    print("Adversarial Dirac search: can window_avg < cascade_TV_W?")
    print("=" * 78)

    test_cases = [
        (2, 10, [3, 5, 7, 5], 4, 1),
        (2, 10, [4, 4, 6, 6], 3, 2),
        (3, 8, [2, 3, 4, 5, 4, 6], 5, 3),
        (2, 5, [3, 2, 1, 4], 2, 3),
        (2, 10, [10, 0, 0, 10], 2, 3),  # Concentrated at edges
        (2, 10, [0, 10, 10, 0], 4, 1),  # Concentrated at center
    ]

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
        step_avg = closed_form_step_window_avg(a, n_half, m, ell, s_lo)
        min_dirac, best_pos = search_min_dirac_window_avg(
            a, n_half, m, ell, s_lo, n_samples=20000)

        # Also try positioning all Diracs at extremes (left or right edge of bin)
        w = 1.0 / (4.0 * n_half)
        # All at left edge
        pos_all_left = np.array([-0.25 + i * w for i in range(d)])
        avg_left = adversarial_dirac_window_avg(a, n_half, m, ell, s_lo,
                                                  pos_all_left)
        # All at right edge
        pos_all_right = np.array([-0.25 + (i + 1) * w - 1e-12
                                    for i in range(d)])
        avg_right = adversarial_dirac_window_avg(a, n_half, m, ell, s_lo,
                                                   pos_all_right)
        # All at center
        pos_all_mid = np.array([-0.25 + (i + 0.5) * w for i in range(d)])
        avg_mid = adversarial_dirac_window_avg(a, n_half, m, ell, s_lo,
                                                 pos_all_mid)

        print(f"\n=== Case {case_idx}: n={n_half}, m={m}, d={d}, ell={ell}, s_lo={s_lo} ===")
        print(f"  a = {a}")
        print(f"  cascade TV_W                   = {cascade_val:.6f}")
        print(f"  step f_a window avg            = {step_avg:.6f}")
        print(f"  Dirac all-left window avg      = {avg_left:.6f}")
        print(f"  Dirac all-mid window avg       = {avg_mid:.6f}")
        print(f"  Dirac all-right window avg     = {avg_right:.6f}")
        print(f"  Dirac MIN (random search)      = {min_dirac:.6f}")
        print(f"  Best positions (relative): {[(p+0.25)/w for p in best_pos]}")
        if min_dirac < cascade_val:
            print(f"  !!! VIOLATION: window_avg ({min_dirac}) < cascade_TV_W ({cascade_val})")
        else:
            print(f"  OK: window_avg >= cascade_TV_W (gap = {min_dirac - cascade_val:+.6e})")


if __name__ == '__main__':
    main()
