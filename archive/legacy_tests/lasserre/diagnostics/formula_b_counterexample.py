"""
Numerical verification of the Formula B counterexample documented in
proof/formula_b_coarse_grid_proof.md (Section 5.1).

Setup (matches the proof document exactly):
  d   = 4 bins        (so n = d/2 = 2, giving the 4n/ell = 4/2 = 2 ratio
                       claimed in the proof; the document quotes "ratio 3.07
                       ~ 4n/ell = 4" for d=4 -- we verify the actual ratio
                       directly here.)
  m   = 10            (mass quantization)
  c   = (2, 1, 1, 6)  (composition, sum = m)
  h   = 1/(4n) = 1/8  (bin width covering [-1/4, 1/4])

Window scan is over knots x_k = -1/2 + k*h, k = 0,...,4n.  We focus on
the narrow window (ell=2, s=6) called out in the proof.

Formula B per-window pruning bound (claimed sound by C&S eq(1)):
    boundToBeat = c_target + 1/m^2 + 2*W/m
where W = sum of g_c masses (in physical units) over bins contributing
to the window.

We compare two quantities:
  (TV)  TV(c; ell, s) = (4n/ell) sum_{k=s..s+ell-2} MC_w[k]
                      = average of (g_c*g_c)(x_{k+1}) over the window
  (B)   the Formula B pruning threshold for a hypothetical c_target
  (E)   the actual per-window error: |TV(c; ell, s) - TV(mu; ell, s)|
        for the worst admissible cumulative-floor mass perturbation mu

If E > (B - c_target) for some admissible mu with high MC_mu max, then
Formula B's correction is too small -- pruning would be unsound.
"""

from __future__ import annotations

import itertools
import numpy as np


def step_autoconv_at_knot(heights: np.ndarray, h: float, k: int) -> float:
    """
    Exact value of (g*g)(x_k) where g is a step function with the given
    heights on bins of width h, x_k = -1/2 + k*h.

    Formula:  (g*g)(x_k) = h * sum_{i+j = k-1} g_i * g_j
    """
    d = len(heights)
    total = 0.0
    for i in range(d):
        j = k - 1 - i
        if 0 <= j < d:
            total += heights[i] * heights[j]
    return h * total


def mass_conv(masses: np.ndarray, s: int) -> float:
    """MC_mu[s] = sum_{i+j = s} mu_i mu_j."""
    d = len(masses)
    total = 0.0
    for i in range(d):
        j = s - i
        if 0 <= j < d:
            total += masses[i] * masses[j]
    return total


def test_value(masses: np.ndarray, n: int, ell: int, s: int) -> float:
    """TV at window (ell, s):  (4n/ell) * sum_{k=s..s+ell-2} MC_mu[k]."""
    return (4.0 * n / ell) * sum(mass_conv(masses, k) for k in range(s, s + ell - 1))


def main() -> None:
    # ---- problem setup -----------------------------------------------------
    d = 4
    n = d // 2          # = 2   (bin layout: 2n = d bins of width h = 1/(4n))
    m = 10
    h = 1.0 / (4 * n)    # = 0.125
    c = np.array([2, 1, 1, 6], dtype=int)
    assert c.sum() == m

    w = c / m            # discrete masses on bins
    a = w / h            # step-function heights = 4n * w

    # window of interest (per proof doc Section 5.1)
    ell, s = 2, 6

    # contributing bins for window (ell, s):
    # the knots in the window are x_{s+1}, ..., x_{s+ell-1}
    # equivalently we use MC_w[k] for k = s, ..., s+ell-2
    # contributing pairs (i,j): 0 <= i,j < d, i+j in [s, s+ell-2]
    contrib_bins = set()
    for k in range(s, s + ell - 1):
        for i in range(d):
            j = k - i
            if 0 <= j < d:
                contrib_bins.add(i)
                contrib_bins.add(j)
    B = sorted(contrib_bins)
    W_int = int(c[B].sum())            # integer mass over contributing bins
    W = W_int / m                      # physical mass

    # ---- discrete TV at this composition / window --------------------------
    TV_w = test_value(w, n, ell, s)

    # also compute (g_c * g_c)(x_k) at the relevant knots
    knot_vals = {k: step_autoconv_at_knot(a, h, k) for k in range(4 * n + 1)}
    peak_knot = max(knot_vals, key=knot_vals.get)
    peak_val = knot_vals[peak_knot]

    # ---- Formula B's claimed correction ------------------------------------
    formulaB_correction = 1.0 / m**2 + 2.0 * W / m

    # ---- worst-case mass-space error under cumulative-floor perturbation ---
    #
    # Cumulative-floor structure: there exist mass shifts delta_i = w_i - mu_i
    # with cumulative sums sigma_k = sum_{j<k} delta_j in (-1/m, 0].
    # That means each delta_i lies in (-1/m, 1/m) but the *cumulative* sum
    # is bounded.  We enumerate cumulative shifts on a fine grid in (-1/m, 0]
    # and recover delta_i = sigma_{i+1} - sigma_i (with sigma_0 = sigma_d = 0).
    #
    # For each such mu we measure both:
    #   - per-window TV gap  : |TV(w) - TV(mu)|
    #   - the max MC_mu peak : max_s MC_mu[s] (relevant for pruning soundness)
    grid = np.linspace(-1.0 / m + 1e-9, 0.0, 41)
    best_gap = 0.0
    best_mu = None
    best_peak_mu = 0.0
    for sigmas in itertools.product(grid, repeat=d - 1):
        sigma = (0.0,) + tuple(sigmas) + (0.0,)
        deltas = np.array([sigma[i + 1] - sigma[i] for i in range(d)])
        mu = w - deltas
        if (mu < -1e-12).any():
            continue
        TV_mu = test_value(mu, n, ell, s)
        gap = abs(TV_w - TV_mu)
        if gap > best_gap:
            best_gap = gap
            best_mu = mu.copy()
            best_peak_mu = 4 * n * max(mass_conv(mu, k) for k in range(2 * d - 1))

    # ---- sharper threshold suggested by Theorem 3 (factor 4n/ell) ----------
    sharp_correction = (4 * n / ell) * formulaB_correction

    # ---- report -----------------------------------------------------------
    print("=" * 70)
    print("Formula B counterexample verification (proof Section 5.1)")
    print("=" * 70)
    print(f"  d={d}, n={n}, m={m}, h={h:.6f}")
    print(f"  composition c = {tuple(c.tolist())},  sum = {c.sum()}")
    print(f"  bin masses  w = {w.tolist()}")
    print(f"  step heights a = {a.tolist()}    (a_i = 4n * w_i)")
    print(f"  window (ell, s) = ({ell}, {s})")
    print(f"  contributing bins B = {B}")
    print(f"  W_int = {W_int}    W = {W:.4f}")
    print()
    print("Knot values of (g_c * g_c)(x_k):")
    for k, v in knot_vals.items():
        marker = "  <-- peak" if k == peak_knot else ""
        in_win = "  [in window]" if s <= k - 1 < s + ell - 1 else ""
        print(f"    x_{k:>2}: {v:.6f}{in_win}{marker}")
    print()
    print(f"  TV(c; ell={ell}, s={s})            = {TV_w:.6f}")
    print(f"  ||g_c * g_c||_inf (over knots)  = {peak_val:.6f}")
    print()
    print(f"  Formula B correction (1/m^2 + 2W/m) = {formulaB_correction:.6f}")
    print(f"  Sharper Theorem 3 correction        = {sharp_correction:.6f}")
    print(f"      ratio (4n/ell)                  = {4 * n / ell:.4f}")
    print()
    print("Worst-case admissible mass perturbation mu (cumulative-floor):")
    print(f"  best mu found              = {best_mu.tolist()}")
    print(f"  per-window TV gap |TV_w - TV_mu| = {best_gap:.6f}")
    print(f"  ||f_step(mu) * f_step(mu)||_inf  = {best_peak_mu:.6f}")
    print()

    # Counterexample: choose c_target slightly below TV_w so Formula B prunes,
    # but the worst-case continuous TV at the same window dips below c_target.
    c_target = TV_w - formulaB_correction - 1e-6
    formulaB_threshold = c_target + formulaB_correction
    sharp_threshold    = c_target + sharp_correction
    print(f"  Pick c_target = TV_w - FormulaB_correction - eps = {c_target:.6f}")
    print(f"  -> Formula B threshold = {formulaB_threshold:.6f}")
    print(f"     TV(c) = {TV_w:.6f} > Formula B threshold  =>  Formula B PRUNES")
    print(f"  -> Sharper  threshold  = {sharp_threshold:.6f}")
    if TV_w > sharp_threshold:
        print("     TV(c) > sharp threshold  =>  Sharper bound also prunes")
    else:
        print("     TV(c) < sharp threshold  =>  Sharper bound does NOT prune (safe)")
    print()
    worst_TV_mu = TV_w - best_gap
    print(f"  But worst-case continuous TV(mu; ell, s) = {worst_TV_mu:.6f}")
    if worst_TV_mu < c_target:
        print("     => continuous TV at this same window can lie BELOW c_target")
        print("        while Formula B still prunes c.  Per-window UNSOUNDNESS.")
    print()
    print("=" * 70)
    print("Quantitative summary (matches proof doc):")
    print(f"  Actual per-window error       : {best_gap:.4f}")
    print(f"  Formula B's claimed correction: {formulaB_correction:.4f}")
    print(f"  ratio error / correction      : {best_gap / formulaB_correction:.3f}")
    print(f"  predicted ratio 4n/ell        : {4 * n / ell:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
