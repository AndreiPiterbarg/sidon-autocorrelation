"""Project the R needed at d=64 to clear alpha >= 1.281.

Use the empirical convergence at d=8 to estimate the rate, then apply
to d=64 starting from the measured R=4 alpha.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def main():
    target = 1.281
    val_d = {4: 1.102, 8: 1.205, 12: 1.271, 16: 1.319, 32: 1.336,
             64: 1.384, 128: 1.420}

    # Measured (d, R, alpha) — Z/2, variable lambda
    runs = {
        4: [(4, 1.000), (6, 1.0476), (8, 1.0625), (10, 1.0667), (12, 1.0769)],
        8: [(4, 1.0667), (6, 1.1031), (8, 1.1310), (10, 1.1479),
            (12, 1.1587), (14, 1.1670), (16, 1.1733)],
        12: [(4, 1.0848), (6, 1.1332), (8, 1.1642)],
        16: [(4, 1.0994), (6, 1.1530)],
        64: [(4, 1.1383)],
    }

    print(f"Power-law fit: gap(R) = C * R^{{-a}}")
    print(f"{'d':>4} {'val_d':>8} {'a':>8} {'C':>10} {'pred R for cert>=1.281':>26}")

    fits = {}
    for d, data in runs.items():
        if len(data) < 3:
            continue
        R_arr = np.array([r for r, _ in data], dtype=float)
        a_arr = np.array([al for _, al in data], dtype=float)
        gap = val_d[d] - a_arr
        # Drop first point (often above asymptotic rate)
        if len(R_arr) > 4:
            R_fit = R_arr[1:]
            gap_fit = gap[1:]
        else:
            R_fit = R_arr
            gap_fit = gap
        # Linear fit log(gap) = log(C) - a*log(R)
        coeffs = np.polyfit(np.log(R_fit), np.log(gap_fit), 1)
        slope, intercept = coeffs
        a = -slope
        C = np.exp(intercept)
        fits[d] = (a, C)
        # R needed: gap(R) <= val_d - target
        eps = val_d[d] - target
        if eps <= 0:
            R_needed = float("inf")
            note = f"impossible (val({d})={val_d[d]} < {target})"
        else:
            R_needed = (C / eps) ** (1.0 / a)
            note = f"{R_needed:.2f}"
        print(f"{d:>4} {val_d[d]:>8.3f} {a:>8.3f} {C:>10.4f} {note:>26}")

    # Project d=64 using d=8 fit as anchor (since we only have R=4 at d=64)
    print("\n--- Projection for d=64 ---")
    a8, C8 = fits[8]
    a16, C16 = fits.get(16, (a8, C8))
    print(f"d=8 fit: a={a8:.3f}, C={C8:.4f}")
    print(f"d=16 fit: a={a16:.3f}, C={C16:.4f}")

    # Anchor d=64 fit using R=4 data point
    R64_anchor, a64_anchor = runs[64][0]
    gap64_R4 = val_d[64] - a64_anchor
    print(f"d=64 R=4: alpha={a64_anchor:.4f}, gap={gap64_R4:.4f}")
    # Assume same rate as d=8, so C_64 = gap_64_R4 * R^a8
    C_proj = gap64_R4 * (R64_anchor ** a8)
    print(f"Projected C(d=64) = {C_proj:.4f} assuming a={a8:.3f}")
    eps = val_d[64] - target
    R_needed = (C_proj / eps) ** (1.0 / a8)
    print(f"To clear cert >= {target} (need gap <= {eps:.3f}): R >= {R_needed:.1f}")

    # Also project for two stronger rates
    for a_try in [a8, a16, 1.5]:
        R_t = (C_proj / eps) ** (1.0 / a_try)
        print(f"  with rate a={a_try:.2f}: R >= {R_t:.1f}")

    # LP size at projected R
    print("\nLP size at d=64 (Z/2 d_eff=32) for various R:")
    from math import comb
    d_eff = 32
    print(f"{'R':>3} {'n_eq':>15} {'n_vars (approx)':>18}")
    for R in (4, 6, 8, 10, 12):
        n_eq = comb(d_eff + R, R)
        n_q = comb(d_eff + R - 1, R - 1)
        print(f"{R:>3} {n_eq:>15,} {n_eq + n_q + 4096 + 1:>18,}")


if __name__ == "__main__":
    main()
