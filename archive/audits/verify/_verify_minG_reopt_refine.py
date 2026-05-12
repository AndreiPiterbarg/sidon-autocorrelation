"""Refine min_G certification near the argmin cell (~ x = 0.246) using
many more cells + higher precision, to determine whether the re-opt G's
true min on [0, 1/4] is actually below 1 or whether the Taylor-B&B
enclosure slack is the only reason we saw lo < 1.
"""
from __future__ import annotations

from flint import fmpq, arb, ctx

from delsarte_dual.grid_bound.G_min import (
    min_G_lower_bound, G_enclosure_taylor, _eval_G_at_point,
)
import _K26_full_sweep_reopt as reopt


U_FMPQ = fmpq(638, 1000)
DENOM = 10 ** 8


def floats_to_fmpq_8digits(a_float_list):
    return [fmpq(int(round(float(v) * DENOM)), DENOM) for v in a_float_list]


def main():
    a_opt, S1, mG_num, status = reopt.solve_QP([0.138, 0.045], [0.85, 0.15])
    assert status == "ok"
    coeffs = floats_to_fmpq_8digits(a_opt)
    print(f"Numerical min_G on 5001-pt grid: {mG_num:.12f}")

    # 1) Global B&B with many more cells and higher precision
    for n_cells, prec in [(4096, 192), (16384, 256), (65536, 320)]:
        encl, center = min_G_lower_bound(coeffs, U_FMPQ,
                                         n_cells=n_cells, prec_bits=prec)
        lo = float(encl.lower()); up = float(encl.upper())
        print(f"  n_cells={n_cells:>6}  prec={prec:>3}  "
              f"min_G LB={lo:.12f}  UB={up:.12f}  "
              f"center={float(center.p)/float(center.q):.6f}")

    # 2) Direct point-eval (rigorous arb) near x = 0.246 to confirm where G is
    print()
    print("Point evaluations of G near x ~ 0.246 (rigorous arb):")
    ctx.prec = 320
    for x_f in [0.240, 0.243, 0.244, 0.245, 0.2455, 0.246, 0.2465, 0.247, 0.248, 0.249, 0.250]:
        x_q = fmpq(int(round(x_f * 10**6)), 10**6)
        g = _eval_G_at_point(coeffs, x_q, U_FMPQ)
        print(f"  x = {x_f:.6f}  G(x) in [{float(g.lower()):.10f}, {float(g.upper()):.10f}]")

    # 3) Numerical (cosines) point eval to ultra-fine grid to find true min
    print()
    import math
    import numpy as np
    a = np.asarray(a_opt, dtype=np.float64)
    js = np.arange(1, 120)
    xs = np.linspace(0.0, 0.25, 5_000_001)  # 5M-pt grid
    # G(x) = sum_j a_j cos(2 pi j x / U)
    G = np.cos(2 * np.pi * np.outer(xs, js) / 0.638) @ a
    print(f"Numerical min_G on 5M-pt grid (float64): {G.min():.12f} at x={xs[G.argmin()]:.7f}")


if __name__ == "__main__":
    main()
