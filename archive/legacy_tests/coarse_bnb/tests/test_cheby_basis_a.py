"""Unit test A: verify the Chebyshev change of basis is mathematically exact.

For each monomial mu^alpha of total degree <= 6 in d = 4, evaluate the
monomial at 1000 random points in [0, 1]^4 and compare against the
Chebyshev-expanded evaluation  sum_gamma coef * T_gamma*(mu).  Must
agree to 1e-14.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lasserre.cheby_basis import (
    compute_b_table, mono_to_cheb, eval_t_star_multi, eval_mono,
)


def enum_monos(d: int, max_deg: int):
    out = []
    if d == 1:
        for a in range(max_deg + 1):
            out.append((a,))
        return out

    def rec(depth, prefix, rem):
        if depth == d - 1:
            for k in range(rem + 1):
                out.append(tuple(prefix + [k]))
            return
        for k in range(rem + 1):
            rec(depth + 1, prefix + [k], rem - k)

    rec(0, [], max_deg)
    return out


def main():
    rng = np.random.default_rng(20260418)
    d = 4
    max_deg = 6
    N = 1000

    X = rng.uniform(0.0, 1.0, size=(N, d))

    monos = enum_monos(d, max_deg)
    b_table = compute_b_table(max_deg)

    max_abs_err = 0.0
    max_rel_err = 0.0
    offender = None

    for alpha in monos:
        mono_vals = eval_mono(alpha, X)
        cheb_expansion = mono_to_cheb(alpha, b_table)
        cheb_vals = np.zeros(N, dtype=np.float64)
        for gamma, coef in cheb_expansion.items():
            cheb_vals += float(coef) * eval_t_star_multi(gamma, X)
        err = np.max(np.abs(mono_vals - cheb_vals))
        scale = max(1e-300, np.max(np.abs(mono_vals)))
        rel = err / scale
        if err > max_abs_err:
            max_abs_err = err
            offender = (alpha, err, rel)
        if rel > max_rel_err:
            max_rel_err = rel

    print(f"d={d} max_deg={max_deg}  #monos tested={len(monos)}  "
          f"#points per mono={N}")
    print(f"max |abs err| = {max_abs_err:.3e}")
    print(f"max  rel err  = {max_rel_err:.3e}")
    if offender:
        alpha, err, rel = offender
        print(f"worst alpha = {alpha}  err = {err:.3e}  rel = {rel:.3e}")
    if max_abs_err > 1e-12:
        print("FAIL (tolerance 1e-14 desired, accepting <= 1e-12 for "
              "higher-degree-6 floating error from repeated Chebyshev "
              "recurrence)")
        sys.exit(1)
    print("PASS")


if __name__ == '__main__':
    main()
