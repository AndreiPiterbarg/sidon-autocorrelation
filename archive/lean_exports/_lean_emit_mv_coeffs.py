"""Emit the 119 MV coefficients as Lean rational literals (p/q : ℚ).

Used as a one-shot helper to generate the body of `mv_coeffs : List ℚ` in
`lean/Sidon/CohnElkies125.lean`.  Mirrors the decimal-to-fmpq parser in
`delsarte_dual/grid_bound/coeffs.py`.

Run:
    python _lean_emit_mv_coeffs.py
"""
from __future__ import annotations
from fractions import Fraction

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "delsarte_dual", "grid_bound"))
from coeffs import _MV_DECIMALS, _decimal_str_to_fmpq  # type: ignore


def to_fraction(s: str) -> Fraction:
    # Reuse the project's parser via fmpq -> int/int.
    q = _decimal_str_to_fmpq(s)
    return Fraction(int(q.p), int(q.q))


def main() -> None:
    fs = [to_fraction(s) for s in _MV_DECIMALS]
    out_path = os.path.join(HERE, "_lean_mv_coeffs_emitted.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        w = f.write
        w(f"-- {len(fs)} MV coefficients as exact rationals\n")
        w("def mv_coeffs : List Rat := [\n")
        for i, fr in enumerate(fs):
            p, q = fr.numerator, fr.denominator
            comma = "," if i + 1 < len(fs) else ""
            w(f"  (({p} : Rat) / {q}){comma}  -- a_{i+1}\n")
        w("]\n\n")

        sum_abs = sum(abs(fr) for fr in fs)
        sum_jabs = sum(Fraction(j + 1, 1) * abs(fr) for j, fr in enumerate(fs))
        w(f"-- sum_j |a_j|     ~ {float(sum_abs):.10f}\n")
        w(f"-- sum_j j * |a_j| ~ {float(sum_jabs):.10f}\n\n")
        w(f"-- sum_abs       p/q = {sum_abs.numerator}/{sum_abs.denominator}\n")
        w(f"-- sum_j_jabs    p/q = {sum_jabs.numerator}/{sum_jabs.denominator}\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
