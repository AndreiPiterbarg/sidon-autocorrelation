"""Tests for certified_lasserre.bessel_kernel. Run as a script or via pytest."""

from __future__ import annotations

import sys
import traceback

from flint import arb, ctx, fmpq

from certified_lasserre.bessel_kernel import (
    bessel_K_matrix,
    bilinear_ff_at,
    normalisation_row,
)


PREC = 384
TOL = 1e-12


def test_K_symmetric() -> None:
    """K(t) = K(t)^T at t = 0, 1/4, 1/2, for P = 4."""
    P = 4
    for t in (fmpq(0), fmpq(1, 4), fmpq(1, 2)):
        M = bessel_K_matrix(t, P, prec_bits=PREC)
        for r in range(P):
            for c in range(P):
                diff = M[r][c] - M[c][r]
                assert abs(float(diff)) == 0.0, (
                    f"asymmetry at t={t}, (r,c)=({r},{c})"
                )


def test_K_matches_direct_integral() -> None:
    """P=3, a=(1, 0.3, -0.1): a^T K(0) a matches mpmath quadrature to 12 digits."""
    from mpmath import mp, mpf, quad
    mp.dps = 30
    a = [1.0, 0.3, -0.1]

    def phi(j, x):
        if x < -mpf(1) / 4 or x > mpf(1) / 4:
            return mpf(0)
        return (1 - 16 * x * x) ** (j - mpf(1) / 2)

    def f(x):
        return sum(mpf(aj) * phi(jj, x) for jj, aj in enumerate(a, start=1))

    direct = quad(lambda x: f(x) * f(-x), [-mpf(1) / 4, mpf(1) / 4])
    bilinear = bilinear_ff_at(fmpq(0), a, prec_bits=PREC)
    assert abs(float(direct) - float(bilinear)) < TOL, (
        f"mismatch: mpmath={direct}, arb={float(bilinear)}"
    )


def test_normalisation_ints() -> None:
    """beta_j = sqrt(pi) Gamma(j+1/2) / (4 Gamma(j+1)) for j = 1..5."""
    from mpmath import mp, mpf, sqrt, pi, gamma
    mp.dps = 40
    for idx, b in enumerate(normalisation_row(5)):
        j = idx + 1
        expected = sqrt(pi) * gamma(j + mpf(1) / 2) / (4 * gamma(j + 1))
        assert abs(float(b) - float(expected)) < TOL, f"beta_{j} mismatch"


def test_rechnitzer_reproduction() -> None:
    """With arbitrary Rechnitzer-like coefficients, bilinear form is positive
    at t=0 and stable across precisions."""
    a = [1.0, 0.7, 0.35, 0.1]
    v256 = bilinear_ff_at(fmpq(0), a, prec_bits=256)
    v384 = bilinear_ff_at(fmpq(0), a, prec_bits=384)
    assert abs(float(v256) - float(v384)) < 1e-60
    assert float(v384) > 0


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _run_one(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
        return True
    except AssertionError as e:
        print(f"FAIL  {name}: {e}")
        return False
    except Exception:
        print(f"ERROR {name}:")
        traceback.print_exc()
        return False


def main() -> int:
    ctx.prec = PREC
    tests = [
        ("test_K_symmetric", test_K_symmetric),
        ("test_K_matches_direct_integral", test_K_matches_direct_integral),
        ("test_normalisation_ints", test_normalisation_ints),
        ("test_rechnitzer_reproduction", test_rechnitzer_reproduction),
    ]
    ok = all(_run_one(n, f) for n, f in tests)

    # Deliverable report
    print()
    print("=== Deliverable numbers ===")
    a = [1.0, 0.3, -0.1]
    v = bilinear_ff_at(fmpq(0), a, prec_bits=384)
    print(f"a^T K(0) a  with a=(1, 0.3, -0.1), prec=384 bits:")
    print(f"  {v}")

    K = bessel_K_matrix(fmpq(1, 4), 3, prec_bits=384)
    print(f"K_{{1,1}}(1/4) (i.e. phi_1 * phi_1 at t=1/4), prec=384 bits:")
    print(f"  {K[0][0]}")
    # Also give K indexed from 0 in case the caller thinks j starts at 0
    print(
        "Note: this module indexes j=1..P; 'K_{0,0}' in the prompt's 0-based"
        " notation is our K_{1,1} since j=0 gives phi_0=(1-16x^2)^{-1/2}"
        " whose self-convolution is logarithmically divergent at t=0."
    )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
