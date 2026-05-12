"""Bessel-ansatz convolution kernel K(t) for the Sidon autocorrelation constant.

Ansatz (Rechnitzer, rescaled to support [-1/4, 1/4]):
    phi_j(x) = (1 - 16 x^2)^{j - 1/2}  on [-1/4, 1/4], j = 1,...,P.
    f(x)     = sum_j a_j phi_j(x),   (f*f)(t) = a^T K(t) a.

Rigorous evaluation of K_{jm}(t) = (phi_j*phi_m)(t) via:
  (C) closed form at t=0: K_{jm}(0) = sqrt(pi) Gamma(j+m)/(4 Gamma(j+m+1/2))
  (B) Arb's certified adaptive integrator on
        int_{a(t)}^{b(t)} phi_j(x) phi_m(t-x) dx,
      a(t)=max(-1/4, t-1/4), b(t)=min(1/4, t+1/4).
  Zero for |t| >= 1/2 (support of f*f).

Indexing: j=1..P maps to array positions 0..P-1.  See
bessel_kernel_derivation.md for the Fourier/Poisson derivation of
hat(phi_j) and the Bessel/Lommel form (not used here: finite integral (B)
is equally closed-form and cleaner)."""

from __future__ import annotations

from typing import Sequence, Union

from flint import acb, arb, ctx, fmpq


Number = Union[fmpq, int, float, arb]


# ---------------------------------------------------------------------------
# Normalisation helpers (beta_j = int phi_j) -- exact gamma form
# ---------------------------------------------------------------------------

def _arb_beta_j(j: int) -> arb:
    """beta_j = sqrt(pi) * Gamma(j + 1/2) / (4 * Gamma(j+1)).

    Computed via the recurrence beta_j = (2j-1)/(2j) * beta_{j-1},
    beta_1 = pi/8.  Exact rational * pi, returned as an Arb ball.
    """
    if j < 1:
        raise ValueError("beta_j is only defined for j >= 1 in this module")
    val = arb.pi() / arb(8)  # beta_1
    for k in range(2, j + 1):
        val = val * arb(2 * k - 1) / arb(2 * k)
    return val


def normalisation_row(P: int) -> list[arb]:
    """Return [beta_1, ..., beta_P] as Arb balls.

    User constraint int f = 1 becomes  sum_j a_j * beta_j == 1.
    """
    return [_arb_beta_j(j) for j in range(1, P + 1)]


def _arb_K_at_zero(j: int, m: int) -> arb:
    """K_{jm}(0) = sqrt(pi) * Gamma(j+m) / (4 * Gamma(j+m+1/2)).

    Uses beta-function / gamma recurrence to stay rational times sqrt(pi)
    where possible.  Here j,m >= 1 so j+m >= 2 and everything is finite.
    """
    n = j + m  # integer >= 2
    # sqrt(pi) * Gamma(n) / (4 * Gamma(n + 1/2))
    #   = sqrt(pi) * (n-1)! / (4 * Gamma(n+1/2))
    # Gamma(n + 1/2) = (2n)! * sqrt(pi) / (4^n * n!)
    # So ratio = (n-1)! * 4^n * n! / (4 * (2n)!)
    num = arb(1)
    for k in range(1, n):  # (n-1)!
        num = num * arb(k)
    fn = arb(1)
    for k in range(1, n + 1):  # n!
        fn = fn * arb(k)
    f2n = arb(1)
    for k in range(1, 2 * n + 1):  # (2n)!
        f2n = f2n * arb(k)
    four_n = arb(4) ** n
    return num * four_n * fn / (arb(4) * f2n)


# ---------------------------------------------------------------------------
# Kernel entry K_{jm}(t) via formula (C) at t=0 or (B) via Arb integration
# ---------------------------------------------------------------------------

def _coerce_fmpq(t: Number) -> fmpq:
    if isinstance(t, fmpq):
        return t
    if isinstance(t, int):
        return fmpq(t)
    if isinstance(t, float):
        # convert float to nearest fmpq deterministically
        return fmpq(t.as_integer_ratio()[0], t.as_integer_ratio()[1])
    if isinstance(t, arb):
        # Must be an exact dyadic / rational; reject otherwise
        raise TypeError(
            "bessel_K_matrix expects an exact rational t; got arb."
        )
    raise TypeError(f"Unsupported t type: {type(t)}")


def _acb_integrand_factory(j: int, m: int, t_acb: acb):
    """Build an integrand x -> phi_j(x) * phi_m(t-x) as an acb callable."""
    # phi_k(y) = (1 - 16 y^2)^{k - 1/2}
    # For acb: fractional powers are handled via complex log; we stay on
    # the positive real axis so the principal branch is correct.
    one = acb(1)
    sixteen = acb(16)
    half = acb(fmpq(1, 2))

    def integrand(x, _ctx):
        # x is acb
        u = one - sixteen * x * x
        v = one - sixteen * (t_acb - x) * (t_acb - x)
        # Clamp tiny negative spillover from ball inflation at endpoints:
        # not needed because the adaptive integrator avoids crossing zero
        # when we pass exact rational endpoints.
        eu = acb(j) - half   # exponent of u
        ev = acb(m) - half   # exponent of v
        return u.pow(eu) * v.pow(ev)

    return integrand


def _K_entry(t_q: fmpq, j: int, m: int, prec_bits: int) -> arb:
    """Rigorous Arb ball for K_{jm}(t_q)."""
    # Outside support: zero.
    half = fmpq(1, 2)
    if t_q >= half or t_q <= -half:
        return arb(0)

    # Closed form at t=0
    if t_q == 0:
        return _arb_K_at_zero(j, m)

    # Endpoints of the convolution integral
    quarter = fmpq(1, 4)
    a_q = max(fmpq(-1, 4), t_q - quarter)
    b_q = min(fmpq(1, 4), t_q + quarter)
    if not (a_q < b_q):
        return arb(0)

    t_acb = acb(t_q)
    integrand = _acb_integrand_factory(j, m, t_acb)

    # Enforce working precision for the integrator
    old_prec = ctx.prec
    ctx.prec = prec_bits
    try:
        res = acb.integral(integrand, acb(a_q), acb(b_q))
    finally:
        ctx.prec = old_prec

    # Result must be real; drop tiny imaginary ball via .real
    return res.real


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bessel_K_matrix(t: Number, P: int, prec_bits: int = 384) -> list[list[arb]]:
    """Return the P x P symmetric matrix K_{jm}(t), j,m = 1..P.

    Entries are Arb balls rigorously enclosing the true value.
    """
    if P < 1:
        raise ValueError("P must be >= 1")
    t_q = _coerce_fmpq(t)

    old_prec = ctx.prec
    ctx.prec = prec_bits
    try:
        M: list[list[arb]] = [[arb(0)] * P for _ in range(P)]
        for r in range(P):
            j = r + 1
            for c in range(r, P):
                m = c + 1
                val = _K_entry(t_q, j, m, prec_bits)
                M[r][c] = val
                if c != r:
                    M[c][r] = val
    finally:
        ctx.prec = old_prec
    return M


def bilinear_ff_at(
    t: Number, a: Sequence[float], prec_bits: int = 384
) -> arb:
    """Return a^T K(t) a as a rigorous Arb ball.

    `a` is a sequence of length P = len(a); a[r] corresponds to j = r+1.
    Each a_j is embedded in Arb exactly (floats via their binary value,
    ints/fmpq exactly).
    """
    P = len(a)
    M = bessel_K_matrix(t, P, prec_bits=prec_bits)

    old_prec = ctx.prec
    ctx.prec = prec_bits
    try:
        # Embed a into Arb
        a_arb: list[arb] = []
        for x in a:
            if isinstance(x, arb):
                a_arb.append(x)
            elif isinstance(x, fmpq):
                a_arb.append(arb(x.p) / arb(x.q))
            elif isinstance(x, int):
                a_arb.append(arb(x))
            elif isinstance(x, float):
                a_arb.append(arb(x))
            else:
                raise TypeError(f"Unsupported coefficient type: {type(x)}")

        total = arb(0)
        for r in range(P):
            for c in range(P):
                total = total + a_arb[r] * M[r][c] * a_arb[c]
    finally:
        ctx.prec = old_prec
    return total


__all__ = [
    "bessel_K_matrix",
    "bilinear_ff_at",
    "normalisation_row",
]
