"""Library of admissible f's with rigorous Fourier-moment computation.

An admissible f satisfies:
    f : R -> R_{>=0}, supp(f) subset [-1/4, 1/4], int f = 1.

Each entry below exposes:
    name : str
    moments(N, prec_bits) -> list[arb] of length 2*N, giving (a_1, b_1, ..., a_N, b_N)

For library f's with closed-form Fourier integrals, moments are arb balls with
tight enclosure.  For step-function approximations of smoother f's, moments
are computed by summing over exact rational pieces + a bounded residual.

These f's serve as admissibility witnesses: every filter must return
ACCEPT (or at worst UNCLEAR, not REJECT) for each library f.  A filter
that rejects any library f is mathematically invalid and must be removed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from flint import arb, acb, fmpq, ctx


@dataclass
class AdmissibleF:
    name: str
    moments_fn: Callable[[int, int], list[arb]]    # (N, prec_bits) -> [a_1, b_1, ...]
    description: str = ""

    def moments(self, N: int, prec_bits: int = 256) -> list[arb]:
        return self.moments_fn(N, prec_bits)


# =============================================================================
#  F1: Uniform on [-1/4, 1/4]
# =============================================================================
#  f(x) = 2 * 1_{[-1/4, 1/4]}(x),  int f = 1.
#  hat f(j) = 2 int_{-1/4}^{1/4} exp(-2 pi i j x) dx = 2 sin(pi j / 2) / (pi j).
#  => a_j = 2 sin(pi j / 2) / (pi j),  b_j = 0.
#  At j = 1:  2/pi.   j = 2: 0.   j = 3: -2/(3 pi).   j = 4: 0.
# =============================================================================

def _moments_uniform(N: int, prec_bits: int) -> list[arb]:
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        ab = []
        for j in range(1, N + 1):
            arg = arb.pi() * arb(j) / arb(2)   # pi j / 2
            a_j = arb(2) * arg.sin() / (arb.pi() * arb(j))
            ab.append(a_j)
            ab.append(arb(0))
        return ab
    finally:
        ctx.prec = old


# =============================================================================
#  F2: Triangular peaked at 0 on [-1/4, 1/4]
# =============================================================================
#  f(x) = 8 (1 - 4|x|) for |x| <= 1/4, else 0.
#    int f = 2 * int_0^{1/4} 8(1 - 4x) dx = 2 * [8x - 16 x^2]_0^{1/4}
#          = 2 * (2 - 1) = 2. Wait that's 2, we need 1.
#  Correct normalisation: f(x) = 16 (1/4 - |x|) * (something).
#  int_{-1/4}^{1/4} (1/4 - |x|) dx = 2 * int_0^{1/4} (1/4 - x) dx
#                                   = 2 * [x/4 - x^2/2]_0^{1/4}
#                                   = 2 * (1/16 - 1/32) = 2/32 = 1/16.
#  So f(x) = 16 * (1/4 - |x|) has int = 1.  
#
#  hat f(j) = 16 int_{-1/4}^{1/4} (1/4 - |x|) exp(-2 pi i j x) dx
#           = 32 int_0^{1/4} (1/4 - x) cos(2 pi j x) dx  (by symmetry)
#  Let u = 2 pi j, substitute y = u x:  x = y/u, dx = dy/u
#    = 32 int_0^{u/4} (1/4 - y/u) cos(y) (dy/u)
#    = (32/u) int_0^{u/4} (1/4) cos(y) dy  -  (32/u^2) int_0^{u/4} y cos(y) dy
#    = (8/u) [sin(u/4)]  -  (32/u^2) [cos(u/4) + (u/4) sin(u/4) - 1]
#  Simpler closed form: use int_0^a (a - x) cos(u x) dx = (1 - cos(a u))/u^2:
#   wait -- let me redo.   int_0^a (a - x) cos(u x) dx = a * sin(au)/u - (integral x cos(u x))
#         int_0^a x cos(ux) dx = [x sin(ux)/u]_0^a - int sin(ux)/u dx = a sin(au)/u + cos(au)/u^2 - 1/u^2
#   Therefore int_0^a (a-x) cos(ux) dx = a sin(au)/u - a sin(au)/u - cos(au)/u^2 + 1/u^2
#                                        = (1 - cos(au)) / u^2.
#   So hat f(j) = 32 * (1 - cos(2 pi j / 4)) / (2 pi j)^2
#               = 32 * (1 - cos(pi j / 2)) / (4 pi^2 j^2)
#               = 8 (1 - cos(pi j / 2)) / (pi^2 j^2).
#
#   j=1: 8 (1 - 0) / pi^2 = 8 / pi^2 ≈ 0.8106.
#   j=2: 8 (1 - (-1)) / (4 pi^2) = 4 / pi^2 ≈ 0.4053.
#   j=3: 8 (1 - 0) / (9 pi^2) = 8 / (9 pi^2) ≈ 0.0901.
#   j=4: 8 (1 - 1) / (16 pi^2) = 0.
#   All real & non-negative => b_j = 0.
# =============================================================================

def _moments_triangular(N: int, prec_bits: int) -> list[arb]:
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        ab = []
        pi2 = arb.pi() * arb.pi()
        for j in range(1, N + 1):
            arg = arb.pi() * arb(j) / arb(2)
            one_minus_cos = arb(1) - arg.cos()
            a_j = arb(8) * one_minus_cos / (pi2 * arb(j) * arb(j))
            ab.append(a_j)
            ab.append(arb(0))
        return ab
    finally:
        ctx.prec = old


# =============================================================================
#  F3: Indicator of [-alpha, alpha], alpha in (0, 1/4]
# =============================================================================
#  f(x) = (1/(2 alpha)) * 1_{[-alpha, alpha]}(x),  int f = 1.
#  hat f(j) = (1/(2 alpha)) * 2 sin(2 pi j alpha) / (2 pi j)
#           = sin(2 pi j alpha) / (2 pi j alpha).
#  b_j = 0 (real f, even support).
# =============================================================================

def _moments_indicator_factory(alpha: fmpq) -> Callable[[int, int], list[arb]]:
    if not (0 < alpha <= fmpq(1, 4)):
        raise ValueError(f"indicator alpha must be in (0, 1/4]; got {alpha}")

    def fn(N: int, prec_bits: int) -> list[arb]:
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            ab = []
            two_pi_alpha = arb(2) * arb.pi() * arb(alpha)
            for j in range(1, N + 1):
                arg = two_pi_alpha * arb(j)
                sinc = arg.sin() / arg
                a_j = sinc
                ab.append(a_j)
                ab.append(arb(0))
            return ab
        finally:
            ctx.prec = old

    return fn


# =============================================================================
#  F4: Shifted indicator of [c - alpha, c + alpha] (non-symmetric support)
# =============================================================================
#  f(x) = (1/(2 alpha)) * 1_{[c-alpha, c+alpha]}(x),  int f = 1, b_j != 0.
#  hat f(j) = (1/(2 alpha)) int_{c-alpha}^{c+alpha} exp(-2 pi i j x) dx
#           = exp(-2 pi i j c) * sin(2 pi j alpha) / (2 pi j alpha).
#  Support constraint: [c - alpha, c + alpha] ⊆ [-1/4, 1/4]
#    i.e., |c| + alpha <= 1/4.
# =============================================================================

def _moments_shifted_indicator_factory(
    center: fmpq, alpha: fmpq
) -> Callable[[int, int], list[arb]]:
    if not (abs(center) + alpha <= fmpq(1, 4)):
        raise ValueError(
            f"support out of [-1/4, 1/4]: |center|+alpha = {abs(center) + alpha}"
        )
    if alpha <= 0:
        raise ValueError(f"alpha must be positive; got {alpha}")

    def fn(N: int, prec_bits: int) -> list[arb]:
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            ab = []
            two_pi_alpha = arb(2) * arb.pi() * arb(alpha)
            two_pi_c = arb(2) * arb.pi() * arb(center)
            for j in range(1, N + 1):
                sinc_arg = two_pi_alpha * arb(j)
                sinc = sinc_arg.sin() / sinc_arg
                shift = two_pi_c * arb(j)
                a_j = sinc * shift.cos()
                b_j = -sinc * shift.sin()    # hat f(j) = e^{-2 pi i j c} * sinc(...)
                ab.append(a_j)
                ab.append(b_j)
            return ab
        finally:
            ctx.prec = old

    return fn


# =============================================================================
#  F5: Convex combination (mixture) of two admissible f's
# =============================================================================
#  If f_1, f_2 are admissible and t in [0, 1] then f = t f_1 + (1-t) f_2 is
#  also admissible (f >= 0 pointwise, int f = 1, support as big as union but
#  still inside [-1/4, 1/4]).  Moments are linear: hat f(j) = t hat f_1(j) + (1-t) hat f_2(j).
# =============================================================================

def _moments_mixture_factory(
    t: fmpq,
    f1_moments: Callable[[int, int], list[arb]],
    f2_moments: Callable[[int, int], list[arb]],
) -> Callable[[int, int], list[arb]]:
    if not (0 <= t <= 1):
        raise ValueError(f"mixing weight must be in [0,1]; got {t}")

    def fn(N: int, prec_bits: int) -> list[arb]:
        m1 = f1_moments(N, prec_bits)
        m2 = f2_moments(N, prec_bits)
        t_arb = arb(t)
        one_minus_t = arb(1) - t_arb
        return [t_arb * a + one_minus_t * b for a, b in zip(m1, m2)]

    return fn


# =============================================================================
#  Public library
# =============================================================================

def build_library() -> list[AdmissibleF]:
    """Return a library of admissible f's for filter-validity testing."""
    lib = [
        AdmissibleF(
            "uniform_[-1/4,1/4]",
            _moments_uniform,
            "f = 2 * 1_{[-1/4,1/4]}",
        ),
        AdmissibleF(
            "triangular_peak_0",
            _moments_triangular,
            "f(x) = 16 (1/4 - |x|)",
        ),
    ]
    for alpha in [fmpq(1, 20), fmpq(1, 10), fmpq(3, 20), fmpq(1, 5), fmpq(1, 4)]:
        lib.append(AdmissibleF(
            f"indicator_[-{alpha}, {alpha}]",
            _moments_indicator_factory(alpha),
            f"f = (1/(2*{alpha})) * 1_{{[-{alpha}, {alpha}]}}",
        ))
    for (c, alpha) in [
        (fmpq(1, 10),  fmpq(1, 10)),
        (fmpq(-1, 10), fmpq(1, 10)),
        (fmpq(1, 20),  fmpq(1, 20)),
    ]:
        lib.append(AdmissibleF(
            f"shifted_indicator_c={c}_alpha={alpha}",
            _moments_shifted_indicator_factory(c, alpha),
            f"f = (1/(2*{alpha})) * 1_{{[{c}-{alpha}, {c}+{alpha}]}}",
        ))
    lib.append(AdmissibleF(
        "mixture_uniform_triangular_t=1/3",
        _moments_mixture_factory(fmpq(1, 3), _moments_uniform, _moments_triangular),
        "1/3 * uniform + 2/3 * triangular",
    ))
    lib.append(AdmissibleF(
        "mixture_narrow_indicators",
        _moments_mixture_factory(
            fmpq(1, 2),
            _moments_indicator_factory(fmpq(1, 20)),
            _moments_indicator_factory(fmpq(1, 5)),
        ),
        "1/2 * (alpha=1/20) + 1/2 * (alpha=1/5)  [both centred on 0]",
    ))
    return lib


__all__ = ["AdmissibleF", "build_library"]
