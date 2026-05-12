"""Matolcsi-Vinuesa baseline parameters as exact rationals.

The 119 cosine coefficients of the cosine multiplier ``G`` used in the
original Matolcsi-Vinuesa proof of ``C_{1a} >= 1.27481`` are reproduced
verbatim from arXiv:0907.1379v2, Appendix.  Each coefficient is given in
the paper as an 8-mantissa-digit float; we treat the decimal string
literally as the rational ``p / 10^k``, with no rounding.

These coefficients are used by the verifier ``certify.py`` as the
baseline single-scale check and by the entry points in
``delsarte_dual.grid_bound.bisect`` for the reproduction of the
``M_cert = 1.27481`` single-scale lower bound.

The production multi-scale lower bound ``C_{1a} >= 1.292`` (the
Piterbarg-Bajaj-Vincent Bound) uses a distinct ``G`` re-optimised
against the multi-scale kernel; those coefficients are produced at
runtime by the QP solver and stored in the emitted JSON certificate.
"""
from __future__ import annotations

from flint import fmpq


_MV_DECIMALS = [
    "+2.16620392",       "-1.87775750",       "+1.05828868",       "-7.29790538e-01",
    "+4.28008515e-01",   "+2.17832838e-01",   "-2.70415201e-01",   "+2.72834790e-02",
    "-1.91721888e-01",   "+5.51862060e-02",   "+3.21662512e-01",   "-1.64478392e-01",
    "+3.95478603e-02",   "-2.05402785e-01",   "-1.33758316e-02",   "+2.31873221e-01",
    "-4.37967118e-02",   "+6.12456374e-02",   "-1.57361919e-01",   "-7.78036253e-02",
    "+1.38714392e-01",   "-1.45201483e-04",   "+9.16539824e-02",   "-8.34020840e-02",
    "-1.01919986e-01",   "+5.94915025e-02",   "-1.19336618e-02",   "+1.02155366e-01",
    "-1.45929982e-02",   "-7.95205457e-02",   "+5.59733152e-03",   "-3.58987179e-02",
    "+7.16132260e-02",   "+4.15425065e-02",   "-4.89180454e-02",   "+1.65425755e-03",
    "-6.48251747e-02",   "+3.45951253e-02",   "+5.32122058e-02",   "-1.28435276e-02",
    "+1.48814403e-02",   "-6.49404547e-02",   "-6.01344770e-03",   "+4.33784473e-02",
    "-2.53362778e-04",   "+3.81674519e-02",   "-4.83816002e-02",   "-2.53878079e-02",
    "+1.96933442e-02",   "-3.04861682e-03",   "+4.79203471e-02",   "-2.00930265e-02",
    "-2.73895519e-02",   "+3.30183589e-03",   "-1.67380508e-02",   "+4.23917582e-02",
    "+3.64690190e-03",   "-1.79916104e-02",   "+7.31661649e-05",   "-2.99875575e-02",
    "+2.71842526e-02",   "+1.41806855e-02",   "-6.01781076e-03",   "+5.86806100e-03",
    "-3.32350597e-02",   "+9.23347466e-03",   "+1.47071722e-02",   "-7.42858080e-04",
    "+1.63414270e-02",   "-2.87265671e-02",   "-1.64287280e-03",   "+8.02601605e-03",
    "-7.62613027e-04",   "+2.18735533e-02",   "-1.78816282e-02",   "-6.58341101e-03",
    "+2.67706547e-03",   "-6.25261247e-03",   "+2.24942824e-02",   "-8.10756022e-03",
    "-5.68160823e-03",   "+7.01871209e-05",   "-1.15294332e-02",   "+1.83608944e-02",
    "-1.20567880e-03",   "-3.13147456e-03",   "+1.39083675e-03",   "-1.49312478e-02",
    "+1.32106694e-02",   "+1.73474188e-03",   "-8.53469045e-04",   "+4.03211203e-03",
    "-1.55352991e-02",   "+8.74711543e-03",   "+1.93998895e-03",   "-2.71357322e-05",
    "+6.13179585e-03",   "-1.41983972e-02",   "+5.84710551e-03",   "+9.22578333e-04",
    "-2.16583469e-04",   "+7.07919829e-03",   "-1.18488582e-02",   "+4.39698322e-03",
    "-8.91346785e-05",   "-3.42086367e-04",   "+6.46355636e-03",   "-8.87555371e-03",
    "+3.56799654e-03",   "-4.97335419e-04",   "-8.04560326e-04",   "+5.55076717e-03",
    "-7.13560569e-03",   "+4.53679038e-03",   "-3.33261516e-03",   "+2.35463427e-03",
    "+2.04023789e-04",   "-1.27746711e-03",   "+1.81247830e-04",
]
assert len(_MV_DECIMALS) == 119, (
    f"Matolcsi-Vinuesa has 119 coefficients; got {len(_MV_DECIMALS)}"
)


def _decimal_str_to_fmpq(s: str) -> fmpq:
    """Parse a signed decimal string (optional ``eNN`` exponent) to exact fmpq.

    Handles forms like ``"+2.16620392"``, ``"-7.29790538e-01"``, ``"+1e2"``,
    ``"-1"``, ``"3.14"``.  No rounding.
    """
    s = s.strip()
    sign = 1
    if s.startswith("+"):
        s = s[1:]
    elif s.startswith("-"):
        sign = -1
        s = s[1:]

    if "e" in s or "E" in s:
        mantissa, exp_part = s.replace("E", "e").split("e", 1)
        exp = int(exp_part)
    else:
        mantissa, exp = s, 0

    if "." in mantissa:
        int_part, frac_part = mantissa.split(".", 1)
    else:
        int_part, frac_part = mantissa, ""
    if int_part == "":
        int_part = "0"
    digits_after_point = len(frac_part)
    mantissa_int = int(int_part + frac_part) if (int_part + frac_part) else 0

    net_exp = exp - digits_after_point
    if net_exp >= 0:
        return fmpq(sign * mantissa_int * (10 ** net_exp), 1)
    return fmpq(sign * mantissa_int, 10 ** (-net_exp))


def mv_coeffs_fmpq() -> list:
    """Return the 119 Matolcsi-Vinuesa cosine coefficients as ``fmpq``.

    The convention is 1-based in the paper and 0-based in the list:
    ``mv_coeffs_fmpq()[j - 1]`` is the coefficient ``a_j``.
    """
    return [_decimal_str_to_fmpq(s) for s in _MV_DECIMALS]


#: Half-width of the single-scale arcsine kernel.
MV_DELTA = fmpq(138, 1000)

#: Period parameter ``u = 1/2 + delta``.
MV_U = fmpq(638, 1000)

#: Number of cosine modes in the single-scale baseline.
MV_N = 119

#: Martin-O'Bryant surrogate ``||K||_2^2 = 0.5747 / delta``
#: (arXiv:0807.5121, Lemma 3.2).  Used by the single-scale arcsine
#: baseline; the production multi-scale pipeline computes ``K_2`` directly
#: from cross-Bessel integrals (see
#: :class:`delsarte_dual.grid_bound_alt_kernel.kernels.MultiScaleArcsineKernel`).
MV_K2_NUMERATOR = fmpq(5747, 10000)


__all__ = [
    "mv_coeffs_fmpq",
    "MV_DELTA",
    "MV_U",
    "MV_N",
    "MV_K2_NUMERATOR",
]
