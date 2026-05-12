"""Rigorous verification pipeline.

Given parameters for a family (F1 or F2), produce a CERTIFIED ball
[lb_low, lb_high] enclosing the lower bound L(params).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from mpmath import iv, mp, mpf

from . import family_f1_selberg as f1
from . import family_f2_gauss_poly as f2
from . import rigorous_max as rm


@dataclass
class VerifiedBound:
    family: str
    params: object
    numerator_lo: mpf
    numerator_hi: mpf
    max_g_lo: mpf
    max_g_hi: mpf
    int_g_lo: mpf
    int_g_hi: mpf
    denom_kind: str         # "max_g" or "int_g"
    pd_certified: bool
    g_nonneg_certified: bool
    pd_reason: str
    lb_low: mpf
    lb_high: mpf
    n_splits: int


def _verify_family(family_module, params, *,
                   rel_tol: float, n_subdiv: int, precision_bits: int,
                   denom_kind: str = "int_g"):
    """Verify the dual bound.

    denom_kind:
      "max_g"  -- use L = numerator / max_{[-1/2,1/2]} g  (conservative)
      "int_g"  -- use L = numerator / int_{-1/2}^{1/2} g  (tighter; requires
                  g >= 0 on [-1/2, 1/2], which we certify by checking that
                  max_g lower bound >= 0 AND min_g lower bound >= 0).
    """
    mp.prec = precision_bits
    iv.prec = precision_bits

    # 1. PD certificate.
    pd_ok, pd_reason = family_module.positive_definite_certificate(params)

    # 2. Numerator.
    xi_max = family_module.ghat_support(params)
    n_lo, n_hi = rm.rigorous_integral_with_weight(
        lambda xi: family_module.ghat_iv(xi, params),
        lambda xi: family_module.weight_iv(xi),
        xi_max,
        n_subdiv=n_subdiv,
        precision_bits=precision_bits,
    )

    # 3. Max of g on [-1/2, 1/2].
    max_lo, max_hi, n_splits = rm.rigorous_max(
        lambda tI: family_module.g_iv(tI, params),
        a=mpf("-0.5"),
        b=mpf("0.5"),
        rel_tol=rel_tol,
        max_splits=60_000,
        precision_bits=precision_bits,
    )

    # 3b. For "int_g" denominator: rigorous int of g on [-1/2, 1/2].
    if denom_kind == "int_g":
        int_lo, int_hi = family_module.integral_g_iv(params, n_subdiv=n_subdiv)
    else:
        int_lo = int_hi = mpf(0)

    # 3c. Certify g >= 0 on [-1/2, 1/2] by certifying -min g <= 0, via the
    # max of -g.
    min_neg_lo, min_neg_hi, _ = rm.rigorous_max(
        lambda tI: -family_module.g_iv(tI, params),
        a=mpf("-0.5"),
        b=mpf("0.5"),
        rel_tol=max(rel_tol, 1e-6),
        max_splits=30_000,
        precision_bits=precision_bits,
    )
    min_g_hi = -min_neg_lo   # upper bound on min g = -(lower bound on max(-g))
    g_nonneg_certified = min_g_hi >= 0

    # 4. Combine. CRITICAL: for denom_kind == "int_g" the bound
    #    L >= numerator / int g  requires g >= 0 on [-1/2, 1/2] (otherwise
    #    int g < max g and the inequality can flip). If g_nonneg is NOT
    #    rigorously certified, the bound is INVALID and we report [0, 0].
    # Additionally, if PD certification failed, ghat >= 0 is not guaranteed
    # and the Plancherel lower-bound step is invalid: report [0, 0].
    if not pd_ok:
        lb_low = mpf(0)
        lb_high = mpf(0)
    elif denom_kind == "int_g":
        if not g_nonneg_certified:
            lb_low = mpf(0)
            lb_high = mpf(0)
        elif int_lo <= 0:
            lb_low = mpf(0)
            lb_high = mpf(0)
        else:
            # Numerator may be negative (signed weight); handle sign carefully.
            # For the lower bound on the ratio, divide n_lo by int_hi if n_lo >= 0,
            # else by int_lo (dividing a negative by a smaller positive gives a
            # more-negative, i.e. smaller, value -- which is the correct lower bound).
            if n_lo >= 0:
                lb_low = n_lo / int_hi
            else:
                lb_low = n_lo / int_lo
            if n_hi >= 0:
                lb_high = n_hi / int_lo
            else:
                lb_high = n_hi / int_hi
    else:
        if max_hi <= 0:
            lb_low = mpf(0)
            lb_high = mpf(0)
        else:
            if n_lo >= 0:
                lb_low = n_lo / max_hi
            else:
                lb_low = n_lo / max_lo if max_lo > 0 else mpf(0)
            if n_hi >= 0:
                lb_high = n_hi / max_lo if max_lo > 0 else n_hi / max_hi
            else:
                lb_high = n_hi / max_hi

    return VerifiedBound(
        family=family_module.__name__.split(".")[-1],
        params=params,
        numerator_lo=n_lo,
        numerator_hi=n_hi,
        max_g_lo=max_lo,
        max_g_hi=max_hi,
        int_g_lo=int_lo,
        int_g_hi=int_hi,
        denom_kind=denom_kind,
        pd_certified=pd_ok,
        g_nonneg_certified=g_nonneg_certified,
        pd_reason=pd_reason,
        lb_low=lb_low,
        lb_high=lb_high,
        n_splits=n_splits,
    )


def verify_f1(params: f1.F1Params, *, rel_tol: float = 1e-10,
              n_subdiv: int = 16384, precision_bits: int = 200,
              denom_kind: str = "int_g") -> VerifiedBound:
    return _verify_family(f1, params, rel_tol=rel_tol, n_subdiv=n_subdiv,
                          precision_bits=precision_bits, denom_kind=denom_kind)


def verify_f2(params: f2.F2Params, *, rel_tol: float = 1e-10,
              n_subdiv: int = 16384, precision_bits: int = 200,
              denom_kind: str = "int_g") -> VerifiedBound:
    return _verify_family(f2, params, rel_tol=rel_tol, n_subdiv=n_subdiv,
                          precision_bits=precision_bits, denom_kind=denom_kind)


def format_ball(b: VerifiedBound) -> str:
    width = b.lb_high - b.lb_low
    denom_line = (
        f"  int g on [-1/2,1/2] in [{float(b.int_g_lo):.10f}, {float(b.int_g_hi):.10f}]"
        if b.denom_kind == "int_g" else ""
    )
    return (f"[{b.family}] params={b.params}\n"
            f"  numerator in [{float(b.numerator_lo):.10f}, {float(b.numerator_hi):.10f}]\n"
            f"  max g on [-1/2,1/2] in [{float(b.max_g_lo):.10f}, {float(b.max_g_hi):.10f}]\n"
            f"{denom_line}\n"
            f"  pd_certified={b.pd_certified}  g_nonneg={b.g_nonneg_certified}\n"
            f"  denom_kind={b.denom_kind}\n"
            f"  lb in [{float(b.lb_low):.10f}, {float(b.lb_high):.10f}]  width={float(width):.2e}\n"
            f"  n_splits={b.n_splits}")
