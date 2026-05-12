"""Forbidden-region function ``Phi(M, y)`` for the MV master inequality.

Sign convention
---------------

    ``Phi(M, y) >= 0``   =>   ``(M, y)`` is admissible (master inequality holds).
    ``Phi(M, y) <  0``   =>   ``(M, y)`` is forbidden (master inequality violated).

The variable is ``y := z_1^2 = |hat f(1)|^2``.  In the writeup notation
(Section 2.2),

    ``Phi(M, y) = M + 1 + 2 y k_1
                 + sqrt((M - 1 - 2 y^2)_+) sqrt((K_2 - 1 - 2 k_1^2)_+)
                 - (2/u + a)``,

where

  * ``M``        -- the claim ``||f * f||_inf = M``,
  * ``y``        -- ``z_1^2`` in ``[0, mu(M)]``, ``mu(M) := M sin(pi/M)/pi``,
  * ``k_1``      -- ``hat K(1)``,
  * ``K_2``      -- ``||K||_{L^2(R)}^2``,
  * ``u``        -- the period parameter,
  * ``a``        -- the gain term ``(4/u) m_G^2 / S_1``.

Contrapositive of admissibility: if ``Phi(M, y) < 0`` for every ``y`` in
the admissible box, no admissible ``f`` exists at that ``M``, hence ``M``
is a rigorous lower bound on ``C_{1a}``.

All algebraic inputs (``delta``, ``u``, ``K_2 = p / delta``, the cosine
coefficients) are exact ``fmpq``; the Bessel evaluations are
``flint.arb`` interval enclosures.  The returned ``Phi`` is also an
``arb``; cell certification rejects a cell iff ``Phi.upper() < 0``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from flint import arb, fmpq, ctx

from .bessel import j0_pi_j_delta_over_u, k1_period_one
from .coeffs import mv_coeffs_fmpq, MV_DELTA, MV_U, MV_K2_NUMERATOR
from .G_min import min_G_lower_bound


def safe_sqrt(x: arb) -> arb:
    """Rigorous arb enclosure of ``sqrt`` on the non-negative part of ``x``.

    Behaviour by cases:

      * ``x.upper() < 0``  -- raise ``ValueError`` (nothing to take sqrt of).
      * ``x.lower() >= 0`` -- return the exact ``x.sqrt()``.
      * otherwise          -- return the hull ``[0, sqrt(x.upper())]``,
                              a conservative superset of
                              ``{ sqrt(v) : v in x and v >= 0 }``.
    """
    x_up = x.upper()
    if x_up < 0:
        raise ValueError(
            f"safe_sqrt: x.upper() = {x_up} < 0; no non-negative sub-interval"
        )
    if x.lower() >= 0:
        return x.sqrt()
    return arb(0).union(x_up.sqrt())


@dataclass(frozen=True)
class PhiParams:
    """Compiled rigorous inputs to ``Phi(M, y)``.

    All fields are exact ``fmpq`` or rigorous ``arb`` enclosures.
    Construct with :meth:`from_mv` for the single-scale arcsine baseline
    or with the dataclass constructor for the multi-scale production
    pipeline (see
    :func:`delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel.compile_phi_params_for_kernel`).
    """

    delta:    fmpq
    u:        fmpq
    K2:       arb
    k1:       arb
    gain_a:   arb
    min_G:    arb
    S1:       arb
    n_coeffs: int
    min_G_center: fmpq

    @classmethod
    def from_mv(
        cls,
        delta: fmpq = MV_DELTA,
        u: fmpq = MV_U,
        coeffs: Sequence[fmpq] = None,
        K2_times_delta: fmpq = MV_K2_NUMERATOR,
        n_cells_min_G: int = 8192,
        prec_bits: int = 256,
    ) -> "PhiParams":
        """Compile Phi inputs from the single-scale Matolcsi-Vinuesa parameters.

        Recovers the baseline ``C_{1a} >= 1.27481`` certificate when run
        with the published 119-coefficient ``G``.
        """
        if coeffs is None:
            coeffs = mv_coeffs_fmpq()
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            k1_arb = k1_period_one(delta, prec_bits=prec_bits)
            K2_arb = arb(K2_times_delta) / arb(delta)
            S1 = arb(0)
            for j, a_j in enumerate(coeffs, start=1):
                j0 = j0_pi_j_delta_over_u(j, delta, u, prec_bits=prec_bits)
                S1 = S1 + (arb(a_j) * arb(a_j)) / (j0 * j0)
            min_G_encl, min_G_center = min_G_lower_bound(
                coeffs, u, n_cells=n_cells_min_G, prec_bits=prec_bits
            )
            min_G_cert_arb = min_G_encl.lower()
            if min_G_cert_arb.upper() <= 0:
                raise ValueError(
                    f"min G certified lower bound is non-positive: "
                    f"{min_G_cert_arb}"
                )
            gain_a = (
                (arb(4) / arb(u)) * (min_G_cert_arb * min_G_cert_arb) / S1
            )
            return cls(
                delta=delta,
                u=u,
                K2=K2_arb,
                k1=k1_arb,
                gain_a=gain_a,
                min_G=min_G_cert_arb,
                S1=S1,
                n_coeffs=len(coeffs),
                min_G_center=min_G_center,
            )
        finally:
            ctx.prec = old


def mu_of_M(M: arb) -> arb:
    """``mu(M) = M sin(pi / M) / pi``  (the writeup's bathtub bound on
    ``|hat h(1)|`` for any admissible test function ``h``).
    """
    return M * (arb.pi() / M).sin() / arb.pi()


def phi_N1(M: arb, y: arb, params: PhiParams) -> arb:
    """Rigorous arb enclosure of ``Phi(M, y)``.

    ``M.upper() < 0`` of the returned value certifies that ``(M, y)`` is
    in the forbidden region of the master inequality.
    """
    two = arb(2)
    y_sq = y * y
    rad1 = M - arb(1) - two * y_sq
    rad2 = params.K2 - arb(1) - two * params.k1 * params.k1
    s1 = safe_sqrt(rad1)
    s2 = safe_sqrt(rad2)
    rhs = M + arb(1) + two * y * params.k1 + s1 * s2
    lhs = arb(2) / arb(params.u) + params.gain_a
    return rhs - lhs


__all__ = [
    "PhiParams",
    "phi_N1",
    "mu_of_M",
    "safe_sqrt",
]
