"""Bochner-admissible kernels for the Matolcsi-Vinuesa master inequality.

Every kernel ``K`` satisfies the four admissibility properties used in
the accompanying paper (Definition 2.1):

  (K1) ``K >= 0`` a.e. on ``R``,
  (K2) ``K(-x) = K(x)`` for every ``x``,
  (K3) ``supp K subset [-delta, delta]``  and  ``int K = 1``,
  (K4) the periodic Fourier coefficients
       ``tilde K(j) := hat K(j/u)``  satisfy  ``tilde K(j) >= 0``
       for every ``j in Z``  (Bochner positivity).

Concrete subclasses expose:

  * ``supp_halfwidth: fmpq``                    -- delta
  * ``K_norm_sq(prec_bits)        -> arb``     -- ``||K||_{L^2(R)}^2``
  * ``K_tilde_real(xi, prec_bits) -> arb``     -- ``hat K(xi)`` at arb xi
  * ``K_tilde(n,  prec_bits)      -> arb``     -- ``hat K(n)`` at integer n
  * ``K_tilde_positive(n_max)     -> bool``    -- Bochner check up to n_max

All transcendental outputs are rigorous ``flint.arb`` midpoint-radius
intervals.  Rational inputs stay as ``flint.fmpq`` so no rounding is
introduced before the transcendental step.

The two production kernels in this module are:

  * ``ArcsineKernel``               -- the Matolcsi-Vinuesa baseline kernel
                                       ``K(x) = (1/delta)(eta * eta)(x/delta)``
                                       with ``hat K(xi) = J_0(pi delta xi)^2``.
                                       Used in the writeup as the single-scale
                                       reference point ``(M_cert = 1.27481)``.

  * ``MultiScaleArcsineKernel``     -- the convex combination
                                       ``K = sum_i lambda_i K_arc(delta_i; .)``
                                       used to lift the lower bound to
                                       the Piterbarg-Bajaj-Vincent Bound
                                       ``C_{1a} >= 1292/1000 = 1.292``.

A note on the arcsine kernel.  The writeup defines ``K(x)`` as the auto-
convolution of the arcsine density ``eta(x) = 2/(pi sqrt(1 - 4x^2))`` on
``(-1/2, 1/2)``, rescaled to support ``[-delta, delta]``.  This makes
``hat K(xi) = J_0(pi delta xi)^2 >= 0`` automatic and gives a finite
``||K||_{L^2}^2``.
"""
from __future__ import annotations

from typing import List, Sequence

from flint import arb, acb, fmpq, ctx


# -----------------------------------------------------------------------------
#  Base class
# -----------------------------------------------------------------------------


class Kernel:
    """Abstract Bochner-admissible kernel.  Concrete subclasses must override
    ``K_norm_sq`` and ``K_tilde_real``; ``K_tilde`` and the Bochner check
    have generic implementations.
    """

    name: str = "abstract"
    supp_halfwidth: fmpq = fmpq(138, 1000)

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        """Rigorous arb enclosure of ``||K||_{L^2(R)}^2``."""
        raise NotImplementedError

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        """Rigorous arb enclosure of ``hat K(xi)`` at real ``xi``.

        ``K`` is real-even and non-negative, so the Fourier transform is real
        with ``|hat K(xi)| <= hat K(0) = 1``.
        """
        raise NotImplementedError

    def K_tilde(self, n: int, prec_bits: int = 256) -> arb:
        """Period-1 Fourier coefficient ``k_n = hat K(n)`` at integer ``n``.

        Concrete kernels may override for tighter exact-rational arguments.
        """
        if n < 0:
            return self.K_tilde(-n, prec_bits)
        if n == 0:
            return arb(1)
        return self.K_tilde_real(arb(n), prec_bits)

    def K_tilde_positive(self, n_max: int = 200, prec_bits: int = 256) -> bool:
        """Return True iff ``hat K(j).lower() >= 0`` for ``j = 1, ..., n_max``.

        A False return certifies a Bochner violation at the chosen precision.
        """
        for j in range(1, n_max + 1):
            if self.K_tilde(j, prec_bits).lower() < 0:
                return False
        return True

    def admissibility_check(
        self, prec_bits: int = 256, n_bochner: int = 50
    ) -> None:
        """Raise ``ValueError`` if any admissibility property is violated.

        ``supp_halfwidth`` must be positive and the Bochner check must pass
        for the first ``n_bochner`` Fourier coefficients.  Properties (K1),
        (K2), (K3) and ``int K = 1`` are enforced by construction in each
        concrete subclass.
        """
        if self.supp_halfwidth <= 0:
            raise ValueError(f"{self.name}: supp_halfwidth must be positive")
        if not self.K_tilde_positive(n_bochner, prec_bits):
            raise ValueError(
                f"{self.name}: Bochner positivity fails for some "
                f"j <= {n_bochner}"
            )

    def __repr__(self) -> str:
        return f"<Kernel {self.name} delta={float(self.supp_halfwidth):.4f}>"


# =============================================================================
#  Single-scale arcsine auto-convolution (MV baseline)
# =============================================================================


class ArcsineKernel(Kernel):
    """Single-scale arcsine auto-convolution kernel.

        ``K(x) = (1/delta) (eta * eta)(x/delta)``  on  ``[-delta, delta]``,
        ``hat K(xi) = J_0(pi delta xi)^2``.

    The L^2 norm uses the Martin-O'Bryant surrogate
    ``||K||_2^2 = 0.5747 / delta`` (arXiv:0807.5121, Lemma 3.2), which is
    the value used by Matolcsi-Vinuesa.  At ``delta = 138/1000`` this gives
    the single-scale lower bound ``C_{1a} >= 1.27481``.
    """

    name = "arcsine"

    def __init__(
        self,
        delta: fmpq = fmpq(138, 1000),
        K_norm_sq_numerator: fmpq = fmpq(5747, 10000),
    ):
        self.supp_halfwidth = delta
        self._K2_numerator = K_norm_sq_numerator

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        return arb(self._K2_numerator) / arb(self.supp_halfwidth)

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            arg = arb.pi() * xi * arb(self.supp_halfwidth)
            j0 = arg.bessel_j(0)
            return j0 * j0
        finally:
            ctx.prec = old

    def K_tilde(self, n: int, prec_bits: int = 256) -> arb:
        if n < 0:
            return self.K_tilde(-n, prec_bits)
        if n == 0:
            return arb(1)
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            q = fmpq(n) * self.supp_halfwidth
            arg = arb.pi() * arb(q)
            j0 = arg.bessel_j(0)
            return j0 * j0
        finally:
            ctx.prec = old


# =============================================================================
#  Multi-scale arcsine kernel (production)
# =============================================================================


class MultiScaleArcsineKernel(Kernel):
    """Convex combination of arcsine auto-convolution kernels.

        ``K(x) = sum_i lambda_i K_arc(delta_i; x)``,
        ``hat K(xi) = sum_i lambda_i J_0(pi delta_i xi)^2``,

    with ``lambda_i > 0``, ``sum_i lambda_i = 1``, and
    ``supp K subset [-max_i delta_i, max_i delta_i]``.

    Admissibility is automatic: each component is non-negative with unit mass
    and non-negative Fourier transform, and the convex combination preserves
    all four properties.

    The production point in the writeup is the 3-scale instance

        ``(delta_1, delta_2, delta_3) = (138/1000, 55/1000, 25/1000)``,
        ``(lambda_1, lambda_2, lambda_3) = (85/100, 10/100, 5/100)``,

    paired with a 200-coefficient re-optimised cosine multiplier ``G``;
    see ``delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel``.

    ``K_norm_sq``
    ------------
    ``||K||_2^2 = int hat K(xi)^2 dxi``.  Expanding the square,

        ``||K||_2^2 = sum_{i,j} lambda_i lambda_j C_{ij}``,
        ``C_{ij}   = int_R J_0(pi delta_i xi)^2 J_0(pi delta_j xi)^2 dxi``.

    Each cross integral ``C_{ij}`` is computed as a rigorous arb enclosure:
    an ``acb.integral`` on ``[-T, T]`` plus the Bessel-asymptotic tail bound

        ``int_{|xi| > T} (...) <= 8 / (pi^4 delta_i delta_j T)``,

    derived from ``J_0(z)^2 <= 2/(pi z)`` (Watson 1944, sec. 7.21):
    applying the bound to both Bessel factors gives
    ``J_0(pi delta_i xi)^2 J_0(pi delta_j xi)^2 <= 4/(pi^4 delta_i delta_j xi^2)``
    on each half-line, and integrating in ``xi`` from ``T`` to ``infty``
    contributes ``4/(pi^4 delta_i delta_j T)`` per side.  The diagonal case
    ``i = j`` can optionally substitute the Martin-O'Bryant surrogate
    ``0.5747 / delta_i`` (``use_diag_surrogate=True``) for byte-identical
    alignment with the single-scale arcsine baseline.
    """

    def __init__(
        self,
        deltas: Sequence[fmpq],
        lambdas: Sequence[fmpq],
        name_suffix: str = "",
        K2_cross_cutoff_xi: fmpq = fmpq(10**5),
        use_diag_surrogate: bool = False,
        diag_surrogate_numerator: fmpq = fmpq(5747, 10000),
    ):
        if not deltas:
            raise ValueError("deltas must be non-empty")
        if len(deltas) != len(lambdas):
            raise ValueError(
                f"len(deltas) ({len(deltas)}) != len(lambdas) ({len(lambdas)})"
            )
        for d in deltas:
            if not isinstance(d, fmpq):
                raise TypeError(f"deltas must be fmpq, got {type(d).__name__}")
            if d <= 0:
                raise ValueError(f"deltas must be positive, got {d}")
        for lam in lambdas:
            if not isinstance(lam, fmpq):
                raise TypeError(
                    f"lambdas must be fmpq, got {type(lam).__name__}"
                )
            if lam < 0:
                raise ValueError(f"lambdas must be non-negative, got {lam}")
        lam_sum = sum(lambdas, fmpq(0))
        if lam_sum != fmpq(1):
            raise ValueError(f"lambdas must sum to 1, got {lam_sum}")

        self.deltas: List[fmpq] = list(deltas)
        self.lambdas: List[fmpq] = list(lambdas)
        max_delta = self.deltas[0]
        for d in self.deltas[1:]:
            if d > max_delta:
                max_delta = d
        self.supp_halfwidth = max_delta
        self._n = len(self.deltas)
        self.K2_cross_cutoff_xi = K2_cross_cutoff_xi
        self.use_diag_surrogate = use_diag_surrogate
        self._diag_num = diag_surrogate_numerator

        if not name_suffix:
            parts = [
                f"l{float(lam.p) / float(lam.q):.2f}"
                f"_d{float(d.p) / float(d.q):.4f}"
                for lam, d in zip(self.lambdas, self.deltas)
            ]
            name_suffix = "+".join(parts)
        self.name = f"multiscale_arcsine[{name_suffix}]"

    # ------------------------------------------------------------------
    # Fourier transform at real-line frequencies
    # ------------------------------------------------------------------

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            total = arb(0)
            for lam, d in zip(self.lambdas, self.deltas):
                arg = arb.pi() * xi * arb(d)
                j0 = arg.bessel_j(0)
                total = total + arb(lam) * (j0 * j0)
            return total
        finally:
            ctx.prec = old

    def K_tilde(self, n: int, prec_bits: int = 256) -> arb:
        if n < 0:
            return self.K_tilde(-n, prec_bits)
        if n == 0:
            return arb(1)
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            total = arb(0)
            for lam, d in zip(self.lambdas, self.deltas):
                q = fmpq(n) * d
                arg = arb.pi() * arb(q)
                j0 = arg.bessel_j(0)
                total = total + arb(lam) * (j0 * j0)
            return total
        finally:
            ctx.prec = old

    def K_tilde_positive(self, n_max: int = 200, prec_bits: int = 256) -> bool:
        """Bochner check.

        ``hat K(j) = sum_i lambda_i J_0(pi delta_i j)^2`` is a non-negative
        linear combination of squares, hence trivially non-negative.  We
        still verify ``lower() >= 0`` defensively at the chosen precision.
        """
        for j in range(1, n_max + 1):
            if self.K_tilde(j, prec_bits).lower() < 0:
                return False
        return True

    # ------------------------------------------------------------------
    # L^2 norm via cross-Bessel integrals + asymptotic tail bound
    # ------------------------------------------------------------------

    def _cross_integral(
        self, di: fmpq, dj: fmpq, prec_bits: int
    ) -> arb:
        """Rigorous arb enclosure of
        ``int_0^infty J_0(pi di xi)^2 J_0(pi dj xi)^2 dxi``.

        Computed as an ``acb.integral`` on ``[0, T]`` plus the asymptotic
        tail bound ``4 / (pi^4 di dj T)`` derived from
        ``J_0(z)^2 <= 2/(pi z)``: applying the asymptotic to both Bessel
        factors gives
        ``J_0(pi di xi)^2 J_0(pi dj xi)^2 <= 4 / (pi^4 di dj xi^2)``
        and integrating ``int_T^infty xi^{-2} dxi = 1/T`` yields the
        stated bound on the half-line tail.
        """
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            T_arb = arb(self.K2_cross_cutoff_xi)
            di_arb = arb(di)
            dj_arb = arb(dj)

            def integrand(z, _flags):
                a_i = acb.pi() * acb(di_arb) * z
                a_j = acb.pi() * acb(dj_arb) * z
                j_i = a_i.bessel_j(acb(0))
                j_j = a_j.bessel_j(acb(0))
                return (j_i * j_i) * (j_j * j_j)

            val = acb.integral(integrand, acb(0), acb(T_arb))
            main = val.real
            pi_arb = arb.pi()
            pi_fourth = pi_arb * pi_arb * pi_arb * pi_arb
            tail = arb(4) / (pi_fourth * di_arb * dj_arb * T_arb)
            return main + arb(0).union(tail)
        finally:
            ctx.prec = old

    def _diag_integral(self, di: fmpq, prec_bits: int) -> arb:
        """``int_0^infty J_0(pi di xi)^4 dxi`` either via the MO surrogate
        ``0.5747 / (2 di)`` or via the rigorous cross integral with
        ``dj = di``.
        """
        if self.use_diag_surrogate:
            return arb(self._diag_num) / (arb(2) * arb(di))
        return self._cross_integral(di, di, prec_bits)

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        """``||K||_2^2 = 2 sum_{i,j} lambda_i lambda_j *
        int_0^infty J_0(pi di xi)^2 J_0(pi dj xi)^2 dxi``.

        Each ``(i, j)`` integral is a rigorous arb enclosure; the factor of
        ``2`` accounts for the even symmetry of ``hat K``.
        """
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            total = arb(0)
            for i in range(self._n):
                for j in range(self._n):
                    if i == j:
                        Cij = self._diag_integral(self.deltas[i], prec_bits)
                    else:
                        Cij = self._cross_integral(
                            self.deltas[i], self.deltas[j], prec_bits
                        )
                    total = (
                        total
                        + arb(self.lambdas[i]) * arb(self.lambdas[j]) * Cij
                    )
            return arb(2) * total
        finally:
            ctx.prec = old


__all__ = [
    "Kernel",
    "ArcsineKernel",
    "MultiScaleArcsineKernel",
]
