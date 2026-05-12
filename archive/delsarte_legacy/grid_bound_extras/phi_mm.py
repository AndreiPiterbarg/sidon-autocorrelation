"""Multi-moment Phi for arbitrary N >= 1 using the MM-10 master inequality.

Derivation (see ``delsarte_dual/multi_moment_derivation.md`` Theorem 2 + Lemma 1):
  Let N = {1, ..., N_max}, let h = f*f, M = ||h||_inf, and z_n := |hat f(n)|
  (period-1 Fourier).  Parseval splits int (f*f) K exactly over N and the
  tail; applying Cauchy-Schwarz only to the tail gives

    2/u + a  <=  M + 1 + 2 sum_{n in N} z_n^2 k_n
                 + sqrt((M - 1 - 2 sum z_n^4)_+) * sqrt((K2 - 1 - 2 sum k_n^2)_+).
                                                                 (MM-10)

  Here we use the bound Re hat h(n) <= |hat h(n)| = z_n^2 to upgrade the
  exact Parseval term  2 sum_{n>=1} Re hat h(n) k_n  to  2 sum z_n^2 k_n.
  This is conservative (gives a LARGER RHS), which in our sign convention
  means Phi(M, (a,b)) with z_n^2 = a_n^2+b_n^2 is an UPPER BOUND on the
  true Phi.  Therefore ``Phi.upper() < 0`` still certifies forbidden.

Setting N_max = 1 recovers Phi_N1 = MV eq. (10).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from flint import arb, fmpq, ctx

from .bessel import j0_pi_j_delta_over_u, k1_period_one
from .coeffs import mv_coeffs_fmpq, MV_DELTA, MV_U, MV_K2_NUMERATOR
from .G_min import min_G_lower_bound


def _arb_pi_n_delta(n: int, delta: fmpq) -> arb:
    """pi * n * delta as arb at current precision."""
    return arb.pi() * arb(fmpq(n) * delta)


def kn_period_one(n: int, delta: fmpq, prec_bits: int = 256) -> arb:
    """k_n = hat K(n) = |J_0(pi n delta)|^2 for n >= 1, period-1 convention.

    Derivation: K(x) = (1/delta) beta(x/delta) with beta the arcsine density
    on (-1/2, 1/2); its real-line Fourier transform at xi is J_0(pi xi);
    the period-1 Fourier coefficient at integer n is J_0(pi n delta)^2.
    (``multi_moment_derivation.md`` §1 and ``mv_construction_detailed.md``
    Section 2 Remark.)
    """
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        j0 = _arb_pi_n_delta(n, delta).bessel_j(0)
        return j0 * j0
    finally:
        ctx.prec = old


def _safe_sqrt(x: arb) -> arb:
    """Rigorous sqrt clamped to non-negative domain.

    If x.upper() < 0: raise (non-physical radicand for the MM-10 inequality).
    If x.lower() >= 0: return x.sqrt().
    Else: return arb(0).union(x.upper().sqrt()) -- enclosure of sqrt on the
    non-negative sub-interval of x's ball.
    """
    x_up = x.upper()
    if x_up < 0:
        raise ValueError(f"_safe_sqrt: x.upper()={x_up} < 0")
    x_lo = x.lower()
    if x_lo >= 0:
        return x.sqrt()
    return arb(0).union(x_up.sqrt())


def arb_sqr(x: arb) -> arb:
    """Dependency-aware square: rigorous enclosure of {x_0^2 : x_0 in x}.

    python-flint's ``x * x`` treats the two factors as independent, giving
    the loose enclosure [-(max|x|)^2, (max|x|)^2] whenever x straddles zero.
    This helper returns the tight [min(|x|)^2, max(|x|)^2].

    Correctness: for a single-variable x,
      * if 0 in x: range of x^2 is [0, max(|x_lo|, |x_up|)^2].
      * else:      range is [min(|x_lo|,|x_up|)^2, max^2]  (x monotone on side).
    ``arb.abs_lower()`` returns min|x| in x's ball, ``arb.abs_upper()`` max|x|;
    both are scalars (tight arbs), so ``a * a`` on them is also tight.  Taking
    the union of the two squared endpoints gives a rigorous enclosure of the
    true range with no dependency artefact.
    """
    al = x.abs_lower()
    au = x.abs_upper()
    return (al * al).union(au * au)


@dataclass(frozen=True)
class PhiMMParams:
    """Precompiled rigorous inputs for phi_mm at arbitrary N_max.

    All arithmetic fields are arb intervals except the exact rationals
    (delta, u, min_G_center).  The k_n vector is indexed 1..N_max;
    ``k_arb[n-1]`` is k_n.  ``sum_kn_sq_arb`` = sum_{n=1..N_max} k_n^2 is
    precomputed for the MM-10 tail radicand.
    """
    delta:        fmpq
    u:            fmpq
    K2:           arb
    k_arb:        tuple            # length N_max
    sum_kn_sq_arb: arb             # sum_n k_n^2 (the tail subtraction)
    gain_a:       arb
    min_G:        arb
    S1:           arb
    n_coeffs:     int
    N_max:        int
    min_G_center: fmpq

    @classmethod
    def from_mv(
        cls,
        N_max: int,
        delta: fmpq = MV_DELTA,
        u: fmpq = MV_U,
        coeffs: Sequence[fmpq] = None,
        K2_times_delta: fmpq = MV_K2_NUMERATOR,
        n_cells_min_G: int = 8192,
        prec_bits: int = 256,
    ) -> "PhiMMParams":
        if N_max < 1:
            raise ValueError("N_max must be >= 1")
        if coeffs is None:
            coeffs = mv_coeffs_fmpq()
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            # k_n, n = 1..N_max
            k_list = tuple(
                kn_period_one(n, delta, prec_bits=prec_bits)
                for n in range(1, N_max + 1)
            )
            sum_kn_sq = arb(0)
            for k in k_list:
                sum_kn_sq = sum_kn_sq + k * k

            K2_arb = arb(K2_times_delta) / arb(delta)

            S1 = arb(0)
            for j, a_j in enumerate(coeffs, start=1):
                j0 = j0_pi_j_delta_over_u(j, delta, u, prec_bits=prec_bits)
                S1 = S1 + (arb(a_j) * arb(a_j)) / (j0 * j0)

            min_G_encl, min_G_center = min_G_lower_bound(
                coeffs, u, n_cells=n_cells_min_G, prec_bits=prec_bits
            )
            min_G_cert = min_G_encl.lower()
            if min_G_cert.upper() <= 0:
                raise ValueError(f"min G cert lower <= 0: {min_G_cert}")
            gain_a = (arb(4) / arb(u)) * (min_G_cert * min_G_cert) / S1

            return cls(
                delta=delta,
                u=u,
                K2=K2_arb,
                k_arb=k_list,
                sum_kn_sq_arb=sum_kn_sq,
                gain_a=gain_a,
                min_G=min_G_cert,
                S1=S1,
                n_coeffs=len(coeffs),
                N_max=N_max,
                min_G_center=min_G_center,
            )
        finally:
            ctx.prec = old


def mu_of_M(M: arb) -> arb:
    """mu(M) := M sin(pi/M) / pi  (uniform bathtub bound, Lemma 1)."""
    return M * (arb.pi() / M).sin() / arb.pi()


def phi_mm(
    M: arb,
    ab_vec: Sequence[arb],          # length 2*N_max: (a_1, b_1, a_2, b_2, ..., a_N, b_N)
    params: PhiMMParams,
) -> arb:
    """Arb enclosure of Phi_MM(M, (a_1, b_1, ..., a_N, b_N)) per (MM-10).

    Spec sign:  Phi_MM >= 0  <=>  (M, ab) consistent with admissibility.
    Phi_MM.upper() < 0 certifies (M, ab) is forbidden.

    The formula depends on the moments through z_n^2 = a_n^2 + b_n^2 and
    z_n^4 = (a_n^2 + b_n^2)^2; phases (arg hat f(n)) drop out because (MM-10)
    only bounds |Re hat h(n)| by |hat h(n)| = z_n^2.
    """
    N = params.N_max
    if len(ab_vec) != 2 * N:
        raise ValueError(
            f"ab_vec length {len(ab_vec)} != 2*N_max={2*N}"
        )
    # Compute z_n^2, z_n^4, and the two M-dependent sums.
    # We use arb_sqr (dependency-aware) so that cells straddling zero give
    # tight [0, max(|.|)^2] rather than [-max^2, max^2].
    sum_zk = arb(0)    # sum_n z_n^2 k_n
    sum_z4 = arb(0)    # sum_n z_n^4
    for n in range(1, N + 1):
        a_n = ab_vec[2 * (n - 1)]
        b_n = ab_vec[2 * (n - 1) + 1]
        z_n_sq = arb_sqr(a_n) + arb_sqr(b_n)
        sum_zk = sum_zk + z_n_sq * params.k_arb[n - 1]
        sum_z4 = sum_z4 + arb_sqr(z_n_sq)

    two = arb(2)
    rad1 = M - arb(1) - two * sum_z4
    rad2 = params.K2 - arb(1) - two * params.sum_kn_sq_arb
    s1 = _safe_sqrt(rad1)
    s2 = _safe_sqrt(rad2)
    rhs = M + arb(1) + two * sum_zk + s1 * s2
    lhs = arb(2) / arb(params.u) + params.gain_a
    return rhs - lhs


__all__ = ["PhiMMParams", "phi_mm", "kn_period_one", "mu_of_M", "_safe_sqrt"]
