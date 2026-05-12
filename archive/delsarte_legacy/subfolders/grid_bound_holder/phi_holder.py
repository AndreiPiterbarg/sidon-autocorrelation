"""Hoelder-generalised Phi for arbitrary N >= 1 using the HM-10 master inequality.

Derivation: see ``delsarte_dual/grid_bound_holder/derivation.md`` Theorem H.
Unconditional on the MV premises (Lemma 3.3 + MV's surrogate K_2 bound),
and does NOT depend on MO Conjecture 2.9.

At (p, q) = (2, 2), phi_holder reduces bit-for-bit to phi_mm; see
``tests/test_holder.py::test_reduces_to_mv_at_p_equals_2``.

Spec sign: Phi >= 0 <=> admissible.  Phi.upper() < 0 => forbidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from flint import arb, fmpq, ctx

# Import-only reuse of Phase 1/2 machinery.  No modification of upstream files.
from delsarte_dual.grid_bound.bessel import j0_pi_j_delta_over_u
from delsarte_dual.grid_bound.coeffs import (
    mv_coeffs_fmpq, MV_DELTA, MV_U, MV_K2_NUMERATOR,
)
from delsarte_dual.grid_bound.G_min import min_G_lower_bound
from delsarte_dual.grid_bound.phi_mm import (
    PhiMMParams, kn_period_one, mu_of_M, _safe_sqrt, arb_sqr,
)


# ------------------------------------------------------------------
# q-norm of the kernel Fourier sequence: K_q := sum_{j in Z} k_j^q.
# ------------------------------------------------------------------

def kq_kernel_upper(
    q: fmpq,
    delta: fmpq,
    K2_if_q_eq_2: arb,
    *,
    J_tail: int = 1024,
    prec_bits: int = 256,
) -> arb:
    """Rigorous upper arb on K_q = sum_{j in Z} k_j^q for q in (1, 2].

    At q == 2 exactly, returns ``K2_if_q_eq_2`` verbatim so that phi_holder
    collapses bit-for-bit to phi_mm (see derivation.md §5).

    For q != 2, we decompose

        K_q  =  1  +  2 * sum_{j=1..J} k_j^q  +  2 * sum_{j>J} k_j^q

    and bound the tail using Krasikov's inequality (Krasikov 2001):
        J_0(x)^2  <=  (2/pi) / sqrt(x^2 + 3/2)   for x >= sqrt(5/2).
    Dropping the +3/2 term in the denominator gives the looser but clean
    bound  J_0(x)^2 <= (2/pi) / x  for x >= sqrt(5/2),  hence

        k_j^q  <=  (2/(pi^2 j delta))^q   for  j >= ceil(sqrt(5/2)/(pi*delta)).

    Using the integral bound  sum_{j>J} j^{-q} <= J^{1-q}/(q-1) for q > 1:

        2 sum_{j>J} k_j^q  <=  2 * (2/(pi^2 delta))^q * J^{1-q} / (q - 1).

    For our kernel (delta = 0.138) with J >= 4, the threshold is comfortably
    satisfied.  The direct sum uses J = J_tail (default 1024).

    NB: the Krasikov bound assumes q >= 1.
    """
    if q == fmpq(2):
        return K2_if_q_eq_2
    if q < fmpq(1):
        raise ValueError(f"kq_kernel_upper requires q >= 1, got {q}")
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        q_arb = arb(q)
        pi_arb = arb.pi()
        delta_arb = arb(delta)

        # Direct sum: j = 1 .. J_tail (paired with -j gives factor 2)
        total = arb(1)             # k_0^q = 1 exactly
        for j in range(1, J_tail + 1):
            k_j = kn_period_one(j, delta, prec_bits=prec_bits)
            k_lo = k_j.lower()
            if k_lo > 0:
                kq = k_j ** q_arb
            else:
                k_u = k_j.upper()
                if k_u <= 0:
                    kq = arb(0)
                else:
                    kq = arb(0).union(arb(k_u) ** q_arb)
            total = total + arb(2) * kq

        # Krasikov tail bound (rigorous for J_tail s.t. pi*(J_tail+1)*delta >= sqrt(5/2))
        # sqrt(5/2) ~ 1.581; pi*delta = 0.4335; so J_tail + 1 >= 4 suffices.
        pd = pi_arb * delta_arb
        # Check threshold: pi*(J_tail+1)*delta >= sqrt(5/2)
        # We need J_tail >= 4 at least; enforce a margin.
        if J_tail < 8:
            raise ValueError(f"J_tail must be >= 8 for Krasikov tail, got {J_tail}")
        # Envelope: k_j^q <= (2/(pi^2 j delta))^q = (2/(pi*pd))^q / j^q
        #         = ((2/pi) / pd)^q / j^q.  With C := (2/(pi*pd))^q:
        C = (arb(2) / (pi_arb * pd)) ** q_arb
        # sum_{j > J_tail} 1/j^q <= J_tail^{1-q} / (q - 1)
        J_arb = arb(J_tail)
        one = arb(1)
        tail_sum_bound = one / ((q_arb - one) * (J_arb ** (q_arb - one)))
        tail_pair = arb(2) * C * tail_sum_bound
        return total + tail_pair
    finally:
        ctx.prec = old


def _safe_root(x: arb, exp_q: fmpq) -> arb:
    """Rigorous x^{exp_q} clamped to non-negative domain.

    * exp_q = 1/p with p >= 2 means exp_q in (0, 1/2].
    * If x.upper() < 0: raise (non-physical radicand).
    * If x.lower() >= 0: return x ** arb(exp_q).
    * Else: return enclosure [0, (x.upper())^{exp_q}].
    """
    x_up = x.upper()
    if x_up < 0:
        raise ValueError(f"_safe_root: x.upper()={x_up} < 0 for exp_q={exp_q}")
    x_lo = x.lower()
    if x_lo >= 0:
        return x ** arb(exp_q)
    return arb(0).union(arb(x_up) ** arb(exp_q))


# ------------------------------------------------------------------
# PhiHolderParams: precompiled inputs for (HM-10) at given (p, q).
# ------------------------------------------------------------------

@dataclass(frozen=True)
class PhiHolderParams:
    """Precompiled rigorous inputs for phi_holder at arbitrary N_max and (p, q).

    Extends PhiMMParams' data by storing the Hoelder exponents and the
    kernel-side q-norm bound K_q.

    At (p, q) = (2, 2), K_q is taken to be PhiMMParams.K2 exactly so that
    phi_holder == phi_mm at arb precision.
    """
    # Base Phi_MM params (reused verbatim)
    delta:        fmpq
    u:            fmpq
    K2:           arb                   # MV's ||K||_2^2 (surrogate)
    k_arb:        tuple                 # length N_max, k_n values
    sum_kn_sq_arb: arb                  # sum_n k_n^2 (MM-10 tail subtraction)
    gain_a:       arb
    min_G:        arb
    S1:           arb
    n_coeffs:     int
    N_max:        int
    min_G_center: fmpq

    # Hoelder-specific
    p:            fmpq                  # p >= 2
    q:            fmpq                  # q = p/(p-1)
    K_q_upper:    arb                   # rigorous upper on sum_j k_j^q
    sum_kn_q_arb: arb                   # sum_{n=1..N_max} k_n^q (tail subtraction)

    @classmethod
    def from_mv(
        cls,
        N_max: int,
        p: fmpq,
        *,
        delta: fmpq = MV_DELTA,
        u: fmpq = MV_U,
        coeffs: Sequence[fmpq] = None,
        K2_times_delta: fmpq = MV_K2_NUMERATOR,
        n_cells_min_G: int = 8192,
        J_tail: int = 512,
        prec_bits: int = 256,
    ) -> "PhiHolderParams":
        if N_max < 1:
            raise ValueError("N_max must be >= 1")
        if p < fmpq(2):
            raise ValueError(f"Hoelder requires p >= 2, got {p}")
        if coeffs is None:
            coeffs = mv_coeffs_fmpq()

        q = p / (p - fmpq(1))    # conjugate

        # Build the underlying PhiMMParams and then lift it.
        base = PhiMMParams.from_mv(
            N_max=N_max,
            delta=delta,
            u=u,
            coeffs=coeffs,
            K2_times_delta=K2_times_delta,
            n_cells_min_G=n_cells_min_G,
            prec_bits=prec_bits,
        )

        old = ctx.prec
        ctx.prec = prec_bits
        try:
            # K_q_upper
            K_q_up = kq_kernel_upper(
                q, delta, base.K2,
                J_tail=J_tail, prec_bits=prec_bits,
            )
            # sum_{n=1..N_max} k_n^q
            q_arb = arb(q)
            sum_kn_q = arb(0)
            for k_n in base.k_arb:
                if q == fmpq(2):
                    sum_kn_q = sum_kn_q + k_n * k_n
                else:
                    sum_kn_q = sum_kn_q + k_n ** q_arb
            return cls(
                delta=base.delta,
                u=base.u,
                K2=base.K2,
                k_arb=base.k_arb,
                sum_kn_sq_arb=base.sum_kn_sq_arb,
                gain_a=base.gain_a,
                min_G=base.min_G,
                S1=base.S1,
                n_coeffs=base.n_coeffs,
                N_max=base.N_max,
                min_G_center=base.min_G_center,
                p=p,
                q=q,
                K_q_upper=K_q_up,
                sum_kn_q_arb=sum_kn_q,
            )
        finally:
            ctx.prec = old

    def as_phi_mm_params(self) -> PhiMMParams:
        """Return the underlying PhiMMParams (sans Hoelder-specific fields).

        Useful for reusing MM-10 helpers (e.g. cell-search starting boxes).
        """
        return PhiMMParams(
            delta=self.delta,
            u=self.u,
            K2=self.K2,
            k_arb=self.k_arb,
            sum_kn_sq_arb=self.sum_kn_sq_arb,
            gain_a=self.gain_a,
            min_G=self.min_G,
            S1=self.S1,
            n_coeffs=self.n_coeffs,
            N_max=self.N_max,
            min_G_center=self.min_G_center,
        )


# ------------------------------------------------------------------
# Phi_Hoelder (HM-10)
# ------------------------------------------------------------------

def phi_holder(
    M: arb,
    ab_vec: Sequence[arb],
    params: PhiHolderParams,
) -> arb:
    """Arb enclosure of Phi_Hoelder(M, (a_1, b_1, ..., a_N, b_N)) per (HM-10).

    Spec sign:  Phi >= 0 <=> (M, ab) consistent with admissibility.
    Phi.upper() < 0 certifies (M, ab) is forbidden.

    At (p, q) = (2, 2), this function is bit-identical to phi_mm at the
    arb level (the two code paths use the same K_2 and the same sum_kn_sq).
    """
    N = params.N_max
    p = params.p
    q = params.q
    if len(ab_vec) != 2 * N:
        raise ValueError(
            f"ab_vec length {len(ab_vec)} != 2*N_max={2*N}"
        )

    # === Fast path at p = q = 2 (for bit-identical MM-10 agreement) ===
    if p == fmpq(2) and q == fmpq(2):
        two = arb(2)
        sum_zk = arb(0)
        sum_z4 = arb(0)
        for n in range(1, N + 1):
            a_n = ab_vec[2 * (n - 1)]
            b_n = ab_vec[2 * (n - 1) + 1]
            z_n_sq = arb_sqr(a_n) + arb_sqr(b_n)
            sum_zk = sum_zk + z_n_sq * params.k_arb[n - 1]
            sum_z4 = sum_z4 + arb_sqr(z_n_sq)
        rad1 = M - arb(1) - two * sum_z4
        rad2 = params.K2 - arb(1) - two * params.sum_kn_sq_arb
        s1 = _safe_sqrt(rad1)
        s2 = _safe_sqrt(rad2)
        rhs = M + arb(1) + two * sum_zk + s1 * s2
        lhs = arb(2) / arb(params.u) + params.gain_a
        return rhs - lhs

    # === General Hoelder (p, q) path (TIGHT h-tail via Lemma 1) ===
    # Tight tail bound (derivation.md Lemma 1 + Parseval):
    #   sum_{|j|>N} |hat h(j)|^p  <=  mu(M)^{p-2} * (M - 1 - 2 sum_{n=1..N} z_n^4)
    # (uses |hat h(n)| = z_n^2 <= mu(M) pointwise).
    # At p = 2, mu^0 = 1 and this collapses to MM-10's radicand exactly.
    # At p > 2 it is STRICTLY tighter than the naive Hausdorff-Young bound
    # ``M - 1 - 2 sum z_n^{2p}`` used in earlier versions of this file.
    two = arb(2)
    p_arb = arb(p)
    inv_p = fmpq(1) / p
    inv_q = fmpq(1) / q

    sum_zk = arb(0)          # sum_n z_n^2 k_n
    sum_z4 = arb(0)          # sum_n z_n^4 (for the tight h-tail bound)
    for n in range(1, N + 1):
        a_n = ab_vec[2 * (n - 1)]
        b_n = ab_vec[2 * (n - 1) + 1]
        z_n_sq = arb_sqr(a_n) + arb_sqr(b_n)     # z_n^2 in [0, ...]
        sum_zk = sum_zk + z_n_sq * params.k_arb[n - 1]
        sum_z4 = sum_z4 + arb_sqr(z_n_sq)         # z_n^4 in [0, ...]

    # Tight h-tail: mu(M)^{p-2} * (M - 1 - 2 sum z_n^4)^{1/p}
    mu = mu_of_M(M)
    rad_h = M - arb(1) - two * sum_z4
    # mu(M)^{p-2}: mu in (0, 1), p-2 >= 0 so mu^{p-2} in (0, 1].
    if p == fmpq(2):
        mu_power = arb(1)
    else:
        # fmpq exponent p-2 >= 0; mu.lower() > 0 at the M values of interest
        mu_lo = mu.lower()
        if mu_lo > 0:
            mu_power = mu ** arb(p - fmpq(2))
        else:
            # Degenerate mu straddling 0: enclose in [0, mu.upper()^{p-2}].
            mu_u = mu.upper()
            if mu_u <= 0:
                mu_power = arb(0)
            else:
                mu_power = arb(0).union(arb(mu_u) ** arb(p - fmpq(2)))

    # (mu^{p-2} * rad_h)^{1/p}
    scaled_rad_h = mu_power * rad_h
    fh = _safe_root(scaled_rad_h, inv_p)

    # k-side radicand: K_q - 1 - 2 sum k_n^q
    rad_k = params.K_q_upper - arb(1) - two * params.sum_kn_q_arb
    fk = _safe_root(rad_k, inv_q)

    rhs = M + arb(1) + two * sum_zk + fh * fk
    lhs = arb(2) / arb(params.u) + params.gain_a
    return rhs - lhs


__all__ = [
    "PhiHolderParams",
    "phi_holder",
    "kq_kernel_upper",
    "_safe_root",
]
