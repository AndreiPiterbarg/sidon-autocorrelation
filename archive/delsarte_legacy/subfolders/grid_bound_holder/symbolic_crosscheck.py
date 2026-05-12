"""Symbolic cross-check of the Hoelder-generalised RHS of the MV/MM-10
master inequality against the Cauchy-Schwarz MM-10 special case.

Background
----------
Theorem 2 of ``delsarte_dual/multi_moment_derivation.md`` (equation MM-10)
bounds the tail ``sum_{|j|>N} hat h(j) k_j`` by Cauchy-Schwarz, giving

    RHS_CS = M + 1 + 2 * sum_{n=1..N} z_n^2 k_n
             + sqrt(M - 1 - 2 * sum_n z_n^4)
             * sqrt(K2 - 1 - 2 * sum_n k_n^2).

The Hoelder generalisation replaces the ``(1/2, 1/2)`` split with a
conjugate pair ``(1/p_hy', 1/q_kernel)`` subject to
``1/p_hy' + 1/q_kernel = 1`` and ``p_hy in [1, 2]``:

    | sum_tail hat h(j) k_j |
        <= ( sum_tail |hat h(j)|^{p_hy'} )^{1/p_hy'}
         * ( sum_tail k_j^{q_kernel}   )^{1/q_kernel}.

Hausdorff-Young at exponent ``p_hy in [1,2]`` gives
    ( sum_j |hat h(j)|^{p_hy'} )^{1/p_hy'}  <=  ||h||_{p_hy},
where ``p_hy' = p_hy/(p_hy-1) >= 2`` is the conjugate.  Since ``0 <= h <= M``
and ``int h = 1``,
    ||h||_{p_hy}^{p_hy} = int h^{p_hy} <= M^{p_hy - 1} * int h = M^{p_hy - 1}.
Hence
    sum_j |hat h(j)|^{p_hy'}  <=  ||h||_{p_hy}^{p_hy'}  =  M^{(p_hy-1) * p_hy'/p_hy}
                                                       =  M^{p_hy'/(p_hy-1) * (p_hy-1)/p_hy * ... }
    # Simplify: (p_hy - 1) * p_hy' / p_hy = (p_hy - 1)/p_hy * p_hy/(p_hy - 1) = 1
    # So ||h||_{p_hy}^{p_hy'} = M^{ p_hy' * (p_hy-1)/p_hy }.  We'll simplify
    # directly with sympy below using the substitution p_hy_prime = p_hy/(p_hy-1).

For the MM-10 / Cauchy-Schwarz case ``p_hy = 2``, ``p_hy' = 2``,
``q_kernel = 2``; ``M^{p_hy' * (p_hy-1)/p_hy} = M^{2 * 1/2} = M`` and the
formula reduces to MM-10.

Exponent-bookkeeping reconciliation
-----------------------------------
In the task prompt the generalised RHS is written as

    RHS_Holder = M + 1 + 2 sum z_n^2 k_n
                 + ( M^{1/(p_hy - 1)} - 1 - 2 sum z_n^{2 p_hy'} )^{1/p_hy'}
                 * ( K_q_total - 1 - 2 sum k_n^{q_kernel} )^{1/q_kernel}.

At ``p_hy = 2`` one has ``1/(p_hy - 1) = 1``, ``p_hy' = 2``, and
``2 p_hy' = 4``, so the first factor becomes ``sqrt(M - 1 - 2 sum z_n^4)``
and everything coincides with ``RHS_CS``.  This script verifies that
reduction symbolically, and also prints the generalised form for a
concrete ``N_max = 2``.
"""

from __future__ import annotations

import sympy as sp


def build_expressions(N_max: int = 2):
    """Return a dict of sympy expressions used in the cross-check."""
    # Scalars.
    M = sp.symbols("M", positive=True)
    K2 = sp.symbols("K2", positive=True)           # kept for the CS-case printout
    K_q_total = sp.symbols("K_q_total", positive=True)
    p_hy = sp.symbols("p_hy", positive=True)       # Hausdorff-Young exponent, in [1, 2]
    q_kernel = sp.symbols("q_kernel", positive=True)

    # Conjugate of p_hy.
    p_hy_prime = p_hy / (p_hy - 1)

    # Moment and kernel symbols.
    z = sp.symbols(" ".join(f"z{n}" for n in range(1, N_max + 1)), positive=True)
    k = sp.symbols(" ".join(f"k{n}" for n in range(1, N_max + 1)), positive=True)
    # When N_max == 1 sympy returns a Symbol rather than a tuple; normalise.
    if N_max == 1:
        z = (z,)
        k = (k,)

    # Cauchy-Schwarz MM-10 RHS (Theorem 2).
    sum_zk_CS = sum(z[n] ** 2 * k[n] for n in range(N_max))
    sum_z4_CS = sum(z[n] ** 4 for n in range(N_max))
    sum_kn2_CS = sum(k[n] ** 2 for n in range(N_max))
    RHS_CS = (
        M + 1
        + 2 * sum_zk_CS
        + sp.sqrt(M - 1 - 2 * sum_z4_CS)
        * sp.sqrt(K2 - 1 - 2 * sum_kn2_CS)
    )

    # Generalised Hoelder RHS.
    #
    #   first_factor = ( M^{1/(p_hy - 1)} - 1 - 2 * sum_n z_n^{2 * p_hy'} )^{1/p_hy'}
    #   second_factor = ( K_q_total - 1 - 2 * sum_n k_n^{q_kernel}       )^{1/q_kernel}
    #
    # with 1/p_hy' + 1/q_kernel = 1 (not enforced symbolically here;
    # enforced only at the MM-10 reduction where p_hy = 2 forces
    # p_hy' = q_kernel = 2).
    sum_zk_Holder = sum(z[n] ** 2 * k[n] for n in range(N_max))
    sum_z_to_2pprime = sum(z[n] ** (2 * p_hy_prime) for n in range(N_max))
    sum_kn_to_q = sum(k[n] ** q_kernel for n in range(N_max))

    first_radicand = M ** (1 / (p_hy - 1)) - 1 - 2 * sum_z_to_2pprime
    second_radicand = K_q_total - 1 - 2 * sum_kn_to_q
    RHS_Holder = (
        M + 1
        + 2 * sum_zk_Holder
        + first_radicand ** (1 / p_hy_prime)
        * second_radicand ** (1 / q_kernel)
    )

    return {
        "N_max": N_max,
        "M": M,
        "K2": K2,
        "K_q_total": K_q_total,
        "p_hy": p_hy,
        "p_hy_prime": p_hy_prime,
        "q_kernel": q_kernel,
        "z": z,
        "k": k,
        "RHS_CS": RHS_CS,
        "RHS_Holder": RHS_Holder,
    }


def verify_reduction(ctx: dict) -> sp.Expr:
    """Substitute ``p_hy = 2`` and ``q_kernel = 2`` into ``RHS_Holder``
    and ``K_q_total = K2`` and return sympy.simplify(RHS_Holder_sub - RHS_CS).
    """
    RHS_Holder = ctx["RHS_Holder"]
    RHS_CS = ctx["RHS_CS"]
    p_hy = ctx["p_hy"]
    q_kernel = ctx["q_kernel"]
    K_q_total = ctx["K_q_total"]
    K2 = ctx["K2"]

    # At the CS case p_hy = 2  =>  p_hy' = 2/(2-1) = 2; q_kernel = 2;
    # and the total-l^q-norm symbol ``K_q_total`` equals ``K2`` by definition.
    RHS_Holder_sub = RHS_Holder.subs(
        {p_hy: 2, q_kernel: 2, K_q_total: K2}
    )
    diff = sp.simplify(RHS_Holder_sub - RHS_CS)
    return diff, RHS_Holder_sub


def main() -> int:
    banner = "=" * 72
    print(banner)
    print("Symbolic cross-check: Hoelder-generalised RHS  vs.  MM-10 (MV eq. 10)")
    print(banner)

    ctx = build_expressions(N_max=2)

    print()
    print("[1] Cauchy-Schwarz MM-10 RHS (N_max = 2):")
    print("    RHS_CS =")
    sp.pprint(ctx["RHS_CS"], use_unicode=False)

    print()
    print("[2] Generalised Hoelder RHS (general p_hy, q_kernel):")
    print("    RHS_Holder =")
    sp.pprint(ctx["RHS_Holder"], use_unicode=False)

    print()
    print("[3] Substituting p_hy = 2, q_kernel = 2, K_q_total = K2:")
    diff, RHS_Holder_sub = verify_reduction(ctx)
    print("    RHS_Holder |_{p_hy=2, q_kernel=2, K_q_total=K2} =")
    sp.pprint(sp.simplify(RHS_Holder_sub), use_unicode=False)

    print()
    print("[4] simplify(RHS_Holder_sub - RHS_CS) =")
    sp.pprint(diff, use_unicode=False)

    ok = diff == 0 or sp.simplify(diff) == 0
    print()
    print(f"[5] Reduction verified symbolically: {ok}")

    # Also spot-check a random numeric evaluation at CS case: both sides
    # should agree exactly at any admissible numerical point.
    numeric = {
        ctx["M"]: sp.Rational(6, 5),         # M = 1.2
        ctx["K2"]: sp.Rational(21, 5),        # K2 = 4.2
        ctx["z"][0]: sp.Rational(1, 5),       # z1 = 0.2
        ctx["z"][1]: sp.Rational(1, 10),      # z2 = 0.1
        ctx["k"][0]: sp.Rational(17, 20),     # k1 = 0.85
        ctx["k"][1]: sp.Rational(1, 2),       # k2 = 0.5
    }
    cs_val = ctx["RHS_CS"].subs(numeric)
    holder_val = RHS_Holder_sub.subs(numeric)
    print()
    print("[6] Numeric spot-check at a small test point:")
    print(f"    RHS_CS        = {sp.N(cs_val, 20)}")
    print(f"    RHS_Holder_sub= {sp.N(holder_val, 20)}")
    print(f"    |difference|  = {sp.N(sp.Abs(cs_val - holder_val), 20)}")

    if not ok:
        print("FAIL: symbolic reduction did not simplify to 0.")
        return 1
    print()
    print("PASS: RHS_Holder at (p_hy, q_kernel) = (2, 2) reduces to RHS_CS (MM-10).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
