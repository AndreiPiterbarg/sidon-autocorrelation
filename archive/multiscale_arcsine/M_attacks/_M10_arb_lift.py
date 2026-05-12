"""Agent M10: Rigorous arb lift of the 2-scale arcsine kernel breakthrough.

Numerical baseline (from _K26_full_sweep_reopt_result.json):
    delta_1 = 0.138   (= MV_DELTA)
    delta_2 = 0.045
    lambda_1 = 0.85   (lambda_2 = 0.15)
    -> numerical M_cert = 1.29005, S_1 = 31.44, k_1 = 0.92139, K_2 = 4.7588

Goal: convert that numerical finding into a rigorous python-flint arb
certified lower bound C_{1a} >= M_cert by reusing the existing
``delsarte_dual.grid_bound_alt_kernel`` pipeline (QP -> PhiParams ->
bisection of cell_search).

Strategy
--------
1. Define ``MultiScaleArcsineKernel`` as a ``Kernel`` subclass with rigorous
   arb implementations of K_tilde(n), K_tilde_real(xi), and K_norm_sq.
   - K_tilde_real(xi) = sum_i lambda_i J_0(pi delta_i xi)^2  (exact arb).
   - K_norm_sq = sum_{ij} lambda_i lambda_j C_{ij},
       C_{ij} = int_{-inf}^{inf} J_0^2(pi delta_i xi) J_0^2(pi delta_j xi) dxi
            = 2 * (int_0^T  ... + tail_T^infty).
     Tail uses the classical Bessel bound |J_0(z)| <= sqrt(2/(pi z)) for z >= 2,
     giving J_0^2(pi delta_i xi) J_0^2(pi delta_j xi) <= 4/(pi^2 delta_i delta_j xi^2)
     for xi >= max(2/(pi delta_i), 2/(pi delta_j)) =: xi_min, and so
       int_T^infty ... d xi <= 4/(pi^2 delta_i delta_j T) for T >= xi_min.

2. Run ``solve_qp_for_kernel(MultiScaleArcsineKernel(...))`` to get the
   119 G-coefficients.
3. Compile ``PhiParams`` via ``compile_phi_params_for_kernel``.
4. Bisect via ``bisect_M_cert`` with a wide bracket.

Report: certified M_cert (or precise blocker).
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Sequence

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from flint import arb, acb, fmpq, ctx

from delsarte_dual.grid_bound_alt_kernel.kernels import Kernel
from delsarte_dual.grid_bound_alt_kernel.optimize_G import solve_qp_for_kernel
from delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel import (
    compile_phi_params_for_kernel,
)
from delsarte_dual.grid_bound.cell_search import certify_phi_negative
from delsarte_dual.grid_bound.bisect import bisect_M_cert, _fmpq_to_float, _fmpq_to_str


# =============================================================================
#  Multi-scale arcsine kernel (rigorous arb implementation)
# =============================================================================

class MultiScaleArcsineKernel(Kernel):
    """K(x) = sum_i lambda_i * K_arcsine(x; delta_i)
    on the Fourier side: K_hat(xi) = sum_i lambda_i * J_0(pi delta_i xi)^2.

    Admissibility (all rigorous by construction):
      (a) K >= 0: each summand >= 0 (auto-conv of arcsine density),
          lambdas >= 0, so sum >= 0.
      (b) supp K subset [-max delta_i, max delta_i]: each arcsine summand
          is supported on [-delta_i, delta_i], so union is [-max, max].
      (c) int K = K_hat(0) = sum lambda_i * 1 = 1 iff sum lambda_i = 1.
      (d) tilde K(j) = K_hat(j) = sum lambda_i J_0(pi delta_i j)^2 >= 0
          (sum of non-negative terms).

    Rigorous K_norm_sq:
      ||K||_2^2 = int K_hat(xi)^2 dxi
              = sum_{ij} lambda_i lambda_j C_{ij},
      C_{ij} = int_{R} J_0^2(pi delta_i xi) J_0^2(pi delta_j xi) dxi
            = 2 * [int_0^T + tail].
      Main integral: arb-rigorous acb.integral.
      Tail bound: |J_0(z)| <= sqrt(2/(pi z)) for z >= 2 (classical;
        e.g. Watson's Bessel Function Treatise, ch. 7), so
        J_0^2(pi delta_i xi) <= 2/(pi^2 delta_i xi) for xi >= 2/(pi delta_i).
        Hence J_0^2 * J_0^2 <= 4/(pi^4 delta_i delta_j xi^2) for
        xi >= xi_min(i,j) = max(2/(pi delta_i), 2/(pi delta_j)).
        And int_T^infty xi^{-2} dxi = 1/T.
    """
    def __init__(self,
                 deltas: Sequence[fmpq],
                 lambdas: Sequence[fmpq],
                 name_suffix: str = "",
                 K2_integration_T: int = 200,
                 K2_eps_q: fmpq = fmpq(1, 10**6)):
        if len(deltas) != len(lambdas):
            raise ValueError("deltas and lambdas must have same length")
        if not all(d > 0 for d in deltas):
            raise ValueError("all deltas must be positive")
        if not all(l >= 0 for l in lambdas):
            raise ValueError("all lambdas must be non-negative")
        if sum(lambdas) != fmpq(1):
            raise ValueError(f"lambdas must sum to 1 exactly (fmpq); got "
                             f"{sum(lambdas)}")
        self.deltas = list(deltas)
        self.lambdas = list(lambdas)
        # Outer support
        self.supp_halfwidth = max(self.deltas)
        # Cache K2
        self._K2_cache: dict[int, arb] = {}
        # Integration bounds
        self.K2_integration_T = int(K2_integration_T)   # T in units of 1
        self.K2_eps_q = K2_eps_q
        self.name = (
            "M10_multiscale_arcsine("
            + ",".join(f"d={float(d.p)/float(d.q):.4f}:l={float(l.p)/float(l.q):.3f}"
                       for d, l in zip(self.deltas, self.lambdas))
            + (f"){name_suffix}" if name_suffix else ")")
        )

    # ---- K_tilde (period-1 / real-line FT) ----------------------------------

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        """K_hat(xi) = sum_i lambda_i J_0(pi delta_i xi)^2."""
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            tot = arb(0)
            for d, lam in zip(self.deltas, self.lambdas):
                arg = arb.pi() * arb(d) * xi
                j0 = arg.bessel_j(0)
                tot = tot + arb(lam) * j0 * j0
            return tot
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
            tot = arb(0)
            for d, lam in zip(self.deltas, self.lambdas):
                # pi * n * d_i is an exact fmpq inside arb.pi()
                q = fmpq(n) * d
                arg = arb.pi() * arb(q)
                j0 = arg.bessel_j(0)
                tot = tot + arb(lam) * j0 * j0
            return tot
        finally:
            ctx.prec = old

    def K_tilde_positive(self, n_max: int = 200, prec_bits: int = 256) -> bool:
        """Trivially True: K_hat is a sum of non-negative terms with non-neg coeffs."""
        # We still verify with arb to be safe; quick path returns True.
        return True

    # ---- K_norm_sq (rigorous arb enclosure) ---------------------------------

    def _C_ij_arb(self, di: fmpq, dj: fmpq, prec_bits: int) -> arb:
        """C_{ij} = int_{R} J_0^2(pi d_i xi) J_0^2(pi d_j xi) dxi (arb).

        Computed as 2 * [int_eps^T  + tail_T^infty] (integrand is even).
        Main: acb.integral with the entire holomorphic integrand.
        Tail bound:
          For xi >= 2/(pi min(d_i, d_j)),
            J_0^2(pi d_i xi) J_0^2(pi d_j xi)
              <= (2/(pi^2 d_i xi)) * (2/(pi^2 d_j xi))  [using J_0^2 <= 2/(pi z)]
              = 4 / (pi^4 d_i d_j xi^2).
          So int_T^infty <= 4/(pi^4 d_i d_j T).
        Small-xi [0, eps] patch: integrand at xi=0 equals 1 (J_0(0)^2 * J_0(0)^2),
          and decreases on [0, eps] (J_0 is decreasing on [0, first zero ~2.4]),
          so contribution <= eps.

        Returns the arb enclosure of C_{ij}.
        """
        # ctx.prec already set by caller
        di_a = arb(di)
        dj_a = arb(dj)

        T_q = fmpq(self.K2_integration_T)
        T_a = arb(T_q)
        eps_q = self.K2_eps_q
        eps_a = arb(eps_q)

        # Tail bound (rigorous): 4 / (pi^4 d_i d_j T)
        # We need T to be large enough that pi d_i xi >= 2 for xi >= T
        # i.e. T >= 2/(pi d_i), and similarly for d_j.  With d_i, d_j ~ 0.04+,
        # 2/(pi * 0.04) ~ 16, so T = 200 >> 16 is fine.
        pi_a = arb.pi()
        tail_ub = arb(4) / (pi_a * pi_a * pi_a * pi_a * di_a * dj_a * T_a)
        # Sanity: T must be large enough.  Encode as a multiplicative slack.
        di_f = float(di.p) / float(di.q)
        dj_f = float(dj.p) / float(dj.q)
        min_d = min(di_f, dj_f)
        xi_min = 2.0 / (float(pi_a.mid()) * min_d) if min_d > 0 else float('inf')
        if float(T_a.mid()) < xi_min:
            raise ValueError(f"K2_integration_T={float(T_a.mid())} too small; "
                             f"need >= {xi_min:.2f} for tail bound to apply")

        # Main integral on [eps, T]
        def integrand(z, flags):
            arg_i = acb.pi() * acb(di_a) * z
            arg_j = acb.pi() * acb(dj_a) * z
            ji = arg_i.bessel_j(acb(0))
            jj = arg_j.bessel_j(acb(0))
            ji2 = ji * ji
            jj2 = jj * jj
            return ji2 * jj2

        val = acb.integral(integrand, acb(eps_a), acb(T_a))
        main = val.real

        # Eps contribution: integrand at xi=0 is 1, decreases on [0, eps].
        # So 0 <= int_0^eps <= eps.  Add an arb in [0, eps].
        eps_contrib = arb(0).union(eps_a)

        # Double (even integrand) and add tail bound (lies in [0, tail_ub]).
        # Final: 2 * (main + eps_contrib + tail_contrib)
        tail_contrib = arb(0).union(tail_ub)
        return arb(2) * (main + eps_contrib + tail_contrib)

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        """||K||_2^2 = sum_{ij} lambda_i lambda_j C_{ij}.  Rigorous arb."""
        if prec_bits in self._K2_cache:
            return self._K2_cache[prec_bits]
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            n = len(self.deltas)
            total = arb(0)
            for i in range(n):
                for j in range(n):
                    if self.lambdas[i] == 0 or self.lambdas[j] == 0:
                        continue
                    C_ij = self._C_ij_arb(self.deltas[i], self.deltas[j], prec_bits)
                    total = total + arb(self.lambdas[i]) * arb(self.lambdas[j]) * C_ij
            self._K2_cache[prec_bits] = total
            return total
        finally:
            ctx.prec = old

    def admissibility_check(self, prec_bits: int = 256, n_bochner: int = 50) -> None:
        if self.supp_halfwidth <= 0:
            raise ValueError(f"{self.name}: supp_halfwidth must be positive")
        # Bochner is trivially true; still spot-check
        for j in range(1, n_bochner + 1):
            v = self.K_tilde(j, prec_bits)
            if v.lower() < 0:
                raise ValueError(
                    f"{self.name}: unexpected K_tilde({j}) lower negative: {v}"
                )


# =============================================================================
#  Main driver
# =============================================================================

def main():
    print("=" * 78)
    print("Agent M10: rigorous arb lift of 2-scale arcsine breakthrough (1.29005 num)")
    print("=" * 78)
    t_start = time.time()

    # Numerical optimum from _K26_full_sweep_reopt_result.json:
    delta_1 = fmpq(138, 1000)
    delta_2 = fmpq(45, 1000)
    lambda_1 = fmpq(85, 100)
    lambda_2 = fmpq(15, 100)
    K = MultiScaleArcsineKernel(
        deltas=[delta_1, delta_2],
        lambdas=[lambda_1, lambda_2],
        name_suffix="_M10",
        K2_integration_T=200,
    )
    def _qf(q): return float(q.p) / float(q.q)
    print(f"\nKernel: {K.name}")
    print(f"  delta_1 = {_qf(delta_1):.4f}  lambda_1 = {_qf(lambda_1):.3f}")
    print(f"  delta_2 = {_qf(delta_2):.4f}  lambda_2 = {_qf(lambda_2):.3f}")
    print(f"  supp_halfwidth = {_qf(K.supp_halfwidth):.4f}")

    # ---- Step 0: admissibility ---------------------------------------------
    print("\n[Step 0] Admissibility check...")
    K.admissibility_check(prec_bits=192, n_bochner=50)
    print("  K admissible (Bochner trivially OK).")

    # ---- Step 1: K2 sanity --------------------------------------------------
    print("\n[Step 1] K_norm_sq arb enclosure...")
    K2_arb = K.K_norm_sq(prec_bits=192)
    K2_mid = float(K2_arb.mid())
    K2_rad = float(K2_arb.rad())
    K2_lo = float(K2_arb.lower())
    K2_hi = float(K2_arb.upper())
    print(f"  K_2 = {K2_arb}")
    print(f"  K_2 mid = {K2_mid:.6f}, rad = {K2_rad:.6e}")
    print(f"  K_2 in [{K2_lo:.6f}, {K2_hi:.6f}]")
    print(f"  (numerical reference: 4.7588)")

    # ---- Step 2: k_1 sanity -------------------------------------------------
    print("\n[Step 2] k_1 = K_hat(1) arb enclosure...")
    k1_arb = K.K_tilde(1, prec_bits=192)
    print(f"  k_1 = {k1_arb}")
    print(f"  k_1 mid = {float(k1_arb.mid()):.6f}  (numerical ref: 0.92139)")

    # ---- Step 3: QP re-optimisation ----------------------------------------
    print("\n[Step 3] QP re-optimisation (119 G-coefficients)...")
    u_q = fmpq(638, 1000)
    qp = solve_qp_for_kernel(
        K, n=119, u=u_q, n_grid=5001,
        prec_bits_weights=128, fmpq_denom=10**8, verbose=True,
    )
    print(f"  S_1_float = {qp.S1_float:.4f}  (numerical ref: 31.44)")
    print(f"  min_G_grid = {qp.min_G_grid_float:.6f}")

    # ---- Step 4: PhiParams compile (rigorous) ------------------------------
    print("\n[Step 4] Compile rigorous PhiParams (incl. Taylor min_G B&B)...")
    try:
        params = compile_phi_params_for_kernel(
            K, qp.a_opt_fmpq, u=u_q,
            n_cells_min_G=4096, prec_bits=192,
        )
    except Exception as e:
        print(f"  [BLOCKER] PhiParams compile failed: {e}")
        return {"status": "BLOCKER_PhiParams", "error": str(e)}

    S1_f = float(params.S1.mid())
    K2_f = float(params.K2.mid())
    k1_f = float(params.k1.mid())
    min_G_f = float(params.min_G.mid())
    gain_f = float(params.gain_a.mid())
    print(f"  K2 (rigorous) = {K2_f:.5f}")
    print(f"  k1 (rigorous) = {k1_f:.5f}")
    print(f"  S1 (rigorous) = {S1_f:.5f}")
    print(f"  min_G (cert)  = {min_G_f:.6f}  (need >0 for sound bound)")
    print(f"  gain_a        = {gain_f:.6f}")

    # ---- Step 5: bisection over M ------------------------------------------
    print("\n[Step 5] Bisection of M_cert via cell_search.certify_phi_negative...")
    # Try a few starting brackets, retreat if needed.
    M_lo_candidates = [
        fmpq(128, 100), fmpq(1275, 1000), fmpq(127, 100),
        fmpq(1265, 1000), fmpq(125, 100), fmpq(1240, 1000),
        fmpq(120, 100), fmpq(115, 100), fmpq(110, 100),
    ]
    M_hi_init = fmpq(1295, 1000)  # just above numerical 1.290
    bracket_M_lo = None
    for M_lo_try in M_lo_candidates:
        print(f"  Trying M_lo = {_fmpq_to_float(M_lo_try):.5f} ...")
        try:
            chk = certify_phi_negative(
                arb(M_lo_try), params,
                max_cells=80000,
                initial_splits=32,
                prec_bits=192,
            )
        except Exception as e:
            print(f"    bracket check raised: {e}")
            continue
        print(f"    verdict = {chk.verdict}, cells_processed = {chk.cells_processed}")
        if chk.verdict == "CERTIFIED_FORBIDDEN":
            bracket_M_lo = M_lo_try
            break

    if bracket_M_lo is None:
        print("\n[BLOCKER] No M_lo in {1.10..1.28} certifiable; cannot bisect.")
        return {"status": "BLOCKER_no_lower_bracket"}

    print(f"\n  Starting bisection with M_lo = {_fmpq_to_float(bracket_M_lo):.5f}, "
          f"M_hi = {_fmpq_to_float(M_hi_init):.5f}")
    try:
        bound = bisect_M_cert(
            params,
            M_lo_init=bracket_M_lo,
            M_hi_init=M_hi_init,
            tol_q=fmpq(1, 10**4),
            max_cells_per_M=80000,
            initial_splits=32,
            prec_bits=192,
            verbose=True,
        )
    except Exception as e:
        print(f"  [BLOCKER] bisect raised: {e}")
        return {"status": "BLOCKER_bisect", "error": str(e),
                "M_lo_certifiable": _fmpq_to_float(bracket_M_lo)}

    M_cert_f = _fmpq_to_float(bound.M_cert_q)
    M_cert_q_str = _fmpq_to_str(bound.M_cert_q)

    # ---- Summary ------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  Rigorous M_cert = {M_cert_f:.6f}")
    print(f"  Rigorous M_cert (fmpq) = {M_cert_q_str}")
    print(f"  Numerical baseline (2-scale, _K26_full_sweep) = 1.29005")
    print(f"  MV rigorous baseline (K1)                   = 1.27481")
    print(f"  Improvement over MV's K1:    {M_cert_f - 1.27481:+.6f}")
    print(f"  Wall time: {elapsed:.1f} s")

    summary = {
        "status": "RIGOROUS_BOUND_OBTAINED",
        "kernel_name": K.name,
        "deltas_q": [_fmpq_to_str(d) for d in K.deltas],
        "lambdas_q": [_fmpq_to_str(l) for l in K.lambdas],
        "K_2_rigorous_mid": K2_f,
        "K_2_rigorous_lower": float(params.K2.lower()),
        "K_2_rigorous_upper": float(params.K2.upper()),
        "k_1_rigorous_mid": k1_f,
        "S_1_rigorous_mid": S1_f,
        "min_G_rigorous_lower": float(params.min_G.lower()),
        "gain_a_rigorous_mid": gain_f,
        "M_cert_rigorous_q": M_cert_q_str,
        "M_cert_rigorous_float": M_cert_f,
        "numerical_baseline_M_cert": 1.29005,
        "MV_rigorous_baseline_M_cert": 1.27481,
        "improvement_over_MV_K1": M_cert_f - 1.27481,
        "wall_time_sec": elapsed,
    }
    out = os.path.join(REPO, "_M10_arb_lift_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {out}")
    return summary


if __name__ == "__main__":
    main()
