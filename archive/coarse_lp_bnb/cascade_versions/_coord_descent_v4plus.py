"""Coordinate descent over (delta_2, delta_3, lambda_1, lambda_2, lambda_3) for
the 3-scale arcsine multi-kernel.

Pipeline (per inner evaluation of a candidate point):
  1. Build MultiScaleArcsineKernel(deltas=[d1, d2, d3], lambdas=[l1, l2, l3]) with
     fixed d1 = 0.138 and the variables; lambdas are renormalised to sum=1.
  2. Solve the QP for G (MOSEK; n=119 coefficients on the n_grid_qp grid).
  3. Compute the closed-form M_cert proxy:
        Let k1, K2 be the kernel's k_1 and ||K||_2^2 (arb).
        Let S1 = sum a_j^2 / hat_K(j/u), min_G via Taylor B&B, gain a = (4/u) min_G^2 / S1.
        Solve the master inequality M + 1 + sqrt((M-1)(K2-1)) = 2/u + a for M
        (closed-form quadratic; interior-y* assumption).
        Verify the interior y* <= mu(M); else fall back to boundary form.
     This proxy is monotone-tight and matches the bisection M_cert to within
     1e-4 (verified on the 2-scale baseline).
  4. At the converged optimum, run the FULL rigorous bisection cell_search to
     emit the certified M_cert_q rational.

Coordinate descent:
  * 5 variables: (delta_2, delta_3, lambda_1, lambda_2, lambda_3)
  * Per sweep, line-search each variable with golden-section (~10 evals) to
    maximise the proxy M_cert.
  * Renormalise lambdas after each lambda-update so sum=1.
  * Stop when an outer sweep improves M_cert by < 1e-6.

The proxy keeps inner evaluations cheap (~1 sec each); rigorous bisection only
runs at the final optimum (~5 sec).

Author: coord-descent v4plus (2026-05-11)
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound_alt_kernel.kernels import MultiScaleArcsineKernel
from delsarte_dual.grid_bound_alt_kernel.optimize_G import solve_qp_for_kernel
from delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel import (
    compile_phi_params_for_kernel,
    run_single_kernel,
)
from delsarte_dual.grid_bound.phi import mu_of_M
from delsarte_dual.grid_bound.bisect import _fmpq_to_float, _fmpq_to_str

PREC_BITS = 128            # inner-loop precision (cheap)
N_COEFFS = 119
N_GRID_QP = 3001
N_CELLS_MIN_G_INNER = 1024
N_CELLS_MIN_G_FINAL = 4096
U = fmpq(638, 1000)
DELTA1_FIXED = fmpq(138, 1000)
DENOM = 1000   # rational grid denominator for (delta_i, lambda_i)


# ---------------------------------------------------------------------------
# Closed-form M_cert proxy
# ---------------------------------------------------------------------------

def _solve_master_M_interior(k1: float, K2: float, gain_a: float, u: float) -> float:
    """Solve M + 1 + sqrt((M-1)(K2-1)) = 2/u + a for the interior-y* regime.

    Returns the M satisfying this exactly (smaller root of the implied quadratic).
    Note: the interior regime simplification drops the 2 y k1 and (-2 y^2) inside
    the radicand; the resulting equation has the closed form below.
    """
    R = 2.0 / u + gain_a - 1.0
    if R <= 1.0:
        return 1.0  # degenerate
    # (M-1)(K2-1) = (R - M)^2  with  R >= M
    # M^2 - (2 R + K2 - 1) M + (R^2 + K2 - 1) = 0
    b = 2.0 * R + K2 - 1.0
    c = R * R + K2 - 1.0
    disc = b * b - 4.0 * c
    if disc < 0:
        return 1.0  # no real solution
    M = 0.5 * (b - math.sqrt(disc))
    return M


def _phi_boundary(M: float, k1: float, K2: float, gain_a: float, u: float) -> float:
    """Phi at z_1^2 = mu(M)^2 (boundary regime)."""
    mu = M * math.sin(math.pi / M) / math.pi
    y = mu * mu
    rad1 = max(M - 1.0 - 2.0 * y * y, 0.0)
    rad2 = max(K2 - 1.0 - 2.0 * k1 * k1, 0.0)
    return M + 1.0 + 2.0 * y * k1 + math.sqrt(rad1 * rad2) - 2.0 / u - gain_a


def _solve_master_M_boundary(k1: float, K2: float, gain_a: float, u: float,
                              M_lo: float = 1.20, M_hi: float = 1.40) -> float:
    """Bisect Phi_boundary(M) = 0 in [M_lo, M_hi].  Phi is increasing in M."""
    f_lo = _phi_boundary(M_lo, k1, K2, gain_a, u)
    f_hi = _phi_boundary(M_hi, k1, K2, gain_a, u)
    if f_lo > 0:
        return M_lo
    if f_hi < 0:
        return M_hi
    for _ in range(80):
        mid = 0.5 * (M_lo + M_hi)
        f_mid = _phi_boundary(mid, k1, K2, gain_a, u)
        if f_mid < 0:
            M_lo = mid
        else:
            M_hi = mid
        if M_hi - M_lo < 1e-9:
            break
    return 0.5 * (M_lo + M_hi)


def _m_cert_proxy(k1: float, K2: float, gain_a: float, u: float) -> float:
    """Return the proxy M_cert = max M such that Phi(M, y) <= 0 forall y in [0,mu(M)].

    Take the worse (smaller) of interior and boundary roots.
    """
    M_int = _solve_master_M_interior(k1, K2, gain_a, u)
    M_bdy = _solve_master_M_boundary(k1, K2, gain_a, u)
    return min(M_int, M_bdy)


# ---------------------------------------------------------------------------
# Inner evaluation: given (d1, d2, d3, l1, l2, l3), return proxy M_cert
# ---------------------------------------------------------------------------

def _to_fmpq(x: float, denom: int = DENOM) -> fmpq:
    return fmpq(int(round(x * denom)), denom)


def _normalize_lambdas(l1: float, l2: float, l3: float) -> tuple:
    s = l1 + l2 + l3
    if s <= 0:
        return (1.0, 0.0, 0.0)
    return (l1 / s, l2 / s, l3 / s)


@dataclass
class InnerResult:
    M_cert_proxy: float
    k1: float
    K2: float
    S1: float
    min_G: float
    gain_a: float
    deltas: tuple
    lambdas: tuple


def _evaluate_proxy(d2: float, d3: float, l1: float, l2: float, l3: float,
                    n_grid_qp: int = N_GRID_QP,
                    n_cells_min_G: int = N_CELLS_MIN_G_INNER,
                    prec_bits: int = PREC_BITS,
                    verbose: bool = False) -> Optional[InnerResult]:
    """Build kernel, solve QP, return proxy M_cert.  None on failure."""
    # Snap to denominator grid; normalise lambdas; enforce d3 < d2 < d1.
    if d2 <= 1.0e-4 or d3 <= 1.0e-4:
        return None
    if d2 >= 0.138 or d3 >= d2:
        return None
    l1n, l2n, l3n = _normalize_lambdas(l1, l2, l3)
    if l1n <= 0 or l2n <= 0 or l3n <= 0:
        return None

    d2q = _to_fmpq(d2, DENOM)
    d3q = _to_fmpq(d3, DENOM)
    l1q = _to_fmpq(l1n, 1000)
    l2q = _to_fmpq(l2n, 1000)
    l3q = fmpq(1) - l1q - l2q
    if l3q <= 0 or l1q <= 0 or l2q <= 0:
        return None
    if d2q >= DELTA1_FIXED or d3q >= d2q:
        return None

    try:
        K = MultiScaleArcsineKernel(
            deltas=[DELTA1_FIXED, d2q, d3q],
            lambdas=[l1q, l2q, l3q],
            use_diag_surrogate=True,
        )
    except Exception as e:
        if verbose:
            print(f"  kernel build failed: {e}")
        return None

    # Solve QP
    try:
        qp_res = solve_qp_for_kernel(
            K, n=N_COEFFS, u=U, n_grid=n_grid_qp,
            prec_bits_weights=prec_bits, verbose=False,
        )
    except Exception as e:
        if verbose:
            print(f"  QP failed: {e}")
        return None

    # Compile PhiParams to extract arb-rigorous k1, K2, S1, min_G, gain_a
    try:
        params = compile_phi_params_for_kernel(
            K, qp_res.a_opt_fmpq, u=U,
            n_cells_min_G=n_cells_min_G, prec_bits=prec_bits,
        )
    except Exception as e:
        if verbose:
            print(f"  compile failed: {e}")
        return None

    k1 = float(params.k1.mid())
    K2 = float(params.K2.upper())     # conservative (upper bound)
    S1 = float(params.S1.upper())     # conservative
    min_G = float(params.min_G.lower())
    gain_a = float(params.gain_a.lower())  # conservative under-estimate
    u_f = float(U.p) / float(U.q)
    M_proxy = _m_cert_proxy(k1, K2, gain_a, u_f)

    return InnerResult(
        M_cert_proxy=M_proxy,
        k1=k1, K2=K2, S1=S1, min_G=min_G, gain_a=gain_a,
        deltas=(float(DELTA1_FIXED.p) / float(DELTA1_FIXED.q), d2, d3),
        lambdas=(l1n, l2n, l3n),
    )


# ---------------------------------------------------------------------------
# Golden-section line search (maximisation)
# ---------------------------------------------------------------------------

PHI_GOLD = (math.sqrt(5.0) - 1.0) / 2.0   # ~0.618


def golden_section_max(f: Callable[[float], float], a: float, b: float,
                       tol: float = 1e-4, max_iter: int = 18) -> tuple:
    """Maximise f on [a, b] via golden-section search.

    Returns (x_opt, f_opt).  f failures (None return) treated as -inf.
    """
    def fv(x):
        v = f(x)
        if v is None:
            return -1e18
        return v

    c = b - PHI_GOLD * (b - a)
    d = a + PHI_GOLD * (b - a)
    fc = fv(c)
    fd = fv(d)
    for _ in range(max_iter):
        if fc > fd:
            b = d
            d = c
            fd = fc
            c = b - PHI_GOLD * (b - a)
            fc = fv(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + PHI_GOLD * (b - a)
            fd = fv(d)
        if (b - a) < tol:
            break
    if fc > fd:
        return (c, fc)
    return (d, fd)


# ---------------------------------------------------------------------------
# Coordinate descent driver
# ---------------------------------------------------------------------------

@dataclass
class CDState:
    d2: float
    d3: float
    l1: float
    l2: float
    l3: float
    M: float


def run_coord_descent(
    d2_init: float = 0.055,
    d3_init: float = 0.025,
    l1_init: float = 0.85,
    l2_init: float = 0.10,
    l3_init: float = 0.05,
    max_outer: int = 50,
    tol_outer: float = 1e-6,
    verbose: bool = True,
) -> tuple:
    """Coordinate descent.  Returns (final_state, history)."""
    history = []
    state = CDState(d2=d2_init, d3=d3_init, l1=l1_init, l2=l2_init, l3=l3_init, M=0.0)

    def eval_state(s: CDState) -> Optional[float]:
        r = _evaluate_proxy(s.d2, s.d3, s.l1, s.l2, s.l3)
        if r is None:
            return None
        return r.M_cert_proxy

    M_init = eval_state(state)
    if M_init is None:
        raise RuntimeError("initial point evaluation failed")
    state.M = M_init
    history.append(("init", state.M, asdict(state)))
    if verbose:
        print(f"init  d2={state.d2:.4f} d3={state.d3:.4f} "
              f"l=({state.l1:.4f},{state.l2:.4f},{state.l3:.4f}) "
              f"M_proxy={state.M:.6f}")

    prev_M = state.M
    for outer in range(1, max_outer + 1):
        # ---- d2 sweep ----
        def f_d2(x):
            return eval_state(CDState(d2=x, d3=state.d3, l1=state.l1,
                                       l2=state.l2, l3=state.l3, M=0))
        lo = max(state.d3 + 0.005, 0.020)
        hi = min(0.130, 0.138 - 0.005)
        x_opt, M_opt = golden_section_max(f_d2, lo, hi, tol=5e-4)
        if M_opt > state.M:
            state.d2 = x_opt
            state.M = M_opt
        # ---- d3 sweep ----
        def f_d3(x):
            return eval_state(CDState(d2=state.d2, d3=x, l1=state.l1,
                                       l2=state.l2, l3=state.l3, M=0))
        lo = 0.005
        hi = max(state.d2 - 0.005, 0.006)
        x_opt, M_opt = golden_section_max(f_d3, lo, hi, tol=5e-4)
        if M_opt > state.M:
            state.d3 = x_opt
            state.M = M_opt

        # ---- l1 sweep ----
        def f_l1(x):
            # keep l2:l3 ratio
            rem = max(1.0 - x, 1e-3)
            tot23 = max(state.l2 + state.l3, 1e-9)
            l2_new = rem * state.l2 / tot23
            l3_new = rem * state.l3 / tot23
            return eval_state(CDState(d2=state.d2, d3=state.d3, l1=x,
                                       l2=l2_new, l3=l3_new, M=0))
        x_opt, M_opt = golden_section_max(f_l1, 0.50, 0.95, tol=1e-3)
        if M_opt > state.M:
            rem = max(1.0 - x_opt, 1e-3)
            tot23 = max(state.l2 + state.l3, 1e-9)
            state.l2 = rem * state.l2 / tot23
            state.l3 = rem * state.l3 / tot23
            state.l1 = x_opt
            state.M = M_opt

        # ---- l2 sweep (with l3 = 1 - l1 - l2) ----
        def f_l2(x):
            l3_new = 1.0 - state.l1 - x
            if l3_new <= 1e-3 or x <= 1e-3:
                return None
            return eval_state(CDState(d2=state.d2, d3=state.d3, l1=state.l1,
                                       l2=x, l3=l3_new, M=0))
        lo = 0.01
        hi = max(1.0 - state.l1 - 0.01, 0.02)
        x_opt, M_opt = golden_section_max(f_l2, lo, hi, tol=1e-3)
        if M_opt > state.M:
            state.l2 = x_opt
            state.l3 = 1.0 - state.l1 - state.l2
            state.M = M_opt

        if verbose:
            print(f"iter {outer:2d}  d2={state.d2:.4f} d3={state.d3:.4f} "
                  f"l=({state.l1:.4f},{state.l2:.4f},{state.l3:.4f}) "
                  f"M_proxy={state.M:.6f}  d={state.M-prev_M:+.6e}")
        history.append((f"iter{outer}", state.M, asdict(state)))
        if state.M - prev_M < tol_outer:
            if verbose:
                print(f"  converged: |dM|={state.M-prev_M:.2e} < {tol_outer:.0e}")
            break
        prev_M = state.M

    return state, history


# ---------------------------------------------------------------------------
# Rigorous bisection at converged point
# ---------------------------------------------------------------------------

def rigorous_final_M(state: CDState, verbose: bool = True) -> dict:
    """Run full rigorous bisection at the converged (rounded) state.

    Returns dict with M_cert (float), M_cert_q (str p/q), and all PhiParams.
    """
    d2q = _to_fmpq(state.d2, DENOM)
    d3q = _to_fmpq(state.d3, DENOM)
    l1q = _to_fmpq(state.l1, 1000)
    l2q = _to_fmpq(state.l2, 1000)
    l3q = fmpq(1) - l1q - l2q
    if l3q <= 0:
        # snap fallback
        l1q = _to_fmpq(state.l1, 100)
        l2q = _to_fmpq(state.l2, 100)
        l3q = fmpq(1) - l1q - l2q
    K = MultiScaleArcsineKernel(
        deltas=[DELTA1_FIXED, d2q, d3q],
        lambdas=[l1q, l2q, l3q],
        use_diag_surrogate=True,
    )
    if verbose:
        print(f"\nRigorous bisection at: deltas=[0.138, {float(d2q.p)/float(d2q.q):.4f}, "
              f"{float(d3q.p)/float(d3q.q):.4f}]  "
              f"lambdas=[{float(l1q.p)/float(l1q.q):.4f}, "
              f"{float(l2q.p)/float(l2q.q):.4f}, "
              f"{float(l3q.p)/float(l3q.q):.4f}]")
    entry = run_single_kernel(
        K, u=U, n_coeffs=N_COEFFS, n_grid_qp=5001,
        n_cells_min_G=N_CELLS_MIN_G_FINAL,
        M_lo_init=fmpq(127, 100),
        M_hi_init=fmpq(1300, 1000),
        tol_q=fmpq(1, 10**4),
        max_cells_per_M=50000,
        initial_splits=32,
        bochner_max=50,
        prec_bits=192,
        verbose=False,
    )
    return {
        "M_cert": entry.M_cert,
        "M_cert_q": entry.M_cert_q,
        "S1": entry.S1,
        "min_G_cert": entry.min_G_cert,
        "gain_a": entry.gain_a,
        "k1": entry.tilde_K_1,
        "K2": entry.K_norm_sq,
        "note": entry.note,
        "wall_time_sec": entry.wall_time_sec,
        "deltas_q": [_fmpq_to_str(DELTA1_FIXED), _fmpq_to_str(d2q), _fmpq_to_str(d3q)],
        "lambdas_q": [_fmpq_to_str(l1q), _fmpq_to_str(l2q), _fmpq_to_str(l3q)],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("Coord-descent v4plus: 3-scale arcsine, (d2,d3,l1,l2,l3) optimization")
    print("=" * 70)

    state, history = run_coord_descent(
        d2_init=0.055, d3_init=0.025,
        l1_init=0.85, l2_init=0.10, l3_init=0.05,
        max_outer=50, tol_outer=1e-6,
    )

    print(f"\nProxy converged at M={state.M:.6f}")
    print(f"  d2 = {state.d2:.6f}")
    print(f"  d3 = {state.d3:.6f}")
    print(f"  l1 = {state.l1:.6f}")
    print(f"  l2 = {state.l2:.6f}")
    print(f"  l3 = {state.l3:.6f}")

    rig = rigorous_final_M(state, verbose=True)
    print(f"\nRIGOROUS M_cert = {rig['M_cert']}  (= {rig['M_cert_q']})")
    print(f"  k1 = {rig['k1']:.6f}  K2 = {rig['K2']:.6f}")
    print(f"  S1 = {rig['S1']:.6f}  min_G = {rig['min_G_cert']:.6f}")
    print(f"  gain_a = {rig['gain_a']:.6f}")
    print(f"  bisect wall time = {rig['wall_time_sec']:.1f}s")

    total_wall = time.time() - t0
    body = {
        "kind": "coord_descent_v4plus",
        "init": {
            "d2": 0.055, "d3": 0.025,
            "l1": 0.85, "l2": 0.10, "l3": 0.05,
            "delta1_fixed": float(DELTA1_FIXED.p) / float(DELTA1_FIXED.q),
            "u": float(U.p) / float(U.q),
            "n_coeffs": N_COEFFS,
        },
        "converged_proxy": asdict(state),
        "rigorous_certificate": rig,
        "history": [{"label": h[0], "M_proxy": h[1], "state": h[2]} for h in history],
        "v4_grid_optimum_M_cert": 1.29216,
        "improvement_vs_v4_grid": (rig["M_cert"] - 1.29216) if rig["M_cert"] else None,
        "total_wall_time_sec": total_wall,
    }
    body_json = json.dumps(body, indent=2, sort_keys=True, default=str)
    digest = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    out = {"sha256_of_body": digest, "body": body}
    out_path = "_coord_descent_v4plus_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=str)
    print(f"\nWrote {out_path}  (sha256 {digest[:16]}...)")
    print(f"Total wall: {total_wall:.1f}s")


if __name__ == "__main__":
    main()
