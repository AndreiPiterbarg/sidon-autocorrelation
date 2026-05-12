"""Atomic-ν Lasserre SDP — single convex SDP lower bound on val(d), and
hence on C_{1a}.

Mathematical reformulation
--------------------------
For any probability measure ν = Σ_i w_i δ_{t_i} on [-1/2, 1/2] and any
admissible f (nonneg, supp ⊂ [-1/4,1/4], ∫f = 1):

    ‖f*f‖_∞  ≥  Σ_i w_i (f*f)(t_i)    since max ≥ conv. combination,

so
    C_{1a}  ≥  inf_{f admissible}  Σ_i w_i (f*f)(t_i)  =:  λ(ν).

On the d-bin piecewise-constant relaxation, each shift t_i is represented
by a window W_i = (ell, s_lo) whose matrix M_{W_i} encodes an averaged
autocorrelation value.  Weighting gives the aggregated matrix

    M_ν := Σ_i w_i M_{W_i}

and the discrete bound

    λ_d(ν) := min_{μ ∈ Δ_d} μ^T M_ν μ  ≤  val(d)  ≤  C_{1a}.

The order-k Lasserre SDP of λ_d(ν) is a SINGLE convex SDP (no max over W,
no bisection) whose dual gives a rigorous lower bound on λ_d(ν) once we
round the float dual to exact rationals via the existing
certified_lasserre.round_repair / certify pipeline.

This module is a THIN adapter over certified_lasserre.build_sdp.build_sdp_data,
which already accepts `lam` = window weights and assembles exactly M_ν.  New
here:

  * ν-point → window-weight projection (project_to_windows).
  * Three ν-design strategies (uniform_grid_nu, peak_concentrated_nu,
    adaptive_from_solution).
  * solve_atomic_nu_sdp: returns a dataclass with the numerical bound,
    solver diagnostics, and (optionally) the rigorous rational bound.

No new SDP mathematics is introduced — the correctness of λ_d(ν) ≤ val(d)
follows from the existing Lasserre soundness proof applied to the fixed
linear objective c^T y with c = vec(M_ν).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from lasserre.core import build_window_matrices
from certified_lasserre.build_sdp import build_sdp_data, SDPData
from certified_lasserre.bm_solver import solve_primal_dual, SolverResult


# =====================================================================
# Window ↔ shift geometry
# =====================================================================

def window_shift(ell: int, s_lo: int, d: int) -> float:
    """Center shift t ∈ [-1/2, 1/2] of window (ell, s_lo) for a d-bin grid.

    Bin centers are c_i = -1/4 + (i + 1/2) / (2d) for i ∈ [0, d-1], so the
    autocorrelation shift at conv-index k = i+j is

        t_k = c_i + c_j = (k + 1 - d) / (2d),     k ∈ [0, 2d-2].

    Window (ell, s_lo) averages conv-entries k ∈ [s_lo, s_lo+ell-2] (length
    ell-1), so its center shift is the average of t_k over that range.
    """
    k_mean = s_lo + (ell - 2) / 2.0
    return (k_mean + 1 - d) / (2.0 * d)


def list_windows(d: int) -> List[Tuple[int, int]]:
    """All (ell, s_lo) tuples in the same order as build_window_matrices(d)."""
    windows, _ = build_window_matrices(d)
    return windows


def windows_of_length(d: int, ell: int) -> List[Tuple[int, Tuple[int, int]]]:
    """[(global_win_index, (ell, s_lo)), …] for windows of a fixed length ell.

    Useful for placing atomic ν-points onto length-ell ("point"-like) windows.
    Each point t_i is then mapped to the single nearest window of this length.
    """
    if ell < 2 or ell > 2 * d:
        raise ValueError(f"ell={ell} out of range [2, {2*d}]")
    out = []
    for w_idx, (e, s) in enumerate(list_windows(d)):
        if e == ell:
            out.append((w_idx, (e, s)))
    return out


# =====================================================================
# ν-point → window-weight projection
# =====================================================================

def project_to_windows(nu_points: Sequence[float],
                       nu_weights: Sequence[float],
                       d: int,
                       *,
                       ell: Union[int, Sequence[int], str] = 2) -> np.ndarray:
    """Project continuous (t_i, w_i) pairs onto the window-weight vector.

    For each (t_i, w_i): find, for every ell in the requested list, the
    length-ell window whose center shift is closest to t_i, and accumulate
    w_i / |ell_list| onto that window.  When ell is a single int, this
    reduces to the original single-length behavior.

    Returns lam: np.ndarray of length n_win (matching build_window_matrices
    output order), entries >= 0, sum = sum(nu_weights).  Normalization is
    left to the caller; build_sdp_data will re-normalize anyway.

    Parameters
    ----------
    ell : int | Sequence[int] | "all"
        - int (default 2): project to a single window length.  ell=2 is the
          most localized ("atomic") choice — one window per conv-index.
        - Sequence[int]: spread each point's weight equally across one
          nearest window of each listed length.  Useful when val(d) needs
          contributions from both short (localized) and long (averaged)
          windows — the min-max formulation uses all lengths in {2,…,2d}.
        - "all": shortcut for list(range(2, 2*d+1)).
    """
    nu_points = np.asarray(nu_points, dtype=np.float64)
    nu_weights = np.asarray(nu_weights, dtype=np.float64)
    if nu_points.shape != nu_weights.shape:
        raise ValueError(f"nu_points {nu_points.shape} vs nu_weights {nu_weights.shape}")
    if nu_points.size == 0:
        raise ValueError("empty nu (no points) — cannot project")
    if np.any(nu_weights < 0):
        raise ValueError("nu_weights must be nonneg")
    if nu_weights.sum() <= 0:
        raise ValueError("nu_weights sum to zero")

    # Resolve ell argument to a concrete list of window lengths.
    if isinstance(ell, str):
        if ell == "all":
            ell_list = list(range(2, 2 * d + 1))
        else:
            raise ValueError(f"ell={ell!r}; only 'all' is a valid string")
    elif isinstance(ell, (int, np.integer)):
        ell_list = [int(ell)]
    else:
        ell_list = [int(e) for e in ell]
        if not ell_list:
            raise ValueError("ell list is empty")

    n_win = len(list_windows(d))
    lam = np.zeros(n_win, dtype=np.float64)

    # Precompute shifts and global indices for each ell in the list.
    shift_table: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for e in ell_list:
        if e < 2 or e > 2 * d:
            raise ValueError(f"ell={e} out of range [2, {2*d}]")
        ew = windows_of_length(d, e)
        if not ew:
            raise ValueError(f"no windows of length ell={e} at d={d}")
        shift_table[e] = (
            np.array([window_shift(e, s, d) for (_, (_, s)) in ew]),
            np.array([gi for (gi, _) in ew]),
        )

    share = 1.0 / len(ell_list)
    for t, w in zip(nu_points, nu_weights):
        if w == 0.0:
            continue
        for e in ell_list:
            shifts, gidx = shift_table[e]
            j = int(np.argmin(np.abs(shifts - t)))
            lam[gidx[j]] += float(w) * share
    return lam


# =====================================================================
# ν-design strategies
# =====================================================================

def uniform_grid_nu(d: int, K: int,
                    *,
                    ell: Union[int, Sequence[int], str] = 2,
                    interval: Tuple[float, float] = (-0.5, 0.5)) -> np.ndarray:
    """Uniform ν on K points equispaced in `interval`.

    With ell=2 and K > 2d-1, multiple ν-points map to the same window, so
    the effective support size is capped at min(K, #length-ell windows).
    Pass ell="all" or a list of ells to spread mass across window lengths.
    """
    lo, hi = interval
    if K <= 0:
        raise ValueError("K must be positive")
    pts = np.linspace(lo, hi, K)
    wts = np.full(K, 1.0 / K)
    return project_to_windows(pts, wts, d, ell=ell)


def uniform_over_all_windows(d: int) -> np.ndarray:
    """lam with uniform weight 1/n_win over every window in build_window_matrices(d).

    This is the raw aggregated baseline — equivalent to passing lam=None to
    build_sdp_data.  Exposed here so atomic-ν callers can name the choice
    explicitly and compose with adaptive refinement on top.
    """
    n = len(list_windows(d))
    return np.full(n, 1.0 / n, dtype=np.float64)


def peak_concentrated_nu(d: int, t_star: float, width: float, K: int,
                         *,
                         ell: Union[int, Sequence[int], str] = 2,
                         profile: str = "gaussian") -> np.ndarray:
    """ν concentrated around shift t_star with effective width `width`.

    profile: "gaussian" (truncated normal) or "triangular".
    Weights are renormalized to sum to 1 before projection.  K points are
    sampled on a uniform grid over [t_star - 2*width, t_star + 2*width]
    (or clipped to [-1/2, 1/2]).
    """
    if width <= 0:
        raise ValueError("width must be positive")
    lo = max(-0.5, t_star - 2.0 * width)
    hi = min(0.5, t_star + 2.0 * width)
    if K < 2 or hi <= lo:
        return project_to_windows([t_star], [1.0], d, ell=ell)
    pts = np.linspace(lo, hi, K)
    if profile == "gaussian":
        w = np.exp(-0.5 * ((pts - t_star) / width) ** 2)
    elif profile == "triangular":
        w = np.maximum(0.0, 1.0 - np.abs(pts - t_star) / (2.0 * width))
    else:
        raise ValueError(f"unknown profile {profile!r}")
    s = w.sum()
    if s <= 0:
        return project_to_windows([t_star], [1.0], d, ell=ell)
    w = w / s
    return project_to_windows(pts, w, d, ell=ell)


def seed_from_joint_bisect(d: int, order: int,
                           *,
                           t_lo: float = 1.0,
                           t_hi: float = 1.5,
                           tol: float = 1e-5,
                           verbose: bool = False) -> Tuple[np.ndarray, float]:
    """Return (lam_win, t_hi) from the min-max Lasserre bisection.

    Runs certified_lasserre.joint_bisect.bisect_joint_sdp which solves
        L_k(d) = min t s.t. t·M_{k-1}(y) − M_{k-1}(q_W y) ⪰ 0 ∀W, …
    and reads the optimal window-localizing PSD dual S_W at the feasible
    bracket t_hi.  BisectResult exposes
        lam_win[W] ≈ S_W[0,0]  (normalized, nonneg)
    which IS the optimal atomic-ν distribution on windows — plugging it
    back into solve_atomic_nu_sdp recovers the bisection value to within
    solver tolerance.  This is the "oracle seed" for adaptive CG, and
    serves as the primary equivalence check that the atomic-ν formulation
    saturates val(d) when ν is chosen well.

    Requires MOSEK (joint_bisect uses Fusion directly).
    """
    from certified_lasserre.joint_bisect import bisect_joint_sdp
    res = bisect_joint_sdp(d=d, order=order, t_lo=t_lo, t_hi=t_hi,
                           tol=tol, verbose=verbose)
    lam = np.asarray(res.lam_win, dtype=np.float64)
    if lam.sum() <= 0:
        raise RuntimeError(
            f"joint_bisect returned all-zero lam_win at d={d} order={order} "
            f"t_hi={res.t_hi:.6f} — dual extraction failed")
    return lam, float(res.t_hi)


def adaptive_from_solution(prev: "AtomicNuResult",
                           K: int,
                           *,
                           floor: float = 1e-3,
                           temperature: float = 1.0) -> np.ndarray:
    """Column-generation-style refinement of ν from a previous SDP solution.

    Given a previous AtomicNuResult with primal moments y*, evaluate
    f_W(y*) := Σ M_W[i,j] y*_{e_i+e_j} for every window W (this is the
    "window value" at the current pseudo-primal), then upweight windows
    where f_W(y*) is largest — those are the shifts where (f*f) is hot and
    where tightening ν is most effective.

    The new ν-weights are
        lam_W ∝ max(0, f_W(y*) - floor) ** (1/temperature)
    restricted to the top-K windows by f_W(y*).  Returns a normalized
    nonneg vector of length n_win.

    temperature → 0 gives argmax (single-point ν); temperature → ∞ gives
    uniform weighting.  floor clips near-zero values to avoid numerical
    noise.
    """
    f_vals = prev.window_values
    if f_vals is None:
        raise ValueError("prev.window_values is None — pass prev from a solve "
                         "that completed with a primal.")
    f = np.asarray(f_vals, dtype=np.float64)
    # Top-K by value
    if K <= 0 or K > len(f):
        K = len(f)
    top = np.argsort(-f)[:K]
    lam = np.zeros_like(f)
    keep = np.maximum(0.0, f[top] - floor)
    if keep.sum() <= 0:
        # All below floor — fall back to top-K uniform.
        lam[top] = 1.0 / K
        return lam
    weighted = keep ** (1.0 / max(temperature, 1e-6))
    lam[top] = weighted / weighted.sum()
    return lam


# =====================================================================
# Top-level solve entry point
# =====================================================================

@dataclass
class AtomicNuResult:
    d: int
    order: int
    lam_windows: np.ndarray           # shape (n_win,) nonneg, sum=1
    lb_numerical: float               # float SDP primal objective
    lb_dual: float                    # float SDP dual objective (lower bound)
    lb_rigorous: Optional[float]      # rational-rounded rigorous lower bound, if certify=True
    solver: str
    status: str
    solve_time: float
    y: np.ndarray                     # primal pseudo-moments
    lambda_A: np.ndarray              # equality duals
    S_blocks: List[np.ndarray]        # PSD block duals
    window_values: Optional[np.ndarray] = None  # f_W(y*) for adaptive design
    stationarity_res_inf: Optional[float] = None


def _window_values(sdp: SDPData, y: np.ndarray) -> np.ndarray:
    """f_W(y) = Σ M_W[i,j] y_{e_i+e_j} for each window W.

    Uses the precomputed F_scipy sparse matrix (n_win × n_y) from the
    precompute cache, if available; otherwise computes directly from
    build_window_matrices and the moment index map.
    """
    from lasserre.precompute import _precompute
    P = _precompute(sdp.d, sdp.order, verbose=False)
    F = P['F_scipy']
    return np.asarray(F @ y, dtype=np.float64)


def solve_atomic_nu_sdp(
    lam_windows: np.ndarray,
    d: int,
    order: int = 2,
    *,
    solver: str = "auto",
    certify: bool = False,
    verbose: bool = False,
    compute_window_values: bool = True,
) -> AtomicNuResult:
    """Solve the aggregated atomic-ν Lasserre SDP and return a structured result.

    Parameters
    ----------
    lam_windows : array of length n_win = len(build_window_matrices(d)[0])
        Window weights defining ν.  Need not be normalized; nonneg required.
        If all zero, raises ValueError.
    d : number of bins.
    order : Lasserre order k (≥2 for meaningful localization).
    solver : 'auto' | 'mosek' | 'clarabel' | 'scs' | 'bm'.
    certify : if True, run the rational rounding pipeline and return
        lb_rigorous; otherwise leave it None.
    verbose : pass-through to solver.
    compute_window_values : if True, evaluate f_W(y*) for every W; used
        by adaptive_from_solution.

    Returns
    -------
    AtomicNuResult with lb_numerical <= val(d) numerically (and
    lb_rigorous <= val(d) rigorously when certify=True).
    """
    lam = np.asarray(lam_windows, dtype=np.float64)
    if np.any(lam < -1e-15):
        raise ValueError("lam_windows must be nonneg")
    lam = np.maximum(lam, 0.0)
    s = lam.sum()
    if s <= 0:
        raise ValueError("lam_windows sums to zero — ν is the empty measure")
    lam = lam / s

    sdp = build_sdp_data(d=d, order=order, lam=lam, verbose=verbose)
    res: SolverResult = solve_primal_dual(sdp, solver=solver, verbose=verbose)

    # Sanity: solver must return a usable status.  Fail loud on infeasible/unbounded.
    status_upper = res.status.upper()
    bad = ("INFEASIBLE" in status_upper and "NOT" not in status_upper) or \
          "UNBOUNDED" in status_upper or "FAILURE" in status_upper
    if bad:
        raise RuntimeError(
            f"atomic-ν SDP solver returned status {res.status!r}; refusing to "
            f"return nonsense. d={d} order={order} solver={res.solver}")

    # Dual lower bound: for a feasibility-bounded SDP, the dual objective
    # is b^T lambda_A + Σ <F_j(0), S_j>.  Since F_j(0)=0 in our encoding
    # (block operators are linear in y, zero at y=0), the dual objective
    # reduces to b^T lambda_A.  Equivalently, strong duality → res.obj.
    lb_numerical = float(res.obj)
    lb_dual = float(sdp.b @ res.lambda_A)

    # Stationarity residual (diagnostic for certifiability)
    r_inf = None
    try:
        r = res.stationarity_residual(sdp)
        r_inf = float(np.max(np.abs(r)))
    except Exception:
        pass

    wvals = _window_values(sdp, res.y) if compute_window_values else None

    lb_rig = None
    if certify:
        # Delegate to the existing rational-rounding pipeline.  Soft import
        # so that users without python-flint can still run certify=False.
        try:
            from certified_lasserre.round_repair import round_and_repair
            from certified_lasserre.certify import certify_bound
            rr = round_and_repair(sdp, res)
            cb = certify_bound(sdp, rr)
            lb_rig = float(cb['lb']) if 'lb' in cb else None
        except Exception as e:
            if verbose:
                print(f"[atomic-ν] certify failed: {type(e).__name__}: {e}")
            lb_rig = None

    return AtomicNuResult(
        d=d, order=order,
        lam_windows=lam,
        lb_numerical=lb_numerical,
        lb_dual=lb_dual,
        lb_rigorous=lb_rig,
        solver=res.solver,
        status=res.status,
        solve_time=res.time,
        y=res.y,
        lambda_A=res.lambda_A,
        S_blocks=res.S_blocks,
        window_values=wvals,
        stationarity_res_inf=r_inf,
    )


# =====================================================================
# Column-generation driver (Phase 1.3)
# =====================================================================

def column_generation_atomic_nu(
    d: int,
    order: int = 2,
    *,
    initial_K: int = 16,
    refine_K: int = 8,
    rounds: int = 5,
    solver: str = "auto",
    temperature: float = 0.5,
    verbose: bool = False,
) -> List[AtomicNuResult]:
    """Iterate: start with uniform ν, solve, read f_W(y*), refine.

    Each round produces a new AtomicNuResult.  λ(ν) is non-monotone under
    refinement in general, but in practice the best-so-far tracks val(d)
    as ν concentrates on active windows.

    Returns the full sequence of results (caller can pick the best by
    max(lb_numerical)).
    """
    history: List[AtomicNuResult] = []

    lam = uniform_grid_nu(d, K=initial_K, ell=2)
    res = solve_atomic_nu_sdp(lam, d=d, order=order, solver=solver,
                              verbose=verbose, compute_window_values=True)
    history.append(res)
    if verbose:
        print(f"[cg] round 0: lb={res.lb_numerical:.6f} "
              f"({res.solver}, {res.solve_time:.1f}s)")

    for r in range(1, rounds):
        lam_new = adaptive_from_solution(res, K=refine_K,
                                         temperature=temperature)
        # Mix 50/50 with previous lam to prevent oscillation.
        lam = 0.5 * lam + 0.5 * lam_new
        lam = lam / lam.sum()
        res = solve_atomic_nu_sdp(lam, d=d, order=order, solver=solver,
                                  verbose=verbose,
                                  compute_window_values=True)
        history.append(res)
        if verbose:
            print(f"[cg] round {r}: lb={res.lb_numerical:.6f} "
                  f"({res.solver}, {res.solve_time:.1f}s)")

    return history


# =====================================================================
# CLI
# =====================================================================

def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Atomic-ν Lasserre SDP lower bound on val(d).")
    ap.add_argument("--d", type=int, required=True, help="number of bins")
    ap.add_argument("--order", type=int, default=2, help="Lasserre order k")
    ap.add_argument("--nu-strategy", choices=("uniform", "peak", "adaptive"),
                    default="uniform")
    ap.add_argument("--K", type=int, default=32,
                    help="number of ν atoms / CG refinement size")
    ap.add_argument("--ell", type=int, default=2,
                    help="window length for ν projection (2 = point windows)")
    ap.add_argument("--t-star", type=float, default=0.0,
                    help="peak center (only for --nu-strategy peak)")
    ap.add_argument("--width", type=float, default=0.1,
                    help="peak width (only for --nu-strategy peak)")
    ap.add_argument("--rounds", type=int, default=5,
                    help="CG rounds (only for --nu-strategy adaptive)")
    ap.add_argument("--solver", default="auto")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.nu_strategy == "uniform":
        lam = uniform_grid_nu(args.d, K=args.K, ell=args.ell)
        res = solve_atomic_nu_sdp(lam, d=args.d, order=args.order,
                                  solver=args.solver, verbose=args.verbose)
        _print_result(res)
    elif args.nu_strategy == "peak":
        lam = peak_concentrated_nu(args.d, t_star=args.t_star,
                                   width=args.width, K=args.K, ell=args.ell)
        res = solve_atomic_nu_sdp(lam, d=args.d, order=args.order,
                                  solver=args.solver, verbose=args.verbose)
        _print_result(res)
    else:  # adaptive
        history = column_generation_atomic_nu(
            args.d, order=args.order,
            initial_K=args.K, refine_K=max(args.K // 2, 4),
            rounds=args.rounds, solver=args.solver, verbose=args.verbose)
        best = max(history, key=lambda r: r.lb_numerical)
        print("--- CG history ---")
        for i, r in enumerate(history):
            print(f"  round {i}: lb={r.lb_numerical:.6f} "
                  f"time={r.solve_time:.2f}s status={r.status}")
        print("--- best ---")
        _print_result(best)
    return 0


def _print_result(r: AtomicNuResult) -> None:
    print(f"d={r.d} order={r.order} solver={r.solver} status={r.status}")
    print(f"  lb_numerical = {r.lb_numerical:.6f}")
    print(f"  lb_dual      = {r.lb_dual:.6f}")
    if r.lb_rigorous is not None:
        print(f"  lb_rigorous  = {r.lb_rigorous:.6f}")
    print(f"  supp(nu) = {int(np.sum(r.lam_windows > 1e-12))}"
          f" / {len(r.lam_windows)} windows")
    print(f"  solve_time   = {r.solve_time:.2f}s")
    if r.stationarity_res_inf is not None:
        print(f"  ||r||_inf    = {r.stationarity_res_inf:.2e}")


if __name__ == "__main__":
    raise SystemExit(_main())
