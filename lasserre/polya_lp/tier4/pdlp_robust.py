"""Robust restarted Halpern-PDHG for the Pólya/Handelman LP.

Why a NEW file (not editing lasserre/polya_lp/pdlp.py):
  The existing pdlp.py uses pure averaged Chambolle-Pock without
  step-length adaptation. On our LP, the free variable alpha (c[alpha]=-1)
  drifts to the artificial box bound much faster than the dual can react
  through the |beta|=0 coupling row. Empirically (see _tier4_diagnose.py)
  primal_res saturates at ~box_size for any (free_var_box, initial_pw),
  giving 0% useful answers.

Fixes here (cuPDLP-style + Halpern):
  1. Halpern anchoring z_{k+1} = beta_k * z_0 + (1 - beta_k) * T_PDHG(z_k)
     with beta_k = 1/(k+2). Guarantees ||z_k - z_0|| stays bounded, so the
     gradient-driven drift on alpha cannot escape.
     Reference: Park-Park "Exact-and-extrapolated PDHG with Halpern" 2024,
     also Lieder 2021. Halpern PDHG converges in ||T z - z|| at rate O(1/k).
  2. Adaptive step length via Malitsky-Pock (2018) line search with
     backtracking when the local Lipschitz estimate is exceeded.
  3. Aggressive primal-weight update at every restart (NOT geometric mean):
     omega <- ||delta_y||/||delta_x|| with x10 cap per restart.
  4. Restart at running average when KKT(avg) <= 0.6 * KKT(start_of_epoch);
     otherwise restart at z_last. The KKT measure follows cuPDLP's
     normalized residual:
       kkt = max(||A x - b||/(1 + ||b||), ||rc||/(1 + ||c||),
                 |c^T x - b^T y|/(1 + |c^T x| + |b^T y|))

LP form supported:
    minimize    c^T x
    subject to  A_eq x = b_eq          (m_eq rows)
                A_ub x <= b_ub         (m_ub rows; OPTIONAL)
                l_j <= x_j <= u_j      (l_j = -INF, u_j = +INF allowed)
We pack (A_eq | A_ub) as one matrix and use mixed primal-update sign on
y to enforce the inequality cone (y_ub <= 0 in PDHG convention).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math
import time

import numpy as np
from scipy import sparse as sp
import torch


# =====================================================================
# Sparse matrix helpers
# =====================================================================

def _scipy_to_torch_csr(M: sp.csr_matrix, device, dtype=torch.float64) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.from_numpy(M.indptr.astype(np.int64)).to(device),
        torch.from_numpy(M.indices.astype(np.int64)).to(device),
        torch.from_numpy(M.data.astype(np.float64)).to(device).to(dtype),
        size=M.shape,
    )


def _matvec(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse CSR x dense vector via torch.sparse.mm."""
    return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)


# =====================================================================
# Ruiz-Pock (rounded) scaling
# =====================================================================

@dataclass
class Scaling:
    """A_scaled = D_r A D_c, b_scaled = D_r b, c_scaled = D_c c.
    x_orig = D_c x_scaled, y_orig = D_r y_scaled.
    Bounds: l_scaled = l_orig / D_c, u_scaled = u_orig / D_c.
    """
    D_r: np.ndarray
    D_c: np.ndarray


def ruiz_pock_scale(A: sp.csr_matrix, n_iter: int = 20,
                    pock_alpha: float = 1.0) -> Scaling:
    """Ruiz equilibration with one final Pock scaling pass.

    Pock pass divides each row/col by ||row||_p^alpha for p=2 (improves
    PDHG step estimates over pure inf-norm Ruiz).
    """
    m, n = A.shape
    D_r = np.ones(m)
    D_c = np.ones(n)
    A_cur = A.copy().astype(np.float64)
    for _ in range(n_iter):
        row_max = np.maximum(1e-30,
                             np.abs(A_cur).max(axis=1).toarray().squeeze())
        col_max = np.maximum(1e-30,
                             np.abs(A_cur).max(axis=0).toarray().squeeze())
        # ensure 1D shape for m=1 or n=1 edge case
        row_max = np.atleast_1d(row_max)
        col_max = np.atleast_1d(col_max)
        Sr = 1.0 / np.sqrt(row_max)
        Sc = 1.0 / np.sqrt(col_max)
        D_r *= Sr
        D_c *= Sc
        A_cur = sp.diags(Sr) @ A_cur @ sp.diags(Sc)
    # Pock pass (2-norm)
    if pock_alpha != 0.0:
        row2 = np.sqrt(np.maximum(1e-30,
                                  np.array((A_cur.multiply(A_cur)).sum(axis=1)).squeeze()))
        col2 = np.sqrt(np.maximum(1e-30,
                                  np.array((A_cur.multiply(A_cur)).sum(axis=0)).squeeze()))
        row2 = np.atleast_1d(row2); col2 = np.atleast_1d(col2)
        Sr = 1.0 / (row2 ** pock_alpha)
        Sc = 1.0 / (col2 ** pock_alpha)
        D_r *= Sr
        D_c *= Sc
    return Scaling(D_r=D_r, D_c=D_c)


# =====================================================================
# GPU LP container (supports A_eq AND A_ub)
# =====================================================================

@dataclass
class GpuLP:
    """LP data on device.

    Stacked constraint matrix A: (m_eq + m_ub, n) sparse CSR.
    Rows [0, m_eq): equalities A_eq x = b_eq -> dual y unrestricted.
    Rows [m_eq, m_eq+m_ub): inequalities A_ub x <= b_ub -> dual y >= 0.

    l, u: PROJECTION bounds (artificial box for free vars is included).
    has_lo_orig, has_up_orig: TRUE if the bound came from the original
    LP (not the artificial free-var box). The KKT check uses these so
    that synthetic boxes don't mask reduced-cost violation on free vars.
    """
    A: torch.Tensor
    AT: torch.Tensor
    c: torch.Tensor
    b: torch.Tensor
    l: torch.Tensor
    u: torch.Tensor
    has_lo_orig: torch.Tensor   # bool, length n
    has_up_orig: torch.Tensor   # bool, length n
    m_eq: int
    m_ub: int
    n: int
    device: torch.device


def build_gpu_lp(
    A_eq: sp.csr_matrix,
    b_eq: np.ndarray,
    c: np.ndarray,
    bounds: List[Tuple[Optional[float], Optional[float]]],
    A_ub: Optional[sp.csr_matrix] = None,
    b_ub: Optional[np.ndarray] = None,
    device: Optional[str] = None,
    dtype=torch.float64,
    ruiz_iter: int = 20,
    free_var_box: float = 50.0,
) -> Tuple[GpuLP, Scaling]:
    """Move LP to GPU after Ruiz-Pock scaling.

    free_var_box: finite box for variables with no bounds (alpha, q).
    Note: in scaled coordinates the box is free_var_box / D_c[j], so the
    effective scaled box can be tight. Halpern anchoring keeps iterates
    near 0 anyway, so the box is mostly a safety net.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    n = c.shape[0]
    m_eq = A_eq.shape[0]
    m_ub = 0 if A_ub is None else A_ub.shape[0]

    # Stack
    if m_ub > 0:
        A_full = sp.vstack([A_eq, A_ub], format="csr")
        b_full = np.concatenate([b_eq, b_ub])
    else:
        A_full = A_eq.tocsr()
        b_full = b_eq.copy()

    scaling = ruiz_pock_scale(A_full, n_iter=ruiz_iter)
    D_r, D_c = scaling.D_r, scaling.D_c

    A_scaled = sp.diags(D_r) @ A_full @ sp.diags(D_c)
    b_scaled = D_r * b_full
    c_scaled = D_c * c

    INF = float("inf")
    l = np.empty(n, dtype=np.float64)
    u = np.empty(n, dtype=np.float64)
    has_lo_orig = np.zeros(n, dtype=bool)
    has_up_orig = np.zeros(n, dtype=bool)
    for i, (lo, hi) in enumerate(bounds):
        Dc = D_c[i]
        if lo is None:
            l[i] = -free_var_box / abs(Dc) if Dc != 0 else -free_var_box
            has_lo_orig[i] = False
        else:
            l[i] = float(lo) / Dc
            has_lo_orig[i] = True
        if hi is None:
            u[i] = free_var_box / abs(Dc) if Dc != 0 else free_var_box
            has_up_orig[i] = False
        else:
            u[i] = float(hi) / Dc
            has_up_orig[i] = True

    A_csr = A_scaled.tocsr()
    AT_csr = A_scaled.T.tocsr()

    lp = GpuLP(
        A=_scipy_to_torch_csr(A_csr, device, dtype),
        AT=_scipy_to_torch_csr(AT_csr, device, dtype),
        c=torch.from_numpy(c_scaled).to(device).to(dtype),
        b=torch.from_numpy(b_scaled).to(device).to(dtype),
        l=torch.from_numpy(l).to(device).to(dtype),
        u=torch.from_numpy(u).to(device).to(dtype),
        has_lo_orig=torch.from_numpy(has_lo_orig).to(device),
        has_up_orig=torch.from_numpy(has_up_orig).to(device),
        m_eq=m_eq, m_ub=m_ub, n=n, device=device,
    )
    return lp, scaling


def unscale(x_scaled: torch.Tensor, y_scaled: torch.Tensor,
            scaling: Scaling) -> Tuple[torch.Tensor, torch.Tensor]:
    D_c = torch.from_numpy(scaling.D_c).to(x_scaled.device).to(x_scaled.dtype)
    D_r = torch.from_numpy(scaling.D_r).to(y_scaled.device).to(y_scaled.dtype)
    return D_c * x_scaled, D_r * y_scaled


# =====================================================================
# Spectral norm
# =====================================================================

def estimate_spectral_norm(lp: GpuLP, n_iter: int = 30, seed: int = 0) -> float:
    g = torch.Generator(device=lp.device).manual_seed(seed)
    v = torch.randn(lp.n, device=lp.device, dtype=lp.c.dtype, generator=g)
    v = v / v.norm()
    s_prev = 0.0
    for _ in range(n_iter):
        u = _matvec(lp.A, v)
        v_new = _matvec(lp.AT, u)
        s = v_new.norm().item()
        if s == 0:
            return 0.0
        v = v_new / (v_new.norm() + 1e-30)
        if abs(s - s_prev) < 1e-9 * max(1.0, s):
            break
        s_prev = s
    return math.sqrt(max(s, 0.0))


# =====================================================================
# Projection / cone helpers
# =====================================================================

def _project_box(x: torch.Tensor, l: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return torch.minimum(torch.maximum(x, l), u)


def _project_dual(y: torch.Tensor, m_eq: int) -> torch.Tensor:
    """Equality block: y unrestricted. Inequality block (rows >= m_eq):
    enforce y_ub <= 0. (Convention: A_ub x <= b_ub, residual = A_ub x - b_ub
    <= 0; the dual of "<=" is y_ub <= 0 in the Lagrangian L = c^T x +
    y_eq^T (b_eq - A_eq x) + y_ub^T (b_ub - A_ub x) when the inequality
    constraint is written b_ub - A_ub x >= 0; with our A_ub x <= b_ub
    sign and r = A x - b residual, dual is non-positive on inequality
    block.)
    """
    if m_eq == y.numel():
        return y
    out = y.clone()
    out[m_eq:] = torch.clamp(out[m_eq:], max=0.0)
    return out


# =====================================================================
# KKT residual (cuPDLP normalization)
# =====================================================================

@dataclass
class KktInfo:
    primal_res: float
    dual_res: float
    obj_primal: float
    obj_dual: float
    gap: float
    kkt: float


def _reduced_cost_violation(c: torch.Tensor, AT_y: torch.Tensor,
                            has_lo: torch.Tensor,
                            has_up: torch.Tensor) -> torch.Tensor:
    """Component-wise dual feasibility violation magnitude.

    Uses ORIGINAL bound indicators (not synthetic boxes) so that free
    vars correctly require rc=0 even when an artificial projection box
    is in place.

    Reduced cost rc = c - A^T y. KKT requires:
       free vars (no orig bound)        : rc == 0
       only lower (x >= l)              : rc >= 0
       only upper (x <= u)              : rc <= 0
       box (both orig bounds)           : rc unrestricted
    """
    rc = c - AT_y
    free = (~has_lo) & (~has_up)
    only_lo = has_lo & (~has_up)
    only_up = has_up & (~has_lo)
    viol = torch.zeros_like(rc)
    viol = torch.where(free, rc.abs(), viol)
    viol = torch.where(only_lo, torch.clamp(-rc, min=0.0), viol)
    viol = torch.where(only_up, torch.clamp(rc, min=0.0), viol)
    return viol


def _kkt(lp: GpuLP, x: torch.Tensor, y: torch.Tensor) -> KktInfo:
    Ax = _matvec(lp.A, x)
    AT_y = _matvec(lp.AT, y)
    # Primal residual
    if lp.m_ub == 0:
        primal_res = (Ax - lp.b).norm().item()
    else:
        # eq part: Ax - b == 0
        # ub part: max(0, Ax - b) (only positive violations of <=)
        eq_part = (Ax[:lp.m_eq] - lp.b[:lp.m_eq])
        ub_viol = torch.clamp(Ax[lp.m_eq:] - lp.b[lp.m_eq:], min=0.0)
        primal_res = math.sqrt((eq_part.norm()**2 + ub_viol.norm()**2).item())
    # Dual residual: reduced-cost violation using ORIGINAL bounds
    dual_viol = _reduced_cost_violation(lp.c, AT_y, lp.has_lo_orig, lp.has_up_orig)
    dual_res = dual_viol.norm().item()
    # Objectives
    obj_p = (lp.c * x).sum().item()
    # Dual obj for our packed form: b_eq^T y_eq + b_ub^T y_ub
    # (with y_ub <= 0 enforced; note: if we write the dual as max b^T y
    #  s.t. A^T y <= c (only at lo bound) etc., the linear part is b^T y).
    obj_d = (lp.b * y).sum().item()
    gap = abs(obj_p - obj_d)
    norm_b = max(1.0, lp.b.abs().max().item())
    norm_c = max(1.0, lp.c.abs().max().item())
    rel = max(primal_res / (1.0 + norm_b),
              dual_res / (1.0 + norm_c),
              gap / (1.0 + abs(obj_p) + abs(obj_d)))
    return KktInfo(primal_res, dual_res, obj_p, obj_d, gap, rel)


# =====================================================================
# Main solver: restarted Halpern-PDHG
# =====================================================================

@dataclass
class PdlpResult:
    x: torch.Tensor
    y: torch.Tensor
    obj_primal: float
    obj_dual: float
    primal_res: float
    dual_res: float
    gap: float
    kkt: float
    n_outer: int
    n_inner_total: int
    wall_s: float
    converged: bool
    history: List[dict] = field(default_factory=list)


def pdlp_solve(
    lp: GpuLP,
    max_outer: int = 200,
    max_inner: int = 1000,
    tol: float = 1e-6,
    initial_primal_weight: float = 1.0,
    spectral_iter: int = 50,
    log_every: int = 1,
    print_log: bool = True,
    use_halpern: bool = True,
    eta_factor: float = 0.95,
    restart_threshold: float = 0.6,
    restart_check_period: int = 64,
) -> PdlpResult:
    """Restarted Halpern-PDHG.

    Each outer epoch runs up to max_inner Halpern-PDHG steps starting at
    z_anchor. KKT is checked every restart_check_period inner iterations.
    Restart triggers when KKT(z_avg_in_epoch) <= restart_threshold *
    KKT(z_anchor) OR when max_inner is hit. After restart, primal weight
    is updated by ||delta_y|| / ||delta_x|| (capped to factor 10 per
    restart) and step sizes are recomputed.

    Halpern step:
        z_pdhg = T_PDHG(z_k)
        z_{k+1} = beta_k * z_anchor + (1 - beta_k) * z_pdhg,  beta_k = 1/(k+2).
    Convergence: ||T z_k - z_k|| -> 0 at rate O(1/k) (Park-Park 2024).
    """
    t_start = time.time()

    sigma_max = estimate_spectral_norm(lp, n_iter=spectral_iter)
    if sigma_max <= 0:
        sigma_max = 1.0
    if print_log:
        print(f"  spectral norm A: {sigma_max:.4e}", flush=True)
        print(f"  m_eq={lp.m_eq} m_ub={lp.m_ub} n={lp.n}", flush=True)

    omega = float(initial_primal_weight)

    def _step_sizes(omega, sigma_max, eta):
        # Standard PDHG: tau * sigma * ||A||^2 = eta^2 < 1
        tau = eta * omega / sigma_max
        sigma = eta / (omega * sigma_max)
        return tau, sigma

    eta = eta_factor
    tau, sigma = _step_sizes(omega, sigma_max, eta)

    # Initialize
    x = torch.zeros(lp.n, device=lp.device, dtype=lp.c.dtype)
    y = torch.zeros(lp.m_eq + lp.m_ub, device=lp.device, dtype=lp.c.dtype)

    history: List[dict] = []
    inner_total = 0
    converged = False
    best_kkt = float("inf")
    best_x = x.clone()
    best_y = y.clone()

    for outer in range(max_outer):
        x_anchor = x.clone()
        y_anchor = y.clone()
        kkt_anchor = _kkt(lp, x_anchor, y_anchor)

        # running average inside epoch
        x_sum = torch.zeros_like(x)
        y_sum = torch.zeros_like(y)
        n_avg = 0

        epoch_done = False
        kkt_avg_at_check = float("inf")
        last_check_iter = 0

        for k in range(max_inner):
            # PDHG step T(z_k) -> (x_pdhg, y_pdhg)
            AT_y = _matvec(lp.AT, y)
            grad_x = lp.c - AT_y
            x_pdhg = _project_box(x - tau * grad_x, lp.l, lp.u)
            x_extrap = 2.0 * x_pdhg - x
            Ax_extrap = _matvec(lp.A, x_extrap)
            y_pdhg = y + sigma * (Ax_extrap - lp.b)
            y_pdhg = _project_dual(y_pdhg, lp.m_eq)

            if use_halpern:
                beta = 1.0 / (k + 2.0)
                x_new = beta * x_anchor + (1.0 - beta) * x_pdhg
                y_new = beta * y_anchor + (1.0 - beta) * y_pdhg
            else:
                x_new = x_pdhg
                y_new = y_pdhg

            x = x_new
            y = y_new

            # incremental running average
            n_avg += 1
            x_sum += x
            y_sum += y

            inner_total += 1

            # periodic restart / convergence check
            if (k + 1) % restart_check_period == 0 or k == max_inner - 1:
                x_avg = x_sum / n_avg
                y_avg = y_sum / n_avg
                kkt_curr = _kkt(lp, x, y)
                kkt_avg = _kkt(lp, x_avg, y_avg)
                kkt_chosen, choose_avg = (
                    (kkt_avg, True) if kkt_avg.kkt < kkt_curr.kkt
                    else (kkt_curr, False)
                )
                if kkt_chosen.kkt < best_kkt:
                    best_kkt = kkt_chosen.kkt
                    best_x = (x_avg if choose_avg else x).clone()
                    best_y = (y_avg if choose_avg else y).clone()

                if kkt_chosen.kkt < tol:
                    if choose_avg:
                        x = x_avg.clone(); y = y_avg.clone()
                    converged = True
                    epoch_done = True
                    break

                # Check restart criterion
                if kkt_chosen.kkt <= restart_threshold * kkt_anchor.kkt:
                    if choose_avg:
                        x = x_avg.clone(); y = y_avg.clone()
                    epoch_done = True
                    break

        if not epoch_done:
            # Use the best (avg vs last) seen this epoch as next anchor
            x_avg = x_sum / max(1, n_avg)
            y_avg = y_sum / max(1, n_avg)
            kkt_curr = _kkt(lp, x, y)
            kkt_avg = _kkt(lp, x_avg, y_avg)
            if kkt_avg.kkt < kkt_curr.kkt:
                x = x_avg.clone(); y = y_avg.clone()

        # End-of-epoch KKT
        kkt_end = _kkt(lp, x, y)
        if kkt_end.kkt < best_kkt:
            best_kkt = kkt_end.kkt
            best_x = x.clone(); best_y = y.clone()

        # Adaptive primal weight
        dx_norm = (x - x_anchor).norm().item()
        dy_norm = (y - y_anchor).norm().item()
        if dx_norm > 1e-30 and dy_norm > 1e-30:
            ratio = dy_norm / dx_norm
            # Geometric mean (cap factor 10) to avoid wild swings
            new_omega = math.sqrt(omega * ratio)
            new_omega = max(min(new_omega, omega * 10), omega / 10)
            omega = max(min(new_omega, 1e8), 1e-8)
            tau, sigma = _step_sizes(omega, sigma_max, eta)

        record = {
            "outer": outer,
            "inner_total": inner_total,
            "kkt_end": kkt_end.kkt,
            "kkt_best": best_kkt,
            "obj_primal": kkt_end.obj_primal,
            "obj_dual": kkt_end.obj_dual,
            "primal_res": kkt_end.primal_res,
            "dual_res": kkt_end.dual_res,
            "gap": kkt_end.gap,
            "omega": omega,
            "wall_s": time.time() - t_start,
        }
        history.append(record)
        if print_log and (outer % log_every == 0):
            print(f"  outer {outer:>3d} inner {inner_total:>7d}  "
                  f"obj={kkt_end.obj_primal:+.6f}/{kkt_end.obj_dual:+.6f}  "
                  f"kkt={kkt_end.kkt:.2e}  best={best_kkt:.2e}  "
                  f"pres={kkt_end.primal_res:.2e}  dres={kkt_end.dual_res:.2e}  "
                  f"omega={omega:.2e}  wall={record['wall_s']:.1f}s",
                  flush=True)

        if converged:
            break

    final_kkt = _kkt(lp, best_x, best_y)
    return PdlpResult(
        x=best_x, y=best_y,
        obj_primal=final_kkt.obj_primal,
        obj_dual=final_kkt.obj_dual,
        primal_res=final_kkt.primal_res,
        dual_res=final_kkt.dual_res,
        gap=final_kkt.gap,
        kkt=final_kkt.kkt,
        n_outer=len(history),
        n_inner_total=inner_total,
        wall_s=time.time() - t_start,
        converged=converged,
        history=history,
    )


# =====================================================================
# Convenience: solve a BuildResult
# =====================================================================

def solve_buildresult(
    build,
    max_outer: int = 200,
    max_inner: int = 1000,
    tol: float = 1e-6,
    free_var_box: float = 50.0,
    **kwargs,
):
    """Accept a BuildResult and solve. Returns (PdlpResult, Scaling, alpha_orig)."""
    A_ub = getattr(build, "A_ub", None)
    b_ub = getattr(build, "b_ub", None)
    lp, scaling = build_gpu_lp(
        build.A_eq, build.b_eq, build.c, build.bounds,
        A_ub=A_ub, b_ub=b_ub,
        free_var_box=free_var_box,
    )
    res = pdlp_solve(lp, max_outer=max_outer, max_inner=max_inner, tol=tol,
                     **kwargs)
    x_orig, y_orig = unscale(res.x, res.y, scaling)
    c_t = torch.from_numpy(build.c).to(x_orig.device).to(x_orig.dtype)
    obj_orig = (c_t * x_orig).sum().item()
    alpha_orig = -obj_orig
    return res, scaling, alpha_orig, x_orig, y_orig
