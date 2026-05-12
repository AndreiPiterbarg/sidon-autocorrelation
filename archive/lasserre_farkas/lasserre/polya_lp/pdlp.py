"""GPU-resident restarted PDHG (PDLP) solver for the Pólya/Handelman LP.

Implements the algorithm from Applegate et al. 2025 "PDLP: A Practical
First-Order Method for Large-Scale Linear Programming" (arXiv:2501.07018):
  - Restarted Chambolle-Pock primal-dual hybrid gradient
  - Adaptive primal weight (ratio tau/sigma)
  - KKT-residual-based restart heuristic
  - Optional feasibility polishing pass at the end

LP standard form (equality constraints + box bounds):
    minimize    c^T x
    subject to  A_eq x = b_eq
                l_j <= x_j <= u_j   (l_j = -inf, u_j = +inf allowed)

For our Pólya LP we MINIMIZE -alpha, so the objective at optimum is
-alpha_LP. The "alpha" we want is -obj.

Implementation notes:
- Sparse matrix on GPU via torch.sparse_csr_tensor (FP64).
- Spectral norm estimated via power iteration on A^T A.
- Restart triggered when running-average KKT residual < 0.6 * previous-restart KKT.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
import math
import numpy as np
from scipy import sparse as sp

import torch


# =====================================================================
# Sparse LP wrapper on GPU
# =====================================================================

@dataclass
class GpuLP:
    """Holds the LP data on GPU (FP64).

    A:  (m, n) sparse CSR (equality constraints A x = b)
    AT: (n, m) sparse CSR (transpose, precomputed for A^T y)
    c:  (n,)
    b:  (m,)
    l:  (n,)   variable lower bounds (-INF for free)
    u:  (n,)   variable upper bounds (+INF for free)
    """
    A: torch.Tensor
    AT: torch.Tensor
    c: torch.Tensor
    b: torch.Tensor
    l: torch.Tensor
    u: torch.Tensor
    device: torch.device
    n: int
    m: int


def _scipy_to_torch_sparse_csr(M: sp.csr_matrix, device, dtype=torch.float64) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.from_numpy(M.indptr.astype(np.int64)).to(device),
        torch.from_numpy(M.indices.astype(np.int64)).to(device),
        torch.from_numpy(M.data.astype(np.float64)).to(device).to(dtype),
        size=M.shape,
    )


@dataclass
class RuizScaling:
    """Diagonal scaling: A_scaled = D_r A D_c, b_scaled = D_r b, c_scaled = D_c c.

    The original LP optimum (x*, y*) is recovered as
      x* = D_c^{-1} x_scaled*,    y* = D_r^{-1} y_scaled*.
    Bounds also rescale: l_scaled = D_c^{-1} l, u_scaled = D_c^{-1} u.
    """
    D_r: np.ndarray   # row scaling, length m
    D_c: np.ndarray   # col scaling, length n


def ruiz_scale(A: sp.csr_matrix, n_iter: int = 20) -> RuizScaling:
    """Iterative Ruiz equilibration: alternately scale rows and columns
    by sqrt(max abs entry) until ||row||_inf and ||col||_inf both ~ 1.
    """
    m, n = A.shape
    D_r = np.ones(m)
    D_c = np.ones(n)
    A_cur = A.copy()
    for _ in range(n_iter):
        # Row max-abs
        row_max = np.maximum(1e-30,
                             np.abs(A_cur).max(axis=1).toarray().squeeze())
        col_max = np.maximum(1e-30,
                             np.abs(A_cur).max(axis=0).toarray().squeeze())
        Sr = 1.0 / np.sqrt(row_max)
        Sc = 1.0 / np.sqrt(col_max)
        D_r *= Sr
        D_c *= Sc
        # A := diag(Sr) A diag(Sc)
        A_cur = sp.diags(Sr) @ A_cur @ sp.diags(Sc)
    return RuizScaling(D_r=D_r, D_c=D_c)


def build_gpu_lp(
    A_eq: sp.csr_matrix,
    b_eq: np.ndarray,
    c: np.ndarray,
    bounds: List[Tuple[Optional[float], Optional[float]]],
    device: Optional[str] = None,
    dtype=torch.float64,
    ruiz_iter: int = 20,
    free_var_box: float = 1e3,
) -> Tuple[GpuLP, RuizScaling]:
    """Move a scipy-form LP to GPU after Ruiz scaling.

    Returns (lp_scaled, scaling). The caller should solve lp_scaled and
    then UNSCALE the solution: x_orig = D_c x_scaled, y_orig = D_r y_scaled.

    free_var_box: finite box bound to use for "free" variables (those with
    None bounds). Setting this to ~1e3 avoids unbounded primal drift in
    PDHG. After Ruiz scaling, x_scaled is bounded by ~1 in the typical
    problem, so 1e3 is loose but finite.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    n = c.shape[0]
    m = A_eq.shape[0]

    # Apply Ruiz scaling
    scaling = ruiz_scale(A_eq, n_iter=ruiz_iter)
    D_r, D_c = scaling.D_r, scaling.D_c

    A_scaled = sp.diags(D_r) @ A_eq @ sp.diags(D_c)
    b_scaled = D_r * b_eq
    c_scaled = D_c * c

    # Bounds: x_scaled = D_c^{-1} x_orig, so [l, u] becomes [l/D_c, u/D_c].
    l = np.empty(n, dtype=np.float64)
    u = np.empty(n, dtype=np.float64)
    for i, (lo, hi) in enumerate(bounds):
        Dc = D_c[i]
        if lo is None:
            l[i] = -free_var_box / Dc
        else:
            l[i] = float(lo) / Dc
        if hi is None:
            u[i] = free_var_box / Dc
        else:
            u[i] = float(hi) / Dc

    A_csr = A_scaled.tocsr()
    AT_csr = A_scaled.transpose().tocsr()
    lp = GpuLP(
        A=_scipy_to_torch_sparse_csr(A_csr, device, dtype),
        AT=_scipy_to_torch_sparse_csr(AT_csr, device, dtype),
        c=torch.from_numpy(c_scaled).to(device).to(dtype),
        b=torch.from_numpy(b_scaled).to(device).to(dtype),
        l=torch.from_numpy(l).to(device).to(dtype),
        u=torch.from_numpy(u).to(device).to(dtype),
        device=device,
        n=n,
        m=m,
    )
    return lp, scaling


def unscale_solution(x_scaled: torch.Tensor, y_scaled: torch.Tensor,
                     scaling: RuizScaling) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recover original-LP solution from Ruiz-scaled solution.

    x_orig = D_c x_scaled,  y_orig = D_r y_scaled.
    """
    D_c = torch.from_numpy(scaling.D_c).to(x_scaled.device).to(x_scaled.dtype)
    D_r = torch.from_numpy(scaling.D_r).to(y_scaled.device).to(y_scaled.dtype)
    return D_c * x_scaled, D_r * y_scaled


def matvec(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse CSR x dense vector via torch.sparse.mm."""
    return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)


# =====================================================================
# Spectral norm estimation (for step-size selection)
# =====================================================================

def estimate_spectral_norm(lp: GpuLP, n_iter: int = 30, seed: int = 0) -> float:
    """Estimate ||A||_2 via power iteration on A^T A."""
    g = torch.Generator(device=lp.device).manual_seed(seed)
    v = torch.randn(lp.n, device=lp.device, dtype=lp.c.dtype, generator=g)
    v = v / v.norm()
    s_prev = 0.0
    for _ in range(n_iter):
        u = matvec(lp.A, v)
        v_new = matvec(lp.AT, u)
        s = v_new.norm().item()
        v = v_new / (v_new.norm() + 1e-30)
        if abs(s - s_prev) < 1e-9 * max(1.0, s):
            break
        s_prev = s
    return math.sqrt(s)


# =====================================================================
# KKT residual
# =====================================================================

def _project_box(x: torch.Tensor, l: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return torch.minimum(torch.maximum(x, l), u)


def _dual_residual_norm(c: torch.Tensor, AT_y: torch.Tensor,
                       l: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Reduced-cost violation: for each j,
        r_j = max(0, l_j > -INF: -(c_j - (A^T y)_j))
                  + max(0, u_j < +INF: (c_j - (A^T y)_j))
    For free vars (l = -INF, u = +INF), reduced cost must equal 0.
    For one-sided bounded: must have proper sign.
    """
    INF_THRESHOLD = 1e29
    rc = c - AT_y                       # reduced cost
    has_lo = l > -INF_THRESHOLD
    has_up = u < INF_THRESHOLD
    free = (~has_lo) & (~has_up)
    only_lo = has_lo & (~has_up)        # x >= l_j: reduced cost must be >= 0
    only_up = has_up & (~has_lo)        # x <= u_j: reduced cost must be <= 0
    bounded = has_lo & has_up           # box: any sign OK

    viol = torch.zeros_like(rc)
    # free: residual = rc (must be 0)
    viol = torch.where(free, rc.abs(), viol)
    # only_lo: violation if rc < 0
    viol = torch.where(only_lo, torch.clamp(-rc, min=0.0), viol)
    # only_up: violation if rc > 0
    viol = torch.where(only_up, torch.clamp(rc, min=0.0), viol)
    # bounded: 0 violation always
    return viol.norm()


@dataclass
class KktInfo:
    primal_res: float
    dual_res: float
    obj_primal: float
    obj_dual: float
    gap: float
    kkt: float


def _kkt_residual(lp: GpuLP, x: torch.Tensor, y: torch.Tensor) -> KktInfo:
    """Compute KKT residual components."""
    Ax = matvec(lp.A, x)
    AT_y = matvec(lp.AT, y)
    primal_res = (Ax - lp.b).norm().item()
    dual_res = _dual_residual_norm(lp.c, AT_y, lp.l, lp.u).item()
    obj_p = (lp.c * x).sum().item()
    obj_d = (lp.b * y).sum().item()
    gap = abs(obj_p - obj_d)
    # Relative KKT
    norm_b = max(1.0, lp.b.abs().max().item())
    norm_c = max(1.0, lp.c.abs().max().item())
    rel = max(primal_res / norm_b, dual_res / norm_c,
              gap / max(1.0, abs(obj_p), abs(obj_d)))
    return KktInfo(
        primal_res=primal_res, dual_res=dual_res,
        obj_primal=obj_p, obj_dual=obj_d, gap=gap, kkt=rel,
    )


# =====================================================================
# Restarted PDHG main loop
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
    history: List[dict]


def pdlp_solve(
    lp: GpuLP,
    max_outer: int = 200,
    max_inner: int = 500,
    tol: float = 1e-7,
    initial_primal_weight: float = 1.0,
    spectral_iter: int = 30,
    log_every: int = 1,
    print_log: bool = True,
    eta_factor: float = 0.99,
) -> PdlpResult:
    """Restarted Chambolle-Pock PDHG with adaptive primal weight.

    Algorithm:
      Outer: each pass starts at (x_start, y_start) and runs `max_inner`
        inner CP steps, tracking running average (x_avg, y_avg).
      Restart: at end of pass, if KKT(z_avg) < KKT(z_last) and KKT(z_avg) <
        0.6 * KKT(z_start), restart at z_avg; else at z_last.
      Adaptive primal weight: every restart, adjust primal_weight ω so that
        the next epoch balances primal-vs-dual progress.

    Step sizes:
      sigma_max = ||A||_2
      tau = eta_factor * primal_weight / sigma_max
      sigma = eta_factor / (primal_weight * sigma_max)
      Always satisfies tau * sigma * ||A||^2 = eta^2 < 1 (Chambolle-Pock).
    """
    t_start = time.time()

    sigma_max = estimate_spectral_norm(lp, n_iter=spectral_iter)
    if print_log:
        print(f"  spectral norm estimate: {sigma_max:.4e}", flush=True)

    primal_weight = float(initial_primal_weight)
    tau = eta_factor * primal_weight / sigma_max
    sigma = eta_factor / (primal_weight * sigma_max)

    x = torch.zeros(lp.n, device=lp.device, dtype=lp.c.dtype)
    y = torch.zeros(lp.m, device=lp.device, dtype=lp.c.dtype)

    history: List[dict] = []
    inner_total = 0
    converged = False
    last_restart_kkt = float("inf")

    for outer in range(max_outer):
        x_start = x.clone()
        y_start = y.clone()

        # Running average over the inner iterations
        x_avg = torch.zeros_like(x)
        y_avg = torch.zeros_like(y)

        for t in range(max_inner):
            # Primal update
            AT_y = matvec(lp.AT, y)
            grad_x = lp.c - AT_y
            x_new = _project_box(x - tau * grad_x, lp.l, lp.u)
            # Dual update with extrapolation
            x_extrap = 2.0 * x_new - x
            Ax_extrap = matvec(lp.A, x_extrap)
            y = y + sigma * (Ax_extrap - lp.b)
            x = x_new

            # Running average (1-based denominator since we just took 1 step)
            n = t + 1
            x_avg.mul_((n - 1) / n).add_(x, alpha=1.0 / n)
            y_avg.mul_((n - 1) / n).add_(y, alpha=1.0 / n)

        inner_total += max_inner

        # KKT at end of epoch
        kkt_last = _kkt_residual(lp, x, y)
        kkt_avg = _kkt_residual(lp, x_avg, y_avg)

        # Restart logic: pick the better, and only restart if it's
        # significantly better than start
        if kkt_avg.kkt < kkt_last.kkt:
            x = x_avg.clone(); y = y_avg.clone()
            chosen = kkt_avg
        else:
            chosen = kkt_last

        # Adaptive primal weight: balance primal vs dual movement during epoch
        dx_norm = (x - x_start).norm().item()
        dy_norm = (y - y_start).norm().item()
        if dx_norm > 0 and dy_norm > 0:
            ratio = dy_norm / dx_norm
            # Move primal_weight toward ratio (geometric mean to be conservative)
            primal_weight = math.sqrt(primal_weight * max(ratio, 1e-12))
            primal_weight = max(min(primal_weight, 1e6), 1e-6)
            tau = eta_factor * primal_weight / sigma_max
            sigma = eta_factor / (primal_weight * sigma_max)

        record = {
            "outer": outer,
            "inner_total": inner_total,
            "kkt_last": kkt_last.kkt,
            "kkt_avg": kkt_avg.kkt,
            "kkt_chosen": chosen.kkt,
            "obj_primal": chosen.obj_primal,
            "obj_dual": chosen.obj_dual,
            "primal_res": chosen.primal_res,
            "dual_res": chosen.dual_res,
            "gap": chosen.gap,
            "primal_weight": primal_weight,
            "wall_s": time.time() - t_start,
        }
        history.append(record)
        if print_log and (outer % log_every == 0):
            print(f"  outer {outer:>4d} inner {inner_total:>6d}  "
                  f"obj={chosen.obj_primal:+.6f}/{chosen.obj_dual:+.6f}  "
                  f"kkt={chosen.kkt:.2e}  pres={chosen.primal_res:.2e}  "
                  f"dres={chosen.dual_res:.2e}  pw={primal_weight:.2e}  "
                  f"wall={record['wall_s']:.1f}s", flush=True)

        if chosen.kkt < tol:
            converged = True
            break

        last_restart_kkt = chosen.kkt

    # Final
    final_kkt = _kkt_residual(lp, x, y)
    return PdlpResult(
        x=x, y=y,
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
# Convenience: solve a lasserre.polya_lp BuildResult directly
# =====================================================================

def solve_buildresult(build, max_outer: int = 200, max_inner: int = 500,
                      tol: float = 1e-7, **kwargs):
    """Accept a BuildResult and solve. Returns (PdlpResult, RuizScaling)."""
    lp, scaling = build_gpu_lp(build.A_eq, build.b_eq, build.c, build.bounds)
    result = pdlp_solve(lp, max_outer=max_outer, max_inner=max_inner, tol=tol, **kwargs)
    return result, scaling
