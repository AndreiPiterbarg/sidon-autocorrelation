r"""GPU implementation of the SCS algorithm (homogeneous self-dual embedding).

Implements the EXACT same algorithm as SCS 3.x (O'Donoghue 2021,
"Operator Splitting for a Homogeneous Embedding of the Linear
Complementarity Problem"), with GPU-accelerated PSD cone projections
via batched torch.linalg.eigh.

The iteration (from the SCS source code, scs.c):
  1. Construct RHS:  u_t_x =  R_x * v_x,  u_t_y = -R_y * v_y
  2. Solve KKT:      [(R_x + P)  A'; A  -R_y] u_t = u_t
  3. Compute tau:     quadratic formula (root_plus)
  4. Subtract:        u_t -= tau * g   (g precomputed)
  5. Reflect:         u = 2*u_t - v
  6. Cone project:    u[n:n+m] = proj_dual_cone(u[n:n+m])
                      u[n+m] = max(u[n+m], 0)
  7. Moreau:          rsk = R * (v + u - 2*u_t)
  8. DR update:       v += alpha * (u - u_t)
  9. Normalize:       v *= sqrt(l) / ||v||

Usage:
    from scs_gpu import scs_gpu_solve
    sol = scs_gpu_solve(A, b, c, cone, device='cuda')
"""
import numpy as np
import torch
from scipy import sparse as sp
import time as _time
import math

SQRT2 = 1.4142135623730951
INV_SQRT2 = 0.7071067811865476


# =====================================================================
# Svec helpers (same as before)
# =====================================================================

def _build_svec_indices(n, device):
    dim = n * (n + 1) // 2
    rows, cols, is_off = [], [], []
    for j in range(n):
        for i in range(j, n):
            rows.append(i); cols.append(j)
            is_off.append(i != j)
    r = torch.tensor(rows, dtype=torch.long, device=device)
    c = torch.tensor(cols, dtype=torch.long, device=device)
    io = torch.tensor(is_off, dtype=torch.bool, device=device)
    us = torch.ones(dim, dtype=torch.float64, device=device)
    us[io] = INV_SQRT2
    ps = torch.ones(dim, dtype=torch.float64, device=device)
    ps[io] = SQRT2
    return {'n': n, 'dim': dim, 'rows': r, 'cols': c,
            'unpack_scale': us, 'pack_scale': ps}

_svec_cache = {}
def _get_svec(n, dev):
    k = (n, dev)
    if k not in _svec_cache: _svec_cache[k] = _build_svec_indices(n, dev)
    return _svec_cache[k]


class ConeInfo:
    def __init__(self, cone, device):
        self.device = device
        self.n_zero = cone.get('z', 0)
        self.n_nonneg = cone.get('l', 0)
        self.psd_sizes = list(cone.get('s', []))
        psd_start = self.n_zero + self.n_nonneg
        self.psd_offsets = []
        off = psd_start
        for nd in self.psd_sizes:
            sd = nd*(nd+1)//2
            self.psd_offsets.append((off, sd, nd))
            off += sd
        self.total_s_dim = off
        self.n_psd = len(self.psd_sizes)
        self.size_groups = {}
        for i,(so,sd,md) in enumerate(self.psd_offsets):
            self.size_groups.setdefault(md,[]).append((i,so,sd))
        self.svec_indices = {md: _get_svec(md, device) for md in self.size_groups}


# =====================================================================
# Cone projection onto K (primal cone), used for PSD eigendecomp
# =====================================================================

def _project_onto_Kstar(s, cone_info):
    """Project s onto dual cone K* = R^z × R_+^l × S_+^{s1} × ...
    Modifies s in-place.  Vectorized scatter/gather for PSD blocks.

    K* is the dual of K = {0}^z × R_+^l × S_+^{s1} × ...
      - Zero cone {0} has dual R^z (free, no projection needed)
      - R_+ is self-dual
      - S_+ is self-dual
    """
    z = cone_info.n_zero
    l = cone_info.n_nonneg
    # Zero cone dual = R^z: leave s[:z] unchanged
    if l > 0:
        s[z:z+l].clamp_(min=0.0)
    # PSD cones: batched eigendecomp per size group
    for md, grp in cone_info.size_groups.items():
        nc = len(grp)
        idx = cone_info.svec_indices[md]
        rows, cols = idx['rows'], idx['cols']
        us, ps = idx['unpack_scale'], idx['pack_scale']
        sd = idx['dim']
        # Vectorized gather: extract all svec blocks at once
        offsets = torch.tensor([so for _, so, _ in grp],
                               dtype=torch.long, device=cone_info.device)
        # Build index tensor for all cones: (nc, sd)
        local_idx = torch.arange(sd, device=cone_info.device).unsqueeze(0)
        global_idx = offsets.unsqueeze(1) + local_idx  # (nc, sd)
        svec_all = s[global_idx] * us.unsqueeze(0)  # (nc, sd)
        # Scatter into batch matrices
        batch = torch.zeros(nc, md, md, dtype=torch.float64, device=cone_info.device)
        batch[:, rows, cols] = svec_all
        batch[:, cols, rows] = svec_all
        # Batched eigendecomp
        ev, evec = torch.linalg.eigh(batch)
        ev.clamp_(min=0.0)
        proj = evec @ torch.diag_embed(ev) @ evec.transpose(-1, -2)
        # Vectorized scatter back
        svec_proj = proj[:, rows, cols] * ps.unsqueeze(0)  # (nc, sd)
        s[global_idx] = svec_proj


# =====================================================================
# Ruiz equilibration
# =====================================================================

def _cone_blocks(cone, m):
    """Return list of (start, end) for STRUCTURED cone blocks only.

    SCS's enforce_cone_boundaries skips the first z+l rows entirely
    (free/linear cones keep per-row scaling). Only PSD/SOC blocks
    get their rows replaced by the aggregate.
    """
    blocks = []
    off = cone.get('z', 0) + cone.get('l', 0)  # skip z+l
    for nd in cone.get('s', []):
        sd = nd * (nd + 1) // 2
        blocks.append((off, off + sd)); off += sd
    return blocks


def _apply_limit(x):
    """SCS apply_limit: values < 1e-4 map to 1.0 (not 1e-4), cap at 1e4."""
    out = np.where(x < 1e-4, 1.0, x)
    return np.minimum(out, 1e4)


def _enforce_cone_inf(d, blocks):
    """Replace d within each cone block by inf-norm of that block."""
    for s, e in blocks:
        if e > s:
            d[s:e] = np.max(np.abs(d[s:e]))


def _enforce_cone_mean(d, blocks):
    """Replace d within each cone block by mean of that block."""
    for s, e in blocks:
        if e > s:
            d[s:e] = np.mean(d[s:e])


def _ruiz_equilibrate(A_csc, b, c, cone, n_ruiz=25, n_l2=1):
    """Ruiz equilibration matching SCS 3.2.11 exactly.

    25 inf-norm passes + 1 L2 pass.
    Key: row AND col norms computed from the SAME A before applying either.
    Only structured cones (PSD) get aggregated; z+l rows keep per-row scaling.
    """
    A = A_csc.copy().astype(np.float64)
    m, n = A.shape
    D = np.ones(m)
    E = np.ones(n)
    blocks = _cone_blocks(cone, m)

    # --- 25 Ruiz inf-norm passes ---
    for _ in range(n_ruiz):
        # Compute BOTH norms from the SAME A (before applying either)
        A_abs = A.copy(); A_abs.data = np.abs(A_abs.data)
        row_norms = np.array(A_abs.max(axis=1).todense()).ravel()
        col_norms = np.array(A_abs.max(axis=0).todense()).ravel()

        # Enforce cone boundaries on row norms only (SCS: norm_inf)
        _enforce_cone_inf(row_norms, blocks)

        row_norms = _apply_limit(row_norms)
        col_norms = _apply_limit(col_norms)

        d = 1.0 / np.sqrt(row_norms)
        e = 1.0 / np.sqrt(col_norms)

        D *= d; E *= e
        A = sp.diags(d) @ A @ sp.diags(e)  # apply both at once

    # --- 1 L2 pass ---
    for _ in range(n_l2):
        A_csr = A.tocsr()
        # Row L2 norms
        row_l2 = np.zeros(m)
        for i in range(m):
            row = A_csr.data[A_csr.indptr[i]:A_csr.indptr[i+1]]
            row_l2[i] = math.sqrt(np.dot(row, row)) if len(row) > 0 else 0.0
        _enforce_cone_mean(row_l2, blocks)
        row_l2 = _apply_limit(row_l2)

        # Col L2 norms (computed from SAME A)
        A_csc_t = A.tocsc()
        col_l2 = np.zeros(n)
        for j in range(n):
            col = A_csc_t.data[A_csc_t.indptr[j]:A_csc_t.indptr[j+1]]
            col_l2[j] = math.sqrt(np.dot(col, col)) if len(col) > 0 else 0.0
        col_l2 = _apply_limit(col_l2)

        d = 1.0 / np.sqrt(row_l2)
        e = 1.0 / np.sqrt(col_l2)
        D *= d; E *= e
        A = sp.diags(d) @ A @ sp.diags(e)

    # --- Scale b, c ---
    b_s = D * b
    c_s = E * c
    sigma = max(np.max(np.abs(b_s)), np.max(np.abs(c_s)))
    # SCS apply_limit: <1e-4 → 1.0
    sigma = 1.0 if sigma < 1e-4 else min(sigma, 1e4)
    sigma = 1.0 / sigma
    b_s *= sigma
    c_s *= sigma

    A = A.tocsc()
    A.sort_indices()
    return A, b_s, c_s, D, E, sigma


def _scipy_to_torch_csr(A_csc, device):
    A_csr = A_csc.tocsr()
    return torch.sparse_csr_tensor(
        torch.tensor(A_csr.indptr, dtype=torch.int64, device=device),
        torch.tensor(A_csr.indices, dtype=torch.int64, device=device),
        torch.tensor(A_csr.data, dtype=torch.float64, device=device),
        size=A_csr.shape, dtype=torch.float64, device=device)


# =====================================================================
# Anderson acceleration (matches SCS acceleration_lookback)
# =====================================================================

class _AndersonAccel:
    """Type-II Anderson acceleration for the DR fixed-point iteration.

    Stores the last `mem` iterates and computes optimal mixing via
    least-squares on residual differences.  SCS default: mem=10, interval=10.
    """

    def __init__(self, mem, dim, device, dtype=torch.float64):
        self.mem = mem
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.k = 0
        # Circular buffers
        self.S = torch.zeros(mem, dim, dtype=dtype, device=device)  # iterates
        self.Y = torch.zeros(mem, dim, dtype=dtype, device=device)  # residuals (f(x)-x)

    def apply(self, v_new, v_prev):
        """Apply AA: given v_prev (input to iteration) and v_new (output),
        return accelerated v.  v_new = G(v_prev) where G is the DR map."""
        f = v_new - v_prev  # residual

        idx = self.k % self.mem
        self.S[idx] = v_prev.clone()
        self.Y[idx] = f.clone()
        self.k += 1

        mk = min(self.k, self.mem)
        if mk < 2:
            return v_new

        # Build dY matrix (differences of consecutive residuals)
        indices = [(self.k - 1 - i) % self.mem for i in range(mk)]
        Y_act = self.Y[indices]  # (mk, dim)
        dY = Y_act[:-1] - Y_act[1:]  # (mk-1, dim)
        if dY.shape[0] == 0:
            return v_new

        # Solve least-squares: min ||f - dY^T gamma||
        G = dY @ dY.T
        G += 1e-10 * torch.eye(G.shape[0], dtype=self.dtype, device=self.device)
        rhs_aa = dY @ f

        try:
            gamma = torch.linalg.solve(G, rhs_aa)
        except Exception:
            return v_new

        # Accelerated iterate
        dS = self.S[indices[:-1]] - self.S[indices[1:]]
        v_acc = v_new - gamma @ (dY + dS)

        # Safeguard: if AA step is much larger than standard step, reject
        aa_norm = torch.norm(v_acc - v_prev).item()
        std_norm = torch.norm(v_new - v_prev).item()
        if aa_norm > 10.0 * max(std_norm, 1e-10):
            return v_new

        return v_acc

    def reset(self):
        self.k = 0
        self.S.zero_()
        self.Y.zero_()


def _torch_cg(mvfn, rhs, x0, maxiter=200, tol=1e-12):
    x = x0.clone(); r = rhs - mvfn(x); p = r.clone()
    rs = torch.dot(r,r)
    if rs.sqrt() < tol: return x
    for _ in range(maxiter):
        Ap = mvfn(p); pAp = torch.dot(p,Ap)
        if pAp.abs() < 1e-30: break
        a = rs/pAp; x = x+a*p; r = r-a*Ap
        rs2 = torch.dot(r,r)
        if rs2.sqrt() < tol: break
        p = r+(rs2/rs)*p; rs = rs2
    return x


# =====================================================================
# SCS HSDE solver
# =====================================================================

def scs_gpu_solve(A_csc, b_np, c_np, cone, *,
                  max_iters=10000, eps_abs=1e-9, eps_rel=1e-9,
                  alpha=1.5, scale_input=True, rho_x=1e-6,
                  scs_scale=0.1, device='cuda',
                  warm_start=None, verbose=False, check_interval=25):
    """SCS on GPU — exact HSDE algorithm with GPU batched eigendecomp."""
    t_start = _time.time()
    m_orig, n_orig = A_csc.shape

    # ═══ SCALING ═══
    if scale_input:
        A_sc, b_sc, c_sc, D_vec, E_vec, sigma = _ruiz_equilibrate(A_csc, b_np, c_np, cone)
    else:
        A_sc = A_csc.copy().tocsc(); b_sc = b_np.astype(np.float64).copy()
        c_sc = c_np.astype(np.float64).copy()
        D_vec = np.ones(m_orig); E_vec = np.ones(n_orig); sigma = 1.0

    m, n = A_sc.shape
    l = n + m + 1   # total HSDE dimension

    # ═══ GPU TRANSFER ═══
    A_gpu = _scipy_to_torch_csr(A_sc, device)
    AT_gpu = _scipy_to_torch_csr(A_sc.T.tocsc(), device)
    b = torch.tensor(b_sc, dtype=torch.float64, device=device)
    c = torch.tensor(c_sc, dtype=torch.float64, device=device)
    cone_info = ConeInfo(cone, device)

    # ═══ DIAGONAL SCALING R ═══
    # R_x = rho_x, R_y = 1/scs_scale, R_tau = TAU_FACTOR
    TAU_FACTOR = 10.0
    diag_r = torch.zeros(l, dtype=torch.float64, device=device)
    diag_r[:n] = rho_x
    # For zero cone entries, use very small R (SCS uses 1/(1000*scale))
    z_n = cone_info.n_zero
    if z_n > 0:
        diag_r[n:n+z_n] = 1.0 / (1000.0 * scs_scale)
    # For all other cone entries
    diag_r[n+z_n:n+m] = 1.0 / scs_scale
    diag_r[n+m] = TAU_FACTOR

    # ═══ FACTOR KKT SYSTEM ═══
    # [(R_x)  A']   solved as  (R_x + A' R_y^{-1} A) p_x = rhs_x + A' R_y^{-1} rhs_y
    # [ A   -R_y]              p_y = R_y^{-1}(A p_x - rhs_y)
    #
    # For P=0: the system is [(R_x) A'; A -R_y] which gives
    #   R_x p_x + A' p_y = rhs_x
    #   A p_x - R_y p_y = rhs_y
    # => p_y = R_y^{-1}(A p_x - rhs_y)
    # => (R_x + A' R_y^{-1} A) p_x = rhs_x + A' R_y^{-1} rhs_y
    #
    # Let Ry_inv = 1/diag_r[n:n+m]
    Ry = diag_r[n:n+m]
    Ry_inv = 1.0 / Ry

    # M = R_x I + A' diag(Ry_inv) A
    use_dense = n < 5000
    if use_dense:
        # M = R_x I + A' diag(Ry_inv) A — build on CPU as dense, transfer
        Ry_inv_np = Ry_inv.cpu().numpy()
        A_dense_np = A_sc.toarray().astype(np.float64)
        # A' diag(Ry_inv) A
        ATA_np = (A_dense_np.T * Ry_inv_np[None, :]) @ A_dense_np
        M_np = rho_x * np.eye(n) + ATA_np
        M_dense = torch.tensor(M_np, dtype=torch.float64, device=device)
        L_factor = torch.linalg.cholesky(M_dense)
        del M_dense, ATA_np, M_np

        def kkt_solve_x(rhs_x):
            return torch.cholesky_solve(rhs_x.unsqueeze(-1), L_factor).squeeze(-1)
        solver_type = 'cholesky'
    else:
        def kkt_matvec(v):
            Av = torch.mv(A_gpu, v)
            return rho_x * v + torch.mv(AT_gpu, Ry_inv * Av)
        _cg_prev = [torch.zeros(n, dtype=torch.float64, device=device)]
        def kkt_solve_x(rhs_x):
            x = _torch_cg(kkt_matvec, rhs_x, _cg_prev[0])
            _cg_prev[0] = x.clone()
            return x
        solver_type = 'cg'

    def solve_kkt(rhs_x, rhs_y):
        """Solve [(R_x) A'; A -R_y] [p_x; p_y] = [rhs_x; rhs_y].
        Returns (p_x, p_y)."""
        # Schur: (R_x + A' Ry^{-1} A) p_x = rhs_x + A' Ry^{-1} rhs_y
        ATRy_inv_rhs_y = torch.mv(AT_gpu, Ry_inv * rhs_y)
        p_x = kkt_solve_x(rhs_x + ATRy_inv_rhs_y)
        # p_y = Ry^{-1}(A p_x - rhs_y)
        p_y = Ry_inv * (torch.mv(A_gpu, p_x) - rhs_y)
        return p_x, p_y

    # ═══ PRECOMPUTE g = KKT^{-1} [c; -b] ═══
    # Note the sign: g is from [c; -b], NOT [c; b]
    g = torch.zeros(n + m, dtype=torch.float64, device=device)
    g_x, g_y = solve_kkt(c, -b)
    g[:n] = g_x
    g[n:] = g_y

    # Precompute 'a' for root_plus: a = TAU_FACTOR + g' R g
    gRg = torch.dot(g[:n+m] * diag_r[:n+m], g[:n+m])
    root_a = TAU_FACTOR + gRg.item()

    setup_time = _time.time() - t_start
    if verbose:
        print(f"  SCS-GPU setup: {setup_time:.3f}s ({solver_type}, "
              f"n={n}, m={m}, PSD={cone_info.n_psd})")

    # ═══ INITIALIZE ═══
    v = torch.zeros(l, dtype=torch.float64, device=device)
    v[n+m] = math.sqrt(l)  # tau = sqrt(l) after normalization

    u_t = torch.zeros(l, dtype=torch.float64, device=device)
    u = torch.zeros(l, dtype=torch.float64, device=device)
    rsk = torch.zeros(l, dtype=torch.float64, device=device)

    if warm_start is not None:
        # TODO: proper warm-start with scaling
        pass

    # ═══ Helper: rebuild KKT factor + g for current scale ═══
    current_scale = [scs_scale]

    def _set_diag_r(sc):
        diag_r[:n] = rho_x
        z_n_ = cone_info.n_zero
        if z_n_ > 0:
            diag_r[n:n+z_n_] = 1.0 / (1000.0 * sc)
        diag_r[n+z_n_:n+m] = 1.0 / sc
        diag_r[n+m] = TAU_FACTOR

    def _refactor():
        """Rebuild Cholesky/CG and recompute g for current diag_r."""
        nonlocal L_factor, Ry_inv, root_a
        Ry_inv = 1.0 / diag_r[n:n+m]
        if use_dense:
            Ry_inv_np_ = Ry_inv.cpu().numpy()
            ATA_ = (A_dense_np.T * Ry_inv_np_[None, :]) @ A_dense_np
            M_ = rho_x * np.eye(n) + ATA_
            L_factor = torch.linalg.cholesky(
                torch.tensor(M_, dtype=torch.float64, device=device))
        # Recompute g
        gx, gy = solve_kkt(c, -b)
        g[:n] = gx
        g[n:] = gy
        gRg_ = torch.dot(g[:n+m] * diag_r[:n+m], g[:n+m])
        root_a = TAU_FACTOR + gRg_.item()

    # Keep A_dense_np around for refactoring (small n only)
    if use_dense:
        A_dense_np = A_sc.toarray().astype(np.float64)

    # ═══ MAIN ITERATION ═══
    t_solve = _time.time()
    status = 'solved_inaccurate'
    final_k = max_iters
    FEASIBLE_ITERS = 1

    # Anderson acceleration (SCS default: lookback=10, interval=10)
    AA_LOOKBACK = 10
    AA_INTERVAL = 10
    aa = _AndersonAccel(AA_LOOKBACK, l, device) if AA_LOOKBACK > 0 else None
    v_prev = v.clone()

    # Adaptive scale tracking (SCS update_scale)
    RESCALING_MIN_ITERS = 100
    sum_log_factor = 0.0
    n_log_factor = 0
    last_scale_update = 0

    # Pre-allocate temp vectors to avoid per-iteration allocation
    Rg = diag_r[:n+m] * g  # R-weighted g, constant between scale updates
    sqrt_l = math.sqrt(l)

    prev_resid_norm = float('inf')  # for AA restart safeguard

    for k in range(max_iters):
        # ── Anderson acceleration (applied at start, like SCS) ──
        if aa is not None and k > 0 and k % AA_INTERVAL == 0:
            v_candidate = aa.apply(v, v_prev)
            # Residual-based restart: if AA made things worse, reject and reset
            resid = torch.norm(v_candidate - v).item()
            if resid > 2.0 * prev_resid_norm and prev_resid_norm > 0:
                aa.reset()  # flush history, fall back to standard DR step
            else:
                v = v_candidate
            prev_resid_norm = resid
        # Only clone v_prev when AA needs it (next AA step)
        if aa is not None and (k + 1) % AA_INTERVAL == 0:
            v_prev = v.clone()

        # ── Normalize v (after first iteration) ──
        if k >= FEASIBLE_ITERS:
            v_norm = torch.norm(v).item()
            if v_norm > 1e-15:
                v *= sqrt_l / v_norm

        # ── Step 1: Construct RHS and solve KKT ──
        rhs_x = diag_r[:n] * v[:n]
        rhs_y = diag_r[n:n+m] * v[n:n+m]
        rhs_y.neg_()  # in-place negate

        p_x, p_y = solve_kkt(rhs_x, rhs_y)
        u_t[:n] = p_x
        u_t[n:n+m] = p_y

        # ── Step 2: Compute tau via root_plus ──
        if k < FEASIBLE_ITERS:
            tau_tilde = 1.0
        else:
            # Use pre-computed Rg to avoid recomputing diag_r * g each iter
            vRg = torch.dot(v[:n+m], Rg).item()
            pRg = torch.dot(u_t[:n+m], Rg).item()
            root_b = vRg - 2.0 * pRg - v[n+m].item() * TAU_FACTOR

            Rp = diag_r[:n+m] * u_t[:n+m]
            pRp = torch.dot(u_t[:n+m], Rp).item()
            pRv = torch.dot(Rp, v[:n+m]).item()
            root_c = pRp - pRv

            disc = root_b * root_b - 4.0 * root_a * root_c
            disc = max(disc, 0.0)
            tau_tilde = (-root_b + math.sqrt(disc)) / (2.0 * root_a)

        u_t[n+m] = tau_tilde

        # ── Step 3: Subtract g * tau ──
        if tau_tilde != 0.0:
            u_t[:n+m].sub_(g, alpha=tau_tilde)  # u_t -= tau * g (in-place)

        # ── Step 4: Reflect: u = 2*u_t - v ──
        torch.mul(u_t, 2.0, out=u)
        u.sub_(v)

        # ── Step 5: Cone projection ──
        _project_onto_Kstar(u[n:n+m], cone_info)
        if k < FEASIBLE_ITERS:
            u[n+m] = 1.0
        else:
            u[n+m] = max(u[n+m].item(), 0.0)

        # ── Step 6: Compute rsk = R * (v + u - 2*u_t) ──
        # rsk = diag_r * (v + u - 2*u_t)  rewritten to reduce allocs
        torch.add(v, u, out=rsk)
        rsk.sub_(u_t, alpha=2.0)
        rsk.mul_(diag_r)

        # ── Step 7: DR update: v += alpha * (u - u_t) ──
        # v += alpha * (u - u_t) = v + alpha*u - alpha*u_t
        v.add_(u, alpha=alpha)
        v.sub_(u_t, alpha=alpha)

        # ── Step 8: Convergence check + adaptive scale ──
        if (k + 1) % check_interval == 0:
            tau = abs(u[n+m].item())
            kap = abs(rsk[n+m].item())

            if tau > 1e-10:
                x_u = u[:n]
                y_u = u[n:n+m]
                s_rsk = rsk[n:n+m]

                Ax = torch.mv(A_gpu, x_u)
                pri_vec = Ax + s_rsk - b * tau
                pri_res = torch.norm(pri_vec, p=float('inf')).item() / tau

                ATy = torch.mv(AT_gpu, y_u)
                dual_vec = ATy + c * tau
                dual_res = torch.norm(dual_vec, p=float('inf')).item() / tau

                pobj = torch.dot(c, x_u).item() / tau
                dobj = -torch.dot(b, y_u).item() / tau
                gap = abs(pobj - dobj)

                eps_p = eps_abs + eps_rel * max(
                    torch.norm(Ax, p=float('inf')).item()/tau,
                    torch.norm(s_rsk, p=float('inf')).item()/tau,
                    torch.norm(b, p=float('inf')).item())
                eps_d = eps_abs + eps_rel * max(
                    torch.norm(ATy, p=float('inf')).item()/tau,
                    torch.norm(c, p=float('inf')).item())

                if verbose and (k+1) % (check_interval*4) == 0:
                    print(f"    [{k+1}] pri={pri_res:.2e} dual={dual_res:.2e} "
                          f"gap={gap:.2e} tau={tau:.4f} "
                          f"scale={current_scale[0]:.2e} pobj={pobj:.6f}")

                eps_g = eps_abs + eps_rel * max(abs(pobj), abs(dobj))
                if pri_res < eps_p and dual_res < eps_d and gap < eps_g:
                    status = 'solved'
                    final_k = k + 1
                    break

                # ── Adaptive scale (SCS update_scale) ──
                # Accumulate log(pri/dual) ratio
                rel_pri = pri_res / max(
                    torch.norm(Ax, p=float('inf')).item()/tau,
                    torch.norm(s_rsk, p=float('inf')).item()/tau,
                    torch.norm(b, p=float('inf')).item(), 1e-18)
                rel_dual = dual_res / max(
                    torch.norm(ATy, p=float('inf')).item()/tau,
                    torch.norm(c, p=float('inf')).item(), 1e-18)
                rel_pri = max(rel_pri, 1e-18)
                rel_dual = max(rel_dual, 1e-18)
                sum_log_factor += math.log(rel_pri) - math.log(rel_dual)
                n_log_factor += 1

                if (k - last_scale_update >= RESCALING_MIN_ITERS
                        and n_log_factor > 0):
                    factor = math.sqrt(math.exp(
                        sum_log_factor / n_log_factor))
                    if factor > math.sqrt(10) or factor < 1.0/math.sqrt(10):
                        new_scale = current_scale[0] * factor
                        new_scale = max(min(new_scale, 1e6), 1e-6)
                        if verbose:
                            print(f"    [scale] {current_scale[0]:.2e} "
                                  f"-> {new_scale:.2e} (factor={factor:.2f})")
                        current_scale[0] = new_scale
                        # Rebuild diag_r, refactor, recompute g
                        _set_diag_r(new_scale)
                        _refactor()
                        Rg = diag_r[:n+m] * g  # update cached Rg
                        # Remap v: v = rsk/diag_r_new + 2*u_t - u
                        v = rsk / diag_r + 2.0 * u_t - u
                        # Reset accumulators + AA
                        sum_log_factor = 0.0
                        n_log_factor = 0
                        last_scale_update = k
                        if aa is not None:
                            aa.reset()
                        v_prev = v.clone()

            # Infeasibility check
            if tau < 1e-8 and kap > 1e-6:
                y_cert = u[n:n+m]
                y_norm = torch.norm(y_cert).item()
                if y_norm > 1e-12:
                    ATy_c = torch.mv(AT_gpu, y_cert)
                    by = torch.dot(b, y_cert).item()
                    cert_res = torch.norm(ATy_c).item() / y_norm
                    if by < -1e-8 and cert_res < eps_abs * 100:
                        status = 'infeasible'
                        final_k = k + 1
                        break

    solve_time = _time.time() - t_solve

    # ═══ EXTRACT AND UNSCALE ═══
    tau_final = abs(u[n+m].item())
    if tau_final > 1e-10 and status in ('solved', 'solved_inaccurate'):
        # x_orig = E * x_scaled / (sigma * tau)
        # y_orig = D * y_scaled / (sigma * tau)
        # s_orig = s_scaled / (D * sigma * tau)
        x_out = u[:n].cpu().numpy() * E_vec / (sigma * tau_final)
        y_out = u[n:n+m].cpu().numpy() * D_vec / (sigma * tau_final)
        s_out = rsk[n:n+m].cpu().numpy() / (D_vec * sigma * tau_final)
        pobj_out = float(c_np @ x_out)
        dobj_out = float(-b_np @ y_out)
    else:
        x_out = np.zeros(n_orig)
        y_out = np.zeros(m_orig)
        s_out = np.zeros(m_orig)
        pobj_out = float('inf')
        dobj_out = float('-inf')
        if status == 'solved_inaccurate':
            status = 'infeasible_inaccurate'

    if verbose:
        print(f"  SCS-GPU: {final_k} iters, {solve_time:.3f}s "
              f"({solve_time/max(final_k,1)*1000:.2f}ms/iter), "
              f"status={status}, tau={tau_final:.6f}")

    return {
        'x': x_out, 'y': y_out, 's': s_out,
        'info': {'iter': final_k, 'status': status,
                 'pobj': pobj_out, 'dobj': dobj_out,
                 'setup_time': setup_time, 'solve_time': solve_time}
    }


# =====================================================================
# Standalone test
# =====================================================================
if __name__ == '__main__':
    print("SCS-GPU (HSDE) — sanity test")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # min t s.t. [[t,x],[x,t]] >= 0, t+x=1, x>=0
    A = sp.csc_matrix(([1,1,-1,-1,-SQRT2,-1],
                        ([0,0,1,2,3,4],[0,1,0,1,0,1])), shape=(5,2))
    b = np.array([1.0, 0, 0, 0, 0])
    c = np.array([0.0, 1.0])
    cone = {'z': 1, 'l': 1, 's': [2]}

    sol = scs_gpu_solve(A, b, c, cone, device=device,
                        max_iters=5000, eps_abs=1e-8, eps_rel=1e-8,
                        verbose=True)
    print(f"\nGPU: status={sol['info']['status']}, iters={sol['info']['iter']}")
    print(f"  x={sol['x']}, t={sol['x'][1]:.8f} (expect 0.5)")

    try:
        import scs
        s = scs.SCS({'A': A, 'b': b, 'c': c}, cone,
                    max_iters=5000, eps_abs=1e-8, eps_rel=1e-8, verbose=False)
        sc = s.solve()
        print(f"\nCPU: status={sc['info']['status']}, iters={sc['info']['iter']}")
        print(f"  x={sc['x']}, t={sc['x'][1]:.8f}")
    except ImportError:
        pass
