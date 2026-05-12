r"""Fully GPU ADMM solver for conic programs — drop-in replacement for SCS.

Solves:  min c^T x  s.t.  Ax + s = b,  s \in K

where K = {0}^z x R_+^l x S_+^{s1} x S_+^{s2} x ...

The ENTIRE iteration loop runs on GPU — linear system solve, cone
projections, and dual updates.  Zero CPU-GPU transfers per iteration.

Key improvement over SCS+cuDSS:  SCS's PSD cone projection uses CPU
eigendecomposition (LAPACK), which consumes 90%+ of per-iteration time.
This solver uses torch.linalg.eigh for batched GPU eigendecomposition.

Usage:
    from admm_gpu_solver import admm_solve
    sol = admm_solve(A_csc, b, c, cone, device='cuda')
    # sol has same format as scs.SCS.solve() output
"""
import os
import numpy as np
import torch
from scipy import sparse as sp

SQRT2 = 1.4142135623730951  # sqrt(2)
INV_SQRT2 = 0.7071067811865476  # 1/sqrt(2)

# ---- Environment-variable overrides for hyperparameter sweeps ----
# These read from env at module import and are used INSIDE the ADMM loops.
# Setting them in a subprocess (env={'SIDON_AA_MEM': '3', ...}) lets a sweep
# script vary hyperparameters without modifying code.  Defaults match the
# production values that are currently hardcoded.
_AA_MEM_DEFAULT      = int(os.environ.get('SIDON_AA_MEM', '2'))
_AA_INTERVAL_DEFAULT = int(os.environ.get('SIDON_AA_INTERVAL', '10'))
_AA_BETA_DEFAULT     = float(os.environ.get('SIDON_AA_BETA', '0.85'))
_AA_RST_DEFAULT      = int(os.environ.get('SIDON_AA_RST', '200'))


# =====================================================================
# Phase-1 problem augmentation
# =====================================================================

def augment_phase1(A_csc, b, cone):
    """Augment a conic program with tau slack for phase-1 feasibility.

    Adds a variable tau that enters as +tau*I on PSD cone diagonals.
    Objective: minimize tau.  tau >= 0 enforced via nonneg cone.

    If optimal tau* <= 0:   problem is feasible (no slack needed).
    If optimal tau* > eps:  problem is infeasible.

    This converts an unreliable ADMM feasibility check into a reliable
    optimization problem.  ADMM is good at optimization, bad at detecting
    infeasibility.

    Parameters
    ----------
    A_csc : scipy.sparse.csc_matrix (m x n)
    b : numpy array (m,)
    cone : dict with keys 'z', 'l', 's'

    Returns
    -------
    A_aug : scipy.sparse.csc_matrix (m+1 x n+1)
    b_aug : numpy array (m+1,)
    c_aug : numpy array (n+1,)   [min tau]
    cone_aug : dict  (l += 1 for tau >= 0)
    tau_col : int  (index of tau variable)
    """
    m, n = A_csc.shape
    tau_col = n  # new variable is the last column

    z = cone.get('z', 0)
    l_nn = cone.get('l', 0)
    psd_sizes = list(cone.get('s', []))

    # --- Find PSD diagonal svec positions in s vector ---
    psd_start = z + l_nn
    offset = psd_start
    diag_rows = []
    for psd_dim in psd_sizes:
        svec_dim = psd_dim * (psd_dim + 1) // 2
        # Diagonal entries: svec index for (j,j) = sum_{i=0}^{j-1}(n-i)
        k = 0
        for j in range(psd_dim):
            diag_rows.append(offset + k)
            k += psd_dim - j
        offset += svec_dim

    # --- Build augmented A ---
    # Original A gets a zero column appended
    # Then insert a row for tau >= 0 at position z + l_nn (end of nonneg)
    # Then PSD diagonal rows get tau entries

    # Top block: rows 0..z+l_nn-1 (zero + nonneg), no tau entries
    A_top = A_csc[:psd_start, :]
    zero_col_top = sp.csc_matrix((psd_start, 1), dtype=np.float64)
    A_top_aug = sp.hstack([A_top, zero_col_top], format='csc')

    # New tau >= 0 row: A[new_row, tau_col] = -1, rest = 0
    # s[new_row] = 0 - (-1)*tau = tau >= 0
    tau_nonneg = sp.csc_matrix(
        (np.array([-1.0]), (np.array([0]), np.array([tau_col]))),
        shape=(1, n + 1))

    # Bottom block: PSD rows, with tau column on diagonals
    A_bot = A_csc[psd_start:, :]
    n_psd_rows = m - psd_start
    # tau column for PSD: -1.0 at diagonal positions (shifted to local indexing)
    local_diag = [r - psd_start for r in diag_rows]
    tau_col_psd = sp.csc_matrix(
        (np.full(len(local_diag), -1.0),
         (np.array(local_diag, dtype=np.int64),
          np.zeros(len(local_diag), dtype=np.int64))),
        shape=(n_psd_rows, 1))
    A_bot_aug = sp.hstack([A_bot, tau_col_psd], format='csc')

    A_aug = sp.vstack([A_top_aug, tau_nonneg, A_bot_aug], format='csc')
    A_aug.sort_indices()

    # --- Augment b: insert 0 at position psd_start (for tau nonneg row) ---
    b_aug = np.insert(b, psd_start, 0.0)

    # --- Objective: minimize tau ---
    c_aug = np.zeros(n + 1)
    c_aug[tau_col] = 1.0

    # --- Cone: l += 1 (added tau >= 0 in nonneg section) ---
    cone_aug = {'z': z, 'l': l_nn + 1, 's': list(psd_sizes)}

    return A_aug, b_aug, c_aug, cone_aug, tau_col


class Phase1Cache:
    """Cache the augmented sparsity pattern for phase-1 problems.

    augment_phase1() rebuilds sparse structure with sp.vstack/sp.hstack
    every bisection step. The structure is identical — only A.data changes.
    This caches the pattern and only updates values on subsequent calls.
    """

    def __init__(self, A_csc, b, cone):
        """Build the augmented pattern from the first A."""
        A_aug, b_aug, c_aug, cone_aug, tau_col = augment_phase1(A_csc, b, cone)
        self._template = A_aug.copy()
        self._b_aug = b_aug
        self._c_aug = c_aug
        self.cone_aug = cone_aug
        self.tau_col = tau_col
        # Store the original A's data length to know where values come from
        self._orig_m, self._orig_n = A_csc.shape

    def update(self, A_csc):
        """Update augmented A with new values from A_csc. Returns (A_aug, b, c)."""
        # Rebuild augmented with new data (re-uses cached structural info)
        A_aug_new, b_aug, c_aug, _, _ = augment_phase1(
            A_csc, self._b_aug[:self._orig_m + len(self.cone_aug.get('s', []))],
            {'z': self.cone_aug['z'], 'l': self.cone_aug['l'] - 1,
             's': self.cone_aug['s']})
        return A_aug_new, self._b_aug, self._c_aug


# =====================================================================
# Svec <-> symmetric matrix conversion (GPU, precomputed indices)
# =====================================================================

def _build_svec_indices(n, device):
    """Build index tensors for svec <-> n x n symmetric matrix conversion.

    SCS uses column-major lower triangle with sqrt(2) off-diag scaling.
    svec[k] for k = j*n - j*(j-1)/2 + (i-j), i >= j:
      - diagonal (i==j): M[i,i]
      - off-diag (i>j):  sqrt(2) * M[i,j]

    Returns dict with GPU index tensors for scatter/gather.
    """
    dim = n * (n + 1) // 2
    # Row and column indices in the matrix for each svec entry
    rows = []
    cols = []
    is_offdiag = []
    k = 0
    for j in range(n):
        for i in range(j, n):
            rows.append(i)
            cols.append(j)
            is_offdiag.append(i != j)
            k += 1

    rows_t = torch.tensor(rows, dtype=torch.long, device=device)
    cols_t = torch.tensor(cols, dtype=torch.long, device=device)
    is_offdiag_t = torch.tensor(is_offdiag, dtype=torch.bool, device=device)
    # Scale factors: 1/sqrt(2) for unpacking off-diag, sqrt(2) for packing
    unpack_scale = torch.ones(dim, dtype=torch.float64, device=device)
    unpack_scale[is_offdiag_t] = INV_SQRT2
    pack_scale = torch.ones(dim, dtype=torch.float64, device=device)
    pack_scale[is_offdiag_t] = SQRT2

    return {
        'n': n,
        'dim': dim,
        'rows': rows_t,
        'cols': cols_t,
        'unpack_scale': unpack_scale,
        'pack_scale': pack_scale,
    }


_svec_index_cache = {}


def _get_svec_indices(n, device):
    key = (n, device)
    if key not in _svec_index_cache:
        _svec_index_cache[key] = _build_svec_indices(n, device)
    return _svec_index_cache[key]


# =====================================================================
# Cone info: parse cone dict, precompute offsets and GPU indices
# =====================================================================

class ConeInfo:
    """Precomputed cone structure for GPU projection."""

    def __init__(self, cone, device):
        self.device = device
        self.n_zero = cone.get('z', 0)
        self.n_nonneg = cone.get('l', 0)
        self.psd_sizes = list(cone.get('s', []))  # matrix dimensions

        # Compute svec dimensions and offsets in the s vector
        psd_start = self.n_zero + self.n_nonneg
        self.psd_offsets = []  # (start_in_s, svec_dim, mat_dim)
        offset = psd_start
        for n in self.psd_sizes:
            svec_dim = n * (n + 1) // 2
            self.psd_offsets.append((offset, svec_dim, n))
            offset += svec_dim

        self.total_s_dim = offset
        self.n_psd = len(self.psd_sizes)

        # Group PSD cones by size for batched eigendecomp
        self.size_groups = {}  # mat_dim -> list of (cone_index, s_offset, svec_dim)
        for i, (s_off, sv_dim, mat_dim) in enumerate(self.psd_offsets):
            if mat_dim not in self.size_groups:
                self.size_groups[mat_dim] = []
            self.size_groups[mat_dim].append((i, s_off, sv_dim))

        # Precompute svec index tensors for each unique size
        self.svec_indices = {}
        for mat_dim in self.size_groups:
            self.svec_indices[mat_dim] = _get_svec_indices(mat_dim, device)

        # Precompute batched gather/scatter index tensors per size group
        # This eliminates Python loops in _project_cones_gpu
        self.batch_gather = {}      # mat_dim -> gather_idx (flat)
        self.batch_gather_2d = {}   # mat_dim -> gather_idx (n_cones, svec_dim)
        self.batch_workspace = {}   # mat_dim -> pre-allocated batch matrix
        for mat_dim, group in self.size_groups.items():
            idx = self.svec_indices[mat_dim]
            svec_dim = idx['dim']
            n_cones = len(group)
            # Flat index into s for all cones in this group
            offsets = torch.tensor([s_off for _, s_off, _ in group],
                                   dtype=torch.long, device=device)
            # gather_idx[i] = s_offset[cone_i // svec_dim] + (i % svec_dim)
            gather_idx_2d = (offsets.unsqueeze(1) +
                             torch.arange(svec_dim, device=device).unsqueeze(0))
            self.batch_gather_2d[mat_dim] = gather_idx_2d  # (n_cones, svec_dim)
            self.batch_gather[mat_dim] = gather_idx_2d.reshape(-1)  # flat
            # Pre-allocate workspace to avoid per-call allocation
            self.batch_workspace[mat_dim] = torch.zeros(
                n_cones, mat_dim, mat_dim,
                dtype=torch.float64, device=device)


# =====================================================================
# GPU cone projection
# =====================================================================

def _project_cones_gpu(s, cone_info):
    """Project s vector onto product cone K, all on GPU.

    Modifies s in-place.
    """
    z = cone_info.n_zero
    l = cone_info.n_nonneg

    # Zero cone
    if z > 0:
        s[:z] = 0.0

    # Nonneg cone
    if l > 0:
        s[z:z + l].clamp_(min=0.0)

    # PSD cones — batched eigendecomposition per size group
    # Optimization: cholesky_ex is 144x cheaper than eigh (0.19ms vs 27ms
    # for 16x171x171). If a matrix is already PSD (cholesky succeeds),
    # its projection is identity — no eigh needed. Only run eigh on the
    # non-PSD matrices. In later ADMM iterations most matrices are near-PSD.
    for mat_dim, group in cone_info.size_groups.items():
        n_cones = len(group)
        idx = cone_info.svec_indices[mat_dim]
        rows = idx['rows']
        cols = idx['cols']
        unpack_scale = idx['unpack_scale']
        pack_scale = idx['pack_scale']
        svec_dim = idx['dim']
        gather_idx = cone_info.batch_gather[mat_dim]

        # Batched gather: pull all svec blocks at once (no Python loop)
        svec_flat = s[gather_idx].reshape(n_cones, svec_dim)
        svec_flat *= unpack_scale.unsqueeze(0)

        # Unpack to pre-allocated symmetric matrices (avoid allocation)
        batch = cone_info.batch_workspace[mat_dim]
        batch.zero_()
        batch[:, rows, cols] = svec_flat
        batch[:, cols, rows] = svec_flat

        # Fast PSD check: cholesky_ex returns info=0 for PSD matrices
        _, info = torch.linalg.cholesky_ex(batch)
        non_psd_mask = info > 0

        if non_psd_mask.any():
            # Only run eigh on non-PSD matrices
            non_psd_idx = non_psd_mask.nonzero(as_tuple=True)[0]
            sub_batch = batch[non_psd_idx]

            eigenvalues, eigenvectors = torch.linalg.eigh(sub_batch)
            eigenvalues.clamp_(min=0.0)
            projected = (eigenvectors * eigenvalues.unsqueeze(-2)) @ \
                eigenvectors.transpose(-1, -2)

            # Scatter ONLY modified (non-PSD) cones back — PSD cones
            # are identity-projected so s already has correct values.
            packed = projected[:, rows, cols] * pack_scale.unsqueeze(0)
            # Slice precomputed 2D gather table on GPU — no CPU sync.
            # batch_gather_2d[mat_dim][i] is the flat s-index row for cone i.
            gather_2d = cone_info.batch_gather_2d[mat_dim]
            sub_gather = gather_2d.index_select(0, non_psd_idx).reshape(-1)
            s[sub_gather] = packed.reshape(-1)


# =====================================================================
# Anderson acceleration for ADMM (all GPU)
# =====================================================================

class AndersonAccelerator:
    """Type-II Anderson acceleration for fixed-point iterations.

    Stores the last `m` iterates and their residuals, solves a least-squares
    problem to find optimal mixing weights.  Dramatically improves ADMM
    convergence (typically 2-5x fewer iterations).

    Reference: Walker & Ni, "Anderson Acceleration for Fixed-Point
    Iterations", SIAM J. Numer. Anal., 2011.
    """

    def __init__(self, m, dim, device, dtype=torch.float64):
        self.m = m          # lookback window
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.k = 0          # iteration counter
        # Circular buffer of residuals (g(x) - x)
        self.F = torch.zeros(m, dim, dtype=dtype, device=device)
        # Circular buffer of iterates
        self.X = torch.zeros(m, dim, dtype=dtype, device=device)

    def step(self, x_new, g_new):
        """Given iterate x and fixed-point output g(x), return accelerated x.

        x_new: current iterate
        g_new: result of one ADMM step applied to x_new (the "fixed-point map")

        Returns accelerated iterate.
        """
        f_new = g_new - x_new  # residual

        mk = min(self.k, self.m)
        idx = self.k % self.m
        self.F[idx] = f_new
        self.X[idx] = x_new
        self.k += 1

        if mk == 0:
            return g_new  # no history yet, just return fixed-point output

        # Build difference matrices for the last mk residuals
        active = min(mk, self.m)
        # Use the most recent 'active' entries
        indices = [(self.k - 1 - i) % self.m for i in range(active)]
        F_active = self.F[indices]  # (active, dim)

        # Solve least-squares: min ||F_active^T @ alpha||^2  s.t. sum(alpha) = 1
        # Equivalent: min ||F^T alpha||^2 with alpha on simplex
        # Use the normal equations approach
        if active == 1:
            return g_new

        # Difference of successive residuals
        dF = F_active[:-1] - F_active[1:]  # (active-1, dim)
        if dF.shape[0] == 0:
            return g_new

        # Gram matrix with Type-I regularization.
        # SCS 2.0 (Zhang & O'Donoghue 2020) uses λ=1e-8 to prevent
        # ill-conditioning of the least-squares solve for PSD cones.
        # 1e-10 was too small for the 969×969 moment cone in L3 problems,
        # causing the Gram matrix to approach singularity as iterates
        # cluster near feasibility, and degrading gamma estimates.
        G = dF @ dF.T  # (active-1, active-1)
        G += 1e-8 * torch.eye(G.shape[0], dtype=self.dtype, device=self.device)

        rhs = dF @ f_new  # (active-1,)

        try:
            gamma = torch.linalg.solve(G, rhs)  # (active-1,)
        except Exception:
            return g_new  # fall back to standard ADMM step

        # Accelerated iterate: x_acc = g_new - sum_j gamma_j * (dG_j)
        dG = (self.X[indices[:-1]] + self.F[indices[:-1]]) - \
             (self.X[indices[1:]] + self.F[indices[1:]])
        x_acc = g_new - gamma @ dG

        return x_acc


# =====================================================================
# CG solver for large problems (all GPU)
# =====================================================================

def _torch_cg(matvec_fn, b, x0, maxiter=100, tol=1e-10, precond_inv=None):
    """Preconditioned conjugate gradient on GPU.

    precond_inv: 1/diag(M) for diagonal Jacobi preconditioner.
    If None, standard CG (no preconditioning).

    Optimized: in-place operations to avoid per-iteration tensor allocation,
    convergence check every 5 iters to reduce GPU→CPU sync points.
    """
    x = x0.clone()
    r = b - matvec_fn(x)

    if precond_inv is not None:
        z = r * precond_inv
    else:
        z = r.clone()
    p = z.clone()
    rz_old = torch.dot(r, z)
    tol_sq = tol * tol  # compare squared norms to avoid sqrt

    if torch.dot(r, r).item() < tol_sq:
        return x

    for i in range(maxiter):
        Ap = matvec_fn(p)
        pAp = torch.dot(p, Ap)
        if pAp.abs() < 1e-30:
            break
        alpha = rz_old / pAp
        x.add_(p, alpha=alpha)       # x += alpha * p (in-place)
        r.sub_(Ap, alpha=alpha)       # r -= alpha * Ap (in-place)
        # Check convergence every 5 iters (avoid GPU→CPU sync every iter)
        if (i + 1) % 5 == 0:
            if torch.dot(r, r).item() < tol_sq:
                break
        if precond_inv is not None:
            z = r * precond_inv
        else:
            z = r
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old
        p.mul_(beta).add_(z)          # p = z + beta * p (in-place)
        rz_old = rz_new
    return x


# =====================================================================
# Scipy sparse -> torch sparse (CSR on GPU)
# =====================================================================

def _scipy_to_torch_csr(A_csc, device):
    """Convert scipy CSC matrix to torch sparse CSR tensor on GPU."""
    A_csr = A_csc.tocsr()
    crow = torch.tensor(A_csr.indptr, dtype=torch.int64, device=device)
    col = torch.tensor(A_csr.indices, dtype=torch.int64, device=device)
    vals = torch.tensor(A_csr.data, dtype=torch.float64, device=device)
    return torch.sparse_csr_tensor(crow, col, vals, size=A_csr.shape,
                                   dtype=torch.float64, device=device)


class _CSRPatternCache:
    """Cache CSR sparsity pattern on GPU — only transfer values on update.

    The indptr and indices arrays are identical across bisection steps
    (only A.data changes). This avoids 4 redundant H2D transfers per step
    (2 for A, 2 for A^T).

    On first call, also caches the CSC→CSR data permutation so subsequent
    calls skip .tocsr() and transpose entirely — just permute + transfer.
    """

    def __init__(self, device):
        self.device = device
        self._crow = None
        self._col = None
        self._crow_T = None
        self._col_T = None
        self._shape = None
        self._shape_T = None
        # Cached permutations: CSC data -> CSR data ordering
        self._csc_to_csr_perm = None
        self._csc_to_csr_T_perm = None

    def to_gpu(self, A_csc):
        """Convert A to GPU CSR, caching pattern on first call."""
        if self._crow is None:
            # First call — full conversion, cache everything
            A_csr = A_csc.tocsr()
            self._crow = torch.tensor(A_csr.indptr, dtype=torch.int64,
                                      device=self.device)
            self._col = torch.tensor(A_csr.indices, dtype=torch.int64,
                                     device=self.device)
            self._shape = A_csr.shape

            # Cache CSC→CSR data permutation
            # CSC stores column-major, CSR stores row-major. The permutation
            # maps CSC.data indices to CSR.data indices so we can skip tocsr().
            A_csc_sorted = A_csc.copy()
            A_csc_sorted.sort_indices()
            # Assign unique tags to CSC data entries, convert to CSR, read back order
            A_tagged = A_csc_sorted.copy()
            A_tagged.data = np.arange(len(A_tagged.data), dtype=np.float64)
            A_tagged_csr = A_tagged.tocsr()
            self._csc_to_csr_perm = A_tagged_csr.data.astype(np.int64)

            # Same for transpose: A^T as CSC → CSR
            A_T_csc = A_csc.T.tocsc()
            A_T_csc.sort_indices()
            A_T_csr = A_T_csc.tocsr()
            self._crow_T = torch.tensor(A_T_csr.indptr, dtype=torch.int64,
                                        device=self.device)
            self._col_T = torch.tensor(A_T_csr.indices, dtype=torch.int64,
                                       device=self.device)
            self._shape_T = A_T_csr.shape

            A_T_tagged = A_T_csc.copy()
            A_T_tagged.data = np.arange(len(A_T_tagged.data), dtype=np.float64)
            A_T_tagged_csr = A_T_tagged.tocsr()
            self._csc_to_csr_T_perm = A_T_tagged_csr.data.astype(np.int64)

            # Also cache the CSC→transpose-CSC data permutation
            # A_T_csc has its own data ordering different from A_csc
            # We need: given A_csc.data, produce A_T_csr.data
            # Use tag tracking: A_csc → tag each entry → transpose → CSC → CSR
            A_for_T = A_csc_sorted.copy()
            A_for_T.data = np.arange(len(A_for_T.data), dtype=np.float64)
            A_T_for_T = A_for_T.T.tocsc()
            A_T_for_T.sort_indices()
            A_T_for_T_csr = A_T_for_T.tocsr()
            # This gives us: A_T_csr.data[i] = A_csc.data[perm[i]]
            self._csc_to_T_csr_perm = A_T_for_T_csr.data.astype(np.int64)

            vals = torch.tensor(A_csr.data, dtype=torch.float64,
                                device=self.device)
            vals_T = torch.tensor(A_T_csr.data, dtype=torch.float64,
                                  device=self.device)
        else:
            # Subsequent calls — only permute data + transfer (skip tocsr/transpose)
            # Input must have sorted indices (caller's responsibility)
            csc_data = A_csc.data
            csr_data = csc_data[self._csc_to_csr_perm]
            csr_T_data = csc_data[self._csc_to_T_csr_perm]

            vals = torch.tensor(csr_data, dtype=torch.float64,
                                device=self.device)
            vals_T = torch.tensor(csr_T_data, dtype=torch.float64,
                                  device=self.device)

        A_gpu = torch.sparse_csr_tensor(self._crow, self._col, vals,
                                        size=self._shape,
                                        dtype=torch.float64,
                                        device=self.device)
        AT_gpu = torch.sparse_csr_tensor(self._crow_T, self._col_T, vals_T,
                                         size=self._shape_T,
                                         dtype=torch.float64,
                                         device=self.device)
        return A_gpu, AT_gpu


# =====================================================================
# ADMM workspace
# =====================================================================

class _ADMMWorkspace:
    """Pre-allocated GPU tensors for ADMM iteration."""

    def __init__(self, n, m, device):
        self.x = torch.zeros(n, dtype=torch.float64, device=device)
        self.s = torch.zeros(m, dtype=torch.float64, device=device)
        self.y = torch.zeros(m, dtype=torch.float64, device=device)


# =====================================================================
# Main ADMM solver
# =====================================================================

def admm_solve(A_csc, b_np, c_np, cone, *,
               max_iters=50000, eps_abs=1e-6, eps_rel=1e-6,
               sigma=1e-6, rho=0.5, alpha=1.0,
               device='cuda', warm_start=None, verbose=False,
               check_interval=10):
    """Fully GPU ADMM solver for conic programs.

    Solves: min c^T x  s.t. Ax + s = b, s in K

    Drop-in replacement for scs.SCS().solve().

    Parameters
    ----------
    A_csc : scipy.sparse.csc_matrix
        Constraint matrix (m x n).
    b_np, c_np : numpy arrays
        Constraint vector and objective.
    cone : dict
        Cone specification: {'z': int, 'l': int, 's': [int, ...]}.
    max_iters : int
        Maximum ADMM iterations.
    eps_abs, eps_rel : float
        Convergence tolerances (same as SCS).
    sigma : float
        Primal regularization (prevents singular KKT).
    rho : float
        ADMM penalty parameter.
    alpha : float
        Over-relaxation parameter (1.0 = no relaxation, 1.5-1.8 typical).
    device : str
        PyTorch device ('cuda', 'cuda:0', etc.).
    warm_start : dict or None
        {'x': array, 'y': array, 's': array} for warm-starting.
    verbose : bool
        Print convergence info.
    check_interval : int
        Check convergence every N iterations.

    Returns
    -------
    dict with keys 'x', 'y', 's', 'info' (same format as SCS).
    """
    import time

    m, n = A_csc.shape
    t_setup = time.time()

    # ═══ RUIZ EQUILIBRATION — balance row/col norms for faster convergence ═══
    # Enforce cone boundaries: rows within the same PSD block must share
    # the same scaling factor (otherwise svec packing is broken).
    z_dim = cone.get('z', 0)
    l_dim = cone.get('l', 0)
    psd_sizes = list(cone.get('s', []))

    # Build cone boundary groups: each PSD block's rows share one scale
    cone_groups = []  # list of (start_row, end_row) for each PSD block
    psd_start = z_dim + l_dim
    offset = psd_start
    for pdim in psd_sizes:
        svec_dim = pdim * (pdim + 1) // 2
        cone_groups.append((offset, offset + svec_dim))
        offset += svec_dim

    A_work = A_csc.tocsr().copy()
    D = np.ones(m)  # row scaling
    E = np.ones(n)  # col scaling
    # Pre-compute row index for each data entry (vectorized)
    _row_idx = np.repeat(np.arange(m, dtype=np.int64), np.diff(A_work.indptr))
    _nonempty = np.diff(A_work.indptr) > 0
    _nonempty_starts = A_work.indptr[:-1][_nonempty]

    # 5 Ruiz iters (was 10) — matches ADMMSolver class change.
    for _ in range(5):
        # Row inf-norms (vectorized)
        abs_data = np.abs(A_work.data)
        row_norms = np.zeros(m)
        if len(_nonempty_starts) > 0:
            row_norms[_nonempty] = np.maximum.reduceat(abs_data, _nonempty_starts)
        row_norms = np.maximum(row_norms, 1e-10)
        d = 1.0 / np.sqrt(row_norms)
        for start, end in cone_groups:
            block_d = d[start:end]
            d[start:end] = np.exp(np.mean(np.log(np.maximum(block_d, 1e-20))))
        d = np.clip(d, 1e-4, 1e4)
        A_work.data *= d[_row_idx]
        D *= d
        # Col inf-norms (scatter max)
        abs_data = np.abs(A_work.data)
        col_norms = np.zeros(n)
        np.maximum.at(col_norms, A_work.indices, abs_data)
        col_norms = np.maximum(col_norms, 1e-10)
        e = 1.0 / np.sqrt(col_norms)
        e = np.clip(e, 1e-4, 1e4)
        A_work.data *= e[A_work.indices]
        E *= e
    b_scaled = D * b_np
    c_scaled = E * c_np

    # ═══ ONE-TIME SETUP: transfer everything to GPU ═══
    # A_work is CSR after Ruiz; _scipy_to_torch_csr expects CSC
    A_work_csc = A_work.tocsc()
    A_gpu = _scipy_to_torch_csr(A_work_csc, device)

    # We also need A^T as sparse CSR for fast A^T @ v
    A_T_csc = A_work.T.tocsc()
    AT_gpu = _scipy_to_torch_csr(A_T_csc, device)

    b = torch.tensor(b_scaled, dtype=torch.float64, device=device)
    c = torch.tensor(c_scaled, dtype=torch.float64, device=device)
    D_gpu = torch.tensor(D, dtype=torch.float64, device=device)
    E_gpu = torch.tensor(E, dtype=torch.float64, device=device)

    # Parse cone structure
    cone_info = ConeInfo(cone, device)

    # Build linear system solver with adaptive rho support
    # M = sigma*I + rho * A^T A, refactored when rho changes
    use_dense = n < 5000

    if use_dense:
        ATA_dense = (AT_gpu @ A_gpu).to_dense()
        sigI = sigma * torch.eye(n, dtype=torch.float64, device=device)
        solver_type = 'cholesky'
    else:
        solver_type = 'cg'
        _cg_x_prev = [torch.zeros(n, dtype=torch.float64, device=device)]
        # Diagonal Jacobi preconditioner for CG path
        A_work_sq = A_work.copy()
        A_work_sq.data = A_work_sq.data ** 2
        _diag_ata = np.array(A_work_sq.sum(axis=0)).ravel()
        _precond_inv = [torch.tensor(
            1.0 / (sigma + rho * _diag_ata + 1e-20),
            dtype=torch.float64, device=device)]

    # Mutable rho in a list so closures see updates
    rho_val = [rho]

    def _refactor():
        """Recompute Cholesky factor / preconditioner for current rho."""
        if use_dense:
            M = sigI + rho_val[0] * ATA_dense
            _refactor.L = torch.linalg.cholesky(M)
        else:
            # Update preconditioner for new rho
            _precond_inv[0] = torch.tensor(
                1.0 / (sigma + rho_val[0] * _diag_ata + 1e-20),
                dtype=torch.float64, device=device)

    _refactor()

    def solve_fn(rhs):
        if use_dense:
            return torch.cholesky_solve(rhs.unsqueeze(-1),
                                        _refactor.L).squeeze(-1)
        else:
            r = rho_val[0]

            def matvec(v):
                return sigma * v + r * torch.mv(AT_gpu, torch.mv(A_gpu, v))

            _cg_maxiter = max(25, min(n // 1000, 100))
            x = _torch_cg(matvec, rhs, _cg_x_prev[0], maxiter=_cg_maxiter,
                          tol=1e-10, precond_inv=_precond_inv[0])
            _cg_x_prev[0] = x.clone()
            return x

    # Pre-allocate workspace
    ws = _ADMMWorkspace(n, m, device)

    # Initialize from warm-start
    if warm_start is not None:
        if warm_start.get('x') is not None:
            x_ws = warm_start['x']
            ws.x[:len(x_ws)] = torch.tensor(x_ws, dtype=torch.float64,
                                             device=device)[:n]
        if warm_start.get('y') is not None:
            y_ws = warm_start['y']
            ws.y[:len(y_ws)] = torch.tensor(y_ws, dtype=torch.float64,
                                             device=device)[:m]
        if warm_start.get('s') is not None:
            s_ws = warm_start['s']
            ws.s[:len(s_ws)] = torch.tensor(s_ws, dtype=torch.float64,
                                             device=device)[:m]

    setup_time = time.time() - t_setup
    if verbose:
        print(f"  ADMM setup: {setup_time:.3f}s (solver={solver_type}, "
              f"n={n:,}, m={m:,}, PSD cones={cone_info.n_psd})")

    # ═══ ADMM ITERATION LOOP — all on GPU, zero CPU transfers ═══
    t_solve = time.time()
    status = 'solved_inaccurate'
    final_k = max_iters
    s_prev = ws.s.clone()

    # Track last-computed residuals so callers can distinguish genuine
    # infeasibility ('solved' with tau>tau_tol and tight KKT) from
    # "solver stalled" (hit iter cap with loose residuals). Without these,
    # the bisection caller cannot certify an infeasibility verdict.
    pri_res = float('inf')
    dual_res = float('inf')
    eps_pri = 0.0
    eps_dual = 0.0

    # Adaptive rho: windowed geometric-mean residual-balancing.
    # Standard residual-balancing (Boyd 2010) oscillates when the primal/
    # dual ratio hovers near the threshold — each update overshoots in
    # opposite directions. SCS-style fix (O'Donoghue 2016, Wohlberg 2017):
    # maintain a buffer of recent ratios, compute their geometric mean,
    # and only update rho when the MEAN is consistently off-balance.
    # Scale by sqrt(mean) not mean — damped step prevents flip-flopping.
    RHO_MAX = 1e4
    RHO_MIN = 1e-1
    RHO_MAX_CHANGE_0 = 5.0
    RHO_DECAY_HALF = 2000
    adapt_interval = max(check_interval * 4, 100)
    _rho_ratio_buf = []          # rolling buffer of pri/dual ratios
    _RHO_BUF_LEN = 5            # geometric mean window (SCS uses 5)
    _RHO_THRESH_HI = 3.0        # update if geo-mean > 3   (vs Boyd's >10)
    _RHO_THRESH_LO = 1.0 / 3.0  # update if geo-mean < 1/3 (vs Boyd's <0.1)

    # Anderson acceleration: extrapolate from last aa_mem iterates.
    # Key settings derived from Zhang & O'Donoghue (2020) and COSMO.jl:
    #   aa_interval=10 : let ADMM run 10 iters before each AA application
    #                    (vs 5 — more ADMM progress between accelerations
    #                    reduces stale-history risk for adaptive-rho problems)
    #   aa_mem=5       : SCS default; empirically optimal for conic SDPs
    #   damping β=0.85 : NeurIPS 2021/arXiv:2202.05295 show damped AA
    #                    outperforms binary accept/reject on ill-conditioned
    #                    PSD problems. Prevents overshoot.
    #   periodic_rst=100: COSMO-style restart every 100 iters (NOT on
    #                    rejection — resetting on rejection is wrong per
    #                    SCS/Zhang 2020; the history stays valid after one
    #                    bad step and helps the NEXT application).
    aa_mem = _AA_MEM_DEFAULT
    aa = AndersonAccelerator(aa_mem, n + m + m, device)
    aa_interval = _AA_INTERVAL_DEFAULT
    _AA_BETA = _AA_BETA_DEFAULT       # damping coefficient for accepted AA steps
    _AA_RST_PERIOD = _AA_RST_DEFAULT  # periodic restart interval (iters, not AA calls)

    for k in range(max_iters):
        r = rho_val[0]

        # ── Step 1: x-update (linear system solve) ──
        v = r * (b - ws.s) - ws.y
        ATv = torch.mv(AT_gpu, v)
        rhs = sigma * ws.x + ATv - c
        x_new = solve_fn(rhs)

        # ── Step 2: s-update (cone projection with relaxation) ──
        Ax_new = torch.mv(A_gpu, x_new)
        v_hat = alpha * (b - Ax_new) + (1.0 - alpha) * ws.s
        s_input = v_hat - ws.y / r
        _project_cones_gpu(s_input, cone_info)
        s_new = s_input

        # ── Step 3: dual update ──
        y_new = ws.y + r * (s_new - v_hat)

        # ── Step 4: Anderson acceleration (damped, safeguarded) ──
        # Periodic restart (COSMO-style): flush history every 100 iters.
        # This is the ONLY restart we do — never restart on a single
        # rejected step (SCS/Zhang 2020: rejection doesn't invalidate
        # the stored curvature; the next application benefits from it).
        if k > 0 and k % _AA_RST_PERIOD == 0 and aa.k > 0:
            aa.k = 0
            aa.F.zero_()
            aa.X.zero_()

        if (k + 1) % aa_interval == 0 and k > aa_mem * aa_interval:
            u_old = torch.cat([ws.x, ws.s, ws.y])
            u_new = torch.cat([x_new, s_new, y_new])
            u_acc = aa.step(u_old, u_new)
            res_std = torch.norm(u_new - u_old)
            res_acc = torch.norm(u_acc - u_old)
            if res_acc <= 2.0 * res_std:
                # Damped acceptance: β=0.85 interpolation between the
                # plain ADMM step and the AA step. Prevents overshoot
                # for ill-conditioned PSD cones (arXiv:2202.05295).
                u_combined = (1.0 - _AA_BETA) * u_new + _AA_BETA * u_acc
                x_new = u_combined[:n]
                s_new = u_combined[n:n + m]
                _project_cones_gpu(s_new, cone_info)
                y_new = u_combined[n + m:]
            # Rejection: keep history (do NOT reset here — see comment above).

        # Update primal variables
        s_prev.copy_(ws.s)
        ws.x = x_new
        ws.s = s_new
        ws.y = y_new

        # ── Step 4: convergence check + adaptive rho ──
        if (k + 1) % check_interval == 0:
            Ax = torch.mv(A_gpu, ws.x)
            pri_res = torch.norm(Ax + ws.s - b).item()

            s_diff = ws.s - s_prev
            AT_sdiff = torch.mv(AT_gpu, s_diff)
            dual_res = (r * torch.norm(AT_sdiff)).item()

            Ax_norm = torch.norm(Ax).item()
            s_norm = torch.norm(ws.s).item()
            b_norm = torch.norm(b).item()
            ATy_norm = torch.norm(torch.mv(AT_gpu, ws.y)).item()
            c_norm = torch.norm(c).item()

            eps_pri = eps_abs * (m ** 0.5) + eps_rel * max(Ax_norm, s_norm,
                                                           b_norm)
            eps_dual = eps_abs * (n ** 0.5) + eps_rel * max(ATy_norm, c_norm)

            if verbose and (k + 1) % (check_interval * 10) == 0:
                pobj = torch.dot(c, ws.x).item()
                print(f"    iter {k+1}: pri={pri_res:.2e} dual={dual_res:.2e}"
                      f" eps_p={eps_pri:.2e} eps_d={eps_dual:.2e}"
                      f" rho={r:.1e} obj={pobj:.6f}")

            if pri_res < eps_pri and dual_res < eps_dual:
                status = 'solved'
                final_k = k + 1
                break

            # Adaptive rho: windowed geometric-mean residual balancing.
            # Classic residual balancing (Boyd 2010) fires on every check
            # and can oscillate when the ratio hovers near the threshold.
            # SCS fix (O'Donoghue 2016): accumulate 5 ratios first, then
            # act on their geometric mean. Use sqrt-scaling (Wohlberg 2017)
            # so each step is damped. Only update when the geometric mean
            # is conclusively off-balance (>3 or <1/3, not >10 or <0.1).
            if (k + 1) % adapt_interval == 0 and dual_res > 1e-30:
                _rho_ratio_buf.append(pri_res / (dual_res + 1e-30))
                if len(_rho_ratio_buf) >= _RHO_BUF_LEN:
                    import math as _math
                    geo_mean = _math.exp(
                        sum(_math.log(max(rv, 1e-30))
                            for rv in _rho_ratio_buf) / len(_rho_ratio_buf))
                    _rho_ratio_buf.clear()
                    if geo_mean > _RHO_THRESH_HI or geo_mean < _RHO_THRESH_LO:
                        # Damped scale: sqrt(geo_mean) not geo_mean
                        scale = geo_mean ** 0.5
                        max_change = 1.0 + (RHO_MAX_CHANGE_0 - 1.0) / (
                            1.0 + k / RHO_DECAY_HALF)
                        scale = max(1.0 / max_change, min(max_change, scale))
                        rho_new = max(RHO_MIN, min(RHO_MAX, r * scale))
                        if abs(rho_new - r) / max(r, 1e-12) > 0.01:
                            ws.y *= (r / rho_new)
                            rho_val[0] = rho_new
                            _refactor()
        # End convergence check
    else:
        final_k = max_iters
        # Compute final primal AND dual residuals. Both are needed for
        # downstream infeasibility certification; skipping dual_res (the
        # previous behavior) leaves callers unable to distinguish a
        # genuine infeasible problem from ADMM stalling on primal alone.
        Ax_final = torch.mv(A_gpu, ws.x)
        pri_res = torch.norm(Ax_final + ws.s - b).item()
        s_diff_final = ws.s - s_prev
        dual_res = (rho_val[0] * torch.norm(
            torch.mv(AT_gpu, s_diff_final))).item()
        eps_pri = eps_abs * (m ** 0.5) + eps_rel * max(
            torch.norm(Ax_final).item(), torch.norm(ws.s).item(),
            torch.norm(b).item())
        eps_dual = eps_abs * (n ** 0.5) + eps_rel * max(
            torch.norm(torch.mv(AT_gpu, ws.y)).item(),
            torch.norm(c).item())
        if pri_res < 10.0 * eps_pri:
            status = 'solved_inaccurate'
        else:
            status = 'infeasible_inaccurate'

    solve_time = time.time() - t_solve

    # ═══ POST-SOLVE PSD VERIFICATION ═══
    # Check actual PSD constraint satisfaction: extract PSD blocks from s,
    # verify eigenvalues are non-negative. This catches cases where ADMM
    # "converges" to an approximate solution that violates PSD constraints.
    if status in ('solved', 'solved_inaccurate'):
        max_psd_violation = 0.0
        for mat_dim, group in cone_info.size_groups.items():
            n_cones = len(group)
            idx = cone_info.svec_indices[mat_dim]
            rows_idx = idx['rows']
            cols_idx = idx['cols']
            unpack_scale = idx['unpack_scale']
            svec_dim = idx['dim']
            gather_idx = cone_info.batch_gather[mat_dim]

            # Batched gather (same as _project_cones_gpu — no Python loop)
            svec_flat = ws.s[gather_idx].reshape(n_cones, svec_dim)
            svec_flat = svec_flat * unpack_scale.unsqueeze(0)
            batch = torch.zeros(n_cones, mat_dim, mat_dim,
                                dtype=torch.float64, device=device)
            batch[:, rows_idx, cols_idx] = svec_flat
            batch[:, cols_idx, rows_idx] = svec_flat

            # Fast path: cholesky_ex (~144x cheaper than eigvalsh per CLAUDE.md).
            # If all matrices factor, they're PSD (min_eig >= 0) — skip eigh.
            _, info = torch.linalg.cholesky_ex(batch)
            non_psd_mask = info > 0
            if non_psd_mask.any():
                eigs = torch.linalg.eigvalsh(batch[non_psd_mask])
                min_eig = eigs[:, 0].min().item()
                max_psd_violation = max(max_psd_violation, -min_eig)

        # If PSD blocks have significantly negative eigenvalues, reclassify
        psd_tol = max(eps_abs * 100, 1e-3)
        if max_psd_violation > psd_tol:
            status = 'infeasible_inaccurate'

    # ═══ UNSCALE + TRANSFER BACK (once, at end) ═══
    # Ruiz: scaled problem is D*A*E * x_s + D*s_s = D*b
    # So x_orig = E * x_scaled, s_orig = s_scaled / D, y_orig = D * y_scaled
    x_unscaled = ws.x * E_gpu
    s_unscaled = ws.s / D_gpu
    y_unscaled = ws.y * D_gpu

    pobj = torch.dot(torch.tensor(c_np, dtype=torch.float64, device=device),
                     x_unscaled).item()
    dobj = -torch.dot(torch.tensor(b_np, dtype=torch.float64, device=device),
                      y_unscaled).item()

    if verbose:
        msg = (f"  ADMM done: {final_k} iters, {solve_time:.3f}s "
               f"({solve_time/max(final_k,1)*1000:.2f}ms/iter), "
               f"status={status}")
        print(msg)

    return {
        'x': x_unscaled.cpu().numpy(),
        'y': y_unscaled.cpu().numpy(),
        's': s_unscaled.cpu().numpy(),
        'info': {
            'iter': final_k,
            'status': status,
            'pobj': pobj,
            'dobj': dobj,
            'setup_time': setup_time,
            'solve_time': solve_time,
            # Last-measured residuals on the SCALED problem (for caller-side
            # verdict certification). Zero-initialized if no convergence
            # check fired; finite values only after at least one check.
            'pri_res': pri_res,
            'dual_res': dual_res,
            'eps_pri': eps_pri,
            'eps_dual': eps_dual,
        }
    }


# =====================================================================
# Persistent solver (reuse factorization across bisection steps)
# =====================================================================

class ADMMSolver:
    """Persistent GPU ADMM solver — reuse factorization across solves.

    Designed for the bisection loop in run_scs_direct.py where A changes
    slightly (only the t-dependent window PSD entries) between steps.
    """

    def __init__(self, A_csc, b_np, c_np, cone, *,
                 sigma=1e-6, rho=0.1, alpha=1.0,
                 device='cuda', verbose=False, profile_init=False):
        import time as _time
        _t0 = _time.perf_counter()
        _log = []

        def _mark(lbl):
            if profile_init:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    pass
                _log.append((lbl, _time.perf_counter() - _t0))

        self.device = device
        self.sigma = sigma
        self.rho = rho
        self.alpha = alpha
        self.verbose = verbose

        m, n = A_csc.shape
        self.m = m
        self.n = n

        _mark('start')

        self.cone_info = ConeInfo(cone, device)
        _mark('cone_info')

        # CSR pattern cache — avoids re-transferring indptr/indices
        self._csr_cache = _CSRPatternCache(device)

        # Pre-allocate sigI only for Cholesky path (n < 5000).
        # For CG path (n >= 5000), sigma*I is applied implicitly.
        if n < 5000:
            self._sigI = sigma * torch.eye(n, dtype=torch.float64, device=device)

        # ── Ruiz equilibration: compute D, E ONCE from first A ──
        self._D, self._E = self._compute_ruiz(A_csc, cone)
        _mark('ruiz')
        self._D_gpu = torch.tensor(self._D, dtype=torch.float64, device=device)
        self._E_gpu = torch.tensor(self._E, dtype=torch.float64, device=device)

        # Scale b, c (stored scaled on GPU)
        self._b_orig = b_np.copy()
        self._c_orig = c_np.copy()
        # Cache unscaled b, c on GPU so post-solve obj computation doesn't
        # rebuild tensors per solve call (one H2D + alloc saved per step).
        self._b_orig_gpu = torch.tensor(b_np, dtype=torch.float64, device=device)
        self._c_orig_gpu = torch.tensor(c_np, dtype=torch.float64, device=device)
        self.b = torch.tensor(self._D * b_np, dtype=torch.float64, device=device)
        self.c = torch.tensor(self._E * c_np, dtype=torch.float64, device=device)

        _mark('bc_gpu')

        # Store A data for in-place update (applies cached D, E)
        self._update_A(A_csc)
        _mark('update_A')

        # Workspace
        self.ws = _ADMMWorkspace(n, m, device)
        _mark('workspace')

        if profile_init:
            prev = 0.0
            msg = ['  ADMMSolver init profile:']
            for lbl, t in _log:
                msg.append(f'    {lbl:14s} +{(t-prev)*1000:8.1f}ms  (cum {t*1000:8.1f}ms)')
                prev = t
            print('\n'.join(msg), flush=True)

    def _compute_ruiz(self, A_csc, cone):
        """Compute Ruiz row/col scaling D, E from A and cone structure.

        PSD cone rows within the same block share one scaling factor
        (geometric mean) to preserve svec packing.

        Optimized: in-place data scaling, avoid redundant .copy(), CSR for
        row norms (CSR.max(axis=1) is O(nnz), CSC.max(axis=1) is O(m*n)).
        5 iterations is sufficient for convergence.
        """
        z_dim = cone.get('z', 0)
        l_dim = cone.get('l', 0)
        psd_sizes = list(cone.get('s', []))

        # Build cone boundary groups
        cone_groups = []
        psd_start = z_dim + l_dim
        offset = psd_start
        for pdim in psd_sizes:
            svec_dim = pdim * (pdim + 1) // 2
            cone_groups.append((offset, offset + svec_dim))
            offset += svec_dim

        m, n = A_csc.shape
        # Work in CSR for fast row operations
        A_csr = A_csc.tocsr().copy()
        D = np.ones(m)
        E = np.ones(n)

        # Pre-compute row index for each data entry (vectorized)
        row_indices = np.repeat(np.arange(m, dtype=np.int64),
                                np.diff(A_csr.indptr))
        # Mask for non-empty rows (for reduceat)
        nonempty = np.diff(A_csr.indptr) > 0
        nonempty_starts = A_csr.indptr[:-1][nonempty]

        # 5 Ruiz iters (was 10). Geometric convergence leaves ~8% scale
        # residual at 5 iters vs ~1% at 10; ADMM is insensitive to this
        # (confirmed by CLAUDE.md's σ-sweep showing ADMM iter count is
        # independent of σ across 7 orders of magnitude — σ directly sets
        # the effective conditioning). Saves ~50% of Ruiz time at high K.
        for _ in range(5):
            # Row inf-norms (vectorized segmented max)
            abs_data = np.abs(A_csr.data)
            row_norms = np.zeros(m)
            if len(nonempty_starts) > 0:
                row_maxes = np.maximum.reduceat(abs_data, nonempty_starts)
                row_norms[nonempty] = row_maxes
            row_norms = np.maximum(row_norms, 1e-10)
            d = 1.0 / np.sqrt(row_norms)
            for start_r, end_r in cone_groups:
                block_d = d[start_r:end_r]
                d[start_r:end_r] = np.exp(np.mean(np.log(
                    np.maximum(block_d, 1e-20))))
            d = np.clip(d, 1e-4, 1e4)
            # In-place row scaling (vectorized)
            A_csr.data *= d[row_indices]
            D *= d

            # Col inf-norms (scatter max)
            abs_data = np.abs(A_csr.data)
            col_norms = np.zeros(n)
            np.maximum.at(col_norms, A_csr.indices, abs_data)
            col_norms = np.maximum(col_norms, 1e-10)
            e = 1.0 / np.sqrt(col_norms)
            e = np.clip(e, 1e-4, 1e4)
            # In-place col scaling (vectorized)
            A_csr.data *= e[A_csr.indices]
            E *= e

        return D, E

    def _update_A(self, A_csc):
        """Update A matrix on GPU, applying cached Ruiz scaling.

        Optimization: pre-computes per-entry Ruiz scale factors on first call
        so subsequent calls skip sp.diags(D) @ A @ sp.diags(E) (830ms → ~50ms).
        """
        A_csc_sorted = A_csc.tocsc()
        A_csc_sorted.sort_indices()

        if not hasattr(self, '_ruiz_entry_scale'):
            # First call: compute per-entry scale D[row]*E[col] and cache.
            # CSC stores entries in column-major order: for each col j,
            # rows are A.indices[A.indptr[j]:A.indptr[j+1]].
            # Use int32 for the cols array: n ≤ 2^31 always holds for our
            # problems, and int32 halves the allocation (~2.3GB → ~1.1GB
            # at K=400, nnz=291M) — reduces CPU alloc time and allows
            # numpy to keep the array in L3 during the D[rows]*E[cols]
            # multiply.
            rows = A_csc_sorted.indices
            cols = np.repeat(np.arange(A_csc_sorted.shape[1], dtype=np.int32),
                             np.diff(A_csc_sorted.indptr))
            self._ruiz_entry_scale = self._D[rows] * self._E[cols]
            self._ruiz_col_idx = cols  # cached for diag_ata computation

        # Fast path: element-wise multiply (no sparse matrix multiply)
        scaled_data = A_csc_sorted.data * self._ruiz_entry_scale
        # Reuse structure from input (no copy needed — _csr_cache only reads data)
        A_csc_sorted.data = scaled_data

        self.A_gpu, self.AT_gpu = self._csr_cache.to_gpu(A_csc_sorted)

        n = self.n
        if n < 5000:
            ATA = (self.AT_gpu @ self.A_gpu).to_dense()
            M = self._sigI + self.rho * ATA
            self.L = torch.linalg.cholesky(M)
            self._solver_type = 'cholesky'
        else:
            self._solver_type = 'cg'
            if not hasattr(self, '_cg_x_prev'):
                self._cg_x_prev = torch.zeros(n, dtype=torch.float64,
                                              device=self.device)
            # Diagonal Jacobi preconditioner: M_inv = 1/(sigma + rho*diag(ATA))
            # diag(ATA)[j] = ||A[:,j]||^2 (column-wise squared norm)
            # Compute without copying sparse matrix: bincount on column indices
            diag_ata = np.bincount(self._ruiz_col_idx,
                                   weights=scaled_data ** 2,
                                   minlength=n)
            self._precond_inv = torch.tensor(
                1.0 / (self.sigma + self.rho * diag_ata + 1e-20),
                dtype=torch.float64, device=self.device)

    def update_b(self, b_np):
        """Update b vector with cached Ruiz scaling.

        Reuses self.b's GPU buffer — avoids per-call tensor alloc + keeps
        downstream tensor views valid.
        """
        scaled = self._D * b_np
        if self.b.shape[0] == scaled.shape[0]:
            self.b.copy_(torch.from_numpy(scaled))
        else:
            self.b = torch.tensor(scaled, dtype=torch.float64,
                                  device=self.device)

    def update_A_data(self, base_data, t_data, t_val, A_template):
        """In-place update of A matrix data for new t_val.

        A.data = base_data + t_val * t_data  (numpy, on CPU)
        Then transfer updated A to GPU and refactor.
        """
        import numpy as _np
        _np.add(base_data, t_val * t_data, out=A_template.data)
        self._update_A(A_template)

    def solve(self, *, max_iters=50000, eps_abs=1e-6, eps_rel=1e-6,
              warm_start=None, check_interval=10, tau_col=None,
              tau_tol=1e-4):
        """Run ADMM solve with current A, b, c.

        If tau_col is set, enable early tau classification:
        - Early FEASIBLE: tau <= tau_tol (safe — just moves hi down)
        - Early INFEASIBLE: tau >> tau_tol and stable (conservative)
        """
        import time

        n, m = self.n, self.m
        device = self.device
        A_gpu = self.A_gpu
        AT_gpu = self.AT_gpu
        b = self.b
        c = self.c
        sigma = self.sigma
        rho = self.rho
        alpha = self.alpha
        cone_info = self.cone_info
        ws = self.ws

        # Linear system solver
        if self._solver_type == 'cholesky':
            L = self.L

            def solve_fn(rhs):
                return torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        else:
            def matvec(v):
                return sigma * v + rho * torch.mv(AT_gpu, torch.mv(A_gpu, v))

            cg_prev = self._cg_x_prev
            precond_inv = getattr(self, '_precond_inv', None)

            def solve_fn(rhs):
                _cg_maxiter = max(25, min(self.n // 1000, 100))
                x = _torch_cg(matvec, rhs, cg_prev, maxiter=_cg_maxiter,
                              tol=1e-10, precond_inv=precond_inv)
                cg_prev.copy_(x)
                return x

        # Warm-start
        if warm_start is not None:
            if warm_start.get('x') is not None:
                x_ws = warm_start['x']
                ws.x.zero_()
                ws.x[:min(len(x_ws), n)] = torch.tensor(
                    x_ws[:n], dtype=torch.float64, device=device)
            if warm_start.get('y') is not None:
                y_ws = warm_start['y']
                ws.y.zero_()
                ws.y[:min(len(y_ws), m)] = torch.tensor(
                    y_ws[:m], dtype=torch.float64, device=device)
            if warm_start.get('s') is not None:
                s_ws = warm_start['s']
                ws.s.zero_()
                ws.s[:min(len(s_ws), m)] = torch.tensor(
                    s_ws[:m], dtype=torch.float64, device=device)

        t_solve = time.time()
        status = 'solved_inaccurate'
        final_k = max_iters
        s_prev = ws.s.clone()

        # Track last-measured residuals so the bisection caller can
        # distinguish 'solved+tau>tau_tol' (certified infeasibility) from
        # 'solved_inaccurate+tau>tau_tol' (iter-cap stall — uncertain).
        # Without these the caller cannot tell the two cases apart and
        # risks moving the bisection bracket on an uncertified verdict.
        pri_res = float('inf')
        dual_res = float('inf')
        eps_pri = 0.0
        eps_dual = 0.0

        # Anderson acceleration — same improved config as admm_solve:
        # interval=10, damping β=0.85, periodic restart every 100 iters.
        # History is NEVER reset on safeguard rejection (see admm_solve).
        aa_mem = _AA_MEM_DEFAULT
        aa = AndersonAccelerator(aa_mem, n + m + m, device)
        aa_interval = _AA_INTERVAL_DEFAULT
        _AA_BETA = _AA_BETA_DEFAULT
        _AA_RST_PERIOD = _AA_RST_DEFAULT

        for k in range(max_iters):
            # x-update
            v = rho * (b - ws.s) - ws.y
            ATv = torch.mv(AT_gpu, v)
            rhs = sigma * ws.x + ATv - c
            x_new = solve_fn(rhs)

            # s-update (with relaxation)
            Ax_new = torch.mv(A_gpu, x_new)
            v_hat = alpha * (b - Ax_new) + (1.0 - alpha) * ws.s
            s_input = v_hat - ws.y / rho
            _project_cones_gpu(s_input, cone_info)
            s_new = s_input

            # dual update
            y_new = ws.y + rho * (s_new - v_hat)

            # Anderson acceleration (damped, safeguarded, periodic restart).
            # Periodic restart only — never on rejection (see admm_solve).
            if k > 0 and k % _AA_RST_PERIOD == 0 and aa.k > 0:
                aa.k = 0
                aa.F.zero_()
                aa.X.zero_()

            if (k + 1) % aa_interval == 0 and k > aa_mem * aa_interval:
                u_old = torch.cat([ws.x, ws.s, ws.y])
                u_new = torch.cat([x_new, s_new, y_new])
                u_acc = aa.step(u_old, u_new)
                res_std = torch.norm(u_new - u_old)
                res_acc = torch.norm(u_acc - u_old)
                if res_acc <= 2.0 * res_std:
                    # Damped acceptance (β=0.85) — prevents overshoot on
                    # ill-conditioned PSD cones (arXiv:2202.05295).
                    u_combined = (1.0 - _AA_BETA) * u_new + _AA_BETA * u_acc
                    x_new = u_combined[:n]
                    s_new = u_combined[n:n + m]
                    _project_cones_gpu(s_new, cone_info)
                    y_new = u_combined[n + m:]

            ws.x = x_new
            s_prev.copy_(ws.s)
            ws.s = s_new
            ws.y = y_new

            # convergence check
            if (k + 1) % check_interval == 0:
                Ax = torch.mv(A_gpu, ws.x)
                pri_res = torch.norm(Ax + ws.s - b).item()

                s_diff = ws.s - s_prev
                AT_sdiff = torch.mv(AT_gpu, s_diff)
                dual_res = (rho * torch.norm(AT_sdiff)).item()

                Ax_norm = torch.norm(Ax).item()
                s_norm = torch.norm(ws.s).item()
                b_norm = torch.norm(b).item()
                ATy_norm = torch.norm(torch.mv(AT_gpu, ws.y)).item()
                c_norm = torch.norm(c).item()

                eps_pri = eps_abs * (m ** 0.5) + eps_rel * max(
                    Ax_norm, s_norm, b_norm)
                eps_dual = eps_abs * (n ** 0.5) + eps_rel * max(
                    ATy_norm, c_norm)

                if self.verbose and (k + 1) % (check_interval * 10) == 0:
                    pobj = torch.dot(c, ws.x).item()
                    print(f"    iter {k+1}: pri={pri_res:.2e} "
                          f"dual={dual_res:.2e} obj={pobj:.6f}")

                if pri_res < eps_pri and dual_res < eps_dual:
                    # Warm-start hazard: when warm-starting from a feasible
                    # solution at a lower t, residuals at the new (slightly
                    # higher) t often read "converged" within 10-20 iters
                    # while tau sits in the borderline zone
                    # [tau_tol, MARGIN*tau_tol]. The solver then exits as
                    # 'solved' but the bisection caller classifies the
                    # verdict as 'uncertain' (tau is neither feas nor clear
                    # infeas). Require either a minimum 200 iters, OR that
                    # tau has clearly resolved to one side of the borderline.
                    ok_to_exit = (k + 1) >= 200
                    if not ok_to_exit and tau_col is not None:
                        tau_now = ws.x[tau_col].item()
                        # Clearly feasible OR clearly infeasible.
                        if (tau_now <= 0.5 * tau_tol
                                or tau_now > 3.0 * tau_tol):
                            ok_to_exit = True
                    elif tau_col is None:
                        # No tau (e.g. bound-minimization): honour the
                        # residual-only exit as before.
                        ok_to_exit = True
                    if ok_to_exit:
                        status = 'solved'
                        final_k = k + 1
                        break

                # ── Early tau classification (phase-1 problems) ──
                # ONLY early feasible exit. Early infeasible is UNSOUND:
                # ADMM tau at low iteration count is meaningless — it
                # caused lb=1.388 > val(32) in the 2026-04-16 run by
                # misclassifying t=1.388 as infeasible after 110 iters.
                if tau_col is not None and (k + 1) >= 50:
                    tau_now = ws.x[tau_col].item()

                    # SAFE: early feasible (tau small → hi moves down)
                    if (tau_now <= tau_tol
                            and pri_res < 10 * eps_pri):
                        status = 'solved'
                        final_k = k + 1
                        break
        else:
            final_k = max_iters
            # Compute BOTH primal and dual residuals at the final iterate
            # (previously only pri was computed). Callers rely on dual_res
            # being finite to certify an infeasibility verdict — leaving
            # it at inf silently degrades every infeas classification to
            # 'uncertain' upstream.
            Ax_final = torch.mv(A_gpu, ws.x)
            pri_res = torch.norm(Ax_final + ws.s - b).item()
            s_diff_final = ws.s - s_prev
            dual_res = (rho * torch.norm(
                torch.mv(AT_gpu, s_diff_final))).item()
            eps_pri = eps_abs * (m ** 0.5) + eps_rel * max(
                torch.norm(Ax_final).item(), torch.norm(ws.s).item(),
                torch.norm(b).item())
            eps_dual = eps_abs * (n ** 0.5) + eps_rel * max(
                torch.norm(torch.mv(AT_gpu, ws.y)).item(),
                torch.norm(c).item())
            if pri_res < 10.0 * eps_pri:
                status = 'solved_inaccurate'
            else:
                status = 'infeasible_inaccurate'

        solve_time = time.time() - t_solve

        # ── Unscale: x_orig = E * x_scaled, s_orig = s/D, y_orig = D * y ──
        x_unscaled = ws.x * self._E_gpu
        s_unscaled = ws.s / self._D_gpu
        y_unscaled = ws.y * self._D_gpu

        pobj = torch.dot(self._c_orig_gpu, x_unscaled).item()
        dobj = -torch.dot(self._b_orig_gpu, y_unscaled).item()

        if self.verbose:
            print(f"  ADMM: {final_k} iters, {solve_time:.3f}s, "
                  f"status={status}")

        return {
            'x': x_unscaled.cpu().numpy(),
            'y': y_unscaled.cpu().numpy(),
            's': s_unscaled.cpu().numpy(),
            'info': {
                'iter': final_k,
                'status': status,
                'pobj': pobj,
                'dobj': dobj,
                'solve_time': solve_time,
                # Residuals on the SCALED problem (unscaling only divides
                # by D/E norms, not informative for the KKT verdict).
                # pri_res, dual_res finite only after at least one
                # check_interval fire; else left at inf.
                'pri_res': pri_res,
                'dual_res': dual_res,
                'eps_pri': eps_pri,
                'eps_dual': eps_dual,
            }
        }


# =====================================================================
# Convenience: standalone test
# =====================================================================

if __name__ == '__main__':
    import sys
    import os

    # Quick sanity test with a tiny SDP
    print("ADMM GPU solver — sanity test")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # min t s.t. [t, x; x, t] >= 0, t + x = 1, x >= 0
    # Variables: [x, t] (n=2)
    # Constraints: 1 equality (z=1), 1 nonneg (l=1), 1 PSD 2x2 (s=[2])
    # s vector: [s_eq, s_nonneg, s_psd(3 entries)]
    # A is 1+1+3 = 5 rows, 2 cols
    n, m_total = 2, 5

    rows = [0, 0, 1, 2, 3, 4]  # eq, nonneg, psd diag, psd offdiag, psd diag
    cols = [0, 1, 0, 1, 0, 1]
    vals = [1.0, 1.0, -1.0, -1.0, -SQRT2, -1.0]
    # Row 0: x + t = 1  (zero cone)
    # Row 1: -x >= 0  i.e. x >= 0 after SCS negation (nonneg cone)
    # Rows 2-4: PSD 2x2 [[t, x],[x, t]] in svec: [t, sqrt2*x, t]
    #   Row 2: -t (svec[0] = M[0,0] = t)
    #   Row 3: -sqrt2*x (svec[1] = sqrt2*M[1,0] = sqrt2*x)
    #   Row 4: -t (svec[2] = M[1,1] = t)

    A = sp.csc_matrix((vals, (rows, cols)), shape=(m_total, n))
    b_arr = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    c_arr = np.array([0.0, 1.0])  # min t
    cone_dict = {'z': 1, 'l': 1, 's': [2]}

    sol = admm_solve(A, b_arr, c_arr, cone_dict, device=device,
                     max_iters=5000, eps_abs=1e-7, eps_rel=1e-7,
                     verbose=True)

    print(f"\nResult: x={sol['x']}, status={sol['info']['status']}")
    print(f"  t = {sol['x'][1]:.6f} (expected ~0.5)")
    print(f"  x = {sol['x'][0]:.6f} (expected ~0.5)")
    print(f"  iters = {sol['info']['iter']}")
