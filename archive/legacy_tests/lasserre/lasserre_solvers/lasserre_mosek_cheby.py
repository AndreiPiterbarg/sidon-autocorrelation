#!/usr/bin/env python
"""MOSEK full Lasserre solver — SHIFTED CHEBYSHEV BASIS.

Solves exactly the same SDP as ``tests/lasserre_mosek_tuned.py`` but
with every moment, localizing, and window PSD cone re-expressed in the
shifted-Chebyshev basis  T_k*(mu) = T_k(2 mu - 1).  The feasible set
and optimal value are mathematically identical; only the numerical
representation changes.

The monomial moment matrix M_k(y) and the Chebyshev moment matrix
M_k^{cheb}(c) satisfy  M_k(y) = V M_k^{cheb}(c) V^T  for a triangular
integer/rational matrix V (change of basis from Chebyshev to
monomials on ``basis``), so the two PSD cones are V-similar and thus
equivalent.  Chebyshev entries are bounded |T_k*| <= 1 on [0, 1],
giving a near-orthonormal parameterisation that dramatically reduces
the condition number of the moment matrix compared with the
Vandermonde-like monomial basis.

Basis-change arithmetic uses ``fractions.Fraction`` so the derivation
is integer-exact; float64 only appears when the final MOSEK sparse
matrices are assembled.

Mode 'baseline' disables MOSEK tuning; mode 'tuned' applies the same
parameter suite as ``lasserre_mosek_tuned.py``.  No Z/2 variant is
exposed — that is an orthogonal reduction and can be re-introduced
once the Chebyshev change of basis is locked in.
"""
from __future__ import annotations

import os

# Change #5 — thread env vars MUST be set before importing numpy / scipy,
# otherwise NumPy/BLAS seize every logical core on import and steal the
# thread pool from MOSEK.  MKL_NUM_THREADS=1 & OPENBLAS_NUM_THREADS=1
# restrict BLAS to a single thread; OMP_NUM_THREADS controls the OpenMP
# pool that MOSEK itself uses.  Default OMP to physical-core count
# (os.cpu_count() is logical — assume HT and halve).  User can override
# any of these by exporting before launch.
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_phys_cores_for_env = max(1, (os.cpu_count() or 2) // 2)
os.environ.setdefault('OMP_NUM_THREADS', str(_phys_cores_for_env))

import argparse
import gc
import json
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import mosek
from mosek.fusion import (Domain, Expr, Matrix, Model, ObjectiveSense,
                          SolutionStatus)


# =====================================================================
# MOSEK log handler — streams MOSEK's per-iteration IPM log so we can
# estimate remaining wall time during long solves.
# =====================================================================

class MosekLogStream:
    """Stream adapter that forwards MOSEK log lines to a Python sink
    with a prefix and (optionally) captures per-iteration IPM rows for
    post-solve timing analysis.

    MOSEK's SDP IPM log format (one row per iteration):
        ITE   PFEAS      DFEAS      GFEAS     PRSTATUS    POBJ        DOBJ        MU         TIME
    MU halves ~every iteration; with N iterations done and k remaining
    (based on MU reaching solver tolerance), remaining wall ≈ k *
    mean_iter_wall.  We parse this live.
    """

    _ITER_RE = None  # lazily compiled

    def __init__(self, prefix: str = '[MOSEK] ',
                  sink=sys.stdout, capture_iters: bool = True):
        self.prefix = prefix
        self.sink = sink
        self.capture_iters = capture_iters
        self.iter_rows: List[Dict[str, Any]] = []
        self._buffer = ''
        self._solve_start_time: Optional[float] = None

    def mark_solve_start(self) -> None:
        self._solve_start_time = time.time()
        self.iter_rows = []

    def _try_parse_iter_row(self, line: str) -> None:
        """Match one IPM iteration row and stash a dict with the fields."""
        import re
        if MosekLogStream._ITER_RE is None:
            MosekLogStream._ITER_RE = re.compile(
                r'^\s*(\d+)\s+'
                r'([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+'
                r'(\S+)\s+'
                r'([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+'
                r'([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$'
            )
        m = MosekLogStream._ITER_RE.match(line)
        if not m:
            return
        try:
            iter_no = int(m.group(1))
            pfeas = float(m.group(2))
            dfeas = float(m.group(3))
            gfeas = float(m.group(4))
            prstatus = m.group(5)
            pobj = float(m.group(6))
            dobj = float(m.group(7))
            mu = float(m.group(8))
            tm = float(m.group(9))
        except ValueError:
            return
        row = {
            'iter': iter_no, 'pfeas': pfeas, 'dfeas': dfeas,
            'gfeas': gfeas, 'prstatus': prstatus,
            'pobj': pobj, 'dobj': dobj, 'mu': mu, 'mosek_time': tm,
            'wall_elapsed': (time.time() - self._solve_start_time
                              if self._solve_start_time else None),
        }
        self.iter_rows.append(row)
        # Live ETA estimate: MU is shrinking roughly geometrically;
        # if target tol is t, remaining ≈ log(mu/t) / log(mu_prev/mu) iters.
        if len(self.iter_rows) >= 3:
            self._print_eta(row)

    def _print_eta(self, cur: Dict[str, Any]) -> None:
        """Print a one-line ETA estimate after this iteration."""
        # Estimate iteration rate (wall seconds per iter).
        wall = cur.get('wall_elapsed')
        if wall is None or wall < 1.0:
            return
        n = cur['iter']
        if n < 2:
            return
        per_iter = wall / n
        # Estimate remaining iterations until mu hits ~1e-8 (typical
        # MOSEK SDP convergence floor).  mu decreases geometrically;
        # compute the effective rate from the last two iterations.
        mu_now = cur['mu']
        if len(self.iter_rows) >= 2:
            mu_prev = self.iter_rows[-2]['mu']
            if mu_prev > 0 and mu_now > 0 and mu_prev > mu_now:
                import math
                rate = math.log(mu_prev / mu_now)  # log10 reduction per iter
                if rate > 1e-3:
                    remain_iters = max(0.0, math.log(mu_now / 1e-8) / rate)
                    remain_s = remain_iters * per_iter
                    eta_msg = (f"{self.prefix}"
                                 f"ETA: iter {n}, mu={mu_now:.2e}, "
                                 f"per-iter={per_iter:.1f}s, "
                                 f"~{remain_iters:.0f} iters remain, "
                                 f"eta={remain_s/60:.1f} min\n")
                    try:
                        self.sink.write(eta_msg)
                        self.sink.flush()
                    except Exception:
                        pass

    def write(self, msg: str) -> None:
        """MOSEK writes chunks that may not be complete lines; buffer
        and split."""
        if not msg:
            return
        self._buffer += msg
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            prefixed = self.prefix + line + '\n'
            try:
                self.sink.write(prefixed)
                self.sink.flush()
            except Exception:
                pass
            if self.capture_iters:
                self._try_parse_iter_row(line)

    def flush(self) -> None:
        if self._buffer:
            try:
                self.sink.write(self.prefix + self._buffer)
                self.sink.flush()
            except Exception:
                pass
            self._buffer = ''


# =====================================================================
# Watcher thread — emits RSS/CPU/thread-count every N seconds so we can
# see that the solver is making progress during MOSEK's serial phases
# (presolve, symbolic factor, Schur allocation).  These phases print
# nothing to MOSEK's log, so from the outside the process looks frozen.
# =====================================================================

class SolverWatcher:
    """Background thread that samples process state periodically and
    writes one-line status records to a sink.  Also snapshots peak RSS.

    Usage:
        w = SolverWatcher(interval_s=15, tag='build'); w.start(); ...
        w.stop()

    Samples include: wall, RSS MB, %CPU (since last sample), thread count,
    and number of threads at >5% CPU (to detect the single-vs-parallel
    transition).  Requires ``psutil`` — falls back to /proc scraping if
    unavailable.
    """

    def __init__(self, interval_s: float = 15.0,
                  sink=sys.stdout, tag: str = 'watch'):
        self.interval_s = float(interval_s)
        self.sink = sink
        self.tag = tag
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_wall: Optional[float] = None
        self.samples: List[Dict[str, Any]] = []
        self.peak_rss_mb: int = 0
        self._psutil_proc = None  # cached psutil.Process(pid) handle
        try:
            import psutil
            self._psutil_proc = psutil.Process(os.getpid())
            # Prime the CPU-percent cache — first call always returns 0.
            self._psutil_proc.cpu_percent(interval=None)
            self._have_psutil = True
        except ImportError:
            self._have_psutil = False

    def _read_proc_stat(self) -> Dict[str, Any]:
        """Read /proc/self/status + /proc/self/stat on Linux; fall back
        to psutil if available.  Returns a dict of sampled fields."""
        out: Dict[str, Any] = {'ts_s': time.time()}
        if self._have_psutil and self._psutil_proc is not None:
            p = self._psutil_proc
            with p.oneshot():
                out['rss_mb'] = int(p.memory_info().rss / 1024 / 1024)
                # cpu_percent returns % CPU averaged since the PREVIOUS
                # call on this Process handle.  We primed in __init__ and
                # keep reusing the same handle, so this is now correct.
                out['cpu_pct'] = p.cpu_percent(interval=None)
                out['threads'] = int(p.num_threads())
            return out
        # Linux /proc fallback.
        try:
            with open(f'/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        out['rss_mb'] = int(line.split()[1]) // 1024
                    elif line.startswith('Threads:'):
                        out['threads'] = int(line.split()[1])
        except Exception:
            out.setdefault('rss_mb', -1)
            out.setdefault('threads', -1)
        out.setdefault('cpu_pct', -1)
        out.setdefault('busy_approx', -1)
        return out

    def _loop(self) -> None:
        assert self._start_wall is not None
        while not self._stop_event.is_set():
            self._stop_event.wait(self.interval_s)
            if self._stop_event.is_set():
                break
            try:
                s = self._read_proc_stat()
            except Exception:
                continue
            elapsed = time.time() - self._start_wall
            rss = s.get('rss_mb', -1)
            if rss > self.peak_rss_mb:
                self.peak_rss_mb = rss
            cpu = s.get('cpu_pct', -1)
            thr = s.get('threads', -1)
            self.samples.append({
                'elapsed_s': round(elapsed, 1), **s,
            })
            try:
                self.sink.write(
                    f"[WATCH:{self.tag}] t={elapsed:6.0f}s  "
                    f"rss={rss:>7d}MB  cpu={cpu:6.1f}%  "
                    f"threads={thr:>3d}  peak_rss={self.peak_rss_mb:>7d}MB\n")
                self.sink.flush()
            except Exception:
                pass

    def start(self) -> None:
        self._start_wall = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

from lasserre_fusion import (build_window_matrices, collect_moments,
                               enum_monomials)
from lasserre_scalable import _precompute
from lasserre.cheby_basis import (
    build_B_matrix,
    build_moment_matrix_map,
    build_loc_matrix_map,
    build_t_matrix_map,
    build_window_Q_map,
)


# =====================================================================
# Known val(d) for gap-closure reporting
# =====================================================================
val_d_known = {
    4: 1.10233, 6: 1.17110, 8: 1.20464, 10: 1.24137,
    12: 1.27072, 14: 1.28396, 16: 1.31852,
    32: 1.336, 64: 1.384, 128: 1.420, 256: 1.448,
}


# =====================================================================
# MOSEK parameter tuning — applied to a Fusion Model
# =====================================================================

def _physical_core_count() -> int:
    try:
        import psutil
        c = psutil.cpu_count(logical=False)
        if c:
            return int(c)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def apply_tuned_params(mdl: Model, *, tol: float = 1e-6,
                        threads: Optional[int] = None,
                        solve_form: str = 'dual',
                        order_method: str = 'forceGraphpar',
                        max_iterations: int = 1600,
                        scaling: str = 'free',
                        presolve_lindep: str = 'on',
                        verbose: bool = True) -> Dict[str, Any]:
    """Apply the suite of MOSEK tuning parameters to a Fusion model.

    Incorporates levers learned from the d=10 pod run:

      • Aggressive constraint scaling     — Lever 1.
      • Increased iteration cap (1600)     — Lever 5.
      • (Retry ladder in _check_feasible) — Levers 2 + 4.

    Returns the settings dict actually applied (for logging).
    """
    if threads is None:
        threads = _physical_core_count()

    # Solve form: primal / dual / free.
    solve_form = solve_form.lower()
    if solve_form == 'dual':
        mdl.setSolverParam('intpntSolveForm', 'dual')
    elif solve_form == 'primal':
        mdl.setSolverParam('intpntSolveForm', 'primal')
    else:
        mdl.setSolverParam('intpntSolveForm', 'free')

    # Basis identification — turn off for pure SDP (meaningless cost).
    mdl.setSolverParam('intpntBasis', 'never')

    # Tolerances (all interior-point CO tolerances to `tol`).
    mdl.setSolverParam('intpntCoTolRelGap', tol)
    mdl.setSolverParam('intpntCoTolPfeas', tol)
    mdl.setSolverParam('intpntCoTolDfeas', tol)
    mdl.setSolverParam('intpntCoTolMuRed', tol)

    # Iteration cap — default 400 is tight for large SDP near boundary.
    mdl.setSolverParam('intpntMaxIterations', int(max_iterations))

    # Constraint scaling — AGGRESSIVE equilibrates row norms aggressively
    # before factorization; beats default FREE on ill-conditioned SDPs.
    sc = scaling.lower()
    if sc in ('aggressive', 'moderate', 'none', 'free'):
        try:
            mdl.setSolverParam('intpntScaling', sc)
        except Exception:
            pass

    # Presolve improvements.  Under pre_elim the Z/2 equalities are
    # gone, so the expensive lindep search will find nothing; caller
    # may pass presolve_lindep='off' to skip it entirely.
    mdl.setSolverParam('presolveUse', 'on')
    mdl.setSolverParam('presolveLindepUse',
                        'on' if presolve_lindep == 'on' else 'off')

    # Reordering heuristic (graph-partitioned AMD, usually best for banded).
    om = order_method.lower()
    try:
        mdl.setSolverParam('intpntOrderMethod', om)
    except Exception:
        # Older MOSEK may not accept 'experimental'; try fallback.
        try:
            mdl.setSolverParam('intpntOrderMethod', 'tryGraphpar')
        except Exception:
            pass

    # Threads.
    mdl.setSolverParam('numThreads', int(threads))

    applied = {
        'solve_form': solve_form, 'basis': 'never',
        'tol': tol,
        'presolve_lindep': presolve_lindep,
        'order_method': om, 'num_threads': int(threads),
        'max_iterations': int(max_iterations), 'scaling': sc,
    }
    if verbose:
        print(f"  Tuning: {applied}", flush=True)
    return applied


def apply_baseline_params(mdl: Model, *, verbose: bool = True
                           ) -> Dict[str, Any]:
    """Match what lasserre_scalable.solve_cg does: only `intpntCoTolRelGap
    = 1e-7` and defaults everywhere else.  This is our "untuned" baseline.
    """
    mdl.setSolverParam('intpntCoTolRelGap', 1e-7)
    applied = {'baseline': True, 'intpntCoTolRelGap': 1e-7}
    if verbose:
        print(f"  Baseline params: {applied}", flush=True)
    return applied


# =====================================================================
# Build the full Lasserre model (t as Variable — linear objective)
# =====================================================================

def _scipy_to_fusion(mat: sp.csr_matrix) -> Matrix:
    """Convert a scipy CSR/COO sparse matrix to MOSEK Fusion Matrix."""
    coo = mat.tocoo()
    return Matrix.sparse(
        int(coo.shape[0]), int(coo.shape[1]),
        coo.row.astype(np.int32).tolist(),
        coo.col.astype(np.int32).tolist(),
        coo.data.astype(np.float64).tolist())


def _build_full_model(
    P: Dict[str, Any], *,
    add_upper_loc: bool = True,
    verbose: bool = True,
    return_B: bool = False,
) -> Tuple[Model, Any, Any, Dict[str, Any]]:
    """Construct the full Lasserre feasibility model IN THE SHIFTED
    CHEBYSHEV BASIS.

    Variable ``c`` of length ``n_y`` represents the Chebyshev moment
    vector, indexed identically to the original monomial moment vector
    ``y`` (both over ``P['mono_list']``).  The two are linked by

        y_alpha = sum_gamma B[alpha, gamma] c_gamma        (exact)

    where B is the triangular change-of-basis matrix built by
    ``build_B_matrix``.  ``B[alpha, gamma]`` equals
    ``prod_i b[alpha_i][gamma_i]`` with ``b`` the integer-exact 1-D
    table from ``lasserre.cheby_basis.compute_b_table``.

    The SDP constraints become:

      * y_0 = 1                         ->  c_0 = 1  (B[0,0] = 1).
      * y_alpha >= 0 for all alpha      ->  (B @ c) >= 0 component-wise.
      * Consistency A_con @ y = 0       ->  (A_con @ B) @ c = 0.
      * Moment PSD M_k(y) >= 0          ->  M_k^cheb(c) >= 0  (V-similar).
      * Localizing M_{k-1}(mu_i y) >= 0 ->  M_{k-1}^cheb(mu_i c) >= 0.
      * Upper localizing                ->  M_{k-1}^cheb(c) - M_{k-1}^cheb(mu_i c) >= 0.
      * Scalar window t >= f_W(y)       ->  t >= (F @ B) @ c.
      * Window PSD t M_{k-1}(y) - Q_W(y) ->  t M_{k-1}^cheb(c) - Q_W^cheb(c) >= 0.

    Every Chebyshev entry map is assembled in ``lasserre.cheby_basis``
    using ``fractions.Fraction`` and converted to float64 only at the
    very end when the sparse MOSEK matrices are created.

    Returns (model, c_var, t_param, stats_dict [, B_sparse]).
    """
    d = P['d']
    n_y = P['n_y']
    n_basis = P['n_basis']
    n_loc = P['n_loc']
    n_win = P['n_win']
    idx = P['idx']
    mono_list = P['mono_list']
    basis = P['basis']
    loc_basis = P['loc_basis']

    t0 = time.time()

    # ---- Change-of-basis matrix B:  y = B c  -----------------------
    B_sp = build_B_matrix(mono_list, idx)
    if verbose:
        print(f"  B matrix: shape {B_sp.shape}, nnz={B_sp.nnz}",
              flush=True)

    mdl = Model('lasserre_cheby')
    c = mdl.variable(n_y, Domain.unbounded())
    t_var = mdl.parameter('t')

    # ---- c_0 = 1  (y_0 = B[0,0] c_0 + 0 = c_0, and y_0 = 1) -------
    zero = tuple(0 for _ in range(d))
    mdl.constraint(c.index(idx[zero]), Domain.equalsTo(1.0))

    # ---- y >= 0 component-wise via (B @ c) >= 0  ------------------
    B_mosek = _scipy_to_fusion(B_sp)
    mdl.constraint(Expr.mul(B_mosek, c), Domain.greaterThan(0.0))

    # ---- Consistency (A_con @ B) @ c = 0  -------------------------
    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']
    c_rows, c_cols, c_vals = [], [], []
    n_consist = 0
    for r in range(len(P['consist_mono'])):
        ai = int(consist_idx[r])
        if ai < 0:
            continue
        child_idx = consist_ei_idx[r]
        coef_by_col: Dict[int, float] = {}
        has_child = False
        for ci in range(d):
            cc = int(child_idx[ci])
            if cc >= 0:
                coef_by_col[cc] = coef_by_col.get(cc, 0.0) + 1.0
                has_child = True
        if not has_child:
            continue
        for cc, coef in coef_by_col.items():
            c_rows.append(n_consist)
            c_cols.append(cc)
            c_vals.append(coef)
        c_rows.append(n_consist)
        c_cols.append(ai)
        c_vals.append(-1.0)
        n_consist += 1
    if n_consist > 0:
        A_con_y = sp.coo_matrix(
            (c_vals, (c_rows, c_cols)), shape=(n_consist, n_y)).tocsr()
        A_con_c = (A_con_y @ B_sp).tocoo()
        A_con_c.sum_duplicates()
        A_con_c.eliminate_zeros()
        A_con_mosek = _scipy_to_fusion(A_con_c)
        mdl.constraint(Expr.mul(A_con_mosek, c),
                       Domain.equalsTo(0.0))
    if verbose:
        print(f"  Consistency: {n_consist} equations", flush=True)

    # ---- Moment matrix PSD:  M_k^cheb(c) ⪰ 0  ---------------------
    G_M = build_moment_matrix_map(basis, idx, n_y)
    if verbose:
        print(f"  Moment map G_M: shape {G_M.shape}, nnz={G_M.nnz}",
              flush=True)
    G_M_mosek = _scipy_to_fusion(G_M)
    M_flat = Expr.mul(G_M_mosek, c)
    M_mat = Expr.reshape(M_flat, n_basis, n_basis)
    mdl.constraint(M_mat, Domain.inPSDCone(n_basis))

    # ---- Localizing (mu_i c)  and  upper-localizing ---------------
    #      and pre-build the T-matrix map shared with the window PSDs.
    if P['order'] >= 2:
        G_T = build_t_matrix_map(loc_basis, idx, n_y)
        G_T_mosek = _scipy_to_fusion(G_T)
        T_flat = Expr.mul(G_T_mosek, c)

        loc_maps: List[sp.csr_matrix] = []
        for i_var in range(d):
            G_L_i = build_loc_matrix_map(loc_basis, idx, n_y, i_var)
            loc_maps.append(G_L_i)
            G_L_mosek = _scipy_to_fusion(G_L_i)
            L_flat = Expr.mul(G_L_mosek, c)
            L_mat = Expr.reshape(L_flat, n_loc, n_loc)
            mdl.constraint(L_mat, Domain.inPSDCone(n_loc))
        if verbose:
            avg_nnz = (sum(m.nnz for m in loc_maps) / len(loc_maps)
                       if loc_maps else 0)
            print(f"  Localizing maps: {len(loc_maps)} cones, "
                  f"avg nnz={avg_nnz:.0f}", flush=True)

        if add_upper_loc:
            for i_var in range(d):
                G_diff = G_T - loc_maps[i_var]
                G_diff_mosek = _scipy_to_fusion(G_diff)
                diff_flat = Expr.mul(G_diff_mosek, c)
                diff_mat = Expr.reshape(diff_flat, n_loc, n_loc)
                mdl.constraint(diff_mat, Domain.inPSDCone(n_loc))
    else:
        G_T = None
        G_T_mosek = None
        loc_maps = []

    # ---- Scalar window constraints:  t >= (F @ B) @ c  ------------
    F_y = sp.csr_matrix(
        (P['f_v'], (P['f_r'], P['f_c'])), shape=(n_win, n_y),
        dtype=np.float64)
    F_c = (F_y @ B_sp).tocoo()
    F_c.sum_duplicates()
    F_c.eliminate_zeros()
    if F_c.nnz > 0:
        F_mosek = _scipy_to_fusion(F_c)
        f_all = Expr.mul(F_mosek, c)
        ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
        t_rep = Expr.flatten(Expr.mul(ones_col, Expr.reshape(t_var, 1, 1)))
        mdl.constraint(Expr.sub(t_rep, f_all),
                       Domain.greaterThan(0.0))

    # ---- Window PSD cones:  t M_{k-1}^cheb(c) - Q_W^cheb(c) ⪰ 0 ---
    n_win_psd_added = 0
    win_active = list(P['nontrivial_windows'])
    if P['order'] >= 2 and G_T is not None:
        # Reuse the same T-flat times t for every window.
        t_T_flat = Expr.mul(t_var, Expr.mul(G_T_mosek, c))
        for w in win_active:
            Mw = P['M_mats'][w]
            G_Q = build_window_Q_map(loc_basis, idx, n_y, Mw)
            if G_Q.nnz == 0:
                # No mu_i mu_j terms: just t M_{k-1}^cheb(c) ⪰ 0.
                Lw_mat = Expr.reshape(t_T_flat, n_loc, n_loc)
                mdl.constraint(Lw_mat, Domain.inPSDCone(n_loc))
                n_win_psd_added += 1
                continue
            G_Q_mosek = _scipy_to_fusion(G_Q)
            Q_flat = Expr.mul(G_Q_mosek, c)
            Lw_flat = Expr.sub(t_T_flat, Q_flat)
            Lw_mat = Expr.reshape(Lw_flat, n_loc, n_loc)
            mdl.constraint(Lw_mat, Domain.inPSDCone(n_loc))
            n_win_psd_added += 1

    mdl.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    build_time = time.time() - t0
    stats = {
        'build_time_s': build_time,
        'n_y': n_y,
        'n_basis': n_basis,
        'n_loc': n_loc,
        'n_consist': n_consist,
        'n_win_psd': n_win_psd_added,
        'n_win_active': len(win_active),
        'n_win_original': len(P['nontrivial_windows']),
        'n_loc_active': d,
        'n_loc_original': d,
        'z2_mode': 'off',
        'add_upper_loc': add_upper_loc,
        'B_nnz': int(B_sp.nnz),
        'basis_kind': 'chebyshev',
    }
    if verbose:
        print(f"  Build (cheby): n_y={n_y}, n_basis={n_basis}, "
              f"n_loc={n_loc}, n_consist={n_consist}, "
              f"n_win_psd={n_win_psd_added}, "
              f"build_time={build_time:.2f}s",
              flush=True)
    if return_B:
        return mdl, c, t_var, stats, B_sp
    return mdl, c, t_var, stats


# =====================================================================
# Top-level solve
# =====================================================================

def _export_proof_artefacts(
    mdl: Model, y: Any, t_param: Any, t_val: float,
    verdict: str, stat_str: str,
    dest_dir: str, tag: str, *,
    d: int, order: int, mode: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Dump everything needed for an independently verifiable proof.

    Writes to `dest_dir/`:
      • {tag}.task.gz       — the complete MOSEK problem (reproducible by
                               any MOSEK 10+ installation).
      • {tag}.y.npy         — moment vector y at t = t_val.
      • {tag}.meta.json     — t_val, verdict, primal/dual statuses, MOSEK
                               version, solver parameters, timestamp.

    If ``verdict == 'infeas'`` (i.e. primal infeasibility certified by a
    dual certificate) the dual-variable vector IS the mathematical proof:
    any reviewer can verify A^T·y_cert = 0, y_cert ∈ K*, b^T·y_cert > 0
    using only the task file's A, b and the y.npy.
    """
    import json as _json
    os.makedirs(dest_dir, exist_ok=True)
    task_path = os.path.join(dest_dir, f'{tag}.task.gz')
    y_path = os.path.join(dest_dir, f'{tag}.y.npy')
    meta_path = os.path.join(dest_dir, f'{tag}.meta.json')

    # 1. Full task file.  Fusion's writeTask serialises the problem
    # (A, b, c, cones) in MOSEK's native format.  Reproducible anywhere.
    try:
        mdl.writeTask(task_path)
    except Exception as exc:
        task_path = None
        if verbose:
            print(f"  proof: writeTask failed: {exc}", flush=True)

    # 2. Primal solution (only meaningful if status is Optimal/Feasible).
    try:
        import numpy as _np
        if verdict == 'feas':
            _np.save(y_path, _np.asarray(y.level(), dtype=_np.float64))
        else:
            # Still save whatever MOSEK last computed, for debugging.
            try:
                _np.save(y_path, _np.asarray(y.level(), dtype=_np.float64))
            except Exception:
                y_path = None
    except Exception:
        y_path = None

    # 3. Metadata.
    meta = {
        'd': d, 'order': order, 'mode': mode,
        't_val': float(t_val),
        'verdict': verdict,
        'status_string': stat_str,
        'mosek_version': str(mosek.Env.getversion()),
        'task_file': os.path.basename(task_path)
            if task_path else None,
        'y_file': os.path.basename(y_path) if y_path else None,
        'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                         time.gmtime()),
    }
    with open(meta_path, 'w') as f:
        _json.dump(meta, f, indent=2)

    if verbose:
        print(f"  proof: wrote {task_path}, {y_path}, {meta_path}",
              flush=True)
    return {'task_path': task_path, 'y_path': y_path,
            'meta_path': meta_path, 'meta': meta}


def solve_mosek_tuned(
    d: int, order: int, *,
    mode: str = 'tuned',
    add_upper_loc: bool = True,
    n_bisect: int = 15,
    t_lo: float = 0.5,
    t_hi: Optional[float] = None,
    proof_dir: Optional[str] = None,
    primary_tol: float = 1e-6,
    order_method: str = 'forceGraphpar',
    force_lindep_off: bool = False,
    watcher_interval_s: float = 15.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve the full d × L{order} Lasserre SDP IN THE SHIFTED CHEBYSHEV
    BASIS via MOSEK Fusion and bisection on t.  Reuses a single Model
    across bisection steps — only t's value changes.

    mode ∈ {baseline, tuned}
      baseline : solver defaults (no tuning)
      tuned    : tuned params (same parameter suite as
                 ``lasserre_mosek_tuned.py`` mode=tuned)

    Returns dict with lb, status, timings, per-solve wall times.
    """
    if mode not in ('baseline', 'tuned'):
        raise ValueError(f"Unknown mode {mode!r}")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"MOSEK Lasserre (CHEBYSHEV): d={d} L{order} mode={mode}")
        print(f"{'=' * 60}", flush=True)

    # --- Precompute ---
    P = _precompute(d, order, verbose=verbose)

    t_build_start = time.time()
    mdl, y, t_param, build_stats = _build_full_model(
        P, add_upper_loc=add_upper_loc, verbose=verbose)
    build_time = time.time() - t_build_start

    # --- Apply params ---
    if mode == 'baseline':
        params = apply_baseline_params(mdl, verbose=verbose)
    else:
        # Lever 4 (only enable when user opts in via --lindep-off):
        # under pre_elim MOSEK's lindep search finds nothing, but
        # empirically at small d disabling it is ~10% SLOWER (MOSEK
        # finds other micro-redundancies).  Default safe: keep it on.
        lindep = 'off' if force_lindep_off else 'on'
        params = apply_tuned_params(
            mdl, tol=primary_tol,
            order_method=order_method,
            presolve_lindep=lindep,
            verbose=verbose)

    # --- MOSEK log stream for per-iteration ETAs ---
    # Attach a line-buffered capture that parses MOSEK's IPM iteration
    # rows live and emits an in-flight ETA after each row.  Essential
    # for long d=16 solves where the verdict can be hours away.
    log_stream = MosekLogStream(prefix='[MOSEK] ')
    try:
        mdl.setLogHandler(log_stream)
        # Max out MOSEK's own logging across every phase so we see where
        # time goes — otherwise the symbolic/allocation phase emits
        # nothing for tens of minutes.
        mdl.setSolverParam('log', 10)
        mdl.setSolverParam('logIntpnt', 10)
        # Print factorization breakdowns (setup, ordering, pivoting).
        try:
            mdl.setSolverParam('logIntpntFactor', 10)
        except Exception:
            pass
        # Log presolve iterations at higher detail.
        try:
            mdl.setSolverParam('logPresolve', 10)
        except Exception:
            pass
        # Log how often MOSEK rewrites progress: 1 = every iter line.
        try:
            mdl.setSolverParam('logIntpntFreq', 1)
        except Exception:
            pass
    except Exception as exc:
        if verbose:
            print(f"  log handler attach failed: {exc}", flush=True)

    # --- Watcher thread: emits a status row every N s during solve.
    # Unlike MOSEK's log, which is quiet during the serial symbolic
    # phase, the watcher samples the process from the outside so we
    # always know the solver is alive and how much RAM it's using.
    watcher = SolverWatcher(interval_s=watcher_interval_s, tag='solve')

    # --- Bisect on t (model reused across all steps) ---
    val = val_d_known.get(d)
    if t_hi is None:
        t_hi = (val + 0.05) if val else 2.0

    per_solve_times: List[float] = []
    err: Optional[str] = None
    ok = True

    def _classify(ps, ds) -> str:
        """Combine primal and dual status into a verdict.

        • 'feas'    : primal Optimal/Feasible  (certifies feasibility)
        • 'infeas'  : dual Optimal with primal infeasible certificate,
                      OR primal unknown + dual strong infeasibility signal
        • 'uncertain' : neither side conclusive.
        """
        if ps in (SolutionStatus.Optimal, SolutionStatus.Feasible):
            return 'feas'
        # MOSEK Fusion signals certified primal infeasibility through a
        # Dual "certificate" status.
        if ds == SolutionStatus.Certificate:
            return 'infeas'
        # Primal Undefined with Dual Unknown/Undefined — can't decide.
        return 'uncertain'

    # Retry ladder: when the primary solve returns 'uncertain' we
    # re-solve with alternative (tolerance, solve_form) combinations
    # and accept the first that certifies.
    #
    # Design goal: be robust across choices of primary_tol.
    #   • Loose tolerances (1e-3, 1e-4) rescue ill-conditioned boundary
    #     cases where tight tolerance IPM stalls — the central path is
    #     less aggressive and the conditioning cliff near val(d) is
    #     sidestepped.
    #   • Tight tolerances (1e-7) rescue cases where the primary was
    #     too loose to produce a clean certificate.
    #   • Primal and dual solve forms construct the Schur complement
    #     on opposite sides; one often certifies where the other fails.
    #
    # We skip any tolerance that equals primary_tol (already tried).
    def _make_ladder(primary: float) -> List[Tuple[float, str]]:
        candidate_tols = (1e-4, 1e-5, 1e-6, 1e-7, 1e-3)
        rungs: List[Tuple[float, str]] = []
        for tol in candidate_tols:
            if abs(tol - primary) < primary * 0.01:
                continue  # already tried at primary
            for form in ('primal', 'dual'):
                rungs.append((tol, form))
        return rungs

    RETRY_LADDER: List[Tuple[float, str]] = _make_ladder(primary_tol)

    def _check_feasible(tv: float, retry_on_unknown: bool = True
                         ) -> Tuple[str, float, str]:
        """Solve at t = tv.  Returns (verdict, wall_s, status_string).

        verdict ∈ {'feas', 'infeas', 'uncertain'}.  On 'uncertain' we
        step through RETRY_LADDER; each retry tries an alternative
        (tolerance, solve form) pair.  First non-uncertain verdict wins.
        """
        t_param.setValue(tv)
        ts = time.time()
        try:
            log_stream.mark_solve_start()
            watcher.start()
            try:
                mdl.solve()
            finally:
                watcher.stop()
            ps = mdl.getPrimalSolutionStatus()
            ds = mdl.getDualSolutionStatus()
            verdict = _classify(ps, ds)
            stat_str = (f"{str(ps).split('.')[-1]}/"
                        f"{str(ds).split('.')[-1]}")
        except Exception as exc:
            return 'uncertain', time.time() - ts, \
                    f'error:{type(exc).__name__}'

        # Remember the primary config so we can restore it after the ladder.
        if verdict == 'uncertain' and retry_on_unknown:
            original_tol = primary_tol
            original_form = 'dual'
            original_iters = 1600
            for retry_tol, retry_form in RETRY_LADDER:
                if verdict != 'uncertain':
                    break
                try:
                    mdl.setSolverParam('intpntSolveForm', retry_form)
                    mdl.setSolverParam('intpntCoTolRelGap', retry_tol)
                    mdl.setSolverParam('intpntCoTolPfeas', retry_tol)
                    mdl.setSolverParam('intpntCoTolDfeas', retry_tol)
                    mdl.setSolverParam('intpntCoTolMuRed', retry_tol)
                    mdl.setSolverParam('intpntMaxIterations', 2400)
                    mdl.solve()
                    ps2 = mdl.getPrimalSolutionStatus()
                    ds2 = mdl.getDualSolutionStatus()
                    v2 = _classify(ps2, ds2)
                    stat_str += (
                        f" -> {retry_form}/{retry_tol:.0e}:"
                        f"{str(ps2).split('.')[-1]}/"
                        f"{str(ds2).split('.')[-1]}"
                    )
                    if v2 != 'uncertain':
                        verdict = v2
                        break
                except Exception:
                    pass
            # Restore primary config regardless of outcome.
            try:
                mdl.setSolverParam('intpntSolveForm', original_form)
                mdl.setSolverParam('intpntCoTolRelGap', original_tol)
                mdl.setSolverParam('intpntCoTolPfeas', original_tol)
                mdl.setSolverParam('intpntCoTolDfeas', original_tol)
                mdl.setSolverParam('intpntCoTolMuRed', original_tol)
                mdl.setSolverParam('intpntMaxIterations', original_iters)
            except Exception:
                pass

        return verdict, time.time() - ts, stat_str

    proof_records: List[Dict[str, Any]] = []

    def _maybe_export(tag: str, tv: float, verdict: str,
                      stat_str: str) -> None:
        if proof_dir is None:
            return
        # For the record attempt we care about ANY certified verdict.
        # infeas  → dual certificate is the proof.
        # feas    → primal solution shows t_val is achievable.
        if verdict in ('feas', 'infeas'):
            rec = _export_proof_artefacts(
                mdl, y, t_param, tv, verdict, stat_str,
                proof_dir, tag,
                d=d, order=order, mode=mode, verbose=verbose)
            proof_records.append(rec)

    # Ensure the upper bound is feasible.
    lo, hi = float(t_lo), float(t_hi)
    v_hi, dt_hi, stat_hi = _check_feasible(hi)
    per_solve_times.append(dt_hi)
    _maybe_export(f'hi_probe_t{hi:.6f}', hi, v_hi, stat_hi)
    if verbose:
        print(f"\n  hi probe: t={hi:.6f} -> {v_hi} {stat_hi} "
              f"({dt_hi:.2f}s)", flush=True)
    tries = 0
    while v_hi != 'feas' and tries < 4:
        hi *= 1.5
        v_hi, dt_hi, stat_hi = _check_feasible(hi)
        per_solve_times.append(dt_hi)
        if verbose:
            print(f"  hi probe: t={hi:.6f} -> {v_hi} {stat_hi} "
                  f"({dt_hi:.2f}s)", flush=True)
        tries += 1

    if v_hi != 'feas':
        err = (f"upper bound t={hi} not feasibly solved "
                f"after {tries + 1} tries")
        ok = False

    # Bisection.  Only commit the bracket on 'feas' or 'infeas' verdicts.
    # 'uncertain' steps are recorded but do NOT shrink the bracket
    # (protecting the soundness of the certified lower bound).
    # When the bracket is unchanged by an uncertain step we perturb the
    # next t to avoid re-hitting the same numerically-borderline point.
    history = []
    n_uncertain = 0
    consecutive_uncertain = 0
    if ok:
        pending_offset = 0.0  # perturbation for next step
        for step in range(n_bisect):
            mid = 0.5 * (lo + hi) + pending_offset
            mid = max(lo + 1e-9, min(hi - 1e-9, mid))
            verdict, dt, stat = _check_feasible(mid)
            per_solve_times.append(dt)
            history.append({'step': step, 't': mid, 'status': stat,
                              'verdict': verdict, 'wall_s': dt})
            _maybe_export(f'step{step + 1:02d}_t{mid:.6f}',
                          mid, verdict, stat)
            if verdict == 'feas':
                hi = mid
                pending_offset = 0.0
                consecutive_uncertain = 0
            elif verdict == 'infeas':
                lo = mid
                pending_offset = 0.0
                consecutive_uncertain = 0
            else:
                n_uncertain += 1
                consecutive_uncertain += 1
                # Offset to a different point at roughly 30% from lo
                # (favor exploring the infeas side — that's where lb
                # can still grow).  Alternate sign on repeated uncert.
                width = hi - lo
                if consecutive_uncertain == 1:
                    pending_offset = -0.20 * width  # closer to lo
                elif consecutive_uncertain == 2:
                    pending_offset = +0.20 * width  # closer to hi
                else:
                    pending_offset = (-1) ** consecutive_uncertain \
                                        * 0.10 * width
            if verbose:
                marker = {'feas': 'feas', 'infeas': 'infeas',
                           'uncertain': '?????'}[verdict]
                print(f"  [{step + 1}/{n_bisect}] t={mid:.8f}  "
                      f"{marker:6s} {stat:40s} ({dt:.2f}s)  "
                      f"[{lo:.6f}, {hi:.6f}]", flush=True)
            # Abandon after 3 consecutive uncertain — numerical boundary
            # is intrinsically fuzzy here, no more progress possible.
            if consecutive_uncertain >= 3:
                if verbose:
                    print(f"    3 consecutive uncertain; stopping "
                          f"bisection (lb preserved).", flush=True)
                break

    lb = lo  # largest infeasible is the certified lower bound

    # --- Gap closure ---
    gc_pct: Optional[float] = None
    if val and lb is not None and val > 1.0:
        gc_pct = 100.0 * (lb - 1.0) / (val - 1.0)

    # --- Cleanup ---
    try:
        mdl.dispose()
    except Exception:
        pass
    gc.collect()

    total_solve_time = sum(per_solve_times)
    n_solves = len(per_solve_times)
    avg_solve = total_solve_time / n_solves if n_solves else 0.0

    result = {
        'd': d, 'order': order, 'mode': mode,
        'lb': lb, 'val_d': val, 'gc_pct': gc_pct,
        'ok': ok, 'error': err,
        'build_time_s': build_time,
        'total_solve_time_s': total_solve_time,
        'n_solves': n_solves,
        'n_uncertain': n_uncertain,
        'avg_solve_time_s': avg_solve,
        'total_time_s': build_time + total_solve_time,
        'per_solve_times_s': per_solve_times,
        'build_stats': build_stats,
        'params': params,
        'history': history,
        'proof_dir': proof_dir,
        'proof_records': [
            {k: v for k, v in rec.items() if k != 'meta'}
            for rec in proof_records
        ],
        # Last solve's MOSEK iteration trace — useful for retrospective
        # timing analysis (mu decay rate, per-iter wall, etc.).
        'last_solve_iter_trace': list(log_stream.iter_rows),
        # Watcher samples (peak-RSS, CPU timeline).  Lets us reconstruct
        # how much RAM a future run would need on a smaller SKU.
        'watcher_peak_rss_mb': watcher.peak_rss_mb,
        'watcher_samples': list(watcher.samples),
    }
    if verbose:
        print(f"\n  lb={lb:.6f}  gc={gc_pct:.2f}%  "
              f"build={build_time:.2f}s  "
              f"avg_solve={avg_solve:.2f}s  "
              f"total_solve={total_solve_time:.2f}s  "
              f"n_solves={n_solves}", flush=True)
    return result


# =====================================================================
# CLI
# =====================================================================

def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--order', type=int, default=3)
    p.add_argument('--mode', choices=('baseline', 'tuned'),
                    default='tuned')
    p.add_argument('--no-upper-loc', action='store_true',
                    help='Omit (1-μ_i) upper-localizing cones.')
    p.add_argument('--n-bisect', type=int, default=15)
    p.add_argument('--t-lo', type=float, default=0.5)
    p.add_argument('--t-hi', type=float, default=None)
    p.add_argument('--json', type=str, default=None,
                    help='Optional JSON result output path.')
    p.add_argument('--proof-dir', type=str, default=None,
                    help='Directory to dump full proof artefacts — '
                         'MOSEK task file, primal/dual vectors, meta '
                         'JSON — one bundle per certified bisection '
                         'step.')
    p.add_argument('--primary-tol', type=float, default=1e-6,
                    help='Primary MOSEK IPM tolerance.  Default 1e-6.')
    p.add_argument('--order-method', type=str, default='forceGraphpar',
                    choices=('free', 'none', 'appminloc',
                              'experimental', 'tryGraphpar',
                              'forceGraphpar'),
                    help='MOSEK AMD ordering method.')
    p.add_argument('--watcher-interval', type=float, default=15.0,
                    help='Seconds between watcher status rows.  '
                         'Reduce for short debug runs.')
    p.add_argument('--lindep-off', action='store_true',
                    help='Disable MOSEK lindep presolve.')
    args = p.parse_args()

    r = solve_mosek_tuned(
        args.d, args.order,
        mode=args.mode,
        add_upper_loc=not args.no_upper_loc,
        n_bisect=args.n_bisect,
        t_lo=args.t_lo,
        t_hi=args.t_hi,
        proof_dir=args.proof_dir,
        primary_tol=args.primary_tol,
        order_method=args.order_method,
        force_lindep_off=args.lindep_off,
        watcher_interval_s=args.watcher_interval,
        verbose=True)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(r, f, indent=2, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(_main())
