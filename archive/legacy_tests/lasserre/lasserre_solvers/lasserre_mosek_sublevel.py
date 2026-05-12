#!/usr/bin/env python
"""MOSEK-tuned full Lasserre solver — with Z/2 symmetry and an optional
SUBLEVEL-hierarchy driver (Lasserre–Henrion).

This file is a superset of ``lasserre_mosek_tuned.py``.  The full
solver, Z/2 modes, retry ladder, proof export and MOSEK tuning are
unchanged.  In addition it exposes ``solve_mosek_sublevel``: a
sequence of SDPs with progressively larger moment-matrix bases

      B_0  ⊂  B_1  ⊂  ...  ⊂  B_K  =  full L{order} basis

At each step B_k the SDP is a VALID Lasserre relaxation: we keep the
full L{order} moment vector y and EVERY non-moment-matrix PSD / linear
constraint at L{order} size; the only thing restricted is the moment
matrix M_k(y) → its principal submatrix on B_k × B_k.  The principal
submatrix is PSD iff every principal sub-principal-submatrix is PSD,
so every B_k yields a RELAXATION of L{order}:

      lb(B_0)  ≤  lb(B_1)  ≤  ...  ≤  lb(B_K)  =  lb(L{order})  ≤  val(d).

Monotone non-decreasing by construction; match at top level by
construction.  No term sparsity, no chordal extension — the only lever
is basis growth.

Because every entry of y referenced by the (fixed) localising /
window PSD cones is already a moment variable, each B_k is
**automatically closed under the Lasserre structure** — no moment
has to be implicitly set to zero, no localising cone has to be
dropped.  Soundness is preserved step-by-step.

Full solver:

    • MSK_IPAR_INTPNT_SOLVE_FORM = DUAL       (often 3-10× on Lasserre SDP)
    • MSK_IPAR_INTPNT_BASIS = NEVER            (skip basis ID, meaningless
                                                 for SDP)
    • MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-6    (tolerance > bisection res)
    • MSK_IPAR_PRESOLVE_LINDEP_USE = ON        (prune redundant equalities)
    • MSK_IPAR_INTPNT_ORDER_METHOD = EXPERIMENTAL (graph-partitioned AMD)
    • MSK_IPAR_NUM_THREADS                     (physical core count)

Optional lossless symmetry modes (do NOT change the SDP value):

    z2_mode='off'        : baseline — no Z/2.
    z2_mode='equalities' : add Z/2 equality constraints y_α = y_{σ(α)}.
                           Post-presolve this roughly halves n_y.
    z2_mode='blockdiag'  : Z/2 equalities + block-diagonalize the
                           (σ-invariant) moment matrix M_k into its
                           sym/anti pieces.  ~4× on the moment-matrix
                           Schur contribution.

All variants use the FULL formulation — every window is an active PSD
cone, every localizing / upper-localizing / pairwise L3 constraint is
present.  No constraint generation; no relaxation.

USAGE
-----

    python tests/lasserre_mosek_tuned.py --d 4 --order 3 --mode baseline
    python tests/lasserre_mosek_tuned.py --d 6 --order 3 --mode tuned
    python tests/lasserre_mosek_tuned.py --d 8 --order 3 --mode z2_eq
    python tests/lasserre_mosek_tuned.py --d 8 --order 3 --mode z2_bd
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
from lasserre.z2_symmetry import z2_symmetry_pairs
from lasserre.z2_blockdiag import (build_blockdiag_picks,
                                     orbit_decomposition,
                                     localizing_sigma_reps,
                                     window_sigma_reps)
from lasserre.z2_elim import canonicalize_z2


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

def _build_full_model(
    P: Dict[str, Any], *,
    z2_mode: str = 'off',
    add_upper_loc: bool = True,
    pre_elim: bool = False,
    verbose: bool = True,
) -> Tuple[Model, Any, Any, Dict[str, Any]]:
    """Construct the full Lasserre feasibility model with t as a MOSEK
    Parameter.

    Window PSD constraints `t·M_{k-1}(y) − Q_W(y) ⪰ 0` are bilinear in
    (t, y); MOSEK (and any convex SDP solver) cannot handle that with t
    as a variable.  So we keep t as a Parameter and bisect.  Bisection
    REUSES the same Model, only changing t's value — no rebuild per
    step, very little overhead.

    z2_mode:
      'off'        — no Z/2.
      'equalities' — add y_α = y_{σ(α)} constraints.
      'blockdiag'  — add Z/2 equalities AND replace the moment PSD with
                     its sym/anti blocks.
      'full'       — blockdiag PLUS: drop one of every σ-paired localizing
                     cone (and upper-localizing cone) and one of every
                     σ-paired window PSD cone.  Under the Z/2 equalities
                     on y the dropped cones are provably redundant
                     (permutation-similar to the retained ones).

    Returns (model, y_var, t_param, stats_dict).
    """
    d = P['d']
    n_y = P['n_y']
    n_basis = P['n_basis']
    n_loc = P['n_loc']
    n_win = P['n_win']
    idx = P['idx']
    mono_list = P['mono_list']

    t0 = time.time()

    # Change #6 — omit constraint names; Fusion autogenerates enough for
    # diagnostics and dropping names removes the per-constraint setName
    # allocation/copy overhead for the ~10^6 constraints at d=16 L3.
    mdl = Model('lasserre_full')
    y = mdl.variable(n_y, Domain.greaterThan(0.0))
    t_var = mdl.parameter('t')

    # y_0 = 1
    zero = tuple(0 for _ in range(d))
    mdl.constraint(y.index(idx[zero]), Domain.equalsTo(1.0))

    # Moment consistency: sum over i of y_{α+e_i} = y_α  (as equalities
    # when every child α+e_i is in the reduced moment set S).
    #
    # Under pre_elim the α-indexed and σ(α)-indexed equations collapse to
    # the same reduced equation (Z/2 symmetry on consist_idx).  We dedup
    # by canonical-ai and aggregate child coefficients when two children
    # share a canonical orbit representative.
    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']
    c_rows, c_cols, c_vals = [], [], []
    n_consist = 0
    emitted_ai = set() if pre_elim else None
    for r in range(len(P['consist_mono'])):
        ai = int(consist_idx[r])
        if ai < 0:
            continue
        if emitted_ai is not None:
            if ai in emitted_ai:
                continue  # σ-partner α already produced this equation
            emitted_ai.add(ai)
        child_idx = consist_ei_idx[r]
        coef_by_col: Dict[int, float] = {}
        has_child = False
        for ci in range(d):
            c = int(child_idx[ci])
            if c >= 0:
                coef_by_col[c] = coef_by_col.get(c, 0.0) + 1.0
                has_child = True
        if not has_child:
            continue
        for c, coef in coef_by_col.items():
            c_rows.append(n_consist)
            c_cols.append(c)
            c_vals.append(coef)
        c_rows.append(n_consist)
        c_cols.append(ai)
        c_vals.append(-1.0)
        n_consist += 1
    if n_consist > 0:
        coo = sp.coo_matrix(
            (c_vals, (c_rows, c_cols)), shape=(n_consist, n_y))
        coo.sum_duplicates()
        coo.eliminate_zeros()
        A_con = Matrix.sparse(n_consist, n_y,
                               coo.row.tolist(),
                               coo.col.tolist(),
                               coo.data.tolist())
        mdl.constraint(Expr.mul(A_con, y), Domain.equalsTo(0.0))

    # ---- Z/2 equality constraints (all Z/2 modes require these) ----
    # Under pre_elim the equalities are implicit (y_α and y_{σα} are
    # literally the same ỹ coordinate), so we skip injecting them.
    n_z2_eq = 0
    if z2_mode in ('equalities', 'blockdiag', 'full') and not pre_elim:
        pairs = z2_symmetry_pairs(P)
        n_z2_eq = len(pairs)
        if n_z2_eq:
            eq_rows = []
            eq_cols = []
            eq_vals = []
            for r, (i, j) in enumerate(pairs):
                eq_rows.extend([r, r])
                eq_cols.extend([i, j])
                eq_vals.extend([1.0, -1.0])
            A_sym = Matrix.sparse(n_z2_eq, n_y, eq_rows, eq_cols, eq_vals)
            # Z/2 equalities y_i − y_j = 0 where σ-pair (i, j) have the
            # same |γ|, so under the rescale they remain ŷ_i − ŷ_j = 0
            # with unchanged coefficients.
            mdl.constraint(Expr.mul(A_sym, y), Domain.equalsTo(0.0))

    # ---- Precompute σ-representatives for localizing and windows ----
    # For z2_mode == 'full', we only emit PSD constraints at these
    # canonical positions.  Dropped cones are redundant under the
    # σ-equalities on y (permutation-similar to retained cones).
    if z2_mode == 'full':
        loc_fixed, loc_pairs = localizing_sigma_reps(d)
        loc_active = list(loc_fixed) + [p for (p, _) in loc_pairs]
        win_fixed, win_pairs = window_sigma_reps(d, P['windows'])
        # Only keep reps that are also in the nontrivial set.
        nontriv = set(P['nontrivial_windows'])
        win_active = [w for w in (list(win_fixed)
                                   + [p for (p, _) in win_pairs])
                       if w in nontriv]
    else:
        loc_active = list(range(d))
        win_active = list(P['nontrivial_windows'])

    # ---- Moment matrix PSD: full cone or sym+anti block-diag ----
    if z2_mode in ('blockdiag', 'full'):
        bd = build_blockdiag_picks(P['basis'], idx, n_y)
        T_sym = bd['T_sym']
        T_anti = bd['T_anti']
        n_sym = bd['n_sym']
        n_anti = bd['n_anti']

        T_sym_coo = T_sym.tocoo()
        if T_sym_coo.nnz:
            M_sym_mat = Matrix.sparse(
                T_sym.shape[0], n_y,
                T_sym_coo.row.tolist(),
                T_sym_coo.col.tolist(),
                T_sym_coo.data.tolist())
            sym_flat = Expr.mul(M_sym_mat, y)
            sym_2d = Expr.reshape(sym_flat, n_sym, n_sym)
            mdl.constraint(sym_2d, Domain.inPSDCone(n_sym))

        if n_anti > 0:
            T_anti_coo = T_anti.tocoo()
            if T_anti_coo.nnz:
                M_anti_mat = Matrix.sparse(
                    T_anti.shape[0], n_y,
                    T_anti_coo.row.tolist(),
                    T_anti_coo.col.tolist(),
                    T_anti_coo.data.tolist())
                anti_flat = Expr.mul(M_anti_mat, y)
                anti_2d = Expr.reshape(anti_flat, n_anti, n_anti)
                mdl.constraint(anti_2d, Domain.inPSDCone(n_anti))
    else:
        M_mat = Expr.reshape(y.pick(P['moment_pick']), n_basis, n_basis)
        mdl.constraint(M_mat, Domain.inPSDCone(n_basis))

    # ---- Localizing PSD: μ_i ≥ 0 and optionally 1 - μ_i ≥ 0 ----
    # In z2_full mode we emit only σ-representative cones.  Under the
    # σ-equalities on y, the dropped cones are permutation-similar to
    # the retained ones, so the constraint set is equivalent.
    if P['order'] >= 2:
        for i_var in loc_active:
            Li = Expr.reshape(y.pick(P['loc_picks'][i_var]), n_loc, n_loc)
            mdl.constraint(Li, Domain.inPSDCone(n_loc))
        if add_upper_loc:
            for i_var in loc_active:
                sub_moment = y.pick(P['t_pick'])
                mu_i_loc = y.pick(P['loc_picks'][i_var])
                diff_i = Expr.sub(sub_moment, mu_i_loc)
                L_upper = Expr.reshape(diff_i, n_loc, n_loc)
                mdl.constraint(L_upper, Domain.inPSDCone(n_loc))

    # ---- Scalar window constraints: t ≥ f_W(y) for all W ----
    F_mosek = Matrix.sparse(n_win, n_y, P['f_r'], P['f_c'], P['f_v'])
    f_all = Expr.mul(F_mosek, y)
    ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep = Expr.flatten(Expr.mul(ones_col, Expr.reshape(t_var, 1, 1)))
    mdl.constraint(Expr.sub(t_rep, f_all), Domain.greaterThan(0.0))

    # ---- Window PSD cones: t·M_{k-1}(y) - Q_W(y) ⪰ 0 ----
    # In z2_full mode we emit one PSD per σ-orbit (either the σ-fixed
    # window itself or the canonical representative of a σ-pair).  The
    # dropped σ-partner of each retained pair is redundant under the
    # σ-equalities on y.
    if P['order'] >= 2:
        ab_eiej_idx = P['ab_eiej_idx']
        ab_flat = P['ab_flat']
        t_y = Expr.mul(t_var, y.pick(P['t_pick']))
        n_win_psd_added = 0
        for w in win_active:
            Mw = P['M_mats'][w]
            nz_i, nz_j = np.nonzero(Mw)
            if len(nz_i) == 0 or ab_eiej_idx is None:
                Lw_mat = Expr.reshape(t_y, n_loc, n_loc)
                mdl.constraint(Lw_mat, Domain.inPSDCone(n_loc))
                n_win_psd_added += 1
                continue
            y_idx = ab_eiej_idx[:, :, nz_i, nz_j]
            valid = y_idx >= 0
            if not np.any(valid):
                Lw_mat = Expr.reshape(t_y, n_loc, n_loc)
                mdl.constraint(Lw_mat, Domain.inPSDCone(n_loc))
                n_win_psd_added += 1
                continue
            ab_exp = np.broadcast_to(ab_flat[:, :, None], y_idx.shape)
            mw_vals = Mw[nz_i, nz_j]
            mw_exp = np.broadcast_to(mw_vals[None, None, :], y_idx.shape)
            rows = ab_exp[valid].ravel().tolist()
            cols = y_idx[valid].ravel().tolist()
            vals = mw_exp[valid].ravel().tolist()
            flat_size = n_loc * n_loc
            Cw_mosek = Matrix.sparse(flat_size, n_y, rows, cols, vals)
            cw_expr = Expr.mul(Cw_mosek, y)
            Lw_flat = Expr.sub(t_y, cw_expr)
            Lw_mat = Expr.reshape(Lw_flat, n_loc, n_loc)
            mdl.constraint(Lw_mat, Domain.inPSDCone(n_loc))
            n_win_psd_added += 1
    else:
        n_win_psd_added = 0

    mdl.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    build_time = time.time() - t0
    stats = {
        'build_time_s': build_time,
        'n_y': n_y,
        'n_basis': n_basis,
        'n_loc': n_loc,
        'n_consist': n_consist,
        'n_z2_eq': n_z2_eq,
        'n_win_psd': n_win_psd_added,
        'n_win_active': len(win_active),
        'n_win_original': len(P['nontrivial_windows']),
        'n_loc_active': len(loc_active),
        'n_loc_original': d,
        'z2_mode': z2_mode,
        'add_upper_loc': add_upper_loc,
    }
    if z2_mode == 'blockdiag':
        stats['n_moment_sym'] = n_sym
        stats['n_moment_anti'] = n_anti
        stats['n_moment_original'] = n_basis
    if verbose:
        print(f"  Build: n_y={n_y}, n_basis={n_basis}, n_loc={n_loc}, "
              f"n_consist={n_consist}, n_z2_eq={n_z2_eq}, "
              f"n_win_psd={n_win_psd_added}, build_time={build_time:.2f}s",
              flush=True)
        if z2_mode == 'blockdiag':
            print(f"  Block-diag moment: {n_basis} "
                  f"-> sym {stats['n_moment_sym']} + "
                  f"anti {stats['n_moment_anti']}", flush=True)
    return mdl, y, t_var, stats


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
    pre_elim: bool = False,
    primary_tol: float = 1e-6,
    order_method: str = 'forceGraphpar',
    force_lindep_off: bool = False,
    watcher_interval_s: float = 15.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve the full d × L{order} Lasserre SDP via MOSEK Fusion and
    bisection on t.  Reuses a single Model across bisection steps —
    only t's value changes.

    mode ∈ {baseline, tuned, z2_eq, z2_bd, z2_full}
      baseline : solver defaults (no tuning, no Z/2)
      tuned    : tuned params, no Z/2
      z2_eq    : tuned params + Z/2 equalities
      z2_bd    : tuned params + Z/2 block-diag (implies equalities)
      z2_full  : z2_bd + σ-paired window / localizing drop

    pre_elim : if True AND mode is a Z/2 mode, substitute out the Z/2
               equalities in the Python precompute before handing the
               SDP to MOSEK.  Reduces n_y by ~50%, eliminates the
               expensive lindep presolve phase.  Losslessly equivalent.

    Returns dict with lb, status, timings, per-solve wall times.
    """
    if mode not in ('baseline', 'tuned', 'z2_eq', 'z2_bd', 'z2_full'):
        raise ValueError(f"Unknown mode {mode!r}")

    # pre_elim only makes sense for the Z/2 modes.
    if pre_elim and mode in ('baseline', 'tuned'):
        raise ValueError(
            f"pre_elim requires a Z/2 mode (z2_eq / z2_bd / z2_full); "
            f"got mode={mode!r}")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"MOSEK-tuned full Lasserre: d={d} L{order} mode={mode}")
        print(f"{'=' * 60}", flush=True)

    # --- Precompute ---
    P = _precompute(d, order, verbose=verbose)

    # --- Optional Z/2 pre-elimination (Lever 1) ---
    # Must happen BEFORE any SDP-structure counting since it changes n_y
    # and every pick array, yet the formulation is mathematically
    # identical (literal variable substitution on σ-orbits).
    if pre_elim:
        P = canonicalize_z2(P, verbose=verbose)

    # --- Build model ---
    z2_map = {
        'baseline': 'off', 'tuned': 'off',
        'z2_eq': 'equalities', 'z2_bd': 'blockdiag',
        'z2_full': 'full',
    }
    z2_mode = z2_map[mode]

    t_build_start = time.time()
    mdl, y, t_param, build_stats = _build_full_model(
        P, z2_mode=z2_mode,
        add_upper_loc=add_upper_loc,
        pre_elim=pre_elim,
        verbose=verbose)
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
# Sublevel (Lasserre–Henrion) driver
# =====================================================================
#
# Pipeline
# --------
# We reuse the full L{order} precompute (``_precompute(d, order)``).
# That pre-computes every moment y_α, every consistency equation,
# every localising matrix index, every window-localising pick array.
# These are PRESERVED AS-IS across every sublevel step — the only
# per-step change is the MOMENT-MATRIX PSD cone.
#
# active_mask : bool[n_basis]
#    True wherever a basis monomial is currently in B_k.  Starts as
#    ``degrees <= order - 1`` (the L{order-1} sub-basis, i.e. the
#    L2 basis when order=3).  Grows by up to N monomials per step.
#
# Scoring
# -------
# After solving S_k we extract the primal moment vector y_primal.
# For each candidate m ∉ B_k we form the principal submatrix
# Σ_k+{m}(y_primal) of M_k(y_primal) on B_k ∪ {m}, and compute the
# SCHUR COMPLEMENT c_m − b_m^T A⁺ b_m where A = Σ_k(y_primal),
# b_m is the y_primal-row of M_k at m restricted to B_k, c_m is
# (y_primal)_{2m}.  The Schur complement is proportional to the
# determinant of Σ_k+{m}; its SIGN tells whether adding m preserves
# PSD.  Most-negative Schur = monomial whose addition will most
# tighten the bound.  (Ranking identical to the min-eigenvalue score
# up to factors of det(A).)
#
# Termination
# -----------
# Stop on any of:
#   (a) lb_k > target_lb  AND  MOSEK returns a valid dual certificate
#       of infeasibility at t = target_lb or at the bracket's lo edge.
#   (b) B_k = full basis (no more moves; final lb = lb(L{order})).
#   (c) Wall-clock budget exhausted.
#
# Soundness
# ---------
# At every step every moment referenced by every emitted PSD cone
# is a variable of the SDP (we kept the full L{order} moment
# structure), so the model is ALWAYS well-posed — no implicit
# zero, no dropped cone.  MOSEK's own convergence check is
# required at each step (no loosened tolerance, no "accept
# inaccurate" flag).


def _basis_degrees(basis) -> np.ndarray:
    """Array of degrees ``|α|`` for each monomial in ``basis``."""
    return np.array([int(sum(m)) for m in basis], dtype=np.int64)


def _moment_pick_np(P: Dict[str, Any]) -> np.ndarray:
    """Return the flat moment_pick as an ``(n_basis, n_basis)`` np.ndarray.

    Some precompute implementations ship it as a plain list (lasserre_scalable)
    and others as an ``np.ndarray`` (lasserre/precompute).  Normalise here.
    """
    n_basis = P['n_basis']
    mp = P.get('moment_pick_np')
    if mp is None:
        mp = np.asarray(P['moment_pick'], dtype=np.int64)
    return np.asarray(mp).reshape(n_basis, n_basis)


def _sub_moment_pick_list(P: Dict[str, Any],
                          active_idx: np.ndarray) -> List[int]:
    """Flat pick list for ``M_k(y)[active_idx, active_idx]``.

    Shape: ``(n_active, n_active)`` flattened to a length-``n_active^2``
    list of y-indices suitable for ``y.pick(...)``.
    """
    mp = _moment_pick_np(P)
    sub = mp[np.ix_(active_idx, active_idx)].ravel()
    return sub.tolist()


def _verify_basis_closed(P: Dict[str, Any],
                          active_mask: np.ndarray) -> Dict[str, Any]:
    """Sanity check: every moment referenced by the moment-matrix
    block at active_mask AND every moment referenced by the
    (unchanged) localising / window PSD cones is present in y.

    The second half is trivially true (we kept full L{order} structure)
    but we verify it anyway.

    Returns a dict with counts of referenced moment indices, plus any
    violations (expected: zero).
    """
    n_basis = P['n_basis']
    active_idx = np.nonzero(active_mask)[0]
    mp = _moment_pick_np(P)
    moment_refs = mp[np.ix_(active_idx, active_idx)].ravel()
    missing_moment = int((moment_refs < 0).sum())

    missing_loc = 0
    t_pick = P.get('t_pick')
    if t_pick is not None:
        missing_loc += int((np.asarray(t_pick) < 0).sum())
    for lp in P.get('loc_picks', []) or []:
        missing_loc += int((np.asarray(lp) < 0).sum())

    ab = P.get('ab_eiej_idx')
    # ab can legitimately contain -1s for moments outside degree-2k;
    # the sparse window builder masks those away.  So we don't count
    # those as violations — we only need all referenced by
    # nontrivial windows.  Trust the window-PSD builder's own mask.

    return {
        'n_active': int(active_mask.sum()),
        'n_basis': n_basis,
        'n_moment_refs': int(moment_refs.size),
        'missing_moment': missing_moment,
        'missing_localizing': missing_loc,
    }


def _build_sublevel_base_model(
    P: Dict[str, Any], *,
    add_upper_loc: bool = True,
    verbose: bool = True,
) -> Tuple[Model, Any, Any, Dict[str, Any]]:
    """Build the Lasserre SDP SKELETON — every constraint EXCEPT the
    moment-matrix PSD cone.

    Why skeleton?  Across sublevel steps only the moment-matrix basis
    B_k changes; every other constraint (consistency, localising,
    upper-localising, scalar window, window PSD) is fixed at full
    L{order} size and stays structurally identical.  By building the
    skeleton ONCE and using ``constraint.remove() + add`` to swap the
    moment PSD each step, we avoid re-incurring the ~15-min Fusion
    construction at every sublevel step — save 15 min × N steps.

    OPTIMISATION (window-PSD loop):
      • Preserve numpy dtypes all the way to Matrix.sparse (Fusion 11
        accepts int32/float64 numpy arrays directly; .tolist() on
        multi-million-element arrays is ~5 min of pure Python object
        churn at d=16, entirely avoidable).

    Returns (model, y_var, t_param, stats).
    """
    d = P['d']
    n_y = P['n_y']
    n_basis = P['n_basis']
    n_loc = P['n_loc']
    idx = P['idx']
    order = P['order']

    t0 = time.time()
    mdl = Model('lasserre_sublevel')
    y = mdl.variable(n_y, Domain.greaterThan(0.0))
    t_var = mdl.parameter('t')

    # y_0 = 1
    zero = tuple(0 for _ in range(d))
    mdl.constraint(y.index(idx[zero]), Domain.equalsTo(1.0))

    # ---- Consistency (vectorised COO assembly) ----
    consist_idx_arr = np.asarray(P['consist_idx'], dtype=np.int64)
    consist_ei_arr = np.asarray(P['consist_ei_idx'], dtype=np.int64)
    # Row r is kept iff consist_idx[r] >= 0 AND at least one child is valid.
    valid_r_mask = consist_idx_arr >= 0
    if valid_r_mask.any():
        kept_r = np.nonzero(valid_r_mask)[0]
        kept_ai = consist_idx_arr[kept_r]
        kept_children = consist_ei_arr[kept_r]  # (n_kept, d)
        # Some rows still have no valid child.  Skip those.
        has_child_row = (kept_children >= 0).any(axis=1)
        kept_r = kept_r[has_child_row]
        kept_ai = kept_ai[has_child_row]
        kept_children = kept_children[has_child_row]
        n_consist = int(kept_r.size)
    else:
        n_consist = 0
    if n_consist > 0:
        # Children coefficients: +1 per valid child.  Aggregate by
        # scatter-add across row-local child slots.
        row_idx_rep = np.repeat(
            np.arange(n_consist, dtype=np.int64), kept_children.shape[1])
        children_flat = kept_children.ravel()
        child_valid = children_flat >= 0
        row_rows = row_idx_rep[child_valid]
        row_cols = children_flat[child_valid]
        row_vals = np.ones(row_rows.size, dtype=np.float64)
        # Append the -y_{alpha_i} diagonal entry for each row.
        ai_rows = np.arange(n_consist, dtype=np.int64)
        ai_cols = kept_ai
        ai_vals = -np.ones(n_consist, dtype=np.float64)
        all_rows = np.concatenate([row_rows, ai_rows])
        all_cols = np.concatenate([row_cols, ai_cols])
        all_vals = np.concatenate([row_vals, ai_vals])
        coo = sp.coo_matrix(
            (all_vals, (all_rows, all_cols)), shape=(n_consist, n_y))
        coo.sum_duplicates()
        coo.eliminate_zeros()
        A_con = Matrix.sparse(n_consist, n_y,
                              coo.row.astype(np.int32),
                              coo.col.astype(np.int32),
                              coo.data.astype(np.float64))
        mdl.constraint(Expr.mul(A_con, y), Domain.equalsTo(0.0))

    # ---- Localising PSD: FULL L{order} — μ_i ≥ 0 and 1 - μ_i ≥ 0 ----
    if order >= 2:
        for i_var in range(d):
            Li = Expr.reshape(y.pick(P['loc_picks'][i_var]), n_loc, n_loc)
            mdl.constraint(Li, Domain.inPSDCone(n_loc))
        if add_upper_loc:
            for i_var in range(d):
                sub_moment = y.pick(P['t_pick'])
                mu_i_loc = y.pick(P['loc_picks'][i_var])
                diff_i = Expr.sub(sub_moment, mu_i_loc)
                L_upper = Expr.reshape(diff_i, n_loc, n_loc)
                mdl.constraint(L_upper, Domain.inPSDCone(n_loc))

    # ---- Scalar windows: t ≥ f_W(y) ----
    n_win = P['n_win']
    f_r_np = np.asarray(P['f_r'], dtype=np.int32)
    f_c_np = np.asarray(P['f_c'], dtype=np.int32)
    f_v_np = np.asarray(P['f_v'], dtype=np.float64)
    F_mosek = Matrix.sparse(n_win, n_y, f_r_np, f_c_np, f_v_np)
    f_all = Expr.mul(F_mosek, y)
    ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep = Expr.flatten(Expr.mul(ones_col, Expr.reshape(t_var, 1, 1)))
    mdl.constraint(Expr.sub(t_rep, f_all), Domain.greaterThan(0.0))

    # ---- Window PSD: FULL L{order} cones — t·M_{k-1}(y) - Q_W(y) ⪰ 0 ----
    # Pass numpy arrays directly to Matrix.sparse — Fusion 11 accepts
    # int32 / float64 arrays, no .tolist() needed.  This alone is a
    # ~3-5 min build-time cut at d=16 (496 windows × ~1M entries each).
    n_win_psd = 0
    if order >= 2:
        ab_eiej_idx = P['ab_eiej_idx']
        ab_flat = P['ab_flat']  # (n_loc, n_loc) int
        flat_size = n_loc * n_loc
        t_y = Expr.mul(t_var, y.pick(P['t_pick']))
        for w in P['nontrivial_windows']:
            Mw = P['M_mats'][w]
            nz_i, nz_j = np.nonzero(Mw)
            if len(nz_i) == 0 or ab_eiej_idx is None:
                Lw_mat = Expr.reshape(t_y, n_loc, n_loc)
                mdl.constraint(Lw_mat, Domain.inPSDCone(n_loc))
                n_win_psd += 1
                continue
            y_idx = ab_eiej_idx[:, :, nz_i, nz_j]        # (nL, nL, k)
            valid = y_idx >= 0
            if not np.any(valid):
                Lw_mat = Expr.reshape(t_y, n_loc, n_loc)
                mdl.constraint(Lw_mat, Domain.inPSDCone(n_loc))
                n_win_psd += 1
                continue
            ab_exp = np.broadcast_to(ab_flat[:, :, None], y_idx.shape)
            mw_vals = Mw[nz_i, nz_j]
            mw_exp = np.broadcast_to(
                mw_vals[None, None, :], y_idx.shape)
            # Keep numpy dtypes; Matrix.sparse accepts int32/float64.
            rows = ab_exp[valid].ravel().astype(np.int32, copy=False)
            cols = y_idx[valid].ravel().astype(np.int32, copy=False)
            vals = mw_exp[valid].ravel().astype(np.float64, copy=False)
            Cw_mosek = Matrix.sparse(flat_size, n_y, rows, cols, vals)
            cw_expr = Expr.mul(Cw_mosek, y)
            Lw_flat = Expr.sub(t_y, cw_expr)
            Lw_mat = Expr.reshape(Lw_flat, n_loc, n_loc)
            mdl.constraint(Lw_mat, Domain.inPSDCone(n_loc))
            n_win_psd += 1

    mdl.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    build_time = time.time() - t0
    stats = {
        'build_time_s': build_time,
        'n_y': n_y,
        'n_basis': n_basis,
        'n_loc': n_loc,
        'n_consist': n_consist,
        'n_win_psd': n_win_psd,
    }
    if verbose:
        print(f"  [sublevel base build] n_y={n_y}  n_loc={n_loc}  "
              f"n_consist={n_consist}  n_win_psd={n_win_psd}  "
              f"build={build_time:.2f}s", flush=True)
    return mdl, y, t_var, stats


def _add_moment_psd(mdl: Model, y: Any, P: Dict[str, Any],
                     active_mask: np.ndarray) -> Any:
    """Add the moment-matrix PSD cone on the principal submatrix of
    M_k(y) indexed by ``active_mask``.  Returns the constraint handle
    so callers can ``.remove()`` it before adding the next step's cone.
    """
    active_idx = np.nonzero(active_mask)[0]
    n_active = int(active_idx.size)
    assert n_active >= 1, "active_mask must contain at least one entry"
    sub_pick = _sub_moment_pick_list(P, active_idx)
    M_sub = Expr.reshape(y.pick(sub_pick), n_active, n_active)
    cons = mdl.constraint(M_sub, Domain.inPSDCone(n_active))
    return cons


def _build_sublevel_model(
    P: Dict[str, Any], active_mask: np.ndarray, *,
    add_upper_loc: bool = True,
    verbose: bool = True,
) -> Tuple[Model, Any, Any, Dict[str, Any]]:
    """Back-compat wrapper: build skeleton + moment PSD.  Prefer
    ``_build_sublevel_base_model`` + ``_add_moment_psd`` in the
    sublevel loop so the skeleton is reused across steps.
    """
    mdl, y, t_var, stats = _build_sublevel_base_model(
        P, add_upper_loc=add_upper_loc, verbose=verbose)
    _add_moment_psd(mdl, y, P, active_mask)
    stats['n_active_basis'] = int(active_mask.sum())
    return mdl, y, t_var, stats


def _score_candidates_schur(
    y_primal: np.ndarray,
    P: Dict[str, Any],
    active_mask: np.ndarray,
    cand_indices: np.ndarray,
    reg: float = 1e-10,
) -> List[Tuple[int, float]]:
    """Rank candidate basis monomials by the Schur complement of
    Σ_{B_k ∪ {m}}(y_primal).

    A = M(y_primal)[B_k, B_k]  (PSD under the SDP constraint)
    b_m = M(y_primal)[B_k, m]
    c_m = M(y_primal)[m, m]
    Schur_m = c_m − b_m^T A⁺ b_m

    Ranking by Schur_m ascending → most-negative first → the monomial
    whose addition MOST forces the PSD block to become indefinite.

    Note: computed once per sublevel step with a single A-inverse
    factorisation, so cost is O(n_active³) + O(n_cand · n_active²).
    At d=16 L3 with n_active=153, n_cand=816: ~25 MFLOPs — milliseconds.
    """
    mp = _moment_pick_np(P)
    active_idx = np.nonzero(active_mask)[0]
    n_active = int(active_idx.size)
    cand_arr = np.asarray(cand_indices, dtype=np.int64)
    n_cand = int(cand_arr.size)
    if n_cand == 0:
        return []

    base_pick = mp[np.ix_(active_idx, active_idx)]
    A = y_primal[base_pick].astype(np.float64)
    A = 0.5 * (A + A.T)
    A = A + reg * np.eye(n_active)

    # A⁻¹: SPD Cholesky factor for stable inversion.  If not numerically
    # PD, fall back to pseudo-inverse via SVD.
    try:
        cho = np.linalg.cholesky(A)
        # A⁻¹ b for all b in parallel via triangular solves
        solve = lambda B: np.linalg.solve(
            cho.T, np.linalg.solve(cho, B))
    except np.linalg.LinAlgError:
        A_pinv = np.linalg.pinv(A)
        solve = lambda B: A_pinv @ B

    row_picks = mp[cand_arr[:, None], active_idx[None, :]]  # (n_cand, n_active)
    diag_picks = mp[cand_arr, cand_arr]                       # (n_cand,)
    B = y_primal[row_picks].astype(np.float64)
    C = y_primal[diag_picks].astype(np.float64)

    AB = solve(B.T)  # (n_active, n_cand)
    quad = np.einsum('ij,ji->i', B, AB)  # (n_cand,)
    schur = C - quad

    scored = [(int(cand_arr[i]), float(schur[i])) for i in range(n_cand)]
    scored.sort(key=lambda x: x[1])
    return scored


def _classify_sdp(ps, ds) -> str:
    """Local copy of the feasibility classifier; keeps sublevel driver
    self-contained (identical semantics to the full-solver version)."""
    if ps in (SolutionStatus.Optimal, SolutionStatus.Feasible):
        return 'feas'
    if ds == SolutionStatus.Certificate:
        return 'infeas'
    return 'uncertain'


def solve_mosek_sublevel(
    d: int, order: int, *,
    target_lb: float = 1.2802,
    n_per_step: int = 50,
    n_bisect_per_step: int = 6,
    t_lo: float = 0.5,
    t_hi: Optional[float] = None,
    time_budget_s: float = 7200.0,
    add_upper_loc: bool = True,
    primary_tol: float = 1e-6,
    order_method: str = 'forceGraphpar',
    force_lindep_off: bool = False,
    watcher_interval_s: float = 30.0,
    proof_dir: Optional[str] = None,
    progress_path: Optional[str] = None,
    initial_active_mask: Optional[np.ndarray] = None,
    z2_pre_elim: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Sublevel Lasserre driver.  See module docstring.

    Parameters
    ----------
    target_lb
        Stopping criterion — certified infeasibility at t = target_lb
        (or any t > target_lb proven infeasible via bisection) exits
        with ``final_status='target_reached'``.  Must be < val(d).
    n_per_step
        Top-N most-negative-Schur candidates added per step.
    n_bisect_per_step
        Number of bisection halvings inside each sublevel.  Used for
        precise lb_k reporting AND for driving the lower bracket above
        target_lb (early termination).
    time_budget_s
        Wall-clock cap across ALL sublevel steps — the driver exits
        cleanly if exceeded.
    initial_active_mask
        If given, use this as B_0 instead of the degree-≤-(order-1)
        sub-basis.  Useful for warm-starts from a previous run.
    progress_path
        If given, after EVERY sublevel step we write the trajectory so
        far to this path (JSON).  Makes 2-minute remote reporting
        trivial: tail the JSON.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"MOSEK SUBLEVEL Lasserre: d={d} L{order}")
        print(f"  target_lb    = {target_lb}")
        print(f"  n_per_step   = {n_per_step}")
        print(f"  n_bisect/step= {n_bisect_per_step}")
        print(f"  time_budget  = {time_budget_s:.0f}s")
        print(f"{'=' * 60}", flush=True)

    # --- Precompute full L{order} structure.  Shared across every step. ---
    P = _precompute(d, order, verbose=verbose)

    # --- Optional Z/2 pre-elimination (Lever 1 from the full solver) ---
    # Halves n_y losslessly by substituting the Z/2 equalities into the
    # moment vector before the SDP is ever built.  The basis list is
    # UNCHANGED (still all degree-≤-order multi-indices), only the
    # y-indexing shrinks — so active_mask / pool_mask (defined by basis
    # degree) remain meaningful, and all picks already point into the
    # canonicalised (smaller) y.  Downstream sublevel code, including
    # Schur scoring, is agnostic to whether Z/2 is applied.
    if z2_pre_elim:
        if verbose:
            print(f"\n  [Z/2 pre-elim] applying canonicalize_z2 ...",
                  flush=True)
        from lasserre.z2_elim import canonicalize_z2
        t_z2 = time.time()
        P = canonicalize_z2(P, verbose=verbose)
        if verbose:
            print(f"  [Z/2 pre-elim] n_y = {P['n_y']}  "
                  f"(elapsed {time.time() - t_z2:.2f}s)", flush=True)

    basis = P['basis']
    n_basis = P['n_basis']
    degrees = _basis_degrees(basis)

    if initial_active_mask is not None:
        active_mask = np.asarray(initial_active_mask, dtype=bool).copy()
        assert active_mask.shape == (n_basis,)
    else:
        # B_0 = degree ≤ order - 1 sub-basis.
        # For order=3 this is the L2 basis (degree ≤ 2).
        active_mask = degrees <= (order - 1)

    # Pool = "degree == order" monomials we can add.  At order=3 these
    # are exactly the degree-3 monomials (new to L3 vs L2).
    pool_mask = degrees == order
    n_pool_total = int(pool_mask.sum())

    if verbose:
        print(f"  full basis   = {n_basis} monomials (degree ≤ {order})")
        print(f"  |B_0|        = {int(active_mask.sum())} "
              f"(degree ≤ {order - 1})")
        print(f"  |pool|       = {n_pool_total} "
              f"(degree == {order})", flush=True)

    # Closed-set sanity check on B_0.
    closed0 = _verify_basis_closed(P, active_mask)
    if closed0['missing_moment'] > 0:
        raise RuntimeError(
            f"B_0 references {closed0['missing_moment']} moment indices "
            "not present in y; aborting.")

    val = val_d_known.get(d)
    if t_hi is None:
        t_hi = (val + 0.05) if val else 2.0

    trajectory: List[Dict[str, Any]] = []
    t_start = time.time()
    cur_lb: Optional[float] = None
    final_status = 'unknown'
    lb_prev: Optional[float] = None
    proof_records: List[Dict[str, Any]] = []

    # Bracket survives ACROSS sublevel steps.  Rationale: if at step k
    # we certified infeasibility at some t_inf (so lb_k ≥ t_inf), then
    # at step k+1 (a tighter relaxation) we STILL have lb_{k+1} ≥ t_inf
    # by monotonicity.  So we can start the bisection for step k+1 with
    # lo_{k+1} = lo_k.  Equivalently, the feas upper bound only
    # decreases across steps (SDP becomes tighter, t can rise further).
    # But keeping hi = t_hi_initial is safe and simple.
    lo = float(t_lo)
    hi = float(t_hi)

    # --- Build the SKELETON model ONCE.  The skeleton contains every
    # constraint EXCEPT the moment-matrix PSD cone — i.e. consistency,
    # y_0 = 1, localising + upper-localising PSDs, scalar windows,
    # window PSDs.  At every sublevel step we only .remove() the prior
    # moment-matrix PSD and add a fresh one on the expanded basis,
    # saving the full Python-side Fusion build cost (~15 min at d=16
    # L3) on every step after step 0.
    if verbose:
        print(f"\n  [skeleton] building base model ONCE (reused "
              f"across sublevel steps) ...", flush=True)
    t_skel = time.time()
    mdl, y, t_param, skel_stats = _build_sublevel_base_model(
        P, add_upper_loc=add_upper_loc, verbose=verbose)
    skeleton_build_time = time.time() - t_skel
    if verbose:
        print(f"  [skeleton] build complete in "
              f"{skeleton_build_time:.2f}s", flush=True)

    # Apply tuned params ONCE on the shared model.
    lindep = 'off' if force_lindep_off else 'on'
    params = apply_tuned_params(
        mdl, tol=primary_tol,
        order_method=order_method,
        presolve_lindep=lindep,
        verbose=verbose)

    # MOSEK log stream — re-attached before each solve with a per-step
    # prefix so ``tail -f`` readers can distinguish sublevel steps.
    log_stream = MosekLogStream(prefix='[MOSEK] ')
    try:
        mdl.setLogHandler(log_stream)
        mdl.setSolverParam('log', 10)
        mdl.setSolverParam('logIntpnt', 10)
    except Exception:
        pass

    moment_psd_cons: Optional[Any] = None  # current moment-PSD handle

    for step in range(n_basis + 1):  # safety upper bound
        elapsed_total = time.time() - t_start
        if elapsed_total > time_budget_s:
            final_status = 'time_budget_exceeded'
            break

        n_active = int(active_mask.sum())
        is_full = n_active >= n_basis

        if verbose:
            print(f"\n{'*' * 60}")
            print(f"*** Sublevel step {step}: |B_k| = {n_active}/{n_basis}"
                  f"  (elapsed {elapsed_total:.0f}s)")
            print(f"{'*' * 60}", flush=True)

        # Closed-set verification
        closed = _verify_basis_closed(P, active_mask)
        if closed['missing_moment'] > 0:
            raise RuntimeError(
                f"At step {step}, active_mask references "
                f"{closed['missing_moment']} moments not in y")

        # --- Swap the moment-matrix PSD cone on the shared skeleton ---
        # Remove the prior step's cone (if any) and add a fresh one on
        # the current active_mask.  This is the O(|B_k|²)-rows amendment
        # we make per step — vs the O(n_y) rebuild we'd otherwise need.
        t_build = time.time()
        if moment_psd_cons is not None:
            try:
                moment_psd_cons.remove()
            except Exception as exc:
                if verbose:
                    print(f"  WARNING: could not remove prior "
                          f"moment-PSD cons: {exc}", flush=True)
        moment_psd_cons = _add_moment_psd(mdl, y, P, active_mask)
        build_time = time.time() - t_build
        build_stats = dict(skel_stats)
        build_stats['n_active_basis'] = n_active
        build_stats['moment_psd_swap_s'] = build_time
        if step == 0:
            build_stats['skeleton_build_s'] = skeleton_build_time
        if verbose:
            print(f"  [moment-PSD swap] |B_k|={n_active}  "
                  f"swap_time={build_time:.2f}s", flush=True)

        # Rebind the MOSEK log stream to this step's prefix.
        log_stream.prefix = f'[MOSEK s{step}] '

        watcher = SolverWatcher(
            interval_s=watcher_interval_s, tag=f's{step}')

        step_bisect_history: List[Dict[str, Any]] = []
        y_primal_at_feas: Optional[np.ndarray] = None

        def _solve_at(tv: float) -> Tuple[str, float, str,
                                          Optional[np.ndarray]]:
            """Solve once at t = tv; on 'uncertain' try ONE retry with
            the opposite solve form (primal ↔ dual) — MOSEK's Schur
            complement is built on opposite sides so one often
            certifies where the other stalls.  Mirrors the retry
            ladder in solve_mosek_tuned, minus the tolerance sweeps.
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
                verdict = _classify_sdp(ps, ds)
                stat = (f"{str(ps).split('.')[-1]}/"
                        f"{str(ds).split('.')[-1]}")
            except Exception as exc:
                return 'uncertain', time.time() - ts, \
                       f'error:{type(exc).__name__}', None
            if verdict == 'uncertain':
                try:
                    mdl.setSolverParam('intpntSolveForm', 'primal')
                    mdl.solve()
                    ps2 = mdl.getPrimalSolutionStatus()
                    ds2 = mdl.getDualSolutionStatus()
                    v2 = _classify_sdp(ps2, ds2)
                    stat = (f"{stat} -> primal:"
                            f"{str(ps2).split('.')[-1]}/"
                            f"{str(ds2).split('.')[-1]}")
                    if v2 != 'uncertain':
                        verdict = v2
                        ps = ps2
                        ds = ds2
                except Exception:
                    pass
                finally:
                    try:
                        mdl.setSolverParam('intpntSolveForm', 'dual')
                    except Exception:
                        pass
            yv: Optional[np.ndarray] = None
            if verdict == 'feas':
                try:
                    yv = np.asarray(y.level(), dtype=np.float64)
                except Exception:
                    yv = None
            return verdict, time.time() - ts, stat, yv

        # -- Ensure bracket upper bound is feasible at THIS sublevel --
        v_hi, dt_hi, stat_hi, y_hi = _solve_at(hi)
        step_bisect_history.append({'t': hi, 'verdict': v_hi,
                                    'stat': stat_hi, 'wall_s': dt_hi,
                                    'phase': 'hi_probe'})
        if verbose:
            print(f"  hi-probe: t={hi:.6f} -> {v_hi} {stat_hi} "
                  f"({dt_hi:.2f}s)", flush=True)
        tries = 0
        while v_hi != 'feas' and tries < 4:
            hi *= 1.5
            v_hi, dt_hi, stat_hi, y_hi = _solve_at(hi)
            step_bisect_history.append({'t': hi, 'verdict': v_hi,
                                        'stat': stat_hi, 'wall_s': dt_hi,
                                        'phase': 'hi_probe_retry'})
            if verbose:
                print(f"  hi-probe: t={hi:.6f} -> {v_hi} {stat_hi} "
                      f"({dt_hi:.2f}s)", flush=True)
            tries += 1
        if v_hi != 'feas':
            if verbose:
                print(f"  WARNING: hi not feasible at step {step}; "
                      f"abandoning this sublevel", flush=True)
            try:
                mdl.dispose()
            except Exception:
                pass
            gc.collect()
            final_status = 'hi_infeasible'
            break
        if y_hi is not None:
            y_primal_at_feas = y_hi

        # -- Probe target_lb early: certifies if sublevel is enough --
        # If target_lb is INFEASIBLE at this step, lb_k ≥ target_lb and
        # we win.  If FEASIBLE, lb_k < target_lb, need more basis.
        target_probe_done = False
        target_probe_infeas = False
        if lo < target_lb < hi:
            v_tgt, dt_tgt, stat_tgt, y_tgt = _solve_at(target_lb)
            step_bisect_history.append({'t': target_lb, 'verdict': v_tgt,
                                        'stat': stat_tgt, 'wall_s': dt_tgt,
                                        'phase': 'target_probe'})
            if verbose:
                print(f"  target-probe: t={target_lb:.6f} -> {v_tgt} "
                      f"{stat_tgt} ({dt_tgt:.2f}s)", flush=True)
            target_probe_done = True
            if v_tgt == 'infeas':
                target_probe_infeas = True
                lo = target_lb
            elif v_tgt == 'feas':
                hi = target_lb
                if y_tgt is not None:
                    y_primal_at_feas = y_tgt

        # -- Regular bisection (bounded by n_bisect_per_step) --
        # Perturb on uncertain to avoid hammering the same t, and
        # break out after two consecutive uncertain verdicts (the
        # boundary is fuzzy, more bisections won't help).
        consec_unc = 0
        pending_perturb = 0.0
        for bstep in range(n_bisect_per_step):
            if time.time() - t_start > time_budget_s:
                break
            if hi - lo < 1e-4:
                break
            mid = 0.5 * (lo + hi) + pending_perturb
            mid = max(lo + 1e-9, min(hi - 1e-9, mid))
            v_m, dt_m, stat_m, y_m = _solve_at(mid)
            step_bisect_history.append({'t': mid, 'verdict': v_m,
                                        'stat': stat_m, 'wall_s': dt_m,
                                        'phase': f'bisect{bstep+1}'})
            if verbose:
                print(f"  [b{bstep+1}] t={mid:.8f} -> {v_m} {stat_m} "
                      f"({dt_m:.2f}s)  [{lo:.6f}, {hi:.6f}]", flush=True)
            if v_m == 'feas':
                hi = mid
                if y_m is not None:
                    y_primal_at_feas = y_m
                consec_unc = 0
                pending_perturb = 0.0
            elif v_m == 'infeas':
                lo = mid
                consec_unc = 0
                pending_perturb = 0.0
            else:
                # 'uncertain' — don't shrink bracket; perturb next t
                # off the midpoint so we don't re-hit the same number.
                consec_unc += 1
                width = hi - lo
                sign = 1.0 if (consec_unc % 2) else -1.0
                pending_perturb = sign * 0.20 * width
                if consec_unc >= 2:
                    if verbose:
                        print(f"    2 consecutive uncertain; stopping "
                              f"bisection at this sublevel", flush=True)
                    break

        lb_k = lo
        cur_lb = lb_k

        # -- Monotonicity check (invariant of the construction) --
        if lb_prev is not None and lb_k + 1e-6 < lb_prev:
            print(f"  *** MONOTONICITY VIOLATION: "
                  f"lb_k={lb_k:.6f} < lb_prev={lb_prev:.6f}",
                  flush=True)

        # -- Optional proof export --
        if proof_dir is not None:
            try:
                rec = _export_proof_artefacts(
                    mdl, y, t_param, lo,
                    'infeas' if target_probe_infeas else 'feas',
                    'sublevel', proof_dir,
                    f'sublevel_s{step:02d}_lb{lo:.6f}',
                    d=d, order=order, mode='sublevel',
                    verbose=verbose)
                proof_records.append(rec)
            except Exception as exc:
                print(f"  proof export failed: {exc}", flush=True)

        step_time = time.time() - t_build
        entry = {
            'step': step,
            'n_active': n_active,
            'n_active_pct': 100.0 * n_active / n_basis,
            'lb_k': lb_k,
            'hi_bracket': hi,
            'bracket_width': hi - lo,
            'target_probe_infeas': target_probe_infeas,
            'step_wall_s': step_time,
            'build_wall_s': build_time,
            'build_stats': build_stats,
            'n_solves_this_step': len(step_bisect_history),
            'bisect_history': step_bisect_history,
            'elapsed_total_s': time.time() - t_start,
            'params': params,
        }
        trajectory.append(entry)

        if progress_path:
            snapshot = {
                'd': d, 'order': order, 'target_lb': target_lb,
                'step': step, 'current_lb': cur_lb,
                'final_status': 'running',
                'n_active': n_active, 'n_basis_full': n_basis,
                'elapsed_s': time.time() - t_start,
                'trajectory': trajectory,
            }
            try:
                with open(progress_path, 'w') as f:
                    json.dump(snapshot, f, indent=2, default=str)
            except Exception:
                pass

        if verbose:
            pct = 100.0 * n_active / n_basis
            print(f"\n  [summary s{step}] |B_k|={n_active}/{n_basis} "
                  f"({pct:.1f}%)  lb_k={lb_k:.6f}  "
                  f"bracket_w={hi - lo:.6f}  "
                  f"step_wall={step_time:.1f}s  "
                  f"total_wall={time.time() - t_start:.0f}s", flush=True)

        lb_prev = lb_k

        # Note: we DO NOT dispose the model here — it's reused across
        # sublevel steps.  The Fusion-level PSD swap at the top of the
        # next iteration removes the prior step's moment PSD cone.  A
        # single dispose at the end of the run releases everything.

        # -- Termination checks --
        if target_probe_infeas or lb_k > target_lb + 1e-8:
            final_status = 'target_reached'
            break
        if is_full:
            final_status = 'full_basis_reached'
            break

        # -- Score & add top-N --
        remaining = np.nonzero(pool_mask & ~active_mask)[0]
        if remaining.size == 0:
            final_status = 'pool_exhausted'
            break

        if y_primal_at_feas is None:
            if verbose:
                print("  WARNING: no primal solution captured; "
                      "adding first n_per_step from pool uniformly",
                      flush=True)
            to_add = remaining[:n_per_step]
        else:
            scores = _score_candidates_schur(
                y_primal_at_feas, P, active_mask, remaining)
            n_neg = sum(1 for (_, s) in scores if s < -1e-10)
            if verbose:
                if scores:
                    print(f"  candidates: {len(scores)} total, "
                          f"{n_neg} with Schur < -1e-10; "
                          f"top 3: "
                          f"{[(m, f'{s:+.2e}') for m,s in scores[:3]]}",
                          flush=True)
            to_add = np.array(
                [m for m, _ in scores[:n_per_step]], dtype=np.int64)

        new_mask = active_mask.copy()
        new_mask[to_add] = True
        n_added = int(new_mask.sum() - active_mask.sum())
        if verbose:
            print(f"  growing B_k: +{n_added} monomials "
                  f"(|B_{step+1}| = {int(new_mask.sum())})", flush=True)
        if n_added == 0:
            # shouldn't happen (remaining non-empty), but guard anyway
            final_status = 'no_progress'
            break
        active_mask = new_mask

    # Dispose the shared skeleton model once at the end of the run.
    try:
        mdl.dispose()
    except Exception:
        pass
    gc.collect()

    total_wall = time.time() - t_start
    gap_closure = None
    if val and cur_lb is not None and val > 1.0:
        gap_closure = 100.0 * (cur_lb - 1.0) / (val - 1.0)

    result = {
        'd': d, 'order': order,
        'target_lb': target_lb,
        'final_lb': cur_lb,
        'final_status': final_status,
        'final_active_size': int(active_mask.sum()),
        'n_basis_full': n_basis,
        'n_sublevel_steps': len(trajectory),
        'total_wall_s': total_wall,
        'val_d_known': val,
        'gap_closure_pct': gap_closure,
        'trajectory': trajectory,
        'proof_records': [
            {k: v for k, v in rec.items() if k != 'meta'}
            for rec in proof_records
        ],
    }

    if progress_path:
        snapshot = dict(result, final_status=final_status)
        try:
            with open(progress_path, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
        except Exception:
            pass

    if verbose:
        print(f"\n{'#' * 60}")
        print(f"# SUBLEVEL DONE: status={final_status}")
        print(f"#   final lb          = {cur_lb}")
        print(f"#   target_lb         = {target_lb}")
        print(f"#   final |B_k|       = {int(active_mask.sum())}/{n_basis}")
        print(f"#   sublevel steps    = {len(trajectory)}")
        print(f"#   wall time         = {total_wall:.1f}s")
        if gap_closure is not None:
            print(f"#   gap closure       = {gap_closure:.2f}%")
        print(f"{'#' * 60}", flush=True)

    return result


# =====================================================================
# CLI
# =====================================================================

def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--order', type=int, default=3)
    p.add_argument('--mode',
                    choices=('baseline', 'tuned', 'z2_eq', 'z2_bd',
                              'z2_full'),
                    default='tuned')
    p.add_argument('--sublevel', action='store_true',
                    help='Run the sublevel (Lasserre–Henrion) driver '
                         'instead of the full solver.  See module '
                         'docstring for the relaxation chain.')
    p.add_argument('--target-lb', type=float, default=1.2802,
                    help='Stopping criterion for --sublevel: exit '
                         'as soon as lb_k > target_lb is certified '
                         'with a valid dual certificate.')
    p.add_argument('--add-per-step', type=int, default=50,
                    help='Top-N most-negative-Schur candidates to '
                         'add per sublevel step.')
    p.add_argument('--bisect-per-step', type=int, default=6,
                    help='Bisection halvings per sublevel step.')
    p.add_argument('--time-budget-s', type=float, default=7200.0,
                    help='Wall-clock budget for the sublevel driver.')
    p.add_argument('--z2-pre-elim', action='store_true',
                    help='Losslessly halve n_y via Z/2 time-reversal '
                         'canonicalisation of the moment vector before '
                         'the SDP is built.  At d=16 L3 this cuts '
                         'n_y from 74613 to ~37K, roughly halving the '
                         'per-solve MOSEK cost.  The sublevel scheme '
                         '(basis growth via degree) is orthogonal to '
                         'Z/2 and remains sound.')
    p.add_argument('--progress', type=str, default=None,
                    help='Path to a JSON file receiving an updated '
                         'trajectory snapshot after every sublevel '
                         'step.  Use this for remote tailing.')
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
    p.add_argument('--pre-elim', action='store_true',
                    help='Substitute out Z/2 equalities in Python '
                         'before handing the problem to MOSEK.  '
                         'Losslessly halves n_y and skips MOSEK lindep '
                         'presolve.  Requires a Z/2 mode.')
    p.add_argument('--primary-tol', type=float, default=1e-6,
                    help='Primary MOSEK IPM tolerance.  Default 1e-6.  '
                         'For feasibility probes far from val(d) you '
                         'can safely loosen to 1e-4 (Lever 1).  Retry '
                         'ladder tightens or loosens as needed.')
    p.add_argument('--order-method', type=str, default='forceGraphpar',
                    choices=('free', 'none', 'appminloc',
                              'experimental', 'tryGraphpar',
                              'forceGraphpar'),
                    help='MOSEK AMD ordering method.  Default '
                         'forceGraphpar uses ParMETIS which is '
                         'genuinely parallel (unlike experimental/AMD '
                         'which stalled for 22+ min at d=16 L3).')
    p.add_argument('--watcher-interval', type=float, default=15.0,
                    help='Seconds between watcher status rows.  '
                         'Reduce for short debug runs.')
    p.add_argument('--lindep-off', action='store_true',
                    help='Disable MOSEK lindep presolve (Lever 4).  '
                         'Safe under --pre-elim but often SLOWER on '
                         'small d.  Try at large d if build time '
                         'dominates.')
    args = p.parse_args()

    if args.sublevel:
        r = solve_mosek_sublevel(
            args.d, args.order,
            target_lb=args.target_lb,
            n_per_step=args.add_per_step,
            n_bisect_per_step=args.bisect_per_step,
            t_lo=args.t_lo,
            t_hi=args.t_hi,
            time_budget_s=args.time_budget_s,
            add_upper_loc=not args.no_upper_loc,
            primary_tol=args.primary_tol,
            order_method=args.order_method,
            force_lindep_off=args.lindep_off,
            watcher_interval_s=args.watcher_interval,
            proof_dir=args.proof_dir,
            progress_path=args.progress,
            z2_pre_elim=args.z2_pre_elim,
            verbose=True)
    else:
        r = solve_mosek_tuned(
            args.d, args.order,
            mode=args.mode,
            add_upper_loc=not args.no_upper_loc,
            n_bisect=args.n_bisect,
            t_lo=args.t_lo,
            t_hi=args.t_hi,
            proof_dir=args.proof_dir,
            pre_elim=args.pre_elim,
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
