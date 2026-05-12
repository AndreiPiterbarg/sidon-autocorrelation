"""Per-composition post-filters for the cascade — F → FN → Q → L chain.

Sequential AND parallel (multiprocessing.Pool) implementations.

Soundness chain: F-survivors ⊇ FN-survivors ⊇ Q-survivors ⊇ L-survivors.
Each filter removes a strict subset; combining them is just chained set
restriction.

Filters:
- F:  variant F (LP-tight Δ_BB linear + #pairs · h² δ²).  Already in
      run_cascade._prune_dynamic with use_F=True.  Numba JIT, fast.
- FN: F's tight LP linear + N's restricted-spectrum δ² bound
      min(op_rest·d, ell_int_sum)·h².  Sound (Σδ=0 ⇒ all-ones component
      annihilated; spectral theorem on M = A_W − α·11ᵀ).  Numba JIT,
      essentially F-cost (op_rest precomputed once per d).  Empirically
      kills 0.6–28.6% extra over F on d ∈ {6,8,10,12} (`_FN_bench.json`).
- Q:  multi-window joint LP per composition (already enumerates ALL
      windows in `_Q_bench._build_windows`).  Sound, decisive at d ≥ 10
      (57-92% additional pruning over F).  Cost ~5-30 ms per LP at d=8-12.
- L:  Lasserre / Shor SDP per-composition cell certificate.  Theoretical
      SDP ceiling.  Sound (`infeasible` Farkas cert only).  Cost ~50-500 ms
      per SDP at d=8-16.

Performance:
- The single-call apply_Q / apply_L functions are O(N) sequential.
- The parallel variants use multiprocessing.Pool with cached worker state
  (windows / sigmas / A_mats built once per worker via initializer), which
  scales near-linearly to cpu_count() workers.
- Heuristic dispatch: parallel is used iff len(survivors) >= PARALLEL_THRESHOLD,
  to avoid Pool startup overhead on tiny batches.
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np

# Bench files live at repo root; prepend to path for import.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------- imports
def _import_Q_utils():
    try:
        from _Q_bench import (_build_windows, _composition_window_data,
                                _enum_balanced_signs, prune_Q_one)
    except Exception as e:
        raise ImportError(
            f"Variant Q requires _Q_bench.py at repo root and scipy."
            f"  Original error: {e}")
    return _build_windows, _composition_window_data, _enum_balanced_signs, prune_Q_one


def _import_L_utils():
    try:
        from _L_bench import (_build_A_matrices, _build_windows as _Lw,
                                _detect_solver, prune_L_one)
    except Exception as e:
        raise ImportError(
            f"Variant L requires _L_bench.py at repo root, cvxpy, and an "
            f"SDP solver (MOSEK preferred, Clarabel fallback).  "
            f"Original error: {e}")
    return _build_A_matrices, _Lw, _detect_solver, prune_L_one


def _import_FN_utils():
    try:
        from _FN_bench import prune_FN
        from _N_bench import precompute_op_norm_restricted
    except Exception as e:
        raise ImportError(
            f"Variant FN requires _FN_bench.py and _N_bench.py at repo "
            f"root.  Original error: {e}")
    return prune_FN, precompute_op_norm_restricted


def _import_QN_utils():
    try:
        from _QN_bench import prune_QN_one, precompute_window_data
    except Exception as e:
        raise ImportError(
            f"Variant QN-fast requires _QN_bench.py at repo root and scipy."
            f"  Original error: {e}")
    return prune_QN_one, precompute_window_data


# Caches (per-process, used by sequential paths).
_Q_CACHE: dict = {}      # d_child -> (windows, ell_int_sums, sigmas)
_L_CACHE: dict = {}      # d_child -> (windows, A_mats)
_FN_CACHE: dict = {}     # d_child -> (ell_prefix, op_rest_d)
_QN_CACHE: dict = {}     # d_child -> (windows, ell_int_sums, sigmas, m_W_arr)


def _get_q_setup(d_child):
    if d_child in _Q_CACHE:
        return _Q_CACHE[d_child]
    _build_windows, _, _enum_signs, _ = _import_Q_utils()
    ws = _build_windows(d_child)
    if isinstance(ws, tuple):
        windows, ell_int_sums = ws[0], ws[1]
    else:
        windows = ws
        ell_int_sums = np.array([w[2] for w in windows], dtype=np.int64)
    sigmas = _enum_signs(d_child)
    _Q_CACHE[d_child] = (windows, ell_int_sums, sigmas)
    return _Q_CACHE[d_child]


def _get_l_setup(d_child):
    if d_child in _L_CACHE:
        return _L_CACHE[d_child]
    _build_A, _build_W, _, _ = _import_L_utils()
    ws = _build_W(d_child)
    windows = ws[0] if isinstance(ws, tuple) else ws
    A_mats = _build_A(d_child, windows)
    _L_CACHE[d_child] = (windows, A_mats)
    return _L_CACHE[d_child]


def _get_qn_setup(d_child, n_half_child):
    """Precompute QN-fast LP setup for d_child.

    Returns (windows, ell_int_sums, sigmas, m_W_arr) where
        m_W_arr[w] = min(op_rest(A_W) · d, n_pairs_W).

    QN-fast is Q's LP with each per-window n_pairs_W replaced by m_W_arr[w].
    Soundness: each m_W is a sound per-window bound on |δᵀA_Wδ| (see
    `_QN_bench.py:80–89`).  QN-fast prunes a SUPERSET of Q's prunes since
    m_W ≤ n_pairs_W per-window.  Empirically validated 0/500 random
    soundness violations at d ∈ {6, 8, 10}.
    """
    key = (d_child, n_half_child)
    if key in _QN_CACHE:
        return _QN_CACHE[key]
    _, precompute_window_data = _import_QN_utils()
    windows, ell_int_sums, sigmas, m_W_arr, _ = precompute_window_data(
        d_child, n_half_child)
    _QN_CACHE[key] = (windows, ell_int_sums, sigmas, m_W_arr)
    return _QN_CACHE[key]


def _get_fn_setup(d_child, n_half_child):
    """Precompute (ell_prefix, op_rest_d) for FN at this d_child / n_half_child.

    op_rest_d[ell, s_lo] := op_rest[ell, s_lo] * d_child   (Σδ=0 spectral
    bound: |δᵀ A_W δ| ≤ op_rest · ‖δ‖₂² ≤ op_rest · d · h²).
    """
    key = (d_child, n_half_child)
    if key in _FN_CACHE:
        return _FN_CACHE[key]
    _, precompute_op = _import_FN_utils()
    conv_len = 2 * d_child - 1
    max_ell = 2 * d_child
    # ell_prefix matches _M1_bench.prune_F construction
    ell_int_arr = np.empty(conv_len, dtype=np.int64)
    two_n = 2 * n_half_child
    for k in range(conv_len):
        d_idx = abs((k + 1) - two_n)
        v = max(0, two_n - d_idx)
        ell_int_arr[k] = v
    ell_prefix = np.zeros(conv_len + 1, dtype=np.int64)
    for k in range(conv_len):
        ell_prefix[k + 1] = ell_prefix[k] + ell_int_arr[k]
    op_rest, _ = precompute_op(d_child, max_ell, conv_len)
    op_rest_d = op_rest * d_child
    _FN_CACHE[key] = (ell_prefix, op_rest_d)
    return _FN_CACHE[key]


# Threshold above which parallelism is worth the Pool overhead.
PARALLEL_THRESHOLD = 64


# ---------------------------------------------------------- sequential
def apply_FN_filter(survivors, n_half_child, m, c_target):
    """FN post-filter: F's tight LP linear + N's restricted-spectrum δ².

    Numba-parallel batch kernel; no need for a multiprocessing Pool.
    Sound: FN-survivors ⊆ F-survivors.  Empirically strict subset at d ≥ 8
    (16 extra prunes / 1014 F-survivors at d=8; 87 / 558 at d=10).

    See `_FN_bench.py` for the soundness derivation (lines 36–53):
      Σδ=0 ⇒ δᵀ(α·11ᵀ)δ = 0  ⇒  δᵀ A_W δ = δᵀ M δ  with M = A_W − α·11ᵀ
      |δᵀ M δ| ≤ ‖M‖_op · ‖δ‖₂² ≤ op_rest · d · h².
    Combined with F's per-window linear bound via per-window min(...).
    """
    if len(survivors) == 0:
        return survivors
    prune_FN, _ = _import_FN_utils()
    d_child = int(survivors.shape[1])
    ell_prefix, op_rest_d = _get_fn_setup(d_child, n_half_child)
    surv = prune_FN(survivors.astype(np.int32, copy=False),
                    n_half_child, m, c_target, ell_prefix, op_rest_d)
    return survivors[surv]


def apply_FN_filter_parallel(survivors, n_half_child, m, c_target,
                             n_workers=None):
    """FN parallel filter — same as `apply_FN_filter` since prune_FN is
    already a numba `@njit(parallel=True)` kernel that saturates cores
    via `prange`.  The `n_workers` arg is accepted for API parity with
    apply_Q_filter_parallel / apply_L_filter_parallel."""
    return apply_FN_filter(survivors, n_half_child, m, c_target)


def apply_F_FN_fused(batch_int, n_half_child, m, c_target):
    """Single-pass F + FN kernel — walks raw batch ONCE and returns
    FN-survivors directly.

    1.86× faster than running F kernel followed by `apply_FN_filter` on
    F-survivors (validated at d=10: 2.4s → 1.2s on 316k compositions,
    0/408k soundness violations).  Use this when you have a RAW
    composition batch (i.e., not already F-filtered) and want FN-survivors.

    Soundness: equivalent to F-prune ∩ FN-prune.  Returns FN-survivors
    (= F-survivors ∩ FN-survivors since FN ⊆ F).
    """
    from f_fn_fused import prune_F_FN_fused
    if len(batch_int) == 0:
        return batch_int
    d_child = int(batch_int.shape[1])
    ell_prefix, op_rest_d = _get_fn_setup(d_child, n_half_child)
    _surv_F, surv_FN = prune_F_FN_fused(
        batch_int.astype(np.int32, copy=False),
        n_half_child, m, c_target, ell_prefix, op_rest_d)
    return batch_int[surv_FN]


def apply_Q_filter(survivors, n_half_child, m, c_target):
    """Sequential Q post-filter."""
    if len(survivors) == 0:
        return survivors
    _, _, _, prune_Q_one = _import_Q_utils()
    d_child = int(survivors.shape[1])
    windows, ell_int_sums, sigmas = _get_q_setup(d_child)
    keep = np.ones(len(survivors), dtype=bool)
    for i in range(len(survivors)):
        if prune_Q_one(survivors[i], windows, ell_int_sums, sigmas,
                        n_half_child, m, c_target):
            keep[i] = False
    return survivors[keep]


def apply_QN_filter(survivors, n_half_child, m, c_target):
    """Sequential QN-fast post-filter.

    QN-fast = Q's multi-window LP with each per-window n_pairs_W replaced
    by m_W := min(op_rest(A_W) · d, n_pairs_W).  Strictly tighter than Q
    (QN-prunes ⊇ Q-prunes ⇒ QN-survivors ⊆ Q-survivors).

    Soundness derivation (`_QN_bench.py:80–89`):
        sum_W (λ_W/ell_W) · |δᵀA_Wδ|
            ≤ sum_W (λ_W/ell_W) · min(op_rest(A_W)·d·h², n_pairs_W·h²)
            = h² · sum_W (λ_W/ell_W) · m_W
    Each per-window bound is sound; the sum with non-negative weights is
    sound.  Margin: 1e-9·m² same as Q's HiGHS guard.

    Empirically: prunes 62 of 240 Q-survivors at d=10 (n=5,m=5,c=1.28),
    25.8% extra reduction, 0/500 random-(δ,λ) violations.
    """
    if len(survivors) == 0:
        return survivors
    prune_QN_one, _ = _import_QN_utils()
    d_child = int(survivors.shape[1])
    windows, ell_int_sums, sigmas, m_W_arr = _get_qn_setup(
        d_child, n_half_child)
    keep = np.ones(len(survivors), dtype=bool)
    for i in range(len(survivors)):
        if prune_QN_one(survivors[i], windows, ell_int_sums, sigmas,
                          n_half_child, m, c_target, m_W_arr):
            keep[i] = False
    return survivors[keep]


def apply_L_filter(survivors, n_half_child, m, c_target, solver='auto'):
    """Sequential L post-filter.  Uses the direct-MOSEK path when MOSEK is
    chosen and available (20.7x faster at d=6 vs CVXPY+MOSEK); falls back
    to CVXPY+MOSEK / Clarabel on any error per-call."""
    if len(survivors) == 0:
        return survivors
    _, _, _detect_solver, prune_L_one = _import_L_utils()
    d_child = int(survivors.shape[1])
    windows, A_mats = _get_l_setup(d_child)
    chosen_solver = _detect_solver(prefer=solver)

    direct_env = None
    direct_prune = None
    if _USE_DIRECT_MOSEK and chosen_solver == 'MOSEK':
        try:
            import mosek
            from l_direct import prune_L_direct
            direct_env = mosek.Env()
            try:
                direct_env.checkoutlicense(mosek.feature.pton)
            except Exception:
                pass
            direct_prune = prune_L_direct
        except Exception:
            direct_env = None
            direct_prune = None

    keep = np.ones(len(survivors), dtype=bool)
    try:
        for i in range(len(survivors)):
            comp = survivors[i]
            if direct_prune is not None:
                try:
                    pruned, _ = direct_prune(
                        comp, A_mats, windows, n_half_child, m, c_target,
                        env=direct_env)
                    if pruned:
                        keep[i] = False
                    continue
                except Exception:
                    pass
            pruned, _ = prune_L_one(
                comp, A_mats, windows, n_half_child, m, c_target,
                solver=chosen_solver, order=1)
            if pruned:
                keep[i] = False
    finally:
        if direct_env is not None:
            try:
                direct_env.__exit__(None, None, None)
            except Exception:
                pass
    return survivors[keep]


# ---------------------------------------------------------- parallel Q
# Module-level worker state (set by Pool initializer; one set per worker process).
_W_Q_WINDOWS = None
_W_Q_ELL = None
_W_Q_SIGMAS = None
_W_NHALF = None
_W_M = None
_W_C = None
_W_PRUNE_Q = None


def _q_worker_init(windows, ell, sigmas, n_half, m, c_target):
    global _W_Q_WINDOWS, _W_Q_ELL, _W_Q_SIGMAS, _W_NHALF, _W_M, _W_C, _W_PRUNE_Q
    _W_Q_WINDOWS = windows
    _W_Q_ELL = ell
    _W_Q_SIGMAS = sigmas
    _W_NHALF = n_half
    _W_M = m
    _W_C = c_target
    # Lazy import inside worker (clean state per process).
    from _Q_bench import prune_Q_one as _pq
    _W_PRUNE_Q = _pq


def _q_worker_check(comp):
    """Worker: True iff Q prunes."""
    return bool(_W_PRUNE_Q(comp, _W_Q_WINDOWS, _W_Q_ELL, _W_Q_SIGMAS,
                              _W_NHALF, _W_M, _W_C))


def apply_Q_filter_parallel(survivors, n_half_child, m, c_target,
                              n_workers=None):
    """Parallel Q post-filter using multiprocessing.Pool.
    Falls back to sequential for small batches."""
    n = len(survivors)
    if n == 0:
        return survivors
    if n_workers is None:
        n_workers = cpu_count()
    if n < PARALLEL_THRESHOLD or n_workers <= 1:
        return apply_Q_filter(survivors, n_half_child, m, c_target)

    d_child = int(survivors.shape[1])
    windows, ell_int_sums, sigmas = _get_q_setup(d_child)
    chunksize = max(1, n // (n_workers * 4))

    with Pool(processes=n_workers,
                initializer=_q_worker_init,
                initargs=(windows, ell_int_sums, sigmas,
                          n_half_child, m, c_target)) as pool:
        results = pool.map(_q_worker_check, list(survivors), chunksize=chunksize)

    pruned = np.array(results, dtype=bool)
    return survivors[~pruned]


# ---------------------------------------------------------- parallel QN
_W_QN_WINDOWS = None
_W_QN_ELL = None
_W_QN_SIGMAS = None
_W_QN_MW = None
_W_PRUNE_QN = None


def _qn_worker_init(windows, ell, sigmas, m_W_arr, n_half, m, c_target):
    global _W_QN_WINDOWS, _W_QN_ELL, _W_QN_SIGMAS, _W_QN_MW
    global _W_NHALF, _W_M, _W_C, _W_PRUNE_QN
    _W_QN_WINDOWS = windows
    _W_QN_ELL = ell
    _W_QN_SIGMAS = sigmas
    _W_QN_MW = m_W_arr
    _W_NHALF = n_half
    _W_M = m
    _W_C = c_target
    from _QN_bench import prune_QN_one as _pqn
    _W_PRUNE_QN = _pqn


def _qn_worker_check(comp):
    """Worker: True iff QN-fast prunes."""
    return bool(_W_PRUNE_QN(comp, _W_QN_WINDOWS, _W_QN_ELL, _W_QN_SIGMAS,
                              _W_NHALF, _W_M, _W_C, _W_QN_MW))


def apply_QN_filter_parallel(survivors, n_half_child, m, c_target,
                               n_workers=None):
    """Parallel QN-fast post-filter using multiprocessing.Pool.
    Falls back to sequential for small batches."""
    n = len(survivors)
    if n == 0:
        return survivors
    if n_workers is None:
        n_workers = cpu_count()
    if n < PARALLEL_THRESHOLD or n_workers <= 1:
        return apply_QN_filter(survivors, n_half_child, m, c_target)

    d_child = int(survivors.shape[1])
    windows, ell_int_sums, sigmas, m_W_arr = _get_qn_setup(
        d_child, n_half_child)
    chunksize = max(1, n // (n_workers * 4))

    with Pool(processes=n_workers,
                initializer=_qn_worker_init,
                initargs=(windows, ell_int_sums, sigmas, m_W_arr,
                          n_half_child, m, c_target)) as pool:
        results = pool.map(_qn_worker_check, list(survivors),
                            chunksize=chunksize)

    pruned = np.array(results, dtype=bool)
    return survivors[~pruned]


# ---------------------------------------------------------- parallel L
_W_L_WINDOWS = None
_W_L_AMATS = None
_W_L_SOLVER = None
_W_PRUNE_L = None
_W_L_ENV = None      # mosek.Env reused across calls in a worker (direct path)
_W_L_PRUNE_DIRECT = None  # prune_L_direct reference (direct MOSEK path)

# Toggle: prefer the direct-MOSEK path (20.7x faster at d=6, sound).
# Set L_USE_DIRECT_MOSEK=0 in env to fall back to CVXPY+MOSEK.
_USE_DIRECT_MOSEK = os.environ.get('L_USE_DIRECT_MOSEK', '1') != '0'


def _l_worker_init(windows, A_mats, solver, n_half, m, c_target):
    global _W_L_WINDOWS, _W_L_AMATS, _W_L_SOLVER, _W_NHALF, _W_M, _W_C, _W_PRUNE_L
    global _W_L_ENV, _W_L_PRUNE_DIRECT
    _W_L_WINDOWS = windows
    _W_L_AMATS = A_mats
    _W_L_SOLVER = solver
    _W_NHALF = n_half
    _W_M = m
    _W_C = c_target
    # Single-thread MOSEK in workers (so they don't fight over cores).
    os.environ.setdefault('MSK_IPAR_NUM_THREADS', '1')
    _mlf = os.environ.get('MOSEKLM_LICENSE_FILE')
    if not _mlf:
        for _cand in ('/home/ubuntu/mosek/mosek.lic',
                      os.path.expanduser('~/mosek/mosek.lic'),
                      'C:/mosek/mosek.lic'):
            if os.path.exists(_cand):
                os.environ['MOSEKLM_LICENSE_FILE'] = _cand
                break
    if _USE_DIRECT_MOSEK and solver == 'MOSEK':
        try:
            import mosek
            from l_direct import prune_L_direct
            env = mosek.Env()
            try:
                env.checkoutlicense(mosek.feature.pton)
            except Exception:
                pass
            _W_L_ENV = env
            _W_L_PRUNE_DIRECT = prune_L_direct
        except Exception:
            _W_L_ENV = None
            _W_L_PRUNE_DIRECT = None
    from _L_bench import prune_L_one as _pl
    _W_PRUNE_L = _pl


def _l_worker_check(comp):
    """Worker: True iff L prunes.  Uses direct MOSEK path if available
    (20.7x speedup at d=6 vs CVXPY); falls back to cvxpy on any error."""
    if _W_L_ENV is not None and _W_L_PRUNE_DIRECT is not None:
        try:
            pruned, _status = _W_L_PRUNE_DIRECT(
                comp, _W_L_AMATS, _W_L_WINDOWS,
                _W_NHALF, _W_M, _W_C, env=_W_L_ENV)
            return bool(pruned)
        except Exception:
            pass
    pruned, _status = _W_PRUNE_L(comp, _W_L_AMATS, _W_L_WINDOWS,
                                  _W_NHALF, _W_M, _W_C,
                                  solver=_W_L_SOLVER, order=1)
    return bool(pruned)


def apply_L_filter_parallel(survivors, n_half_child, m, c_target,
                              solver='auto', n_workers=None):
    """Parallel L post-filter."""
    n = len(survivors)
    if n == 0:
        return survivors
    if n_workers is None:
        n_workers = cpu_count()
    if n < max(8, PARALLEL_THRESHOLD // 4) or n_workers <= 1:
        return apply_L_filter(survivors, n_half_child, m, c_target, solver=solver)

    d_child = int(survivors.shape[1])
    windows, A_mats = _get_l_setup(d_child)
    _, _, _detect_solver, _ = _import_L_utils()
    chosen_solver = _detect_solver(prefer=solver)
    # L SDPs are slower → smaller chunks, more workers active
    chunksize = max(1, n // (n_workers * 8))

    with Pool(processes=n_workers,
                initializer=_l_worker_init,
                initargs=(windows, A_mats, chosen_solver,
                          n_half_child, m, c_target)) as pool:
        results = pool.map(_l_worker_check, list(survivors), chunksize=chunksize)

    pruned = np.array(results, dtype=bool)
    return survivors[~pruned]


# ---------------------------------------------------------- parallel SP (split-cell)
# Split-cell SDP filter: for each L-survivor, split the parent cell into
# 2^d sub-cells (each x_i either in [c_i-1, c_i] or [c_i, c_i+1]); if EVERY
# sub-cell SDP is `infeasible`, the parent is split-pruned.  Sound:
# sub-cells are tighter relaxations, Farkas certificates compose.
#
# Optimizations vs the smoke baseline (`_smoke_split_cell_SDP.py`, 165 s/cell):
#   (a) Direct MOSEK Task API via `l_direct.prune_L_direct(lo_override, hi_override)`;
#       bypasses 84% CVXPY canonicalization overhead.
#   (b) Reused `mosek.Env` per worker.
#   (c) Smart sigma ordering — try sub-cells whose box-center mean-sum is
#       closest to 4nm FIRST (most likely feasible).  When parent is NOT
#       split-prunable we exit on the first feasible sub-cell.
#   (d) Box-sum pre-screen: Σlo > 4nm or Σhi < 4nm ⇒ infeasible (free).
#
# Validated at d=10 (n=5,m=5,c=1.28): 9/14 split-pruned in 483 s total
# (4.8x faster than baseline; +2 prunes from MOSEK robustness).  Smart
# ordering reduces un-prunable cells from 165s to 4-12s (35x faster).
#
# Should ONLY be used at d ≤ 10 in routine pipelines: cost is 2^d sub-cells
# per L-survivor; at d=12 with direct MOSEK it's ~9 min/cell, at d=14 ~2 hr.
_W_SP_AMATS = None
_W_SP_WINDOWS = None
_W_SP_ENV = None
_W_SP_PRUNE_DIRECT = None


def _sp_worker_init(windows, A_mats):
    global _W_SP_AMATS, _W_SP_WINDOWS, _W_SP_ENV, _W_SP_PRUNE_DIRECT
    _W_SP_AMATS = A_mats
    _W_SP_WINDOWS = windows
    os.environ.setdefault('MSK_IPAR_NUM_THREADS', '1')
    _mlf = os.environ.get('MOSEKLM_LICENSE_FILE')
    if not _mlf:
        for _cand in ('/home/ubuntu/mosek/mosek.lic',
                      os.path.expanduser('~/mosek/mosek.lic'),
                      'C:/mosek/mosek.lic'):
            if os.path.exists(_cand):
                os.environ['MOSEKLM_LICENSE_FILE'] = _cand
                break
    try:
        import mosek
        from l_direct import prune_L_direct
        env = mosek.Env()
        try:
            env.checkoutlicense(mosek.feature.pton)
        except Exception:
            pass
        _W_SP_ENV = env
        _W_SP_PRUNE_DIRECT = prune_L_direct
    except Exception:
        _W_SP_ENV = None
        _W_SP_PRUNE_DIRECT = None


def _sp_worker_check(task):
    """Worker: test one sub-cell SDP.

    Args (tuple): (sigma_idx, c_int_list, sigma_list, n_half, m, c_target).
    Returns: (sigma_idx, sub_pruned: bool, status: str, t: float).
    """
    sigma_idx, c_int_list, sigma_list, n_half, m, c_target = task
    c_int = np.asarray(c_int_list, dtype=np.int32)
    sigma = np.asarray(sigma_list, dtype=np.int8)
    d = len(c_int)
    nm = float(4 * n_half * m)
    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for i in range(d):
        ci = float(c_int[i])
        if sigma[i] > 0:
            lo[i] = max(0.0, ci - 1.0); hi[i] = ci
        else:
            lo[i] = ci; hi[i] = ci + 1.0
    if (np.sum(lo) > nm + 1e-9) or (np.sum(hi) < nm - 1e-9):
        return sigma_idx, True, 'box_sum_pre_infeasible', 0.0
    if _W_SP_PRUNE_DIRECT is None or _W_SP_ENV is None:
        return sigma_idx, False, 'no_direct_mosek', 0.0
    import time as _time
    t0 = _time.time()
    try:
        pruned, status = _W_SP_PRUNE_DIRECT(
            c_int, _W_SP_AMATS, _W_SP_WINDOWS, n_half, m, c_target,
            env=_W_SP_ENV, lo_override=lo, hi_override=hi)
    except Exception as e:
        return sigma_idx, False, f'EXC:{type(e).__name__}', _time.time() - t0
    return sigma_idx, bool(pruned), str(status), _time.time() - t0


def _sp_smart_sigma_order(c_int, n_half, m):
    """Order sigmas by box-center mean-sum closeness to 4nm (likely feasible first)."""
    from itertools import product
    d = int(len(c_int))
    s_target = 4 * n_half * m
    cs = c_int.tolist() if hasattr(c_int, 'tolist') else list(c_int)
    sigmas = list(product([1, -1], repeat=d))
    def slack(sigma):
        ms = 0.0
        for i in range(d):
            ms += (cs[i] - 0.5) if sigma[i] > 0 else (cs[i] + 0.5)
        return abs(ms - s_target)
    sigmas.sort(key=slack)
    return sigmas


def _recursive_subsplit_check(c_int, lo, hi, n_half, m, c_target, depth, max_depth):
    """Recursively prove that no x in [lo, hi] satisfies max_W TV_W(x/m) ≤ c_target.

    Sound: at every leaf, MOSEK Farkas-cert (`prim_infeas_cer`) is required to
    declare the leaf infeasible.  Internal nodes (status `optimal` at depth
    < max_depth) recurse into 2^d sub-boxes (each coord halved).  At max_depth
    a leaf with `optimal` is treated as FEASIBLE (returns False).

    Worker-side helper: uses module-level `_W_SP_*` worker globals.
    """
    nm = float(4 * n_half * m)
    if (np.sum(lo) > nm + 1e-9) or (np.sum(hi) < nm - 1e-9):
        return True
    if _W_SP_PRUNE_DIRECT is None or _W_SP_ENV is None:
        return False
    try:
        pruned, status = _W_SP_PRUNE_DIRECT(
            c_int, _W_SP_AMATS, _W_SP_WINDOWS, n_half, m, c_target,
            env=_W_SP_ENV, lo_override=lo, hi_override=hi)
    except Exception:
        return False
    if pruned:
        return True
    if status != 'optimal' or depth >= max_depth:
        return False

    from itertools import product
    d = len(c_int)
    mid = 0.5 * (lo + hi)
    for sub in product([0, 1], repeat=d):
        sub_lo = np.empty(d, dtype=np.float64)
        sub_hi = np.empty(d, dtype=np.float64)
        for i in range(d):
            if sub[i] == 0:
                sub_lo[i] = lo[i]; sub_hi[i] = mid[i]
            else:
                sub_lo[i] = mid[i]; sub_hi[i] = hi[i]
        if not _recursive_subsplit_check(c_int, sub_lo, sub_hi,
                                           n_half, m, c_target,
                                           depth + 1, max_depth):
            return False
    return True


def _sp_worker_check_recursive(task):
    """Worker variant: when sub-cell SDP returns `optimal`, recurse to
    `max_depth` levels (each level halves the box in every coord).  Sound
    union-of-Farkas-certs at every leaf.

    Args (tuple): (sigma_idx, c_int_list, sigma_list, n_half, m, c_target,
                    max_depth).
    """
    sigma_idx, c_int_list, sigma_list, n_half, m, c_target, max_depth = task
    c_int = np.asarray(c_int_list, dtype=np.int32)
    sigma = np.asarray(sigma_list, dtype=np.int8)
    d = len(c_int)
    nm = float(4 * n_half * m)
    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for i in range(d):
        ci = float(c_int[i])
        if sigma[i] > 0:
            lo[i] = max(0.0, ci - 1.0); hi[i] = ci
        else:
            lo[i] = ci; hi[i] = ci + 1.0
    if (np.sum(lo) > nm + 1e-9) or (np.sum(hi) < nm - 1e-9):
        return sigma_idx, True, 'box_sum_pre_infeasible', 0.0
    import time as _time
    t0 = _time.time()
    try:
        pruned = _recursive_subsplit_check(
            c_int, lo, hi, n_half, m, c_target,
            depth=1, max_depth=max_depth)
    except Exception as e:
        return sigma_idx, False, f'EXC:{type(e).__name__}', _time.time() - t0
    status = 'infeasible' if pruned else 'optimal_at_depth'
    return sigma_idx, bool(pruned), status, _time.time() - t0


# Maximum d for split-cell to be tractable in routine pipelines.
SPLIT_CELL_MAX_D = 10


def apply_split_cell_filter_parallel(survivors, n_half_child, m, c_target,
                                       n_workers=None, max_d=SPLIT_CELL_MAX_D,
                                       early_terminate=True, verbose=False,
                                       max_depth=1):
    """Split-cell SDP filter on L-survivors.

    For each survivor, splits the cell into 2^d sub-cells (binary in each
    coord) and runs the Shor SDP via direct MOSEK on each.  If EVERY sub-cell
    is `infeasible`, the parent is split-pruned.

    Returns survivors that were NOT split-pruned (those with at least one
    feasible sub-cell).

    `max_depth` (default 1): when a sub-cell SDP returns `optimal`, recurse
    into 2^d sub-sub-cells (each coord halved again), repeat to `max_depth`.
    At max_depth, an `optimal` leaf means the cell is genuinely (relaxation-)
    feasible — parent NOT split-prunable.  Cost grows as (2^d)^max_depth in
    the worst case, but in practice only the few "stuck" sub-cells recurse.

    Soundness: each sub-cell SDP is a tighter relaxation than the parent;
    `infeasible` is Farkas-certified.  Union of certificates ⇒ parent
    infeasible (at any depth).  See `l_direct.prune_L_direct` for the SDP
    encoding (now augmented with trace-identity + Cauchy-Schwarz cuts).

    Empirically at d=10 (n=5,m=5,c=1.28) with 12 workers: ~20s per
    split-prunable cell, ~5s per un-prunable cell (smart-order early exit).

    Gated to d ≤ `max_d` (default 10) — cost is 2^d sub-cells per parent.
    Returns input unchanged when d > max_d.
    """
    if len(survivors) == 0:
        return survivors
    d_child = int(survivors.shape[1])
    if d_child > max_d:
        return survivors
    if n_workers is None:
        n_workers = cpu_count()

    windows, A_mats = _get_l_setup(d_child)

    use_recursive = max_depth > 1
    worker_fn = _sp_worker_check_recursive if use_recursive else _sp_worker_check

    keep_mask = np.ones(len(survivors), dtype=bool)
    with Pool(processes=min(n_workers, max(1, 2 ** d_child)),
                initializer=_sp_worker_init,
                initargs=(windows, A_mats)) as pool:
        for idx, c_int in enumerate(survivors):
            sigmas = _sp_smart_sigma_order(c_int, n_half_child, m)
            if use_recursive:
                tasks = [(si, c_int.tolist(), list(sigma),
                          int(n_half_child), int(m), float(c_target),
                          int(max_depth))
                          for si, sigma in enumerate(sigmas)]
            else:
                tasks = [(si, c_int.tolist(), list(sigma),
                          int(n_half_child), int(m), float(c_target))
                          for si, sigma in enumerate(sigmas)]
            n_inf = 0
            feasible = False
            n_total = len(tasks)
            chunksize = max(1, n_total // (n_workers * 4))
            try:
                for sigma_idx, pruned, status, t in pool.imap_unordered(
                        worker_fn, tasks, chunksize=chunksize):
                    if pruned:
                        n_inf += 1
                    else:
                        feasible = True
                        if early_terminate:
                            break
            except Exception as e:
                if verbose:
                    print(f"   SP filter exception on cell {idx}: {e}")
                continue
            if not feasible:
                keep_mask[idx] = False
            if verbose:
                print(f"   SP[{idx+1}/{len(survivors)}]: "
                      f"{'PRUNED' if not feasible else 'survived'} "
                      f"({n_inf} sub-cells inf)")

    return survivors[keep_mask]


# ============================================================================
# Persistent Pool context manager (5.14x speedup for repeated filter calls).
#
# When the cascade dispatches Q/QN/L filters hundreds of times per level,
# `with Pool(...)` per call burns ~50–200 ms each on Pool spawn + initializer
# data pickle.  PersistentPools opens one Pool per active filter and reuses
# it; static state (windows / sigmas / A_mats / m_W_arr / mosek.Env) is set
# once via the initializer; per-call (n_half, m, c_target) travels with each
# task tuple — three scalars per composition is negligible vs. the comp
# vector itself.  Validated 0/N soundness violations on 200 comps × 10 calls.
# ============================================================================
_PW_Q_WINDOWS = None
_PW_Q_ELL = None
_PW_Q_SIGMAS = None
_PW_Q_PRUNE = None


def _pw_q_init(windows, ell, sigmas):
    global _PW_Q_WINDOWS, _PW_Q_ELL, _PW_Q_SIGMAS, _PW_Q_PRUNE
    _PW_Q_WINDOWS = windows
    _PW_Q_ELL = ell
    _PW_Q_SIGMAS = sigmas
    from _Q_bench import prune_Q_one as _pq
    _PW_Q_PRUNE = _pq


def _pw_q_check(task):
    comp, n_half, m, c_target = task
    return bool(_PW_Q_PRUNE(comp, _PW_Q_WINDOWS, _PW_Q_ELL, _PW_Q_SIGMAS,
                              n_half, m, c_target))


_PW_QN_WINDOWS = None
_PW_QN_ELL = None
_PW_QN_SIGMAS = None
_PW_QN_MW = None
_PW_QN_PRUNE = None


def _pw_qn_init(windows, ell, sigmas, m_W_arr):
    global _PW_QN_WINDOWS, _PW_QN_ELL, _PW_QN_SIGMAS, _PW_QN_MW, _PW_QN_PRUNE
    _PW_QN_WINDOWS = windows
    _PW_QN_ELL = ell
    _PW_QN_SIGMAS = sigmas
    _PW_QN_MW = m_W_arr
    from _QN_bench import prune_QN_one as _pqn
    _PW_QN_PRUNE = _pqn


def _pw_qn_check(task):
    comp, n_half, m, c_target = task
    return bool(_PW_QN_PRUNE(comp, _PW_QN_WINDOWS, _PW_QN_ELL, _PW_QN_SIGMAS,
                                n_half, m, c_target, _PW_QN_MW))


_PW_L_WINDOWS = None
_PW_L_AMATS = None
_PW_L_SOLVER = None
_PW_L_ENV = None
_PW_L_PRUNE_DIRECT = None
_PW_L_PRUNE_CVXPY = None


def _pw_l_init(windows, A_mats, solver):
    global _PW_L_WINDOWS, _PW_L_AMATS, _PW_L_SOLVER
    global _PW_L_ENV, _PW_L_PRUNE_DIRECT, _PW_L_PRUNE_CVXPY
    _PW_L_WINDOWS = windows
    _PW_L_AMATS = A_mats
    _PW_L_SOLVER = solver
    os.environ.setdefault('MSK_IPAR_NUM_THREADS', '1')
    os.environ.setdefault('MOSEKLM_LICENSE_FILE',
                            os.environ.get('MOSEKLM_LICENSE_FILE',
                                            '/home/ubuntu/mosek/mosek.lic'))
    if _USE_DIRECT_MOSEK and solver == 'MOSEK':
        try:
            import mosek
            from l_direct import prune_L_direct
            env = mosek.Env()
            try:
                env.checkoutlicense(mosek.feature.pton)
            except Exception:
                pass
            _PW_L_ENV = env
            _PW_L_PRUNE_DIRECT = prune_L_direct
        except Exception:
            _PW_L_ENV = None
            _PW_L_PRUNE_DIRECT = None
    from _L_bench import prune_L_one as _pl
    _PW_L_PRUNE_CVXPY = _pl


def _pw_l_check(task):
    comp, n_half, m, c_target = task
    if _PW_L_ENV is not None and _PW_L_PRUNE_DIRECT is not None:
        try:
            pruned, _ = _PW_L_PRUNE_DIRECT(
                comp, _PW_L_AMATS, _PW_L_WINDOWS,
                n_half, m, c_target, env=_PW_L_ENV)
            return bool(pruned)
        except Exception:
            pass
    pruned, _ = _PW_L_PRUNE_CVXPY(
        comp, _PW_L_AMATS, _PW_L_WINDOWS,
        n_half, m, c_target, solver=_PW_L_SOLVER, order=1)
    return bool(pruned)


class PersistentPools:
    """Context manager holding one persistent Pool per active filter (Q/QN/L).

    FN doesn't need a Pool — its kernel is already numba-prange parallel.

    Usage:
        with PersistentPools(d_child=10, n_half_child=5, use_Q=True,
                              use_QN=True, use_L=True, solver='MOSEK',
                              n_workers=64) as pp:
            for parent in parents:
                # ... process_parent_fused → surv_F ...
                surv_FN = pp.apply_FN(surv_F, n_half, m, c_target)
                surv_Q  = pp.apply_Q(surv_FN, n_half, m, c_target)
                surv_QN = pp.apply_QN(surv_Q, n_half, m, c_target)
                surv_L  = pp.apply_L(surv_QN, n_half, m, c_target)
    """
    def __init__(self, d_child, n_half_child, n_workers=None,
                  use_Q=True, use_QN=True, use_L=True, solver='MOSEK'):
        self.d_child = int(d_child)
        self.n_half_child = int(n_half_child)
        self.n_workers = n_workers or cpu_count()
        self.use_Q = use_Q
        self.use_QN = use_QN
        self.use_L = use_L
        self.solver = solver
        self._pool_q = None
        self._pool_qn = None
        self._pool_l = None
        self._chosen_solver = None

    def __enter__(self):
        if self.use_Q:
            windows, ell_int_sums, sigmas = _get_q_setup(self.d_child)
            self._pool_q = Pool(processes=self.n_workers,
                                  initializer=_pw_q_init,
                                  initargs=(windows, ell_int_sums, sigmas))
        if self.use_QN:
            windows, ell_int_sums, sigmas, m_W_arr = _get_qn_setup(
                self.d_child, self.n_half_child)
            self._pool_qn = Pool(processes=self.n_workers,
                                   initializer=_pw_qn_init,
                                   initargs=(windows, ell_int_sums, sigmas, m_W_arr))
        if self.use_L:
            windows, A_mats = _get_l_setup(self.d_child)
            _, _, _detect_solver, _ = _import_L_utils()
            self._chosen_solver = _detect_solver(prefer=self.solver)
            self._pool_l = Pool(processes=self.n_workers,
                                  initializer=_pw_l_init,
                                  initargs=(windows, A_mats, self._chosen_solver))
        return self

    def __exit__(self, *exc):
        for p in (self._pool_q, self._pool_qn, self._pool_l):
            if p is not None:
                try:
                    p.close(); p.join()
                except Exception:
                    pass
        self._pool_q = self._pool_qn = self._pool_l = None

    def apply_FN(self, survivors, n_half_child, m, c_target):
        return apply_FN_filter(survivors, n_half_child, m, c_target)

    def _run(self, pool, check_fn, survivors, n_half, m, c_target):
        n = len(survivors)
        if n == 0:
            return survivors
        nh, mm, ct = int(n_half), int(m), float(c_target)
        tasks = [(survivors[i], nh, mm, ct) for i in range(n)]
        chunksize = max(1, n // (self.n_workers * 4))
        results = pool.map(check_fn, tasks, chunksize=chunksize)
        pruned = np.array(results, dtype=bool)
        return survivors[~pruned]

    def apply_Q(self, survivors, n_half_child, m, c_target):
        if self._pool_q is None:
            return apply_Q_filter_parallel(
                survivors, n_half_child, m, c_target,
                n_workers=self.n_workers)
        return self._run(self._pool_q, _pw_q_check,
                          survivors, n_half_child, m, c_target)

    def apply_QN(self, survivors, n_half_child, m, c_target):
        if self._pool_qn is None:
            return apply_QN_filter_parallel(
                survivors, n_half_child, m, c_target,
                n_workers=self.n_workers)
        return self._run(self._pool_qn, _pw_qn_check,
                          survivors, n_half_child, m, c_target)

    def apply_L(self, survivors, n_half_child, m, c_target):
        if self._pool_l is None:
            return apply_L_filter_parallel(
                survivors, n_half_child, m, c_target,
                solver=self.solver, n_workers=self.n_workers)
        n = len(survivors)
        if n < max(8, PARALLEL_THRESHOLD // 4):
            return apply_L_filter(survivors, n_half_child, m, c_target,
                                    solver=self.solver)
        return self._run(self._pool_l, _pw_l_check,
                          survivors, n_half_child, m, c_target)


# ---------------------------------------------------------- chain
def apply_post_filter_chain(survivors, n_half_child, m, c_target,
                              use_FN: bool = False,
                              use_Q: bool = False,
                              use_QN: bool = False,
                              use_L: bool = False,
                              use_SP: bool = False,
                              sp_max_d: int = SPLIT_CELL_MAX_D,
                              parallel: bool = True, n_workers=None):
    """F-survivors -> apply FN, Q, QN, L, SP (split-cell) post-filters in order.

    Order is FN → Q → QN-fast → L → SP.  Each is a (weak) subset of the prior;
    SP gated to d ≤ sp_max_d (default 10) — cost is 2^d sub-cells per cell.
    With parallel=True (default), Q/QN/L/SP use multiprocessing.Pool;
    FN uses numba prange."""
    if not (use_FN or use_Q or use_QN or use_L or use_SP):
        return survivors
    if len(survivors) == 0:
        return survivors

    if use_FN:
        survivors = apply_FN_filter(survivors, n_half_child, m, c_target)
    if not (use_Q or use_QN or use_L or use_SP):
        return survivors

    Q_fn = apply_Q_filter_parallel if parallel else apply_Q_filter
    QN_fn = apply_QN_filter_parallel if parallel else apply_QN_filter
    L_fn = apply_L_filter_parallel if parallel else apply_L_filter

    if use_Q and len(survivors) > 0:
        survivors = Q_fn(survivors, n_half_child, m, c_target,
                          n_workers=n_workers) if parallel else Q_fn(
                          survivors, n_half_child, m, c_target)
    if use_QN and len(survivors) > 0:
        survivors = QN_fn(survivors, n_half_child, m, c_target,
                          n_workers=n_workers) if parallel else QN_fn(
                          survivors, n_half_child, m, c_target)
    if use_L and len(survivors) > 0:
        if parallel:
            survivors = L_fn(survivors, n_half_child, m, c_target,
                              n_workers=n_workers)
        else:
            survivors = L_fn(survivors, n_half_child, m, c_target)
    if use_SP and len(survivors) > 0:
        survivors = apply_split_cell_filter_parallel(
            survivors, n_half_child, m, c_target,
            n_workers=n_workers, max_d=sp_max_d)
    return survivors
