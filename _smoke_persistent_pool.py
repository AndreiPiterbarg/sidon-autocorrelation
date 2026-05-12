"""Smoke test: persistent multiprocessing.Pool vs fresh-pool-per-call for the
post-filter chain (apply_Q_filter_parallel).

Motivation: post_filters.py creates a fresh `Pool(...)` PER call, which costs
~50-200 ms (Linux fork) or several hundred ms (Windows spawn) to start up plus
the cost of pickling and shipping the per-window setup data each time.  In a
cascade with hundreds of parents per level, this adds up.

This script:
  1. Generates a batch of test compositions at d=6, m=5, picks 200 of them.
  2. Times 10 sequential calls to apply_Q_filter_parallel (current code path).
  3. Times 10 calls reusing a SINGLE persistent Pool with worker globals
     re-broadcast each call (PersistentPool helper below).
  4. Verifies prune masks match exactly (soundness).

Reports speedup as PERSISTENT_POOL_SPEEDUP: Xx (or POOL_OVERHEAD_NEGLIGIBLE).
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched

import post_filters as pf
from post_filters import (
    apply_Q_filter_parallel,
    apply_Q_filter,
    _get_q_setup,
)


# ----------------------------------------------------------------- worker
# Persistent worker globals.  These are set once per worker via initializer
# (cached static data: windows, ell_int_sums, sigmas).  Per-call args travel
# WITH each task tuple (small overhead — three scalars).  No risk of
# desynchronised workers, no need for a separate broadcast phase.
_PW_WINDOWS = None
_PW_ELL = None
_PW_SIGMAS = None
_PW_PRUNE_Q = None


def _persistent_q_init(windows, ell, sigmas):
    """Pool initializer: cache the *static* Q-setup once per worker."""
    global _PW_WINDOWS, _PW_ELL, _PW_SIGMAS, _PW_PRUNE_Q
    _PW_WINDOWS = windows
    _PW_ELL = ell
    _PW_SIGMAS = sigmas
    from _Q_bench import prune_Q_one as _pq
    _PW_PRUNE_Q = _pq


def _persistent_q_check(task):
    """Worker task: (comp, n_half, m, c_target) → True iff Q prunes.

    Per-call args travel with each task — three scalars per composition is
    negligible compared to pickling the composition vector itself.  This
    sidesteps the pool.map non-round-robin pitfall of any broadcast scheme.
    """
    comp, n_half, m, c_target = task
    return bool(_PW_PRUNE_Q(comp, _PW_WINDOWS, _PW_ELL, _PW_SIGMAS,
                              n_half, m, c_target))


# ----------------------------------------------------------------- helper
class PersistentPoolQ:
    """Persistent Pool reused across multiple apply_Q_filter_parallel calls.

    Per-call args (n_half, m, c_target) are broadcast to all workers via a
    pool.map BEFORE the actual work-map.  Static setup (windows, ell, sigmas)
    is cached once per worker via the initializer.

    Usage:
        with PersistentPoolQ(d_child=6, n_workers=8) as ppool:
            survivors_1 = ppool.filter(survivors, n_half=3, m=5, c=1.28)
            survivors_2 = ppool.filter(survivors_2, n_half=4, m=5, c=1.30)
            ...
    """

    def __init__(self, d_child, n_workers=None):
        self.d_child = int(d_child)
        self.n_workers = n_workers or cpu_count()
        self.pool = None

    def __enter__(self):
        windows, ell, sigmas = _get_q_setup(self.d_child)
        self.pool = Pool(processes=self.n_workers,
                         initializer=_persistent_q_init,
                         initargs=(windows, ell, sigmas))
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def filter(self, survivors, n_half_child, m, c_target):
        """Run apply_Q_filter_parallel-equivalent on the persistent pool."""
        n = len(survivors)
        if n == 0:
            return survivors
        nh, mm, ct = int(n_half_child), int(m), float(c_target)
        tasks = [(survivors[i], nh, mm, ct) for i in range(n)]
        chunksize = max(1, n // (self.n_workers * 4))
        results = self.pool.map(_persistent_q_check, tasks,
                                chunksize=chunksize)
        pruned = np.array(results, dtype=bool)
        return survivors[~pruned]


# ----------------------------------------------------------------- bench
def _gen_test_compositions(d, m, want_n=200, seed=42):
    """Generate a deterministic random sample of `want_n` compositions."""
    rng = np.random.default_rng(seed)
    pool = []
    target = want_n * 6
    for batch in generate_compositions_batched(d, m * d, batch_size=10000):
        pool.append(np.asarray(batch, dtype=np.int32))
        if sum(len(b) for b in pool) >= target:
            break
    arr = np.concatenate(pool, axis=0)
    if len(arr) > want_n:
        idx = rng.choice(len(arr), size=want_n, replace=False)
        arr = arr[idx]
    return arr


def main():
    d = 6
    m = 5
    n_half_child = 3   # n=3 ⇒ 4n = 12 = d * m / ... typical small-cascade params
    c_target = 1.28
    n_workers = min(8, cpu_count())
    n_calls = 10
    batch_size = 200

    print(f"== Persistent-pool smoke test ==")
    print(f"  d={d}, m={m}, n_half_child={n_half_child}, c_target={c_target}")
    print(f"  n_workers={n_workers}, n_calls={n_calls}, batch_size={batch_size}")

    survivors = _gen_test_compositions(d, m, want_n=batch_size, seed=42)
    print(f"  generated {len(survivors)} test compositions, dtype={survivors.dtype}")

    # Pre-warm setup cache so timing focuses on Pool.
    _ = _get_q_setup(d)

    # ---------- Scenario 1: Fresh Pool per call (current code path)
    print("\n[Scenario 1] Fresh Pool per call")
    fresh_results = []
    t0 = time.perf_counter()
    for i in range(n_calls):
        out = apply_Q_filter_parallel(survivors, n_half_child, m, c_target,
                                       n_workers=n_workers)
        fresh_results.append(out)
    t_fresh = time.perf_counter() - t0
    print(f"  total wall time: {t_fresh:.3f} s   "
          f"({t_fresh/n_calls*1000:.1f} ms / call)")

    # ---------- Scenario 2: Persistent Pool reused
    print("\n[Scenario 2] Persistent Pool, 10 calls")
    persistent_results = []
    t0 = time.perf_counter()
    with PersistentPoolQ(d_child=d, n_workers=n_workers) as ppool:
        t_setup = time.perf_counter() - t0
        for i in range(n_calls):
            out = ppool.filter(survivors, n_half_child, m, c_target)
            persistent_results.append(out)
    t_persist = time.perf_counter() - t0
    print(f"  total wall time: {t_persist:.3f} s   "
          f"({t_persist/n_calls*1000:.1f} ms / call, "
          f"setup={t_setup*1000:.0f} ms)")

    # ---------- Soundness: prune sets must match exactly
    print("\n[Soundness] Comparing fresh vs persistent prune masks")
    all_ok = True
    for i, (a, b) in enumerate(zip(fresh_results, persistent_results)):
        if a.shape != b.shape:
            print(f"  call {i}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            all_ok = False
            continue
        if not np.array_equal(a, b):
            print(f"  call {i}: ARRAY MISMATCH (sorted-equal? "
                  f"{np.array_equal(np.sort(a, axis=0), np.sort(b, axis=0))})")
            all_ok = False
    if all_ok:
        print(f"  OK — all {n_calls} prune masks identical")
        print(f"     example: kept {len(persistent_results[0])} / {batch_size}")

    # ---------- Verdict
    print("\n[Verdict]")
    if t_fresh > 0:
        speedup = t_fresh / t_persist
    else:
        speedup = 1.0
    overhead_per_call = (t_fresh - t_persist) / n_calls
    print(f"  speedup        : {speedup:.2f}x")
    print(f"  overhead/call  : {overhead_per_call*1000:+.1f} ms saved")

    if speedup >= 1.5:
        print(f"\nPERSISTENT_POOL_SPEEDUP: {speedup:.2f}x")
    elif speedup < 1.1:
        print("\nPOOL_OVERHEAD_NEGLIGIBLE")
    else:
        print(f"\nPERSISTENT_POOL_SPEEDUP: {speedup:.2f}x")


# ----------------------------------------------------------------- API sketch
"""
Sketch — proposed `post_filters.py` API change (if speedup is large):

    @contextlib.contextmanager
    def persistent_pools(d_child, n_workers=None,
                         use_Q=True, use_QN=False, use_L=False):
        '''Open one Pool per active filter for the lifetime of a cascade level
        (or the whole cascade run).  Worker initializers cache static setup
        (windows / sigmas / A_mats); per-call args (n_half, m, c_target)
        travel WITH each task.

        Yields a `Pools` namedtuple with `.q_pool`, `.qn_pool`, `.l_pool`
        attributes (None for filters not enabled).
        '''
        pools = Pools()
        try:
            if use_Q:
                w, e, s = _get_q_setup(d_child)
                pools.q_pool = Pool(n_workers,
                                    initializer=_q_persistent_init,
                                    initargs=(w, e, s))
            if use_QN:
                w, e, s, mw = _get_qn_setup(d_child, n_half_child)  # n_half also static for QN
                pools.qn_pool = Pool(n_workers,
                                     initializer=_qn_persistent_init,
                                     initargs=(w, e, s, mw))
            if use_L:
                w, A = _get_l_setup(d_child)
                solver = _detect_solver()
                pools.l_pool = Pool(n_workers,
                                    initializer=_l_persistent_init,
                                    initargs=(w, A, solver))
            yield pools
        finally:
            for p in (pools.q_pool, pools.qn_pool, pools.l_pool):
                if p is not None:
                    p.close(); p.join()

The per-task workers receive (comp, n_half, m, c_target) tuples (or, for L,
also order=1).  Cascade callers do:

    with persistent_pools(d_child, use_Q=True, use_L=True) as pools:
        for parent in parents:
            survivors = ... # F filter
            survivors = apply_Q_filter_pool(survivors, n_half, m, c, pools.q_pool)
            survivors = apply_L_filter_pool(survivors, n_half, m, c, pools.l_pool)
            ...

Note: QN's `_get_qn_setup` is keyed on (d_child, n_half_child).  If n_half
varies per parent within a level (it does in the cascade), each (n_half_child)
needs its own QN pool — or refactor m_W_arr to travel per-task too (m_W_arr
is small: shape (n_windows,)).  For Q (no n_half dependence) and L (no
n_half dependence either — just windows/A_mats), a single pool covers a
whole level.
"""


if __name__ == "__main__":
    # On Windows, multiprocessing requires the main-guard.
    main()
