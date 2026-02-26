"""CPU-only cascade prover — no GPU, no dimension limits.

Runs L0 (composition generation + pruning) then cascades through
refinement levels until all survivors are eliminated or a max
dimension is reached.

Usage:
    python -m cloninger-steinerberger.cpu.run_cascade
    python -m cloninger-steinerberger.cpu.run_cascade --n_half 2 --m 20 --c_target 1.30
    python -m cloninger-steinerberger.cpu.run_cascade --n_half 3 --m 50 --c_target 1.30 --max_levels 5
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import itertools

import numpy as np
import numba
from numba import njit, prange

# Path setup — import from archive/cpu
_this_dir = os.path.dirname(os.path.abspath(__file__))
_cs_dir = os.path.dirname(_this_dir)
_archive_cpu = os.path.join(_cs_dir, 'archive', 'cpu')
sys.path.insert(0, _archive_cpu)
sys.path.insert(0, _cs_dir)

from compositions import (generate_canonical_compositions_batched,
                         generate_compositions_batched)
from pruning import (correction, asymmetry_threshold, count_compositions,
                     asymmetry_prune_mask, _canonical_mask)
from test_values import compute_test_values_batch


# =====================================================================
# Dynamic per-window threshold (matches GPU integer-space logic exactly)
# =====================================================================

@njit(parallel=True, cache=True)
def _prune_dynamic(batch_int, n_half, m, c_target):
    """Per-window dynamic threshold matching GPU refinement kernel exactly.

    Works in integer convolution space.  For each window (ell, s_lo),
    computes W_int and the dynamic integer threshold:
        dyn_it = floor((dyn_base + 2*W_int) * ell / (4*n) * (1 - 4*eps))
    where dyn_base = c_target*m^2 + 1 + 1e-9*m^2.

    Returns boolean mask: True = survived (not pruned).
    """
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    m_d = np.float64(m)
    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n = 1.0 / (4.0 * np.float64(n_half))
    DBL_EPS = 2.220446049250313e-16

    for b in prange(B):
        # Integer autoconvolution (same as GPU: conv[k] = sum_{i+j=k} c_i*c_j)
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            conv[2 * i] += ci * ci
            for j in range(i + 1, d):
                conv[i + j] += np.int64(2) * ci * np.int64(batch_int[b, j])
        # Prefix sum
        for k in range(1, conv_len):
            conv[k] += conv[k - 1]

        # Prefix sum of c (for W_int)
        prefix_c = np.zeros(d + 1, dtype=np.int64)
        for i in range(d):
            prefix_c[i + 1] = prefix_c[i] + np.int64(batch_int[b, i])

        # Window scan with per-window dynamic thresholds
        pruned = False
        for ell in range(2, 2 * d + 1):
            if pruned:
                break
            n_cv = ell - 1
            dyn_base_ell = dyn_base * np.float64(ell) * inv_4n
            two_ell_inv_4n = 2.0 * np.float64(ell) * inv_4n
            n_windows = conv_len - n_cv + 1
            for s_lo in range(n_windows):
                s_hi = s_lo + n_cv - 1
                ws = conv[s_hi]
                if s_lo > 0:
                    ws -= conv[s_lo - 1]
                # W_int: sum of c in support bins [lo_bin, hi_bin]
                lo_bin = s_lo - (d - 1)
                if lo_bin < 0:
                    lo_bin = 0
                hi_bin = s_lo + ell - 2
                if hi_bin > d - 1:
                    hi_bin = d - 1
                W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                dyn_x = dyn_base_ell + two_ell_inv_4n * np.float64(W_int)
                dyn_it = np.int64(dyn_x * (1.0 - 4.0 * DBL_EPS))
                if ws > dyn_it:
                    pruned = True
                    break

        if pruned:
            survived[b] = False

    return survived


# =====================================================================
# Level 0: generate all compositions, prune, collect survivors
# =====================================================================

def run_level0(n_half, m, c_target, verbose=True):
    """Run Level 0: enumerate compositions, prune, collect survivors.

    Matches GPU semantics exactly:
      - Canonical palindrome filter (only c <= rev(c))
      - Dynamic per-window threshold (integer-space, W_int-dependent)

    Returns
    -------
    dict with: survivors (N, d) int32, n_survivors, n_pruned_asym,
               n_pruned_test, elapsed, proven
    """
    d = 2 * n_half
    S = m
    n_total = count_compositions(d, S)
    corr = correction(m)

    if verbose:
        print(f"\n[L0] d={d}, m={m}, compositions={n_total:,}")
        print(f"     correction={corr:.6f}, threshold={c_target+corr:.6f}")

    t0 = time.time()
    all_survivors = []
    n_pruned_asym = 0
    n_pruned_test = 0
    n_processed = 0
    n_non_canonical = 0

    for batch in generate_compositions_batched(d, S, batch_size=200_000):
        n_processed += len(batch)

        # Canonical filter (match GPU: only c <= rev(c))
        canon = _canonical_mask(batch)
        n_non_canonical += int(np.sum(~canon))
        batch = batch[canon]
        if len(batch) == 0:
            continue

        # Asymmetry filter
        needs_check = asymmetry_prune_mask(batch, n_half, m, c_target)
        n_asym_batch = int(np.sum(~needs_check))
        n_pruned_asym += n_asym_batch

        candidates = batch[needs_check]
        if len(candidates) == 0:
            continue

        # Dynamic per-window threshold (matches GPU exactly)
        survived_mask = _prune_dynamic(candidates, n_half, m, c_target)
        n_pruned_test += int(np.sum(~survived_mask))

        survivors = candidates[survived_mask]
        if len(survivors) > 0:
            all_survivors.append(survivors)

    elapsed = time.time() - t0

    if all_survivors:
        all_survivors = np.vstack(all_survivors)
    else:
        all_survivors = np.empty((0, d), dtype=np.int32)

    n_survivors = len(all_survivors)
    proven = n_survivors == 0

    if verbose:
        print(f"     {elapsed:.2f}s: {n_processed:,} compositions processed")
        print(f"     asym pruned: {n_pruned_asym:,}, "
              f"test pruned: {n_pruned_test:,}, "
              f"survivors: {n_survivors:,}")
        if proven:
            print(f"     PROVEN at L0!")

    return {
        'survivors': all_survivors,
        'n_survivors': n_survivors,
        'n_pruned_asym': n_pruned_asym,
        'n_pruned_test': n_pruned_test,
        'n_processed': n_processed,
        'elapsed': elapsed,
        'proven': proven,
    }


# =====================================================================
# Refinement: uniform full-bin split (matches GPU semantics)
# =====================================================================

def generate_children_uniform(parent_int, m, c_target):
    """Generate all child compositions from a parent via uniform 2-split.

    Each parent bin c_i is split into (a, b) where a + b = c_i.
    Both sub-bins are capped at x_cap (energy bound) for efficiency.

    Parameters
    ----------
    parent_int : (d_parent,) int array
    m : int
    c_target : float

    Returns
    -------
    (N_children, d_child) int32 array where d_child = 2 * d_parent
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent

    # x_cap: single-bin energy cap (matches GPU host_refine.cuh)
    corr = 2.0 / m + 1.0 / (m * m)
    thresh = c_target + corr + 1e-9
    import math
    x_cap = int(math.floor(m * math.sqrt(thresh / d_child)))
    x_cap = min(x_cap, m)
    x_cap = max(x_cap, 0)

    # Build per-bin split options
    per_bin_choices = []
    for i in range(d_parent):
        b_i = int(parent_int[i])
        lo = max(0, b_i - x_cap)
        hi = min(b_i, x_cap)
        if lo > hi:
            # This parent can't produce valid children
            return np.empty((0, d_child), dtype=np.int32)
        per_bin_choices.append(list(range(lo, hi + 1)))

    # Total children = product of choice counts
    total = 1
    for choices in per_bin_choices:
        total *= len(choices)

    if total == 0:
        return np.empty((0, d_child), dtype=np.int32)

    # For very large expansions, use chunked generation to avoid OOM
    if total > 50_000_000:
        return _generate_children_chunked(parent_int, per_bin_choices,
                                          d_parent, d_child, total)

    children = np.empty((total, d_child), dtype=np.int32)
    idx = 0
    for combo in itertools.product(*per_bin_choices):
        for i in range(d_parent):
            children[idx, 2 * i] = combo[i]
            children[idx, 2 * i + 1] = int(parent_int[i]) - combo[i]
        idx += 1

    return children


def _generate_children_chunked(parent_int, per_bin_choices, d_parent,
                                d_child, total):
    """Generate children in chunks to avoid memory blowup."""
    chunk_size = 10_000_000
    chunks = []
    buf = np.empty((chunk_size, d_child), dtype=np.int32)
    idx = 0

    for combo in itertools.product(*per_bin_choices):
        for i in range(d_parent):
            buf[idx, 2 * i] = combo[i]
            buf[idx, 2 * i + 1] = int(parent_int[i]) - combo[i]
        idx += 1
        if idx == chunk_size:
            chunks.append(buf[:idx].copy())
            idx = 0

    if idx > 0:
        chunks.append(buf[:idx].copy())

    return np.vstack(chunks) if chunks else np.empty((0, d_child), dtype=np.int32)


def test_children(children_int, n_half_child, m, c_target):
    """Prune children via asymmetry + dynamic threshold.

    NO canonical filter at refinement levels — applying it here would
    silently drop canonical children whose parent is non-canonical
    (rev(P) for canonical P), since rev(P) is never in our parent list.
    Instead, survivors are canonicalized and deduped after testing.

    Returns
    -------
    (survivors, stats) where survivors is (K, d_child) int32
    """
    if len(children_int) == 0:
        d_child = children_int.shape[1] if children_int.ndim == 2 else 2
        return np.empty((0, d_child), dtype=np.int32), {
            'n_tested': 0, 'n_canonical': 0, 'n_asym': 0, 'n_test': 0,
            'n_survived': 0
        }

    N, d_child = children_int.shape

    # NOTE: No canonical filter here. At refinement levels, we only refine
    # canonical parents P. Children of P include both canonical and
    # non-canonical compositions. The non-canonical child C has the same
    # autoconvolution as rev(C), which is a child of rev(P). Since rev(P)
    # is not in our parent list, we must test C here (equivalently testing
    # rev(C)). Survivors are canonicalized after testing.

    # Asymmetry filter
    needs_check = asymmetry_prune_mask(children_int, n_half_child, m, c_target)
    n_asym = int(np.sum(~needs_check))
    candidates = children_int[needs_check]

    if len(candidates) > 0:
        # Dynamic per-window threshold
        survived_mask = _prune_dynamic(candidates, n_half_child, m, c_target)
        survivors = candidates[survived_mask]
        n_test = int(np.sum(~survived_mask))
    else:
        survivors = np.empty((0, d_child), dtype=np.int32)
        n_test = 0

    # Canonicalize survivors: replace each with min(c, rev(c)) lex
    if len(survivors) > 0:
        for i in range(len(survivors)):
            rev = survivors[i, ::-1].copy()
            if tuple(rev) < tuple(survivors[i]):
                survivors[i] = rev

    return survivors, {
        'n_tested': N,
        'n_canonical': 0,
        'n_asym': n_asym,
        'n_test': n_test,
        'n_survived': len(survivors),
    }


# =====================================================================
# Multiprocessing helpers
# =====================================================================

def _init_worker_threads(n_threads):
    """Limit Numba parallelism in each worker to avoid oversubscription."""
    numba.set_num_threads(n_threads)


def _process_single_parent(args):
    """Worker: generate children for one parent, prune, return survivors."""
    parent, m, c_target, n_half_child, batch_size = args

    children = generate_children_uniform(parent, m, c_target)
    n_children = len(children)

    if n_children == 0:
        return None, {'children': 0, 'asym': 0, 'test': 0, 'survived': 0}

    parent_survivors = []
    total_asym = 0
    total_test = 0
    total_survived = 0

    for start in range(0, n_children, batch_size):
        end = min(start + batch_size, n_children)
        batch = children[start:end]

        survivors, stats = test_children(batch, n_half_child, m, c_target)
        total_asym += stats['n_asym']
        total_test += stats['n_test']
        total_survived += stats['n_survived']

        if len(survivors) > 0:
            parent_survivors.append(survivors)

    if parent_survivors:
        result = np.vstack(parent_survivors)
    else:
        result = None

    return result, {
        'children': n_children,
        'asym': total_asym,
        'test': total_test,
        'survived': total_survived,
    }


# =====================================================================
# Cascade runner
# =====================================================================

def run_cascade(n_half, m, c_target, max_levels=10, n_workers=None,
                verbose=True):
    """Run the full CPU cascade: L0 + refinement levels.

    Parameters
    ----------
    n_half : int
        Initial n_half (d0 = 2 * n_half).
    m : int
        Grid resolution.
    c_target : float
        Target lower bound.
    max_levels : int
        Max refinement levels after L0.
    n_workers : int or None
        Number of parallel workers for refinement levels.
        None = auto-detect CPU count.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with cascade results.
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    n_workers = max(1, n_workers)

    # Ensure enough file descriptors for spawn-based multiprocessing
    # (each worker needs ~6 fds for pipes/semaphores)
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        needed = n_workers * 8 + 256  # generous headroom
        if soft < needed:
            resource.setrlimit(resource.RLIMIT_NOFILE,
                               (min(needed, hard), hard))
    except (ImportError, ValueError, OSError):
        pass  # Windows or insufficient permissions — proceed anyway
    d0 = 2 * n_half
    corr = correction(m)
    n_total = count_compositions(d0, m)

    if verbose:
        print(f"\n{'='*70}")
        print(f"CPU CASCADE PROVER")
        print(f"  n_half={n_half}, m={m}, d0={d0}, c_target={c_target}")
        print(f"  correction={corr:.6f}, effective threshold={c_target+corr:.6f}")
        print(f"  L0 compositions: {n_total:,}")
        print(f"  workers: {n_workers} (CPUs: {mp.cpu_count()})")
        print(f"  max refinement levels: {max_levels}")
        print(f"{'='*70}")

    t_total = time.time()

    # --- Level 0 ---
    l0 = run_level0(n_half, m, c_target, verbose=verbose)

    info = {
        'n_half': n_half, 'm': m, 'd0': d0, 'c_target': c_target,
        'correction': corr,
        'l0_time': l0['elapsed'],
        'l0_survivors': l0['n_survivors'],
        'l0_pruned_asym': l0['n_pruned_asym'],
        'l0_pruned_test': l0['n_pruned_test'],
        'levels': [],
    }

    if l0['proven']:
        info['proven_at'] = 'L0'
        info['total_time'] = time.time() - t_total
        return info

    # --- Refinement levels ---
    current_configs = l0['survivors']
    d_parent = d0
    n_half_parent = n_half

    for level_num in range(1, max_levels + 1):
        d_child = 2 * d_parent
        n_half_child = 2 * n_half_parent
        n_parents = len(current_configs)

        if n_parents == 0:
            break

        if verbose:
            print(f"\n[L{level_num}] d_parent={d_parent} -> d_child={d_child}, "
                  f"{n_parents:,} parents")

        t_level = time.time()
        all_survivors = []
        total_children = 0
        total_canonical = 0
        total_asym = 0
        total_test = 0
        total_survived = 0

        if n_workers > 1 and n_parents > n_workers:
            # --- Parallel path: distribute parents across workers ---
            numba_threads = max(1, mp.cpu_count() // n_workers)
            work = [(current_configs[i], m, c_target, n_half_child, 500_000)
                    for i in range(n_parents)]
            chunksize = max(1, n_parents // (n_workers * 20))

            if verbose:
                print(f"     (parallel: {n_workers} workers, "
                      f"chunksize={chunksize}, "
                      f"numba_threads={numba_threads})")

            completed = 0
            # Use "spawn" context — "fork" is unsafe after Numba/OpenMP
            # initializes threads in the main process.
            ctx = mp.get_context("spawn")
            with ctx.Pool(n_workers, initializer=_init_worker_threads,
                          initargs=(numba_threads,)) as pool:
                for surv, stats in pool.imap_unordered(
                        _process_single_parent, work,
                        chunksize=chunksize):
                    total_children += stats['children']
                    total_asym += stats['asym']
                    total_test += stats['test']
                    total_survived += stats['survived']
                    completed += 1

                    if surv is not None:
                        all_survivors.append(surv)

                    if verbose and n_parents > 100 and completed % 1000 == 0:
                        elapsed_so_far = time.time() - t_level
                        rate = completed / elapsed_so_far
                        eta = (n_parents - completed) / rate
                        print(f"     [{completed}/{n_parents}] "
                              f"{total_survived:,} survivors so far, "
                              f"ETA {_fmt_time(eta)}")

        else:
            # --- Sequential path (original, for small parent sets) ---
            for p_idx in range(n_parents):
                parent = current_configs[p_idx]

                children = generate_children_uniform(parent, m, c_target)
                n_children = len(children)
                total_children += n_children

                if n_children == 0:
                    continue

                batch_size = 500_000
                for start in range(0, n_children, batch_size):
                    end = min(start + batch_size, n_children)
                    batch = children[start:end]

                    survivors, stats = test_children(
                        batch, n_half_child, m, c_target)

                    total_canonical += stats.get('n_canonical', 0)
                    total_asym += stats['n_asym']
                    total_test += stats['n_test']
                    total_survived += stats['n_survived']

                    if len(survivors) > 0:
                        all_survivors.append(survivors)

                if verbose and n_parents > 100 and (p_idx + 1) % 1000 == 0:
                    elapsed_so_far = time.time() - t_level
                    rate = (p_idx + 1) / elapsed_so_far
                    eta = (n_parents - p_idx - 1) / rate
                    print(f"     [{p_idx+1}/{n_parents}] "
                          f"{total_survived:,} survivors so far, "
                          f"ETA {_fmt_time(eta)}")

        elapsed_level = time.time() - t_level

        if all_survivors:
            all_survivors = np.vstack(all_survivors)
            # Deduplicate — the same canonical survivor can arise from
            # multiple parents (a child of P and a child of Q may
            # canonicalize to the same composition)
            seen = set()
            unique_idx = []
            for i in range(len(all_survivors)):
                key = tuple(all_survivors[i])
                if key not in seen:
                    seen.add(key)
                    unique_idx.append(i)
            all_survivors = all_survivors[unique_idx]
        else:
            all_survivors = np.empty((0, d_child), dtype=np.int32)

        n_survived = len(all_survivors)

        if n_parents > 0:
            factor = n_survived / n_parents
        else:
            factor = 0

        lvl_info = {
            'level': level_num,
            'd_parent': d_parent,
            'd_child': d_child,
            'parents_in': n_parents,
            'total_children': total_children,
            'children_per_parent': total_children / max(1, n_parents),
            'asym_pruned': total_asym,
            'test_pruned': total_test,
            'survivors_out': n_survived,
            'expansion_factor': factor,
            'elapsed': elapsed_level,
        }
        info['levels'].append(lvl_info)

        if verbose:
            print(f"     {elapsed_level:.2f}s: {total_children:,} children "
                  f"({total_children/max(1,n_parents):.1f}/parent)")
            print(f"     asym: {total_asym:,}, "
                  f"test: {total_test:,}, "
                  f"survivors: {n_survived:,} (factor={factor:.4f}x)")
            if n_survived == 0:
                print(f"     PROVEN at L{level_num}!")

        if n_survived == 0:
            info['proven_at'] = f'L{level_num}'
            break

        # Prepare next level
        current_configs = all_survivors
        d_parent = d_child
        n_half_parent = n_half_child

    info['total_time'] = time.time() - t_total

    if verbose:
        print(f"\n{'='*70}")
        if 'proven_at' in info:
            print(f"PROVEN: c >= {c_target} (cascade converges at "
                  f"{info['proven_at']})")
        else:
            n_remain = len(current_configs) if len(current_configs) > 0 else 0
            print(f"NOT PROVEN: {n_remain:,} survivors remain at "
                  f"d={d_parent}")
        print(f"Total time: {_fmt_time(info['total_time'])}")
        print(f"{'='*70}")

    return info


# =====================================================================
# Formatting helpers
# =====================================================================

def _fmt_time(seconds):
    if seconds < 60:
        return f'{seconds:.2f}s'
    if seconds < 3600:
        return f'{seconds/60:.1f}m'
    return f'{seconds/3600:.2f}h'


def print_summary(info):
    """Print a compact summary table."""
    print(f"\nCASCADE SUMMARY: n_half={info['n_half']}, m={info['m']}, "
          f"d0={info['d0']}, c_target={info['c_target']}")
    print(f"  L0: {_fmt_time(info['l0_time'])}, "
          f"{info['l0_survivors']:,} survivors")

    if info.get('levels'):
        print(f"\n  {'Level':>5} | {'Parents':>10} | {'Children':>12} | "
              f"{'Ch/Par':>8} | {'Survivors':>10} | {'Factor':>10} | "
              f"{'Time':>10}")
        print(f"  {'-'*75}")

        for lvl in info['levels']:
            factor = lvl['expansion_factor']
            if factor == 0:
                fstr = '0x'
            elif factor < 0.01:
                fstr = f'{factor:.6f}x'
            else:
                fstr = f'{factor:.4f}x'

            print(f"  L{lvl['level']:>4} | {lvl['parents_in']:>10,} | "
                  f"{lvl['total_children']:>12,} | "
                  f"{lvl['children_per_parent']:>8.1f} | "
                  f"{lvl['survivors_out']:>10,} | "
                  f"{fstr:>10} | "
                  f"{_fmt_time(lvl['elapsed']):>10}")

    proven_at = info.get('proven_at')
    if proven_at:
        print(f"\n  PROVEN at {proven_at} "
              f"(total: {_fmt_time(info['total_time'])})")
    else:
        last_lvl = info['levels'][-1] if info.get('levels') else None
        remain = last_lvl['survivors_out'] if last_lvl else info['l0_survivors']
        print(f"\n  NOT PROVEN — {remain:,} survivors remain "
              f"(total: {_fmt_time(info['total_time'])})")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CPU-only cascade prover (no GPU, no dimension limits)')
    parser.add_argument('--n_half', type=int, default=2,
                        help='Initial n_half (d0 = 2*n_half, default: 2)')
    parser.add_argument('--m', type=int, default=20,
                        help='Grid resolution (default: 20)')
    parser.add_argument('--c_target', type=float, default=1.30,
                        help='Target lower bound (default: 1.30)')
    parser.add_argument('--max_levels', type=int, default=10,
                        help='Max refinement levels after L0 (default: 10)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel workers (default: CPU count)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory (default: data)')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    args = parser.parse_args()

    info = run_cascade(
        n_half=args.n_half,
        m=args.m,
        c_target=args.c_target,
        max_levels=args.max_levels,
        n_workers=args.workers,
        verbose=not args.quiet,
    )

    print_summary(info)

    # Save result
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(args.output_dir,
                        f'cpu_cascade_{ts}.json')

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, 'w') as f:
        json.dump(info, f, indent=2, default=convert)
    print(f"\nResult saved to {path}")


if __name__ == '__main__':
    main()
