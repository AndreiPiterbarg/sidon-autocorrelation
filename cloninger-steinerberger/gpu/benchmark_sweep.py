"""Empirical cascade benchmark for optimal (m, n_half) selection.

Runs L0 fully, samples a fixed number of survivors, then cascades through
refinement levels (L1 -> L2 -> L3 ...) until convergence (0 survivors).
Records time and expansion factor at each level.  Projects full runtime
by linear scaling from the sample.

Usage:
    python -m cloninger-steinerberger.gpu.benchmark_sweep
    python -m cloninger-steinerberger.gpu.benchmark_sweep --m_values 50,100,150,200
    python -m cloninger-steinerberger.gpu.benchmark_sweep --dry_run
    python -m cloninger-steinerberger.gpu.benchmark_sweep --resume
    python -m cloninger-steinerberger.gpu.benchmark_sweep --sample_size 10000
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'archive', 'cpu'))
from pruning import correction, count_compositions
from gpu import wrapper
from gpu.solvers import gpu_run_single_level


# Supported d_parent values for gpu_refine_parents (from dispatch.cuh)
SUPPORTED_D_PARENT = {4, 6, 8, 12, 16, 24}

# Known upper bound on c (Yu 2021). If c_target + correction exceeds this,
# the proof is vacuously true (coarse grid can't represent near-optimal
# functions) and the benchmark result is meaningless.
C_UPPER_BOUND = 1.5029


# ---------------------------------------------------------------------------
# x_cap and refinement count (Python reimplementation of host_refine.cuh)
# ---------------------------------------------------------------------------

def compute_x_cap(m, d_child, c_target):
    """Compute the single-bin energy cap matching host_refine.cuh:387."""
    corr = 2.0 / m + 1.0 / (m * m)
    thresh = c_target + corr + 1e-9
    x_cap = int(math.floor(m * math.sqrt(thresh / d_child)))
    x_cap = min(x_cap, m)
    x_cap = max(x_cap, 0)
    return x_cap


def compute_refs_per_parent(survivors, m, c_target, d_child):
    """Per-parent refinement counts (vectorized numpy).

    Parameters
    ----------
    survivors : np.ndarray of shape (N, d_parent), int32
        Parent survivor configs (each row sums to m).
    m : int
        Grid resolution.
    c_target : float
        Target lower bound.
    d_child : int
        Child dimension for the next level.

    Returns
    -------
    np.ndarray of shape (N,), dtype=int64 : refinement count per parent.
    """
    if len(survivors) == 0:
        return np.array([], dtype=np.int64)

    x_cap = compute_x_cap(m, d_child, c_target)

    # Vectorized: for each bin, eff = min(B, x_cap) - max(0, B - x_cap) + 1
    B = survivors.astype(np.int64)
    lo = np.maximum(0, B - x_cap)
    hi = np.minimum(B, x_cap)
    eff = np.maximum(hi - lo + 1, 0)

    return np.prod(eff, axis=1)


def compute_total_refs(survivors, m, c_target, d_child):
    """Sum of per-parent refinement counts."""
    return int(np.sum(compute_refs_per_parent(survivors, m, c_target, d_child)))


# ---------------------------------------------------------------------------
# Refinement level runner
# ---------------------------------------------------------------------------

def run_refinement_level(parent_configs, d_parent, m, c_target,
                         max_survivors, time_budget_sec=0.0):
    """Run one refinement level on the GPU.

    Thin wrapper around wrapper.refine_parents() returning a standardized dict.

    Parameters
    ----------
    parent_configs : np.ndarray of shape (N, d_parent), int32
    d_parent : int
    m : int
    c_target : float
    max_survivors : int
    time_budget_sec : float

    Returns
    -------
    dict with: parents_in, survivors_out, survivor_configs, n_extracted,
               elapsed, total_refs, throughput, expansion_factor,
               refs_per_parent, timed_out
    """
    parents_in = len(parent_configs)

    t0 = time.time()
    result = wrapper.refine_parents(
        d_parent=d_parent,
        parent_configs_array=parent_configs,
        m=m,
        c_target=c_target,
        max_survivors=max_survivors,
        time_budget_sec=time_budget_sec)
    elapsed = time.time() - t0

    total_refs = (result['total_asym'] + result['total_test'] +
                  result['total_survivors'])
    throughput = total_refs / elapsed if elapsed > 0 else 0
    expansion = result['total_survivors'] / parents_in if parents_in > 0 else 0
    rpp = total_refs / parents_in if parents_in > 0 else 0

    return {
        'parents_in': parents_in,
        'survivors_out': result['total_survivors'],
        'survivor_configs': result['survivor_configs'],
        'n_extracted': result['n_extracted'],
        'elapsed': elapsed,
        'total_refs': total_refs,
        'throughput': throughput,
        'expansion_factor': expansion,
        'refs_per_parent': rpp,
        'timed_out': result.get('timed_out', False),
    }


# ---------------------------------------------------------------------------
# Cascade pipeline
# ---------------------------------------------------------------------------

def run_cascade(n_half, m, c_target, sample_size, max_survivors,
                level_timeout):
    """Run L0 fully, then cascade refinement levels on a sample.

    Parameters
    ----------
    n_half : int
    m : int
    c_target : float
    sample_size : int
        Number of L0 survivors to sample for the cascade.
    max_survivors : int
        Max survivor extraction buffer per level.
    level_timeout : float
        Per-level time budget in seconds (0 = no limit).

    Returns
    -------
    dict with: n_half, m, d0, correction, l0_compositions, l0_time,
               l0_survivors, l0_pruned_asym, l0_pruned_test,
               sample_size, scale_factor, levels[], projected_total, proven_at
    """
    d0 = 2 * n_half
    corr = correction(m)
    n_total = count_compositions(d0, m)

    info = {
        'n_half': n_half,
        'm': m,
        'd0': d0,
        'correction': corr,
        'l0_compositions': n_total,
    }

    print(f"\n{'='*60}")
    print(f"CONFIG: n_half={n_half}, m={m}, d0={d0}, "
          f"corr={corr:.6f}, L0 comps={n_total:,}")
    print(f"{'='*60}")

    # ---- Step 1: Level 0 (GPU) ----
    # Use early stop: collect ~2x sample_size survivors then stop.
    # This avoids enumerating all L0 compositions for large configs.
    # We request 2x to have margin for random sampling.
    l0_target = sample_size * 2
    print(f"\n  [L0] Running Level 0 (target_survivors={l0_target:,})...")
    t0 = time.time()
    l0_result = gpu_run_single_level(
        n_half, m, c_target,
        verbose=False,
        extract_survivors=True,
        stream_to_disk=True,
        target_survivors=l0_target)
    l0_time = time.time() - t0

    # n_survivors is the count from the portion of the space that was
    # actually processed (may be < total if early-stopped).
    l0_survivors_n = l0_result['n_survivors']
    l0_extracted = l0_result['stats'].get('n_extracted', l0_survivors_n)
    l0_processed = l0_result['stats']['n_processed']

    # Estimate total L0 survivors by extrapolation
    if l0_processed > 0 and l0_processed < n_total:
        l0_survivors_est = int(l0_survivors_n * (n_total / l0_processed))
        l0_time_est = l0_time * (n_total / l0_processed)
        early_stopped = True
    else:
        l0_survivors_est = l0_survivors_n
        l0_time_est = l0_time
        early_stopped = False

    info['l0_time'] = l0_time
    info['l0_time_est'] = l0_time_est
    info['l0_survivors'] = l0_survivors_est
    info['l0_survivors_sampled'] = l0_extracted
    info['l0_processed'] = l0_processed
    info['l0_early_stopped'] = early_stopped
    info['l0_pruned_asym'] = l0_result['stats']['n_pruned_asym']
    info['l0_pruned_test'] = l0_result['stats']['n_pruned_test']

    if early_stopped:
        pct = 100.0 * l0_processed / n_total
        print(f"  [L0] Early stop in {l0_time:.2f}s: "
              f"{l0_extracted:,} extracted, {l0_survivors_n:,} counted "
              f"({l0_processed:,}/{n_total:,} = {pct:.1f}% processed)")
        print(f"  [L0] Estimated total: ~{l0_survivors_est:,} survivors, "
              f"~{_fmt_time(l0_time_est)} full L0 time")
    else:
        print(f"  [L0] Done in {l0_time:.2f}s: "
              f"{l0_survivors_n:,} survivors / {n_total:,} total")

    if l0_result['proven']:
        info['proven_at'] = 'L0'
        info['levels'] = []
        info['sample_size'] = 0
        info['scale_factor'] = 1.0
        info['projected_total'] = l0_time_est
        print(f"  [L0] PROVEN at Level 0!")
        _cleanup_survivor_files(l0_result)
        return info

    # Load L0 survivors
    survivors = _load_survivors(l0_result, d0)
    l0_survivor_file = l0_result.get('survivor_file')

    if len(survivors) == 0:
        info['error'] = 'No survivors extracted despite n_survivors > 0'
        _cleanup_survivor_files(l0_result)
        return info

    N0_extracted = len(survivors)
    N0 = l0_survivors_est  # use estimated total for projection

    # ---- Step 2: Sample L0 survivors ----
    actual_sample = min(sample_size, N0_extracted)
    scale = N0 / actual_sample if actual_sample > 0 else 1.0
    info['sample_size'] = actual_sample
    info['scale_factor'] = scale

    if actual_sample < N0_extracted:
        rng = np.random.RandomState(42)
        idx = rng.choice(N0_extracted, size=actual_sample, replace=False)
        sampled = survivors[idx]
        print(f"\n  [Sample] {actual_sample:,} / ~{N0:,} est L0 survivors "
              f"(seed=42, scale={scale:.1f}x)")
    else:
        sampled = survivors
        print(f"\n  [Sample] Using all {N0_extracted:,} extracted survivors "
              f"(scale={scale:.1f}x from ~{N0:,} est total)")

    # Clean up L0 survivor file now that we have the sample in memory
    _cleanup_file(l0_survivor_file)

    # ---- Step 3: Cascade through refinement levels ----
    levels = []
    current_configs = sampled
    d_parent = d0
    level_num = 1
    proven_at = None

    while True:
        d_child = 2 * d_parent

        # Check if d_parent is supported
        if d_parent not in SUPPORTED_D_PARENT:
            print(f"\n  [L{level_num}] d_parent={d_parent} not supported "
                  f"(max supported: {max(SUPPORTED_D_PARENT)}). "
                  f"Stopping cascade.")
            break

        parents_in = len(current_configs)
        if parents_in == 0:
            # Previous level already converged
            break

        print(f"\n  [L{level_num}] Refining: d_parent={d_parent} -> "
              f"d_child={d_child}, {parents_in:,} parents...")

        level_result = run_refinement_level(
            current_configs, d_parent, m, c_target,
            max_survivors=max_survivors,
            time_budget_sec=level_timeout)

        lvl = {
            'level': level_num,
            'd_parent': d_parent,
            'd_child': d_child,
            'parents_in': level_result['parents_in'],
            'survivors_out': level_result['survivors_out'],
            'n_extracted': level_result['n_extracted'],
            'elapsed': level_result['elapsed'],
            'total_refs': level_result['total_refs'],
            'throughput': level_result['throughput'],
            'expansion_factor': level_result['expansion_factor'],
            'refs_per_parent': level_result['refs_per_parent'],
            'timed_out': level_result['timed_out'],
        }
        levels.append(lvl)

        factor_str = f"{level_result['expansion_factor']:.4f}x"
        print(f"  [L{level_num}] Done in {level_result['elapsed']:.2f}s: "
              f"{level_result['survivors_out']:,} survivors / "
              f"{level_result['total_refs']:,} refs "
              f"({level_result['throughput']:,.0f} refs/s), "
              f"factor={factor_str}")

        if level_result['timed_out']:
            print(f"  [L{level_num}] TIMED OUT after {level_timeout:.0f}s")
            break

        if level_result['survivors_out'] == 0:
            proven_at = f'L{level_num}'
            print(f"  [L{level_num}] PROVEN at Level {level_num}!")
            break

        # Prepare next level
        current_configs = level_result['survivor_configs']
        d_parent = d_child
        level_num += 1

    info['levels'] = levels
    info['proven_at'] = proven_at

    # ---- Step 4: Project full runtime ----
    projected = project_full_runtime(info, N0, actual_sample)
    info['projected_total'] = projected

    print(f"\n  [Projection] Full projected time: {_fmt_time(projected)}")

    return info


def project_full_runtime(info, N0, sample_size):
    """Project full runtime from cascade sample using linear scaling.

    Every level's projected time = elapsed * (N0 / sample_size).
    L0 time uses the estimated full-run time (extrapolated if early-stopped).

    Parameters
    ----------
    info : dict
        Cascade result with l0_time_est and levels[].
    N0 : int
        Total L0 survivors (estimated full population).
    sample_size : int
        Number of L0 survivors sampled.

    Returns
    -------
    float : projected total wall time in seconds.
    """
    l0_time = info.get('l0_time_est', info['l0_time'])

    if sample_size == 0 or N0 == 0:
        return l0_time

    scale = N0 / sample_size

    total = l0_time
    for lvl in info.get('levels', []):
        projected_level = lvl['elapsed'] * scale
        total += projected_level

    return total


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_cascade_table(info):
    """Print detailed cascade results for a single config."""
    n_half = info['n_half']
    m = info['m']
    d0 = info['d0']

    print(f"\nCONFIG: n_half={n_half}, m={m}, d0={d0}")
    print(f"  L0: {_fmt_time(info['l0_time'])}, "
          f"{_fmt_count(info['l0_survivors'])} survivors (full)")

    sample = info.get('sample_size', 0)
    N0 = info.get('l0_survivors', 0)
    scale = info.get('scale_factor', N0 / sample if sample > 0 else 1.0)
    is_exact = (scale <= 1.0 + 1e-9)  # no projection needed

    if sample > 0 and N0 > 0:
        if is_exact:
            print(f"  All {N0:,} L0 survivors used (exact, no projection)")
        else:
            print(f"  Sample: {sample:,} / {N0:,} (scale={scale:.1f}x)")

    levels = info.get('levels', [])
    if not levels:
        if info.get('proven_at') == 'L0':
            print(f"  PROVEN at L0")
        return

    proj_col = 'Exact Total' if is_exact else f'Full Proj ({scale:.0f}x)'
    print()
    header = (f"  {'Level':>5} | {'Parents':>10} | {'Survivors':>10} | "
              f"{'Factor':>8} | {'Time':>8} | {'Refs/s':>10} | "
              f"{proj_col:>16}")
    print(header)
    print(f"  {'-'*79}")

    for lvl in levels:
        lnum = f"L{lvl['level']}"
        proj_time = lvl['elapsed'] * scale
        ef = lvl['expansion_factor']
        # Show enough precision to distinguish near-zero from zero
        if ef == 0:
            factor = '0x'
        elif ef < 0.01:
            factor = f"{ef:.4f}x"
        else:
            factor = f"{ef:.2f}x"
        thru = _fmt_count(lvl['throughput'])

        surv = lvl['survivors_out']
        surv_str = _fmt_count(surv)

        row = (f"  {lnum:>5} | {_fmt_count(lvl['parents_in']):>10} | "
               f"{surv_str:>10} | "
               f"{factor:>8} | {_fmt_time(lvl['elapsed']):>8} | "
               f"{thru:>10} | {_fmt_time(proj_time):>16}")
        print(row)
        if lvl.get('timed_out'):
            print(f"         ^ TIMED OUT")

    # Final status
    last_level = levels[-1] if levels else None
    final_survivors = last_level['survivors_out'] if last_level else 0
    projected = info.get('projected_total')

    if info.get('proven_at'):
        label = 'Total (exact)' if is_exact else 'Total projected'
        if projected is not None:
            print(f"\n  {label}: {_fmt_time(projected)}")
        print(f"  PROVEN — cascade converges at {info['proven_at']}")
    elif final_survivors > 0:
        last_d = last_level['d_child'] if last_level else 0
        if projected is not None:
            label = 'Time to reach max depth' if is_exact else 'Projected time to max depth'
            print(f"\n  {label}: {_fmt_time(projected)}")
        print(f"  NOT PROVEN — {final_survivors:,} survivors remain at "
              f"d={last_d} (no further refinement supported)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_survivors(result, d):
    """Load survivors from Level 0 result (in-memory or disk)."""
    sf = result.get('survivor_file')
    if sf and os.path.exists(sf):
        return wrapper.load_survivors_chunk(sf, d)
    return result['survivors']


def _cleanup_survivor_files(result):
    """Delete temporary survivor binary file from a Level 0 result."""
    sf = result.get('survivor_file')
    _cleanup_file(sf)


def _cleanup_file(path):
    """Delete a file if it exists."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _fmt_time(seconds):
    """Format seconds as human-readable string."""
    if seconds is None:
        return 'N/A'
    if seconds == float('inf'):
        return 'inf'
    if seconds < 60:
        return f'{seconds:.2f}s'
    if seconds < 3600:
        return f'{seconds/60:.1f}m'
    return f'{seconds/3600:.2f}h'


def _fmt_count(n):
    """Format large count with SI suffix."""
    if n is None:
        return 'N/A'
    if n < 1000:
        return str(int(n))
    if n < 1_000_000:
        return f'{n/1000:.1f}K'
    if n < 1_000_000_000:
        return f'{n/1_000_000:.2f}M'
    return f'{n/1_000_000_000:.2f}B'


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------

def build_sweep_configs(m_values, n_half_values, c_target):
    """Build configs sorted cheapest-first by L0 composition count.

    Configs where c_target + correction >= C_UPPER_BOUND are flagged as
    vacuous (the proof is trivially true due to coarse discretization).
    """
    configs = []
    skipped = []
    for n_half in n_half_values:
        d = 2 * n_half
        for m in m_values:
            corr = correction(m)
            eff_thresh = c_target + corr
            if eff_thresh >= C_UPPER_BOUND:
                skipped.append((n_half, m, corr, eff_thresh))
                continue
            n_total = count_compositions(d, m)
            configs.append({
                'n_half': n_half,
                'm': m,
                'd': d,
                'n_total': n_total,
                'correction': corr,
            })
    if skipped:
        print(f"Skipping {len(skipped)} vacuous configs "
              f"(c_target + correction >= {C_UPPER_BOUND}):")
        for n_half, m, corr, eff in skipped:
            print(f"  n_half={n_half}, m={m}: "
                  f"correction={corr:.4f}, threshold={eff:.4f}")
    configs.sort(key=lambda c: c['n_total'])
    return configs


def _get_sample_size(m, sample_small, sample_large, m_threshold=30):
    """Return sample size based on m value."""
    return sample_small if m < m_threshold else sample_large


def run_sweep(configs, c_target, sample_small, sample_large, max_survivors,
              level_timeout, output_dir, resume):
    """Run the full sweep, checkpointing after each config."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'benchmark_checkpoint.json')

    # Load checkpoint if resuming
    completed = {}
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            ckpt = json.load(f)
        for r in ckpt.get('results', []):
            key = (r['n_half'], r['m'])
            completed[key] = r
        print(f"Resumed: {len(completed)} configs already completed")

    results = list(completed.values())

    for i, cfg in enumerate(configs):
        key = (cfg['n_half'], cfg['m'])
        if key in completed:
            print(f"\n[{i+1}/{len(configs)}] SKIP n_half={cfg['n_half']}, "
                  f"m={cfg['m']} (already completed)")
            continue

        sample_size = _get_sample_size(cfg['m'], sample_small, sample_large)
        print(f"\n[{i+1}/{len(configs)}] Running n_half={cfg['n_half']}, "
              f"m={cfg['m']} (sample_size={sample_size:,})...")

        try:
            result = run_cascade(
                cfg['n_half'], cfg['m'], c_target,
                sample_size, max_survivors, level_timeout)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            result = {
                'n_half': cfg['n_half'],
                'm': cfg['m'],
                'd0': cfg['d'],
                'error': str(e),
            }
            # Force library reload after GPU failure
            wrapper._lib = None

        results.append(result)
        format_cascade_table(result)

        # Checkpoint
        _save_checkpoint(checkpoint_path, results, c_target,
                         sample_small, sample_large,
                         [c['m'] for c in configs],
                         list(set(c['n_half'] for c in configs)))

    return results


def _save_checkpoint(path, results, c_target, sample_small, sample_large,
                     m_values, n_half_values):
    """Save checkpoint JSON."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj == float('inf'):
            return "inf"
        return obj

    def convert_dict(d):
        return {k: convert(v) for k, v in d.items()}

    def convert_result(r):
        out = {}
        for k, v in r.items():
            if k == 'levels' and isinstance(v, list):
                out[k] = [convert_dict(lvl) for lvl in v]
            else:
                out[k] = convert(v)
        return out

    ckpt = {
        'sweep_params': {
            'c_target': c_target,
            'sample_small': sample_small,
            'sample_large': sample_large,
            'm_values': m_values,
            'n_half_values': n_half_values,
        },
        'results': [convert_result(r) for r in results],
    }
    with open(path, 'w') as f:
        json.dump(ckpt, f, indent=2, default=convert)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_summary_table(results):
    """Print cross-config comparison table with per-level factors."""
    ok = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    def sort_key(r):
        t = r.get('projected_total')
        if t is None or t == float('inf') or t == 'inf':
            return float('inf')
        return t

    ok.sort(key=sort_key)

    # Determine max number of refinement levels across all results
    max_levels = max((len(r.get('levels', [])) for r in ok), default=0)

    print(f"\n{'='*120}")
    print("BENCHMARK SWEEP RESULTS (sorted by projected total time)")
    print(f"{'='*120}")

    # Build header
    hdr = (f"{'n_half':>6} | {'m':>5} | {'L0 surv':>10} | "
           f"{'L0 time':>8} | {'Sample':>8}")
    for i in range(1, max_levels + 1):
        hdr += f" | {'L'+str(i)+' fac':>8} | {'L'+str(i)+' surv':>8}"
    hdr += f" | {'Status':>8} | {'Proj Total':>12}"
    print(hdr)
    print('-' * 120)

    for r in ok:
        l0_surv = r.get('l0_survivors', 0)
        l0_time = r.get('l0_time', 0)
        sample = r.get('sample_size', 0)
        levels = r.get('levels', [])

        row = (f"{r['n_half']:>6} | {r['m']:>5} | "
               f"{_fmt_count(l0_surv):>10} | "
               f"{_fmt_time(l0_time):>8} | {_fmt_count(sample):>8}")

        for i in range(max_levels):
            if i < len(levels):
                lvl = levels[i]
                ef = lvl['expansion_factor']
                if ef == 0:
                    fac = '0x'
                elif ef < 0.01:
                    fac = f"{ef:.4f}x"
                else:
                    fac = f"{ef:.2f}x"
                surv = _fmt_count(lvl['survivors_out'])
            else:
                fac = '-'
                surv = '-'
            row += f" | {fac:>8} | {surv:>8}"

        proven_at = r.get('proven_at')
        if proven_at:
            status = proven_at
        else:
            # Show remaining survivors count
            final_surv = levels[-1]['survivors_out'] if levels else 0
            status = f'{final_surv} left' if final_surv > 0 else '-'
        projected = r.get('projected_total')
        row += f" | {status:>8} | {_fmt_time(projected):>12}"
        print(row)

    if failed:
        print(f"\nFAILED CONFIGS:")
        for r in failed:
            print(f"  n_half={r['n_half']}, m={r['m']}: {r['error']}")


def save_results(results, output_dir):
    """Write JSON + flattened CSV output files."""
    ts = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj == float('inf'):
            return "inf"
        return obj

    def convert_dict(d):
        return {k: convert(v) for k, v in d.items()}

    def convert_result(r):
        out = {}
        for k, v in r.items():
            if k == 'levels' and isinstance(v, list):
                out[k] = [convert_dict(lvl) for lvl in v]
            else:
                out[k] = convert(v)
        return out

    # JSON (with nested levels)
    json_path = os.path.join(output_dir, f'benchmark_sweep_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump([convert_result(r) for r in results],
                  f, indent=2, default=convert)
    print(f"\nJSON saved to {json_path}")

    # CSV (flattened: L1_*, L2_*, L3_* columns)
    max_levels = max((len(r.get('levels', [])) for r in results), default=0)

    base_fields = [
        'n_half', 'm', 'd0', 'correction', 'l0_compositions',
        'l0_survivors', 'l0_time', 'l0_pruned_asym', 'l0_pruned_test',
        'sample_size',
    ]
    level_suffixes = [
        'parents_in', 'survivors_out', 'expansion_factor',
        'elapsed', 'total_refs', 'throughput', 'refs_per_parent',
        'timed_out',
    ]
    level_fields = []
    for i in range(1, max_levels + 1):
        for suf in level_suffixes:
            level_fields.append(f'L{i}_{suf}')

    tail_fields = ['proven_at', 'projected_total', 'error']
    fieldnames = base_fields + level_fields + tail_fields

    csv_path = os.path.join(output_dir, f'benchmark_sweep_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = {k: convert(r.get(k, '')) for k in base_fields + tail_fields}
            for i, lvl in enumerate(r.get('levels', []), 1):
                for suf in level_suffixes:
                    row[f'L{i}_{suf}'] = convert(lvl.get(suf, ''))
            writer.writerow(row)
    print(f"CSV saved to {csv_path}")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run(configs, c_target):
    """Print theoretical quantities and dimension schedules (no GPU)."""
    print(f"\nDRY RUN: c_target={c_target}")
    print(f"{'='*120}")

    header = (f"{'#':>3} | {'n_half':>6} | {'m':>5} | {'d0':>3} | "
              f"{'corr':>8} | {'eff_thresh':>10} | {'L0 comps':>12} | "
              f"{'Dim Schedule':>50}")
    print(header)
    print('-' * 120)

    for i, cfg in enumerate(configs):
        n_half = cfg['n_half']
        m = cfg['m']
        d0 = cfg['d']
        corr = cfg['correction']
        eff_thresh = c_target + corr

        # Build dimension schedule with x_cap info
        dims = []
        d_parent = d0
        while d_parent in SUPPORTED_D_PARENT:
            d_child = 2 * d_parent
            x_cap = compute_x_cap(m, d_child, c_target)
            dims.append(f"d{d_child}(xcap={x_cap})")
            d_parent = d_child
        schedule = f"d{d0} -> " + " -> ".join(dims) if dims else f"d{d0} (no refinement)"

        row = (f"{i+1:>3} | {n_half:>6} | {m:>5} | {d0:>3} | "
               f"{corr:>8.4f} | {eff_thresh:>10.6f} | "
               f"{cfg['n_total']:>12,} | {schedule}")
        print(row)

    print(f"\nDimension schedules (full):")
    for n_half in sorted(set(c['n_half'] for c in configs)):
        d0 = 2 * n_half
        dims = [f"L0:d{d0}"]
        d = d0
        level = 1
        while d in SUPPORTED_D_PARENT:
            d_child = 2 * d
            dims.append(f"L{level}:d{d_child}")
            d = d_child
            level += 1
        print(f"  n_half={n_half}: {' -> '.join(dims)}")
        if d not in SUPPORTED_D_PARENT:
            print(f"    (stops at d_parent={d}: not in supported set "
                  f"{sorted(SUPPORTED_D_PARENT)})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Empirical cascade benchmark for optimal (m, n_half)')
    parser.add_argument('--m_values', type=str,
                        default='10,20,30,40,50',
                        help='Comma-separated m values (default: 10-50 step 10)')
    parser.add_argument('--n_half_values', type=str, default='2,3',
                        help='Comma-separated n_half values (default: 2,3)')
    parser.add_argument('--c_target', type=float, default=1.30,
                        help='Target lower bound (default: 1.30)')
    parser.add_argument('--sample_small', type=int, default=5000,
                        help='Sample size for m < 30 (default: 5000)')
    parser.add_argument('--sample_large', type=int, default=1000,
                        help='Sample size for m >= 30 (default: 1000)')
    parser.add_argument('--max_survivors', type=int, default=10_000_000,
                        help='Max survivor buffer per level (default: 10M)')
    parser.add_argument('--level_timeout', type=float, default=0.0,
                        help='Per-level time budget in seconds (default: 0=no limit)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory (default: data)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print theoretical quantities only (no GPU)')
    return parser.parse_args()


def main():
    args = parse_args()

    m_values = [int(x) for x in args.m_values.split(',')]
    n_half_values = [int(x) for x in args.n_half_values.split(',')]

    configs = build_sweep_configs(m_values, n_half_values, args.c_target)

    print(f"Benchmark sweep: {len(configs)} configs")
    print(f"  m values: {m_values}")
    print(f"  n_half values: {n_half_values}")
    print(f"  c_target: {args.c_target}")
    print(f"  sample: {args.sample_small:,} (m<30), "
          f"{args.sample_large:,} (m>=30)")
    print(f"  max_survivors: {args.max_survivors:,}")
    if args.level_timeout > 0:
        print(f"  level_timeout: {args.level_timeout:.0f}s")

    if args.dry_run:
        dry_run(configs, args.c_target)
        return

    # Check GPU
    if not wrapper.is_available():
        print("ERROR: No CUDA GPU available")
        sys.exit(1)

    dev = wrapper.get_device_name()
    free_mb = wrapper.get_free_memory() / (1024 * 1024)
    print(f"\nGPU: {dev}")
    print(f"Free memory: {free_mb:.0f} MB")

    results = run_sweep(configs, args.c_target,
                        args.sample_small, args.sample_large,
                        args.max_survivors, args.level_timeout,
                        args.output_dir, args.resume)

    format_summary_table(results)
    save_results(results, args.output_dir)


if __name__ == '__main__':
    main()
