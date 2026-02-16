"""Hierarchical multi-level branch-and-prune driver.

Chains base-level enumeration (n=3, d=6) with refinement levels
(n=6 -> n=12 -> n=24), extracting survivors at each level and
refining them at the next.

Usage:
    from gpu.multilevel import hierarchical_prove
    result = hierarchical_prove(c_target=1.20, n_base=3, m=50)
"""
import numpy as np
import time
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cpu'))
from pruning import correction, count_compositions, asymmetry_threshold
from gpu import wrapper
from gpu.solvers import gpu_run_single_level
from gpu.wrapper import load_survivors_chunk


def _checkpoint_path(checkpoint_dir, level):
    """Get checkpoint file paths for a given level."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    meta_path = os.path.join(checkpoint_dir, f'level_{level}_meta.json')
    surv_path = os.path.join(checkpoint_dir, f'level_{level}_survivors.npy')
    return meta_path, surv_path


def save_checkpoint(checkpoint_dir, level, survivors, metadata):
    """Save checkpoint for a completed level."""
    meta_path, surv_path = _checkpoint_path(checkpoint_dir, level)
    np.save(surv_path, survivors)
    metadata['survivor_file'] = surv_path
    metadata['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_checkpoint(checkpoint_dir, level):
    """Load checkpoint for a given level, or return None."""
    meta_path, surv_path = _checkpoint_path(checkpoint_dir, level)
    if not os.path.exists(meta_path) or not os.path.exists(surv_path):
        return None
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    survivors = np.load(surv_path)
    return {'metadata': metadata, 'survivors': survivors}


def hierarchical_prove(c_target, n_base=3, m=50, max_levels=4,
                       time_budget=7000, checkpoint_dir='data/',
                       max_survivors=10000000, verbose=True,
                       resume=False):
    """Hierarchical branch-and-prune: n_base -> 2*n_base -> 4*n_base -> ...

    Parameters
    ----------
    c_target : float
        Target lower bound to prove (e.g. 1.20).
    n_base : int
        Base level n (default 3, giving d=6).
    m : int
        Grid resolution (default 50).
    max_levels : int
        Maximum number of refinement levels (default 4: n=3,6,12,24).
    time_budget : float
        Total time budget in seconds (default 7000s ~ 2 hours).
    checkpoint_dir : str
        Directory for checkpoint files.
    max_survivors : int
        Max survivors to extract per level.
    verbose : bool
        Print progress.
    resume : bool
        If True, attempt to resume from the last completed checkpoint.

    Returns
    -------
    dict with keys:
        proven : bool
        c_target : float
        c_proven : float or None
        levels_completed : int
        remaining_survivors : int
        total_elapsed : float
        level_results : list of dicts
    """
    corr = correction(m)
    prune_target = c_target + corr

    if verbose:
        dev = wrapper.get_device_name()
        print(f"{'='*60}")
        print(f"Hierarchical Branch-and-Prune")
        print(f"{'='*60}")
        print(f"  Target: c >= {c_target:.6f}")
        print(f"  Base level: n={n_base}, m={m}")
        print(f"  Correction: {corr:.6f}")
        print(f"  Prune threshold: {prune_target:.6f}")
        print(f"  Max levels: {max_levels} (n = {', '.join(str(n_base * 2**i) for i in range(max_levels))})")
        print(f"  Time budget: {time_budget:.0f}s")
        print(f"  Device: {dev}")
        print()

    t_start = time.time()
    level_results = []
    survivors = None
    survivor_file = None  # binary file path when using streamed mode
    start_level = 0

    # Check for resume
    if resume:
        for lvl in range(max_levels - 1, -1, -1):
            ckpt = load_checkpoint(checkpoint_dir, lvl)
            if ckpt is not None:
                metadata = ckpt['metadata']
                if (metadata.get('c_target') == c_target and
                    metadata.get('m') == m and
                    metadata.get('n_base') == n_base):
                    survivors = ckpt['survivors']
                    start_level = lvl + 1
                    if verbose:
                        print(f"  Resuming from level {lvl} checkpoint "
                              f"({len(survivors)} survivors)")
                    break

    for level in range(start_level, max_levels):
        n_half = n_base * (2 ** level)
        d = 2 * n_half

        elapsed_so_far = time.time() - t_start
        remaining_budget = time_budget - elapsed_so_far
        if remaining_budget <= 0:
            if verbose:
                print(f"\n  Time budget exhausted before level {level}")
            break

        if verbose:
            print(f"{'─'*60}")
            print(f"Level {level}: n={n_half}, d={d}")
            print(f"  Time remaining: {remaining_budget:.0f}s")

        t_level = time.time()

        if level == 0:
            # Base level: full enumeration with survivor extraction
            # Auto-selects streamed mode for large runs (D>=6, m>=20)
            result = gpu_run_single_level(
                n_half, m, c_target,
                verbose=verbose,
                extract_survivors=True)

            use_streamed = result['stats'].get('streamed', False)

            level_info = {
                'level': level,
                'n_half': n_half,
                'd': d,
                'type': 'base_enumeration',
                'n_pruned_asym': result['stats']['n_pruned_asym'],
                'n_pruned_test': result['stats']['n_pruned_test'],
                'n_survivors': result['n_survivors'],
                'n_extracted': result['stats'].get('n_extracted', 0),
                'min_test_val': result['min_test_val'],
                'streamed': use_streamed,
                'elapsed': time.time() - t_level,
            }

            if result['proven']:
                if verbose:
                    print(f"\n  PROVEN at base level! c >= {c_target:.6f}")
                level_results.append(level_info)
                return {
                    'proven': True,
                    'c_target': c_target,
                    'c_proven': c_target,
                    'levels_completed': level + 1,
                    'remaining_survivors': 0,
                    'total_elapsed': time.time() - t_start,
                    'level_results': level_results,
                }

            if use_streamed and result.get('survivor_file'):
                survivors = None
                survivor_file = result['survivor_file']
            else:
                survivors = result['survivors']
                survivor_file = None
        else:
            # Refinement level: process previous level's survivors
            use_file_mode = (survivors is None and survivor_file is not None)

            if not use_file_mode and (survivors is None or len(survivors) == 0):
                if verbose:
                    print(f"  No survivors to refine — proven!")
                level_results.append({
                    'level': level,
                    'n_half': n_half,
                    'd': d,
                    'type': 'refinement',
                    'n_survivors': 0,
                    'elapsed': 0,
                })
                return {
                    'proven': True,
                    'c_target': c_target,
                    'c_proven': c_target,
                    'levels_completed': level + 1,
                    'remaining_survivors': 0,
                    'total_elapsed': time.time() - t_start,
                    'level_results': level_results,
                }

            d_parent = d // 2

            if use_file_mode:
                import os as _os
                file_size = _os.path.getsize(survivor_file)
                per_surv = d_parent * 4
                num_parents = file_size // per_surv

                if verbose:
                    print(f"  Parents: {num_parents:,} (from file: {survivor_file})")
                    print(f"  Mode: CHUNKED FILE REFINEMENT")

                REFINE_BATCH = 20_000_000  # 20M parents * 24 bytes = 480 MB
                total_asym = 0
                total_test = 0
                total_survived = 0
                total_ref_extracted = 0
                all_child_survivors = []
                timed_out = False
                batch_idx = 0

                for start in range(0, num_parents, REFINE_BATCH):
                    end = min(start + REFINE_BATCH, num_parents)
                    chunk = load_survivors_chunk(
                        survivor_file, d_parent, start, end - start)

                    batch_budget = remaining_budget - (time.time() - t_level)
                    if batch_budget <= 30:
                        timed_out = True
                        break

                    batch_result = wrapper.refine_parents(
                        d_parent=d_parent,
                        parent_configs_array=chunk,
                        m=m,
                        c_target=c_target,
                        max_survivors=max_survivors,
                        time_budget_sec=batch_budget)

                    total_asym += batch_result['total_asym']
                    total_test += batch_result['total_test']
                    total_survived += batch_result['total_survivors']
                    total_ref_extracted += batch_result['n_extracted']

                    if batch_result['n_extracted'] > 0:
                        all_child_survivors.append(
                            batch_result['survivor_configs'])

                    if verbose:
                        print(f"    Batch {batch_idx}: [{start:,}, {end:,}) "
                              f"surv={batch_result['total_survivors']:,} "
                              f"cum={total_survived:,}")

                    if batch_result['timed_out']:
                        timed_out = True
                        break
                    batch_idx += 1

                if all_child_survivors:
                    child_survivors = np.concatenate(
                        all_child_survivors, axis=0)
                else:
                    child_survivors = np.empty((0, d), dtype=np.int32)

                level_info = {
                    'level': level,
                    'n_half': n_half,
                    'd': d,
                    'type': 'refinement_chunked',
                    'num_parents': num_parents,
                    'n_pruned_asym': total_asym,
                    'n_pruned_test': total_test,
                    'n_survivors': total_survived,
                    'n_extracted': total_ref_extracted,
                    'timed_out': timed_out,
                    'batches': batch_idx + 1,
                    'elapsed': time.time() - t_level,
                }

                n_survived_level = total_survived

            else:
                num_parents = len(survivors)

                if verbose:
                    total_refs = 0
                    for p in range(num_parents):
                        N = 1
                        for i in range(d_parent):
                            N *= (2 * int(survivors[p, i]) + 1)
                        total_refs += N
                    print(f"  Parents: {num_parents:,}")
                    print(f"  Total refinements: {total_refs:,.0f}")

                ref_result = wrapper.refine_parents(
                    d_parent=d_parent,
                    parent_configs_array=survivors,
                    m=m,
                    c_target=c_target,
                    max_survivors=max_survivors,
                    time_budget_sec=remaining_budget)

                level_info = {
                    'level': level,
                    'n_half': n_half,
                    'd': d,
                    'type': 'refinement',
                    'num_parents': num_parents,
                    'n_pruned_asym': ref_result['total_asym'],
                    'n_pruned_test': ref_result['total_test'],
                    'n_survivors': ref_result['total_survivors'],
                    'n_extracted': ref_result['n_extracted'],
                    'min_test_val': ref_result['min_test_val'],
                    'timed_out': ref_result['timed_out'],
                    'elapsed': time.time() - t_level,
                }

                child_survivors = ref_result['survivor_configs']
                n_survived_level = ref_result['total_survivors']
                timed_out = ref_result['timed_out']

            if verbose:
                print(f"\n  Level {level} results:")
                print(f"    Asym pruned: {level_info['n_pruned_asym']:,}")
                print(f"    Test pruned: {level_info['n_pruned_test']:,}")
                print(f"    Survivors: {n_survived_level:,}")
                if level_info['n_extracted'] > 0:
                    print(f"    Extracted: {level_info['n_extracted']:,}")
                print(f"    Elapsed: {time.time() - t_level:.1f}s")

            if n_survived_level == 0:
                if verbose:
                    print(f"\n  >>> PROVEN: c >= {c_target:.6f} <<<")
                level_results.append(level_info)
                return {
                    'proven': True,
                    'c_target': c_target,
                    'c_proven': c_target,
                    'levels_completed': level + 1,
                    'remaining_survivors': 0,
                    'total_elapsed': time.time() - t_start,
                    'level_results': level_results,
                }

            survivors = child_survivors
            survivor_file = None  # clear file mode for next level

            if timed_out:
                if verbose:
                    print(f"  WARNING: Level {level} timed out with "
                          f"{n_survived_level} survivors remaining")

        level_results.append(level_info)

        # Save checkpoint
        save_checkpoint(checkpoint_dir, level, survivors, {
            'c_target': c_target,
            'm': m,
            'n_base': n_base,
            'level': level,
            'n_half': n_half,
            'd': d,
            'n_survivors': len(survivors),
            **{k: v for k, v in level_info.items() if k != 'level'},
        })

        if verbose:
            print(f"  Checkpoint saved: {len(survivors)} survivors")

    total_elapsed = time.time() - t_start
    n_remaining = len(survivors) if survivors is not None else 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"Hierarchical proof {'COMPLETE' if n_remaining == 0 else 'INCOMPLETE'}")
        print(f"  Levels completed: {len(level_results)}")
        print(f"  Remaining survivors: {n_remaining:,}")
        print(f"  Total time: {total_elapsed:.1f}s")
        print(f"{'='*60}")

    return {
        'proven': n_remaining == 0,
        'c_target': c_target,
        'c_proven': c_target if n_remaining == 0 else None,
        'levels_completed': len(level_results),
        'remaining_survivors': n_remaining,
        'total_elapsed': total_elapsed,
        'level_results': level_results,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Hierarchical branch-and-prune')
    parser.add_argument('--target', type=float, default=1.10,
                        help='Target lower bound (default: 1.10)')
    parser.add_argument('--n-base', type=int, default=3,
                        help='Base level n (default: 3)')
    parser.add_argument('--m', type=int, default=50,
                        help='Grid resolution (default: 50)')
    parser.add_argument('--max-levels', type=int, default=4,
                        help='Maximum refinement levels (default: 4)')
    parser.add_argument('--time-budget', type=float, default=7000,
                        help='Time budget in seconds (default: 7000)')
    parser.add_argument('--checkpoint-dir', type=str, default='data/',
                        help='Checkpoint directory (default: data/)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    args = parser.parse_args()

    result = hierarchical_prove(
        c_target=args.target,
        n_base=args.n_base,
        m=args.m,
        max_levels=args.max_levels,
        time_budget=args.time_budget,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume)

    # Machine-readable summary
    summary = {
        'proven': result['proven'],
        'c_target': args.target,
        'c_proven': result['c_proven'],
        'n_base': args.n_base,
        'm': args.m,
        'levels_completed': result['levels_completed'],
        'remaining_survivors': result['remaining_survivors'],
        'total_elapsed': result['total_elapsed'],
    }
    summary_path = os.path.join(args.checkpoint_dir, 'run_summary.json')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")
