"""Full cascade runs with capped parents per level.

Runs the COMPLETE cascade (L0 through convergence or L6) for every
non-vacuous (m, n_half) config at c_target=1.40.

At each level, if survivors exceed MAX_PARENTS, we randomly sample
down to MAX_PARENTS before proceeding. This gives us the true
expansion trajectory through all levels.
"""
import math
import os
import sys
import time
import numpy as np

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction, count_compositions
from run_cascade import (run_level0, process_parent_fused,
                         _canonicalize_inplace, _fast_dedup)

C_TARGET = 1.40
C_UPPER_BOUND = 1.5029
MAX_PARENTS = 2000      # cap per level
MAX_LEVELS = 8
TIME_PER_LEVEL = 300    # 5 min max per level

RNG = np.random.default_rng(12345)


def run_one_level(parents, m, c_target, n_half_child, time_budget):
    """Process parents, return (unique_survivors, total_children, n_completed)."""
    d_child = 2 * parents.shape[1]
    all_surv = []
    total_children = 0
    t0 = time.time()

    for i, parent in enumerate(parents):
        if time.time() - t0 > time_budget:
            return all_surv, total_children, i
        surv, nc = process_parent_fused(parent, m, c_target, n_half_child)
        total_children += nc
        if len(surv) > 0:
            all_surv.append(surv)

    return all_surv, total_children, len(parents)


def dedup_survivors(surv_list, d_child):
    """Canonicalize + dedup a list of survivor arrays."""
    if not surv_list:
        return np.empty((0, d_child), dtype=np.int32)
    merged = np.vstack(surv_list)
    _canonicalize_inplace(merged)
    merged = _fast_dedup(merged)
    return merged


def run_full_cascade(n_half, m, c_target, label=""):
    """Run full cascade with parent cap, return list of level results."""
    corr = correction(m)
    threshold = c_target + corr
    if threshold >= C_UPPER_BOUND:
        print(f"  {label}: VACUOUS (threshold={threshold:.4f} >= {C_UPPER_BOUND})")
        return None

    margin = C_UPPER_BOUND - threshold
    d0 = 2 * n_half

    # L0
    t0 = time.time()
    l0 = run_level0(n_half, m, c_target, verbose=False)
    l0_time = time.time() - t0
    n_l0 = l0['n_survivors']

    results = [{
        'level': 0, 'd': d0, 'parents_in': '-',
        'survivors': n_l0, 'expansion': '-',
        'children_tested': l0['n_processed'],
        'time': l0_time, 'sampled': False,
    }]

    if n_l0 == 0:
        return results

    current = l0['survivors']

    for level in range(1, MAX_LEVELS + 1):
        n_parents = len(current)
        if n_parents == 0:
            results.append({
                'level': level, 'd': current.shape[1] * 2 if len(current) > 0 else 0,
                'parents_in': 0, 'survivors': 0,
                'expansion': 0, 'children_tested': 0,
                'time': 0, 'sampled': False,
            })
            break

        d_parent = current.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_child // 2

        # Sample if needed
        sampled = False
        if n_parents > MAX_PARENTS:
            idx = RNG.choice(n_parents, MAX_PARENTS, replace=False)
            sample = current[idx]
            sampled = True
        else:
            sample = current

        t0 = time.time()
        surv_list, total_children, n_completed = run_one_level(
            sample, m, c_target, n_half_child, TIME_PER_LEVEL)
        elapsed = time.time() - t0

        unique_surv = dedup_survivors(surv_list, d_child)
        n_surv = len(unique_surv)

        # Expansion: survivors per parent completed
        if n_completed > 0:
            surv_per_parent = n_surv / n_completed
            # Project to full parent set
            projected_surv = int(surv_per_parent * n_parents)
            expansion = surv_per_parent
        else:
            surv_per_parent = 0
            projected_surv = 0
            expansion = 0

        timed_out = n_completed < len(sample)

        results.append({
            'level': level, 'd': d_child,
            'parents_in': n_parents,
            'parents_processed': n_completed,
            'survivors_raw': sum(len(s) for s in surv_list),
            'survivors_unique': n_surv,
            'projected_full': projected_surv,
            'expansion': expansion,
            'children_tested': total_children,
            'children_per_parent': total_children / max(n_completed, 1),
            'time': elapsed,
            'sampled': sampled,
            'timed_out': timed_out,
        })

        if projected_surv == 0 or n_surv == 0:
            break

        # Next level: use actual survivors (capped)
        current = unique_surv

    return results


def print_cascade(results, label):
    """Pretty-print one cascade result."""
    print(f"\n  {label}")
    print(f"  {'Lvl':>3} {'d':>4} | {'parents_in':>10} | {'tested':>10} | "
          f"{'surv(unique)':>12} | {'projected':>10} | {'exp/parent':>10} | "
          f"{'children/p':>10} | {'time':>6} | notes")
    print(f"  " + "-" * 110)

    for r in results:
        lvl = r['level']
        d = r['d']
        if lvl == 0:
            print(f"  L{lvl:>1} {d:>4} | {'':>10} | {r['children_tested']:>10,} | "
                  f"{r['survivors']:>12,} | {'':>10} | {'':>10} | "
                  f"{'':>10} | {r['time']:>5.1f}s |")
            continue

        pin = r['parents_in']
        proc = r.get('parents_processed', pin)
        su = r.get('survivors_unique', r.get('survivors', 0))
        proj = r.get('projected_full', su)
        exp = r.get('expansion', 0)
        cpp = r.get('children_per_parent', 0)
        t = r['time']
        notes = []
        if r.get('sampled'):
            notes.append(f"sampled {proc}/{pin}")
        if r.get('timed_out'):
            notes.append("TIMEOUT")
        note_str = ", ".join(notes)

        print(f"  L{lvl:>1} {d:>4} | {pin:>10,} | {proc:>10,} | "
              f"{su:>12,} | {proj:>10,} | {exp:>10.1f} | "
              f"{cpp:>10,.0f} | {t:>5.1f}s | {note_str}")


# =========================================================================
# RUN ALL CONFIGS
# =========================================================================

print("=" * 120)
print(f"FULL CASCADE ANALYSIS: c_target = {C_TARGET}, MAX_PARENTS = {MAX_PARENTS}")
print(f"Upper bound: {C_UPPER_BOUND}, Time budget/level: {TIME_PER_LEVEL}s")
print("=" * 120)

configs = [
    (2, 20, "n=2, m=20 (main config)"),
    (2, 25, "n=2, m=25"),
    (2, 30, "n=2, m=30"),
    (2, 40, "n=2, m=40"),
    (2, 50, "n=2, m=50"),
    (3, 20, "n=3, m=20"),
    (3, 25, "n=3, m=25"),
    (3, 30, "n=3, m=30"),
]

all_results = {}

for n_half, m, label in configs:
    print(f"\n{'='*80}")
    print(f"  {label}: n_half={n_half}, m={m}, d0={2*n_half}")
    corr = correction(m)
    print(f"  correction={corr:.6f}, threshold={C_TARGET+corr:.6f}, "
          f"margin={C_UPPER_BOUND-C_TARGET-corr:.6f}")
    print(f"{'='*80}")

    results = run_full_cascade(n_half, m, C_TARGET, label)
    if results is not None:
        all_results[(n_half, m)] = results
        print_cascade(results, label)

        # Check convergence
        last = results[-1]
        proj = last.get('projected_full', last.get('survivors', -1))
        if proj == 0:
            print(f"\n  >>> CONVERGES at L{last['level']} <<<")
        else:
            print(f"\n  >>> DID NOT CONVERGE by L{last['level']} "
                  f"(projected {proj:,} survivors) <<<")


# =========================================================================
# COMPARISON TABLE
# =========================================================================

print("\n\n" + "=" * 120)
print("COMPARISON: Expansion factor trajectory for each config")
print("=" * 120)

# Header
levels_to_show = range(0, MAX_LEVELS + 1)
header = f"{'Config':>20} |"
for l in levels_to_show:
    header += f" {'L'+str(l):>10} |"
header += " Converges?"
print(header)
print("-" * len(header))

for (n_half, m), results in all_results.items():
    label = f"n={n_half}, m={m}"
    row = f"{label:>20} |"

    # Build lookup
    by_level = {r['level']: r for r in results}

    for l in levels_to_show:
        if l in by_level:
            r = by_level[l]
            if l == 0:
                row += f" {r['survivors']:>10,} |"
            else:
                exp = r.get('expansion', 0)
                proj = r.get('projected_full', 0)
                if proj == 0 and l > 0:
                    row += f" {'0 (DONE)':>10} |"
                else:
                    row += f" {exp:>9.1f}x |"
        else:
            row += f" {'':>10} |"

    last = results[-1]
    proj = last.get('projected_full', last.get('survivors', -1))
    if proj == 0:
        row += f" L{last['level']}"
    else:
        row += f" NO"
    print(row)


print("\n\nLEGEND:")
print("  L0 column = number of survivors")
print("  L1+ columns = expansion factor (survivors per parent)")
print("  '0 (DONE)' = 0 survivors, proof complete at this level")
