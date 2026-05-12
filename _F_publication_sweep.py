"""F-sweep: find max c_target such that variant F prunes all L0 compositions.

If max c > 1.2802 anywhere => new published lower bound on C_{1a}.

Strategy:
  For each (n_half, m), run F over the palindromic L0 composition space
  at a grid of c_target values, find the highest c giving 0 survivors.
"""
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)  # so we can import _M1_bench

from compositions import generate_compositions_batched
from pruning import count_compositions
from _M1_bench import prune_F


def n_palindromic_comps(n_half, m):
    return count_compositions(n_half, 2 * n_half * m)


def find_max_c(n_half, m, c_grid, batch_size=200_000, verbose=True):
    """For each c in c_grid (sorted ascending), run F at L0.
    Return list of (c, n_survivors, t_sec).
    Stop early if survivors > some threshold and they're growing."""
    d = 2 * n_half
    S_half = 2 * n_half * m
    n_total = n_palindromic_comps(n_half, m)

    # Warm up JIT once per (n_half, m)
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * n_half * m  # any valid comp
    prune_F(warm, n_half, m, c_grid[0])

    results = []
    for c in c_grid:
        t0 = time.time()
        n_surv = 0
        n_proc = 0
        for half_batch in generate_compositions_batched(n_half, S_half,
                                                          batch_size=batch_size):
            batch = np.empty((len(half_batch), d), dtype=np.int32)
            batch[:, :n_half] = half_batch
            batch[:, n_half:] = half_batch[:, ::-1]
            n_proc += len(batch)
            sF = prune_F(batch, n_half, m, c)
            n_surv += int(sF.sum())
        elapsed = time.time() - t0
        results.append((float(c), int(n_surv), float(elapsed)))
        if verbose:
            print(f"  c={c:.4f}  surv={n_surv:>8d} / {n_proc:,}  "
                  f"[{elapsed:.2f}s]")
    return results


def main():
    # Configurations to try (n_half, m).
    # Composition counts (palindromic, half summing to 2nm):
    #   (3, 50): 46K   (3, 100): 182K  (3, 200): 723K
    #   (4, 20): 716K  (4, 30): 2.4M   (4, 50): 11M   (4, 100): 86M
    #   (5, 10): 4.6M  (5, 20): 27M    (5, 30): 350M (slow)
    #   (6, 5): 350K   (6, 10): 4.7M   (6, 20): big
    #   (7, 5): 2.4M   (7, 10): 12M
    #   (8, 5): 32K    (8, 10): 6M

    # c values to test.  Focus on clearing 1.2802 = the published bound.
    c_grid = [1.28, 1.281, 1.282, 1.285, 1.290, 1.295, 1.300,
               1.310, 1.320, 1.350, 1.400, 1.450, 1.500]

    configs = [
        (3, 50), (3, 100), (3, 200),
        (4, 30), (4, 50), (4, 100),
        (5, 10), (5, 20),
        (6, 5), (6, 10),
        (7, 5), (7, 10),
        (8, 5),
    ]

    all_results = {}
    headline = None
    headline_c = -1.0
    overall_t0 = time.time()

    for n_half, m in configs:
        n_total = n_palindromic_comps(n_half, m)
        print(f"\n=== n_half={n_half}, m={m}  "
              f"(palindromic comps: {n_total:,}) ===")
        if n_total > 200_000_000:
            print(f"  SKIP — too many compositions")
            continue
        results = find_max_c(n_half, m, c_grid)
        all_results[f"{n_half}_{m}"] = {
            'n_half': n_half, 'm': m, 'n_total': int(n_total),
            'results': results,
        }
        max_c_zero = -1.0
        for c, n_surv, _t in results:
            if n_surv == 0 and c > max_c_zero:
                max_c_zero = c
        if max_c_zero > 0:
            print(f"  *** MAX c with 0 survivors: {max_c_zero:.4f}")
            if max_c_zero > headline_c:
                headline_c = max_c_zero
                headline = (n_half, m, max_c_zero)
        else:
            print(f"  No c in grid gave 0 survivors at this (n, m).")

    overall_elapsed = time.time() - overall_t0
    print(f"\n{'='*60}")
    print(f"Total wall: {overall_elapsed:.1f}s")
    if headline:
        nh, m, c = headline
        print(f"\n*** HEADLINE: F proves C_{{1a}} >= {c} via L0 at "
              f"(n_half={nh}, m={m})")
        if c > 1.2802:
            print(f"*** EXCEEDS 1.2802 by {c - 1.2802:.4f} — NEW PUBLISHED BOUND")
        else:
            print(f"*** Does NOT exceed 1.2802 (current best published)")
    else:
        print(f"\nNo (n, m) in this sweep gave a 0-survivor proof.")

    out = os.path.join(_dir, '_F_publication_sweep_results.json')
    with open(out, 'w') as f:
        json.dump({
            'configs': all_results,
            'headline': headline,
            'overall_wall_sec': overall_elapsed,
        }, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
