"""F-cascade benchmark sweep — mirrors tests/benchmark_sweep.py exactly,
but uses variant F (Σδ=0 LP-tight correction) at every level.

Style:
  L0: full enumeration via prune_F.
  L1+: sample SAMPLE parents (constant), run each via process_parent_fused
       (W-refined kernel), then apply prune_F as additional filter.
       Record per-level: avg children, avg W-survivors, avg F-survivors,
       expansion factor, wall time.  Stop when:
         - all F-survivors = 0 at this level (PROVEN with F)
         - extrapolated total children > BUDGET (cascade futile)
         - hit max_levels

  No per-parent skip-on-big-children: every sampled parent runs to completion.

Targets c=1.28 by default.
"""
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)

from compositions import generate_compositions_batched
from pruning import count_compositions, correction
from run_cascade import process_parent_fused
from _M1_bench import prune_F

C_UPPER = 1.5029
BUDGET = 160e12  # total children budget across all levels


def run_L0_F(n_half, m, c_target, batch_size=200_000, verbose=True):
    d = 2 * n_half
    S_full = 4 * n_half * m
    S_half = 2 * n_half * m
    n_total_half = count_compositions(n_half, S_half)
    if verbose:
        print(f"\n=== L0 (n_half={n_half}, m={m}, c_target={c_target}) ===")
        print(f"     d={d}, S=4nm={S_full}, palindromic comps={n_total_half:,}")

    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = S_full
    prune_F(warm, n_half, m, c_target)

    t0 = time.time()
    n_proc = 0
    surv_list = []
    last = t0
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=batch_size):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_proc += len(batch)
        s = prune_F(batch, n_half, m, c_target)
        if s.any():
            surv_list.append(batch[s].copy())
        now = time.time()
        if verbose and now - last >= 5.0:
            n_so_far = sum(len(s) for s in surv_list)
            pct = n_proc / n_total_half * 100
            print(f"     [{pct:5.1f}%] {n_proc:,}/{n_total_half:,}  "
                  f"F-surv: {n_so_far:,}", flush=True)
            last = now
    t = time.time() - t0
    surv = np.vstack(surv_list) if surv_list else np.empty((0, d), dtype=np.int32)
    if verbose:
        print(f"     L0 done: {n_proc:,} processed, "
              f"{len(surv):,} F-survivors  [{t:.1f}s]")
    return surv, t


def run_cascade_F(n_half, m, c_target, sample_n=5, max_levels=5,
                   verbose=True, parent_timeout_sec=120.0):
    """L0 with F, then sample-based cascade reporting expansion factor."""
    out = {'n_half': n_half, 'm': m, 'c_target': c_target,
           'sample_n': sample_n, 'levels': [], 'status': 'incomplete'}

    corr_flat = correction(m, n_half)
    if c_target + corr_flat >= C_UPPER:
        print(f"VACUOUS: c+corr_flat={c_target+corr_flat:.4f} >= {C_UPPER}")
        out['status'] = 'VACUOUS'
        return out

    survivors, t_L0 = run_L0_F(n_half, m, c_target, verbose=verbose)
    n_surv = len(survivors)
    out['levels'].append({
        'level': 0, 'd_parent': 2 * n_half, 'n_parents_in': 0,
        'n_F_survivors': n_surv, 'wall_sec': t_L0,
    })
    if n_surv == 0:
        out['status'] = 'PROVEN_L0'
        if verbose:
            print(f"\n*** PROVEN at L0 with F! ***")
        return out

    level = 1
    rng = np.random.default_rng(0)
    while level <= max_levels:
        d_parent = survivors.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_child // 2
        n_parents = len(survivors)

        # Sample CONSTANT N parents per level (no skipping based on children).
        if n_parents > sample_n:
            idx = rng.choice(n_parents, sample_n, replace=False)
            sample = survivors[idx]
        else:
            sample = survivors
        sample_n_actual = len(sample)

        if verbose:
            print(f"\n--- L{level}: d_parent={d_parent} → d_child={d_child}, "
                  f"{n_parents:,} F-parents in, sampling {sample_n_actual} ---")

        t_level = time.time()
        total_children = 0
        total_W = 0
        total_F = 0
        per_parent_stats = []
        next_F_seeds = []
        for i, parent in enumerate(sample):
            t_p = time.time()
            try:
                W_surv, n_ch = process_parent_fused(
                    parent, m, c_target, n_half_child)
            except Exception as e:
                if verbose:
                    print(f"     [{i+1}] error: {e}")
                continue
            t_w = time.time() - t_p

            # F filter on W-surv
            if len(W_surv) > 0:
                f_mask = prune_F(W_surv, n_half_child, m, c_target)
                F_surv = W_surv[f_mask]
            else:
                F_surv = np.empty((0, d_child), dtype=np.int32)
            t_total = time.time() - t_p

            total_children += n_ch
            total_W += len(W_surv)
            total_F += len(F_surv)
            per_parent_stats.append({
                'children': int(n_ch),
                'W_surv': int(len(W_surv)),
                'F_surv': int(len(F_surv)),
                'wall_sec': float(t_total),
            })
            if len(F_surv) > 0 and sum(len(x) for x in next_F_seeds) < sample_n:
                next_F_seeds.append(F_surv.copy())
            if verbose:
                print(f"     [{i+1}/{sample_n_actual}] "
                      f"{n_ch:,} children → W:{len(W_surv):,} "
                      f"→ F:{len(F_surv):,}  [{t_total:.1f}s]", flush=True)

        elapsed = time.time() - t_level
        n_completed = len(per_parent_stats)
        if n_completed == 0:
            out['status'] = f'NO_COMPLETION_L{level}'
            return out

        avg_children = total_children / n_completed
        avg_W = total_W / n_completed
        avg_F = total_F / n_completed
        F_extra = 100.0 * (total_W - total_F) / max(1, total_W)
        # Expansion factor: F-survivors per parent (aka F-survivors generated
        # at this level / parents at previous level).  expansion < 1 means
        # cascade is contracting.
        expansion = avg_F

        if verbose:
            print(f"\n     L{level} sample summary ({elapsed:.1f}s):")
            print(f"       avg children/parent: {avg_children:,.0f}")
            print(f"       avg W-survivors:     {avg_W:,.1f}")
            print(f"       avg F-survivors:     {avg_F:,.1f}  "
                  f"(F prunes {F_extra:.1f}% MORE than W)")
            print(f"       expansion factor (F-surv/parent): {expansion:.2f}x")
            est_total_F = int(expansion * n_parents)
            est_total_chld = int(avg_children * n_parents)
            print(f"       extrapolated total F-survivors: {est_total_F:,}")
            print(f"       extrapolated total children:    {est_total_chld:.2e}")
            print(f"       budget: {est_total_chld/BUDGET*100:.2f}% of "
                  f"{BUDGET:.0e}")

        out['levels'].append({
            'level': level, 'd_parent': d_parent,
            'n_parents_in': n_parents, 'sample_n': sample_n_actual,
            'avg_children_per_parent': avg_children,
            'avg_W_survivors_per_parent': avg_W,
            'avg_F_survivors_per_parent': avg_F,
            'F_extra_prune_pct': F_extra,
            'expansion_factor_F': expansion,
            'est_total_F': int(expansion * n_parents),
            'est_total_children': int(avg_children * n_parents),
            'wall_sec_sample': elapsed,
            'per_parent': per_parent_stats,
        })

        if total_F == 0:
            print(f"\n*** SAMPLE-PROVEN at L{level} with F "
                  f"(W still has {total_W}) ***")
            out['status'] = f'SAMPLE_PROVEN_F_L{level}'
            return out
        if total_W == 0:
            print(f"\n*** SAMPLE-PROVEN at L{level} with W ***")
            out['status'] = f'SAMPLE_PROVEN_W_L{level}'
            return out

        est_total_chld = int(avg_children * n_parents)
        if est_total_chld > BUDGET:
            print(f"     *** BUDGET EXCEEDED ***")
            out['status'] = f'BUDGET_EXCEEDED_L{level}'
            return out

        if not next_F_seeds:
            print(f"     no F survivors found in sample → terminate")
            out['status'] = f'NO_SAMPLED_F_SURVIVORS_L{level}'
            return out
        survivors = np.vstack(next_F_seeds)
        if len(survivors) > sample_n:
            survivors = survivors[:sample_n]
        level += 1

    out['status'] = f'LEVEL_CAP_L{max_levels}'
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_half', type=int, default=2)
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--c_target', type=float, default=1.28)
    ap.add_argument('--sample', type=int, default=5,
                    help='constant # parents to sample per level')
    ap.add_argument('--max_levels', type=int, default=5)
    ap.add_argument('--sweep', action='store_true')
    args = ap.parse_args()

    results = []
    if args.sweep:
        configs = [
            (2, 10), (2, 20), (2, 30), (2, 50),
            (3, 10), (3, 20), (3, 30),
            (4, 10), (4, 20),
            (5, 5), (5, 10),
            (6, 5),
        ]
        for nh, m in configs:
            print(f"\n{'='*70}")
            print(f"### CONFIG (n_half={nh}, m={m}, c_target={args.c_target}, "
                  f"sample={args.sample}) ###")
            res = run_cascade_F(nh, m, args.c_target,
                                 sample_n=args.sample,
                                 max_levels=args.max_levels)
            results.append(res)
    else:
        res = run_cascade_F(args.n_half, args.m, args.c_target,
                             sample_n=args.sample,
                             max_levels=args.max_levels)
        results.append(res)

    out_path = os.path.join(_dir, '_F_benchmark_sweep_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}\nResults saved to {out_path}")
    print(f"\nSummary (status @ last level, expansion factors):")
    for r in results:
        nh, m, c = r['n_half'], r['m'], r['c_target']
        last = r['levels'][-1]
        exps = [f"L{lv['level']}:{lv.get('expansion_factor_F', '?'):.1f}x"
                if isinstance(lv.get('expansion_factor_F'), float)
                else f"L{lv['level']}:{lv.get('n_F_survivors', '?')}surv"
                for lv in r['levels']]
        print(f"  (n={nh}, m={m}, c={c}): {r['status']}  | "
              f"{', '.join(exps)}")


if __name__ == '__main__':
    main()
