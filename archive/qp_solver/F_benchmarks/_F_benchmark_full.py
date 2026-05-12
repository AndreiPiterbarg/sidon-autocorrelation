"""F-cascade benchmark — full L1, sampled L2+.

For each (n_half, m, c_target):
  L0: full enumeration with prune_F.  All F-survivors counted.
  L1: process EVERY L0 F-survivor through process_parent_fused (W-refined
      kernel) + prune_F filter.  Total L1 F-survivors fully counted.
      ⇒ if total L1 F-survivors == 0, cascade is PROVEN at L1 (rigorous).
  L2+: sample SAMPLE_L2 parents, expansion factor only.

If full L1 takes longer than --max_l1_minutes, fall back to sampling
SAMPLE_L1 parents at L1 and clearly mark the run as "L1 sampled".

Targets c=1.28 by default; sweep over (n_half, m) to find any config that
fully converges at L1.
"""
import os, sys, time, json
import numpy as np
import multiprocessing as mp

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)

from compositions import generate_compositions_batched
from pruning import count_compositions, correction
from run_cascade import process_parent_fused
from _M1_bench import prune_F

C_UPPER = 1.5029


# ---- Multiprocessing worker (importable, picklable) ----
_WORKER_CTX = {}

def _worker_init(m, c_target, n_half_child):
    _WORKER_CTX['m'] = m
    _WORKER_CTX['c_target'] = c_target
    _WORKER_CTX['n_half_child'] = n_half_child

def _worker_one_parent(parent):
    m = _WORKER_CTX['m']
    c_target = _WORKER_CTX['c_target']
    n_half_child = _WORKER_CTX['n_half_child']
    d_parent = len(parent)
    d_child = 2 * d_parent
    t = time.time()
    try:
        W_surv, n_ch = process_parent_fused(parent, m, c_target, n_half_child)
    except Exception as e:
        return (-1, 0, 0, np.empty((0, d_child), dtype=np.int32), 0.0, str(e))
    if len(W_surv) > 0:
        f_mask = prune_F(W_surv, n_half_child, m, c_target)
        F_surv = W_surv[f_mask].copy()
    else:
        F_surv = np.empty((0, d_child), dtype=np.int32)
    return (int(n_ch), int(len(W_surv)), int(len(F_surv)), F_surv,
            time.time()-t, None)


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


def process_one_parent(parent, m, c_target, n_half_child):
    """Run W-refined kernel + F filter on one parent.  Returns
    (n_children, len(W_surv), len(F_surv), F_surv array, wall_sec)."""
    d_parent = len(parent)
    d_child = 2 * d_parent
    t = time.time()
    try:
        W_surv, n_ch = process_parent_fused(parent, m, c_target, n_half_child)
    except Exception as e:
        return -1, 0, 0, np.empty((0, d_child), dtype=np.int32), 0.0, str(e)
    if len(W_surv) > 0:
        f_mask = prune_F(W_surv, n_half_child, m, c_target)
        F_surv = W_surv[f_mask].copy()
    else:
        F_surv = np.empty((0, d_child), dtype=np.int32)
    return int(n_ch), int(len(W_surv)), int(len(F_surv)), F_surv, time.time()-t, None


def run_L1_full(L0_F_survivors, m, c_target, max_minutes=10.0, verbose=True,
                 n_workers=None):
    """Process every L0 F-survivor at L1 in PARALLEL across workers."""
    n_parents = len(L0_F_survivors)
    if n_parents == 0:
        return None, np.empty((0, 0), dtype=np.int32)

    d_parent = L0_F_survivors.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    if n_workers is None:
        n_workers = min(64, max(1, mp.cpu_count() // 4))
    n_workers = min(n_workers, n_parents)

    if verbose:
        print(f"\n--- L1 FULL: {n_parents:,} parents, "
              f"d_parent={d_parent} → d_child={d_child}, "
              f"workers={n_workers} ---")

    t0 = time.time()
    total_children = 0
    total_W = 0
    total_F = 0
    L1_F_survivors_list = []
    n_done = 0
    last_report = t0
    deadline = t0 + 60.0 * max_minutes
    timed_out = False
    errors = 0

    parents_list = [L0_F_survivors[i] for i in range(n_parents)]

    with mp.Pool(n_workers,
                  initializer=_worker_init,
                  initargs=(m, c_target, n_half_child)) as pool:
        for i, (n_ch, n_W, n_F, F_surv, t_p, err) in enumerate(
                pool.imap_unordered(_worker_one_parent, parents_list,
                                    chunksize=4)):
            if err:
                errors += 1
                continue
            n_done += 1
            total_children += n_ch
            total_W += n_W
            total_F += n_F
            if n_F > 0:
                L1_F_survivors_list.append(F_surv)
            now = time.time()
            if verbose and now - last_report >= 5.0:
                print(f"     [{n_done}/{n_parents}] cum: ch={total_children:,}, "
                      f"W:{total_W:,}, F:{total_F:,}  [{now-t0:.0f}s]",
                      flush=True)
                last_report = now
            if now > deadline:
                print(f"     TIMEOUT at parent {n_done}/{n_parents} "
                      f"({60.0*max_minutes:.0f}s budget exhausted)")
                timed_out = True
                pool.terminate()
                break

    elapsed = time.time() - t0
    if n_done == 0:
        return {'n_parents': n_parents, 'n_done': 0, 'errors': errors,
                'wall_sec': elapsed, 'timed_out': timed_out}, np.empty((0, d_child), dtype=np.int32)

    avg_ch = total_children / n_done
    avg_W = total_W / n_done
    avg_F = total_F / n_done
    F_extra = 100.0 * (total_W - total_F) / max(1, total_W)
    if L1_F_survivors_list:
        L1_F = np.vstack(L1_F_survivors_list)
    else:
        L1_F = np.empty((0, d_child), dtype=np.int32)

    out = {
        'mode': 'full',
        'n_parents': n_parents, 'n_done': n_done, 'errors': errors,
        'timed_out': timed_out,
        'avg_children_per_parent': avg_ch,
        'avg_W_survivors_per_parent': avg_W,
        'avg_F_survivors_per_parent': avg_F,
        'total_children': total_children,
        'total_W_survivors': total_W,
        'total_F_survivors': total_F,
        'F_extra_prune_pct': F_extra,
        'wall_sec': elapsed,
    }
    if verbose:
        print(f"\n     L1 FULL summary ({elapsed:.1f}s, "
              f"{'timed out' if timed_out else 'completed'}):")
        print(f"       parents processed: {n_done}/{n_parents} "
              f"({errors} errors)")
        print(f"       avg children/parent: {avg_ch:,.0f}")
        print(f"       avg W-surv/parent:   {avg_W:,.1f}")
        print(f"       avg F-surv/parent:   {avg_F:,.1f}")
        print(f"       total W-survivors:   {total_W:,}")
        print(f"       total F-survivors:   {total_F:,}")
        print(f"       F prunes {F_extra:.1f}% MORE than W")
        if total_F == 0 and not timed_out:
            print(f"       *** RIGOROUS PROOF at L1 with F ({n_done}/{n_parents} parents fully processed) ***")
    return out, L1_F


def run_L2_sample(L1_F_survivors, m, c_target, sample_n=50, verbose=True):
    """Sample N parents at L2, report expansion."""
    n_parents = len(L1_F_survivors)
    if n_parents == 0:
        return None
    d_parent = L1_F_survivors.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    rng = np.random.default_rng(0)
    if n_parents > sample_n:
        idx = rng.choice(n_parents, sample_n, replace=False)
        sample = L1_F_survivors[idx]
    else:
        sample = L1_F_survivors
    sample_n_actual = len(sample)

    n_workers = min(64, max(1, mp.cpu_count() // 4), sample_n_actual)
    if verbose:
        print(f"\n--- L2 SAMPLE: {sample_n_actual} of {n_parents} parents, "
              f"d_parent={d_parent} → d_child={d_child}, "
              f"workers={n_workers} ---")

    t0 = time.time()
    total_children = 0
    total_W = 0
    total_F = 0
    n_done = 0
    parents_list = [sample[i] for i in range(sample_n_actual)]
    with mp.Pool(n_workers,
                  initializer=_worker_init,
                  initargs=(m, c_target, n_half_child)) as pool:
        for i, (n_ch, n_W, n_F, _F_surv, t_p, err) in enumerate(
                pool.imap_unordered(_worker_one_parent, parents_list,
                                    chunksize=2)):
            if err:
                continue
            n_done += 1
            total_children += n_ch
            total_W += n_W
            total_F += n_F
            if verbose and (i+1) % 25 == 0:
                print(f"     [{i+1}/{sample_n_actual}] cum: "
                      f"ch={total_children:.2e}, W:{total_W:,}, F:{total_F:,}",
                      flush=True)
    elapsed = time.time() - t0

    if n_done == 0:
        return {'mode': 'sample', 'n_parents': n_parents,
                'sample_n_target': sample_n, 'n_done': 0,
                'wall_sec': elapsed}

    avg_ch = total_children / n_done
    avg_F = total_F / n_done
    avg_W = total_W / n_done
    F_extra = 100.0 * (total_W - total_F) / max(1, total_W)
    expansion = avg_F  # F-survivors per parent

    if verbose:
        print(f"\n     L2 SAMPLE summary ({elapsed:.1f}s):")
        print(f"       avg children/parent: {avg_ch:,.0f}")
        print(f"       avg W-surv/parent:   {avg_W:,.1f}")
        print(f"       avg F-surv/parent:   {avg_F:,.1f}")
        print(f"       expansion factor:    {expansion:.2f}x")
        if total_F == 0:
            print(f"       *** sample-proven (W={total_W} too) — "
                  f"NOT a rigorous proof, only {sample_n_actual}/{n_parents} parents ***")

    return {
        'mode': 'sample',
        'n_parents': n_parents,
        'sample_n_target': sample_n, 'sample_n_actual': sample_n_actual,
        'n_done': n_done,
        'avg_children_per_parent': avg_ch,
        'avg_W_survivors_per_parent': avg_W,
        'avg_F_survivors_per_parent': avg_F,
        'F_extra_prune_pct': F_extra,
        'expansion_factor': expansion,
        'extrapolated_total_F': int(expansion * n_parents),
        'wall_sec': elapsed,
    }


def run_one_config(n_half, m, c_target, max_l1_minutes=10.0, sample_l2=50,
                   verbose=True):
    out = {'n_half': n_half, 'm': m, 'c_target': c_target,
           'L0': None, 'L1': None, 'L2': None, 'status': 'incomplete'}

    corr_flat = correction(m, n_half)
    if c_target + corr_flat >= C_UPPER:
        out['status'] = 'VACUOUS'
        return out

    L0_surv, t_L0 = run_L0_F(n_half, m, c_target, verbose=verbose)
    out['L0'] = {'n_F_survivors': len(L0_surv), 'wall_sec': t_L0}
    if len(L0_surv) == 0:
        out['status'] = 'PROVEN_L0'
        if verbose:
            print(f"\n*** RIGOROUS PROOF at L0 with F ***")
        return out

    # L1 full
    L1_stats, L1_F = run_L1_full(L0_surv, m, c_target,
                                   max_minutes=max_l1_minutes, verbose=verbose)
    out['L1'] = L1_stats
    if L1_stats is None or L1_stats.get('n_done', 0) == 0:
        out['status'] = 'L1_FAILED'
        return out

    if L1_stats.get('timed_out'):
        out['status'] = f'L1_TIMEOUT_after_{L1_stats["n_done"]}/{L1_stats["n_parents"]}'
        return out

    if L1_stats.get('total_F_survivors', 0) == 0:
        out['status'] = 'PROVEN_L1_FULL'
        return out

    # L2 sample
    L2_stats = run_L2_sample(L1_F, m, c_target, sample_n=sample_l2,
                              verbose=verbose)
    out['L2'] = L2_stats
    if L2_stats and L2_stats.get('expansion_factor', 1.0) == 0:
        out['status'] = 'L2_SAMPLE_PROVEN_NOT_FULL'
    else:
        out['status'] = 'L2_HAS_SURVIVORS'
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--c_target', type=float, default=1.28)
    ap.add_argument('--max_l1_minutes', type=float, default=10.0)
    ap.add_argument('--sample_l2', type=int, default=50)
    args = ap.parse_args()

    # Configurations with non-trivial L0 size where full L1 is plausible.
    # L0 comp counts (palindromic):
    #   (3, 30):  16K   (3, 50):  46K   (3, 100): 181K
    #   (4, 10):  92K   (4, 20):  716K  (4, 30):  2.4M
    #   (5, 10):  4.6M  (5, 5):   316K
    #   (6, 5):   8.3M
    configs = [
        (3, 30), (3, 50), (3, 100),
        (4, 10), (4, 20),
        (5, 5), (5, 10),
    ]

    results = []
    for nh, m in configs:
        print(f"\n{'='*70}")
        print(f"### CONFIG (n_half={nh}, m={m}, c_target={args.c_target}) ###")
        res = run_one_config(nh, m, args.c_target,
                              max_l1_minutes=args.max_l1_minutes,
                              sample_l2=args.sample_l2)
        results.append(res)

    out_path = os.path.join(_dir, '_F_benchmark_full_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}\nResults saved to {out_path}")
    print(f"\nSummary:")
    for r in results:
        nh, m, c = r['n_half'], r['m'], r['c_target']
        l0 = r['L0']['n_F_survivors'] if r.get('L0') else '?'
        l1 = r['L1']
        l2 = r['L2']
        l1_str = (f"L1full:{l1['total_F_survivors']:,}F-surv/{l1['n_parents']:,}p"
                  if l1 and l1.get('total_F_survivors') is not None
                  else 'L1?')
        l2_str = (f"L2sample-exp:{l2['expansion_factor']:.1f}x"
                  if l2 else '')
        print(f"  (n={nh},m={m}): {r['status']:30s} L0:{l0}  {l1_str}  {l2_str}")


if __name__ == '__main__':
    main()
