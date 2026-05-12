"""Quick throughput calibration: per-LP / per-SDP times for Q and L at each d.

Method:
   For each d ∈ {8, 12, 16}:
   1. Generate 200 random valid compositions at d.
   2. Time apply_Q_filter_parallel  → t_Q_per_LP = wall / 200.
   3. Time apply_L_filter_parallel  → t_L_per_SDP = wall / 200.
   4. Time F kernel on representative parent at d_parent = d/2.
      Get F-count distribution.
   5. Derive: typical parent wall ≈ F_kernel_wall + F_count * t_Q + Q_count * t_L.
      Q_count ≈ 0.7 * F (from prior data); L_count ≈ 0.85 * Q.

Output: per-d table with t_Q, t_L, and derived parents/30min for typical parents.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.environ.get('CASCADE_ROOT', '/home/ubuntu')
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, ROOT)

from pruning import correction
from run_cascade import process_parent_fused, run_level0
from post_filters import apply_Q_filter_parallel, apply_L_filter_parallel

N_WORKERS = 64
N_LP_BENCH = 200  # random comps to time Q on
N_SDP_BENCH = 200  # random comps to time L on
N_PARENT_F = 30   # parents to time F kernel on


def gen_random_valid_comps(d, m, n_half, c_target, n_target):
    """Generate n_target random integer compositions at d that pass F.

    We sample compositions of length d with sum 4nm, then filter through
    F to get realistic 'F-survivor like' compositions.
    """
    from compositions import generate_canonical_compositions_batched
    from _M1_bench import prune_F
    S = 4 * n_half * m
    print(f"  generating ≥{n_target} F-survivors at d={d}...", flush=True)
    # Warm
    warm = np.zeros((1, d), dtype=np.int32)
    prune_F(warm, n_half, m, c_target)
    survivors_list = []
    n_seen = 0
    t0 = time.time()
    for batch in generate_canonical_compositions_batched(d, S, batch_size=200_000):
        n_seen += len(batch)
        sF = prune_F(batch, n_half, m, c_target)
        idx = np.where(sF)[0]
        if len(idx) > 0:
            survivors_list.extend(batch[idx])
        if len(survivors_list) >= n_target:
            break
        if time.time() - t0 > 60.0:
            print(f"  [time cap] stopped after {n_seen:,} comps; got {len(survivors_list)}", flush=True)
            break
    arr = np.array(survivors_list[:n_target], dtype=np.int32)
    print(f"  → got {len(arr)} comps from {n_seen:,} scanned in {time.time()-t0:.1f}s",
            flush=True)
    return arr


def time_filter(name, fn, batch, *args, **kwargs):
    t0 = time.time()
    out = fn(batch, *args, **kwargs)
    wall = time.time() - t0
    n = len(batch)
    n_surv = len(out) if out is not None else 0
    per_lp = wall / max(n, 1)
    print(f"    {name}: {n_surv}/{n} survive  wall={wall*1000:.0f}ms  per_LP={per_lp*1000:.2f}ms", flush=True)
    return wall, n_surv, per_lp


def time_F_kernel(parents, n_half_child, m, c_target):
    """Time F kernel on given parents at d_child = 2*n_half_child."""
    walls = []
    F_counts = []
    children = []
    for p in parents:
        t0 = time.time()
        surv_F, n_ch = process_parent_fused(
            p, m, c_target, n_half_child,
            use_flat_threshold=False, use_F=True, use_Q=False,
            skip_sdp_cert=True)
        walls.append(time.time() - t0)
        F_counts.append(int(len(surv_F)))
        children.append(int(n_ch))
    return np.array(walls), np.array(F_counts), np.array(children)


def main():
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(ROOT, f'quick_throughput_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'summary.json')

    # (d_child, n_parent, m, c_target)
    cfgs = [
        (8,  2, 20, 1.25),
        (12, 3, 10, 1.25),
        (16, 4, 10, 1.25),
    ]

    full = {'started': ts, 'configs': cfgs, 'results': []}

    for d_child, n_parent, m, c in cfgs:
        d_parent = 2 * n_parent
        print(f"\n========== d_child={d_child} (n_parent={n_parent}, m={m}, c={c}) ==========",
                flush=True)
        if c + correction(m, n_parent) >= 1.5029:
            print(f"  VACUOUS", flush=True)
            full['results'].append({'d_child': d_child, 'verdict': 'VACUOUS'})
            continue

        # 1) Get random "F-survivor-like" comps at d_child to time Q+L
        n_half_child = d_child // 2
        # for the calibration batch we want ~200 comps at d_child;
        # use n_half_child = d_child / 2 as the "n_half" parameter for F.
        m_calib = 10 if d_child >= 12 else m  # smaller m for tractability at high d
        c_calib = c
        if c_calib + correction(m_calib, n_half_child) >= 1.5029:
            m_calib = max(m_calib, 15)  # bump m if needed to avoid vacuity
        if c_calib + correction(m_calib, n_half_child) >= 1.5029:
            m_calib = 20

        comps = gen_random_valid_comps(d_child, m_calib, n_half_child, c_calib,
                                          N_LP_BENCH)
        if len(comps) < 10:
            print(f"  TOO FEW comps; skip d_child={d_child}", flush=True)
            full['results'].append({'d_child': d_child,
                                      'verdict': 'INSUFFICIENT_COMPS'})
            continue

        # 2) Time Q parallel
        wall_Q, n_Q, t_Q = time_filter(
            'Q parallel', apply_Q_filter_parallel,
            comps, n_half_child, m_calib, c_calib, n_workers=N_WORKERS)
        # Actually for Q we want PRUNE rate too; n_Q is survivors.
        q_kill_rate = 1 - n_Q / max(len(comps), 1)

        # 3) Time L parallel on Q-survivors (or full batch if no Q-survivors filtered)
        l_input = comps if n_Q == 0 else (apply_Q_filter_parallel(comps, n_half_child, m_calib, c_calib, n_workers=N_WORKERS) if n_Q > 0 else comps)
        if len(l_input) == 0:
            l_input = comps
        wall_L, n_L, t_L = time_filter(
            'L parallel', apply_L_filter_parallel,
            l_input[:N_SDP_BENCH], n_half_child, m_calib, c_calib,
            solver='MOSEK', n_workers=N_WORKERS)
        l_kill_rate = 1 - n_L / max(len(l_input[:N_SDP_BENCH]), 1)

        # 4) Time F kernel on representative parents from L0 of (n_parent, m).
        try:
            r0 = run_level0(n_parent, m, c, verbose=False, use_F=True, use_Q=False)
            l0_surv = r0['survivors']
        except Exception as e:
            print(f"  L0 enum failed: {e}", flush=True)
            l0_surv = np.zeros((0, d_parent), dtype=np.int32)
        if len(l0_surv) > N_PARENT_F:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(l0_surv), N_PARENT_F, replace=False)
            parents = l0_surv[idx]
        else:
            parents = l0_surv
        if len(parents) > 0:
            print(f"  Timing F kernel on {len(parents)} parents (d_parent={d_parent})...",
                    flush=True)
            walls_F, F_counts, n_children = time_F_kernel(parents, n_half_child, m, c)
            mean_F = float(np.mean(F_counts))
            median_F = int(np.median(F_counts))
            mean_F_wall = float(np.mean(walls_F))
            median_F_wall = float(np.median(walls_F))
            mean_children = float(np.mean(n_children))
        else:
            mean_F = median_F = -1
            mean_F_wall = median_F_wall = -1
            mean_children = -1

        # 5) Derive throughput
        # Per-parent wall ≈ F_wall + mean_F * t_Q + mean_F * (1-q_kill) * t_L
        #                 = F_wall + mean_F * (t_Q + (1-q_kill) * t_L)
        if mean_F >= 0:
            avg_Q_per_parent = mean_F * (1 - q_kill_rate)
            avg_L_per_parent_input = avg_Q_per_parent
            # L_count is what comes out of L (smaller); but L wall is on Q_count input.
            mean_parent_wall_pred = (mean_F_wall +
                                       mean_F * t_Q +
                                       avg_Q_per_parent * t_L)
            parents_30min = int(1800.0 / max(mean_parent_wall_pred, 1e-3))
        else:
            mean_parent_wall_pred = None
            parents_30min = None

        result = {
            'd_child': d_child, 'n_parent': n_parent, 'm': m, 'c_target': c,
            't_Q_per_LP_s': float(t_Q),
            't_L_per_SDP_s': float(t_L),
            'q_kill_rate': float(q_kill_rate),
            'l_kill_rate': float(l_kill_rate),
            'mean_F_per_parent': mean_F,
            'median_F_per_parent': median_F,
            'mean_F_wall_s': mean_F_wall,
            'median_F_wall_s': median_F_wall,
            'mean_children_per_parent': mean_children,
            'predicted_parent_wall_s': mean_parent_wall_pred,
            'predicted_parents_per_30min': parents_30min,
        }
        full['results'].append(result)
        with open(out_path, 'w') as f:
            json.dump(full, f, indent=2, default=str)

    # Print final table
    print("\n\n=========================================================")
    print("QUICK THROUGHPUT TABLE — parents per 30 min (predicted)")
    print("=========================================================")
    print(f"{'d':>4} {'cfg':>14} {'t_Q (ms/LP)':>12} {'t_L (ms/SDP)':>13} "
          f"{'F_med':>8} {'F_wall':>8} {'p_wall_pred':>12} {'p/30min':>10}")
    for r in full['results']:
        if 'verdict' in r:
            print(f"{r['d_child']:>4}  {r['verdict']:>50}")
            continue
        cs = f"n={r['n_parent']}m={r['m']}"
        print(f"{r['d_child']:>4} {cs:>14} {r['t_Q_per_LP_s']*1000:>11.2f} "
              f"{r['t_L_per_SDP_s']*1000:>12.2f} "
              f"{r['median_F_per_parent']:>8} "
              f"{r['median_F_wall_s']:>7.2f}s "
              f"{(r['predicted_parent_wall_s'] or 0):>11.2f}s "
              f"{(r['predicted_parents_per_30min'] or 0):>10,}")

    print(f"\nFull JSON: {out_path}", flush=True)


if __name__ == '__main__':
    main()
