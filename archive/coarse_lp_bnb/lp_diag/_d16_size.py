"""Empirically measure the d=16 cascade layer size at c=1.281 across m values.

For each m ∈ {10, 15, 20, 30}:
  1. Full L0 enum at d=4 (n_half=2) with F+Q post-filter (parallel).
     → measure: N_L0_surv, L0_wall.
  2. Sample S1 = 10 L0 survivors.  For each, F kernel only at d_child=8.
     → measure: avg_children_d8, avg_F_surv_d8 per L0 parent, walls.
  3. Pool the F-survivors at d=8 across the sample.  Sample S2 = 5 of them.
     For each, F kernel only at d_child=16.
     → measure: avg_children_d16 per L1 F-survivor parent, walls.
  4. Derive:
        L1 total children (d=8 layer)  ≈ N_L0_surv × avg_children_d8
        L1 F-survivors                 ≈ N_L0_surv × avg_F_surv_d8
        **L2 total children (d=16 layer) ≈ L1 F-survivors × avg_children_d16**
        Total wall (full L1)  ≈ N_L0_surv × avg_L1_wall
        Total wall (full L2)  ≈ L1_F_survivors × avg_L2_wall

All wall numbers are EXTRAPOLATED from the small samples × measured per-parent walls.
The size numbers are the relevant ones for cascade-feasibility planning.
Cap per-parent wall at 60s (kernel can't be interrupted; just included as outlier).
"""
import os, sys, time, json
from datetime import datetime, timezone
from math import comb
import numpy as np

ROOT = os.environ.get('CASCADE_ROOT', '/home/ubuntu')
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, ROOT)
from pruning import correction
from run_cascade import run_level0, process_parent_fused

C_TARGET = 1.281
C_UPPER  = 1.5029
M_VALUES = [10, 15, 20, 30]
N_PARENT = 2          # d0 = 4
S1 = 10               # L1 sample size
S2 = 5                # L2 sample size (slow → keep small)


def time_F_kernel(parent, m, c, n_half_child):
    t = time.time()
    sF, n_ch = process_parent_fused(
        parent, m, c, n_half_child,
        use_flat_threshold=False, use_F=True, use_Q=False,
        skip_sdp_cert=True)
    return time.time() - t, int(len(sF)), int(n_ch), sF


def main():
    rows = []
    for m in M_VALUES:
        corr = correction(m, N_PARENT)
        if C_TARGET + corr >= C_UPPER:
            print(f"\nm={m}: VACUOUS (c+corr={C_TARGET+corr:.4f} >= {C_UPPER})")
            continue
        d0 = 2 * N_PARENT
        S0 = 4 * N_PARENT * m
        n_l0_total = comb(S0 + d0 - 1, d0 - 1)
        thresh = C_TARGET + corr
        print(f"\n========= m={m} (d0={d0}, S0={S0}, c+corr={thresh:.4f}) =========",
                flush=True)
        print(f"  L0 total compositions = {n_l0_total:,}", flush=True)

        # 1) L0 with F+Q parallel
        t0 = time.time()
        r0 = run_level0(N_PARENT, m, C_TARGET, verbose=False, use_F=True,
                          use_Q=True, use_L=False)
        l0_wall = time.time() - t0
        L0_surv = r0['survivors']
        N_L0 = int(r0['n_survivors'])
        print(f"  L0 wall = {l0_wall:.2f}s  →  {N_L0:,} survivors "
              f"({100*N_L0/max(n_l0_total,1):.3f}%)", flush=True)

        if N_L0 == 0:
            rows.append({'m': m, 'verdict': 'L0_CLOSED',
                          'l0_wall_s': l0_wall, 'N_L0_surv': 0})
            continue

        # 2) L1 d=8 sample
        rng = np.random.default_rng(7)
        s1 = min(S1, N_L0)
        idx = rng.choice(N_L0, s1, replace=False)
        sample_L0 = L0_surv[idx]

        children_d8 = []
        F_d8 = []
        walls_L1 = []
        L1_pool = []
        print(f"\n  Sampling L1 d=8 on {s1} L0 survivors:", flush=True)
        for i, p in enumerate(sample_L0):
            w, nF, nch, sF = time_F_kernel(p, m, C_TARGET, n_half_child=4)
            walls_L1.append(w)
            children_d8.append(nch)
            F_d8.append(nF)
            if nF > 0:
                L1_pool.append(sF)
            print(f"    [{i+1}/{s1}] children={nch:,}  F={nF}  wall={w:.2f}s", flush=True)

        avg_ch_d8 = float(np.mean(children_d8))
        avg_F_d8 = float(np.mean(F_d8))
        avg_w_L1 = float(np.mean(walls_L1))
        L1_total_children = avg_ch_d8 * N_L0
        L1_total_F_surv = avg_F_d8 * N_L0
        L1_full_wall_estimate = avg_w_L1 * N_L0
        print(f"\n  L1 (d=8) avg children/parent = {avg_ch_d8:,.0f}",
              flush=True)
        print(f"  L1 (d=8) avg F-survivors/parent = {avg_F_d8:.1f}", flush=True)
        print(f"  ⇒ L1 layer total children (extrapolated) = {L1_total_children:,.2e}",
              flush=True)
        print(f"  ⇒ L1 layer total F-survivors (extrapolated) = {L1_total_F_surv:,.0f}",
              flush=True)
        print(f"  ⇒ Full L1 wall estimate = {L1_full_wall_estimate/3600:.2f} hr",
              flush=True)

        # 3) L2 d=16 sample
        L2_size_est = -1
        avg_ch_d16 = -1
        avg_F_d16 = -1
        avg_w_L2 = -1
        if not L1_pool:
            print(f"  No L1 F-survivors in sample → can't probe d=16 directly",
                    flush=True)
        else:
            pool = np.vstack(L1_pool)
            print(f"\n  L1 F-survivor pool (sampled): {len(pool)} compositions",
                    flush=True)
            s2 = min(S2, len(pool))
            idx2 = rng.choice(len(pool), s2, replace=False)
            sample_L1 = pool[idx2]
            children_d16 = []
            F_d16 = []
            walls_L2 = []
            print(f"  Sampling L2 d=16 on {s2} L1 F-survivors:", flush=True)
            for i, p in enumerate(sample_L1):
                w, nF, nch, _sF = time_F_kernel(p, m, C_TARGET, n_half_child=8)
                walls_L2.append(w)
                children_d16.append(nch)
                F_d16.append(nF)
                print(f"    [{i+1}/{s2}] children={nch:,}  F={nF}  wall={w:.2f}s",
                        flush=True)
            avg_ch_d16 = float(np.mean(children_d16))
            avg_F_d16 = float(np.mean(F_d16))
            avg_w_L2 = float(np.mean(walls_L2))
            L2_size_est = L1_total_F_surv * avg_ch_d16
            L2_full_wall_estimate = L1_total_F_surv * avg_w_L2
            print(f"\n  L2 (d=16) avg children/parent = {avg_ch_d16:,.0f}",
                    flush=True)
            print(f"  L2 (d=16) avg F-survivors/parent = {avg_F_d16:.1f}",
                    flush=True)
            print(f"  ⇒ L2 layer total children (extrapolated) = {L2_size_est:,.2e}",
                    flush=True)
            print(f"  ⇒ Full L2 wall estimate = {L2_full_wall_estimate/3600:.2f} hr",
                    flush=True)

        rows.append({
            'm': m, 'd0': d0,
            'l0_wall_s': l0_wall,
            'N_L0_total': n_l0_total,
            'N_L0_surv': N_L0,
            'L0_survival_pct': 100 * N_L0 / max(n_l0_total, 1),
            'L1_avg_children': avg_ch_d8,
            'L1_avg_F_per_parent': avg_F_d8,
            'L1_avg_wall_s': avg_w_L1,
            'L1_total_children_est': L1_total_children,
            'L1_total_F_surv_est': L1_total_F_surv,
            'L1_full_wall_est_hr': L1_full_wall_estimate / 3600,
            'L2_avg_children': avg_ch_d16,
            'L2_avg_F_per_parent': avg_F_d16,
            'L2_avg_wall_s': avg_w_L2,
            'L2_total_children_est': L2_size_est,
        })

    # Final table
    print("\n\n=========================================================")
    print("  D=16 LAYER SIZE BENCHMARK  (c=1.281, d0=4 cascade path)")
    print("=========================================================")
    print(f"{'m':>3} {'L0_total':>12} {'L0_surv':>10} {'L0%':>6} "
          f"{'L1_total':>12} {'L1_Fsurv':>12} {'L2_total':>14} "
          f"{'L1_wall_hr':>11} {'L2_wall_hr':>11}")
    for r in rows:
        if 'verdict' in r:
            print(f"{r['m']:>3}  {r['verdict']}")
            continue
        l1_w = '-' if r.get('L1_full_wall_est_hr') is None else f'{r["L1_full_wall_est_hr"]:.2f}'
        # L2 wall = L1_F_survivors * L2_avg_wall_s / 3600
        l2_w = '-'
        if r['L2_avg_wall_s'] > 0 and r['L1_total_F_surv_est'] > 0:
            l2_w = f"{r['L1_total_F_surv_est'] * r['L2_avg_wall_s'] / 3600:.2f}"
        print(f"{r['m']:>3} {r['N_L0_total']:>12,} {r['N_L0_surv']:>10,} "
              f"{r['L0_survival_pct']:>5.2f}% "
              f"{r['L1_total_children_est']:>12.2e} "
              f"{r['L1_total_F_surv_est']:>12.2e} "
              f"{r['L2_total_children_est']:>14.2e} "
              f"{l1_w:>11} {l2_w:>11}")

    out_path = os.path.join(ROOT, '_d16_size_results.json')
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nJSON: {out_path}")


if __name__ == '__main__':
    main()
