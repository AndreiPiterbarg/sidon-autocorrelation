"""L0 + L1 cascade test at higher c_target.

For c where L0 enumeration with F+Q+L leaves N>0 survivors, this checks
whether expanding those N survivors to L1 (d_child = 2*d_parent) and applying
F filter at d_child closes the cascade.

If F at L1 leaves 0 children-survivors across ALL L0 L-survivors, then
C_{1a} >= c_target is proven via the L0+L1 cascade.

Pipeline (per c):
  1. Enumerate all palindromic compositions at d=2*n_half via batched
     iterators.  Apply F (numba parallel) -> F-survivors.
  2. Apply Q (joint multi-window LP) on F-survivors -> Q-survivors.
  3. Apply L (Lasserre/Shor SDP, MOSEK) on Q-survivors -> L-survivors.
  4. For each L-survivor parent, generate all valid d_child=2d children
     and apply F filter at d_child.  Collect any F-survivors at L1.
  5. Save everything (parents, children counts, survivors).

If step 5 reports 0 cumulative L1 survivors -> PROVEN.

Time budget: each step has a hard wall.  L1 step uses run_cascade's
process_parent_fused with --use_F (Q+L at d=24 is infeasible due to
n_sigma=C(24,12)=2,704,156).
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from math import comb

ROOT = '/home/ubuntu'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger', 'cpu'))

import numpy as np

# Bench imports
from _M1_bench import prune_F
from _Q_bench import _build_windows as q_build_windows, prune_Q_one, _enum_balanced_signs
from _L_bench import _build_A_matrices, prune_L_one, _detect_solver
from compositions import generate_compositions_batched
from pruning import correction
from run_cascade import process_parent_fused


C_UPPER = 1.5029


def run_l0_full(n_half, m, c_target, batch_size=200_000, do_L=True, log=print):
    """Full L0 enumeration with F + Q + L; returns L-survivor compositions."""
    d = 2 * n_half
    half_sum = 2 * n_half * m
    n_total = comb(half_sum + n_half - 1, n_half - 1)
    log(f"L0 ({n_half},{m},d={d}): {n_total:,} compositions")

    windows_obj = q_build_windows(d)
    if isinstance(windows_obj, tuple):
        windows = windows_obj[0]
        ell_int_sums = windows_obj[1]
    else:
        windows = windows_obj
        ell_int_sums = np.array([w[2] for w in windows], dtype=np.int64)
    sigmas = _enum_balanced_signs(d)
    A_mats = _build_A_matrices(d, windows) if do_L else None
    solver = _detect_solver(prefer='MOSEK') if do_L else None
    if do_L:
        log(f"  L solver = {solver}")

    f_surv_total = []  # F-survivor compositions
    q_surv_total = []
    l_surv_total = []
    n_processed = 0
    n_F = n_Q = n_L = 0
    t_F = t_Q = t_L = 0.0
    t0 = time.time()

    # JIT warmup
    warm = np.zeros((1, d), dtype=np.int32)
    prune_F(warm, n_half, m, c_target)

    for batch in generate_compositions_batched_palindromic(n_half, half_sum,
                                                              batch_size, d):
        n_processed += len(batch)

        # F
        tf = time.time()
        sF = prune_F(batch, n_half, m, c_target)
        t_F += time.time() - tf

        # sF True means SURVIVES (consistent w/ the bench).  Index those.
        f_surv_idx = np.where(sF)[0]
        n_F += len(f_surv_idx)
        for idx in f_surv_idx:
            f_surv_total.append(batch[idx].copy())

        # Q on F-survivors.  Convention (matches bench):
        #   prune_Q_one(...) returns True means comp is Q-PRUNED.
        #   sQ_on_F[k] = True iff Q did NOT prune (i.e., k SURVIVES Q).
        sQ_on_F = np.ones(len(f_surv_idx), dtype=bool)
        tq = time.time()
        for k, idx in enumerate(f_surv_idx):
            if prune_Q_one(batch[idx], windows, ell_int_sums, sigmas,
                                n_half, m, c_target):
                sQ_on_F[k] = False  # Q pruned this composition
        t_Q += time.time() - tq
        q_surv_local = f_surv_idx[sQ_on_F]    # actual Q-survivors
        n_Q += len(q_surv_local)
        for idx in q_surv_local:
            q_surv_total.append(batch[idx].copy())

        # L on Q-survivors.  Same convention:
        #   prune_L_one returns (pruned: bool, status).  pruned=True ⇒ L PRUNED.
        #   sL_on_Q[k] = True iff L did NOT prune (k SURVIVES L).
        if do_L and len(q_surv_local) > 0:
            sL_on_Q = np.ones(len(q_surv_local), dtype=bool)
            tl = time.time()
            for k, idx in enumerate(q_surv_local):
                pruned, status = prune_L_one(batch[idx], A_mats, windows,
                                                n_half, m, c_target,
                                                solver=solver, order=1)
                if pruned:
                    sL_on_Q[k] = False
            t_L += time.time() - tl
            l_surv_local = q_surv_local[sL_on_Q]
        else:
            l_surv_local = q_surv_local
        n_L += len(l_surv_local)
        for idx in l_surv_local:
            l_surv_total.append(batch[idx].copy())

        # Periodic progress
        if n_processed % (batch_size * 50) == 0:
            log(f"    progress: {n_processed:,}/{n_total:,} "
                f"({100*n_processed/n_total:.1f}%)  "
                f"F={n_F} Q={n_Q} L={n_L}  "
                f"wall={time.time()-t0:.0f}s")

    return {
        'n_processed': n_processed,
        'n_F': n_F, 'n_Q': n_Q, 'n_L': n_L,
        'l_survivors': np.array(l_surv_total, dtype=np.int32) if l_surv_total
                        else np.zeros((0, d), dtype=np.int32),
        'q_survivors': np.array(q_surv_total, dtype=np.int32) if q_surv_total
                        else np.zeros((0, d), dtype=np.int32),
        't_F': t_F, 't_Q': t_Q, 't_L': t_L,
        'wall_total': time.time() - t0,
    }


def generate_compositions_batched_palindromic(n_half, half_sum, batch_size, d):
    """Yield palindromic compositions of length d with sum = 2*half_sum."""
    for half_batch in generate_compositions_batched(n_half, half_sum,
                                                       batch_size=batch_size):
        n = len(half_batch)
        full = np.empty((n, d), dtype=np.int32)
        full[:, :n_half] = half_batch
        full[:, n_half:] = half_batch[:, ::-1]
        yield full


def l1_expand_test(parents, n_half, m, c_target, log=print, hard_wall=3000):
    """For each parent (d=2*n_half), expand to d=2d via cascade kernel
    and apply F (only F — Q+L too expensive at d=24).  Return any L1
    survivors."""
    d = 2 * n_half
    d_child = 2 * d
    n_half_child = d_child // 2
    n_parents = len(parents)
    log(f"L1: expanding {n_parents} parents from d={d} to d={d_child}")

    total_children = 0
    total_l1_survivors_count = 0
    parent_results = []
    t0 = time.time()
    for i, p in enumerate(parents):
        if time.time() - t0 > hard_wall:
            log(f"  *** wall budget {hard_wall}s exhausted after {i} parents ***")
            break
        tp = time.time()
        try:
            surv_i, n_children_i = process_parent_fused(
                p, m, c_target, n_half_child,
                use_flat_threshold=False, use_F=True, skip_sdp_cert=True)
        except Exception as e:
            log(f"  [parent {i}] EXC: {e}")
            parent_results.append({'idx': i, 'children': -1, 'survivors': -1,
                                    'wall': time.time() - tp, 'error': str(e)})
            continue
        n_surv = len(surv_i)
        wall_i = time.time() - tp
        total_children += int(n_children_i)
        total_l1_survivors_count += n_surv
        log(f"  [parent {i+1}/{n_parents}] children={n_children_i:,} "
            f"L1_surv={n_surv}  wall={wall_i:.1f}s")
        parent_results.append({'idx': i, 'children': int(n_children_i),
                                'survivors': int(n_surv),
                                'wall': round(wall_i, 2)})
    return {
        'n_parents_processed': len([r for r in parent_results if 'error' not in r]),
        'n_parents_total': n_parents,
        'total_children': total_children,
        'total_l1_survivors': total_l1_survivors_count,
        'per_parent': parent_results,
        'wall': time.time() - t0,
    }


def main():
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_dir = f'/home/ubuntu/l1_cascade_{ts}'
    os.makedirs(out_dir, exist_ok=True)

    # c_target = 1.32 has L=234 at L0 from prior cert.
    # We probe c=1.32 first.  If L1 closes, push to 1.33, 1.34.
    n_half = 6
    m = 15
    c_targets = [1.32, 1.33, 1.34]

    log_file = os.path.join(out_dir, 'driver.log')
    log_fp = open(log_file, 'a')

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fp.write(line + '\n')
        log_fp.flush()

    log(f"L1 cascade test @ c in {c_targets} for (n={n_half}, m={m}, d={2*n_half})")
    log(f"Output: {out_dir}")

    overall = {'configs': []}
    for c_target in c_targets:
        log(f"\n========== c_target = {c_target} ==========")

        # --- L0
        log0 = log
        l0 = run_l0_full(n_half, m, c_target, batch_size=200_000, do_L=True,
                          log=log0)
        log(f"L0 done: F={l0['n_F']} Q={l0['n_Q']} L={l0['n_L']}  "
            f"wall={l0['wall_total']:.1f}s")

        # Save L-survivors
        np.save(os.path.join(out_dir, f"L_survivors_c{int(c_target*10000)}.npy"),
                l0['l_survivors'])

        if l0['n_L'] == 0:
            log(f"  *** L0 ALONE closes c={c_target} (L=0).  No L1 needed. ***")
            overall['configs'].append({
                'c_target': c_target, 'l0': {k: v for k, v in l0.items()
                                              if k not in ('l_survivors',
                                                           'q_survivors')},
                'closure': 'L0',
            })
            continue

        # --- L1
        l1 = l1_expand_test(l0['l_survivors'], n_half, m, c_target, log=log)
        log(f"L1 done: parents_processed={l1['n_parents_processed']}/"
            f"{l1['n_parents_total']}  "
            f"total_children={l1['total_children']:,}  "
            f"total_L1_surv={l1['total_l1_survivors']}  "
            f"wall={l1['wall']:.1f}s")

        closure = ('L0+L1' if l1['total_l1_survivors'] == 0
                              and l1['n_parents_processed'] == l1['n_parents_total']
                   else 'NOT_CLOSED')
        log(f"  *** c={c_target}: {closure} ***")

        overall['configs'].append({
            'c_target': c_target,
            'l0': {k: v for k, v in l0.items() if k not in ('l_survivors',
                                                              'q_survivors')},
            'l1': l1,
            'closure': closure,
        })

        if closure == 'NOT_CLOSED':
            log(f"  --- stopping push, c={c_target} did not close ---")
            break

    # Save overall summary
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(overall, f, indent=2, default=str)
    log("\nDONE.")
    log_fp.close()


if __name__ == '__main__':
    main()
