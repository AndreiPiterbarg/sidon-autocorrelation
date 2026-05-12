"""S3 assessment driver: per-parent timing of column-generation LP cert
vs the existing vertex-enumeration LP cert.

Mirrors _stage2_assess.py but adds the active-set version for direct
side-by-side timing.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from cascade_opts import (
    _whole_parent_prune_theorem1,
    lp_dual_certificate,
)
from run_cascade import _prune_dynamic_int32, _compute_bin_ranges

from _S3_active_lp import lp_dual_certificate_active


def assess(n_half, m, c_target, max_parents=60, max_d_orig_lp=14):
    d = 2 * n_half
    S_half = 2 * n_half * m
    print(f"\n=== n_half={n_half}, m={m}, c_target={c_target}, d_parent={d} ===")

    # L0 survivors -> "parents" for L1 cert
    surv = []
    n_proc = 0
    t0 = time.time()
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_proc += len(batch)
        s = _prune_dynamic_int32(batch, n_half, m, c_target,
                                  use_flat_threshold=False)
        if s.any():
            surv.append(batch[s].copy())
        # Cap survivor accumulation at d>=12 (else memory blows up)
        if d >= 12 and surv and sum(len(x) for x in surv) > max_parents * 1000:
            break
    if surv:
        surv = np.vstack(surv)
    else:
        surv = np.empty((0, d), dtype=np.int32)
    t_l0 = time.time() - t0
    print(f"  L0: {n_proc:,} processed, {len(surv):,} survivors  [{t_l0:.2f}s]")
    if len(surv) == 0:
        return {'note': 'no L0 survivors'}

    n_half_child = 2 * n_half
    d_child = 2 * d
    sample = surv[:max_parents]

    # Counters
    n_t1 = 0
    n_examined = 0
    n_lp_orig = n_lp_act = 0
    n_match = orig_T_act_F = orig_F_act_T = 0
    t_orig = t_act = 0.0

    for parent in sample:
        res = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        if res is None:
            continue
        lo_arr, hi_arr, _ = res
        if _whole_parent_prune_theorem1(parent, lo_arr, hi_arr,
                                         int(n_half_child), int(m), c_target):
            n_t1 += 1
            continue
        n_examined += 1

        # Original cert
        if d <= max_d_orig_lp:
            ta = time.time()
            try:
                lp_o = lp_dual_certificate(parent, lo_arr, hi_arr,
                                             int(n_half_child), int(m), c_target)
            except Exception:
                lp_o = False
            t_orig += time.time() - ta
        else:
            lp_o = None

        # Active-set cert
        ta = time.time()
        try:
            lp_a = lp_dual_certificate_active(parent, lo_arr, hi_arr,
                                                int(n_half_child), int(m), c_target)
        except Exception:
            lp_a = False
        t_act += time.time() - ta

        if lp_o is not None:
            n_lp_orig += int(lp_o)
            n_lp_act += int(lp_a)
            if lp_o == lp_a:
                n_match += 1
            elif lp_o and not lp_a:
                orig_T_act_F += 1
            else:
                orig_F_act_T += 1
                print(f"  !! UNSOUND parent={tuple(parent)} "
                      f"lo={tuple(lo_arr)} hi={tuple(hi_arr)}")
        else:
            n_lp_act += int(lp_a)

    per = max(n_examined, 1)
    rec = {
        'n_half': n_half, 'm': m, 'c_target': c_target,
        'd_parent': d, 'n_examined': n_examined,
        't1_cleared': n_t1,
        'lp_original_True': n_lp_orig if d <= max_d_orig_lp else None,
        'lp_active_True': n_lp_act,
        'match': n_match if d <= max_d_orig_lp else None,
        'orig_True_act_False': orig_T_act_F if d <= max_d_orig_lp else None,
        'orig_False_act_True': orig_F_act_T if d <= max_d_orig_lp else None,
        'orig_avg_ms': (1000 * t_orig / per) if d <= max_d_orig_lp else None,
        'active_avg_ms': 1000 * t_act / per,
        'speedup': (t_orig / max(t_act, 1e-9)) if d <= max_d_orig_lp else None,
    }
    print(f"  examined={n_examined}  T1 cleared={n_t1}")
    if d <= max_d_orig_lp:
        print(f"  LP_orig_True={n_lp_orig}   LP_act_True={n_lp_act}")
        print(f"  match={n_match}   orig=T/act=F={orig_T_act_F}   "
              f"orig=F/act=T (UNSOUND)={orig_F_act_T}")
        print(f"  avg ms: orig={1000 * t_orig / per:.1f}   "
              f"active={1000 * t_act / per:.1f}   "
              f"speedup={t_orig / max(t_act, 1e-9):.2f}x")
    else:
        print(f"  LP_act_True={n_lp_act}   "
              f"avg ms: active={1000 * t_act / per:.1f} "
              f"(orig skipped: d={d} > {max_d_orig_lp})")
    return rec


if __name__ == '__main__':
    cases = [
        (3, 10, 1.28),  # d_parent = 6
        (4, 10, 1.28),  # d_parent = 8
        (4, 10, 2.50),  # d_parent = 8 high c
        (5,  8, 1.20),  # d_parent = 10
        (6,  6, 1.20),  # d_parent = 12
    ]
    out = []
    for nh, m, c in cases:
        rec = assess(nh, m, c, max_parents=20)
        out.append(rec)
    with open(os.path.join(_HERE, '_S3_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote _S3_results.json")
