"""Soundness audit: enumerate δ-cell on first 5 EXTRA-pruned comps and check
whether any δ realizes m^2*TV_W <= c_target*m^2 for all W (i.e. comp is FEASIBLE).
If so, the EXTRA cuts are UNSOUND.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
from itertools import product

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))

from compositions import generate_compositions_batched
from pruning import count_compositions
from _M1_bench import prune_F
from _Q_bench import _build_windows, prune_Q_one, _enum_balanced_signs
from _L_bench import _build_A_matrices, _shor_feasibility, _make_cell
from _smoke_RLT_extra import _shor_feasibility_extra


def audit_one(c_int, A_mats, windows, n_half, m, c_target, n_grid=5):
    """Enumerate δ ∈ {-1, -.5, 0, .5, 1}^d ∩ {sum δ = 0, δ_i <= c_i, x = c-δ in cell}.
    For each, compute max_W (m^2·TV_W(a)). If <= c_target·m^2 for ANY δ:
       composition is FEASIBLE -> SDP-prune was unsound!
    """
    d = len(c_int)
    cs_m2 = c_target * m * m
    nm = 4 * n_half * m
    if n_grid == 3:
        levels = [-1, 0, 1]
    elif n_grid == 5:
        levels = [-1, -0.5, 0, 0.5, 1]
    else:
        levels = [-1, 0, 1]

    c = np.asarray(c_int, dtype=np.float64)
    n_checked = 0
    n_feasible = 0
    for delta in product(levels, repeat=d):
        delta = np.array(delta, dtype=np.float64)
        if abs(np.sum(delta)) > 1e-9:
            continue
        x = c - delta
        if np.any(x < 0):
            continue
        n_checked += 1
        # x in [c-1, c+1]; check all window TV
        max_v = -np.inf
        for A_mat, (ell, _) in zip(A_mats, windows):
            v = float(x.T @ A_mat @ x) / (4.0 * n_half * ell)
            if v > max_v:
                max_v = v
        if max_v <= cs_m2 + 1e-6:
            n_feasible += 1
            return True, x, max_v / (m * m)
    return False, None, n_checked


def main():
    n_half, m, c_target = 3, 10, 1.28
    d = 2 * n_half
    S_half = 2 * n_half * m

    windows, ell_int_sums = _build_windows(d)
    A_mats = _build_A_matrices(d, windows)
    sigmas = _enum_balanced_signs(d)

    # F + Q survivors
    f_survivors = []
    for half_batch in generate_compositions_batched(n_half, S_half, batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        sF = prune_F(batch, n_half, m, c_target)
        for k in range(len(batch)):
            if sF[k]:
                f_survivors.append(batch[k].copy())
    q_survivors = [c for c in f_survivors
                   if not prune_Q_one(c, windows, ell_int_sums, sigmas,
                                       n_half, m, c_target)]
    print(f"F={len(f_survivors)}, Q={len(q_survivors)}")

    # Run baseline + extra on first 10 Q-survivors and audit any extra-pruned ones
    n_test = min(10, len(q_survivors))
    n_unsound = 0
    n_sound = 0
    for k, c_int in enumerate(q_survivors[:n_test]):
        lo, hi = _make_cell(c_int, m)
        b_pruned, _ = _shor_feasibility(c_int, lo, hi, A_mats, windows,
                                          n_half, m, c_target,
                                          solver='MOSEK')
        e_pruned, _ = _shor_feasibility_extra(c_int, lo, hi, A_mats, windows,
                                                n_half, m, c_target,
                                                solver='MOSEK', triple_cuts=4)
        if e_pruned and not b_pruned:
            # Audit
            print(f"\n  comp {k}: c={list(c_int)}: extra=PRUNE, baseline=NO_PRUNE")
            print(f"    Auditing 5-grid for feasibility...")
            is_feasible, witness_x, info = audit_one(c_int, A_mats, windows,
                                                       n_half, m, c_target, n_grid=5)
            if is_feasible:
                n_unsound += 1
                print(f"    SOUNDNESS BUG: feasible witness x={witness_x}, max TV*m^2={info:.4f}")
            else:
                n_sound += 1
                print(f"    No 5-grid witness  (checked {info} delta-points)")
        elif e_pruned and b_pruned:
            print(f"  comp {k}: both prune, OK")
        else:
            print(f"  comp {k}: neither prunes")

    print(f"\nUnsound prunes detected: {n_unsound} / {n_test}")
    print(f"Sound (no 5-grid witness): {n_sound} / {n_test}")


if __name__ == '__main__':
    main()
