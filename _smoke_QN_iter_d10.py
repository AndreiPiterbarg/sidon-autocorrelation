"""Smoke test: QN-iterative (joint spectral) at d=10 (n_half=5, m=5, c=1.28).

GOAL
----
Q already prunes everything at d=12 (so QN gave 0 extras there).  At d=10,
F=558 -> Q=240 -> L=14, so Q's residue is 240 cells.  This smoke tests
whether QN-iterative (the JOINT spectral bound from `_QN_bench._qn_bound_iter`)
can prune any of those 240 Q-survivors.  Each extra prune saves L (Lasserre
SDP) work.

PIPELINE
--------
1. Generate F-survivors at (n_half=5, m=5, d=10, c=1.28) via _M1_bench.prune_F.
   Expected ~558.
2. Apply Q via _Q_bench.prune_Q_one. Expected ~240 survivors.
3. Apply QN-iterative on the Q-survivors.
4. Soundness: any QN-survivor that's not a Q-survivor (impossible if sound)
   would be a bug. Also count if QN-iter gives a smaller bound than Q
   (sound regression).
5. Time each phase.

QN-ITER ANALYSIS (from _QN_bench.py)
-------------------------------------
`_qn_bound_iter(c_int, ..., m_W_arr, n_pairs_arr_per_window, A_W_list, max_iter=4)`:

Returns t_best (excess in m^2 units; > margin*m^2 ==> prune).

Mechanism:
  - Iter 0: solves QN-fast LP. If excess>0, return immediately (prune found).
  - Iter k>=1:
      a. Build A_lam = sum_W (lam_W/ell_W) A_W (d-by-d real symmetric).
      b. alpha = 1^T A_lam 1 / d^2.  op_norm = ||A_lam - alpha||_op.
      c. joint_corr = op_norm * d / (4n).
         sum_corr = sum_W (lam_W/ell_W) m_W / (4n) (current LP correction).
      d. If joint_corr < sum_corr (spectral tighter), re-solve LP with
         m_W=0 (no per-W quadratic) and use t_lp_zero - joint_corr.
      e. Update lam if improved.

Soundness: each iterate uses min(spectral, sum), both individually sound.

API: returns scalar t_best (NOT a tuple). prune iff t_best > 1e-9*m^2.

CONSTRAINTS: wall < 10 min.
"""
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from pruning import count_compositions
from _M1_bench import prune_F
from _Q_bench import (_enum_balanced_signs, _build_windows, prune_Q_one)
from _QN_bench import (precompute_window_data, _qn_bound_iter, prune_QN_one,
                        _qn_bound_lp)


def main(n_half=5, m=5, c_target=1.28, batch_size=200_000, max_iter=4,
         margin=1e-9, out_path=None):
    d = 2 * n_half
    S_half = 2 * n_half * m
    n_total_half = count_compositions(n_half, S_half)

    print(f"=== QN-iterative smoke at d={d} ===")
    print(f"  n_half={n_half}, m={m}, c_target={c_target}")
    print(f"  total palindromic compositions = {n_total_half:,}")

    # Precompute window data + op_rest + A_W list (for spectral iter step).
    print("  precomputing window data, op_rest, A_W list...", end=' ', flush=True)
    t_pre = time.time()
    windows, ell_int_sums, sigmas, m_W_arr, A_W_list = \
        precompute_window_data(d, n_half)
    n_win = len(windows)
    n_sigma = len(sigmas)
    pre_secs = time.time() - t_pre
    print(f"n_win={n_win}, n_sigma={n_sigma} [{pre_secs:.2f}s]")

    # n_pairs_arr_per_window (only used for prune_QN_one's bookkeeping;
    # _qn_bound_iter signature accepts it as positional).  Build same as m_W_arr.
    n_pairs_per_window = ell_int_sums.astype(np.float64)

    # Warm JIT
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    prune_F(warm, n_half, m, c_target)

    # ---- Phase F ----
    print("\n[Phase F] running F-prune across all palindromic comps...")
    t0 = time.time()
    F_survivors = []
    n_proc = 0
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=batch_size):
        full = np.empty((len(half_batch), d), dtype=np.int32)
        full[:, :n_half] = half_batch
        full[:, n_half:] = half_batch[:, ::-1]
        n_proc += len(full)
        sF = prune_F(full, n_half, m, c_target)
        if int(sF.sum()) > 0:
            F_survivors.append(full[sF].copy())
    if F_survivors:
        F_survivors = np.concatenate(F_survivors, axis=0)
    else:
        F_survivors = np.empty((0, d), dtype=np.int32)
    t_F = time.time() - t0
    print(f"  processed {n_proc:,}, F-survivors = {len(F_survivors)}  [{t_F:.2f}s]")

    # ---- Phase Q ----
    print("\n[Phase Q] running Q-prune on F-survivors...")
    t0 = time.time()
    sQ = np.ones(len(F_survivors), dtype=bool)
    for k in range(len(F_survivors)):
        c_int = F_survivors[k]
        if prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                        n_half, m, c_target, margin=margin):
            sQ[k] = False
    t_Q = time.time() - t0
    Q_survivors = F_survivors[sQ]
    extra_Q = int(np.sum(~sQ))
    print(f"  Q-survivors = {len(Q_survivors)}  (extra prune over F = {extra_Q})  "
          f"[{t_Q:.2f}s, {1000*t_Q/max(1,len(F_survivors)):.2f} ms/LP]")

    # Soundness sanity: also run QN-fast on F-survivors (should subset Q).
    # We'll do this implicitly inside iter call.

    # ---- Phase QN-iterative on Q-survivors ----
    print(f"\n[Phase QN-iter] running QN-iterative (max_iter={max_iter}) on Q-survivors...")
    t0 = time.time()
    qn_iter_excesses = np.empty(len(Q_survivors), dtype=np.float64)
    qn_fast_excesses = np.empty(len(Q_survivors), dtype=np.float64)
    q_excesses = np.empty(len(Q_survivors), dtype=np.float64)

    # We need the Q-excess for soundness comparison and to confirm Q said
    # "not pruned" (excess <= 0).  Recompute via _q_bound_lp.
    from _Q_bench import _q_bound_lp
    sQN = np.ones(len(Q_survivors), dtype=bool)
    sQN_fast = np.ones(len(Q_survivors), dtype=bool)
    for k in range(len(Q_survivors)):
        c_int = Q_survivors[k]
        # Q-excess (for sanity)
        tQ_val, _ = _q_bound_lp(c_int, windows, ell_int_sums, sigmas,
                                  n_half, m, c_target)
        q_excesses[k] = tQ_val
        # QN-fast (for comparison)
        tQNf, _ = _qn_bound_lp(c_int, windows, ell_int_sums, sigmas,
                                n_half, m, c_target, m_W_arr)
        qn_fast_excesses[k] = tQNf
        if tQNf > margin * m * m:
            sQN_fast[k] = False
        # QN-iterative
        t_iter_val = _qn_bound_iter(c_int, windows, ell_int_sums, sigmas,
                                      n_half, m, c_target,
                                      m_W_arr, n_pairs_per_window, A_W_list,
                                      max_iter=max_iter)
        qn_iter_excesses[k] = t_iter_val
        if t_iter_val > margin * m * m:
            sQN[k] = False
    t_QN_iter = time.time() - t0

    extra_QN_iter_over_Q = int(np.sum(~sQN))   # Q said survive, QN-iter prunes
    extra_QN_fast_over_Q = int(np.sum(~sQN_fast))
    QN_iter_survivors = int(np.sum(sQN))
    QN_fast_survivors = int(np.sum(sQN_fast))

    print(f"  QN-fast survivors      = {QN_fast_survivors}  "
          f"(extra over Q = {extra_QN_fast_over_Q})")
    print(f"  QN-iterative survivors = {QN_iter_survivors}  "
          f"(extra over Q = {extra_QN_iter_over_Q})  "
          f"[{t_QN_iter:.2f}s, "
          f"{1000*t_QN_iter/max(1,len(Q_survivors)):.2f} ms/comp]")

    # ---- Soundness check ----
    # All Q-survivors should have q_excesses <= margin * m^2.
    q_thr = margin * m * m
    q_violators = int(np.sum(q_excesses > q_thr))
    print(f"\n[Soundness] Q-excess on Q-survivors > {q_thr:.2e}? "
          f"{q_violators} violations (expected 0)")

    # qn_fast_excesses[k] should be <= qn_iter_excesses[k] (iter only refines).
    iter_smaller_than_fast = int(np.sum(qn_iter_excesses < qn_fast_excesses - 1e-9))
    print(f"[Soundness] QN-iter < QN-fast (regression)? "
          f"{iter_smaller_than_fast} violations (expected 0)")

    # qn_fast_excesses[k] should be <= q_excesses[k] (QN-fast tighter than Q).
    fast_loose_than_q = int(np.sum(qn_fast_excesses < q_excesses - 1e-9))
    print(f"[Soundness] QN-fast < Q? "
          f"{fast_loose_than_q} violations (expected 0)")

    # ---- Distribution of QN-iter excess on Q-survivors ----
    if len(qn_iter_excesses) > 0:
        print(f"\n[Distribution] QN-iter excess (in m^2 units, >0 means prune):")
        print(f"  min={qn_iter_excesses.min():+.4e}, max={qn_iter_excesses.max():+.4e}, "
              f"mean={qn_iter_excesses.mean():+.4e}")
        print(f"  Q   excess: min={q_excesses.min():+.4e}, max={q_excesses.max():+.4e}")

    # Top 5 wins (QN-iter excess - Q excess)
    if len(Q_survivors) > 0:
        gain = qn_iter_excesses - q_excesses
        top_idx = np.argsort(-gain)[:5]
        print(f"\n[Top wins] QN-iter excess - Q excess (top 5):")
        for ti in top_idx:
            print(f"  c={Q_survivors[ti].tolist()}: "
                  f"Q-exc={q_excesses[ti]:+.4e}, QN-fast={qn_fast_excesses[ti]:+.4e}, "
                  f"QN-iter={qn_iter_excesses[ti]:+.4e}, gain={gain[ti]:+.4e}")

    total_wall = t_F + t_Q + t_QN_iter + pre_secs
    print(f"\nTotal wall: {total_wall:.2f}s "
          f"(pre={pre_secs:.2f} F={t_F:.2f} Q={t_Q:.2f} QN-iter={t_QN_iter:.2f})")

    result = {
        'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d,
        'n_processed': n_proc, 'n_win': n_win, 'n_sigma': n_sigma,
        'max_iter': max_iter, 'margin': margin,
        'surv_F': int(len(F_survivors)),
        'surv_Q': int(len(Q_survivors)),
        'surv_QN_fast': QN_fast_survivors,
        'surv_QN_iter': QN_iter_survivors,
        'extra_Q_over_F': int(np.sum(~sQ)),
        'extra_QN_fast_over_Q': extra_QN_fast_over_Q,
        'extra_QN_iter_over_Q': extra_QN_iter_over_Q,
        't_F': t_F, 't_Q': t_Q, 't_QN_iter': t_QN_iter, 't_pre': pre_secs,
        'wall_total': total_wall,
        'soundness_q_violators': q_violators,
        'soundness_iter_lt_fast': iter_smaller_than_fast,
        'soundness_fast_lt_q': fast_loose_than_q,
        'q_excess_min': float(q_excesses.min()) if len(q_excesses) else None,
        'q_excess_max': float(q_excesses.max()) if len(q_excesses) else None,
        'qn_fast_excess_min': float(qn_fast_excesses.min()) if len(qn_fast_excesses) else None,
        'qn_fast_excess_max': float(qn_fast_excesses.max()) if len(qn_fast_excesses) else None,
        'qn_iter_excess_min': float(qn_iter_excesses.min()) if len(qn_iter_excesses) else None,
        'qn_iter_excess_max': float(qn_iter_excesses.max()) if len(qn_iter_excesses) else None,
    }

    # Write result
    if out_path is None:
        out_path = os.path.join(_dir, '_smoke_QN_iter_d10.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2)
    print(f"\nWrote {out_path}")
    return result


if __name__ == '__main__':
    main()
