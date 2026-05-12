"""Stage 2 assessment: do the existing parent-level certs (Theorem 1,
LP dual, SDP parent) actually FIRE on real cascade survivors?

If they fire often -> existing Stage 2 is already strong; new Stage 2 fix
gives marginal returns.
If they rarely fire -> there IS room for a stronger Stage 2.

We take L0 survivors as 'parents' for L1, and for each test all 3 certs.
"""
import os, sys, time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from pruning import count_compositions
from cascade_opts import (_whole_parent_prune_theorem1,
                          lp_dual_certificate, sdp_certify_parent)
from run_cascade import _prune_dynamic_int32, _compute_bin_ranges


def assess(n_half, m, c_target, max_parents=200):
    d = 2 * n_half
    S_full = 4 * n_half * m
    S_half = 2 * n_half * m
    n_total_half = count_compositions(n_half, S_half)

    print(f"\n=== n_half={n_half}, m={m}, c_target={c_target} ===")
    print(f"d={d}, S_full=4nm={S_full}, total palindromic comps={n_total_half:,}")

    # Get L0 survivors
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
    if surv:
        surv = np.vstack(surv)
    else:
        surv = np.empty((0, d), dtype=np.int32)
    t_l0 = time.time() - t0
    print(f"L0: {n_proc:,} processed, {len(surv):,} survivors  [{t_l0:.2f}s]")

    if len(surv) == 0:
        print("No L0 survivors — nothing to assess.")
        return

    # For Stage 2 assessment: each L0 survivor becomes a parent at L1
    # (n_half_child = 2*n_half).  Apply the 3 certs to it.
    n_half_child = 2 * n_half
    sample = surv[:max_parents]
    n_t1 = n_lp = n_sdp = 0
    n_lp_skipped = n_sdp_skipped = 0
    n_no_prune = 0
    t_t1 = t_lp = t_sdp = 0.0

    for parent in sample:
        # Compute cursor ranges for child
        d_child = 2 * d
        # _compute_bin_ranges signature in run_cascade.py
        try:
            res = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        except Exception as e:
            print(f"  bin_ranges error: {e}")
            continue
        if res is None:
            continue
        lo_arr, hi_arr, total_children = res
        if total_children == 0:
            continue

        # Theorem 1 cert
        ta = time.time()
        t1 = _whole_parent_prune_theorem1(parent, lo_arr, hi_arr,
                                            int(n_half_child), int(m), c_target)
        t_t1 += time.time() - ta
        if t1:
            n_t1 += 1
            continue

        # LP dual cert (skip if d > 10)
        if d <= 10:
            ta = time.time()
            try:
                lp = lp_dual_certificate(parent, lo_arr, hi_arr,
                                          int(n_half_child), int(m), c_target)
            except Exception as e:
                lp = False
            t_lp += time.time() - ta
            if lp:
                n_lp += 1
                continue
        else:
            n_lp_skipped += 1

        # SDP parent cert (skip if d > 12)
        if d <= 12:
            ta = time.time()
            try:
                sdp = sdp_certify_parent(parent, lo_arr, hi_arr,
                                          int(n_half_child), int(m), c_target)
            except Exception as e:
                sdp = False
            t_sdp += time.time() - ta
            if sdp:
                n_sdp += 1
                continue
        else:
            n_sdp_skipped += 1

        n_no_prune += 1

    n_tested = len(sample)
    print(f"\n--- Stage 2 cert assessment on {n_tested} L0-survivor parents ---")
    print(f"  Theorem 1 cleared:        {n_t1:>6}  ({100*n_t1/n_tested:5.1f}%)  "
          f"[{t_t1:.2f}s]")
    print(f"  LP dual cleared:          {n_lp:>6}  ({100*n_lp/n_tested:5.1f}%)  "
          f"[{t_lp:.2f}s, skipped={n_lp_skipped}]")
    print(f"  SDP parent cleared:       {n_sdp:>6}  ({100*n_sdp/n_tested:5.1f}%)  "
          f"[{t_sdp:.2f}s, skipped={n_sdp_skipped}]")
    print(f"  REQUIRES enumeration:     {n_no_prune:>6}  "
          f"({100*n_no_prune/n_tested:5.1f}%)")
    if n_no_prune > 0:
        print(f"  --> {n_no_prune} parents need full child expansion."
              f" Stage 2 improvement would target these.")
    else:
        print(f"  --> All parents handled by existing certs."
              f" Stage 2 improvement gives marginal returns.")


if __name__ == '__main__':
    for nh, m, c in [(2, 30, 1.20), (3, 10, 1.20), (3, 10, 1.28),
                      (3, 20, 1.28), (4, 10, 1.28)]:
        assess(nh, m, c, max_parents=100)
