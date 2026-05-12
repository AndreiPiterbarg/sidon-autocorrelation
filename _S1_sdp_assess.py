"""S1 assessment: do the existing parent-level certs (Theorem 1, LP dual,
SDP MOSEK) actually FIRE on real cascade survivors?

This is a copy of /root/_stage2_assess.py with `sdp_certify_parent`
replaced by `sdp_certify_parent_mosek`. Reports:
  - clearance % per cert.
  - per-call wall time at d_parent = 4, 6, 8, 12.
  - whether MOSEK ever returns True where it shouldn't have (cross-check
    with full child enumeration when d_parent <= 4).

The output is written to `_S1_results.json` for downstream analysis.
"""
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from pruning import count_compositions
from cascade_opts import (_whole_parent_prune_theorem1,
                          lp_dual_certificate, sdp_certify_parent,
                          _theorem1_threshold_table)
from run_cascade import _prune_dynamic_int32, _compute_bin_ranges
from _S1_sdp_mosek import sdp_certify_parent_mosek


def integer_count_surviving(parent, lo_arr, hi_arr, n_half_child, m, c_target):
    """Returns (n_total, n_surv) of integer children. Bound: 5e6 enumeration."""
    d_parent = len(parent)
    d_child = 2 * d_parent
    thr_table = _theorem1_threshold_table(int(n_half_child), int(m), c_target)
    P = parent.astype(np.int64)
    lo = lo_arr.astype(np.int64)
    hi = hi_arr.astype(np.int64)

    rsizes = (hi - lo + 1)
    n_enum = int(np.prod(rsizes))
    if n_enum > 5_000_000:
        return n_enum, None

    cur = lo.copy().astype(np.int64)
    n_total = 0
    n_surv = 0
    conv_len = 2 * d_child - 1
    while True:
        n_total += 1
        child = np.empty(d_child, dtype=np.int64)
        for p in range(d_parent):
            child[2 * p] = cur[p]
            child[2 * p + 1] = 2 * P[p] - cur[p]
        if not np.any(child < 0):
            conv = np.zeros(conv_len, dtype=np.int64)
            for ii in range(d_child):
                ci = child[ii]
                if ci != 0:
                    conv[2 * ii] += ci * ci
                    for jj in range(ii + 1, d_child):
                        cj = child[jj]
                        if cj != 0:
                            conv[ii + jj] += 2 * ci * cj
            pruned = False
            for ell in range(2, 2 * d_child + 1):
                n_cv = ell - 1
                n_win = conv_len - n_cv + 1
                if n_win <= 0:
                    continue
                thr = thr_table[ell - 2]
                for s_lo in range(n_win):
                    ws = int(conv[s_lo:s_lo + n_cv].sum())
                    if ws > thr:
                        pruned = True
                        break
                if pruned:
                    break
            if not pruned:
                n_surv += 1
        i = 0
        cur[0] += 1
        while cur[i] > hi[i]:
            cur[i] = lo[i]
            i += 1
            if i >= d_parent:
                return n_total, n_surv
            cur[i] += 1


def assess(n_half, m, c_target, max_parents=200, soundness_check=True):
    d = 2 * n_half
    S_full = 4 * n_half * m
    S_half = 2 * n_half * m

    print(f"\n=== n_half={n_half}, m={m}, c_target={c_target} ===", flush=True)
    print(f"d={d}, S_full=4nm={S_full}", flush=True)

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
    print(f"L0: {n_proc:,} processed, {len(surv):,} survivors  [{t_l0:.2f}s]", flush=True)

    if len(surv) == 0:
        return {'n_proc': n_proc, 'n_surv': 0}

    n_half_child = 2 * n_half
    sample = surv[:max_parents]
    n_t1 = n_lp = n_cvxpy = n_mosek = 0
    n_lp_skipped = n_sdp_skipped = 0
    n_no_prune = 0
    n_total = 0
    t_t1 = t_lp = t_cvxpy = t_mosek = 0.0
    cvxpy_call_times = []
    mosek_call_times = []
    soundness_violations = []
    soundness_checked = 0

    for parent in sample:
        d_child = 2 * d
        try:
            res = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        except Exception:
            continue
        if res is None:
            continue
        lo_arr, hi_arr, total_children = res
        if total_children == 0:
            continue
        n_total += 1

        ta = time.time()
        t1 = _whole_parent_prune_theorem1(parent, lo_arr, hi_arr,
                                            int(n_half_child), int(m), c_target)
        t_t1 += time.time() - ta
        if t1:
            n_t1 += 1
            continue

        if d <= 10:
            ta = time.time()
            try:
                lp = lp_dual_certificate(parent, lo_arr, hi_arr,
                                          int(n_half_child), int(m), c_target)
            except Exception:
                lp = False
            t_lp += time.time() - ta
            if lp:
                n_lp += 1
                continue
        else:
            n_lp_skipped += 1

        if d > 12:
            n_sdp_skipped += 1
            n_no_prune += 1
            continue

        # CVXPY SDP cert
        ta = time.time()
        try:
            cv = sdp_certify_parent(parent, lo_arr, hi_arr,
                                     int(n_half_child), int(m), c_target)
        except Exception:
            cv = False
        dt_cv = time.time() - ta
        t_cvxpy += dt_cv
        cvxpy_call_times.append(dt_cv)
        if cv:
            n_cvxpy += 1

        # MOSEK SDP cert
        ta = time.time()
        try:
            ms, status = sdp_certify_parent_mosek(parent, lo_arr, hi_arr,
                                                  int(n_half_child), int(m), c_target,
                                                  return_status=True)
        except Exception as e:
            ms, status = False, f"exception:{e}"
        dt_ms = time.time() - ta
        t_mosek += dt_ms
        mosek_call_times.append(dt_ms)
        if ms:
            n_mosek += 1
            # Cross-check: if MOSEK says infeasible AND we can enumerate,
            # check no integer child survives.
            if soundness_check:
                rsizes = hi_arr - lo_arr + 1
                if int(np.prod(rsizes.astype(np.int64))) <= 1_000_000:
                    soundness_checked += 1
                    n_tot_int, n_surv_int = integer_count_surviving(
                        parent, lo_arr, hi_arr, int(n_half_child),
                        int(m), c_target)
                    if n_surv_int and n_surv_int > 0:
                        soundness_violations.append({
                            'parent': parent.tolist(),
                            'lo': lo_arr.tolist(),
                            'hi': hi_arr.tolist(),
                            'mosek_status': status,
                            'n_total_children': int(n_tot_int),
                            'n_surv_children': int(n_surv_int),
                            'first_surv': '<not_extracted>',
                        })

        if not (cv or ms):
            n_no_prune += 1

    print(f"\n--- S1 cert assessment on {n_total} L0-survivor parents (sample of {len(sample)}) ---")
    print(f"  Theorem 1 cleared:    {n_t1:>5}  ({100*n_t1/n_total:5.1f}%)  [{t_t1:.3f}s]")
    print(f"  LP dual cleared:      {n_lp:>5}  ({100*n_lp/n_total:5.1f}%)  [{t_lp:.3f}s, skipped={n_lp_skipped}]")
    print(f"  SDP CVXPY cleared:    {n_cvxpy:>5}  ({100*n_cvxpy/max(n_total,1):5.1f}%)  [{t_cvxpy:.3f}s]")
    print(f"  SDP MOSEK cleared:    {n_mosek:>5}  ({100*n_mosek/max(n_total,1):5.1f}%)  [{t_mosek:.3f}s, skipped={n_sdp_skipped}]")
    print(f"  REQUIRES enumeration: {n_no_prune:>5}  ({100*n_no_prune/max(n_total,1):5.1f}%)")

    cvxpy_arr = np.array(cvxpy_call_times) if cvxpy_call_times else np.array([0.0])
    mosek_arr = np.array(mosek_call_times) if mosek_call_times else np.array([0.0])
    print(f"  CVXPY per-call: avg={1000*cvxpy_arr.mean():.1f} ms  max={1000*cvxpy_arr.max():.1f} ms")
    print(f"  MOSEK per-call: avg={1000*mosek_arr.mean():.1f} ms  max={1000*mosek_arr.max():.1f} ms")
    print(f"  Soundness probe: {len(soundness_violations)} violations (out of {soundness_checked} checked)")
    if soundness_violations:
        print("  VIOLATIONS:")
        for v in soundness_violations:
            print(f"    {v}")

    return {
        'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d,
        'n_proc': int(n_proc), 'n_surv_l0': int(len(surv)),
        'n_total': int(n_total), 'sample_size': int(len(sample)),
        'n_t1': int(n_t1), 'n_lp': int(n_lp), 'n_lp_skipped': int(n_lp_skipped),
        'n_cvxpy': int(n_cvxpy), 'n_mosek': int(n_mosek),
        'n_sdp_skipped': int(n_sdp_skipped),
        'n_no_prune': int(n_no_prune),
        't_t1_total_s': float(t_t1),
        't_lp_total_s': float(t_lp),
        't_cvxpy_total_s': float(t_cvxpy),
        't_mosek_total_s': float(t_mosek),
        'cvxpy_avg_ms': float(1000 * cvxpy_arr.mean()),
        'cvxpy_max_ms': float(1000 * cvxpy_arr.max()),
        'mosek_avg_ms': float(1000 * mosek_arr.mean()),
        'mosek_max_ms': float(1000 * mosek_arr.max()),
        'soundness_checked': int(soundness_checked),
        'soundness_violations': soundness_violations,
    }


if __name__ == '__main__':
    results = []
    for nh, mv, cv in [
        (2, 30, 1.20),
        (3, 10, 1.20),
        (3, 10, 1.28),
        (3, 20, 1.28),
        (4, 10, 1.28),
    ]:
        try:
            r = assess(nh, mv, cv, max_parents=100)
            if r:
                results.append(r)
        except Exception as e:
            print(f"ERROR on ({nh},{mv},{cv}): {e}")
            results.append({'n_half': nh, 'm': mv, 'c_target': cv, 'error': str(e)})

    out = {
        'configs': results,
        'note': ('MOSEK direct Task API; per-call SDP times measured. '
                 'Compares against CVXPY+SCS implementation in cascade_opts.py'),
    }
    with open(os.path.join(_dir, '_S1_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {os.path.join(_dir, '_S1_results.json')}")
