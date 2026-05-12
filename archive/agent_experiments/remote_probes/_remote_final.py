"""FINAL test: per-d feasibility + cert rate report.

For each d in [10, 16, 22, 30]:
  Phase 1: smoke — one solve on a random Dirichlet hw=0.025 box.
           Reports solve time, status, SDP value.
  Phase 2: cert rate — sample 10 random Dirichlet boxes, check LP cert
           and SDP cert at target=1.281 with cushion. Reports
           (LP_cert, SDP_cert_when_LP_failed, total).

This is the definitive feasibility report.
"""
import sys
import time
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)
from interval_bnb.windows import build_windows
from interval_bnb.bound_epigraph import bound_epigraph_lp_float
from interval_bnb.bound_sdp_escalation import (
    build_sdp_escalation_cache, bound_sdp_escalation_lb_float, _safe_cushion,
)


def _smoke(d, hw, target, cache, windows, n_attempts=10):
    """Test up to n_attempts random boxes; collect SDP cert rate.
    Cap each solve at 60s.
    """
    rng = np.random.default_rng(0)
    n_lp_cert = n_lp_far = 0
    sdp_records = []
    cushion_used = 0.0
    for trial in range(50):
        if len(sdp_records) >= n_attempts:
            break
        mu = rng.dirichlet(np.ones(d))
        if mu[0] > mu[-1]:
            mu = mu[::-1]
        lo = np.maximum(mu - hw, 0.0)
        hi = np.minimum(mu + hw, 1.0)
        if lo.sum() > 1.0 or hi.sum() < 1.0:
            continue
        lp = bound_epigraph_lp_float(lo, hi, windows, d)
        if lp >= target:
            n_lp_cert += 1
            continue
        if lp < target - 0.05:
            n_lp_far += 1
            continue
        # SDP escalation
        t0 = time.time()
        res = bound_sdp_escalation_lb_float(lo, hi, windows, d, cache=cache,
                                             time_limit_s=60.0)
        dt = time.time() - t0
        if res['is_feasible_status']:
            cushion = _safe_cushion(res['r_prim'], res['r_dual'],
                                     res['duality_gap'])
            lb_safe = float(res['obj_val_dual']) - cushion
            cert = lb_safe >= target
            sdp_records.append((True, cert, lb_safe, dt, res['status']))
            cushion_used = max(cushion_used, cushion)
        else:
            sdp_records.append((False, False, float('nan'), dt,
                                 res['status']))
        print(f"    trial {trial:>2}: LP={lp:.4f} -> SDP_cold t={dt:.1f}s "
              f"status={res['status']:<25} obj_dual={res.get('obj_val_dual')}",
              flush=True)
    return {
        'lp_cert': n_lp_cert,
        'lp_far': n_lp_far,
        'sdp_records': sdp_records,
        'cushion_max': cushion_used,
    }


def main():
    target = 1.281
    print(f"\n{'=' * 70}", flush=True)
    print(f"FINAL FEASIBILITY REPORT — Lasserre order-2 SDP escalation", flush=True)
    print(f"target={target}, hw=0.025, n_attempts=10 per d", flush=True)
    print(f"{'=' * 70}\n", flush=True)
    for d in (10, 16, 22, 30):
        print(f"--- d={d} ---", flush=True)
        windows = build_windows(d)
        t0 = time.time()
        try:
            cache = build_sdp_escalation_cache(d, windows)
        except Exception as e:
            print(f"  cache build FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        t_cache = time.time() - t0
        n_cliques = len(cache['mom_blocks'])
        max_B = max(b.n_cb for b in cache['mom_blocks'])
        print(f"  cache: {t_cache:.2f}s, n_y={cache['n_y']}, bw={cache['bandwidth']}, "
              f"cliques={n_cliques}, max_B={max_B}, n_W={cache['n_W_kept']}",
              flush=True)

        result = _smoke(d, 0.025, target, cache, windows, n_attempts=10)
        n_sdp_total = len(result['sdp_records'])
        n_sdp_solved = sum(1 for ok, _, _, _, _ in result['sdp_records'] if ok)
        n_sdp_cert = sum(1 for ok, c, _, _, _ in result['sdp_records']
                         if ok and c)
        sdp_times = [t for ok, _, _, t, _ in result['sdp_records'] if ok]
        med_t = np.median(sdp_times) if sdp_times else float('nan')
        max_t = max(sdp_times) if sdp_times else float('nan')
        print(f"  RESULTS: LP cert {result['lp_cert']}, "
              f"LP-too-far {result['lp_far']}, "
              f"SDP attempted {n_sdp_total}, "
              f"SDP solved {n_sdp_solved}, "
              f"SDP cert {n_sdp_cert}/{n_sdp_total}",
              flush=True)
        if sdp_times:
            print(f"  TIMING : SDP median {med_t:.1f}s, max {max_t:.1f}s, "
                  f"cushion_max {result['cushion_max']:.2e}", flush=True)
        print(flush=True)


if __name__ == '__main__':
    main()
