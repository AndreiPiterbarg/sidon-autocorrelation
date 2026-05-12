"""Test Lasserre order-2 SDP cert rate at the d=30 transition zone (hw=0.025).

If SDP cert rate at hw=0.025 d=30 is >>50%, then SDP escalation is the right
fix and unblocks d=30. If <=50%, SDP isn't enough and we need different strategy.

Uses lasserre_box_lb_float from interval_bnb/lasserre_cert.py.
"""
import numpy as np
import sys
import time

sys.path.insert(0, '.')
from interval_bnb.windows import build_windows
from interval_bnb.bound_epigraph import _solve_epigraph_lp
from interval_bnb.lasserre_cert import lasserre_box_lb_float


def random_simplex_box(hw, d, rng):
    for _ in range(20):
        mu_c = rng.dirichlet(np.ones(d))
        if mu_c[0] <= mu_c[-1]:
            break
        mu_c = mu_c[::-1]
    lo = np.maximum(0.0, mu_c - hw); hi = np.minimum(1.0, mu_c + hw)
    hi[0] = min(hi[0], 0.5)
    return lo, hi


def main():
    target = 1.281
    rng = np.random.default_rng(0)

    print('='*80)
    print('LASSERRE-2 SDP cert at d=30 transition zone (hw=0.025)')
    print('Test: among LP-failing boxes, does Lasserre-2 SDP cert?')
    print('='*80)

    # First gather LP-failing boxes
    d = 30
    windows = build_windows(d)
    print(f'Generating LP-failing boxes at d={d} hw=0.025...')

    failing_boxes = []
    n_total = 0
    for trial in range(50):
        if len(failing_boxes) >= 10:
            break
        lo, hi = random_simplex_box(0.025, d, rng)
        if lo.sum() > 1.0 or hi.sum() < 1.0:
            continue
        n_total += 1
        lp_val, *_ = _solve_epigraph_lp(np.asarray(lo, dtype=np.float64),
                                          np.asarray(hi, dtype=np.float64),
                                          windows, d)
        if lp_val < target:
            failing_boxes.append((lo, hi, lp_val))

    print(f'Found {len(failing_boxes)} LP-failing boxes from {n_total} trials')

    # Test Lasserre-2 SDP on each
    print(f'\nLasserre-2 SDP cert test (this is slow — 1-10s per box):')
    print(f'  box # | LP_val   | Lasserre_LB  | SDP_cert(t=1.281)? | time')

    sdp_cert_count = 0
    for k, (lo, hi, lp_val) in enumerate(failing_boxes):
        t0 = time.time()
        try:
            sdp_lb = lasserre_box_lb_float(lo, hi, windows, d, order=2,
                                            solver='MOSEK', verbose=False)
        except Exception as e:
            sdp_lb = float('-inf')
            print(f'  {k:>4}  | {lp_val:.4f} | SDP failed: {type(e).__name__}: {str(e)[:50]}')
            continue
        elapsed = time.time() - t0
        cert = sdp_lb >= target
        if cert: sdp_cert_count += 1
        print(f'  {k:>4}  | {lp_val:.4f} | {sdp_lb:.4f}      | {cert}             | {elapsed:.1f}s')

    n = len(failing_boxes)
    if n > 0:
        rate = 100.0 * sdp_cert_count / n
        print(f'\nLasserre-2 SDP cert rate on LP-failing boxes: {sdp_cert_count}/{n} = {rate:.1f}%')

        if rate >= 70:
            print('VERDICT: Lasserre-2 SDP IS the fix — would unblock d=30 BnB.')
        elif rate >= 30:
            print('VERDICT: Lasserre-2 helps but not dispositive. SDP escalation worth trying.')
        else:
            print('VERDICT: Lasserre-2 does NOT close the d=30 gap. Need different approach.')
    else:
        print('No failing boxes found — should have but did not. Diagnostic broken.')


if __name__ == '__main__':
    main()
