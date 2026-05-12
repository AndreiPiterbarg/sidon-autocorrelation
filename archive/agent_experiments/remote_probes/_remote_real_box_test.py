"""Test K=0 vs full-PSD on REAL d=22 BnB stuck boxes.

Loads stuck_boxes_w*.npz (real residual boxes from a d=22 t=1.281 BnB
run). Picks 2 boxes that are LP-failing (LP val < target). Runs both
the K=0 fast SDP and the full-PSD SDP on each. Reports verdicts and
solve times so we can see the actual tightness loss.
"""
import sys, glob, os, time
import numpy as np
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)
from interval_bnb.windows import build_windows
from interval_bnb.bound_epigraph import bound_epigraph_lp_float
from interval_bnb.bound_sdp_escalation_fast import (
    build_sdp_escalation_cache_fast as build_cache_fast,
    bound_sdp_escalation_lb_float_fast,
)


def load_stuck_boxes(d=22, max_files=20):
    """Load real stuck (in_flight) boxes from BnB dump files."""
    files = sorted(glob.glob('stuck_boxes_w*.npz'))[:max_files]
    boxes = []
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            # Try different keys.
            for key in ('lo', 'hi', 'boxes', 'lo_int', 'hi_int', 'data'):
                if key in data.files:
                    pass
            keys = list(data.files)
            print(f"  {f}: keys={keys}", flush=True)
            # Common structure: arrays of (n_boxes, d) for lo and hi
            if 'lo' in keys and 'hi' in keys:
                lo_arr = data['lo']
                hi_arr = data['hi']
                if lo_arr.ndim == 2:
                    for i in range(lo_arr.shape[0]):
                        boxes.append((lo_arr[i].astype(np.float64),
                                      hi_arr[i].astype(np.float64), f, i))
                else:
                    boxes.append((lo_arr.astype(np.float64),
                                  hi_arr.astype(np.float64), f, 0))
            elif 'lo_int' in keys and 'hi_int' in keys:
                SCALE = 2**60
                lo_int = data['lo_int']
                hi_int = data['hi_int']
                if lo_int.ndim == 2:
                    for i in range(lo_int.shape[0]):
                        lo_f = np.array([float(x) / SCALE for x in lo_int[i]],
                                         dtype=np.float64)
                        hi_f = np.array([float(x) / SCALE for x in hi_int[i]],
                                         dtype=np.float64)
                        boxes.append((lo_f, hi_f, f, i))
        except Exception as e:
            print(f"  {f}: {type(e).__name__}: {e}", flush=True)
    return boxes


def main():
    d = 22
    target = 1.281
    print(f"=== Real-box K=0 vs full-PSD test, d={d}, target={target} ===",
          flush=True)
    windows = build_windows(d)
    boxes = load_stuck_boxes(d=d)
    print(f"\nLoaded {len(boxes)} boxes from dumps.", flush=True)

    # Filter to LP-failing boxes within the SDP-attemptable zone.
    candidates = []
    for (lo, hi, src, idx) in boxes:
        if lo.shape[0] != d:
            continue
        if lo.sum() > 1.0 or hi.sum() < 1.0:
            continue
        try:
            lp = bound_epigraph_lp_float(lo, hi, windows, d)
        except Exception:
            continue
        # We want boxes where LP < target (LP-failing) but close.
        if target - 0.05 <= lp < target:
            hw = float((hi - lo).max() / 2)
            candidates.append((lp, lo, hi, src, idx, hw))
            if len(candidates) >= 8:
                break

    candidates.sort(key=lambda x: x[0], reverse=True)
    test_boxes = candidates[:2]
    print(f"\nFiltered: {len(candidates)} LP-failing boxes in stress zone "
          f"(target-0.05 <= LP < target). Using top 2:", flush=True)
    for k, (lp, lo, hi, src, idx, hw) in enumerate(test_boxes):
        print(f"  box {k}: LP={lp:.6f} hw={hw:.4f} src={src}#{idx}", flush=True)
    if not test_boxes:
        print("NO LP-failing boxes found. Cannot run tightness test.",
              flush=True)
        return

    # Build cache once.
    print(f"\nBuilding shared cache (target={target})...", flush=True)
    t0 = time.time()
    cache = build_cache_fast(d, windows, target=target)
    print(f"  cache build: {time.time()-t0:.2f}s", flush=True)

    # For each test box, try K=0 then K=946 (full PSD) — both at 48 threads.
    K_values = [0, 16, 64, 946]
    print(f"\n--- Running K-sweep on each box ---", flush=True)
    for k_box, (lp, lo, hi, src, idx, hw) in enumerate(test_boxes):
        print(f"\n## BOX {k_box}: LP={lp:.6f} hw={hw:.4f} src={src}#{idx}",
              flush=True)
        for K in K_values:
            t0 = time.time()
            res = bound_sdp_escalation_lb_float_fast(
                lo, hi, windows, d, cache=cache,
                target=target,
                n_window_psd_cones=K,
                time_limit_s=600.0, n_threads=48,
            )
            dt = time.time() - t0
            v = res.get('verdict', 'na')
            lam = res.get('lambda_star', float('nan'))
            print(f"  K={K:>3}  t={dt:6.1f}s  verdict={v:<10}  lambda*={lam:.4f}",
                  flush=True)


if __name__ == '__main__':
    main()
