"""SDP audit on the 748 stuck d=10 boxes.

Samples 50 boxes uniformly at random (seed=42), runs the fast SDP cert
at K=32 and K=999 (full-PSD baseline), with target = 12/10 = 1.2.
Reports cert counts, median wall-clock, and 5 boxes that fail at K=999.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from interval_bnb.box import Box
from interval_bnb.windows import build_windows
from interval_bnb.bound_sdp_escalation_fast import (
    build_sdp_escalation_cache_fast,
    bound_sdp_escalation_int_ge_fast,
)


def main():
    npz = np.load(os.path.join(_HERE, "stuck_d10_master_queue.npz"))
    lo_all = npz["lo"]   # (748, 10)
    hi_all = npz["hi"]   # (748, 10)
    depths = npz["depths"]
    n_total = lo_all.shape[0]
    print(f"loaded {n_total} stuck boxes; lo shape {lo_all.shape}")

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_total, size=50, replace=False)
    sample_idx_sorted = sorted(int(i) for i in sample_idx)
    print(f"sampled 50 indices (seed=42): {sample_idx_sorted[:10]}...")

    d = 10
    target_num, target_den = 12, 10
    target_f = float(target_num) / float(target_den)

    print("building windows + SDP cache...")
    windows = build_windows(d)
    cache = build_sdp_escalation_cache_fast(d, windows, target=target_f)
    print(f"  windows: {len(windows)}; nontrivial: {len(cache['P']['nontrivial_windows'])}")

    results = []
    Ks = [32, 999]

    for k_idx, K in enumerate(Ks):
        print(f"\n=== K = {K} ===")
        for j, i in enumerate(sample_idx_sorted):
            lo = lo_all[i].astype(np.float64)
            hi = hi_all[i].astype(np.float64)
            B = Box(lo=lo, hi=hi)
            lo_int, hi_int = B.to_ints()
            max_w = float((hi - lo).max())
            t0 = time.time()
            try:
                cert = bound_sdp_escalation_int_ge_fast(
                    lo_int, hi_int, windows, d,
                    target_num=target_num, target_den=target_den,
                    cache=cache,
                    n_window_psd_cones=K,
                    n_threads=1,
                    time_limit_s=15.0,
                )
                ok = bool(cert)
                err = None
            except Exception as e:
                ok = False
                err = repr(e)
            wall = time.time() - t0

            if k_idx == 0:
                results.append({
                    "box_idx": int(i),
                    "depth": int(depths[i]),
                    "max_w": max_w,
                    "K32_cert": ok,
                    "K32_wall_s": wall,
                    "K32_err": err,
                })
            else:
                # find existing entry by box_idx
                rec = next(r for r in results if r["box_idx"] == int(i))
                rec["K999_cert"] = ok
                rec["K999_wall_s"] = wall
                rec["K999_err"] = err

            tag = "CERT" if ok else "FAIL"
            print(f"  [{j+1:2d}/50] box={i:4d} depth={int(depths[i]):3d} maxw={max_w:.3e} "
                  f"K={K} {tag} wall={wall:.2f}s")

    # Aggregate
    n_cert_K32 = sum(1 for r in results if r["K32_cert"])
    n_cert_K999 = sum(1 for r in results if r.get("K999_cert"))
    walls_K32 = sorted(r["K32_wall_s"] for r in results)
    walls_K999 = sorted(r["K999_wall_s"] for r in results)
    med_K32 = walls_K32[len(walls_K32) // 2]
    med_K999 = walls_K999[len(walls_K999) // 2]

    failed_K999 = [r for r in results if not r.get("K999_cert")]
    failed_K999.sort(key=lambda r: -r["max_w"])

    summary = {
        "n_sample": 50,
        "target": target_f,
        "n_cert_K32": n_cert_K32,
        "n_cert_K999": n_cert_K999,
        "median_wall_K32_s": med_K32,
        "median_wall_K999_s": med_K999,
        "failed_K999_top5": [
            {"box_idx": r["box_idx"], "depth": r["depth"], "max_w": r["max_w"]}
            for r in failed_K999[:5]
        ],
        "n_failed_K999": len(failed_K999),
        "results": results,
    }

    out_path = os.path.join(_HERE, "sdp_stuck_audit.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_path}")

    print("\n=== SUMMARY ===")
    print(f"  K=32  certs: {n_cert_K32}/50  median wall: {med_K32:.2f}s")
    print(f"  K=999 certs: {n_cert_K999}/50  median wall: {med_K999:.2f}s")
    print(f"  K=999 failed: {len(failed_K999)} boxes")
    print(f"  top 5 failures (by max_w):")
    for r in failed_K999[:5]:
        print(f"    box_idx={r['box_idx']:4d} depth={r['depth']:3d} max_w={r['max_w']:.3e}")


if __name__ == "__main__":
    main()
