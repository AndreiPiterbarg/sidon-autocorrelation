#!/usr/bin/env python
"""L2 CG sweep — fast version. 8 bisection steps, aggressive CG."""
import sys, os, time, json, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

try:
    import psutil
    def mem_gb():
        m = psutil.virtual_memory()
        return m.total / 1024**3, m.available / 1024**3, psutil.Process().memory_info().rss / 1024**3
except ImportError:
    def mem_gb():
        return 0, 0, 0

from lasserre_scalable import solve_cg

val_d = {4:1.102, 6:1.171, 8:1.205, 10:1.241, 12:1.271, 14:1.284, 16:1.319}

# Fewer bisect steps (8 not 15), more windows per round (30), fewer rounds
configs = [
    (16, 2, 8, 3, 30, "L2 d=16"),
    (24, 2, 8, 3, 30, "L2 d=24"),
    (32, 2, 8, 2, 40, "L2 d=32"),
    (48, 2, 6, 2, 50, "L2 d=48"),
    (64, 2, 6, 2, 50, "L2 d=64"),
]

results = []
print("=" * 70)
print("L2 CG SWEEP — FAST (reduced bisection)")
tot, avail, rss = mem_gb()
print(f"RAM: {tot:.0f} GB total, {avail:.0f} GB available")
print("=" * 70, flush=True)

for d, order, n_bisect, cg_rounds, cg_add, desc in configs:
    print(f"\n{'#'*70}")
    print(f"# {desc}  (bisect={n_bisect}, rounds={cg_rounds}, add={cg_add})")
    tot, avail, rss = mem_gb()
    print(f"# RAM: avail={avail:.0f}GB  RSS={rss:.1f}GB")
    print(f"{'#'*70}\n", flush=True)

    if avail < 15:
        print("  SKIP: <15 GB available", flush=True)
        results.append({'d': d, 'desc': desc, 'status': 'skipped-mem'})
        continue

    t0 = time.time()
    try:
        r = solve_cg(d, c_target=1.10, order=order, n_bisect=n_bisect,
                     add_upper_loc=True, cg_rounds=cg_rounds,
                     cg_add_per_round=cg_add, conv_tol=1e-7, verbose=True)
        elapsed = time.time() - t0
        tot, avail, rss = mem_gb()
        r['wall_s'] = elapsed
        r['peak_rss_gb'] = rss
        r['desc'] = desc
        r['status'] = 'ok'
        v = val_d.get(d, 0)
        if v > 1:
            r['gap_closure'] = (r['lb'] - 1.0) / (v - 1.0) * 100
        results.append(r)
        print(f"\n  => d={d} lb={r['lb']:.8f} time={elapsed:.1f}s "
              f"RSS={rss:.1f}GB active={r['n_active_windows']}", flush=True)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAILED ({elapsed:.1f}s): {e}", flush=True)
        traceback.print_exc()
        results.append({'d': d, 'desc': desc, 'status': str(e), 'wall_s': elapsed})
        if 'emory' in str(e):
            print("  OOM — stopping.", flush=True)
            break

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Config':<15} {'lb':>10} {'gap%':>8} {'windows':>10} {'time':>10} {'RSS':>7}")
print("-" * 70)
for r in results:
    lb = r.get('lb', 0)
    gc = r.get('gap_closure', 0)
    nw = r.get('n_active_windows', 0)
    nt = r.get('n_win_total', 0)
    t = r.get('wall_s', 0)
    rss = r.get('peak_rss_gb', 0)
    lb_s = f"{lb:.6f}" if lb else "---"
    gc_s = f"{gc:.1f}%" if gc else "---"
    w_s = f"{nw}/{nt}" if nt else "---"
    print(f"{r['desc']:<15} {lb_s:>10} {gc_s:>8} {w_s:>10} {t:>9.0f}s {rss:>6.1f}G")
print("=" * 70)

outpath = os.path.join('data', f"l2_cg_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json")
os.makedirs('data', exist_ok=True)
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved: {outpath}")
