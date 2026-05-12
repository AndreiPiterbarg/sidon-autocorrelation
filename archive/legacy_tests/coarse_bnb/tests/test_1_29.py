#!/usr/bin/env python
"""Test c=1.29: cascade + box cert analysis at multiple S values."""
import subprocess, sys, time, os

os.chdir("/workspace/sidon-autocorrelation")

configs = [
    # (d0, S, c_target, max_levels)
    (4, 20, 1.29, 8),
    (4, 30, 1.29, 8),
    (4, 50, 1.29, 8),
    (4, 75, 1.29, 8),
    (6, 20, 1.29, 6),
    (6, 30, 1.29, 6),
    (6, 50, 1.29, 6),
    (8, 20, 1.29, 5),
    (8, 30, 1.29, 5),
]

print("=" * 70)
print("CASCADE TESTS at c_target=1.29")
print("val(16)=1.319, margin=0.029 (vs 0.019 at c=1.30)")
print("=" * 70, flush=True)

for d0, S, c, max_lev in configs:
    print(f"\n{'#'*70}")
    print(f"# d0={d0}, S={S}, c={c}, max_levels={max_lev}")
    print(f"{'#'*70}", flush=True)

    t0 = time.time()
    cmd = [
        sys.executable, "-u",
        "cloninger-steinerberger/cpu/run_cascade.py",
        "--coarse", "--d0", str(d0), "--S", str(S),
        "--c_target", str(c), "--max_levels", str(max_lev),
    ]
    # Clean checkpoints
    for f in os.listdir("data"):
        if f.startswith("checkpoint_"):
            try: os.remove(os.path.join("data", f))
            except: pass

    try:
        subprocess.run(cmd, check=False, timeout=3600)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT", flush=True)
    print(f"\n  Elapsed: {time.time()-t0:.1f}s", flush=True)

print("\n" + "=" * 70)
print("ALL DONE")
print("=" * 70, flush=True)
