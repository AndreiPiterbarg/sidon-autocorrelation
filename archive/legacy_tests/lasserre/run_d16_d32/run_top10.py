#!/usr/bin/env python
"""Top 10 cascade experiments — FEASIBLE configs only.

d0=2 is infeasible (billions of survivors at d=8 since val(8)<1.30).
Focus on d0=4,6,8,10 where L0 prunes heavily, leaving manageable survivors.

Uses: run_cascade.py --coarse --S <val>
  -> Theorem 1 (no correction), Gray code kernel, sound box cert
"""
import subprocess, sys, time, os, shutil, glob

os.chdir("/workspace/sidon-autocorrelation")

experiments = [
    # (id, d0, S, c_target, max_levels)
    # --- d0=4: moderate L0, cascade to d=16 ---
    (1, 4, 20, 1.30, 8),
    (2, 4, 30, 1.30, 8),
    (3, 4, 50, 1.30, 8),

    # --- d0=6: heavy L0 pruning, cascade to d=12-24 ---
    (4, 6, 20, 1.30, 6),
    (5, 6, 30, 1.30, 6),
    (6, 6, 50, 1.30, 6),

    # --- d0=8: very heavy L0, cascade to d=16 ---
    (7, 8, 20, 1.30, 5),
    (8, 8, 30, 1.30, 5),
    (9, 8, 50, 1.30, 5),

    # --- d0=10: almost all pruned at L0, cascade to d=20 ---
    (10, 10, 20, 1.30, 4),
]

print("=" * 70)
print("TOP 10 CASCADE EXPERIMENTS — c_target=1.30")
print("COARSE GRID: Theorem 1, no correction, Gray code kernel")
print("d0=4,6,8,10 (skip d0=2 — too many survivors)")
print("=" * 70, flush=True)

for exp_id, d0, S, c, max_lev in experiments:
    # Clean stale checkpoints between experiments
    for f in glob.glob("data/checkpoint_*.npy") + glob.glob("data/checkpoint_*.json"):
        try:
            os.remove(f)
        except OSError:
            pass

    print(f"\n\n{'#'*70}")
    print(f"# EXP {exp_id}: d0={d0}, S={S}, c_target={c}, max_levels={max_lev}")
    print(f"# MODE: --coarse --S {S} (Theorem 1, no correction)")
    print(f"{'#'*70}", flush=True)

    t0 = time.time()
    cmd = [
        sys.executable, "-u",
        "cloninger-steinerberger/cpu/run_cascade.py",
        "--coarse",
        "--d0", str(d0),
        "--S", str(S),
        "--c_target", str(c),
        "--max_levels", str(max_lev),
    ]
    try:
        subprocess.run(cmd, check=False, timeout=7200)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 7200s", flush=True)
    elapsed = time.time() - t0
    print(f"\n  Experiment {exp_id} elapsed: {elapsed:.1f}s", flush=True)

print("\n\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 70, flush=True)
