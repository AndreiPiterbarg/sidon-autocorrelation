"""Re-run the push experiment with the WINDOWED kernels (prune_A, prune_F),
which the agent investigation predicts are sound (in contrast to the unsound
pointeval kernel prune_P).

Hypothesis under test: the windowed cascade should NOT over-certify past
the MV upper bound 1.5098. If A/F stop pruning at some c* <= 1.5098, the
windowed chain is sound and the project's 1.28 / 1.30 claims via the W-
refined main theorem are valid. If A/F also prune through 1.5098, the issue
is deeper than just pointeval.

Phases:
  Phase 1: (n=2, m=20) full enumeration, c sweep 1.20 → 1.55 step 0.01
  Phase 2: (n=2, m=40) full enumeration, c sweep 1.20 → 1.55 step 0.01
  Phase 3: tight refinement near the breakpoint
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "cloninger-steinerberger"))
sys.path.insert(0, os.path.join(_HERE, "cloninger-steinerberger", "cpu"))
from compositions import generate_compositions_batched
from _M1_bench import prune_A, prune_F, prune_P  # windowed + pointeval

print("[warmup]", flush=True)
t0 = time.time()
warm = np.array([[8, 8, 8, 8]], dtype=np.int32)
_ = prune_A(warm, 2, 4, 1.0)
_ = prune_F(warm, 2, 4, 1.0)
_ = prune_P(warm, 2, 4, 1.0)
print(f"[warmup] {time.time()-t0:.1f}s", flush=True)


def full_enum(n_half, m, c_target, kernels):
    """Run all kernels on full enumeration; return survivor counts."""
    d = 2 * n_half
    S = 4 * n_half * m
    counts = {name: 0 for name in kernels}
    total = 0
    for batch in generate_compositions_batched(d, S, batch_size=300_000):
        batch = batch.astype(np.int32)
        total += len(batch)
        for name, ker in kernels.items():
            s = ker(batch, n_half, m, c_target)
            counts[name] += int(s.sum())
    return total, counts


def sweep(n_half, m, c_lo, c_hi, c_step, label):
    print(f"\n{'='*78}", flush=True)
    print(f"{label}", flush=True)
    print(f"{'='*78}", flush=True)
    kernels = {"A": prune_A, "F": prune_F, "P": prune_P}
    print(f"{'c_target':>10}  {'total':>13}  "
          f"{'A_surv':>10}  {'F_surv':>10}  {'P_surv':>10}  {'elapsed':>8}",
          flush=True)
    breakpoints = {name: None for name in kernels}
    c = c_lo
    while c <= c_hi + 1e-9:
        t0 = time.time()
        total, counts = full_enum(n_half, m, c, kernels)
        elapsed = time.time() - t0
        markers = []
        for name in kernels:
            if counts[name] > 0 and breakpoints[name] is None:
                breakpoints[name] = c
                markers.append(f"{name}!")
        marker_str = "  ← " + " ".join(markers) if markers else ""
        print(f"{c:>10.4f}  {total:>13,}  "
              f"{counts['A']:>10,}  {counts['F']:>10,}  {counts['P']:>10,}  "
              f"{elapsed:>7.2f}s{marker_str}", flush=True)
        # If all three have broken, no need to keep going
        if all(bp is not None for bp in breakpoints.values()) and c > min(
                bp for bp in breakpoints.values() if bp is not None) + 0.05:
            break
        c += c_step
    print(f"\n  breakpoints (first c with any survivor):", flush=True)
    for name, bp in breakpoints.items():
        if bp is None:
            print(f"    {name}: never broke up to {c_hi}", flush=True)
        else:
            print(f"    {name}: {bp:.4f}  (highest 0-survivor c ≈ "
                  f"{bp - c_step:.4f})", flush=True)
    return breakpoints


# --- Phase 1: n=2, m=20 -------------------------------------------------
bp1 = sweep(2, 20, 1.20, 1.65, 0.01,
            "PHASE 1 — n=2 m=20 step 0.01 (708,561 comps per c)")

# --- Phase 2: n=2, m=40 -------------------------------------------------
bp2 = sweep(2, 40, 1.20, 1.65, 0.01,
            "PHASE 2 — n=2 m=40 step 0.01 (5,564,321 comps per c)")

# --- Final summary -----------------------------------------------------
print(f"\n{'#'*78}", flush=True)
print(f"# FINAL SUMMARY", flush=True)
print(f"{'#'*78}", flush=True)
print(f"  MV upper bound:  C_{{1a}} ≤ 1.5098 (Matolcsi-Vinuesa 2010)", flush=True)
print(f"  Existing claim:  C_{{1a}} ≥ 1.2802 (via W-refined chain)", flush=True)
print(f"", flush=True)
print(f"  (n=2, m=20):  A breaks at {bp1['A']}  F breaks at {bp1['F']}  "
      f"P breaks at {bp1['P']}", flush=True)
print(f"  (n=2, m=40):  A breaks at {bp2['A']}  F breaks at {bp2['F']}  "
      f"P breaks at {bp2['P']}", flush=True)

print(f"", flush=True)
print(f"  Diagnosis:", flush=True)
for cfg, bp in [("(n=2, m=20)", bp1), ("(n=2, m=40)", bp2)]:
    for name in ("A", "F", "P"):
        if bp[name] is None:
            verdict = "OVER-CERTIFIES (no breakpoint found)"
        elif bp[name] > 1.5098 + 0.005:
            verdict = f"OVER-CERTIFIES past MV (broke at {bp[name]:.4f}, > 1.5098)"
        elif bp[name] >= 1.27:
            verdict = f"sound at this resolution (broke at {bp[name]:.4f})"
        else:
            verdict = f"breaks early ({bp[name]:.4f}) — bound is loose here"
        print(f"    {cfg} {name}: {verdict}", flush=True)

with open("_windowed_push_max.json", "w") as fp:
    json.dump({"phase1": bp1, "phase2": bp2}, fp, indent=2,
              default=lambda x: float(x) if isinstance(x, np.floating) else x)
print(f"\nWrote _windowed_push_max.json", flush=True)
