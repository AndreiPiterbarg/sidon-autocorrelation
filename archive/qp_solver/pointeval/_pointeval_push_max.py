"""Push the point-eval bound as high as it will go.

Strategy:
 Phase 1. At (n=2, m=20) — the existing axiom config — sweep c_target
 in fine steps from 1.30 upward until P leaves survivors.
 Print per-step: total enumerated, P_survivors, sample
 survivor c, timing.

 Phase 2. At (n=2, m=40), repeat — finer height resolution should let
 us push the certified bound higher (smaller correction).

 Phase 3. At (n=3, m=20), repeat — bigger d means more bins; bound
 should also get tighter.

 Phase 4. At (n=3, m=40) — even finer.

 Phase 5. Push d further if interesting (n=4, m=20).

For each phase we run FULL enumeration (NOT palindromic). The first
c_target with any survivor is the upper limit at that resolution.

Heavy logging throughout.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "cloninger-steinerberger"))
sys.path.insert(0, os.path.join(_HERE, "cloninger-steinerberger", "cpu"))
from compositions import generate_compositions_batched
from _M1_bench import prune_P

# JIT warmup
print("[warmup] compiling Numba kernels...", flush=True)
t0 = time.time()
_ = prune_P(np.array([[8, 8, 8, 8]], dtype=np.int32), 2, 4, 1.0)
print(f"[warmup] done in {time.time()-t0:.1f}s", flush=True)


def full_enum_prune(n_half, m, c_target, batch_size=300_000,
 max_survivors_to_collect=10):
 """Full non-palindromic enumeration; run prune_P; return summary."""
 d = 2 * n_half
 S = 4 * n_half * m
 total = 0
 n_surv = 0
 survivors = []
 t0 = time.time()
 for batch in generate_compositions_batched(d, S, batch_size=batch_size):
 batch = batch.astype(np.int32)
 sP = prune_P(batch, n_half, m, c_target)
 total += len(batch)
 n_surv += int(sP.sum())
 if sP.any() and len(survivors) < max_survivors_to_collect:
 for i in np.where(sP)[0]:
 if len(survivors) >= max_survivors_to_collect:
 break
 survivors.append(tuple(int(x) for x in batch[i]))
 return {
 "n_half": n_half, "m": m, "c_target": float(c_target),
 "d": d, "S": S, "total": total, "survivors": n_surv,
 "elapsed": time.time() - t0, "survivor_samples": survivors,
 }


def sweep_c(n_half, m, c_lo, c_hi, c_step, label):
 """Sweep c_target in [c_lo, c_hi] with given step. Print per-step."""
 print(f"\n{'='*72}", flush=True)
 print(f"{label}", flush=True)
 print(f" config: n_half={n_half} m={m} d={2*n_half} S={4*n_half*m}", flush=True)
 print(f" sweep: c_target ∈ [{c_lo}, {c_hi}] step {c_step}", flush=True)
 print(f"{'='*72}", flush=True)
 print(f"{'c_target':>10} {'total':>13} {'P_surv':>10} {'pct':>9} {'elapsed':>8}", flush=True)

 breakpoint_c = None
 results = []
 c = c_lo
 while c <= c_hi + 1e-12:
 r = full_enum_prune(n_half, m, c)
 results.append(r)
 pct = 100.0 * r["survivors"] / r["total"] if r["total"] > 0 else 0.0
 marker = "" if r["survivors"] == 0 else " ← FIRST SURVIVORS"
 print(f"{c:>10.4f} {r['total']:>13,} {r['survivors']:>10,} "
 f"{pct:>8.4f}% {r['elapsed']:>7.2f}s{marker}", flush=True)
 if r["survivors"] > 0 and breakpoint_c is None:
 breakpoint_c = c
 for s in r["survivor_samples"]:
 print(f" survivor: {s}", flush=True)
 # Continue a bit further to see how survivors grow
 c += c_step
 if c > c_hi:
 break
 continue
 c += c_step

 if breakpoint_c is None:
 print(f"\n NO SURVIVORS up to c_target = {c_hi}", flush=True)
 print(f" ⇒ certified at this resolution: c ≤ {c_hi}", flush=True)
 else:
 # Highest c with 0 survivors = breakpoint - step
 last_clean = breakpoint_c - c_step
 print(f"\n ⇒ HIGHEST c with 0 survivors at this resolution: "
 f"c_target = {last_clean:.4f}", flush=True)
 print(f" ⇒ FIRST c with survivors: "
 f"c_target = {breakpoint_c:.4f}", flush=True)
 return results, breakpoint_c


# ─── Phase 1: (n=2, m=20) — existing axiom config ──────────────────────
results_phase = {}
results_phase["n2m20"], bp_n2m20 = sweep_c(
 2, 20, 1.30, 2.05, 0.05,
 label="PHASE 1 — base config (n=2, m=20), step 0.05")
# refine around the breakpoint
if bp_n2m20 is not None:
 results_phase["n2m20_fine"], bp_n2m20_fine = sweep_c(
 2, 20, max(1.30, bp_n2m20 - 0.05), bp_n2m20 + 0.01, 0.005,
 label="PHASE 1b — fine sweep around n=2 m=20 breakpoint")

# ─── Phase 2: (n=2, m=40) — finer m, smaller correction ────────────────
results_phase["n2m40"], bp_n2m40 = sweep_c(
 2, 40, 1.30, 2.05, 0.05,
 label="PHASE 2 — finer m (n=2, m=40), step 0.05")
if bp_n2m40 is not None:
 results_phase["n2m40_fine"], _ = sweep_c(
 2, 40, max(1.30, bp_n2m40 - 0.05), bp_n2m40 + 0.01, 0.005,
 label="PHASE 2b — fine sweep around n=2 m=40 breakpoint")

# ─── Phase 3: (n=3, m=20) — bigger d ───────────────────────────────────
results_phase["n3m20"], bp_n3m20 = sweep_c(
 3, 20, 1.30, 1.80, 0.05,
 label="PHASE 3 — bigger d (n=3, m=20), step 0.05")
if bp_n3m20 is not None:
 results_phase["n3m20_fine"], _ = sweep_c(
 3, 20, max(1.30, bp_n3m20 - 0.05), bp_n3m20 + 0.01, 0.005,
 label="PHASE 3b — fine sweep around n=3 m=20 breakpoint")

# ─── Final summary ──────────────────────────────────────────────────────
print(f"\n{'#'*72}", flush=True)
print(f"# FINAL SUMMARY", flush=True)
print(f"{'#'*72}", flush=True)
print(f" (n=2, m=20): highest 0-survivor c ≈ "
 f"{(bp_n2m20 - 0.05) if bp_n2m20 else '> 2.05'}", flush=True)
print(f" (n=2, m=40): highest 0-survivor c ≈ "
 f"{(bp_n2m40 - 0.05) if bp_n2m40 else '> 2.05'}", flush=True)
print(f" (n=3, m=20): highest 0-survivor c ≈ "
 f"{(bp_n3m20 - 0.05) if bp_n3m20 else '> 1.80'}", flush=True)

with open("_pointeval_push_max.json", "w") as fp:
 json.dump(results_phase, fp, indent=2,
 default=lambda x: int(x) if isinstance(x, np.integer) else x)
print(f"\nWrote _pointeval_push_max.json", flush=True)
