"""Adversarial proof attempt at c_target = 1.281.

For each (n_half, m) config:
 1. Enumerate ALL compositions of S = 4·n_half·m into d = 2·n_half nonneg
 integer bins (NOT just palindromic — that's a stricter test than the
 bench).
 2. Run prune_A, prune_D, prune_F, prune_P; log survivors and timing.
 3. If P leaves any survivors at L0, try one cascade level (refining each
 surviving parent into all valid children at d_child = 2·d_parent and
 m_child = m). Report which parents fail to fully prune their child set.
 4. Surface the actual survivor compositions (up to 20) for inspection.

Usage:
 PYTHONIOENCODING=utf-8 python _pointeval_1p281_proof.py
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "cloninger-steinerberger"))
sys.path.insert(0, os.path.join(_HERE, "cloninger-steinerberger", "cpu"))

from compositions import generate_compositions_batched
from _M1_bench import prune_A, prune_D, prune_F, prune_P

LOG = []

def log(msg=""):
 print(msg, flush=True)
 LOG.append(msg)


def warmup():
 log("[warmup] JIT-compiling kernels...")
 t0 = time.time()
 warm = np.array([[8, 8, 8, 8]], dtype=np.int32)
 _ = prune_A(warm, 2, 4, 1.0)
 _ = prune_D(warm, 2, 4, 1.0)
 _ = prune_F(warm, 2, 4, 1.0)
 _ = prune_P(warm, 2, 4, 1.0)
 log(f"[warmup] done in {time.time()-t0:.1f}s")


def run_full_l0(n_half, m, c_target):
 """Enumerate ALL compositions of S=4nm into d bins; run all prunes."""
 d = 2 * n_half
 S = 4 * n_half * m
 log("")
 log(f"════════════════════════════════════════════════════════════════")
 log(f" L0 FULL ENUMERATION: n_half={n_half} m={m} c_target={c_target}")
 log(f" d={d} S=4nm={S}")
 log(f"════════════════════════════════════════════════════════════════")

 n_total = 0
 n_A_surv = n_D_surv = n_F_surv = n_P_surv = 0
 t_A = t_D = t_F = t_P = 0.0
 P_survivors = [] # collect actual P survivors for inspection
 P_bug_vs_A = 0 # P kept that A pruned (would be SOUNDNESS violation)
 P_bug_vs_F = 0 # P kept that F pruned (also a problem -- F is sound)

 t_total_start = time.time()
 batch_count = 0
 for batch in generate_compositions_batched(d, S, batch_size=300_000):
 batch = batch.astype(np.int32)
 n_total += len(batch)
 batch_count += 1

 t0 = time.time(); sA = prune_A(batch, n_half, m, c_target); t_A += time.time()-t0
 t0 = time.time(); sD = prune_D(batch, n_half, m, c_target); t_D += time.time()-t0
 t0 = time.time(); sF = prune_F(batch, n_half, m, c_target); t_F += time.time()-t0
 t0 = time.time(); sP = prune_P(batch, n_half, m, c_target); t_P += time.time()-t0

 n_A_surv += int(sA.sum())
 n_D_surv += int(sD.sum())
 n_F_surv += int(sF.sum())
 n_P_surv += int(sP.sum())

 # Soundness sanity (P should prune everything A/F prunes; P-survivors
 # should be a SUBSET of A-survivors and F-survivors)
 P_bug_vs_A += int((sP & ~sA).sum())
 P_bug_vs_F += int((sP & ~sF).sum())

 # Collect P survivors
 if sP.any() and len(P_survivors) < 100:
 idx = np.where(sP)[0]
 for i in idx:
 if len(P_survivors) >= 100:
 break
 P_survivors.append(tuple(int(x) for x in batch[i]))

 log(f" [batch {batch_count:>3}] n_total={n_total:>10,} "
 f"A_s={n_A_surv:>9,} D_s={n_D_surv:>9,} "
 f"F_s={n_F_surv:>9,} P_s={n_P_surv:>9,}")

 t_total = time.time() - t_total_start
 log(f"")
 log(f" ── L0 SUMMARY ───────────────────────────────────────────────")
 log(f" total compositions enumerated: {n_total:,} ({t_total:.2f}s)")
 log(f" prune A (W-refined): survivors={n_A_surv:,} "
 f"({100*n_A_surv/n_total:.4f}%) [{t_A:.2f}s]")
 log(f" prune D (variant D): survivors={n_D_surv:,} "
 f"({100*n_D_surv/n_total:.4f}%) [{t_D:.2f}s]")
 log(f" prune F (M1 LP): survivors={n_F_surv:,} "
 f"({100*n_F_surv/n_total:.4f}%) [{t_F:.2f}s]")
 log(f" prune P (point-eval): survivors={n_P_surv:,} "
 f"({100*n_P_surv/n_total:.4f}%) [{t_P:.2f}s]")
 log(f"")
 log(f" soundness sanity:")
 log(f" P-not-A = {P_bug_vs_A} (should be 0; nonzero ⇒ P keeps some c "
 f"A pruned — would be a soundness bug)")
 log(f" P-not-F = {P_bug_vs_F} (should be 0; F is sound)")
 if P_bug_vs_A > 0 or P_bug_vs_F > 0:
 log(f" *** SOUNDNESS WARNING: nonzero bug count ***")

 if n_P_surv == 0:
 log(f" P PRUNES ALL {n_total:,} COMPOSITIONS AT L0 ")
 else:
 log(f" P leaves {n_P_surv:,} survivors at L0 — try cascade L1.")
 log(f" Sample survivors (first 20):")
 for s in P_survivors[:20]:
 log(f" {s}")

 return {
 "n_half": n_half, "m": m, "c_target": c_target,
 "n_total": n_total,
 "A_surv": n_A_surv, "D_surv": n_D_surv,
 "F_surv": n_F_surv, "P_surv": n_P_surv,
 "P_bug_vs_A": P_bug_vs_A, "P_bug_vs_F": P_bug_vs_F,
 "t_A": t_A, "t_D": t_D, "t_F": t_F, "t_P": t_P,
 "wall": t_total,
 "P_survivors_sample": P_survivors,
 }


def cascade_l1(parent, n_half, m, c_target):
 """Try refining ONE parent c into all valid children at d_child=2d.

 Children allow ±1 deviation per bin pair (the canonical_discretization
 floor-rounding slack), with sum = 2*sum(parent). Run prune_P on ALL
 such children; report whether all are pruned.
 """
 d_parent = 2 * n_half
 d_child = 2 * d_parent
 n_half_child = d_parent # = 2*n_half
 sum_parent = int(parent.sum())
 sum_child = 2 * sum_parent

 # Generate all valid children: each pair (c[2i], c[2i+1]) summing to
 # 2*parent[i] ± 1. We enumerate the cartesian product.
 # Pair sums allowed: max(0, 2p-1)..(2p+1).
 pair_options = []
 for p in parent:
 p_int = int(p)
 sums = []
 for ps in range(max(0, 2*p_int - 1), 2*p_int + 2):
 for a in range(0, ps + 1):
 sums.append((a, ps - a))
 pair_options.append(sums)

 # Cartesian product is huge; cap at 50k for sanity
 from itertools import product
 children = []
 for combo in product(*pair_options):
 child = []
 for (a, b) in combo:
 child.append(a); child.append(b)
 if sum(child) == sum_child:
 children.append(child)
 if len(children) > 50_000:
 return None, len(children), "too many children to enumerate"
 if not children:
 return 0, 0, "no valid children"

 batch = np.array(children, dtype=np.int32)
 sP = prune_P(batch, n_half_child, m, c_target)
 n_surv = int(sP.sum())
 return n_surv, len(children), "ok"


if __name__ == "__main__":
 warmup()
 log(f"\n{'#'*72}")
 log(f"# c_target = 1.281 (target just above C&S 32/25 = 1.28)")
 log(f"# Adversarial: full enumeration, all four kernels, soundness sanity")
 log(f"{'#'*72}")

 results = []
 configs = [
 (2, 8), # tiny: S=64, d=4
 (2, 16), # tiny+: S=128, d=4
 (2, 20), # axiom-cfg: S=160, d=4 ← exact axiom config
 (2, 30), # bigger m: S=240, d=4
 (3, 8), # small: S=96, d=6
 (3, 12), # medium: S=144, d=6
 ]
 for (nh, m) in configs:
 r = run_full_l0(nh, m, 1.281)
 results.append(r)

 log(f"\n{'#'*72}")
 log(f"# OVERALL SUMMARY @ c_target = 1.281")
 log(f"{'#'*72}")
 log(f"{'n_half':>7} {'m':>3} {'d':>3} {'S':>6} {'total':>10} "
 f"{'A_s':>10} {'D_s':>10} {'F_s':>10} {'P_s':>10}")
 for r in results:
 log(f"{r['n_half']:>7} {r['m']:>3} {2*r['n_half']:>3} "
 f"{4*r['n_half']*r['m']:>6} {r['n_total']:>10,} "
 f"{r['A_surv']:>10,} {r['D_surv']:>10,} "
 f"{r['F_surv']:>10,} {r['P_surv']:>10,}")

 n_blocked = sum(1 for r in results if r["P_surv"] > 0)
 n_clean = sum(1 for r in results if r["P_surv"] == 0)
 log(f"")
 log(f" L0 fully terminates (P_surv = 0): {n_clean}/{len(results)} configs")
 log(f" L0 has survivors: {n_blocked}/{len(results)} configs")
 n_bug = sum(1 for r in results if r["P_bug_vs_A"] > 0 or r["P_bug_vs_F"] > 0)
 log(f" Soundness violations: {n_bug}/{len(results)} configs")

 out = {"c_target": 1.281, "configs": results}
 with open("_M1_p_1p281_FULL.json", "w") as fp:
 json.dump(out, fp, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else x)
 with open("_M1_p_1p281_FULL.log", "w", encoding="utf-8") as fp:
 fp.write("\n".join(LOG))
 log(f"\nWrote _M1_p_1p281_FULL.json and _M1_p_1p281_FULL.log")
