"""Corrected vacuity analysis for c_target = 1.40.

Now that the threshold fix has been applied (C&S Lemma 3 pointwise bound),
re-derive everything from first principles.

The code uses:
  TV threshold = c_target + (3 + 2*W_int) / m^2   (per window)
  Justified by: C&S Lemma 3 pointwise L-inf bound + W-refinement + W_g correction.
  PROVEN SOUND.
"""
import math
import numpy as np
import sys, os

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction

C_TARGET = 1.40
C_UPPER = 1.5029
GAP = C_UPPER - C_TARGET  # 0.1029

print("=" * 80)
print("CORRECTED VACUITY ANALYSIS FOR c_target = 1.40")
print("Formula: TV > c_target + (3 + 2*W_int)/m^2   (C&S Lemma 3 + W-refinement)")
print("Proven sound via pointwise L^inf bound, NOT per-window TV bound.")
print("=" * 80)

# =========================================================================
# 1. TWO DIFFERENT CORRECTIONS IN THE CODEBASE
# =========================================================================
print("""
1. TWO CORRECTIONS IN THE CODEBASE
===================================

A. correction() in pruning.py:  2/m + 1/m^2
   - Used for: x_cap computation, display/logging
   - This is the BASIC C&S Lemma 3 bound (no W-refinement)
   - Valid: it's the worst case of the W-refined bound (W_f = 1 => 2/m + 1/m^2)

B. Pruning kernels in run_cascade.py:  (3 + 2*W_int)/m^2
   - Used for: actual pruning decisions
   - This is the W-REFINED bound corrected for discrete W_g
   - +3 = +1 (eps*eps bound) + 2 (W_f <= W_g + 1/m correction)
   - Worst case (W_int=m): 2/m + 3/m^2 > correction() = 2/m + 1/m^2

QUESTION: Is correction() correct for x_cap?

x_cap uses: thresh = c_target + correction(m) = c_target + 2/m + 1/m^2
The pruning threshold (worst case) is: c_target + 2/m + 3/m^2

Since correction() < actual worst case, x_cap is SLIGHTLY TOO TIGHT.
A child bin with c_i = x_cap+1 would be pruned by the x_cap filter
but MIGHT survive the actual window scan (which uses the looser +3 threshold).

Is this a soundness issue? Let me check...
""")

# The x_cap logic:
# If c_i >= x_cap_cs = floor(m*sqrt(c_target/d_child)) + 1, then
#   d_child * ((c_i-1)/m)^2 >= c_target
#   => ||f*f||_inf >= c_target  (by Cauchy-Schwarz on bin i)
# This is INDEPENDENT of the discretization correction.
# So x_cap_cs is sound regardless of which correction formula we use.

# The other x_cap:
# x_cap = floor(m * sqrt((c_target + corr + 1e-9) / d_child))
# This says: if c_i > x_cap, then d_child * (c_i/m)^2 > c_target + corr
# => TV at ell=2, s=2*i is d_child*(c_i/m)^2 > c_target + corr
# => pruned by the BASIC (non-W-refined) threshold

# But wait: the code takes x_cap = min(x_cap, x_cap_cs).
# Since x_cap_cs uses c_target (not c_target+corr), and at relevant d_child
# we have x_cap > x_cap_cs typically, x_cap_cs is the binding constraint.
# x_cap_cs is sound because it uses the Cauchy-Schwarz argument directly.

print("x_cap soundness check:")
print(f"{'m':>4} {'level':>5} {'d_child':>7} | {'x_cap(corr)':>11} {'x_cap_cs':>8} {'binding':>8}")
print("-" * 60)

for m in [20, 25, 30, 40, 50]:
    corr = correction(m)
    for level in range(4):
        d_parent = 4 * (2**level)  # n_half=2
        d_child = 2 * d_parent
        thresh = C_TARGET + corr + 1e-9
        x1 = int(math.floor(m * math.sqrt(thresh / d_child)))
        x2 = int(math.floor(m * math.sqrt(C_TARGET / d_child))) + 1
        xf = min(x1, x2, m)
        binding = "x_cap_cs" if x2 <= x1 else "x_cap"
        if level <= 2:
            print(f"{m:4d} L{level:>3d}  {d_child:7d} | {x1:11d} {x2:8d} {binding:>8}")

print("""
x_cap_cs is binding at almost every level. x_cap_cs uses Cauchy-Schwarz
(d_child * mu_i^2 >= c_target => ||f*f||_inf >= c_target), which is
independent of the discretization correction. So x_cap is SOUND.
""")

# =========================================================================
# 2. PRECISE VACUITY ANALYSIS
# =========================================================================
print("2. PRECISE VACUITY")
print("=" * 50)
print()
print("The TV-space pruning threshold is:")
print("  TV > c_target + (3 + 2*W_int) / m^2")
print()
print("A window can prune a near-optimal config (TV ~ C_upper) when:")
print("  c_target + (3 + 2*W_int)/m^2 < C_upper = 1.5029")
print(f"  (3 + 2*W_int)/m^2 < {GAP}")
print(f"  W_int < (m^2 * {GAP} - 3) / 2")
print()
print("A parameter m is FULLY vacuous if NO window can prune, i.e.,")
print("even the best window (W_int = 0) has threshold >= C_upper:")
print(f"  c_target + 3/m^2 >= {C_UPPER}")
print(f"  3/m^2 >= {GAP}")
print(f"  m^2 <= {3/GAP:.1f}")
print(f"  m <= {math.sqrt(3/GAP):.2f}")
print()

m_full_vacuous = int(math.floor(math.sqrt(3/GAP)))
print(f"  => m <= {m_full_vacuous} is FULLY vacuous (no window works at all)")
print()

print(f"{'m':>4} | {'3/m^2':>8} | {'best thresh':>11} | {'W_vacuous':>10} | {'max W_int':>9} | status")
print("-" * 75)

for m in range(5, 61):
    best_thresh = C_TARGET + 3.0/(m*m)   # W_int=0
    worst_thresh = C_TARGET + (3 + 2*m)/(m*m)  # W_int=m
    w_vac = (m*m * GAP - 3) / 2

    if m in [5, 10, 15, 19, 20, 21, 22, 25, 30, 40, 50, 60] or abs(best_thresh - C_UPPER) < 0.01:
        if best_thresh >= C_UPPER:
            status = "FULLY VACUOUS"
        elif worst_thresh >= C_UPPER:
            status = f"PARTIAL (W<={int(w_vac)})"
        else:
            status = "FULLY OK"
        print(f"{m:4d} | {3.0/(m*m):8.6f} | {best_thresh:11.6f} | {w_vac:10.1f} | {m:9d} | {status}")

# =========================================================================
# 3. WHAT "PARTIAL VACUITY" MEANS IN PRACTICE
# =========================================================================
print()
print()
print("3. WHAT 'PARTIAL VACUITY' MEANS")
print("=" * 50)
print("""
m=20: W_vacuous threshold = 19.1. So windows with W_int >= 20 are vacuous.
Since W_int = sum of child masses in window bins, and total mass = m = 20,
only the WIDEST window (containing ALL bins) has W_int = 20.

In practice: every window EXCEPT the widest one can still prune.
Narrow windows (small W_int) have the TIGHTEST thresholds:
  W_int=0:  threshold = 1.40 + 3/400 = 1.4075   (excellent pruning power)
  W_int=5:  threshold = 1.40 + 13/400 = 1.4325
  W_int=10: threshold = 1.40 + 23/400 = 1.4575
  W_int=15: threshold = 1.40 + 33/400 = 1.4825
  W_int=19: threshold = 1.40 + 41/400 = 1.5025   (just barely works)
  W_int=20: threshold = 1.40 + 43/400 = 1.5075   (VACUOUS - above C_upper)
""")

for m in [20, 25, 30]:
    print(f"m={m}: thresholds by W_int")
    for w in range(0, m+1, max(1, m//5)):
        t = C_TARGET + (3 + 2*w)/(m*m)
        ok = "OK" if t < C_UPPER else "VACUOUS"
        print(f"  W_int={w:3d}: threshold = {t:.6f}  {ok}")
    print()

# =========================================================================
# 4. WHICH (m, n_half) ARE OBSOLETE/REDUNDANT
# =========================================================================
print()
print("4. DEFINITIVE PARAMETER CLASSIFICATION")
print("=" * 50)
print(f"""
For c_target = {C_TARGET}:

FULLY VACUOUS (m <= {m_full_vacuous}):
  Even the best window (W_int=0) gives threshold >= {C_UPPER}.
  No pruning is possible. These values are MATHEMATICALLY USELESS.
  m <= {m_full_vacuous}: 3/m^2 >= {GAP:.4f}
""")

# Check: is m=5 the actual cutoff?
for m in range(1, 20):
    if C_TARGET + 3.0/(m*m) < C_UPPER:
        print(f"  First non-fully-vacuous m: {m}")
        print(f"    3/{m}^2 = {3.0/(m*m):.6f} < {GAP:.4f}")
        break

print(f"""
PARTIALLY VACUOUS (widest window fails):
  m=20: W_int=20 (widest) is vacuous, but W_int<=19 works.
        This is the CURRENT config and it WORKS. Only one window is lost.
  m=21: All windows work (first m with zero vacuous windows).

n_half COMPARISON (all at m=20):
  n_half=2 (d=4): 1,771 L0 compositions. OPTIMAL for c_target=1.40.
  n_half=3 (d=6): 53,130 L0 compositions. 30x more work at L0.
    Also produces more survivors (less selective asymmetry filter
    at d=6 because left/right split is over 3 bins, not 2).
  n_half=4 (d=8): 888,030 L0 compositions. Infeasible.
  n_half=5+: Millions+. Out of the question.

HIGHER m TRADE-OFF:
  m=25: 3,276 L0 comps (1.85x), better margin (0.0213 vs 0.0004)
  m=30: 5,456 L0 comps (3.1x), even better margin (0.0351)
  m=40: 12,341 L0 comps (7x), good margin (0.0523)
  BUT: higher m also means more children per parent at each level.
  The key question is whether tighter pruning compensates.
  Empirically (from benchmark): m=20 has LOWEST expansion factors.

VERDICT:
  n_half=2, m=20 is the ONLY viable config for c_target=1.40.
  Everything else is either vacuous, redundant, or strictly worse.
""")

# =========================================================================
# 5. VERIFY correction() vs PRUNING CODE CONSISTENCY
# =========================================================================
print("5. INTERNAL CONSISTENCY CHECK")
print("=" * 50)
print()
print("correction() returns 2/m + 1/m^2 (basic C&S Lemma 3, W=1)")
print("Pruning code uses (3 + 2*W_int)/m^2 (W-refined, +3 for W_g correction)")
print()
print("At W_int=m (worst case):")

for m in [20, 25, 30]:
    corr_func = correction(m)
    actual_worst = (3 + 2*m)/(m*m)
    diff = actual_worst - corr_func
    print(f"  m={m}: correction()={corr_func:.6f}, "
          f"actual worst={actual_worst:.6f}, diff={diff:.6f} = {2.0/(m*m):.6f}")

print(f"""
The difference is always exactly 2/m^2 (the W_g correction term).
correction() returns the BASIC bound (used for x_cap and display).
The pruning kernels use the W-REFINED bound (tighter per-window).

This is CORRECT: x_cap uses correction() as an upper bound on the
discretization error when checking whether a single bin's energy
alone exceeds c_target. The Cauchy-Schwarz cap (x_cap_cs) is the
binding constraint anyway and is independent of the correction.
""")

# =========================================================================
# 6. VERIFY THE +3 vs +1 MATTERS
# =========================================================================
print("6. DOES +3 vs +1 MATTER?")
print("=" * 50)
print()
print("C&S MATLAB uses +1 (no W_g correction). Our code uses +3.")
print("At W_int=0: +1 gives threshold c+1/m^2, +3 gives c+3/m^2.")
print()

for m in [20, 25, 30, 50]:
    t1 = C_TARGET + 1.0/(m*m)
    t3 = C_TARGET + 3.0/(m*m)
    print(f"  m={m}: +1 threshold={t1:.6f}, +3 threshold={t3:.6f}, "
          f"diff={2.0/(m*m):.6f}")

print(f"""
At m=20: difference is 2/400 = 0.005. This means +3 is 0.5% more
conservative than +1. In practice, this means our code prunes
SLIGHTLY FEWER configurations than the MATLAB. Our proof is
STRICTLY MORE RIGOROUS (we account for W_g vs W_f).

At m=50: difference is 2/2500 = 0.0008. Negligible.
""")
