# CPU Cascade Prover: 5 Validated High-Impact Ideas

> *Historical session note. For current project state see README.md and NOTES_INDEX.md. Both lower-bound proofs are now complete; the framing below dates from earlier exploration.*


> **Root cause diagnosis**: The CPU fine cascade is infeasible at L3+ (d_parent >= 16)
> because the Cartesian product of cursor ranges is ~160^16 = 10^35 children per
> parent. Even at 7M children/sec/core on 64 cores, a single parent takes 10^19
> CPU-years. The bottleneck is purely COMBINATORIAL: too many children to enumerate.
>
> These 5 ideas attack the combinatorial explosion from complementary angles:
> (A) tighter pruning thresholds, (B) smaller search space via dead-value elimination,
> (C) smarter tree search that prunes subtrees, (D) parent-level elimination,
> (E) symmetry exploitation. Combined multiplicative effect: 10^8 - 10^15 reduction.

---

## IDEA 1: Theorem 1 Pruning (Drop Correction Term)

### What
Use the exact Theorem 1 bound (no correction) instead of C&S Lemma 3 (with correction
2/m + 1/m^2) for ALL pruning decisions in the fine cascade.

### Why it works
**Theorem 1**: For any nonneg f with bin masses mu_i, max(f*f) >= TV_W(mu). This is
EXACT -- no step-function approximation, no correction term.

The fine cascade computes integer conv sums from compositions c_i summing to S = 4nm.
These define bin masses mu_i = c_i/S. Theorem 1 applies directly.

**Current threshold**: `floor((c_target*m^2 + 1 + W_int/(2n) + eps) * 4n*ell)`
**Theorem 1 threshold**: `floor(c_target * m^2 * 4n * ell - eps)`

The Theorem 1 threshold is LOWER by `(1 + W_int/(2n)) * 4n * ell`, meaning it prunes
MORE children. At m=20, d=32, this is ~3% tighter, translating to ~30% fewer survivors
per level (compositions near the boundary switch from survivor to pruned).

**Caveat**: Requires box certification at the final level (already implemented in
`run_cascade_coarse_v2.py`). Implementation: one-line threshold change.

### Expected benefit
- 30% fewer survivors per level; compound over 5 levels: (0.7)^5 = 0.17 -> **6x total**
- Implementation: trivial (one-line change)

---

## IDEA 2: Per-Bin Cursor Range Tightening (Constraint Propagation)

### What
Before enumeration, for each parent bin i and cursor value v, check if v is guaranteed
to be pruned regardless of other cursor values (via interval arithmetic lower bounds on
window sums). Remove dead values. Iterate (AC-3) until stable.

### Why it works
For cursor value v at bin i, the self-contribution to conv is FIXED (v^2, 2v(2P-v),
(2P-v)^2). Cross-contributions with other bins have computable MINIMUMS (using lo/hi
ranges of other bins). If self + min_cross > threshold for some window, v is dead.

**Key observation**: Extreme cursor values (v near 0 or 2P_i) produce peaked
distributions easily killed by ell=2 windows. Typical parents have ~60% of cursor
values dead at the extremes.

**Soundness**: Only provably-dead values are removed. The remaining range is a superset
of all ranges needed by any surviving child. AC-3 iteration is monotone and terminates.

**Tightening (verified 2026-04-14)**: The per-value dead check uses a window sum
lower bound. Use the **4-corner joint minimum** for cross-term middle entries
(bilinear -> corner is optimal, verified exact against brute force). For entries
with self-terms (c_p^2), add the interior critical point `x = clamp(2P - x_other,
lo, hi)`. This makes per-entry bounds EXACT, and window sum bounds 50-100% tight
(limited only by the sum-of-entry-mins inherent looseness).

### Expected benefit
- 5-20x range reduction per bin -> (1/5)^d to (1/20)^d total reduction
- At d_parent=16 with 5x: 10^11 reduction. With 20x: 10^21 reduction.
- Cost: O(d^2 * max_range) per parent, negligible vs enumeration

---

## IDEA 3: Full Branch-and-Bound at All Levels

### What
Replace the Gray code enumeration (visits every child) with depth-first B&B tree search
at ALL levels, not just L0. At each internal node (after fixing bins 0..k), compute
partial autoconvolution and check if partial window sum already exceeds threshold.
If yes, prune the entire subtree.

### Why it works
All c_i >= 0, so all conv cross-terms are non-negative. Therefore:
partial_conv(fixed bins) <= full_conv(all bins).

If partial_ws > threshold, full_ws > threshold, so the entire subtree is pruned.
This is the SAME argument used in `_l0_bnb_inner` (lines 428-705), extended to L1+.

**Strictly better than Gray code + J_MIN=7**: Gray code only checks for subtree
pruning when digit 7 advances. Full B&B checks at EVERY depth, pruning exponentially
larger subtrees earlier in the tree.

**Tightening (verified 2026-04-14)**:
1. **Partial conv + min_contrib**: At each B&B node, compute partial conv from fixed
   bins PLUS the guaranteed minimum contributions from unfixed bins (4-corner joint
   minimums for cross-terms). This is already done in the subtree pruning code
   (lines 1994-2100) but should be applied at EVERY B&B node.
2. **Tighter W_int_max**: Instead of `2*sum(unfixed parents)`, use per-child-bin
   maximums: `max(c_{2j}) = hi_j`, `max(c_{2j+1}) = 2P_j - lo_j`. This is tighter
   when cursor ranges are restricted.
3. **Theorem 1 threshold**: Use the correction-free threshold (Idea 1) for the
   partial conv check, making it strictly tighter.

### Expected benefit
- At L3 (d=16): B&B can reduce 10^35 to 10^10-10^15 by pruning subtrees at depth 2-4
- Combines multiplicatively with range tightening (shallower tree after range reduction)
- Zero overhead for surviving branches (same work as Gray code per leaf)

---

## IDEA 4: Whole-Parent Interval Arithmetic Pre-Pruning

### What
Before entering ANY enumeration for a parent, compute guaranteed LOWER BOUNDS on all
window sums using interval arithmetic over ALL conv entries. If any window's lower
bound exceeds the threshold, skip the parent entirely (zero enumeration).

### Why it works
For each conv[k], compute min_conv[k] = minimum over all valid children:
- Self-terms within parent bin: min(v^2) = lo^2
- Cross-terms between parents: min(2*c_i*c_j) = 2*lo_i*lo_j (independent)
- For each window: min_ws >= sum_{k in W} min_conv[k]

**Strictly more powerful than `block_mass_prune_mask`** which only checks contiguous
blocks at ell=4k. This checks ALL conv entries and ALL windows.

### Tightening (verified 2026-04-14)
Three levels of tightness:

1. **Sum-of-per-term-mins** (original): 20-100% of true min. LOOSE for middle
   cross-terms (conv[2i+2j+1]) where sum-of-mins is 30-78% below true min.

2. **4-corner joint minimum** per parent pair: EXACT for all cross-term entries.
   For the middle term, evaluate `f(xi,xj) = 4*xi*Pj + 4*xj*Pi - 4*xi*xj`
   at 4 corners. Bilinear -> corner is optimal. Also add interior critical points
   for entries with self-terms (convex in one variable, min can be interior).

3. **2^d_parent corner evaluation** of the full window sum: NEAR-EXACT (matches
   brute force for 64/66 tested windows). Cost: 2^16 * d^2 * n_windows ≈ 3*10^8
   operations per parent. CHEAP for d_parent <= 16.

### Expected benefit
- O(d^2) per parent with level-2 bounds; O(2^d * d^2) with level-3 bounds
- With tightened bounds: prunes 20-60% of parents at L2+
- Combined with range tightening: when lo > 0, all cross-term mins become positive

---

## IDEA 5: Canonical (Palindromic) Enumeration

### What
Enumerate only canonical children (c <= rev(c) lexicographically) during B&B traversal,
instead of generating all children and deduplicating post-hoc.

### Why it works
conv(c) is identical to conv(rev(c)) because autoconvolution is symmetric:
conv'[k] = conv[2d-2-k], and the window scan checks all positions.

Therefore c and rev(c) are always both pruned or both survivors. Testing both is
redundant. By enforcing c[0] <= c[d-1] (and recursively for ties) in the B&B tree,
we enumerate exactly one representative per equivalence class.

### Expected benefit
- **Exact 2x speedup** (half the children tested, zero approximation)
- Eliminates post-hoc deduplication
- Combines multiplicatively with all other ideas

---

## Combined Impact Projection

| Level | Current Search Space | After Ideas 1-5 | Feasible? |
|-------|---------------------|------------------|-----------|
| L2 (d=8) | ~10^9/parent | ~10^4/parent | YES (already feasible) |
| L3 (d=16) | ~10^35/parent | ~10^12-10^16/parent | BORDERLINE (hours-days) |
| L4 (d=32) | ~10^75/parent | ~10^30-10^40/parent | NO (need GPU) |

Key: Ideas 2+3 provide the largest combined reduction because they synergize
exponentially (tighter ranges make B&B pruning fire earlier and deeper).
