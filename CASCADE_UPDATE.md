# CPU Cascade: 5 Validated High-Impact Ideas

> *Historical session note. For current project state see README.md and NOTES_INDEX.md. Both lower-bound proofs are now complete; the framing below dates from earlier exploration.*


> **Root cause diagnosis**: The cascade is infeasible at L3+ (d_parent >= 16) because
> the Cartesian product of cursor ranges is ~160^16 = 10^35 children per parent.
> Even at 7M children/sec/core, this takes 10^21 CPU-years per parent.
>
> The ideas below attack this from multiple angles: (A) TIGHTER PRUNING to reduce
> survivors at each level, (B) SMARTER ENUMERATION to avoid visiting dead children,
> (C) SMALLER SEARCH SPACE by eliminating provably-dead cursor values before enumeration.
> Combined multiplicative effect: estimated 10^8 - 10^15 reduction in work.
>
> **VERIFICATION STATUS (2026-04-14)**: All 5 ideas independently verified:
> - Idea 1: Algebraic verification of threshold formulas + Theorem 1 applicability
> - Idea 2: Sum-of-minimums bound proven; per-term cross formulas verified against brute-force
> - Idea 3: partial_conv <= full_conv proven (non-negative terms); W_int_max bound verified
> - Idea 4: Full interval arithmetic verified against brute-force for d_parent=3
>   (min_conv_ia <= min_conv_bf for all entries, all windows). Tightness: 20-100%.
> - Idea 5: conv(rev(c)) == rev(conv(c)) verified numerically; W_int matching proven
>   algebraically and verified for all windows at d=8.

---

## IDEA 1: Theorem 1 Pruning in Fine Cascade (Drop Correction Term)

### What
Replace the C&S Lemma 3 threshold (which includes a positive correction term) with
the Theorem 1 threshold (exact, no correction) for ALL pruning in the fine cascade.
This is a one-line change that makes pruning strictly tighter.

### Mathematical Validity (PROVEN)

**Theorem 1** (from `run_cascade_coarse.py` line 1-13): For any nonneg f on [-1/4,1/4]
with integral 1, if f has bin masses mu_i = integral_{B_i} f, then for any window (ell, s):

    max_{|t|<=1/2} (f*f)(t) >= TV_W(mu) = (2d/ell) * sum_{k=s..s+ell-2} sum_{i+j=k} mu_i*mu_j

This is **EXACT** -- no correction term, no step-function approximation.

**Key insight**: The fine cascade enumerates compositions c_0,...,c_{d-1} summing to
S = 4nm. These define bin masses mu_i = c_i/S. Theorem 1 applies to ANY f with
those bin masses, including the true function f we're trying to bound. Therefore:

    If TV_W(mu) >= c_target, then max(f*f) >= c_target.

**Current fine threshold** (C&S Lemma 3 + W-refinement):

    dyn_it = floor((c_target*m^2 + 1 + W_int/(2n) + eps) * 4n*ell)

**Theorem 1 threshold** (no correction):

    dyn_it = floor(c_target * m^2 * 4n * ell - eps)

The Theorem 1 threshold is LOWER by `(1 + W_int/(2n)) * 4n * ell`. Since pruning
fires when `ws > dyn_it`, a LOWER threshold means MORE children are pruned.

**Verification**: At m=20, n_half=16 (d=32), ell=20, W_int=500:
- Correction delta = (1 + 500/32) * 64 * 20 = 16.625 * 1280 = 21280
- Theorem 1 threshold = 1.4 * 400 * 64 * 20 = 716800
- Relative reduction: 21280/716800 = 3.0%

**Soundness of Theorem 1 in the cascade**: The cascade proves that for every mass
distribution mu, some window TV exceeds c_target. Theorem 1 guarantees that this
implies max(f*f) >= c_target for ALL f with those masses. The fine grid enumeration
visits specific grid points; between grid points, **box certification** (first + second
order Taylor bounds, as in `run_cascade_coarse_v2.py`) handles the continuous case.

**Caveat**: The current fine cascade is self-certifying (correction handles continuous
case). With Theorem 1, we need box certification at the final level. This is O(d^2)
per composition -- negligible compared to enumeration. Box certification is already
implemented in `run_cascade_coarse_v2.py` (lines 48-101) with sound second-order
quadratic correction for the indefinite Hessian of TV_W.

**VERIFIED (2026-04-14)**: Threshold algebra confirmed:
Fine threshold = Theorem 1 threshold + (1 + W_int/(2n)) * 4n * ell.
At m=20, n=16, ell=20, W_int=500: delta = 21280, Th1 threshold = 716800, +3.0%.

### Expected Benefit
- ~3% lower threshold -> ~30% fewer survivors per level (compositions near boundary switch from survivor to pruned)
- Compound across 5 levels: (0.7)^5 = 0.17 -> **83% total work reduction**
- Implementation: change one line (threshold formula)
- Box certification adds negligible cost

---

## IDEA 2: Per-Bin Cursor Range Tightening (Constraint Propagation)

### What
Before enumerating children of a parent, compute for each parent bin i and each
cursor value v in [lo_i, hi_i] whether v is GUARANTEED to lead to pruning regardless
of all other cursor values. Remove dead values. Iterate (AC-3 style) until stable.

### Mathematical Validity (PROVEN)

For parent bin i with cursor value v: child[2i] = v, child[2i+1] = 2P_i - v.

**Self-contribution** to conv entries (EXACT, independent of other bins):
- conv[4i] gets v^2
- conv[4i+1] gets 2*v*(2P_i - v)
- conv[4i+2] gets (2P_i - v)^2

**Cross-contribution** with parent bin j (LOWER BOUNDED by interval arithmetic):
For other bins c_{2j} in [lo_j, hi_j] and c_{2j+1} = 2P_j - c_{2j}:
- conv[2i+2j] gets >= 2 * min(v, v) * lo_j = 2 * v * lo_j  (if v > 0, lo_j > 0)
- Similarly for all 4 cross-term positions

For each window W, compute:

    ws_min(v, i) = sum_{k in W} [self_contrib(v, i, k) + sum_{j != i} min_cross_contrib(v, i, j, k)]

If `ws_min(v, i) > threshold(W)` for SOME window W, then cursor value v at bin i
leads to pruning for ALL choices of other bins. Remove v from [lo_i, hi_i].

**Iteration**: After removing dead values from bin i, the min_cross_contrib for OTHER
bins that cross-reference bin i becomes tighter (because the range [lo_i, hi_i] shrunk).
Iterate until convergence. This is the AC-3 algorithm from constraint satisfaction.

**Soundness**: We only remove values that are provably dead (guaranteed pruned). The
remaining range is a SUPERSET of the values needed by any surviving child. No false
negatives.

**Why extreme values are easily killed**: When v = 0, child[2i] = 0 and child[2i+1] = 2P_i.
The self-contribution to conv[4i+2] is (2P_i)^2, which is very large. With typical
P_i ~ 80, this gives 25600 at a single conv entry. Even without cross-terms, this
can exceed the ell=2 threshold for that position.

Similarly, v = 2P_i gives the same effect at conv[4i].

### Expected Benefit

**Conservative estimate** (5x range reduction per bin):
- Current product at d_parent=16: 160^16 = 10^35
- After 5x reduction: 32^16 = 10^24 (10^11 improvement)

**Optimistic estimate** (20x range reduction per bin):
- After 20x reduction: 8^16 = 10^14 (10^21 improvement)

**Cost**: O(d^2 * max_range) per parent per iteration, with ~3-5 iterations to converge.
At d=32, max_range=300: ~3M operations per parent. NEGLIGIBLE vs 10^35 enumeration.

### Implementation
```python
@njit(cache=True)
def _tighten_ranges(parent_int, n_half_child, m, c_target, lo_arr, hi_arr):
    """Tighten cursor ranges via interval arithmetic constraint propagation."""
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    # ... threshold setup same as pruning kernel ...
    
    changed = True
    while changed:
        changed = False
        for i in range(d_parent):
            new_lo = lo_arr[i]
            new_hi = hi_arr[i]
            # Scan from lo upward: find first non-dead value
            for v in range(lo_arr[i], hi_arr[i] + 1):
                if _is_dead_value(v, i, parent_int, lo_arr, hi_arr, thresholds):
                    new_lo = v + 1
                else:
                    break
            # Scan from hi downward: find first non-dead value
            for v in range(hi_arr[i], lo_arr[i] - 1, -1):
                if _is_dead_value(v, i, parent_int, lo_arr, hi_arr, thresholds):
                    new_hi = v - 1
                else:
                    break
            if new_lo > lo_arr[i] or new_hi < hi_arr[i]:
                lo_arr[i] = new_lo
                hi_arr[i] = new_hi
                changed = True
    return lo_arr, hi_arr
```

---

## IDEA 3: Full Branch-and-Bound Tree at All Levels

### What
Replace the Gray code enumeration (which visits every element of the Cartesian product)
with a depth-first branch-and-bound tree search at ALL cascade levels. Currently, only
L0 uses B&B (`_l0_bnb_inner`, lines 428-705). L1+ uses Gray code with limited subtree
pruning at depth J_MIN=7.

The full B&B checks partial autoconvolution at EVERY internal node, not just at depth 7.

### Mathematical Validity (PROVEN)

**Key property**: All masses c_i >= 0, so all conv cross-terms are non-negative.
Therefore:

    partial_conv(bins 0..k) <= full_conv(all bins)

Consequence: if the partial window sum (from fixed bins 0..k) already exceeds the
threshold, then the FULL window sum (with all bins) must also exceed it. The entire
subtree below this node can be pruned.

**For the flat threshold** (W_int-independent): directly sound, since the threshold
doesn't grow with more bins.

**For the W-refined threshold**: use W_int_max = W_int_fixed + 2 * sum(unfixed parent bins).
This gives a valid upper bound on W_int for any child in the subtree, and thus a valid
threshold lower bound.

This is EXACTLY the same argument used in `_l0_bnb_inner` (lines 663-698). The only
difference is applying it at L1+ instead of just L0.

### Why This Is Strictly Better Than Gray Code + J_MIN=7

The Gray code checks for subtree pruning ONLY when digit J_MIN (= 7) advances. This
means:
- Digits 0-6: no subtree pruning, enumerate all combinations
- Digit 7: check once, potentially skip the inner sweep

The full B&B checks at EVERY depth:
- After fixing bin 0: check partial conv -> potentially prune 1/R of the tree
- After fixing bin 0,1: check -> potentially prune 1/R^2
- ...
- After fixing bins 0..k: check -> potentially prune 1/R^(d-k)

At each node, the partial check costs O(k^2) for the partial autoconvolution. The total
cost across the tree is O(N_visited * d^2), where N_visited << N_total because of pruning.

### Expected Benefit

**At L2 (d_parent=8)**: subtree pruning at depth 2-3 typically eliminates 60-90% of
the search space. Combined with cursor range tightening (Idea 2), the tree is much
shallower and pruning is more effective.

**At L3 (d_parent=16)**: the B&B tree can potentially reduce 10^35 to 10^10-10^15
by pruning large subtrees at shallow depths.

**At L4 (d_parent=32)**: similar exponential reduction.

The key advantage over Gray code: B&B prunes ENTIRE SUBTREES (exponential savings)
while Gray code visits every element (no subtree skipping except at J_MIN).

### Implementation

Extend `_l0_bnb_inner` to work at arbitrary levels:
```python
@njit(cache=True)
def _bnb_inner(parent_int, d_parent, lo_arr, hi_arr, 
               n_half_child, m, c_target, out_buf, use_flat_threshold):
    """Full B&B for any cascade level. Same logic as _l0_bnb_inner but
    for child generation (each parent bin i splits into child bins 2i, 2i+1)."""
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1
    conv = np.zeros(conv_len, dtype=np.int32)
    child = np.zeros(d_child, dtype=np.int32)
    
    # DFS over cursor positions 0..d_parent-1
    pos = 0
    child[0] = lo_arr[0]
    child[1] = 2 * parent_int[0] - lo_arr[0]
    
    while True:
        # ... standard DFS with carry/backtrack ...
        # At each internal node: compute partial conv, check windows
        # If partial ws > threshold: prune subtree (skip to next sibling)
        # At leaf: full conv check, store survivors
```

---

## IDEA 4: Whole-Parent Minimum-TV Pre-Pruning (Generalized Interval Arithmetic)

### What
Before entering ANY enumeration loop for a parent, compute a guaranteed LOWER BOUND
on the window sum for every window (ell, s_lo) across ALL possible children. If any
window's lower bound exceeds the threshold, skip the parent entirely with ZERO enumeration.

This generalizes `block_mass_prune_mask` (which only checks contiguous self-convolution
blocks) to consider ALL conv entries using interval arithmetic.

### Mathematical Validity (PROVEN)

For each conv entry conv[k] = sum_{i+j=k} c_i * c_j, we compute the minimum over
all valid child assignments.

**Within-parent-bin terms** (self-contribution of parent bin i):
The 3 conv entries at positions [4i, 4i+1, 4i+2] sum to (2P_i)^2 (CONSTANT, proven
algebraically: v^2 + 2v(2P-v) + (2P-v)^2 = (2P)^2).

Individual entry minimums:
- conv[4i] = c_{2i}^2, minimum = lo_i^2
- conv[4i+2] = c_{2i+1}^2 = (2P_i - c_{2i})^2, minimum = (2P_i - hi_i)^2
- conv[4i+1] = 2*c_{2i}*c_{2i+1}, minimum at endpoints of concave function:
  min(2*lo_i*(2P_i-lo_i), 2*hi_i*(2P_i-hi_i))

**Cross-parent terms** (parent bins i, j with i < j):
- conv[2i+2j]: gets 2*c_{2i}*c_{2j}, min = 2*lo_i*lo_j
- conv[2i+2j+1]: gets 2*c_{2i}*c_{2j+1} + 2*c_{2i+1}*c_{2j},
  min >= 2*lo_i*(2P_j - hi_j) + 2*(2P_i - hi_i)*lo_j
- conv[2i+2j+2]: gets 2*c_{2i+1}*c_{2j+1}, min = 2*(2P_i-hi_i)*(2P_j-hi_j)

**Soundness of sum-of-minimums bound**:
The minimum of a sum >= sum of minimums (when terms share variables). So:

    min_{child} ws(W) >= sum_{k in W} min_{child} conv[k] >= sum_{k in W} min_conv[k]

If sum_{k in W} min_conv[k] > threshold(W), then ALL children are pruned.

**This is strictly more powerful than `block_mass_prune_mask` because**:
1. It considers cross-terms between NON-CONTIGUOUS parent bins
2. It considers ALL possible windows, not just self-convolution windows at ell=4k
3. It uses per-entry bounds from cursor ranges, not just total block mass

**VERIFIED (2026-04-14)**: Tested against brute-force enumeration at d_parent=3
(7875 children). All conv entry lower bounds valid. All window sum lower bounds valid.
Tightness of per-entry bounds: 20-100% of true min (conservative but sound).
Looseness comes from sum-of-per-term-minimums: two terms sharing the same conv entry
but from different parent pairs are minimized at different cursor configurations.
Can be tightened by checking all 4 box corners per parent pair.

**IMPORTANT CORRECTNESS NOTE**: Each conv[r] receives contributions from ALL child
pairs (p,q) with p+q=r, which may span multiple parent pairs. E.g., at d_child=8,
conv[6] receives terms from parent pairs (0,3), (0,2), (1,2), and (1,1). The code
must iterate over ALL contributing parent pairs, not just one. The implementation
below correctly accumulates per-term minimums from all pairs via += operators.

### Expected Benefit
- O(d^2) per parent (compute min_conv) + O(d^2) for window scan = O(d^2) total.
- Could prune 10-50% of parents at L2+, eliminating need for ANY enumeration.
- Combined with range tightening (Idea 2): when lo > 0 (after tightening), ALL
  cross-term minimums become positive, dramatically increasing the lower bounds.

### Implementation
```python
@njit(cache=True)
def _whole_parent_prune(parent_int, lo_arr, hi_arr, n_half_child, m, c_target):
    """Check if ALL children of this parent are pruned via interval arithmetic."""
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1
    min_conv = np.zeros(conv_len, dtype=np.int64)
    
    # Self-contributions
    for i in range(d_parent):
        lo = np.int64(lo_arr[i])
        hi = np.int64(hi_arr[i])
        P2 = np.int64(2) * np.int64(parent_int[i])
        comp_lo = P2 - hi  # min of c_{2i+1}
        comp_hi = P2 - lo  # max of c_{2i+1}
        min_conv[4*i] += lo * lo
        min_conv[4*i + 2] += comp_lo * comp_lo
        mut_a = np.int64(2) * lo * (P2 - lo)
        mut_b = np.int64(2) * hi * (P2 - hi)
        min_conv[4*i + 1] += min(mut_a, mut_b)
    
    # Cross-contributions
    for i in range(d_parent):
        for j in range(i + 1, d_parent):
            lo_i, lo_j = np.int64(lo_arr[i]), np.int64(lo_arr[j])
            P2_i = np.int64(2) * np.int64(parent_int[i])
            P2_j = np.int64(2) * np.int64(parent_int[j])
            hi_comp_i = P2_i - np.int64(hi_arr[i])  # min c_{2i+1}
            hi_comp_j = P2_j - np.int64(hi_arr[j])  # min c_{2j+1}
            min_conv[2*i + 2*j] += np.int64(2) * lo_i * lo_j
            min_conv[2*i + 2*j + 2] += np.int64(2) * hi_comp_i * hi_comp_j
            # Middle term: min over box boundary
            min_conv[2*i + 2*j + 1] += np.int64(2) * (
                lo_i * hi_comp_j + hi_comp_i * lo_j)
    
    # Window scan
    for ell in range(2, 2 * d_child + 1):
        # ... sliding window over min_conv ...
        if ws_min > threshold:
            return True  # ALL children pruned
    return False
```

---

## IDEA 5: Canonical (Palindromic) Enumeration During Generation

### What
Currently, ALL children are generated, tested, canonicalized (min(c, rev(c))), and then
deduplicated post-hoc. Since every child c has a symmetric twin rev(c) with IDENTICAL
autoconvolution, this exactly doubles the work. Instead, enumerate ONLY canonical
children (c <= rev(c) lexicographically) during the B&B/Gray code traversal.

### Mathematical Validity (PROVEN)

**Claim**: conv(c) = conv(rev(c)) for any composition c.

**Proof**: conv[k] = sum_{i+j=k} c_i * c_j. Under reversal c_i -> c_{d-1-i}:
conv'[k] = sum_{i+j=k} c_{d-1-i} * c_{d-1-j}. Let i' = d-1-i, j' = d-1-j.
Then i'+j' = 2d-2-k, so conv'[k] = conv[2d-2-k]. But the window scan checks
ALL positions, and the conv array is scanned symmetrically (windows from both ends).
So the set of window sums is identical for c and rev(c).

Therefore: c survives iff rev(c) survives. Testing both is redundant.

**Canonical constraint**: c <= rev(c) lex means:
- c[0] < c[d-1], OR
- c[0] = c[d-1] AND c[1] < c[d-2], OR
- c[0] = c[d-1] AND c[1] = c[d-2] AND c[2] < c[d-3], OR ...

For the cascade, child bins come in parent-linked pairs:
(child[0], child[1]) from parent[0]; (child[d-2], child[d-1]) from parent[d/2-1].

The canonical constraint on the outermost pair:
child[0] <= child[d-1], i.e., cursor[0] <= 2*P_{d_parent-1} - cursor[d_parent-1].

In the B&B tree, when assigning paired bins (cursor[i] and cursor[d_parent-1-i]),
we can enforce the lex constraint directly:

**For the outermost pair (i=0, j=d_parent-1)**:
- If cursor[0] < 2*P_j - cursor[j]: canonical (proceed to inner pairs)
- If cursor[0] > 2*P_j - cursor[j]: non-canonical (skip entire subtree)
- If cursor[0] = 2*P_j - cursor[j]: check next pair

**For inner pairs**: same logic, but only activated when outer pairs are tied.

### Expected Benefit
- **Exact 2x speedup** for enumeration (half the children tested).
- ZERO mathematical approximation -- exact equivalence class enumeration.
- Combines multiplicatively with all other ideas.
- Also eliminates the need for post-hoc deduplication of symmetric pairs.

### Implementation
In the B&B tree, add a lex constraint tracker:
```python
# Track lex state: LEX_LT (canonical confirmed), LEX_EQ (tied so far)
lex_state = LEX_EQ  # start tied

# When assigning cursor[i] (from left) and cursor[j] (from right):
if lex_state == LEX_EQ:
    child_left = cursor[i]
    child_right_rev = 2 * parent_int[j] - cursor[j]  # rev(child)[i]
    if child_left < child_right_rev:
        lex_state = LEX_LT  # confirmed canonical
    elif child_left > child_right_rev:
        continue  # skip: non-canonical
    # else: still tied, check inner pair

# In the B&B backtrack: restore lex_state
```

---

---

## TIGHTENING ANALYSIS (2026-04-14)

### Cross-term middle entries (Ideas 2 & 4): 4-corner minimum

**Problem**: The sum-of-per-term-minimums for conv[2i+2j+1] is 30-78% below the true
minimum. The two terms `2*c_{2i}*c_{2j+1}` and `2*c_{2i+1}*c_{2j}` share cursor
variables from parents i and j. Minimizing each independently wastes the correlation.

**Fix**: Compute the JOINT minimum at the 4 box corners:
```
f(xi, xj) = 2*xi*(2Pj - xj) + 2*(2Pi - xi)*xj = 4*xi*Pj + 4*xj*Pi - 4*xi*xj
```
This is bilinear in (xi, xj), so min is at one of 4 corners. Evaluate all 4, take min.

**Verified**: 4-corner minimum matches brute force EXACTLY for all middle cross-terms.

### Conv entries with self-terms: interior critical points

**Problem**: Entries like `conv[2] = 2*c_0*c_2 + c_1^2` are CONVEX in x_0 (due to
the c_1^2 = (2P_0 - x_0)^2 self-term). The minimum is in the INTERIOR, not at corners.

**Fix**: For entries containing a self-term c_p^2 where p is from parent pi,
compute the critical point: `x_pi = clamp(2*P_pi - x_other, lo_pi, hi_pi)` for
each corner assignment of the other involved parents.

**Verified**: With corner + interior evaluation, ALL per-entry minimums are EXACT.

### Window sum minimization: 2^d_parent corner check

**Problem**: Even with exact per-entry mins, the window sum bound is only ~50%
tight (different entries achieve their mins at different cursor configurations).

**Fix**: For the whole-parent pre-pruning (Idea 4), evaluate the window sum
directly at ALL 2^d_parent corners of the cursor box. The window sum is a
quadratic in the cursor variables; empirically, its minimum is at a box corner
for 64 out of 66 tested windows (even when the Hessian is indefinite).

**Cost**: 2^d_parent * O(d^2) per parent per window.
- d_parent=16: 65536 * 256 * 20 windows ≈ 3.4 * 10^8 operations. CHEAP.
- d_parent=32: 4.3 * 10^9 * 1024 * 40 ≈ 1.8 * 10^14. TOO EXPENSIVE for corner check.

For d_parent >= 20, fall back to per-entry interval arithmetic (sum-of-4-corner-mins).
For d_parent <= 16, use the full 2^d_parent corner evaluation for NEAR-EXACT bounds.

**For the ONE case (ell=2, boundary windows) where the corner min is not exact**:
add the interior critical point check (clamp the gradient-zero point to the box).
This costs O(d_parent) per parent per window, negligible.

---

## Summary Table

| Idea | Type | Mechanism | Expected Speedup | Math Valid |
|------|------|-----------|-----------------|------------|
| 1. Theorem 1 pruning | Tighter threshold | Drop correction term | 5-6x (compound over levels) | EXACT (Theorem 1) |
| 2. Range tightening | Smaller search space | Kill dead cursor values | 10^11 - 10^21 at d=16 | SOUND (interval arith) |
| 3. Full B&B all levels | Smarter enumeration | Prune subtrees at any depth | 10^5 - 10^15 at d=16 | EXACT (monotone conv) |
| 4. Whole-parent pruning | Skip entire parents | Interval arith all entries | 1.1-2x (parent elimination) | SOUND (sum-of-min) |
| 5. Canonical enumeration | Halve search space | Skip non-canonical children | 2x | EXACT (conv symmetry) |

**Recommended execution order**:
1. IDEA 1 (Theorem 1) -- trivial, one-line change, immediate 30% survivor reduction
2. IDEA 4 (whole-parent) -- easy, O(d^2), eliminates parents before enumeration
3. IDEA 5 (canonical enum) -- moderate, 2x speedup, no approximation
4. IDEA 2 (range tightening) -- moderate, largest single impact
5. IDEA 3 (full B&B) -- most complex, but most powerful at high d

**Combined projected impact at L3 (d_parent=16)**:
- Current: ~10^35 children per parent, infeasible
- Ideas 1+4: ~30% fewer survivors + ~30% parents eliminated -> ~0.5x
- Idea 2 (5x range reduction/bin): (1/5)^16 = 7 * 10^(-12) -> 5 * 10^23 per parent
- Idea 3 (B&B prunes 99% of remaining tree): ~5 * 10^21 per parent
- Idea 5 (canonical): -> ~2.5 * 10^21 per parent
- At 7M children/sec: 3.6 * 10^14 seconds. Still too much for one parent.
- BUT: range tightening + B&B synergize exponentially. If range reduces 20x/bin:
  (1/20)^16 * B&B factor -> potentially 10^14 total, feasible.
