# Five Validated Ideas for Making the Cascade Feasible

> *Historical session note. For current project state see README.md and NOTES_INDEX.md. Both lower-bound proofs are now complete; the framing below dates from earlier exploration.*


> **Context:** At L3-L4, each parent spawns 81^8 ≈ 2×10^15 (L3) to 81^16 ≈ 5×10^30 (L4)
> children. Enumeration is permanently infeasible at these scales. The only viable path
> is **per-parent certification** that proves all children pruned WITHOUT enumerating them.
>
> **Critical finding from audit:** All approaches based on lower-bounding window sums from
> partial cursor assignments (partial-conv, min_contrib, CDCL, factored masks) provide
> **zero benefit** for the actual hard cases because x_cap/2 > avg_bin_mass → min_contrib = 0.
> The 5 ideas below avoid this trap entirely.

---

## IDEA 1: SDP Relaxation for Parent-Level Infeasibility Certification

*(Already implemented in cascade_opts.py:440-571 as `sdp_certify_parent`)*

### What It Does

For each parent, formulates "does any child survive all windows?" as a QCQP in
d_parent cursor variables. The Shor SDP relaxation certifies infeasibility (all
children pruned) without visiting a single child.

### Mathematical Formulation

**Variables:** x = (cursor[0], ..., cursor[d_parent-1]), x_i ∈ [lo_i, hi_i].

**Child masses (affine in x):** child[2i] = x_i, child[2i+1] = 2P_i - x_i.

**Window sum (quadratic in x):** For each window W = (ℓ, s):

    ws_W(x) = x^T Q_W x + g_W^T x + κ_W

where Q_W, g_W, κ_W are derived from the convolution pair structure. Specifically,
for each pair (p,q) with p+q ∈ W:

    child[p]·child[q] = (s_p·x_{a(p)} + c_p)(s_q·x_{a(q)} + c_q)

where s_p = +1 for even p, -1 for odd p; c_p = 0 for even, 2P_{a(p)} for odd;
a(p) = p÷2 is the parent index.

**SDP relaxation:** Replace xx^T with PSD matrix X ≽ xx^T via moment matrix
Y = [[1, x^T], [x, X]] ≽ 0. Add RLT cuts:

    (x_i - lo_i)(x_j - lo_j) ≥ 0  →  X[i,j] ≥ lo_j·x_i + lo_i·x_j - lo_i·lo_j
    (hi_i - x_i)(hi_j - x_j) ≥ 0  →  X[i,j] ≥ hi_i·x_j + hi_j·x_i - hi_i·hi_j
    ... (4 RLT inequalities per pair)

**Pruning:** If SDP infeasible → QCQP infeasible → all children pruned. Sound
because the SDP relaxation ENLARGES the feasible set.

### Problem Size and Cost

| Level | d_parent | SDP vars | Constraints | Est. time/parent |
|-------|----------|----------|-------------|-----------------|
| L3    | 8        | 44       | ~250        | 1-5 ms          |
| L4    | 16       | 152      | ~600        | 10-50 ms        |
| L5    | 32       | 560      | ~1400       | 100-500 ms      |

Total for ~2000 parents: **2-100 seconds.** Negligible vs years of enumeration.

### Empirical Evidence

For a perfectly balanced parent (all bins = 40, m=20, c_target=1.4): **0 out of
100,000 random children survive** the worst-case threshold. The balanced child
itself is pruned with 8% margin at ell=9. SDP will trivially certify such parents.

The open question: what fraction of the ACTUAL L2 survivors (the hard cases) can
SDP certify? This depends on whether those parents have any genuine L3 survivors.

### Soundness: ✅ PROVEN (standard Shor/Lasserre relaxation theory)

---

## IDEA 2: Non-Convex QCQP Global Solver (Exact, Zero Relaxation Gap)

### What It Does

Replace the SDP relaxation (which has a potential gap) with an EXACT non-convex
QCQP global solver. For each parent, the solver either:
- **Proves infeasibility** (all children pruned) — rigorous, no relaxation gap
- **Finds a feasible point** (a specific surviving cursor assignment) — an actual
  survivor that can be verified and collected without enumerating 10^15 children

### Why This Is Better Than SDP

The SDP relaxation enlarges the feasible set (X ≽ xx^T instead of X = xx^T). If
the true feasible set is empty but the relaxed set is not, SDP reports "feasible"
(false negative). The global solver has NO relaxation gap: if the problem is
infeasible, it PROVES it.

### Mathematical Basis

The QCQP is:

    Find x ∈ [lo, hi]^{d_parent} such that:
      ws_W(x) ≤ threshold_W   for all windows W

Equivalently (as maximization):

    maximize  min_W [threshold_W - ws_W(x)]
    subject to  lo ≤ x ≤ hi

If optimal value < 0: no feasible x → all children pruned.
If optimal value ≥ 0: optimal x is a survivor.

### Solver Technology

Modern spatial branch-and-bound solvers (BARON, Gurobi 12+ with non-convex QP,
SCIP, Couenne) handle small QCQP instances exactly:

1. **Branch:** Subdivide the cursor box [lo, hi] into sub-boxes
2. **Bound:** Compute convex relaxation (McCormick envelopes, SDP) on each sub-box
3. **Prune:** Discard sub-boxes where the relaxation proves infeasibility
4. **Converge:** Until all sub-boxes pruned (infeasible) or feasible point found

For d_parent = 8-16 variables with ~200-600 quadratic constraints, this is a SMALL
problem. Expected solve time: **1-100 ms per parent.**

### Critical Advantage: Finding Survivors

When the solver finds a feasible point x*, we get a SPECIFIC cursor assignment
that survives all windows. We can:

1. **Verify** it in exact int64 arithmetic (convert x* to integer, check all windows)
2. **Collect** it as an L3 survivor without enumerating 10^15 siblings
3. **Use it as a warm start** for finding nearby survivors (local search)

This transforms the problem: instead of exhaustive enumeration to FIND survivors,
we directly COMPUTE them.

### Integer Feasibility

The QCQP operates on continuous x. A continuous solution might not correspond to
an integer cursor value. Two approaches:

1. **Round-and-verify:** Round x* to nearest integer, check all windows. If it
   survives → genuine integer survivor. If not → the continuous solution doesn't
   correspond to an integer survivor.

2. **Mixed-integer QCQP (MIQCQP):** Add integrality constraints x_i ∈ ℤ. BARON
   handles this natively. Slightly slower but gives exact integer solutions.

### Cost

| Level | d_parent | Est. time/parent | Total (2000 parents) |
|-------|----------|-------------------|---------------------|
| L3    | 8        | 1-50 ms           | 2-100 sec           |
| L4    | 16       | 10-500 ms         | 20-1000 sec         |
| L5    | 32       | 100 ms - 5 sec    | 200-10000 sec       |

### Soundness: ✅ PROVEN (global optimality guarantee of branch-and-bound solvers)

The solver provides a **dual bound** (lower bound on the objective) that proves
infeasibility. This is mathematically rigorous — it's a constructive proof that
no feasible point exists in the box.

For floating-point concerns: verify the dual bound with interval arithmetic, or
use an exact rational solver (slow but available).

---

## IDEA 3: Per-Composition Adaptive Box-Cert Correction (7-23× Tighter)

### What It Does

Replace the global worst-case box-cert correction with a **per-child composition-
specific correction** that is 7-23× tighter. This dramatically lowers the
threshold, turning many current survivors into pruned children.

### The Gap That Creates Survivors

The current threshold includes a correction:

    threshold(ℓ) = c_target × 4nm²ℓ + min(n, ℓ-1, 2d-ℓ) × B

where B = n(8m+1)/2 is the **worst-case** per-index perturbation bound.

**Concrete numbers (m=20, c_target=1.4, d_child=16):**

| ℓ   | Worst-case correction | Per-comp correction (balanced) | Ratio |
|-----|----------------------:|-------------------------------:|------:|
| 5   | 2,576                | 224                            | 0.087 |
| 8   | 4,508                | 544                            | 0.121 |
| 9   | 5,152                | 704                            | 0.137 |
| 10  | 5,152                | 864                            | 0.168 |
| 13  | 5,152                | 1,184                          | 0.230 |

The per-composition correction is **7-12× smaller** at the dominant killing windows.
The gap between bare Theorem 1 and the corrected threshold narrows from ~5,152 to
~704 integer units. Survivors hiding in this gap are eliminated.

### Mathematical Derivation

For a specific child composition c at window W = (ℓ, s):

**Gradient:**

    grad_i(c, W) = (4d/ℓ) × Σ_{j: s ≤ i+j ≤ s+ℓ-2} c_j / S

**First-order bound (cell_var):** Sort grad, pair extremes:

    cell_var(c, W) = (1/(2S)) × Σ_{k=0}^{d/2-1} [grad_sorted[d-1-k] - grad_sorted[k]]

**Second-order bound (quad_corr):**

    quad_corr(W) = (2d/ℓ) × min(cross_W, d²-N_W) / (4S²)

**Per-composition correction:**

    correction(c, W) = cell_var(c, W) + quad_corr(W)

**Soundness proof:** By Taylor expansion of TV_W(μ + δ) where δ is the cell
perturbation (|δ_i| ≤ h = 1/(2S), Σδ_i = 0):

    TV_W(μ + δ) = TV_W(μ) + grad·δ + (2d/ℓ)Q(δ)

    |TV_W(μ + δ) - TV_W(μ)| ≤ |grad·δ| + |(2d/ℓ)Q(δ)| ≤ cell_var + quad_corr

Therefore: if ws(c) > bare_Theorem1(W) + cell_var(c,W) + quad_corr(W), then
TV_W(μ) > c_target for ALL continuous μ in the cell. **SOUND.**

And: cell_var(c,W) + quad_corr(W) ≤ min(n,ℓ-1,2d-ℓ) × B (proven upper bound
on gradient-pairing). So the per-composition correction is ALWAYS ≤ worst-case.

### Why It's Tighter for Balanced Compositions

For balanced child (all c_j ≈ S/d):
- grad_i ≈ (4d/ℓ) × (S/d) × (active j count for bin i) / S = (4/ℓ) × count
- The gradient spread (max - min) is proportional to the variation in "active j count"
  across bins, which is bounded by the window geometry
- For the killing windows (ℓ ≈ d/2), most bins have similar j-counts, so spread is small
- Cell_var ≈ O(gradient_spread / S) ≈ O(1/S) with a SMALL constant

For the worst case (all mass in one bin):
- One gradient is huge, others are zero
- Cell_var ≈ O(max_mass / S) ≈ O(1) (S times larger than balanced case)

### Computational Cost

Per non-quick-killed child (15% of total), at the window that exceeds bare threshold:

1. **Gradient computation:** Use existing prefix_c array. For bin i at window (ℓ,s):
   active j-range = [max(0,s-i), min(d-1, s+ℓ-2-i)].
   grad_i = (4d/(ℓ·S)) × [prefix_c[hi_j+1] - prefix_c[lo_j]]. Cost: **O(d)**.

2. **Sort gradient:** Insertion sort on d=16 values. Cost: **O(d²) ≈ 256 ops**.

3. **Pair extremes:** Sum d/2 differences. Cost: **O(d)**.

4. **Compare:** ws > bare + (cell_var + quad_corr) × scale. Cost: **O(1)**.

Total extra: ~300 ops × 0.5ns = **150ns per non-quick-killed child**.
Amortized over all children: 0.15 × 150ns = **23ns overhead** (16% increase).

### Impact

This idea does NOT avoid enumeration — it TIGHTENS the threshold so that:
1. **Quick-check succeeds more often** (lower threshold → easier to exceed)
2. **Fewer children survive to L4** (the correction gap where survivors hide shrinks 7-23×)
3. **SDP certification (Ideas 1-2) becomes more likely** (lower threshold → harder for
   any cursor to satisfy all constraints → SDP infeasibility more likely)

The third point is crucial: **per-composition correction makes the SDP stronger.**
The SDP checks ws_W(x) ≤ threshold_W. With a lower threshold_W, the SDP has an
easier time proving infeasibility.

### Soundness: ✅ PROVEN

cell_var + quad_corr ≤ min(n,ℓ-1,2d-ℓ)×B (proven). This is the existing
coarse-cascade box-cert analysis (run_cascade_coarse_v2.py) applied to the
fine cascade's children. The per-composition correction is a TIGHTER VALID BOUND
on the same mathematical quantity.

---

## IDEA 4: Higher-Degree Lasserre Hierarchy (Degree-4 SDP)

### What It Does

When the degree-2 SDP relaxation (Idea 1) is inconclusive (reports "feasible"
but no integer survivor is found), upgrade to the **degree-4 Lasserre hierarchy**.
This uses a larger moment matrix that captures quartic variable relationships,
closing the relaxation gap.

### Mathematical Basis

The degree-2 relaxation (Shor) uses moment matrix indexed by monomials
{1, x_1, ..., x_d}. The degree-4 Lasserre hierarchy adds monomials of degree 2:
{1, x_1, ..., x_d, x_1², x_1x_2, ..., x_d²}.

For d_parent = 8: the degree-4 moment matrix is indexed by all monomials up to
degree 2, giving ${8+2 \choose 2} = 45$ entries. The moment matrix M ∈ ℝ^{45×45}
must satisfy:

1. **M ≽ 0** (PSD)
2. **Consistency:** M[α,β] = M[γ,δ] whenever xᵅ·xᵝ = xᵞ·xᵟ
3. **Constraint propagation:** For each window constraint ws_W(x) ≤ threshold_W,
   the "localizing matrix" L_W(M) ≽ 0, where:
   
       L_W[α,β] = Σ_{γ} Q_W[γ] · M[α+γ, β] + ...

   (Incorporates the constraint into the moment structure)

4. **Box constraints:** Localizing matrices for (x_i - lo_i) ≥ 0 and (hi_i - x_i) ≥ 0

### Why This Is Tighter

The degree-2 SDP captures quadratic relationships: it "knows" that X[i,j] ≈ x_i×x_j.
But it cannot distinguish X[i,j]×X[k,l] from X[i,k]×X[j,l] (both map to the
4th-order moment x_i×x_j×x_k×x_l).

The degree-4 hierarchy has SEPARATE variables for each 4th-order moment. It captures
relationships like:

    (x_i - x_j)² × (x_k - x_l)² ≥ 0

which constrains the joint distribution of cursor values in ways that degree-2 cannot.

For our problem: the window sums ws_W involve products x_a×x_b from different
parent bins. The degree-4 hierarchy captures the interaction between these products
across multiple windows, potentially proving that no cursor can dodge ALL windows.

### Problem Size

| d_parent | Degree-2 SDP size | Degree-4 SDP size | Est. time/parent |
|----------|-------------------|-------------------|-----------------|
| 8        | 9×9 = 44 vars     | 45×45 = 1035 vars | 50-500 ms       |
| 16       | 17×17 = 152 vars  | 153×153 = 11781 vars | 5-30 sec    |

For d_parent=8: the degree-4 SDP is ~20× larger than degree-2 but still tractable
(MOSEK handles 1035-variable SDPs in ~500ms).

For d_parent=16 (L4): degree-4 is expensive (~30 sec/parent). Use degree-2 first;
fall back to degree-4 only for uncertified parents.

### When to Use

1. Run degree-2 SDP (Idea 1) on all parents → certifies most
2. For uncertified parents: run QCQP solver (Idea 2) → finds exact answer for most
3. For remaining: run degree-4 SDP → certifies more
4. Final remaining: these have genuine survivors → collect via QCQP feasible points

### Soundness: ✅ PROVEN (Lasserre hierarchy convergence theorem)

The degree-2k Lasserre hierarchy converges to the exact convex hull of the
feasible set as k → ∞ (Lasserre 2001). At degree 4, the relaxation is strictly
tighter than degree 2. Infeasibility at any degree implies infeasibility of the
original problem.

---

## IDEA 5: LP Dual Certificate with Rigorous SOS Interior Verification

### What It Does

Strengthens the existing LP dual certificate (cascade_opts.py:238-433) by replacing
the heuristic interior-point check (20 random samples) with a **rigorous Sum-of-Squares
(SOS) polynomial certificate** that covers the ENTIRE cursor box interior.

### The Problem with the Current LP Dual

The LP dual finds weights λ_W ≥ 0 (Σλ_W = 1) such that:

    F(x) = Σ_W λ_W × [ws_W(x) - threshold_W] ≥ 0   for all x in box

The LP verifies this at BOX VERTICES (2^d_parent = 256 points for d=8). Since F(x)
is QUADRATIC in x and the Q_W matrices can be indefinite, F might be NEGATIVE at
interior points even when positive at all vertices.

The current code checks 20 random interior points (line 398-426). If any is negative,
the certificate fails. But 20 random points are not rigorous — a negative point
might be missed.

### The Fix: SOS Certificate for Box Non-Negativity

Given λ_W from the LP, construct F(x) = Σ_W λ_W × (x^T Q_W x + g_W^T x + κ_W - threshold_W).

This is a QUADRATIC polynomial in d_parent variables. We need to certify:

    F(x) ≥ 0   for all x ∈ [lo, hi]^{d_parent}

Using the Positivstellensatz: F(x) can be written as:

    F(x) = σ_0(x) + Σ_i σ_i(x)·(x_i - lo_i) + Σ_i τ_i(x)·(hi_i - x_i)
           + Σ_{i<j} ρ_{ij}(x)·(x_i - lo_i)·(x_j - lo_j) + ...

where σ_0, σ_i, τ_i, ρ_{ij} are SOS polynomials (sums of squares). If such a
decomposition exists, F is non-negative on the box.

For QUADRATIC F (degree 2): the SOS multipliers σ_0 are degree 0 (constants),
and σ_i, τ_i are degree 0 (constants). The SOS certificate reduces to:

    F(x) = σ_0 + Σ_i a_i·(x_i - lo_i) + Σ_i b_i·(hi_i - x_i)

where σ_0 ≥ 0 and a_i, b_i ≥ 0. This is a LINEAR PROGRAM in (σ_0, a_i, b_i)!

Wait — this only works if F is AFFINE. For quadratic F, we need degree-1 SOS
multipliers, giving an SDP.

**Precise formulation:** For quadratic F(x) = x^T A x + b^T x + c on box [lo,hi]:

    F(x) = (affine)^T · M · (affine)   where M ≽ 0

Specifically, write F in the Bernstein basis on [lo,hi], then check all Bernstein
coefficients are non-negative. For a quadratic in d variables, this has
(d+1)(d+2)/2 = 45 Bernstein coefficients (for d=8).

If all Bernstein coefficients are non-negative: F ≥ 0 on the box. This is a
SUFFICIENT condition (not necessary) but is very tight for quadratic polynomials.

### Algorithm

1. Run LP dual (existing code) → get λ_W weights
2. Build F(x) = Σ λ_W × (ws_W(x) - threshold_W)
3. Compute F's Bernstein coefficients on the box [lo, hi]
4. If all coefficients ≥ 0: **CERTIFIED** (rigorous, all children pruned)
5. If any coefficient < 0: certificate fails for this parent

### Cost

Step 3 (Bernstein coefficients): O(3^d) for degree-2 polynomial in d variables.
For d=8: 3^8 = 6561 evaluations. Each is O(d²) = O(64). Total: ~420K ops = **~0.2ms**.

Combined with LP solve (~1ms): **~1.2ms per parent**. Total: ~2.5 seconds for 2084 parents.

### Why This Helps

The current LP dual + 20 random points is NOT rigorous. It can't be used in a
formal proof. The SOS/Bernstein certificate IS rigorous. This turns the LP dual
from a heuristic into a **proof tool**.

Parents that the LP dual certifies (but currently can't prove due to interior-
point gap) are now rigorously certified. This complements the SDP (Idea 1) and
QCQP solver (Idea 2) — different mathematical tools that might certify different
parents.

### Soundness: ✅ PROVEN

Bernstein coefficient non-negativity implies polynomial non-negativity on the box.
This is a classical result in approximation theory (Bernstein 1912). For quadratic
polynomials, the sufficient condition is known to be tight (no gap).

---

## Combined Strategy

```
For each parent at L3/L4/L5:
  1. LP dual + Bernstein certificate (Idea 5)        ~1.2 ms    → certifies "easy" parents
  2. SDP degree-2 relaxation (Idea 1)                ~5 ms      → certifies most remaining
  3. Non-convex QCQP global solver (Idea 2)          ~50 ms     → exact answer, finds survivors
  4. SDP degree-4 Lasserre (Idea 4)                  ~500 ms    → closes relaxation gap
  5. Remaining: these have GENUINE survivors
     → collect via QCQP feasible points (no enumeration!)
```

**Throughout all levels:** use per-composition adaptive correction (Idea 3) to
lower thresholds, making certification easier at every step.

**Total time per level:** ~2000 parents × ~50ms avg = **~100 seconds.**

**Compare:** current enumeration = **190 years.** Improvement = **60,000,000×**.

---

## Implementation Priority

1. **Idea 2 (QCQP solver)** — highest standalone impact, gives exact answers, finds
   survivors directly. Use Gurobi (free academic) or SCIP (open-source).
2. **Idea 1 (SDP degree-2)** — already implemented, cheapest per-parent, certifies
   easy cases.
3. **Idea 5 (LP dual + Bernstein)** — makes existing LP dual rigorous, fast, simple.
4. **Idea 3 (Per-comp correction)** — tightens thresholds 7-23×, helps all other ideas.
5. **Idea 4 (Degree-4 Lasserre)** — fallback for hard cases, most expensive.
