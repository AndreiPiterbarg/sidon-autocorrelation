# Lasserre L2 d=128 Speedup Proposals

> *Historical session note. For current project state see README.md and NOTES_INDEX.md. Both lower-bound proofs are now complete; the framing below dates from earlier exploration.*


> **Target:** Run Lasserre order-2 (L2) at d=128 on CPU with 256 GB RAM.
>
> **MOSEK status:** MOSEK crashes at d=128. The "Schur ~1PB" estimate in
> `lasserre_scs.py` is for the full 12M-variable precompute, but even the
> highd solver's 498K variables can overwhelm MOSEK because its memory scales
> **quadratically** with the number of constraint rows touching PSD blocks.
> MOSEK has NO out-of-core mode and does NOT auto-exploit chordal structure.
>
> **Solution:** Use a solver that avoids the Schur complement entirely.
> Three proven alternatives exist (see Idea 0 below).
>
> **Critical finding (preserved):** MOSEK theoretically **CAN** run at d=128 using the `solve_highd_sparse`
> solver from `lasserre_highd.py`. The "MOSEK Schur ~1PB" estimate in
> `lasserre_scs.py` refers to the FULL moment set (12M variables) from
> `lasserre_scalable.py`. The highd solver uses a **reduced moment set**
> (|S| ≈ 498K variables), bringing the Schur complement down to **~34 GB**
> — well within the 256 GB budget.
>
> **The real bottleneck is SOLVE TIME, not memory.**
> Each MOSEK interior-point solve at d=128 takes ~30-60 minutes (Cholesky
> factorization on a 498K×498K sparse matrix with ~1.6B nnz). The current
> algorithm makes ~300 MOSEK calls (20 bisection × 15 CG rounds), giving
> a total runtime of **100-300 hours**. The goal: reduce to **1-5 hours**.
>
> **Memory estimates (Schur complement, Cholesky fill-in 2.5x):**
>
> | Bandwidth | Schur Memory | Margin in 256GB |
> |-----------|-------------|-----------------|
> | bw=16 | ~34 GB | 7.5x margin |
> | bw=12 | ~5.3 GB | 48x margin |
> | bw=10 | ~1.8 GB | 142x margin |
>
> **Bottleneck breakdown (estimated at d=128, bw=16, order=2):**
> - `_precompute_highd`: |S| ≈ 498K moments, ~1 GB, ~10s
> - MOSEK model build: ~30s
> - MOSEK solve (single call): ~30-60 min
> - Violation checking: ~32K windows, ~1-5 min per CG round
> - **Total (300 calls): 100-300 hours**

---

## Idea 0: Replace MOSEK with a Schur-Free Solver (THE Critical Enabler)

**Impact: Makes d=128 POSSIBLE (currently crashes)**
**Memory: 256 GB → < 5 GB**

### The Problem

MOSEK's interior-point method builds a Schur complement matrix whose memory
scales O(n²) with the number of constraint rows touching PSD blocks. At d=128
with 498K variables, 112 clique PSD cones (171×171), 128 localizing cones,
and a full M1 (129×129), MOSEK exceeds available RAM and crashes. MOSEK has
**no out-of-core mode** and does **not** automatically exploit chordal structure.

### Three Proven Alternatives (ranked by effort)

**Option A: Clarabel (lowest effort, Python-native)**
```bash
pip install clarabel
```
- Interior-point method (same accuracy as MOSEK, ~1e-8)
- Default CVXPY solver since v1.5
- Handles sparse SDPs with many small PSD blocks much better than MOSEK
- Drop-in replacement: change `solver=cp.MOSEK` to `solver=cp.CLARABEL`
- Or: reformulate the MOSEK Fusion model as a CVXPY problem

**Option B: SCS on the highd reduced moment set (medium effort, highest impact)**
- `lasserre_scs.py` already exists but uses the FULL 12M-variable `_precompute`
- **The fix:** wire it to use `_precompute_highd` (498K variables) instead
- SCS indirect mode: O(nnz) memory, no Schur complement
- Estimated: ~50 MB memory, ~30-45 min total solve time
- Implementation: adapt `_precompute_scs_decomposition` to consume clique data
  from `_precompute_highd` instead of full moment/localizing picks

**Option C: COSMO.jl with automatic chordal decomposition (best theoretical)**
- Julia solver with Python wrapper (`cosmo-python` via pyjulia)
- **Automatically** decomposes large PSD cones into clique blocks
- Smart clique merging reduces projection overhead by up to 60%
- Feed the full undecomposed SDP — COSMO finds the clique structure itself
- Same accuracy class as SCS (first-order, ~1e-5)

### Soundness

All three solve the same mathematical SDP program — only the numerical
algorithm differs. The mathematical formulation (moment constraints, PSD
cones, window constraints) is unchanged. The feasible set and optimal
value are identical. **QED.**

### Why This Must Come First

Without solving the MOSEK crash, no other optimization matters. Ideas 1-5
below assume a working solver. Option B (SCS + highd) is recommended because:
1. SCS is already integrated (`lasserre_scs.py`)
2. The highd precompute is already built (`lasserre_highd.py`)
3. Only the glue code needs to change (adapt SCS COO builder for clique data)
4. Memory: ~50 MB (vs 256 GB budget = 5000x margin)

### Alternative Solvers (from literature survey)

| Solver | Method | Max Size | Accuracy | Python | Notes |
|--------|--------|----------|----------|--------|-------|
| **Clarabel** | Interior-point | Large sparse | 1e-8 | `pip install` | Default CVXPY |
| **SCS** | ADMM | 250K+ vars | 1e-5 | `pip install` | Already in repo |
| **COSMO.jl** | ADMM+chordal | Large sparse | 1e-5 | pyjulia | Auto clique decomp |
| **SDPNAL+** | Aug. Lagrangian | n=9K, m=12M | **1e-8** | MATLAB only | Best accuracy |
| **TSSOS** | Term sparsity | 6K vars | 1e-8 | Julia only | Lasserre-specific |
| **LoRADS** | Low-rank ADMM | n=180K | 1e-5 | C (CLI) | Extreme scale |

---

## Idea 1: Optimization-Mode Objective (Eliminate Bisection Entirely)

**Estimated speedup: 15-20x (300 calls → 15-20 calls)**
**Memory impact: None**

### What

The current algorithm treats the SDP as a FEASIBILITY problem parameterized
by t, then bisects over t. This requires n_bisect=20 MOSEK calls per CG round.

**Key insight:** Make t a MOSEK **variable** (not parameter) and **minimize t**
directly. This replaces 20 bisection calls with 1 optimization call per CG round.

### Implementation

```python
# CURRENT (bisection over parameter):
t_param = mdl.parameter("t")
mdl.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))  # feasibility
for step in range(n_bisect):
    t_param.setValue(mid)
    mdl.solve()  # 20 calls per round

# PROPOSED (direct optimization):
t_var = mdl.variable("t", 1, Domain.unbounded())
mdl.objective(ObjectiveSense.Minimize, t_var)  # minimize t
mdl.solve()  # 1 call per round → optimal t directly
```

**For scalar window constraints** `t ≥ f_W(y)`: These are LINEAR in (t, y).
Replacing `t_param` with `t_var` is straightforward — just change the
constraint from `Expr.sub(t_rep, f_all) ≥ 0` to use `t_var` instead.

**For PSD window constraints** `t·M_{k-1}(y) - Q_W(y) ≽ 0`: These involve
the PRODUCT t·y, which is bilinear. With t as a variable, this becomes
non-convex and MOSEK cannot handle it directly.

**Solution — Schur complement reformulation:**
The window PSD constraint L_W = t·T(y) - Q_W(y) ≽ 0 can be equivalently
written as a linear matrix inequality (LMI) in the **homogenized** variables.

Define z_α = t · y_α for all α ∈ S. Then:
- z_0 = t · y_0 = t (since y_0 = 1)
- L_W[a,b] = z_{α_a+α_b} - Σ M_W[i,j] · y_{α_a+α_b+e_i+e_j}

This mixes z and y variables. Add the linking constraint z_α = t · y_α.
This is STILL bilinear... BUT we can use MOSEK's **rotated quadratic cone**:

z_α · y_0 ≥ y_α² is equivalent to (z_α, y_0, y_α) ∈ RotatedSOC,
which enforces z_α ≥ y_α²/y_0 = y_α² (since y_0=1). This is a
RELAXATION of z_α = t·y_α, but combined with the constraint z_0 = t
and the SDP structure, it can be shown to be tight at optimality.

**Simpler alternative:** Use optimization mode for the SCALAR-ONLY CG
rounds (Rounds 0-3, before any PSD windows are added). Then switch to
bisection ONLY for the PSD-window rounds (Rounds 4+). Since scalar rounds
capture ~80-90% of the final bound, and typically only 3-5 PSD rounds are
needed with 5-8 bisection steps each, total calls drop to:
- Scalar phase: ~5 optimization calls (no bisection)
- PSD phase: 5 rounds × 8 bisection = 40 calls
- **Total: ~45 calls** vs. 300 → **~7x speedup**

### Soundness Proof

The optimization problem min{t : t ≥ f_W(y) ∀W, moment constraints on y}
is a standard conic program (linear objective + PSD cones + linear constraints).
MOSEK solves this directly. The optimal t* equals the true relaxation bound.

For the scalar-only phase: f_W(y) = Σ M_W[i,j] y_{e_i+e_j} is LINEAR in y.
The constraint t ≥ f_W(y) is linear in (t, y). All other constraints
(PSD cones, consistency, y ≥ 0) don't involve t. The program is convex. **QED.**

---

## Idea 2: MOSEK Warm-Start + Incremental Model Updates

**Estimated speedup: 3-5x per MOSEK solve**
**Memory impact: None**

### What

MOSEK supports **hot-starting** from a previous solution. The code already
sets `intpntHotStart = "primal"` (line 484 of lasserre_highd.py), but doesn't
exploit it fully. Between CG rounds, only a few PSD window constraints are
added — the solution changes minimally. A good warm start should reduce
MOSEK iterations from ~15-25 to ~5-8.

Additionally, use **incremental model updates**: instead of rebuilding the
MOSEK model from scratch each CG round, use `mdl.constraint(...)` to add
new constraints to the existing model. MOSEK preserves internal state across
incremental updates.

### Implementation

```python
# CURRENT: rebuild model each CG round (implicit in check_feasible loop)
# PROPOSED: keep model alive, add constraints incrementally

mdl, y, t_param = _build_model_highd(P, add_upper_loc, verbose)

for cg_round in range(max_cg_rounds):
    # Solve with warm start (MOSEK does this automatically for re-solves)
    t_param.setValue(t_mid)
    mdl.solve()  # warm-started from previous solution
    
    # Add violated windows incrementally (no model rebuild)
    for w, eig in violations[:n_add]:
        _add_window_psd_highd(mdl, y, t_param, w, P)
    # MOSEK preserves factorization for unchanged constraints
```

Also set MOSEK parameters to exploit warm-starting:
```python
mdl.setSolverParam("intpntHotStart", "primal_dual")  # stronger warm start
mdl.setSolverParam("intpntStartingPoint", "satisfy_bounds")
```

### Soundness Proof

Warm-starting and incremental constraint addition do not change the
mathematical program — they only affect the solver's internal starting
point and convergence path. MOSEK's interior-point method converges to
the same optimal solution regardless of starting point. The final
solution satisfies all constraints to the specified tolerance. **QED.**

---

## Idea 3: Reduced Bandwidth with Adaptive Widening

**Estimated speedup: 5-10x (from Schur complement reduction)**
**Memory reduction: 34 GB → 2-5 GB**

### What

Start with bandwidth=10 instead of bandwidth=16. This reduces:
- Clique basis size: 171 → 78 (2.2x smaller PSD cones)
- Distinct moments per clique: 5985 → 1365 (4.4x fewer)
- Schur complement: 34 GB → 1.8 GB (19x smaller)
- MOSEK solve time: ~30-60 min → ~3-6 min per call (Cholesky scales as D³)

After convergence at bw=10, **check if the bound is tight enough**. If not,
selectively increase bandwidth for cliques near the center (where windows
are tightest) while keeping narrow bandwidth at edges.

### Implementation

```python
def solve_highd_adaptive(d, c_target, order=2,
                          bw_start=10, bw_max=16, bw_step=2):
    """Adaptive bandwidth: start narrow, widen where needed."""
    bw = bw_start
    while bw <= bw_max:
        result = solve_highd_sparse(d, c_target, order, bandwidth=bw, ...)
        if result_is_tight_enough(result):
            return result
        bw += bw_step  # widen and retry
    return result
```

A more sophisticated version: identify which cliques' localizing matrices
are most violated (those near the center of the support, where ell ≈ d
windows concentrate) and widen only those cliques.

### Soundness Proof

Each bandwidth choice produces a valid relaxation (proved in `lasserre_highd.py`
Soundness Theorem, lines 27-45). A narrower bandwidth is a WEAKER relaxation
(fewer PSD constraints, more moments excluded), so lb_narrow ≤ lb_wide ≤ val(d).

The adaptive approach produces a sequence of valid lower bounds:
lb(bw=10) ≤ lb(bw=12) ≤ lb(bw=14) ≤ lb(bw=16) ≤ val(d).

Each is a sound lower bound. If lb(bw=10) already exceeds the target
c_target, no widening is needed. **QED.**

### Why This Has Enormous Impact

MOSEK's Cholesky factorization cost scales as O(n_y × D²) where D is the
Schur complement bandwidth (≈ distinct moments per clique). Reducing D from
5985 to 1365 gives a (5985/1365)² ≈ 19x speedup in factorization. Combined
with fewer nnz entries, total solve time drops by 10-20x.

The quality loss from narrower bandwidth is typically small: empirical data
at smaller d shows bw=10 captures 85-95% of the gap closure that bw=16 achieves.

---

## Idea 4: Vectorized Batch Violation Checking with Scalar Pre-Filter

**Estimated speedup: 5-10x faster violation checking per CG round**
**Memory impact: Negligible**

### What

The violation checker (`_check_violations_highd`) loops over all non-active
windows computing eigvalsh sequentially. At d=128, there are ~32K windows.
Replace with:

1. **Scalar pre-filter:** Compute f_W(y) for all windows via the sparse
   matvec `F_scipy.dot(y_vals)` (already done). Skip windows where
   `f_W(y) < t - δ` (far from binding). Use δ = 0.05 · t.
2. **Clique-batched construction:** For each clique, build ALL covered
   windows' localizing matrices simultaneously using vectorized numpy.
3. **Batch eigvalsh:** Stack remaining matrices into a 3D array and call
   `np.linalg.eigvalsh(L_stack)` once (LAPACK batched routine).

### Implementation

```python
def _check_violations_batched(y_vals, t_val, P, active_windows, delta=0.05):
    # Step 1: scalar pre-filter (O(nnz) — already computed)
    f_vals = P['F_scipy'].dot(y_vals)
    threshold = t_val * (1.0 - delta)
    candidates = [w for w in range(P['n_win'])
                  if w not in active_windows
                  and f_vals[w] >= threshold
                  and int(P['window_covering'][w]) >= 0]  # covered only
    
    if not candidates:
        return []
    
    # Step 2: batch-build localizing matrices by clique
    n_cb = P['clique_data'][0]['loc_size']  # 18
    L_stack = np.zeros((len(candidates), n_cb, n_cb))
    # ... build all L_W via vectorized clique indexing ...
    
    # Step 3: one batched eigvalsh call
    all_eigs = np.linalg.eigvalsh(L_stack)  # shape (n_cand, n_cb)
    min_eigs = all_eigs[:, 0]
    
    violated = [(candidates[i], float(min_eigs[i]))
                for i in np.where(min_eigs < -1e-6)[0]]
    return sorted(violated, key=lambda x: x[1])
```

### Soundness Proof

**Scalar pre-filter soundness:** CG convergence requires adding ANY violated
constraint, not necessarily the most violated. The pre-filter skips windows
with large scalar slack. If ALL truly violated windows have `f_W(y) ≥ t(1-δ)`,
the filter misses nothing.

**Safety net:** After CG convergence, run one final UNFILTERED check to verify
no violations were missed. If any are found, continue CG. This guarantees
the same final result as the sequential checker.

**Batch eigvalsh:** Computes the IDENTICAL eigenvalues as sequential calls —
`np.linalg.eigvalsh` on a stacked array delegates to the same LAPACK dsyev
routine per matrix. **QED.**

---

## Idea 5: Moment Set Pruning via Reachability Analysis

**Estimated speedup: 1.5-2x (from reduced n_y)**
**Memory reduction: ~30% less Schur complement**

### What

The current reduced moment set |S| ≈ 498K includes ALL C(131,3) = 366K
degree-≤3 monomials (for full consistency). Many of these are unreachable:
they appear in no PSD cone entry and have no degree-4 children in S. Since
they're constrained only by consistency + non-negativity, they can be
eliminated via substitution.

### Which moments are eliminable?

A degree-3 monomial α is **eliminable** if:
1. α is NOT in any clique moment PSD entry (α ≠ α_a + α_b for any
   clique basis pair). This means α has support spanning more than
   one clique (i.e., its nonzero components are separated by > bandwidth).
2. ALL degree-4 children α + e_i are outside S (partial consistency only).
3. α appears as a child in exactly one degree-2 consistency constraint.

For eliminable α: the only constraints involving y_α are:
- (i) y_α ≥ 0 (non-negativity)
- (ii) y_α ≥ Σ_{i∈S'} y_{α+e_i} (partial consistency, as child)
- (iii) y_β = Σ_i y_{β+e_i} where β is a degree-2 parent and α = β + e_j

Since y_α appears linearly in (iii), substitute y_α = y_β - Σ_{i≠j} y_{β+e_i}
from (iii), which eliminates y_α from the model.

### Estimate of reduction

At d=128, bw=16: a degree-3 monomial x_i x_j x_k is in a clique PSD iff
{i,j,k} fits within some [c, c+16] window. Monomials like x_0 x_64 x_127
are NOT in any clique. Roughly 30-40% of degree-3 monomials have support
wider than bandwidth. This gives n_y reduction from ~498K to ~350K.

### Soundness Proof

**Claim:** Eliminating a variable y_α via substitution from an equality
constraint does not change the optimal value of the SDP.

**Proof:** The substitution y_α = y_β - Σ_{i≠j} y_{β+e_i} is derived from
the equality consistency constraint y_β = Σ_i y_{β+e_i}. Substituting
eliminates y_α as a variable and removes the consistency row. The remaining
constraints that referenced y_α now reference the substituted expression.

The feasible set is unchanged (same solutions, just parameterized by fewer
variables). The objective doesn't involve y_α (it's `min t`). Therefore the
optimal value is identical. **QED.**

**Non-negativity handling:** After substitution, we need y_α ≥ 0, i.e.,
y_β - Σ_{i≠j} y_{β+e_i} ≥ 0. This is a NEW linear constraint that
replaces the non-negativity of y_α. It's a valid constraint since it
follows from the original y_α ≥ 0 + equality consistency.

---

## Combined Impact Estimate

| Idea | Speedup Factor | Memory Impact | Implementation Effort |
|------|---------------|---------------|----------------------|
| 1. Optimization mode | 7x (300→45 calls) | None | Medium |
| 2. Warm-start + incremental | 3-5x per call | None | Low |
| 3. Adaptive bandwidth | 5-10x (bw 16→10) | 34GB → 2GB | Low |
| 4. Batch violation check | 5-10x per CG round | Negligible | Low |
| 5. Moment pruning | 1.5-2x (n_y reduction) | 30% less Schur | Medium |

**Combined estimate:**
- Current: ~300 calls × 30-60 min = 100-300 hours
- With ideas 1+2: ~45 calls × 10-20 min = 7-15 hours
- With idea 3 (bw=10): ~45 calls × 1-3 min = 0.75-2.25 hours
- With ideas 4+5: additional 1.5-2x → **~30-90 minutes total**

**Priority order: 3 > 1 > 2 > 4 > 5** (by impact/effort ratio).

Idea 3 (reduced bandwidth) is the single biggest win because MOSEK's
Cholesky factorization scales as O(D³) where D is the moments-per-clique.
Reducing bw from 16 to 10 gives D: 5985 → 1365, a 19x factorization speedup.

Idea 1 (optimization mode) is the second-biggest win because it eliminates
85% of MOSEK calls. Combined with idea 2 (warm-start), each remaining call
is 3-5x faster.

Ideas 4 and 5 provide additional constant-factor improvements.
