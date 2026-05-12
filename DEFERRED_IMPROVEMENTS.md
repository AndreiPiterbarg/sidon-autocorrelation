# Deferred Algorithmic Improvements

> *Historical session note. For current project state see README.md and NOTES_INDEX.md. Both lower-bound proofs are now complete; the framing below dates from earlier exploration.*


These changes are mathematically sound and would improve speed/gap-closure,
but were deferred due to implementation complexity. All are well-defined and
ready to implement once the current proof run completes.

---

## 1. Z/2 Time-Reversal Symmetry Block-Diagonalization

**Expected speedup:** 4× on the dominant 969×969 moment PSD cone (→ 2× overall bisection speedup)

**Math:** The autoconvolution objective and all Lasserre constraints are invariant under
bin-reversal σ: i → d−1−i. Formally:
- Window matrices: M_W[i,j] = M_{W'}[d−1−i, d−1−j] where W' is the reflected window.
  The set of all windows is closed under reflection.
- Moment PSD: M_3(y∘σ) ⪰ 0 iff M_3(y) ⪰ 0 (σ acts as a linear permutation).
- Localizing constraints: set {M_2(μ_i·y) ⪰ 0, i=0..d−1} is permuted by σ, not destroyed.
- Consistency: y_α = Σ_i y_{α+e_i} maps to y_{σ(α)} = Σ_j y_{σ(α)+e_j} (re-index). ✓

**Symmetrization argument (why bound is preserved):** Any optimal μ* can be averaged
with its reflection (μ* + σ(μ*))/2, achieving the same objective. So restricting to
σ-symmetric moments gives the same val(d).

**Implementation (3–5 days):**
1. Partition the moment basis into self-paired (α = σ(α)) and paired (α, σ(α)) with α ≠ σ(α).
2. Introduce even/odd coordinates: u_α = (y_α + y_{σ(α)})/√2, v_α = (y_α − y_{σ(α)})/√2.
3. Add equality constraints: v_α = 0 for all pairs (enforcing symmetry).
4. The 969×969 moment PSD cone splits into two ~485×485 blocks (even and odd subspace).
5. Re-index all ab_eiej, t_pick arrays into the new basis.

**Reference:** Gatermann & Parrilo (2004), "Symmetry groups, semidefinite programs,
and sums of squares."

---

## 2. Bisection Elimination — Single Solve with Lazy Cuts

**Expected speedup:** 10× on per-round wall time

**Math:** Instead of running 12 bisection steps per CG round to find the feasibility
boundary, solve the direct epigraph problem once:
  min t  s.t.  (base constraints) ∧ (spectral cuts for active windows)
The ADMM optimal t* IS the lower bound directly. No bisection needed.

After each outer iteration, check violations at y*(t*) and add new spectral cuts.
This is one ADMM solve instead of 12, converging to the same bound asymptotically.

**Implementation (3–4 days):**
1. Add t as a variable in the minimize objective (already done in base problem).
2. Remove the phase-1 feasibility augmentation; solve the direct min-t problem.
3. Use the ADMM dual objective for the lower bound certificate.
4. Between outer iterations: add violation cuts, warm-start from current iterate.
5. The CG linear system updates: only A.data changes (Sherman-Morrison-trivial for
   scalar cuts), so the Cholesky/CG preconditioner can be warm-updated.

---

## 3. Block-Jacobi CG Preconditioner on Banded Clique Blocks

**Expected speedup:** 2–3× on CG iterations per ADMM step (→ ~2× ADMM speedup)

**Math:** The linear system (σI + ρA^TA) that CG solves inherits block structure from
the banded cliques. Specifically, the clique-restricted moment and localizing matrices
produce dense blocks of size n_cb × n_cb = 153 × 153 in A^TA. The block-Jacobi
preconditioner:
  P_block = diag(σI_b + ρ A_b^T A_b) for each clique block b

Applied via batched triangular solves (Cholesky per block, cached).

**Implementation (2–3 days):**
1. Identify which columns of A belong to each clique's localizing basis.
2. Compute A_b (the sub-matrix for clique b) once per round.
3. Cholesky-factor (σI + ρ A_b^T A_b) for each block.
4. In `_torch_cg`, replace `precond_inv` (diagonal) with block-Cholesky application.
5. Re-factor only when ρ changes.

---

## 4. Syevdx Range-Selection Eigendecomposition

**Expected speedup:** 2–3× on PSD projection for the 969×969 moment cone

**Math:** `torch.linalg.eigh` computes ALL n eigenpairs of an n×n matrix, but for
PSD projection Π_+(X) = V max(Λ,0) V^T, only the NEGATIVE eigenvalue directions
matter. cuSOLVER's `syevdx` computes eigenpairs in a specified range [λ_lo, λ_hi],
e.g., [−∞, 0] to get only negative eigenpairs.

For typical ADMM iterates near convergence, most matrices are near-PSD and have
only k << n negative eigenvalues. Computing only those k gives O(n·k²) instead of
O(n³).

**Implementation (2–3 days):**
1. Use `torch.linalg.eigh` but discard positive eigenvalues before reconstruction.
   (Simple version, same cost but less GPU memory for eigenvector storage.)
2. Full version: call `cusolverDnXsyevdx` directly via ctypes or a PyTorch extension,
   requesting only eigenvalues in [−∞, −eps].
3. Warm-start the eigenbasis from the previous ADMM iteration (Krylov subspace warm).

---

## 5. σ Auto-Scaling for the CG Linear System

**Expected speedup:** 1.5–2× on CG iterations

**Math:** The linear system (σI + ρA^TA)x = b has condition number κ ≈ ρ‖A‖²/σ
when σ << ρ‖A‖². After Ruiz equilibration ‖A_scaled‖_∞ ≈ 1, so ‖A_scaled‖_F² ≈ nnz.
Setting σ = max(1e-10, nnz/n × 1e-6) balances the regularizer with the problem scale.

Currently σ=1e-6 constant regardless of problem size, leaving the system poorly
conditioned when nnz/n >> 1 (which it is: nnz≈1.5M, n=74K, ratio≈20 → σ should be ~2e-5).

**Implementation (30 min):**
```python
# In admm_solve and ADMMSolver.__init__, after Ruiz equilibration:
nnz = A_work.nnz
n = A_csc.shape[1]
sigma = max(1e-10, 1e-6 * nnz / n)
```

---

## 6. Cloninger-Steinerberger Warm-Start for Pseudo-Moments

**Expected speedup:** Round 0 convergence in <100 iters (vs 290 currently)

**Math:** The extremal distribution μ* at d=16 is numerically known from the cascade
solver. Its pseudo-moments y*_α = E_{μ*}[x^α] can be computed by numerical integration:
μ* is a discrete measure, so y*_α = Σ_i (μ*)_i^{|α|} × basis_integral(α, bin_i).
Initialize ADMM workspace with these moments as the starting primal iterate.

**Implementation (1–2 days):**
1. Run `cloninger-steinerberger/cpu/run_cascade.py` at d=16 to get the extremal μ*.
2. Compute y*_α for all α in S using the measure's bin weights.
3. In `admm_solve`, initialize `warm_start={'x': y*_extended}` before the solve.

---

## 7. Adaptive Restart of Spectral Cut Eigenvectors

**Expected improvement:** Tighter gap closure per round

**Math:** Currently, spectral cut eigenvectors are computed once (at violation detection)
and kept fixed across bisection steps. As y moves during ADMM, the cuts may become
stale. An adaptive scheme:
- After each CG round, recompute eigenvectors of L_W(y_last_feas) for active windows.
- Replace the stored eigenvectors with the updated ones.
- This ensures cuts always point in the currently most-violated direction.

**Implementation (1 day):** In the end-of-round violation check, for ACTIVE windows
(not just violated ones), recompute L_W(y_last_feas), run eigh, and update active_cuts.

---

## Priority Order (once proof run completes)

If `lb > 1.2802` is reached: DONE. Certificate extraction only.

If `lb` plateaus below 1.2802 (unlikely for L3 d=16):
1. **Item 7** (adaptive eigenvectors) — 1 day, likely fixes gap
2. **Item 1** (Z/2 symmetry) — 3 days, structural improvement
3. **Item 2** (bisection elimination) — 3 days, large wall-time reduction
4. **Item 5** (σ scaling) — 30 min, try immediately
5. **Item 3** (block-Jacobi) — 2 days
6. **Item 4** (syevdx) — 2 days
