/-
Copyright (c) 2026 Sidon Project. All rights reserved.

# Principal Submatrix of PSD Matrix is PSD

This file proves that principal submatrices of positive semidefinite matrices
are positive semidefinite. This is the mathematical foundation for the
clique-restricted Lasserre relaxation being sound.

## Code correspondence
- `lasserre/cliques.py:_add_sparse_moment_constraints`: clique moment PSD
  is a principal submatrix of the full moment matrix M_k(y)
- `lasserre_highd.py` lines 37-38: "Clique M_2^{I_c}(y*) ≽ 0: principal
  submatrix of M_2(y*) ≽ 0"
- `lasserre_highd.py:_check_violations_highd` lines 693-696: "L_W^{I_c}
  is a principal submatrix of L_W. By Cauchy interlacing..."

## Mathematical statement
If M ∈ ℝⁿˣⁿ is PSD and e : m → n is any function, then M.submatrix e e is PSD.
For injective e (embedding), this gives the principal submatrix result.

The proof is standard: for any w ∈ ℝᵐ,
  w* · (M.submatrix e e) · w = v* · M · v ≥ 0
where v = ∑ᵢ wᵢ · δ_{e(i)} is the zero-extension of w.
-/
import Mathlib

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Lasserre.PSDSubmatrix

/-! ## Quadratic form characterization of PSD -/

/-- A real symmetric matrix is PSD iff its quadratic form is nonneg.
    This is the working definition used throughout the Lasserre code:
    `np.linalg.eigvalsh(L)[:, 0] ≥ -tol` in `_check_violations_highd`. -/
theorem psd_iff_quadform_nonneg {n : Type*} [Fintype n] [DecidableEq n]
    (M : Matrix n n ℝ) (hM_sym : M.IsHermitian) :
    M.PosSemidef ↔ (M.IsHermitian ∧ ∀ x : n → ℝ, 0 ≤ x ⬝ᵥ M.mulVec x) :=
  ⟨fun h => ⟨h.1, h.2⟩, fun h => ⟨h.1, h.2⟩⟩

/-! ## Principal submatrix preserves PSD

The key mathematical fact: if M is PSD over indices n, and we restrict to
a subset of indices via e : m → n, then M.submatrix e e is PSD.

This justifies the clique-restricted Lasserre relaxation:
- Full moment matrix M_k(y) is PSD (for true moments)
- Clique basis restricts to indices I_c ⊆ {all monomials}
- M_k^{I_c}(y) = M_k(y).submatrix e e is PSD (principal submatrix)
- Therefore imposing M_k^{I_c}(y) ≽ 0 is a NECESSARY condition
- This makes the clique relaxation SOUND (never excludes true moments)
-/

/-- Principal submatrix of a PSD matrix is PSD.

    Proof sketch (formalized via Mathlib's Matrix.PosSemidef.submatrix):
    For any w : m → ℝ, define v : n → ℝ by v_j = Σ_{i : e(i)=j} w_i.
    Then w^T (M.submatrix e e) w = v^T M v ≥ 0 (since M is PSD).

    In the code, this is used for:
    1. Clique moment PSD: M_k(y)[I_c, I_c] ≽ 0  (`_add_sparse_moment_constraints`)
    2. Clique localizing PSD: L_i(y)[I_c, I_c] ≽ 0  (`_add_sparse_localizing_constraints`)
    3. Window localizing PSD: L_W(y)[I_c, I_c] ≽ 0  (`_add_window_psd_highd`)
    4. Violation checking via Cauchy interlacing  (`_check_violations_highd`) -/
theorem principal_submatrix_psd {n m : Type*}
    [Fintype n] [DecidableEq n] [Fintype m] [DecidableEq m]
    (M : Matrix n n ℝ) (hM : M.PosSemidef) (e : m → n) :
    (M.submatrix e e).PosSemidef :=
  hM.submatrix e

/-! ## Cauchy interlacing consequence

The Cauchy interlacing theorem implies that if A is a principal submatrix of B,
then λ_min(A) ≥ λ_min(B). Equivalently, if λ_min(A) < 0, then λ_min(B) < 0.

This is used in `_check_violations_highd` (line 693-696): if the clique-
restricted localizing matrix L_W^{I_c} has a negative eigenvalue, then the
full L_W also has a negative eigenvalue — it's a GENUINE violation. -/

/-- If a principal submatrix is not PSD, the full matrix is not PSD.
    This is the contrapositive of `principal_submatrix_psd`.

    Used in: `_check_violations_highd` — violations detected on the
    clique-restricted matrix are genuine violations of the full matrix. -/
theorem not_psd_of_submatrix_not_psd {n m : Type*}
    [Fintype n] [DecidableEq n] [Fintype m] [DecidableEq m]
    (M : Matrix n n ℝ) (e : m → n)
    (h : ¬ (M.submatrix e e).PosSemidef) :
    ¬ M.PosSemidef :=
  fun hM => h (principal_submatrix_psd M hM e)

end Lasserre.PSDSubmatrix
