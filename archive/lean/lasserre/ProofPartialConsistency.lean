/-
Copyright (c) 2026 Sidon Project. All rights reserved.

# Partial Consistency Soundness for Lasserre SDP Hierarchy

This file proves the mathematical correctness of using INEQUALITY (not equality)
for partial consistency constraints when not all children moments are in the
reduced moment set S.

## Code correspondence
- `lasserre_highd.py` lines 486-502 (partial consistency inequality construction)
- `run_scs_direct.py` lines 157-163 (SCS nonneg cone encoding of inequalities)
- CLAUDE.md: "Partial consistency must use INEQUALITY ... not equality.
  Equality with missing children forces unmapped moments to zero → lb > val(d)."

## Mathematical statement
For true moments y* of a probability distribution μ on Δ_d:
- Full consistency: y*_α = Σ_i y*_{α+e_i}  (from Σ x_i = 1)
- Since y*_{α+e_i} ≥ 0 (nonnegativity of μ on ℝ^d_{≥0}), we have:
    y*_α ≥ Σ_{i∈S'} y*_{α+e_i}  for any subset S' ⊆ {0,...,d-1}

Using equality with partial children overconstrain the problem and is UNSOUND.
-/
import Mathlib

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Finset
open scoped BigOperators

namespace Lasserre.PartialConsistency

variable {d : ℕ}

/-! ## Core partial consistency theorems -/

/-- Sum over a subset ≤ sum over the full set when all terms are nonneg.
    This is the mathematical justification for using `≥` instead of `=`
    in `lasserre_highd.py` line 497-501 when `partial_mask` is true.

    Corresponds to: `consist_iq_lists` construction in `_precompute_highd`. -/
theorem sum_subset_le_sum_univ (y : Fin d → ℝ)
    (hy_nonneg : ∀ i, 0 ≤ y i) (S : Finset (Fin d)) :
    ∑ i ∈ S, y i ≤ ∑ i : Fin d, y i :=
  Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ S)
    (fun x _ _ => hy_nonneg x)

/-- Partial consistency soundness: if parent = Σ_all children (full consistency)
    and all children are ≥ 0, then parent ≥ Σ_subset children.

    This is the theorem that justifies partial consistency INEQUALITY in the
    Lasserre hierarchy. The code at `_precompute_highd` lines 486-502 correctly
    constructs `consist_iq_lists` with this inequality for parents whose children
    are not all in the reduced moment set S. -/
theorem partial_consistency_sound (children : Fin d → ℝ)
    (parent : ℝ)
    (h_full : parent = ∑ i : Fin d, children i)
    (h_nonneg : ∀ i, 0 ≤ children i)
    (S : Finset (Fin d)) :
    ∑ i ∈ S, children i ≤ parent := by
  rw [h_full]
  exact sum_subset_le_sum_univ children h_nonneg S

/-- The complement sum is nonneg — this is the "missing mass" that
    partial equality would force to zero.

    At d=128 with bandwidth 16, a degree-3 parent might have ~16 children
    in S out of 128 total. The complement contains ~112 nonneg terms. -/
theorem complement_sum_nonneg (y : Fin d → ℝ)
    (hy_nonneg : ∀ i, 0 ≤ y i) (S : Finset (Fin d)) :
    0 ≤ ∑ i ∈ Sᶜ, y i :=
  Finset.sum_nonneg (fun i _ => hy_nonneg i)

/-! ## Unsoundness of partial equality -/

/-- **UNSOUNDNESS OF PARTIAL EQUALITY**: If we use y_α = Σ_{i∈S'} y_{α+e_i}
    (equality with partial children), then every child NOT in S' must be zero.

    This overconstrain forces unmapped moments to zero, shrinking the
    feasible set and producing lb > val(d) — an UNSOUND lower bound.

    Corresponds to: CLAUDE.md warning "Equality with missing children forces
    unmapped moments to zero → lb > val(d)."

    Empirical verification: at d=8, using equality gives lb=1.28 > val(8)=1.21. -/
theorem partial_equality_forces_zero (children : Fin d → ℝ)
    (parent : ℝ)
    (h_full : parent = ∑ i : Fin d, children i)
    (h_nonneg : ∀ i, 0 ≤ children i)
    (S : Finset (Fin d))
    (h_partial_eq : parent = ∑ i ∈ S, children i) :
    ∀ i, i ∉ S → children i = 0 := by
  intro i hi
  -- Step 1: decompose full sum = sum over S + sum over complement
  have h_decomp : (∑ j ∈ S, children j) + ∑ j ∈ Sᶜ, children j =
      ∑ j : Fin d, children j :=
    Finset.sum_add_sum_compl S children
  -- Step 2: complement sum = 0
  have h_compl_zero : ∑ j ∈ Sᶜ, children j = 0 := by linarith
  -- Step 3: children i ≤ complement sum (single nonneg term in nonneg sum)
  have h_mem_compl : i ∈ Sᶜ := Finset.mem_compl.mpr hi
  have h_le_compl : children i ≤ ∑ j ∈ Sᶜ, children j :=
    Finset.single_le_sum (fun j _ => h_nonneg j) h_mem_compl
  -- Step 4: 0 ≤ children i ≤ 0, so children i = 0
  linarith [h_nonneg i]

/-- Every element of the complement is forced to zero — the bulk variant. -/
theorem partial_equality_forces_all_complement_zero (children : Fin d → ℝ)
    (parent : ℝ)
    (h_full : parent = ∑ i : Fin d, children i)
    (h_nonneg : ∀ i, 0 ≤ children i)
    (S : Finset (Fin d))
    (h_partial_eq : parent = ∑ i ∈ S, children i) :
    ∀ i ∈ Sᶜ, children i = 0 :=
  fun i hi => partial_equality_forces_zero children parent h_full h_nonneg S h_partial_eq i
    (Finset.mem_compl.mp hi)

end Lasserre.PartialConsistency
