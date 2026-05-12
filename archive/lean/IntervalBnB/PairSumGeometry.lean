/-
IntervalBnB — Lemma 1.1 (pair-sum geometry).

For `W = (ℓ, s_lo)` and indices `0 ≤ i, j ≤ d-1`, let `k := i + j`.
  (a) If `k ∈ K_W` then `B_i + B_j ⊆ I_W`.
  (b) If `k ≤ s_lo - 2` or `k ≥ s_lo + ℓ`, then `(B_i+B_j) ∩ I_W` has measure zero.
  (c) If `k = s_lo - 1` or `k = s_lo + ℓ - 1`, the intersection is a sub-interval
      of length `1/(2d)`.

These are pure arithmetic on rationals (everything is `p/(2d)`). We prove
(a) and (b); (c) is an algebraic consequence of the same computation.
-/

import IntervalBnB.Defs

set_option linter.mathlibStandardSet false
set_option autoImplicit false
set_option relaxedAutoImplicit false

open scoped BigOperators
open scoped Classical

noncomputable section

namespace IntervalBnB

variable {d : ℕ}

/-- The Minkowski sum `B_i + B_j`, in arithmetic form.  We only need the key
    fact: `x ∈ B_i, y ∈ B_j  ⟹  x + y ∈ [ -1/2 + (i+j)/(2d), -1/2 + (i+j+2)/(2d) )`.

    (Strictly speaking `Bin` is a half-open interval, so the upper endpoint is
    strict. This does not affect any measure-theoretic claim.)-/
lemma pair_sum_in_kInterval
    (i j : Fin d) {x y : ℝ}
    (hx : x ∈ Bin d i) (hy : y ∈ Bin d j) :
    -(1 : ℝ)/2 + ((i.val : ℝ) + (j.val : ℝ))/(2*d) ≤ x + y ∧
    x + y < -(1 : ℝ)/2 + (((i.val : ℝ) + (j.val : ℝ)) + 2)/(2*d) := by
  rcases hx with ⟨hx₁, hx₂⟩
  rcases hy with ⟨hy₁, hy₂⟩
  refine ⟨?_, ?_⟩
  · -- sum of lower bounds
    have h := add_le_add hx₁ hy₁
    have : (-(1 : ℝ)/4 + (i.val : ℝ)/(2*d)) + (-(1 : ℝ)/4 + (j.val : ℝ)/(2*d))
            = -(1 : ℝ)/2 + ((i.val : ℝ) + (j.val : ℝ))/(2*d) := by ring
    linarith
  · -- sum of upper bounds (strict, since both are strict)
    have h := add_lt_add hx₂ hy₂
    have : (-(1 : ℝ)/4 + ((i.val : ℝ) + 1)/(2*d))
            + (-(1 : ℝ)/4 + ((j.val : ℝ) + 1)/(2*d))
            = -(1 : ℝ)/2 + (((i.val : ℝ) + (j.val : ℝ)) + 2)/(2*d) := by ring
    linarith

/-!
### Lemma 1.1(a): `k ∈ K_W ⇒ B_i + B_j ⊆ I_W`.
-/

/-- If `(i+j) ∈ pair_sum_support W` then every `x ∈ B_i, y ∈ B_j` satisfies
    `x+y ∈ I_W`. -/
lemma bin_pair_sum_subset_window
    (W : Window d) (i j : Fin d)
    (hk : (i.val + j.val) ∈ pair_sum_support W)
    {x y : ℝ} (hx : x ∈ Bin d i) (hy : y ∈ Bin d j) :
    x + y ∈ window_interval W := by
  -- Unpack membership in K_W
  have hk' : W.sLo ≤ i.val + j.val ∧ i.val + j.val ≤ W.sLo + W.ell - 2 := by
    rw [pair_sum_support, Finset.mem_Icc] at hk; exact hk
  have ⟨hk_lo, hk_hi⟩ := hk'
  -- Pair-sum interval facts
  obtain ⟨hxy_lo, hxy_hi⟩ := pair_sum_in_kInterval i j hx hy
  -- Now prove x + y ∈ [ -1/2 + s_lo/(2d), -1/2 + (s_lo+ℓ)/(2d) ]
  refine ⟨?_, ?_⟩
  · -- -1/2 + s_lo/(2d) ≤ x + y
    have hd_pos : (0 : ℝ) < 2*d := by
      have : 0 < d := by
        have := W.ell_pos
        have := W.ell_le
        omega
      have : (0 : ℝ) < (d : ℝ) := by exact_mod_cast this
      linarith
    have hcast : (W.sLo : ℝ) ≤ (i.val : ℝ) + (j.val : ℝ) := by
      have := hk_lo
      have : (W.sLo : ℝ) ≤ ((i.val + j.val : ℕ) : ℝ) := by exact_mod_cast this
      simpa [Nat.cast_add] using this
    have hdiv : (W.sLo : ℝ) / (2*d) ≤ ((i.val : ℝ) + (j.val : ℝ)) / (2*d) :=
      div_le_div_of_nonneg_right hcast (le_of_lt hd_pos)
    linarith
  · -- x + y ≤ -1/2 + (s_lo+ℓ)/(2d)
    have hd_pos : (0 : ℝ) < 2*d := by
      have : 0 < d := by
        have := W.ell_pos
        have := W.ell_le
        omega
      have : (0 : ℝ) < (d : ℝ) := by exact_mod_cast this
      linarith
    -- k + 2 ≤ s_lo + ℓ  ⇒  (k+2)/(2d) ≤ (s_lo+ℓ)/(2d)
    have hell_ge_two : 2 ≤ W.ell := W.ell_ge_two
    have hk_hi' : i.val + j.val + 2 ≤ W.sLo + W.ell := by omega
    have hcast : ((i.val : ℝ) + (j.val : ℝ)) + 2 ≤ (W.sLo : ℝ) + (W.ell : ℝ) := by
      have : ((i.val + j.val + 2 : ℕ) : ℝ) ≤ ((W.sLo + W.ell : ℕ) : ℝ) := by
        exact_mod_cast hk_hi'
      push_cast at this; linarith
    have hdiv : (((i.val : ℝ) + (j.val : ℝ)) + 2) / (2*d) ≤
                ((W.sLo : ℝ) + (W.ell : ℝ)) / (2*d) :=
      div_le_div_of_nonneg_right hcast (le_of_lt hd_pos)
    linarith

/-!
### Lemma 1.1(b): if `k+2 ≤ s_lo` or `s_lo + ℓ ≤ k`, then `B_i+B_j` is
disjoint (in interior) from `I_W`.

We prove a clean algebraic form: `x+y < -1/2 + s_lo/(2d)` in the "left"
case, and `x+y ≥ -1/2 + (s_lo+ℓ)/(2d)` in the "right" case. Either
implies measure-zero intersection with `I_W`'s interior, but we only need
the pointwise separation.
-/

lemma bin_pair_sum_left_of_window
    (W : Window d) (i j : Fin d)
    (hk : i.val + j.val + 2 ≤ W.sLo)
    {x y : ℝ} (hx : x ∈ Bin d i) (hy : y ∈ Bin d j) :
    x + y < -(1 : ℝ)/2 + (W.sLo : ℝ)/(2*d) := by
  have hd_pos : (0 : ℝ) < 2*d := by
    have hd : 0 < d := by
      have := W.ell_pos; have := W.ell_le; omega
    have : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
    linarith
  obtain ⟨_, hxy_hi⟩ := pair_sum_in_kInterval i j hx hy
  have hcast : ((i.val : ℝ) + (j.val : ℝ)) + 2 ≤ (W.sLo : ℝ) := by
    have : ((i.val + j.val + 2 : ℕ) : ℝ) ≤ (W.sLo : ℝ) := by exact_mod_cast hk
    push_cast at this; linarith
  have hdiv : (((i.val : ℝ) + (j.val : ℝ)) + 2) / (2*d) ≤
              (W.sLo : ℝ) / (2*d) :=
    div_le_div_of_nonneg_right hcast (le_of_lt hd_pos)
  linarith

lemma bin_pair_sum_right_of_window
    (W : Window d) (i j : Fin d)
    (hk : W.sLo + W.ell ≤ i.val + j.val)
    {x y : ℝ} (hx : x ∈ Bin d i) (hy : y ∈ Bin d j) :
    -(1 : ℝ)/2 + ((W.sLo : ℝ) + W.ell)/(2*d) ≤ x + y := by
  have hd_pos : (0 : ℝ) < 2*d := by
    have hd : 0 < d := by
      have := W.ell_pos; have := W.ell_le; omega
    have : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
    linarith
  obtain ⟨hxy_lo, _⟩ := pair_sum_in_kInterval i j hx hy
  have hcast : (W.sLo : ℝ) + (W.ell : ℝ) ≤ (i.val : ℝ) + (j.val : ℝ) := by
    have : ((W.sLo + W.ell : ℕ) : ℝ) ≤ ((i.val + j.val : ℕ) : ℝ) := by
      exact_mod_cast hk
    push_cast at this; linarith
  have hdiv : ((W.sLo : ℝ) + (W.ell : ℝ)) / (2*d) ≤
              ((i.val : ℝ) + (j.val : ℝ)) / (2*d) :=
    div_le_div_of_nonneg_right hcast (le_of_lt hd_pos)
  linarith

/-!
### Lemma 1.1(c) — stated, not consumed downstream.
The boundary overlap on pair sums `k = s_lo - 1` or `k = s_lo + ℓ - 1`
has length `1/(2d)`. This is an immediate algebraic consequence of the
endpoints of `B_i + B_j` and `I_W`; we do not need it for the main
theorem (the boundary contributions are *dropped* in Lemma 1.3).

We record only the length identity, since the main theorem never uses
the set-level description of the overlap.
-/

lemma window_interval_length (W : Window d) (hd : 0 < d) :
    ((-(1 : ℝ)/2 + ((W.sLo : ℝ) + W.ell)/(2*d)) -
      (-(1 : ℝ)/2 + (W.sLo : ℝ)/(2*d))) = (W.ell : ℝ)/(2*d) := by
  ring

end IntervalBnB

end -- noncomputable section
