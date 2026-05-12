/-
IntervalBnB — Section 2 of THEOREM.md.  Scale-factor derivation.

Key identity:
    |I_W| = ℓ / (2d).

This is exactly the length of the merged union of `ℓ - 1` half-overlapping
intervals of length `1/d`. The arithmetic identity is immediate:

    (s_lo + ℓ)/(2d) - s_lo/(2d) = ℓ/(2d).

For the "merged block" interpretation, note that each pair-sum `k` in
`K_W = {s_lo, …, s_lo + ℓ - 2}` contributes an interval of half-width
units `[k, k+2]`, and consecutive intervals overlap by one half-width.
The union runs from `s_lo` to `s_lo + ℓ` in half-width units, giving
total length `ℓ/(2d)`.
-/

import IntervalBnB.Defs

set_option linter.mathlibStandardSet false
set_option autoImplicit false
set_option relaxedAutoImplicit false

open scoped BigOperators
open scoped Classical
open MeasureTheory

noncomputable section

namespace IntervalBnB

variable {d : ℕ}

/-- The length of `I_W` as a real number.  Pure arithmetic identity. -/
lemma window_interval_endpoint_diff (W : Window d) :
    (-(1 : ℝ)/2 + ((W.sLo : ℝ) + W.ell)/(2*d)) -
      (-(1 : ℝ)/2 + (W.sLo : ℝ)/(2*d)) = (W.ell : ℝ)/(2*d) := by
  ring

/-- Lebesgue measure of `I_W` as a real number. -/
lemma window_interval_volume (W : Window d) (hd : 0 < d) :
    (MeasureTheory.volume (window_interval W)).toReal = (W.ell : ℝ)/(2*d) := by
  have hlt : -(1 : ℝ)/2 + (W.sLo : ℝ)/(2*d) ≤
              -(1 : ℝ)/2 + ((W.sLo : ℝ) + W.ell)/(2*d) := by
    have hd_pos : (0 : ℝ) < 2*d := by
      have : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
      linarith
    have hell_pos : (0 : ℝ) ≤ (W.ell : ℝ) := by exact_mod_cast Nat.zero_le _
    have h1 : (W.sLo : ℝ)/(2*d) ≤ ((W.sLo : ℝ) + W.ell)/(2*d) := by
      apply div_le_div_of_nonneg_right _ (le_of_lt hd_pos); linarith
    linarith
  rw [window_interval, Real.volume_Icc]
  rw [window_interval_endpoint_diff]
  have hnn : (0 : ℝ) ≤ (W.ell : ℝ)/(2*d) := by
    apply div_nonneg
    · exact_mod_cast Nat.zero_le _
    · have : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
      linarith
  simp [ENNReal.toReal_ofReal, hnn]

/-- Positivity of `|I_W|`. -/
lemma window_interval_length_pos (W : Window d) (hd : 0 < d) :
    (0 : ℝ) < (W.ell : ℝ) / (2*d) := by
  have hell : 0 < W.ell := W.ell_pos
  have hell_R : (0 : ℝ) < (W.ell : ℝ) := by exact_mod_cast hell
  have hd_R : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
  positivity

end IntervalBnB

end -- noncomputable section
