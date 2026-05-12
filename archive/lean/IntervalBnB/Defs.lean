/-
IntervalBnB — Core Definitions

These definitions EXACTLY mirror the Python sources that build the same
objects:
  * `lasserre/core.py`        — `build_window_matrices` (prefactor 2d/ell)
  * `interval_bnb/windows.py` — window enumeration (ell, s_lo)
  * `interval_bnb/symmetry.py` — the σ involution

All of the Lean objects below are indexed by a dimension `d : ℕ` with
`d ≥ 2` as a running hypothesis (where applicable).
-/

import Mathlib

set_option linter.mathlibStandardSet false
set_option autoImplicit false
set_option relaxedAutoImplicit false

open scoped BigOperators
open scoped Classical

noncomputable section

namespace IntervalBnB

/-!
## Bins `B_i` and bin masses `μ_i`.

Matches `lasserre/core.py` implicitly (the bin structure) and the Section 1.1
setup in `interval_bnb/THEOREM.md`.
-/

/-- Bin `B_i = [ -1/4 + i/(2d),  -1/4 + (i+1)/(2d) )`.
    Matches THEOREM.md §1.1 "B_i" definition. -/
def Bin (d : ℕ) (i : Fin d) : Set ℝ :=
  Set.Ico (-(1 : ℝ)/4 + (i.val : ℝ)/(2*d)) (-(1 : ℝ)/4 + ((i.val : ℝ)+1)/(2*d))

/-- Bin mass μ_i := ∫_{B_i} f.
    Matches THEOREM.md §1.1 and `bin_masses` in `lean/Sidon/Defs.lean`. -/
def bin_mass (d : ℕ) (f : ℝ → ℝ) (i : Fin d) : ℝ :=
  MeasureTheory.integral MeasureTheory.volume (Set.indicator (Bin d i) f)

/-!
## Windows `W = (ℓ, s_lo)` and derived objects.

Matches `lasserre/core.py` lines 141-157 (`build_window_matrices`) and
`interval_bnb/windows.py` (`build_windows`).
-/

/-- The window enumeration `W_d = {(ℓ, s_lo) : 2 ≤ ℓ ≤ 2d, 0 ≤ s_lo ≤ 2d-ℓ}`.
    Matches `lasserre/core.py:149-150`:
      `windows = [(ell, s) for ell in range(2, 2d+1) for s in range(2d-ell+1)]`. -/
def Window (d : ℕ) : Type :=
  {p : ℕ × ℕ // 2 ≤ p.1 ∧ p.1 ≤ 2*d ∧ p.2 + p.1 ≤ 2*d}

namespace Window

variable {d : ℕ}

/-- Length component ℓ of window `W = (ℓ, s_lo)`. -/
def ell (W : Window d) : ℕ := W.val.1

/-- Lower endpoint component `s_lo` of window `W = (ℓ, s_lo)`. -/
def sLo (W : Window d) : ℕ := W.val.2

lemma ell_ge_two (W : Window d) : 2 ≤ W.ell := W.property.1

lemma ell_le (W : Window d) : W.ell ≤ 2*d := W.property.2.1

lemma sLo_add_ell_le (W : Window d) : W.sLo + W.ell ≤ 2*d := W.property.2.2

lemma ell_pos (W : Window d) : 0 < W.ell := by
  have := W.ell_ge_two; omega

end Window

/-- Pair-sum support `K_W = {s_lo, s_lo+1, …, s_lo+ℓ-2}`, cardinality `ℓ-1`.
    Matches `lasserre/core.py:155` and `interval_bnb/windows.py:58-60`. -/
def pair_sum_support {d : ℕ} (W : Window d) : Finset ℕ :=
  Finset.Icc W.sLo (W.sLo + W.ell - 2)

/-- Window matrix entry `M_W[i,j]`.
    Matches `lasserre/core.py:155-156`:
      `mask = (sums >= s_lo) & (sums <= s_lo + ell - 2)`
      `(2.0 * d / ell) * mask.astype(np.float64)`. -/
def window_matrix {d : ℕ} (W : Window d) (i j : Fin d) : ℝ :=
  if (i.val + j.val) ∈ pair_sum_support W then (2*d : ℝ) / W.ell else 0

/-- The window `t`-interval `I_W = [ -1/2 + s_lo/(2d),  -1/2 + (s_lo+ℓ)/(2d) ]`.
    Matches THEOREM.md §1.2. -/
def window_interval {d : ℕ} (W : Window d) : Set ℝ :=
  Set.Icc (-(1 : ℝ)/2 + (W.sLo : ℝ)/(2*d))
          (-(1 : ℝ)/2 + ((W.sLo : ℝ) + W.ell)/(2*d))

/-!
## Simplex and `val(d)`.
-/

/-- The probability simplex `Δ_d := {μ ∈ ℝ^d : μ_i ≥ 0, ∑_i μ_i = 1}`. -/
def simplex_d (d : ℕ) : Set (Fin d → ℝ) :=
  {μ | (∀ i, 0 ≤ μ i) ∧ ∑ i, μ i = 1}

/-- Quadratic form `μ^T M_W μ`. -/
def quadForm {d : ℕ} (W : Window d) (μ : Fin d → ℝ) : ℝ :=
  ∑ i, ∑ j, μ i * window_matrix W i j * μ j

/-- `val(d) := inf_{μ ∈ Δ_d}  max_{W ∈ W_d}  μ^T M_W μ`.

    Matches THEOREM.md §1:
      `val(d) = min_{μ ∈ Δ_d} max_{W ∈ W_d} μ^T M_W μ`. -/
def val_d (d : ℕ) : ℝ :=
  sInf ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) '' simplex_d d)

/-!
## Admissible `f` for the Sidon problem.

Matches THEOREM.md opening paragraph: nonneg, Lebesgue measurable,
supported in `[-1/4, 1/4]`, with `∫ f = 1`.
-/

/-- The pointwise autoconvolution `f * f`.  For admissible `f` this is
    a nonneg L¹ function on ℝ, hence has a well-defined essential
    supremum. -/
def autoconv (f : ℝ → ℝ) : ℝ → ℝ :=
  MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume

/-- Essential supremum of `f * f` on the reference interval `[-1/2, 1/2]`.
    This is the `max` that appears in the Sidon definition of `C_{1a}`. -/
def essSupConv (f : ℝ → ℝ) : ℝ :=
  (MeasureTheory.eLpNorm
      (Set.indicator (Set.Icc (-(1 : ℝ)/2) (1/2)) (autoconv f))
      ⊤ MeasureTheory.volume).toReal

/-- `f` is admissible if it is nonnegative, measurable, supported in
    `[-1/4, 1/4]`, and integrates to 1.  We also require `f` to be
    Bochner integrable (which holds automatically for any nonneg
    `f` with `∫ f = 1`, but we record it explicitly as a separate
    regularity assumption). -/
structure Admissible (f : ℝ → ℝ) : Prop where
  nonneg      : ∀ x, 0 ≤ f x
  measurable  : Measurable f
  support     : ∀ x, f x ≠ 0 → x ∈ Set.Icc (-(1 : ℝ)/4) (1/4)
  integral_one : MeasureTheory.integral MeasureTheory.volume f = 1
  integrable   : MeasureTheory.Integrable f MeasureTheory.volume
  /-- Genuine regularity: the autoconvolution is essentially bounded.
      For the Sidon problem we quantify over `f ∈ L¹ ∩ L^∞` supported
      in `[-1/4, 1/4]` (step functions, piecewise-constant, etc. — the
      extremal regime); for every such `f`, `f * f` is continuous (in
      fact Lipschitz) and hence bounded on the reference window.
      This is strictly weaker than the conclusion of Lemma 1.2: it
      just says the essential supremum is some real number, without
      specifying any quantitative bound relating the supremum to the
      integral. -/
  autoconv_ess_lt_top :
    MeasureTheory.eLpNorm
      (Set.indicator (Set.Icc (-(1 : ℝ)/2) (1/2)) (autoconv f))
      ⊤ MeasureTheory.volume < ⊤

end IntervalBnB

end -- noncomputable section
