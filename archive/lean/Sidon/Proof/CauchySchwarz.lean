/-
Sidon Autocorrelation Project — Bin Definitions

Bin interval, f_bin (restriction of f to a bin), and basic properties.
-/

import Mathlib
import Sidon.Defs

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-- The i-th bin interval [-(1/4) + i·δ, -(1/4) + (i+1)·δ). -/
noncomputable def bin_interval (n : ℕ) (i : Fin (2 * n)) : Set ℝ :=
  let δ := 1 / (4 * n : ℝ)
  let a := -(1/4 : ℝ) + i * δ
  let b := -(1/4 : ℝ) + (i + 1) * δ
  Set.Ico a b

/-- Restriction of f to bin i using bin_interval. -/
noncomputable def f_bin (f : ℝ → ℝ) (n : ℕ) (i : Fin (2 * n)) : ℝ → ℝ :=
  Set.indicator (bin_interval n i) f

lemma f_bin_nonneg (f : ℝ → ℝ) (hf : 0 ≤ f) (n : ℕ) (i : Fin (2 * n)) :
  0 ≤ f_bin f n i := by
    apply Set.indicator_nonneg; intro x hx; exact hf x

lemma integral_f_bin (f : ℝ → ℝ) (n : ℕ) (i : Fin (2 * n)) :
  MeasureTheory.integral MeasureTheory.volume (f_bin f n i) = bin_masses f n i := by
    have h_integral : MeasureTheory.integral MeasureTheory.MeasureSpace.volume (f_bin f n i) = MeasureTheory.integral MeasureTheory.MeasureSpace.volume (Set.indicator (bin_interval n i) f) := by
      simp [f_bin];
    convert h_integral using 1

lemma f_bin_integrable (f : ℝ → ℝ) (hf : MeasureTheory.Integrable f MeasureTheory.volume) (n : ℕ) (i : Fin (2 * n)) :
  MeasureTheory.Integrable (f_bin f n i) MeasureTheory.volume := by
  refine' MeasureTheory.Integrable.mono' _ _ _;
  refine' fun x => |f x|;
  · exact hf.norm;
  · exact MeasureTheory.AEStronglyMeasurable.indicator ( hf.1 ) ( measurableSet_Ico );
  · filter_upwards [ ] with x using by rw [ f_bin ] ; rw [ Set.indicator_apply ] ; aesop;

end -- noncomputable section
