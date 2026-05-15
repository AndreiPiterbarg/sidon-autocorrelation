/-
Sidon Autocorrelation Project — Core Definitions

This module provides the core definitions used by the main proof:

* `autoconvolution_ratio f := ‖f * f‖_∞ / (∫ f)²` — the ratio whose
  supremum over admissible `f` defines the Sidon autocorrelation
  constant C_{1a}.

* `convolution_nonneg` — pointwise nonnegativity of the convolution of
  two nonneg functions.

The headline theorems in `Sidon.MultiScale` and `Sidon.MultiScaleSchwartz`
bound `autoconvolution_ratio f` from below.
-/

import Mathlib

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

-- ═══════════════════════════════════════════════════════════════════════════════
-- Core Definitions
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The autoconvolution ratio R(f) = ‖f*f‖_∞ / (∫f)². -/
noncomputable def autoconvolution_ratio (f : ℝ → ℝ) : ℝ :=
  let conv := MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume
  let norm_inf := (MeasureTheory.eLpNorm conv ⊤ MeasureTheory.volume).toReal
  let integral := MeasureTheory.integral MeasureTheory.volume f
  norm_inf / (integral ^ 2)

/-- Convolution of nonneg functions is nonneg. -/
theorem convolution_nonneg {f g : ℝ → ℝ} (hf : ∀ x, 0 ≤ f x) (hg : ∀ x, 0 ≤ g x) :
    ∀ x, 0 ≤ MeasureTheory.convolution f g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
  intro x
  simp [MeasureTheory.convolution]
  exact MeasureTheory.integral_nonneg fun t => mul_nonneg (hf t) (hg (x - t))

end -- noncomputable section
