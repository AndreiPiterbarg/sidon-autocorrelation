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
-- Correction Term Support Lemmas
-- Source: output (6).lean (UUID: db9a6f0e)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Floor division approximation bound. -/
lemma nat_floor_approx (x : ℝ) (m : ℕ) (hm : m > 0) (h : 0 ≤ x) :
    |x / m - (Nat.floor x : ℝ) / m| ≤ 1 / m := by
  field_simp;
  cases abs_cases ( ( x - Nat.floor x ) / ( m : ℝ ) ) <;> nlinarith [ Nat.floor_le h, Nat.lt_floor_add_one x, mul_div_cancel₀ ( x - Nat.floor x ) ( by positivity : ( m : ℝ ) ≠ 0 ) ]

/-- Product approximation error bound. -/
lemma product_approx_error (x1 x2 y1 y2 : ℝ) (_hx1 : 0 ≤ x1) (hx2 : 0 ≤ x2) (hy1 : 0 ≤ y1) (_hy2 : 0 ≤ y2)
    (h1 : |x1 - y1| ≤ 1) (h2 : |x2 - y2| ≤ 1) :
    |x1 * x2 - y1 * y2| ≤ y1 + y2 + 1 := by
  exact abs_le.mpr ⟨ by nlinarith [ abs_le.mp h1, abs_le.mp h2 ], by nlinarith [ abs_le.mp h1, abs_le.mp h2 ] ⟩

end -- noncomputable section
