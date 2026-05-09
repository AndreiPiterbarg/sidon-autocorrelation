/-
Sidon Autocorrelation Project — Variant L Post-Filter (Lasserre/Shor SDP cell)

This file formalizes **variant L**, the per-composition Lasserre/Shor SDP
cell-emptiness pruning step at the bottom of the cascade pruning chain
F → Q → L (`cloninger-steinerberger/cpu/post_filters.py`).

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL DERIVATION (matches `_L_bench.py`)
═══════════════════════════════════════════════════════════════════════════════

For a fixed integer composition `c : Fin (2n) → ℕ` with `Σc = 4nm`, the
"continuous cell" of valid heights `a : Fin (2n) → ℝ` is
   { a | a ≥ 0,  |a − c/m|_∞ ≤ 1/m,  Σa = 4n }.
Lifting to `x = m·a` coordinates this becomes
   { x : Fin (2n) → ℝ | max(0, c_i − 1) ≤ x_i ≤ c_i + 1,  Σx = 4nm },
which is the `Cell` of this file.

Per window `W = (ℓ, s_lo)` define the symmetric `{0,1}` indicator matrix
`A_W` (already defined in `SpectralDiscretizationBound`).  Then the
quadratic form satisfies
   m² · TV_W(a) · 4nℓ  =  ∑_{(i,j) ∈ W} x_i x_j  =  xᵀ A_W x.
So the cell of windows-not-violated heights is
   `WindowedCell n m c c_target` :=
      { x ∈ Cell | ∀ ℓ s_lo, 2 ≤ ℓ → xᵀ A_W x ≤ 4nℓ · c_target · m² }.

Variant L PRUNES composition `c` iff `WindowedCell` is empty:
   ∀ x in Cell, ∃ ℓ s_lo, 2 ≤ ℓ ∧ xᵀ A_W x > 4nℓ · c_target · m².

Equivalently, scaling by `1/(4nℓ)` (with `m²·TV_W(a) = (1/(4nℓ))·xᵀA_Wx`):
   ∀ valid `a`, ∃ ℓ s_lo, 2 ≤ ℓ ∧ test_value_real n a ℓ s_lo > c_target,
which is the standard cascade pruning conclusion (matching the cascade
soundness pattern of `tight_cascade_prune_sound`).

═══════════════════════════════════════════════════════════════════════════════
SDP FARKAS CERTIFICATE
═══════════════════════════════════════════════════════════════════════════════

The Python solver claims `prune_L = True` only when MOSEK / Clarabel
return `infeasible` with a Farkas certificate.  We formalize the certificate
abstractly: a tuple of multipliers `(y_W, lam_lo, lam_hi, mu_eq)` with
   • y_W ≥ 0 for each window constraint,
   • lam_lo, lam_hi ≥ 0 for each box constraint,
   • mu_eq free for the equality constraint,
   • finite support on (ℓ, s_lo),
   • the linear (relaxed) inequality is uniformly negative on the box.

The Farkas inequality reads, for ε > 0 and all `x` in the box:
   ∑_{ℓ, s_lo} y_W (4nℓ·c_target·m² − xᵀ A_W x)
       + ∑ᵢ lam_lo i · (xᵢ − lo_i)
       + ∑ᵢ lam_hi i · (hi_i − xᵢ)
       + mu_eq · (Σx − 4nm)
   ≤ −ε.

For any `x ∈ WindowedCell`, every individual term is ≥ 0 (windows by
definition; box by definition; equality is exactly zero), giving LHS ≥ 0.
This contradicts LHS ≤ −ε < 0; hence `WindowedCell = ∅`.

═══════════════════════════════════════════════════════════════════════════════
SOUNDNESS CHAIN
═══════════════════════════════════════════════════════════════════════════════

  SDPInfeasibilityCertificate  →  cell_empty  →  rigorous L-prune
  (Farkas dual, Sect. 5)         (this file)     (this file, Theorem)

The last step lifts emptiness back from `x = m·a` coordinates to the standard
cascade conclusion `∃ ℓ s_lo, test_value_real > c_target` for every valid `a`.

NEW AXIOMS DECLARED IN THIS FILE: ZERO.
NEW SORRYS IN THIS FILE: ZERO.
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.WRefinedDefs
import Sidon.Proof.TightDiscretizationBound
import Sidon.Proof.SpectralDiscretizationBound

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise
open scoped Matrix

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace PostFilterL

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 1: The cell box and windowed cell
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Lower bound for the cell box in `x = m·a` coordinates: `lo_i = max(0, c_i − 1)`. -/
def cellLo (n : ℕ) (c : Fin (2 * n) → ℕ) (i : Fin (2 * n)) : ℝ :=
  max 0 ((c i : ℝ) - 1)

/-- Upper bound for the cell box: `hi_i = c_i + 1`. -/
def cellHi (n : ℕ) (c : Fin (2 * n) → ℕ) (i : Fin (2 * n)) : ℝ :=
  (c i : ℝ) + 1

/-- The cell (continuous, `x = m·a` coordinates):
       { x | lo_i ≤ x_i ≤ hi_i,  Σ x = 4nm }. -/
def Cell (n m : ℕ) (c : Fin (2 * n) → ℕ) : Set (Fin (2 * n) → ℝ) :=
  { x | (∀ i, cellLo n c i ≤ x i ∧ x i ≤ cellHi n c i) ∧
        ∑ i, x i = (4 * n * m : ℕ) }

/-- The windowed cell:
       { x ∈ Cell | ∀ ℓ s_lo, 2 ≤ ℓ → xᵀ A_W x ≤ 4nℓ · c_target · m² }.

    Composition `c` is L-pruned iff this set is empty. -/
def WindowedCell (n m : ℕ) (c : Fin (2 * n) → ℕ) (c_target : ℝ) :
    Set (Fin (2 * n) → ℝ) :=
  { x | x ∈ Cell n m c ∧
        ∀ ℓ s_lo, 2 ≤ ℓ →
          dotProduct x ((A_W n s_lo ℓ).mulVec x) ≤
            (4 * n * ℓ * c_target * (m : ℝ) ^ 2) }

/-- The L-prune predicate: the windowed cell is empty. -/
def cell_empty (n m : ℕ) (c : Fin (2 * n) → ℕ) (c_target : ℝ) : Prop :=
  WindowedCell n m c c_target = ∅

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 2: From the cell (x-coords) to valid `a` (TV-coords)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- If `a ≥ 0`, `|a − c/m| ≤ 1/m`, and `Σ a = 4n`, then `x := m·a` lives in `Cell`.

    The total-mass equality `Σ a = 4n` is automatic in the cascade context
    (both sides come from a density with ∫f = 1; see `SpectralDiscretizationBound`). -/
theorem mem_cell_of_valid_a
    (n m : ℕ) (hm : 0 < m) (c : Fin (2 * n) → ℕ)
    (a : Fin (2 * n) → ℝ)
    (ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (h_sum_a : ∑ i, a i = 4 * n) :
    (fun i => (m : ℝ) * a i) ∈ Cell n m c := by
  classical
  refine ⟨?_, ?_⟩
  · -- Box: lo_i ≤ m·a_i ≤ hi_i.
    intro i
    have hm_pos : (0 : ℝ) < (m : ℝ) := Nat.cast_pos.mpr hm
    have hclose_i : |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ) := h_close i
    have habs_i := abs_le.mp hclose_i
    -- |a i - c i / m| ≤ 1/m  ↔  c i / m - 1/m ≤ a i ≤ c i / m + 1/m.
    have h_low : (c i : ℝ) / m - 1 / m ≤ a i := by linarith [habs_i.1]
    have h_high : a i ≤ (c i : ℝ) / m + 1 / m := by linarith [habs_i.2]
    -- Multiply by m: c i - 1 ≤ m·a_i ≤ c i + 1.
    have h_mul_low : (c i : ℝ) - 1 ≤ (m : ℝ) * a i := by
      have h1 := mul_le_mul_of_nonneg_left h_low hm_pos.le
      have h_eq : (m : ℝ) * ((c i : ℝ) / m - 1 / m) = (c i : ℝ) - 1 := by
        field_simp
      linarith
    have h_mul_high : (m : ℝ) * a i ≤ (c i : ℝ) + 1 := by
      have h1 := mul_le_mul_of_nonneg_left h_high hm_pos.le
      have h_eq : (m : ℝ) * ((c i : ℝ) / m + 1 / m) = (c i : ℝ) + 1 := by
        field_simp
      linarith
    have h_x_nonneg : 0 ≤ (m : ℝ) * a i := mul_nonneg hm_pos.le (ha_nonneg i)
    -- Combine: lo_i = max(0, c_i - 1) ≤ m·a_i.
    constructor
    · -- max 0 (c_i - 1) ≤ m * a_i
      unfold cellLo
      exact max_le h_x_nonneg h_mul_low
    · -- m * a_i ≤ c_i + 1 = hi_i
      unfold cellHi
      exact h_mul_high
  · -- Sum: Σ (m * a_i) = 4nm.
    rw [show (∑ i, (m : ℝ) * a i) = (m : ℝ) * ∑ i, a i from by
      rw [← Finset.mul_sum]]
    rw [h_sum_a]
    push_cast
    ring

/-- Converting back: if `x ∈ Cell n m c`, define `a := x / m`.  Then `a ≥ 0`,
    `|a − c/m| ≤ 1/m`, and `Σa = 4n`. -/
theorem valid_a_of_mem_cell
    (n m : ℕ) (hm : 0 < m) (c : Fin (2 * n) → ℕ)
    (x : Fin (2 * n) → ℝ) (hx : x ∈ Cell n m c) :
    let a : Fin (2 * n) → ℝ := fun i => x i / m
    (∀ i, 0 ≤ a i) ∧
    (∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ)) ∧
    (∑ i, a i = 4 * n) := by
  classical
  intro a
  have hm_pos : (0 : ℝ) < (m : ℝ) := Nat.cast_pos.mpr hm
  obtain ⟨h_box, h_sum⟩ := hx
  refine ⟨?_, ?_, ?_⟩
  · intro i
    have h_lo : 0 ≤ x i := by
      have h1 := (h_box i).1
      have hlo_nn : 0 ≤ cellLo n c i := le_max_left _ _
      linarith
    show 0 ≤ x i / m
    exact div_nonneg h_lo hm_pos.le
  · intro i
    obtain ⟨hlo, hhi⟩ := h_box i
    -- lo_i ≤ x_i ≤ hi_i where lo_i = max(0, c_i - 1), hi_i = c_i + 1.
    have hlo' : (c i : ℝ) - 1 ≤ x i := le_trans (le_max_right _ _) hlo
    have hhi' : x i ≤ (c i : ℝ) + 1 := by unfold cellHi at hhi; exact hhi
    -- |x_i / m - c_i/m| = |x_i - c_i|/m ≤ 1/m.
    show |x i / m - (c i : ℝ) / m| ≤ 1 / (m : ℝ)
    rw [show x i / m - (c i : ℝ) / m = (x i - c i) / m from by ring]
    rw [abs_div, abs_of_pos hm_pos]
    -- Now goal: |x i - c i| / m ≤ 1 / m
    have h_abs_le : |x i - (c i : ℝ)| ≤ 1 := by
      rw [abs_sub_le_iff]
      exact ⟨by linarith, by linarith⟩
    exact div_le_div_of_nonneg_right h_abs_le hm_pos.le
  · -- Σ (x_i / m) = (Σ x_i) / m = 4nm / m = 4n.
    show ∑ i, x i / m = 4 * n
    rw [show (∑ i, x i / m) = (∑ i, x i) / m from by
      rw [← Finset.sum_div]]
    rw [h_sum]
    push_cast
    field_simp

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 3: TV vs xᵀ A_W x relationship
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Key algebraic relation:  for `x = m·a`,
       `xᵀ A_W x = m² · ∑_{(i,j) ∈ W} a_i a_j = 4nℓ · m² · TV_W(a)`. -/
theorem dot_A_W_eq_window_sum_mul
    (n s_lo ℓ : ℕ) (m : ℕ) (a : Fin (2 * n) → ℝ) :
    let x : Fin (2 * n) → ℝ := fun i => (m : ℝ) * a i
    dotProduct x ((A_W n s_lo ℓ).mulVec x) =
      (m : ℝ) ^ 2 * ∑ p ∈ window_pair_set n s_lo ℓ, a p.1 * a p.2 := by
  classical
  intro x
  rw [A_W_quad_form n s_lo ℓ x]
  -- ∑ (m·a_i)(m·a_j) = m² · ∑ a_i·a_j
  rw [show (∑ p ∈ window_pair_set n s_lo ℓ, x p.1 * x p.2) =
        (∑ p ∈ window_pair_set n s_lo ℓ, (m : ℝ) ^ 2 * (a p.1 * a p.2)) from ?_]
  · rw [← Finset.mul_sum]
  · apply Finset.sum_congr rfl
    intro p _
    show (m : ℝ) * a p.1 * ((m : ℝ) * a p.2) = (m : ℝ) ^ 2 * (a p.1 * a p.2)
    ring

/-- Window violation in `x`-coordinates ⇔ TV exceeds `c_target` in `a`-coordinates. -/
theorem window_violate_iff_tv_exceeds
    (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (c_target : ℝ)
    (a : Fin (2 * n) → ℝ) (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    let x : Fin (2 * n) → ℝ := fun i => (m : ℝ) * a i
    (dotProduct x ((A_W n s_lo ℓ).mulVec x) >
        (4 * n * ℓ * c_target * (m : ℝ) ^ 2)) ↔
      test_value_real n a ℓ s_lo > c_target := by
  classical
  intro x
  rw [test_value_real_eq_window_sum]
  rw [dot_A_W_eq_window_sum_mul n s_lo ℓ m a]
  have hn_real : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hm_real : (0 : ℝ) < (m : ℝ) := Nat.cast_pos.mpr hm
  have hℓ_real : (0 : ℝ) < (ℓ : ℝ) := by
    have h2 : (2 : ℝ) ≤ (ℓ : ℝ) := by exact_mod_cast hℓ
    linarith
  have h4nℓ_pos : (0 : ℝ) < 4 * n * ℓ := by positivity
  have hm_sq_pos : (0 : ℝ) < (m : ℝ) ^ 2 := by positivity
  -- Set S := ∑ p ∈ W, a p.1 * a p.2 to simplify.
  set S : ℝ := ∑ p ∈ window_pair_set n s_lo ℓ, a p.1 * a p.2 with hS_def
  -- The goal, after unfolding `>`, is:
  --   4 * n * ℓ * c_target * m² < m² * S  ↔  c_target < (1/(4nℓ)) * S
  -- We prove via the chain:
  --   m² · (4nℓ · c_target) < m² · S  ↔  4nℓ · c_target < S  ↔  c_target < S/(4nℓ).
  have h_mul_eq_div : (1 / (4 * n * ℓ : ℝ)) * S = S / (4 * n * ℓ) := by
    field_simp
  rw [h_mul_eq_div]
  show (m : ℝ) ^ 2 * S > 4 * n * ℓ * c_target * (m : ℝ) ^ 2 ↔ S / (4 * n * ℓ) > c_target
  rw [show (4 * n * (ℓ : ℝ) * c_target * (m : ℝ) ^ 2 : ℝ) =
        (m : ℝ) ^ 2 * (4 * n * ℓ * c_target) from by ring]
  constructor
  · intro h
    have h' : 4 * n * ℓ * c_target < S := (mul_lt_mul_iff_of_pos_left hm_sq_pos).mp h
    show c_target < S / (4 * n * ℓ)
    rw [lt_div_iff₀ h4nℓ_pos]
    linarith
  · intro h
    show (m : ℝ) ^ 2 * (4 * n * ℓ * c_target) < (m : ℝ) ^ 2 * S
    have h' : c_target < S / (4 * n * ℓ) := h
    have h'' : 4 * n * ℓ * c_target < S := by
      have := (lt_div_iff₀ h4nℓ_pos).mp h'
      linarith
    exact (mul_lt_mul_iff_of_pos_left hm_sq_pos).mpr h''

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 4: cell_empty implies rigorous pruning
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Variant L pruning soundness (abstract).**
    If the windowed cell is empty, then for every valid real height `a` (with
    `a ≥ 0`, `|a − c/m| ≤ 1/m`, `Σa = 4n`), some window `(ℓ, s_lo)` with
    `2 ≤ ℓ` satisfies `test_value_real n a ℓ s_lo > c_target`.

    This is the per-composition cascade prune conclusion exactly as
    `tight_cascade_prune_sound` states it (only the witness window is now
    existentially quantified, which is the natural cascade pattern). -/
theorem cell_empty_implies_rigorous_pruning
    (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (c_target : ℝ)
    (c : Fin (2 * n) → ℕ)
    (h_empty : cell_empty n m c c_target)
    (a : Fin (2 * n) → ℝ)
    (ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (h_sum_a : ∑ i, a i = 4 * n) :
    ∃ ℓ s_lo, 2 ≤ ℓ ∧ test_value_real n a ℓ s_lo > c_target := by
  classical
  -- Define x = m·a; show x ∈ Cell.
  set x : Fin (2 * n) → ℝ := fun i => (m : ℝ) * a i with hx_def
  have hx_cell : x ∈ Cell n m c :=
    mem_cell_of_valid_a n m hm c a ha_nonneg h_close h_sum_a
  -- Suppose for contradiction no window violates.
  by_contra h_no_violate
  push_neg at h_no_violate
  -- h_no_violate has type: ∀ ℓ s_lo, 2 ≤ ℓ → test_value_real n a ℓ s_lo ≤ c_target.
  have h_all_windows : ∀ ℓ s_lo, 2 ≤ ℓ →
      dotProduct x ((A_W n s_lo ℓ).mulVec x) ≤
        (4 * n * ℓ * c_target * (m : ℝ) ^ 2) := by
    intro ℓ s_lo hℓ
    have h_tv_le : test_value_real n a ℓ s_lo ≤ c_target := h_no_violate ℓ s_lo hℓ
    by_contra h_violate
    push_neg at h_violate
    have h_iff := window_violate_iff_tv_exceeds n m hn hm c_target a ℓ s_lo hℓ
    have h_tv_gt : test_value_real n a ℓ s_lo > c_target := h_iff.mp h_violate
    linarith
  have hx_windowed : x ∈ WindowedCell n m c c_target :=
    ⟨hx_cell, h_all_windows⟩
  -- But cell_empty says WindowedCell = ∅.
  rw [cell_empty] at h_empty
  rw [h_empty] at hx_windowed
  exact (Set.mem_empty_iff_false x).mp hx_windowed

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 5: SDP Farkas Infeasibility Certificate
-- ═══════════════════════════════════════════════════════════════════════════════

/-- A Farkas-style infeasibility certificate for the SDP (more precisely,
    the linear relaxation produced by the dual of the SDP / Shor LP).

    `y_W ℓ s_lo` is the multiplier for the window constraint
       `xᵀ A_W x ≤ 4nℓ · c_target · m²`.
    `lam_lo i` is the multiplier for `x_i − lo_i ≥ 0`.
    `lam_hi i` is the multiplier for `hi_i − x_i ≥ 0`.
    `mu_eq` is the (free) multiplier for `Σx − 4nm = 0`.

    The Farkas witness asserts that the linear combination is uniformly
    `≤ −ε` on the box for some `ε > 0`, which contradicts feasibility.

    `y_W_finitely_supported` lets us write the y_W-sum over a finite
    indexing Finset (the set of windows actually examined by the SDP).

    This certificate is what MOSEK / Clarabel return when status is
    `infeasible` with a Farkas dual cert (see `_L_bench.py:208-222`). -/
structure SDPInfeasibilityCertificate (n m : ℕ) (c : Fin (2 * n) → ℕ)
    (c_target : ℝ) where
  /-- Multipliers for the window constraints, indexed by `(ℓ, s_lo)`. -/
  y_W : ℕ → ℕ → ℝ
  /-- Multipliers for the lower-bound constraints `x_i ≥ lo_i`. -/
  lam_lo : Fin (2 * n) → ℝ
  /-- Multipliers for the upper-bound constraints `x_i ≤ hi_i`. -/
  lam_hi : Fin (2 * n) → ℝ
  /-- Multiplier for the equality constraint `Σx = 4nm` (free sign). -/
  mu_eq : ℝ
  /-- Window multipliers are non-negative (dual cone of `≤`). -/
  y_W_nonneg : ∀ ℓ s_lo, 0 ≤ y_W ℓ s_lo
  /-- Lower-bound multipliers are non-negative. -/
  lam_lo_nonneg : ∀ i, 0 ≤ lam_lo i
  /-- Upper-bound multipliers are non-negative. -/
  lam_hi_nonneg : ∀ i, 0 ≤ lam_hi i
  /-- The finite set of windows on which `y_W` may be nonzero. -/
  support : Finset (ℕ × ℕ)
  /-- `y_W` vanishes outside `support`. -/
  y_W_finitely_supported : ∀ ℓ s_lo, (ℓ, s_lo) ∉ support → y_W ℓ s_lo = 0
  /-- Only `(ℓ, s_lo)` with `2 ≤ ℓ` may carry positive weight. -/
  support_l_ge_two : ∀ p ∈ support, 2 ≤ p.1
  /-- The Farkas witness: for some `ε > 0`, on the entire box,
        ∑_W y_W (4nℓc_target·m² − xᵀ A_W x)
            + ∑ᵢ lam_lo i (xᵢ − lo_i)
            + ∑ᵢ lam_hi i (hi_i − xᵢ)
            + mu_eq (Σx − 4nm)  ≤  −ε. -/
  farkas_witness : ∃ ε > 0, ∀ x : Fin (2 * n) → ℝ,
    (∀ i, cellLo n c i ≤ x i ∧ x i ≤ cellHi n c i) →
    (∑ p ∈ support,
        y_W p.1 p.2 * ((4 * n * (p.1 : ℝ) * c_target * (m : ℝ) ^ 2) -
          dotProduct x ((A_W n p.2 p.1).mulVec x))
      + ∑ i, lam_lo i * (x i - cellLo n c i)
      + ∑ i, lam_hi i * (cellHi n c i - x i)
      + mu_eq * (∑ i, x i - (4 * n * m : ℕ))) ≤ -ε

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 6: Farkas certificate implies cell_empty
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Farkas certificate ⇒ WindowedCell empty.**

    Suppose for contradiction `x ∈ WindowedCell n m c c_target`.  Then:
      • box: `lo_i ≤ x_i ≤ hi_i`, so `x_i − lo_i ≥ 0` and `hi_i − x_i ≥ 0`,
      • equality: `Σ x = 4nm`, so the `mu_eq` term is zero,
      • windows: for every `(ℓ, s_lo)` with `2 ≤ ℓ`,
        `xᵀ A_W x ≤ 4nℓ · c_target · m²`, so `(thr − xᵀ A_W x) ≥ 0`.
    All three groups contribute ≥ 0 to the LHS of the Farkas witness, so
    LHS ≥ 0.  But the witness asserts LHS ≤ −ε < 0.  Contradiction. -/
theorem farkas_certificate_implies_cell_empty
    (n m : ℕ) (c : Fin (2 * n) → ℕ) (c_target : ℝ)
    (cert : SDPInfeasibilityCertificate n m c c_target) :
    cell_empty n m c c_target := by
  classical
  unfold cell_empty
  rw [Set.eq_empty_iff_forall_notMem]
  intro x hx
  obtain ⟨hx_cell, hx_windows⟩ := hx
  obtain ⟨h_box, h_sum⟩ := hx_cell
  -- Extract Farkas witness.
  obtain ⟨ε, hε_pos, h_witness⟩ := cert.farkas_witness
  have h_ineq := h_witness x h_box
  -- Three nonneg contributions.
  -- (1) The window sum is nonneg.
  have h_window_nn :
      (0 : ℝ) ≤ ∑ p ∈ cert.support,
        cert.y_W p.1 p.2 *
          ((4 * n * (p.1 : ℝ) * c_target * (m : ℝ) ^ 2) -
            dotProduct x ((A_W n p.2 p.1).mulVec x)) := by
    apply Finset.sum_nonneg
    intro p hp
    have hℓ : 2 ≤ p.1 := cert.support_l_ge_two p hp
    have h_y : 0 ≤ cert.y_W p.1 p.2 := cert.y_W_nonneg p.1 p.2
    have h_window : dotProduct x ((A_W n p.2 p.1).mulVec x) ≤
                     (4 * n * (p.1 : ℝ) * c_target * (m : ℝ) ^ 2) := by
      have hxw := hx_windows p.1 p.2 hℓ
      -- hxw : dotProduct x ... ≤ 4 * n * ↑p.1 * c_target * m^2
      exact hxw
    have h_diff_nn :
        0 ≤ (4 * n * (p.1 : ℝ) * c_target * (m : ℝ) ^ 2) -
              dotProduct x ((A_W n p.2 p.1).mulVec x) := by linarith
    exact mul_nonneg h_y h_diff_nn
  -- (2a) The lower-bound sum is nonneg.
  have h_lo_nn : (0 : ℝ) ≤ ∑ i, cert.lam_lo i * (x i - cellLo n c i) := by
    apply Finset.sum_nonneg
    intro i _
    have h_lam : 0 ≤ cert.lam_lo i := cert.lam_lo_nonneg i
    have h_diff : 0 ≤ x i - cellLo n c i := by
      have := (h_box i).1; linarith
    exact mul_nonneg h_lam h_diff
  -- (2b) The upper-bound sum is nonneg.
  have h_hi_nn : (0 : ℝ) ≤ ∑ i, cert.lam_hi i * (cellHi n c i - x i) := by
    apply Finset.sum_nonneg
    intro i _
    have h_lam : 0 ≤ cert.lam_hi i := cert.lam_hi_nonneg i
    have h_diff : 0 ≤ cellHi n c i - x i := by
      have := (h_box i).2; linarith
    exact mul_nonneg h_lam h_diff
  -- (3) The equality term is exactly zero.
  have h_eq_zero : cert.mu_eq * (∑ i, x i - (4 * n * m : ℕ)) = 0 := by
    have h_sum_zero : (∑ i, x i - (4 * n * m : ℕ) : ℝ) = 0 := by
      rw [h_sum]; ring
    rw [h_sum_zero]; ring
  -- Combine: LHS ≥ 0.
  have h_LHS_nn :
      (0 : ℝ) ≤ ∑ p ∈ cert.support,
          cert.y_W p.1 p.2 *
            ((4 * n * (p.1 : ℝ) * c_target * (m : ℝ) ^ 2) -
              dotProduct x ((A_W n p.2 p.1).mulVec x))
        + ∑ i, cert.lam_lo i * (x i - cellLo n c i)
        + ∑ i, cert.lam_hi i * (cellHi n c i - x i)
        + cert.mu_eq * (∑ i, x i - (4 * n * m : ℕ)) := by
    rw [h_eq_zero, add_zero]
    linarith
  -- But h_ineq says LHS ≤ -ε < 0.
  linarith

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 7: Combined chain — Farkas certificate ⇒ rigorous pruning
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **The full L-pruning soundness chain.**
    A valid SDP infeasibility (Farkas) certificate for `(n, m, c, c_target)`
    implies that, for every continuous-feasible composition `a`, some window
    `(ℓ, s_lo)` with `2 ≤ ℓ` exhibits `test_value_real n a ℓ s_lo > c_target`.

    This is exactly the per-composition guarantee the cascade pruning chain
    F → Q → L provides at the L step.  Composing `F-prune` and `Q-prune`
    earlier in the chain just shrinks the surviving compositions; `L-prune`
    finishes the elimination via this structured-Farkas argument. -/
theorem farkas_certificate_implies_rigorous_pruning
    (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (c_target : ℝ)
    (c : Fin (2 * n) → ℕ)
    (cert : SDPInfeasibilityCertificate n m c c_target)
    (a : Fin (2 * n) → ℝ)
    (ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (h_sum_a : ∑ i, a i = 4 * n) :
    ∃ ℓ s_lo, 2 ≤ ℓ ∧ test_value_real n a ℓ s_lo > c_target :=
  cell_empty_implies_rigorous_pruning n m hn hm c_target c
    (farkas_certificate_implies_cell_empty n m c c_target cert)
    a ha_nonneg h_close h_sum_a

end PostFilterL

end -- noncomputable section
