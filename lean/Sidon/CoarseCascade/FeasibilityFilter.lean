/-
Sidon Autocorrelation Project — Feasibility Pre-Filter Soundness

When transitioning from cascade level d to 2d, each parent bin with integer
mass p splits into children (c, p-c) where both c and p-c must satisfy
0 <= c <= x_cap and 0 <= p-c <= x_cap.

This means p <= 2*x_cap is NECESSARY for a valid child split to exist.
Parents with any bin p > 2*x_cap can be safely discarded.

BUG FIX (2026-04-13): Previously used p <= x_cap, which was too aggressive
and discarded valid parents. The correct bound is p <= 2*x_cap because
child splits (c, p-c) allow c up to x_cap AND p-c up to x_cap independently.

Source: run_cascade_coarse.py line 595, run_cascade_coarse_v2.py line 703.
  feasible = np.all(current <= 2 * x_cap, axis=1)
-/

import Sidon.CoarseCascade.IntegerThreshold

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- =============================================================================
-- PART 1: Child Split Feasibility
-- =============================================================================

/-- A child split (c, p-c) is feasible if both halves are in [0, x_cap]. -/
def feasible_split (p x_cap : ℕ) : Prop :=
  ∃ c : ℕ, c ≤ x_cap ∧ p - c ≤ x_cap ∧ c ≤ p

/-- **Feasibility necessary condition:** if a valid split exists, then p <= 2*x_cap.

    Proof: c <= x_cap and p - c <= x_cap (in ℕ), so p ≤ c + x_cap ≤ 2*x_cap. -/
theorem feasibility_necessary (p x_cap : ℕ) :
    feasible_split p x_cap → p ≤ 2 * x_cap := by
  rintro ⟨c, hc, hpc, hcp⟩
  -- c ≤ x_cap and p - c ≤ x_cap and c ≤ p, so p = (p - c) + c ≤ 2 * x_cap
  omega

/-- **Feasibility sufficient condition:** if p <= 2*x_cap, then a valid split exists.

    Construction: c = min(p, x_cap). Then c <= x_cap by definition.
    And p - c <= x_cap when p <= 2*x_cap. -/
theorem feasibility_sufficient (p x_cap : ℕ) (h : p ≤ 2 * x_cap) :
    feasible_split p x_cap := by
  refine ⟨min p x_cap, ?_, ?_, ?_⟩
  · exact min_le_right p x_cap
  · -- p - min(p, x_cap) ≤ x_cap iff p ≤ x_cap + min(p, x_cap)
    -- Case p ≤ x_cap: min = p, so p - p = 0 ≤ x_cap ✓
    -- Case p > x_cap: min = x_cap, so p - x_cap ≤ x_cap ⟺ p ≤ 2*x_cap ✓
    omega
  · exact min_le_left p x_cap

/-- **Feasibility iff:** p <= 2*x_cap is necessary and sufficient. -/
theorem feasibility_iff (p x_cap : ℕ) :
    feasible_split p x_cap ↔ p ≤ 2 * x_cap :=
  ⟨feasibility_necessary p x_cap, feasibility_sufficient p x_cap⟩

-- =============================================================================
-- PART 2: Pre-Filter Soundness
-- =============================================================================

/-- **Pre-filter soundness:** Discarding parents where any bin > 2*x_cap
    does not lose any parent that could produce children with all bins ≤ x_cap.

    If parent bin p_i > 2*x_cap, then no child split of bin i can have both
    halves in [0, x_cap], so any child of this parent must have at least one
    bin > x_cap.

    Source: run_cascade_coarse.py lines 593-599. -/
theorem prefilter_sound {d : ℕ} (S : ℕ) (c_target : ℝ)
    (parent : Fin d → ℕ) (_h_sum : ∑ i, parent i = S)
    (x_cap : ℕ)
    (_h_xcap_def : x_cap = Nat.floor ((S : ℝ) * Real.sqrt (c_target / (d : ℝ))))
    (i : Fin d) (h_infeasible : parent i > 2 * x_cap) :
    ¬∃ child : Fin (2 * d) → ℕ,
      (∀ j : Fin d,
        child ⟨2 * j.val, by omega⟩ + child ⟨2 * j.val + 1, by omega⟩ = parent j) ∧
      (∀ k : Fin (2 * d), child k ≤ x_cap) := by
  rintro ⟨child, h_split, h_bound⟩
  -- For bin i: child[2i] + child[2i+1] = parent_i > 2*x_cap
  -- But each child ≤ x_cap, so child[2i] + child[2i+1] ≤ 2*x_cap. Contradiction.
  have h_pair : child ⟨2 * i.val, by omega⟩ + child ⟨2 * i.val + 1, by omega⟩ = parent i :=
    h_split i
  have h1 : child ⟨2 * i.val, by omega⟩ ≤ x_cap := h_bound _
  have h2 : child ⟨2 * i.val + 1, by omega⟩ ≤ x_cap := h_bound _
  omega

-- =============================================================================
-- PART 3: Old Filter Was Unsound (Counterexample Witness)
-- =============================================================================

/-- The OLD filter `p <= x_cap` was too aggressive: it discarded parents
    that DO have valid children.

    Counterexample: x_cap = 10, parent bin p = 15.
    Split c = 8, p-c = 7: both <= 10. Valid!
    But old filter rejects since 15 > 10. -/
theorem old_filter_unsound :
    ∃ p x_cap : ℕ, p > x_cap ∧ feasible_split p x_cap := by
  refine ⟨15, 10, by norm_num, ?_⟩
  -- 15 ≤ 2 * 10 = 20, so by feasibility_sufficient, feasible.
  exact feasibility_sufficient 15 10 (by norm_num)

-- =============================================================================
-- PART 4: Cursor Range Correctness
-- =============================================================================

/-- The cursor range for parent bin p is [max(0, p - x_cap), min(p, x_cap)].
    This is non-empty iff p <= 2*x_cap (the feasibility condition).

    Source: run_cascade_coarse.py process_parent(), lines 491-497:
      lo = max(0, p - x_cap)
      hi = min(p, x_cap)
      if lo > hi: return empty -/
theorem cursor_range_correct (p x_cap : ℕ) (h : p ≤ 2 * x_cap) :
    let lo := p - min p x_cap  -- max(0, p - x_cap) in natural number arithmetic
    let hi := min p x_cap
    lo ≤ hi ∧
    (∀ c, lo ≤ c → c ≤ hi → c ≤ x_cap ∧ p - c ≤ x_cap ∧ c ≤ p) ∧
    (∀ c, c ≤ x_cap → p - c ≤ x_cap → c ≤ p → lo ≤ c ∧ c ≤ hi) := by
  simp only
  refine ⟨?_, ?_, ?_⟩
  · -- lo = p - min(p, x_cap), hi = min(p, x_cap). Show lo ≤ hi.
    -- Case p ≤ x_cap: min = p, lo = 0, hi = p. 0 ≤ p ✓
    -- Case p > x_cap: min = x_cap, lo = p - x_cap, hi = x_cap.
    --   Need p - x_cap ≤ x_cap, i.e., p ≤ 2 * x_cap ✓ by hypothesis
    omega
  · intro c hc_lo hc_hi
    refine ⟨?_, ?_, ?_⟩
    · -- c ≤ hi = min(p, x_cap) ≤ x_cap
      exact le_trans hc_hi (min_le_right _ _)
    · -- p - c ≤ x_cap
      -- From hc_lo: c ≥ p - min(p, x_cap)
      -- Case p ≤ x_cap: min = p, c ≥ 0, p - c ≤ p ≤ x_cap
      -- Case p > x_cap: min = x_cap, c ≥ p - x_cap, p - c ≤ x_cap
      omega
    · -- c ≤ p: c ≤ hi = min(p, x_cap) ≤ p
      exact le_trans hc_hi (min_le_left _ _)
  · intro c hc hpc hcp
    refine ⟨?_, ?_⟩
    · -- p - min(p, x_cap) ≤ c
      omega
    · -- c ≤ min(p, x_cap)
      exact le_min hcp hc

/-- The number of children per parent bin is (hi - lo + 1) = min(p, x_cap) - max(0, p-x_cap) + 1. -/
theorem children_count (p x_cap : ℕ) (_h : p ≤ 2 * x_cap) :
    let lo := p - min p x_cap
    let hi := min p x_cap
    hi - lo + 1 = min p x_cap - (p - min p x_cap) + 1 := by
  rfl

end -- noncomputable section
