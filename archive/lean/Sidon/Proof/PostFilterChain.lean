/-
Sidon Autocorrelation Project — Post-Filter Chain F ⊇ Q ⊇ L

This file synthesises the per-composition pruning chain F → Q → L
(`cloninger-steinerberger/cpu/post_filters.py`) as a sequence of subset
theorems on the prune sets.  Equivalently, on survivor sets:
   F-survivors ⊇ Q-survivors ⊇ L-survivors
i.e. each filter is at least as restrictive as the previous.

═══════════════════════════════════════════════════════════════════════════════
THE CHAIN
═══════════════════════════════════════════════════════════════════════════════

We package three pruning predicates:

  • `f_pruned n m c c_target` — variant F prunes c iff there exists a
    window (ℓ, s_lo) with 2 ≤ ℓ such that `test_value > c_target +
    correction_F`.  This is the LP-tight Δ_BB-based correction.

  • `q_pruned n m c c_target` — variant Q prunes c iff there exist a
    finite set `S` of windows (each with ℓ ≥ 2) and a probability
    distribution `λ` on `S` such that the convex combination of test
    values exceeds `c_target + corr_Q_m2(λ)/m²`.

  • `l_pruned n m c c_target` — variant L prunes c iff the SDP
    "windowed cell" is empty: no continuous-feasible `a` simultaneously
    satisfies all window constraints `test_value_real ≤ c_target`.
    (The SDP Farkas certificate produces this; see `PostFilterL`.)

The chain is captured by three theorems:

  1. `f_pruned_implies_q_pruned` — F ⊆ Q (Dirac λ at the F-witness window
     recovers F's bound).
  2. `q_pruned_implies_l_pruned` — Q ⊆ L (Q-prune is a sound LP relaxation
     of cell-emptiness; if Q-prunes, the cell ∩ window-constraints is
     empty).
  3. `l_pruned_implies_rigorous_pruning` — L is sound: cell empty implies
     for every valid `a`, some window's test value exceeds `c_target`.

The full chain `F ⇒ Q ⇒ L ⇒ rigorous` is composed from these.

═══════════════════════════════════════════════════════════════════════════════
SOUNDNESS — ALL THREE VARIANTS PRODUCE THE SAME CASCADE CONCLUSION
═══════════════════════════════════════════════════════════════════════════════

The bottom of the chain is the rigorous-pruning conclusion shared by all
three variants:
   ∃ ℓ s_lo, 2 ≤ ℓ ∧ test_value_real n a ℓ s_lo > c_target.

Each `*_pruned_implies_rigorous_pruning` theorem yields this, given the
standard cascade hypotheses on `a` (`a ≥ 0`, `|a − c/m| ≤ 1/m`, and
`Σ a = 4n`) and the cascade simplex `Σ c = 4nm`.

═══════════════════════════════════════════════════════════════════════════════
PYTHON CROSS-REFERENCE
═══════════════════════════════════════════════════════════════════════════════
  - `cloninger-steinerberger/cpu/post_filters.py:apply_post_filter_chain`
     (run F → Q → L in sequence on survivors).
  - `_M1_bench.py:prune_F`, `_Q_bench.py:prune_Q_one`, `_L_bench.py:prune_L_one`.

NEW AXIOMS DECLARED IN THIS FILE: ZERO.
NEW SORRYS IN THIS FILE: ZERO.
═══════════════════════════════════════════════════════════════════════════════
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.WRefinedDefs
import Sidon.Proof.TightDiscretizationBound
import Sidon.Proof.SpectralDiscretizationBound
import Sidon.Proof.PostFilterF
import Sidon.Proof.PostFilterQ
import Sidon.Proof.PostFilterL

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

namespace PostFilterChain

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 1: Pruning predicates
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Variant F prunes composition `c` iff some window `(ℓ, s_lo)` with `2 ≤ ℓ`
    has `test_value > c_target + correction_F`. -/
def f_pruned (n m : ℕ) (c : Fin (2 * n) → ℕ) (c_target : ℝ) : Prop :=
  ∃ ℓ s_lo, 2 ≤ ℓ ∧ test_value n m c ℓ s_lo > c_target + correction_F n m c ℓ s_lo

/-- Variant Q prunes composition `c` iff there is a finite multi-window
    distribution `(S, λ)` with `λ ≥ 0`, `Σλ = 1`, all windows in `S` have
    `2 ≤ ℓ`, and the weighted test value exceeds `c_target + corr_Q_m2/m²`. -/
def q_pruned (n m : ℕ) (c : Fin (2 * n) → ℕ) (c_target : ℝ) : Prop :=
  ∃ (S : Finset (ℕ × ℕ)) (lam : ℕ × ℕ → ℝ),
    (∀ W ∈ S, 0 ≤ lam W) ∧
    (∑ W ∈ S, lam W = 1) ∧
    (∀ W ∈ S, 2 ≤ W.1) ∧
    weighted_test_value n m c S lam > c_target + corr_Q_m2 n c S lam / (m : ℝ) ^ 2

/-- Variant L prunes composition `c` iff the windowed cell is empty
    (the SDP-cell formulation; see `PostFilterL.cell_empty`).

    The SDP returns a Farkas certificate (`SDPInfeasibilityCertificate`) that
    proves this emptiness; here we package L-pruning as the abstract emptiness
    predicate, and `farkas_certificate_implies_l_pruned` below shows the
    certificate produces it. -/
def l_pruned (n m : ℕ) (c : Fin (2 * n) → ℕ) (c_target : ℝ) : Prop :=
  PostFilterL.cell_empty n m c c_target

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 2: F ⊆ Q
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **F-pruned ⇒ Q-pruned** (subset chain step 1).

    Setting `λ` to the Dirac at the F-witness window `W₀` recovers F's
    per-window bound exactly (proven in `f_prune_implies_q_prune`).  Hence
    every F-pruned composition is Q-pruned.

    The cascade always has `0 < n` and `0 < m`; under these mild assumptions,
    every F-pruned composition is Q-pruned. -/
theorem f_pruned_implies_q_pruned
    (n m : ℕ) (hn : 0 < n) (hm : 0 < m)
    (c : Fin (2 * n) → ℕ) (c_target : ℝ)
    (hF : f_pruned n m c c_target) :
    q_pruned n m c c_target := by
  classical
  obtain ⟨ℓ, s_lo, hℓ, h_F_excess⟩ := hF
  set W₀ : ℕ × ℕ := (ℓ, s_lo) with hW₀_def
  refine ⟨{W₀}, diracLambda W₀, ?_, ?_, ?_, ?_⟩
  · -- λ ≥ 0 on S = {W₀}.
    intro W _
    exact diracLambda_nonneg W₀ W
  · -- Σ λ = 1.
    exact diracLambda_sum W₀
  · -- All windows in S have 2 ≤ ℓ.
    intro W hW
    rw [Finset.mem_singleton] at hW
    rw [hW]
    exact hℓ
  · -- Q-pruning excess at the Dirac λ.  This is exactly
    -- f_prune_implies_q_prune from PostFilterQ.
    have hℓ_W₀ : 2 ≤ W₀.1 := hℓ
    exact f_prune_implies_q_prune n m c_target hn hm c W₀ hℓ_W₀ h_F_excess

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 3: Q ⊆ L (Q-pruned implies the cell is empty)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Q-pruned ⇒ L-pruned** (subset chain step 2).

    Q-pruning means: for every valid `a` (which corresponds to `x = m·a` in
    the cell), some window's test value exceeds `c_target`.  But for any
    `x ∈ WindowedCell`, ALL windows satisfy `test_value_real ≤ c_target`
    (by definition of `WindowedCell`).  Hence no `x ∈ WindowedCell` can
    correspond to a valid `a`.  Every `x ∈ Cell` corresponds to a valid `a`
    (via `valid_a_of_mem_cell`).  Hence `WindowedCell = ∅`, i.e., L-pruned.

    The cascade simplex `Σ c = 4nm` is always satisfied for compositions
    enumerated by `compositions.py`.  Under this assumption (and positivity),
    Q-pruned compositions are L-pruned. -/
theorem q_pruned_implies_l_pruned
    (n m : ℕ) (hn : 0 < n) (hm : 0 < m)
    (c : Fin (2 * n) → ℕ)
    (h_simplex : ∑ i, (c i : ℕ) = 4 * n * m)
    (c_target : ℝ)
    (hQ : q_pruned n m c c_target) :
    l_pruned n m c c_target := by
  classical
  unfold l_pruned PostFilterL.cell_empty
  rw [Set.eq_empty_iff_forall_notMem]
  intro x hx
  obtain ⟨hx_cell, hx_windows⟩ := hx
  obtain ⟨ha_nonneg, h_close, h_sum_a⟩ :=
    PostFilterL.valid_a_of_mem_cell n m hm c x hx_cell
  set a : Fin (2 * n) → ℝ := fun i => x i / m with ha_def
  -- Σ a = Σ c/m follows from Σ c = 4nm and Σ a = 4n.
  have h_sum_eq : ∑ i, a i = ∑ i, (c i : ℝ) / m := by
    rw [show (∑ i, (c i : ℝ) / m) = (∑ i, (c i : ℝ)) / m from by
      rw [← Finset.sum_div]]
    rw [show (∑ i, (c i : ℝ)) = ((∑ i, c i : ℕ) : ℝ) from by push_cast; rfl]
    rw [h_simplex]
    rw [h_sum_a]
    push_cast
    have hm_pos : (0 : ℝ) < (m : ℝ) := Nat.cast_pos.mpr hm
    have hm_ne : (m : ℝ) ≠ 0 := ne_of_gt hm_pos
    field_simp
  -- Apply q_prune_sound to get a violating window.
  obtain ⟨S, lam, h_lam_nn, h_lam_sum, h_ℓ_pos, h_excess⟩ := hQ
  have h_violate :=
    q_prune_sound n m c_target hn hm c S lam h_lam_nn h_lam_sum h_ℓ_pos
      h_excess a ha_nonneg h_close h_sum_eq
  obtain ⟨W, hW_S, hW_violate⟩ := h_violate
  have hℓ_W : 2 ≤ W.1 := h_ℓ_pos W hW_S
  -- But hx_windows says every window with 2 ≤ ℓ has dotProduct ≤ thr.
  have hx_W : dotProduct x ((A_W n W.2 W.1).mulVec x) ≤
              (4 * n * (W.1 : ℝ) * c_target * (m : ℝ) ^ 2) := by
    exact hx_windows W.1 W.2 hℓ_W
  -- Translate via x = m · a.
  have hx_eq : x = (fun i => (m : ℝ) * a i) := by
    funext i
    show x i = (m : ℝ) * (x i / m)
    have hm_pos : (0 : ℝ) < (m : ℝ) := Nat.cast_pos.mpr hm
    have hm_ne : (m : ℝ) ≠ 0 := ne_of_gt hm_pos
    field_simp
  have h_iff :=
    PostFilterL.window_violate_iff_tv_exceeds n m hn hm c_target a W.1 W.2 hℓ_W
  have h_dot_violate :
      dotProduct (fun i => (m : ℝ) * a i) ((A_W n W.2 W.1).mulVec
        (fun i => (m : ℝ) * a i)) >
        (4 * n * (W.1 : ℝ) * c_target * (m : ℝ) ^ 2) :=
    h_iff.mpr hW_violate
  rw [← hx_eq] at h_dot_violate
  linarith [h_dot_violate, hx_W]

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 4: L-pruned ⇒ rigorous pruning (composing with PostFilterL)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **L-pruned ⇒ rigorous pruning.**

    If `c` is L-pruned (windowed cell empty), then every valid `a` exhibits
    some window with test value exceeding `c_target`.

    This is just `cell_empty_implies_rigorous_pruning` repackaged through
    the chain's predicate. -/
theorem l_pruned_implies_rigorous_pruning
    (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (c_target : ℝ)
    (c : Fin (2 * n) → ℕ)
    (hL : l_pruned n m c c_target)
    (a : Fin (2 * n) → ℝ)
    (ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (h_sum_a : ∑ i, a i = 4 * n) :
    ∃ ℓ s_lo, 2 ≤ ℓ ∧ test_value_real n a ℓ s_lo > c_target :=
  PostFilterL.cell_empty_implies_rigorous_pruning n m hn hm c_target c hL
    a ha_nonneg h_close h_sum_a

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 5: Composed chain F → Q → L → rigorous
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Full chain F ⇒ rigorous (via Q ⇒ L).**
    Composes `f_pruned_implies_q_pruned`, `q_pruned_implies_l_pruned`,
    and `l_pruned_implies_rigorous_pruning`. -/
theorem f_pruned_implies_rigorous_pruning
    (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (c_target : ℝ)
    (c : Fin (2 * n) → ℕ)
    (h_simplex : ∑ i, (c i : ℕ) = 4 * n * m)
    (hF : f_pruned n m c c_target)
    (a : Fin (2 * n) → ℝ)
    (ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (h_sum_a : ∑ i, a i = 4 * n) :
    ∃ ℓ s_lo, 2 ≤ ℓ ∧ test_value_real n a ℓ s_lo > c_target :=
  l_pruned_implies_rigorous_pruning n m hn hm c_target c
    (q_pruned_implies_l_pruned n m hn hm c h_simplex c_target
      (f_pruned_implies_q_pruned n m hn hm c c_target hF))
    a ha_nonneg h_close h_sum_a

/-- **Q ⇒ rigorous (direct, via cell-empty).** -/
theorem q_pruned_implies_rigorous_pruning
    (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (c_target : ℝ)
    (c : Fin (2 * n) → ℕ)
    (h_simplex : ∑ i, (c i : ℕ) = 4 * n * m)
    (hQ : q_pruned n m c c_target)
    (a : Fin (2 * n) → ℝ)
    (ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (h_sum_a : ∑ i, a i = 4 * n) :
    ∃ ℓ s_lo, 2 ≤ ℓ ∧ test_value_real n a ℓ s_lo > c_target :=
  l_pruned_implies_rigorous_pruning n m hn hm c_target c
    (q_pruned_implies_l_pruned n m hn hm c h_simplex c_target hQ)
    a ha_nonneg h_close h_sum_a

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 6: SDP Farkas certificate ⇒ chain
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Farkas SDP certificate ⇒ L-pruned.**
    A valid SDP infeasibility certificate (`PostFilterL.SDPInfeasibilityCertificate`)
    immediately produces L-pruning (the cell is empty).  This is what the Python
    pipeline (`_L_bench.py:prune_L_one`) effectively returns when MOSEK / Clarabel
    yield status `infeasible` with a Farkas dual cert. -/
theorem farkas_certificate_implies_l_pruned
    (n m : ℕ) (c : Fin (2 * n) → ℕ) (c_target : ℝ)
    (cert : PostFilterL.SDPInfeasibilityCertificate n m c c_target) :
    l_pruned n m c c_target :=
  PostFilterL.farkas_certificate_implies_cell_empty n m c c_target cert

end PostFilterChain

end -- noncomputable section

-- ═══════════════════════════════════════════════════════════════════════════════
-- AUDIT BLOCK (chain F ⊇ Q ⊇ L)
-- ═══════════════════════════════════════════════════════════════════════════════
/-
SUBSET CHAIN ON PRUNE SETS:
    F-pruned  ⊆  Q-pruned  ⊆  L-pruned     (more aggressive pruning at each step).
Equivalently on survivor sets:
    F-survivors  ⊇  Q-survivors  ⊇  L-survivors.

THEOREMS ESTABLISHED:
  • `f_pruned_implies_q_pruned`         (F ⊆ Q via Dirac λ).
  • `q_pruned_implies_l_pruned`         (Q ⊆ L via Q ⇒ cell-empty).
  • `l_pruned_implies_rigorous_pruning` (L ⇒ rigorous; composing PostFilterL).
  • `f_pruned_implies_rigorous_pruning` (full chain F ⇒ Q ⇒ L ⇒ rigorous).
  • `q_pruned_implies_rigorous_pruning` (Q ⇒ L ⇒ rigorous, skipping F).
  • `farkas_certificate_implies_l_pruned` (SDP Farkas certificate ⇒ L-pruned).

HYPOTHESES USED PER LEMMA (cascade-context standard):
  All chain theorems use:  0 < n, 0 < m  (positivity, always true for the
  cascade).  The Q ⇒ L step additionally uses Σ c = 4nm (simplex constraint,
  always true for compositions enumerated by `compositions.py`).

NEW AXIOMS DECLARED IN THIS FILE:  ZERO.
NEW SORRYS IN THIS FILE:           ZERO.
═══════════════════════════════════════════════════════════════════════════════
-/

-- Axiom audit (uncomment to verify only kernel axioms used).
#print axioms PostFilterChain.f_pruned_implies_q_pruned
#print axioms PostFilterChain.q_pruned_implies_l_pruned
#print axioms PostFilterChain.l_pruned_implies_rigorous_pruning
#print axioms PostFilterChain.f_pruned_implies_rigorous_pruning
#print axioms PostFilterChain.q_pruned_implies_rigorous_pruning
#print axioms PostFilterChain.farkas_certificate_implies_l_pruned
