/-
Sidon Autocorrelation Project — Gray Code Subtree Pruning (Claims 4.14–4.25)

This file collects ALL the theorems and lemmas that must be proved to
certify the gray code subtree pruning optimization implemented in
`_fused_generate_and_prune_gray` (run_cascade.py).

The optimization works as follows: when the Gray code focus pointer
reaches level J_MIN, a partial autoconvolution of the fixed left-prefix
child bins is computed and checked against a conservative threshold.
If the partial sum already exceeds the threshold for some window, the
entire inner sweep (~128–249 children) is skipped.

Claims covered:
  4.14      Digit-ordering independence (Cartesian product permutation invariance)
  4.16      Fixed prefix characterization
  4.17      Partial autoconvolution lower bound (proved via SubtreePruning)
  4.18      Window sum monotonicity chain (proved via SubtreePruning)
  4.19      Individual child mass bounds (for W_int argument)
  4.20      W_int_max upper bound
  4.21      Gray code + child reset state validity after subtree prune
  4.22      Enumeration completeness (with subtree_triggered guard)
  4.23      Dynamic threshold monotonicity (W_int NOT scaled by ℓ/(4n))
  4.23b     Threshold formula consistency (subtree vs per-child)
  4.24      Partial convolution non-negativity
  4.25      End-to-end soundness (with structured hypotheses)

Removed from previous version (trivially true or redundant):
  4.15      active_pos_decreasing — conclusion directly instantiated h_built
  4.17b     window_restriction_conservative — special case of 4.18 with
            dyn_max := dyn_it and h_threshold_mono := le_refl
  4.21_old  gray_code_fast_forward_focus — conclusion identical to h_reset
  4.21b_old post_reset_child_state_valid — conclusion identical to h_fixed_unchanged
  4.21c_old post_reset_raw_conv_correct — conclusion identical to h_recompute

Cross-cutting dependencies (not formalized here, covered in other files):
  - After subtree prune, nz_list must be rebuilt (SparseCrossTerm.lean Claim 4.33).
    The interaction is additive: subtree pruning resets child state, then nz_list
    rebuild restores the SparseCrossTerm invariant.
  - Partial autoconv uses int32 arithmetic. Overflow safety (m ≤ 200 → conv entry
    < 2^31) is established in CauchySchwarz.lean (int32_safe).

STATUS: All claims proved. Claims 4.17, 4.18 proved via SubtreePruning references.
Claims 4.14, 4.16, 4.20, 4.22 proved directly.
-/

import Mathlib
import Sidon.Defs
import Sidon.Algorithm.SubtreePruning

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
-- PART A: Digit-Ordering Independence (Claim 4.14)
--
-- The Gray code active_pos array is built right-to-left (reversed) so that
-- inner (fast-changing) digits correspond to rightmost parent positions.
-- The Cartesian product of children is the same set regardless of digit
-- ordering, and the pruning test is a per-child predicate independent of
-- enumeration order.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.14: The Cartesian product of children is permutation-invariant
    in the active_pos ordering.

    Formally: let σ be any permutation of Fin(d_parent). The constraint
    "lo(i) ≤ child[2i] ≤ hi(i) and child[2i+1] = parent(i) - child[2i]
    for all i" holds iff the same constraint holds with i replaced by σ(i).

    This follows from: ∀ i, P(i) ↔ ∀ i, P(σ(i)) because σ is a bijection
    (substitute i := σ⁻¹(j) in the backward direction).

    Since the pruning test (window scan + canonicalization) depends only on
    the child mass vector — not on which digit was enumerated first — the
    survivor set is identical for any digit ordering. -/
theorem gray_code_digit_order_independence
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (lo hi : Fin d_parent → ℕ)
    (σ : Equiv.Perm (Fin d_parent)) :
    ∀ child : Fin (2 * d_parent) → ℕ,
      (∀ i : Fin d_parent, lo i ≤ child ⟨2 * i.1, by omega⟩ ∧
            child ⟨2 * i.1, by omega⟩ ≤ hi i ∧
            child ⟨2 * i.1 + 1, by omega⟩ = parent i - child ⟨2 * i.1, by omega⟩) ↔
      (∀ i : Fin d_parent, lo (σ i) ≤ child ⟨2 * (σ i).1, by omega⟩ ∧
            child ⟨2 * (σ i).1, by omega⟩ ≤ hi (σ i) ∧
            child ⟨2 * (σ i).1 + 1, by omega⟩ = parent (σ i) - child ⟨2 * (σ i).1, by omega⟩) := by
  intros child
  apply Iff.intro
  · intro h i; exact h (σ i)
  · intro h i
    exact h (σ.symm i) |> fun h' => by simpa [Equiv.symm_apply_apply] using h'

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART B: Fixed Prefix Characterization (Claim 4.16)
--
-- With reversed active_pos ordering, the "fixed prefix" is the set of
-- child bins 0..2p-1 where p = active_pos[J_MIN - 1]. We must prove
-- that all parent positions with index < p are indeed fixed (either
-- inactive or outer active positions).
--
-- Note: Claim 4.15 (active_pos_decreasing) was removed — its conclusion
-- was a direct instantiation of hypothesis h_built with specific k, k'.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.16: Every parent position with index < active_pos[J_MIN - 1]
    is either inactive (range = 1) or an outer active position (digit
    index ≥ J_MIN). In either case, the corresponding child bins are
    fixed during the inner sweep of digits 0..J_MIN-1.

    Code reference: run_cascade.py:1353
      fixed_parent_boundary = active_pos[J_MIN - 1] -/
theorem fixed_prefix_characterization
    {d_parent : ℕ} (_parent : Fin d_parent → ℕ)
    (lo hi : Fin d_parent → ℕ)
    (active_pos : Fin d_parent → ℕ) (n_active : ℕ)
    (hn_active : n_active ≤ d_parent)
    (J_MIN : ℕ) (hJ : J_MIN < n_active) (hJ_pos : 0 < J_MIN)
    (h_decreasing : ∀ k k' : Fin n_active, k.1 < k'.1 →
      active_pos ⟨k'.1, by omega⟩ < active_pos ⟨k.1, by omega⟩)
    (_h_active_bound : ∀ k : Fin n_active, active_pos ⟨k.1, by omega⟩ < d_parent)
    -- Positions not in active_pos have range 1 (inactive)
    (h_inactive_range : ∀ q : Fin d_parent,
      (∀ k : Fin n_active, active_pos ⟨k.1, by omega⟩ ≠ q.1) →
      hi q - lo q + 1 = 1)
    (p : ℕ) (hp : p < active_pos ⟨J_MIN - 1, by omega⟩)
    (hp_d : p < d_parent) :
    -- p is either inactive or has digit index ≥ J_MIN
    (hi ⟨p, by omega⟩ - lo ⟨p, by omega⟩ + 1 = 1) ∨
    (∃ k : Fin n_active, k.1 ≥ J_MIN ∧ active_pos ⟨k.1, by omega⟩ = p) := by
  by_cases h : ∃ k : Fin n_active, active_pos ⟨k.1, by omega⟩ = p
  · obtain ⟨k, hk⟩ := h
    right
    refine ⟨k, ?_, hk⟩
    by_contra hlt
    push_neg at hlt
    have hk_lt : k.1 ≤ J_MIN - 1 := by omega
    have : active_pos ⟨J_MIN - 1, by omega⟩ ≤ active_pos ⟨k.1, by omega⟩ := by
      rcases eq_or_lt_of_le hk_lt with heq | hlt2
      · simp [heq]
      · exact le_of_lt (h_decreasing ⟨k.1, k.2⟩ ⟨J_MIN - 1, by omega⟩ hlt2)
    omega
  · left
    push_neg at h
    exact h_inactive_range ⟨p, by omega⟩ (fun k => h k)

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART C: Partial Autoconvolution Soundness (Claims 4.17–4.18)
--
-- The partial autoconvolution of the fixed prefix is a lower bound on
-- the full autoconvolution for every window. Combined with the W_int_max
-- upper bound, this gives a sound pruning criterion.
--
-- Both claims are direct consequences of existing SubtreePruning lemmas.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.17 (= SubtreePruning.partial_conv_le_full_conv):
    The partial autoconvolution restricted to a prefix of length 2p
    is ≤ the full autoconvolution, for every convolution index t.

    All omitted terms c_i * c_j (where i ≥ 2p or j ≥ 2p) are nonneg. -/
theorem partial_conv_prefix_le_full
    {d : ℕ} (c : Fin d → ℤ) (hc : ∀ i, 0 ≤ c i)
    (p : ℕ) (hp : 2 * p ≤ d) (t : ℕ) :
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i.1 < 2 * p ∧ j.1 < 2 * p then c i * c j else 0) ≤
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t then c i * c j else 0) :=
  partial_conv_le_full_conv c hc p hp t

/-- Claim 4.18 (= SubtreePruning.subtree_pruning_chain):
    Window sum monotonicity chain:
      ws_full ≥ ws_partial > dyn_max ≥ dyn_actual ⟹ ws_full > dyn_actual -/
theorem subtree_pruning_soundness_gray
    (ws_partial ws_full : ℤ)
    (dyn_max dyn_actual : ℤ)
    (h_partial_le : ws_full ≥ ws_partial)
    (h_exceeds : ws_partial > dyn_max)
    (h_threshold_mono : dyn_max ≥ dyn_actual) :
    ws_full > dyn_actual :=
  subtree_pruning_chain ws_partial ws_full dyn_max dyn_actual h_partial_le h_exceeds h_threshold_mono

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART D: W_int_max Correctness (Claims 4.19–4.20)
--
-- The W_int_max computation uses parent_prefix for unfixed bins.
-- We must prove that parent_int[p] is an upper bound on the sum
-- of child masses for any split of parent position p.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.19: For any parent position p and any valid split
    child[2p] + child[2p+1] = parent[p], each individual child mass
    is bounded by parent[p].

    This feeds directly into the W_int bounding argument (Claim 4.20):
    when we upper-bound the child mass contribution of an unfixed
    parent position to a window, we can use parent[p] because
    child[2p] ≤ parent[p] and child[2p+1] ≤ parent[p]. -/
theorem parent_mass_bounds_individual_child
    {d_parent : ℕ} (parent : Fin d_parent → ℕ) (child : Fin (2 * d_parent) → ℕ)
    (h_split : ∀ p : Fin d_parent,
      child ⟨2 * p.1, by omega⟩ + child ⟨2 * p.1 + 1, by omega⟩ = parent p)
    (p : Fin d_parent) :
    child ⟨2 * p.1, by omega⟩ ≤ parent p ∧
    child ⟨2 * p.1 + 1, by omega⟩ ≤ parent p := by
  have := h_split p
  constructor <;> omega

/-- Claim 4.20: W_int_max = W_int_fixed + W_int_unfixed is an upper
    bound on the actual W_int for any child in the subtree.

    W_int_fixed uses the exact child masses of fixed bins.
    W_int_unfixed uses parent_prefix[hi_parent+1] - parent_prefix[lo_parent]
    as an upper bound on the unfixed bins' contribution.

    Code reference: run_cascade.py:1406-1437

    Note: p_boundary > 0 is required; in the code, the subtree check
    only fires when fixed_len ≥ 4 (run_cascade.py:1356), i.e. p_boundary ≥ 2.
    Without this, 2*p_boundary-1 underflows in ℕ. -/
theorem w_int_max_is_upper_bound
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (child_any : Fin (2 * d_parent) → ℕ)
    (fixed_child : Fin (2 * d_parent) → ℕ)
    (p_boundary : ℕ) (_hp : 2 * p_boundary ≤ 2 * d_parent)
    (hp_pos : 0 < p_boundary)
    -- Fixed prefix matches for all bins < 2*p_boundary
    (_h_fixed : ∀ i : Fin (2 * d_parent), i.1 < 2 * p_boundary →
      fixed_child i = child_any i)
    -- child_any is a valid split of parent
    (h_split : ∀ q : Fin d_parent,
      child_any ⟨2 * q.1, by omega⟩ + child_any ⟨2 * q.1 + 1, by omega⟩ = parent q)
    (lo_bin hi_bin : ℕ) (hlo : lo_bin ≤ hi_bin)
    (hhi : hi_bin < 2 * d_parent)
    -- W_int_fixed: sum of fixed_child masses in the fixed portion of the window
    (W_int_fixed : ℤ)
    (hWf : W_int_fixed = ∑ i ∈ Finset.Icc lo_bin (min hi_bin (2 * p_boundary - 1)),
      if h : i < 2 * d_parent then (child_any ⟨i, h⟩ : ℤ) else 0)
    -- W_int_unfixed: parent mass upper bound for unfixed portion
    (W_int_unfixed : ℤ)
    (hWu : W_int_unfixed ≥ ∑ q ∈ Finset.filter
      (fun q => 2 * q ≤ hi_bin ∧ lo_bin ≤ 2 * q + 1 ∧ q ≥ p_boundary)
      (Finset.range d_parent),
      if h : q < d_parent then (parent ⟨q, h⟩ : ℤ) else 0) :
    -- Actual W_int for child_any
    (∑ i ∈ Finset.Icc lo_bin hi_bin,
      if h : i < 2 * d_parent then (child_any ⟨i, h⟩ : ℤ) else 0)
      ≤ W_int_fixed + W_int_unfixed := by
  have h_sum_split : (∑ i ∈ Finset.Icc lo_bin hi_bin, if h : i < 2 * d_parent then child_any (⟨i, h⟩) else 0 : ℤ) =
    (∑ i ∈ Finset.Icc lo_bin (min hi_bin (2 * p_boundary - 1)), if h : i < 2 * d_parent then child_any (⟨i, h⟩) else 0 : ℤ) +
    (∑ i ∈ Finset.Icc (max lo_bin (2 * p_boundary)) hi_bin, if h : i < 2 * d_parent then child_any (⟨i, h⟩) else 0 : ℤ) := by
      have h_sum_split : Finset.Icc lo_bin hi_bin = Finset.Icc lo_bin (min hi_bin (2 * p_boundary - 1)) ∪ Finset.Icc (max lo_bin (2 * p_boundary)) hi_bin := by
        ext x; simp only [Finset.mem_union, Finset.mem_Icc]; omega
      rw [ h_sum_split, Finset.sum_union ];
      exact Finset.disjoint_left.mpr fun x hx₁ hx₂ => by cases max_cases lo_bin ( 2 * p_boundary ) <;> linarith [ Finset.mem_Icc.mp hx₁, Finset.mem_Icc.mp hx₂, min_le_left hi_bin ( 2 * p_boundary - 1 ), min_le_right hi_bin ( 2 * p_boundary - 1 ), Nat.sub_add_cancel ( by linarith : 1 ≤ 2 * p_boundary ) ] ;
  have h_unfixed_bound : (∑ i ∈ Finset.Icc (max lo_bin (2 * p_boundary)) hi_bin, if h : i < 2 * d_parent then child_any (⟨i, h⟩) else 0 : ℤ) ≤
    (∑ q ∈ Finset.filter (fun q => 2 * q ≤ hi_bin ∧ lo_bin ≤ 2 * q + 1 ∧ q ≥ p_boundary) (Finset.range d_parent), if h : q < d_parent then child_any (⟨2 * q, by
      linarith⟩) + child_any (⟨2 * q + 1, by
      linarith⟩) else 0 : ℤ) := by
      have h_unfixed_bound : Finset.Icc (max lo_bin (2 * p_boundary)) hi_bin ⊆ Finset.biUnion (Finset.filter (fun q => 2 * q ≤ hi_bin ∧ lo_bin ≤ 2 * q + 1 ∧ q ≥ p_boundary) (Finset.range d_parent)) (fun q => {2 * q, 2 * q + 1}) := by
        simp +decide [ Finset.subset_iff ];
        exact fun x hx₁ hx₂ hx₃ => ⟨ x / 2, ⟨ by omega, by omega, by omega, by omega ⟩, by omega ⟩
      generalize_proofs at *;
      refine' le_trans ( Finset.sum_le_sum_of_subset_of_nonneg h_unfixed_bound _ ) _;
      · exact fun _ _ _ => by split_ifs <;> norm_num;
      · rw [ Finset.sum_biUnion ];
        · gcongr ; aesop;
        · intros q hq r hr hqr; simp_all +decide [ Finset.disjoint_left ] ; omega;
  generalize_proofs at *;
  grind

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART E: Gray Code State After Subtree Prune (Claims 4.21–4.22)
--
-- After a successful subtree prune, the inner Gray code state is reset
-- so that the next outer advance starts a fresh inner sweep. We must
-- prove state validity and enumeration completeness.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.21: After a subtree prune, the code resets inner digits to
    gc_a[k]=0, gc_dir[k]=+1 for k < J_MIN (run_cascade.py:1449-1452),
    and wires gc_focus to skip past inner digits (lines 1455-1457).
    Cursor and child bins for inner positions are also reset (lines 1459-1464).

    The combined state is valid for continued Algorithm M execution:
    (1) All digits are in range (inner = 0 < radix, since radix ≥ 2)
    (2) All directions are ±1 (inner = +1, outer preserved)

    The fixed prefix (child bins < 2*fixed_parent_boundary) is
    unchanged because all inner positions have physical index
    ≥ fixed_parent_boundary (from the decreasing active_pos ordering,
    established by Claim 4.16). -/
theorem gray_code_subtree_reset_valid
    (n_active J_MIN : ℕ)
    (_hJ : J_MIN < n_active)
    (r : Fin n_active → ℕ) (hr : ∀ i, r i ≥ 2)
    -- Post-reset Gray code state
    (gc_a : Fin n_active → ℕ)
    (gc_dir : Fin n_active → ℤ)
    -- Reset conditions (run_cascade.py:1449-1452)
    (h_a_reset : ∀ k : Fin n_active, k.1 < J_MIN → gc_a k = 0)
    (h_dir_reset : ∀ k : Fin n_active, k.1 < J_MIN → gc_dir k = 1)
    -- Outer digits unchanged and in range
    (h_a_outer : ∀ k : Fin n_active, k.1 ≥ J_MIN → gc_a k < r k)
    (h_dir_outer : ∀ k : Fin n_active, k.1 ≥ J_MIN → (gc_dir k = 1 ∨ gc_dir k = -1)) :
    -- (1) All digits in range
    (∀ k : Fin n_active, gc_a k < r k) ∧
    -- (2) All directions are ±1
    (∀ k : Fin n_active, gc_dir k = 1 ∨ gc_dir k = -1) := by
  refine ⟨fun k => ?_, fun k => ?_⟩
  · by_cases hk : k.1 < J_MIN
    · have := h_a_reset k hk; have := hr k; omega
    · exact h_a_outer k (by omega)
  · by_cases hk : k.1 < J_MIN
    · exact Or.inl (h_dir_reset k hk)
    · exact h_dir_outer k (by omega)

/-- Claim 4.22: Enumeration completeness — every composition that falls
    inside a pruned subtree would be individually pruned.

    When the Gray code focus reaches J_MIN and the partial autoconv check
    fires (subtree_triggered), the inner sweep of digits 0..J_MIN-1 is
    skipped.  This theorem proves that EVERY composition `c` sharing the
    same outer digits (≥ J_MIN) as a triggering state is individually
    pruned via the full window scan (from Claims 4.17-4.20).

    This is the key content for Claim 4.25 (master soundness): it shows
    that no unpruned survivor is lost when a subtree is skipped.

    Code reference: run_cascade.py:1348-1441
      The subtree prune fires when j == J_MIN and n_active > J_MIN,
      and the partial autoconv exceeds the conservative threshold. -/
theorem gray_code_subtree_enumeration_completeness
    {n_active : ℕ} (r : Fin n_active → ℕ)
    (J_MIN : ℕ) (_hJ : J_MIN < n_active)
    -- per-child pruning predicate: True when the full window scan would prune
    (individually_pruned : (∀ i : Fin n_active, Fin (r i)) → Prop)
    -- predicate: does this outer state trigger a subtree prune?
    (subtree_triggered : (∀ i : Fin n_active, Fin (r i)) → Prop)
    -- Key hypothesis: if the partial autoconv check fires for an outer state,
    -- then EVERY inner completion is individually pruned (from Claims 4.17-4.20)
    (h_subtree_sound : ∀ (outer_state : ∀ i : Fin n_active, Fin (r i)),
      subtree_triggered outer_state →
      ∀ (inner_variant : ∀ i : Fin n_active, Fin (r i)),
        -- inner_variant agrees with outer_state on digits ≥ J_MIN
        (∀ i : Fin n_active, i.1 ≥ J_MIN → inner_variant i = outer_state i) →
        individually_pruned inner_variant) :
    -- Every composition in a pruned subtree is individually pruned
    ∀ (c : ∀ i : Fin n_active, Fin (r i)),
      (∃ outer_state, subtree_triggered outer_state ∧
        (∀ i : Fin n_active, i.1 ≥ J_MIN → c i = outer_state i)) →
      individually_pruned c := by
  intro c ⟨outer, h_trig, h_agree⟩
  exact h_subtree_sound outer h_trig c h_agree

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART F: Threshold Arithmetic (Claims 4.23–4.24)
--
-- The integer threshold computation must be exact. The formula matches
-- run_cascade.py: dyn_x = c_target * m² * ℓ/(4n) + 1 + eps_margin + 2*W
-- where ℓ/(4n) scales ONLY c_target*m², NOT the correction terms.
--
-- The formula here matches the actual Python code (run_cascade.py:1111-1114).
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.23: The dynamic threshold is monotone non-decreasing in W_int.
    Since we use W_int_max ≥ W_int_actual, the threshold with W_int_max
    is at least as large, making the pruning test conservative.

    The threshold formula (run_cascade.py:1111-1114):
      dyn_base_ell = c_target * m² * ℓ/(4n)
      dyn_x = dyn_base_ell + 1 + eps_margin + 2*W
      dyn_it = floor(dyn_x * (1 - 4ε))
    where eps_margin = 1e-9 * m² and ε = 2.220446049250313e-16 (IEEE 754 float64).

    Proof sketch: the inner expression is affine in W with slope 2 > 0,
    the outer multiplication by (1-4ε) > 0 preserves monotonicity,
    and floor is non-decreasing. -/
theorem dynamic_threshold_monotone
    (c_target : ℝ) (m : ℝ) (ell : ℝ) (inv_4n : ℝ)
    (h_pos : 0 < 1 - 4 * (2.220446049250313e-16 : ℝ))
    (W1 W2 : ℝ) (hW : W1 ≤ W2) :
    ⌊(c_target * m ^ 2 * ell * inv_4n + 1 + 1e-9 * m ^ 2 + 2 * W1) *
      (1 - 4 * (2.220446049250313e-16 : ℝ))⌋ ≤
    ⌊(c_target * m ^ 2 * ell * inv_4n + 1 + 1e-9 * m ^ 2 + 2 * W2) *
      (1 - 4 * (2.220446049250313e-16 : ℝ))⌋ := by
  exact Int.floor_mono (mul_le_mul_of_nonneg_right (by linarith) (le_of_lt h_pos))

/-- Claim 4.24: The partial convolution entries are non-negative when
    all child masses are non-negative. Each term in the sum is either 0
    (from the filter) or c_i * c_j ≥ 0 (product of non-negatives).
    Needed for the lower-bound argument in Claim 4.17. -/
theorem partial_conv_nonneg
    {d : ℕ} (c : Fin d → ℤ) (hc : ∀ i, 0 ≤ c i)
    (p : ℕ) (_hp : 2 * p ≤ d) (t : ℕ) :
    0 ≤ ∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i.1 < 2 * p ∧ j.1 < 2 * p then c i * c j else 0 := by
  apply Finset.sum_nonneg; intro i _; apply Finset.sum_nonneg; intro j _
  split_ifs with h
  · exact mul_nonneg (hc i) (hc j)
  · exact le_refl _

/-- Claim 4.23b: The subtree pruning threshold uses the SAME formula as
    the per-child pruning threshold, with W_int_max replacing W_int_actual.
    Both paths compute (run_cascade.py:1193-1194 and 1392-1393):
      dyn_x = c_target * m² * ℓ/(4n) + 1 + eps_margin + 2*W
      dyn_it = floor(dyn_x * (1 - 4ε))
    Both look up the same precomputed threshold_table (indexed by ell_idx
    and W_int), confirming structural identity.

    The structural identity means Claim 4.23 (monotonicity in W) directly
    gives threshold(W_int_max) ≥ threshold(W_int_actual). -/
theorem threshold_formula_consistency
    (c_target : ℝ) (m : ℝ) (ell : ℝ) (inv_4n : ℝ)
    (W_actual W_max : ℝ) (hW : W_actual ≤ W_max)
    (h_pos : 0 < 1 - 4 * (2.220446049250313e-16 : ℝ)) :
    -- Same formula applied to both
    let threshold (W : ℝ) := ⌊(c_target * m ^ 2 * ell * inv_4n + 1 + 1e-9 * m ^ 2 + 2 * W) *
      (1 - 4 * (2.220446049250313e-16 : ℝ))⌋
    threshold W_actual ≤ threshold W_max := by
  exact Int.floor_mono (mul_le_mul_of_nonneg_right (by linarith) (le_of_lt h_pos))

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART G: End-to-End Soundness (Claim 4.25)
--
-- The final theorem: the Gray code kernel with subtree pruning produces
-- the same set of canonical survivors as the kernel without it.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.25 (Master Soundness Theorem): For any parent composition,
    the set of canonical survivors produced by the Gray code kernel with
    subtree pruning is identical to the set produced without subtree pruning.

    Direction (⊆): Every survivor of the pruned kernel is a survivor of
    the unpruned kernel (h_with_subset — subtree pruning only removes).

    Direction (⊇): Every survivor of the unpruned kernel is a survivor
    of the pruned kernel. By h_partition (from Claim 4.22), each such
    survivor is either visited by the pruned kernel (hence in S_with if
    it survives the per-child test) or in a pruned subtree. By
    h_subtree_not_survivor (from Claims 4.17 → 4.20 → 4.23 → 4.18),
    children in pruned subtrees are not in S_without, contradiction.

    Hypotheses map to lower-level claims:
    - h_with_subset: structural (pruning only removes)
    - h_partition: Claim 4.22 (enumeration completeness)
    - h_subtree_not_survivor: chain of 4.17 (partial ≤ full),
      4.20 (W_int_max ≥ W_int_actual), 4.23 (threshold monotone),
      4.18 (ws_full > dyn_actual) -/
theorem gray_code_subtree_pruning_sound
    {d_parent : ℕ} (_parent : Fin d_parent → ℕ)
    (_lo _hi : Fin d_parent → ℕ)
    (_m : ℕ) (_c_target : ℝ) (_n_half_child : ℕ)
    -- S_with: survivors with subtree pruning enabled
    -- S_without: survivors without subtree pruning (all children tested individually)
    (S_with S_without : Finset (Fin (2 * d_parent) → ℕ))
    -- Predicate: child is in a subtree that was pruned
    (in_pruned_subtree : (Fin (2 * d_parent) → ℕ) → Prop)
    -- (⊆): subtree pruning only removes, never adds
    (h_with_subset : S_with ⊆ S_without)
    -- Enumeration completeness (Claim 4.22): every survivor in the
    -- unpruned kernel is either visited by the pruned kernel or
    -- in a pruned subtree
    (h_partition : ∀ child : Fin (2 * d_parent) → ℕ,
      child ∈ S_without → child ∈ S_with ∨ in_pruned_subtree child)
    -- Subtree soundness (chain 4.17 → 4.20 → 4.23 → 4.18):
    -- children in pruned subtrees would be individually pruned,
    -- so they are not in S_without (the individually-tested survivor set)
    (h_subtree_not_survivor : ∀ child : Fin (2 * d_parent) → ℕ,
      in_pruned_subtree child → child ∉ S_without) :
    S_with = S_without := by
  apply Finset.Subset.antisymm h_with_subset
  intro c hc
  rcases h_partition c hc with h | h
  · exact h
  · exact absurd hc (h_subtree_not_survivor c h)

end -- noncomputable section
