/-
Sidon Autocorrelation Project — Sparse Cross-Term Optimization (Claims 4.26–4.35)

This file collects ALL the theorems and lemmas that must be proved to
certify the sparse cross-term optimization implemented in
`_fused_generate_and_prune_gray` (run_cascade.py).

The optimization works as follows: instead of iterating all d_child bins
in the cross-term update loop (checking `if child[j] != 0` for each),
we maintain an explicit nonzero index list `nz_list` with a reverse-index
`nz_pos`. When d_child ≥ 32, the cross-term loop iterates only over
`nz_list`, skipping zero bins entirely.

The nz_list is maintained incrementally: when the Gray code advances and
bins k1, k2 change values, at most 2 add/remove operations on nz_list
keep it in sync. After a subtree prune (which resets child bins and does
a full raw_conv recompute), nz_list is rebuilt from scratch.

STATUS: ALL PROOFS COMPLETE — 9 theorems, 0 sorry.
Dependencies on existing modules (Defs, IncrementalAutoconv, GrayCode)
are noted. All dependencies are passed as hypotheses (self-contained).

AUDIT FIXES (2026-03-28):
- Claim 4.26: Changed conclusion from Finset equality to biconditional form
  (matching downstream consumers in Claims 4.30, 4.31).
- Claim 4.27: Removed (redundant — conclusion restated hypothesis h_forward).
- Claims 4.28–4.29: Changed conclusion from `True` to substantive
  set-membership biconditionals about the updated nz_list.
- Claim 4.30: Added constraining hypotheses (h_k1, h_k2, h_rest) connecting
  nz_list' to the four-case update result.
- Claim 4.31: (a) Removed `child j ≠ 0` from dense-side filter (dense code
  iterates all j ≠ k1,k2; zero terms contribute 0 via 2·δ·0 = 0).
  (b) Added k2 contribution (delta2 · child[j] at k2+j), previously missing.
- Claim 4.32: Added connecting hypotheses (h_dense, h_sparse) defining both
  raw_conv arrays as raw_conv_after_self plus respective cross-term deltas.
- Claim 4.33: Changed conclusion from Finset equality to biconditional form.
- Claim 4.35: Added hypotheses (h_same, h_S_dense, h_S_sparse) defining
  survivor sets via a common enumeration and pruning test.
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

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART A: Nonzero List Invariant (Claim 4.26)
--
-- The nz_list is a faithful representation of the set of nonzero child bins.
-- This invariant must hold at every point where the cross-term loop executes.
--
-- Claim 4.27 (nz_pos consistency) has been removed: its conclusion
-- `nz_list[nz_pos[i]] = i` was the third conjunct of hypothesis h_forward,
-- making the theorem a trivial extraction with no added content.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.26: nz_list invariant — child[i] ≠ 0 iff i appears in
    nz_list[0..nz_count-1].

    This is the core invariant for the sparse cross-term loop.
    It must hold:
      (a) after initialization from the first child,
      (b) after each incremental nz_list update following a Gray code step,
      (c) after nz_list rebuild following a subtree prune.

    The biconditional form is consumed by Claims 4.30, 4.31, 4.32. -/
theorem nz_list_invariant
    {d : ℕ} (child : Fin d → ℤ)
    (nz_list : Fin d → ℕ)
    (nz_count : ℕ) (hnz_count : nz_count ≤ d)
    (h_valid : ∀ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ < d)
    (_h_distinct : ∀ k₁ k₂ : Fin nz_count,
      nz_list ⟨k₁.1, by omega⟩ = nz_list ⟨k₂.1, by omega⟩ → k₁ = k₂)
    (h_nonzero : ∀ k : Fin nz_count,
      child ⟨nz_list ⟨k.1, by omega⟩, h_valid k⟩ ≠ 0)
    (h_complete : ∀ i : Fin d, child i ≠ 0 →
      ∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = i.1) :
    ∀ i : Fin d, child i ≠ 0 ↔
      ∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = i.1 := by
  intro i
  constructor
  · exact h_complete i
  · rintro ⟨k, hk⟩
    have heq : (⟨nz_list ⟨k.1, by omega⟩, h_valid k⟩ : Fin d) = i := Fin.ext hk
    rw [← heq]
    exact h_nonzero k

/-
PROBLEM
═══════════════════════════════════════════════════════════════════════════════
PART B: Incremental Update Correctness (Claims 4.28–4.30)

When the Gray code advances and exactly one cursor position changes,
bins k1 = 2*pos and k2 = 2*pos+1 get new values. The nz_list must be
updated to reflect these changes. There are four cases per bin:
nonzero → zero:   swap-remove from list
zero → nonzero:   append to list
nonzero → nonzero: no change needed
zero → zero:       no change needed
═══════════════════════════════════════════════════════════════════════════════

Claim 4.28: Swap-remove preserves the nz_list invariant (minus one element).

    When removing index i from nz_list, we swap it with the last element
    and decrement nz_count. The resulting nz_list represents exactly the
    original set minus {i}.

    Code reference: run_cascade.py:1304-1309
      p = nz_pos[k]; nz_count -= 1
      last = nz_list[nz_count]; nz_list[p] = last
      nz_pos[last] = p; nz_pos[k] = -1

PROVIDED SOLUTION
For each j : Fin d, prove both directions of the iff.

(→) Suppose ⟨k', hk'⟩ witnesses nz_list'[k'] = j for some k' : Fin nz_count'. We need to find a witness in the original nz_list and show j ≠ i.

Consider whether k'.val = p or k'.val ≠ p.

Case k'.val = p: nz_list'[p] = nz_list[nz_count-1] = j by h_swap (use Fin.val equality). So ⟨nz_count-1, by omega⟩ is a Fin nz_count witness. j ≠ i because if j = i then nz_list[nz_count-1] = i = nz_list[p], so by h_distinct (applied to ⟨nz_count-1, by omega⟩ and ⟨p, hp⟩) we get nz_count-1 = p. But then nz_count' = nz_count - 1 = p, and k'.val = p = nz_count', contradicting k' : Fin nz_count' (i.e., k'.val < nz_count').

Case k'.val ≠ p: k'.val < nz_count' and k'.val ≠ p, so by h_rest, nz_list'[k'] = nz_list[k']. Since nz_list'[k'] = j, we have nz_list[k'] = j. Use ⟨k'.val, by omega⟩ as witness (k'.val < nz_count' ≤ nz_count). j ≠ i because if j = i then nz_list[k'.val] = i.val = nz_list[p] (by h_at_p), so by h_distinct k'.val = p, contradicting k'.val ≠ p.

(←) Suppose ⟨k0, hk0⟩ witnesses nz_list[k0] = j, and j ≠ i. Since nz_list[k0] = j ≠ i = nz_list[p] (by h_at_p), h_distinct gives k0 ≠ p (as Fin nz_count values).

Use Nat.lt_or_eq_of_lt (k0.isLt) to split on whether k0.val < nz_count - 1 or k0.val = nz_count - 1.

Case k0.val < nz_count - 1 (= nz_count'): k0.val ≠ p (since k0 ≠ p as Fin elements), so h_rest gives nz_list'[k0.val] = nz_list[k0.val] = j. Use ⟨k0.val, by omega⟩ as Fin nz_count' witness.

Case k0.val = nz_count - 1: nz_list'[p] = nz_list[nz_count-1] = nz_list[k0] = j by h_swap. p < nz_count' because: p < nz_count (hp), and p ≠ nz_count - 1 (since k0.val = nz_count - 1 and k0 ≠ p implies p ≠ nz_count - 1), so p < nz_count - 1 = nz_count'. Use ⟨p, by omega⟩ as Fin nz_count' witness.

Key technical details: Use Fin.ext for equality of Fin values. Use omega for arithmetic. The h_distinct hypothesis takes Fin nz_count arguments, so construct them with appropriate bounds proofs.
-/
theorem swap_remove_preserves_invariant
    {d : ℕ} (nz_list nz_list' : Fin d → ℕ)
    (nz_count : ℕ) (hnz : 0 < nz_count) (hnz_d : nz_count ≤ d)
    -- No duplicates in original list
    (h_distinct : ∀ k₁ k₂ : Fin nz_count,
      nz_list ⟨k₁.1, by omega⟩ = nz_list ⟨k₂.1, by omega⟩ → k₁ = k₂)
    -- The index being removed
    (i : Fin d) (p : ℕ) (hp : p < nz_count)
    (h_at_p : nz_list ⟨p, by omega⟩ = i.1)
    -- Result of swap-remove
    (nz_count' : ℕ) (h_count' : nz_count' = nz_count - 1)
    (h_swap : nz_list' ⟨p, by omega⟩ = nz_list ⟨nz_count - 1, by omega⟩)
    (h_rest : ∀ (k : ℕ) (_hk : k < nz_count') (_hkp : k ≠ p),
      nz_list' ⟨k, by omega⟩ = nz_list ⟨k, by omega⟩) :
    -- The set in nz_list' = the set in nz_list minus {i}
    ∀ j : Fin d,
      (∃ k : Fin nz_count', nz_list' ⟨k.1, by omega⟩ = j.1) ↔
      ((∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = j.1) ∧ j ≠ i) := by
  all_goals generalize_proofs at *;
  intro j
  constructor;
  · rintro ⟨ k, hk ⟩ ; by_cases hk' : k.val = p <;> simp_all +decide [ Fin.ext_iff ] ;
    · refine' ⟨ ⟨ ⟨ nz_count - 1, by omega ⟩, hk ⟩, _ ⟩;
      intro H; specialize h_distinct ⟨ p, hp ⟩ ⟨ nz_count - 1, by omega ⟩ ; simp_all +decide ;
      linarith [ Fin.is_lt k, Nat.sub_add_cancel hnz ];
    · refine' ⟨ ⟨ ⟨ k, by omega ⟩, _ ⟩, _ ⟩
      all_goals generalize_proofs at *;
      · grind +ring;
      · contrapose! h_distinct;
        use ⟨ k, by omega ⟩, ⟨ p, by omega ⟩ ; aesop;
  · intro hj
    obtain ⟨k, hk⟩ := hj.left
    generalize_proofs at *;
    by_cases hk_eq_p : k.val = p;
    · simp_all +decide [ Fin.ext_iff ];
    · by_cases hk_lt_nz_count' : k.val < nz_count';
      · exact ⟨ ⟨ k, by linarith ⟩, h_rest k hk_lt_nz_count' hk_eq_p ▸ hk ⟩;
      · use ⟨p, by omega⟩
        generalize_proofs at *;
        grind

/-
PROBLEM
(→) If nz_list'[k'] = j, trace k' through swap/rest to find j in nz_list.
j ≠ i by h_distinct: if j = i then nz_list[k''] = nz_list[p],
forcing k'' = p, but position p now holds nz_list[nz_count-1].
(←) If nz_list[k0] = j and j ≠ i:
case k0 < nz_count': if k0 ≠ p then nz_list'[k0] = nz_list[k0] = j;
if k0 = p then nz_list[p] = j = i, contradiction.
case k0 = nz_count-1: nz_list'[p] = nz_list[nz_count-1] = j,
and p < nz_count' (since p = nz_count-1 would give j = i).

Claim 4.29: Append preserves the nz_list invariant (plus one element).

    When adding index i to nz_list, we place it at position nz_count
    and increment nz_count.

    Code reference: run_cascade.py:1308-1309
      nz_list[nz_count] = k; nz_pos[k] = nz_count; nz_count += 1

PROVIDED SOLUTION
For each j : Fin d, prove both directions of the iff.

(→) Suppose ⟨k', hk'⟩ witnesses nz_list'[k'] = j for some k' : Fin nz_count'. Since nz_count' = nz_count + 1, either k'.val < nz_count or k'.val = nz_count.

Case k'.val < nz_count: By h_rest k'.val (by omega), nz_list'[k'.val] = nz_list[k'.val]. Since nz_list'[k'.val] = j (after Fin.val manipulation), nz_list[k'.val] = j. So ⟨k'.val, by omega⟩ : Fin nz_count witnesses j in original list → left disjunct.

Case k'.val = nz_count: nz_list'[nz_count] = i.val by h_append. Since nz_list'[k'] = j and k' has the same index as nz_count (after Fin coercion), j.val = i.val, so j = i by Fin.ext → right disjunct.

(←)
Left disjunct: ⟨k0, hk0⟩ witnesses nz_list[k0] = j with k0 : Fin nz_count. By h_rest k0.val k0.isLt, nz_list'[k0.val] = nz_list[k0.val] = j. And k0.val < nz_count < nz_count' (since nz_count' = nz_count + 1), so ⟨k0.val, by omega⟩ : Fin nz_count' is a valid witness.

Right disjunct: j = i. By h_append, nz_list'[nz_count] = i.val = j.val. nz_count < nz_count' (since nz_count' = nz_count + 1), so ⟨nz_count, by omega⟩ : Fin nz_count' is a valid witness.
-/
theorem append_preserves_invariant
    {d : ℕ} (nz_list nz_list' : Fin d → ℕ)
    (nz_count : ℕ) (hnz_d : nz_count < d)
    -- The index being added
    (i : Fin d)
    -- i is not already in nz_list
    (h_not_in : ∀ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ ≠ i.1)
    -- Result of append
    (nz_count' : ℕ) (h_count' : nz_count' = nz_count + 1)
    (h_append : nz_list' ⟨nz_count, by omega⟩ = i.1)
    (h_rest : ∀ (k : ℕ) (_hk : k < nz_count),
      nz_list' ⟨k, by omega⟩ = nz_list ⟨k, by omega⟩) :
    -- The set in nz_list' = the set in nz_list plus {i}
    ∀ j : Fin d,
      (∃ k : Fin nz_count', nz_list' ⟨k.1, by omega⟩ = j.1) ↔
      ((∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = j.1) ∨ j = i) := by
  intro j; constructor
  · rintro ⟨k', hk'⟩
    by_cases hlt : k'.1 < nz_count
    · left; exact ⟨⟨k'.1, hlt⟩, by rw [← h_rest k'.1 hlt]; exact hk'⟩
    · right
      have hkeq : k'.1 = nz_count := by omega
      apply Fin.ext
      calc j.1 = nz_list' ⟨k'.1, by omega⟩ := hk'.symm
        _ = nz_list' ⟨nz_count, by omega⟩ := by congr 1; exact Fin.ext hkeq
        _ = i.1 := h_append
  · rintro (⟨k, hk⟩ | rfl)
    · exact ⟨⟨k.1, by omega⟩, by rw [h_rest k.1 k.isLt]; exact hk⟩
    · exact ⟨⟨nz_count, by omega⟩, h_append⟩

/-
PROBLEM
(→) If k' < nz_count: nz_list'[k'] = nz_list[k'], giving left disjunct.
If k' = nz_count: nz_list'[nz_count] = i, giving right disjunct.
(←) Left: nz_list[k0] = j, so nz_list'[k0] = nz_list[k0] = j with k0 < nz_count'.
Right: j = i, use h_append with k' = nz_count < nz_count'.

Claim 4.30: After the four-case update (old→new for bins k1, k2),
    the nz_list invariant is restored for the updated child array.

    This composes Claims 4.28–4.29 for the two bins that change in
    each Gray code step. The key insight is that bins k1 and k2 are
    the ONLY bins that change, so the invariant for all other bins
    is trivially preserved.

    The hypotheses h_k1, h_k2, h_rest specify the postcondition of the
    four-case update procedure (established by composing Claims 4.28–4.29
    for each of k1 and k2 as needed).

    Code reference: run_cascade.py:1303-1315 (the four-case block)

PROVIDED SOLUTION
Intro i. Case split on whether i = k1 using by_cases.

Case i = k1: subst i. exact h_k1.symm.

Case i ≠ k1: Case split on whether i = k2 using by_cases.

  Case i = k2: subst i. exact h_k2.symm.

  Case i ≠ k2:
    Have h_eq : child' i = child i := h_unchanged i ‹i ≠ k1› ‹i ≠ k2›
    Rewrite child' i ≠ 0 as child i ≠ 0 using h_eq.
    Then use (h_inv_before i).trans (h_rest i ‹i ≠ k1› ‹i ≠ k2›).symm
    Or equivalently: constructor
      · intro h; rw [h_eq] at h; exact (h_rest i ‹i ≠ k1› ‹i ≠ k2›).mpr ((h_inv_before i).mp h)
      · intro h; rw [h_eq]; exact (h_inv_before i).mpr ((h_rest i ‹i ≠ k1› ‹i ≠ k2›).mp h)
-/
theorem incremental_nz_update_correct
    {d : ℕ} (child child' : Fin d → ℤ)
    (k1 k2 : Fin d) (_hk : k1 ≠ k2)
    -- Only k1, k2 changed
    (h_unchanged : ∀ i : Fin d, i ≠ k1 → i ≠ k2 → child' i = child i)
    -- nz_list was correct before
    (nz_list : Fin d → ℕ) (nz_count : ℕ) (hnz : nz_count ≤ d)
    (h_inv_before : ∀ i : Fin d, child i ≠ 0 ↔
      ∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = i.1)
    -- nz_list' is the result of the four-case update
    (nz_list' : Fin d → ℕ) (nz_count' : ℕ) (hnz' : nz_count' ≤ d)
    -- The update correctly tracks k1
    (h_k1 : (∃ k : Fin nz_count', nz_list' ⟨k.1, by omega⟩ = k1.1) ↔ child' k1 ≠ 0)
    -- The update correctly tracks k2
    (h_k2 : (∃ k : Fin nz_count', nz_list' ⟨k.1, by omega⟩ = k2.1) ↔ child' k2 ≠ 0)
    -- All other indices unchanged in nz_list
    (h_rest : ∀ j : Fin d, j ≠ k1 → j ≠ k2 →
      ((∃ k : Fin nz_count', nz_list' ⟨k.1, by omega⟩ = j.1) ↔
       (∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = j.1))) :
    -- nz_list' is correct for child'
    ∀ i : Fin d, child' i ≠ 0 ↔
      ∃ k : Fin nz_count', nz_list' ⟨k.1, by omega⟩ = i.1 := by
  intro i; by_cases hi : i = k1 <;> by_cases hi' : i = k2 <;> simp_all +decide ;

/-
PROBLEM
Case i = k1: exact h_k1.symm
Case i = k2: exact h_k2.symm
Case i ≠ k1, k2: chain h_unchanged + h_inv_before + h_rest

═══════════════════════════════════════════════════════════════════════════════
PART C: Cross-Term Equivalence (Claims 4.31–4.32)

The sparse cross-term loop computes the same raw_conv updates as the
original dense loop. This is the central correctness theorem.

Key insight: The dense loop iterates all j ≠ k1, k2 (via range boundaries).
When child[j] = 0, the contribution is 2·δ·0 = 0, so zero bins are
harmless. The sparse loop iterates only nz_list entries ≠ k1, k2, which
by the invariant (Claim 4.26) are exactly the nonzero bins. Since zero
bins contribute 0, both loops produce the same sum.

The self-terms (at indices 2·k1, 2·k2) and mutual term (at index k1+k2)
are computed identically by both paths (run_cascade.py:1296-1300, before
the if/else branch), so they cancel. Only the cross-terms differ.
═══════════════════════════════════════════════════════════════════════════════

Claim 4.31: The sparse cross-term sum equals the dense cross-term sum,
    for BOTH the delta1 (at k1+j) and delta2 (at k2+j) contributions.

    Dense path (run_cascade.py:1323-1333):
      for jj in range(k1):          # all j < k1
          raw_conv[k1+jj] += 2·δ₁·child[jj]
          raw_conv[k2+jj] += 2·δ₂·child[jj]
      for jj in range(k2+1, d):     # all j > k2
          raw_conv[k1+jj] += 2·δ₁·child[jj]
          raw_conv[k2+jj] += 2·δ₂·child[jj]

    Sparse path (run_cascade.py:1317-1322):
      for idx in range(nz_count):
          jj = nz_list[idx]
          if jj != k1 and jj != k2:
              raw_conv[k1+jj] += 2·δ₁·child[jj]
              raw_conv[k2+jj] += 2·δ₂·child[jj]

    These are equal because:
      (a) nz_list = {j | child[j] ≠ 0} (Claim 4.26 invariant)
      (b) For j with child[j] = 0: 2·δ·0 = 0 (zero terms contribute nothing)
      (c) Dense skips j ∈ {k1,k2} via range boundaries
      (d) Sparse skips j ∈ {k1,k2} via explicit check

    Note: the dense-side formalization does NOT filter on child j ≠ 0
    (matching the actual code which iterates all j in range). The equality
    holds because zero-valued terms contribute 0 to the sum.

    Depends on: Claim 4.26 (nz_list invariant).

PROVIDED SOLUTION
Intro t. Constructor (for the ∧).

For each component (k1-part and k2-part), the proof is the same structure. Let me describe the k1-part; the k2-part is identical with delta2 replacing delta1 and k2.1 + j.1 replacing k1.1 + j.1.

The key idea: both sums compute the same thing because:
1. When child j = 0, the summand is `if ... then 2 * delta1 * 0 else 0 = 0` regardless of the condition.
2. The nz_list bijects onto exactly the nonzero entries.

More precisely, define f(j) = if j ≠ k1 ∧ j ≠ k2 ∧ k1.1 + j.1 = t then 2 * delta1 * child j else 0.

Step 1: Show ∑ j : Fin d, f(j) = ∑ j ∈ Finset.univ.filter (fun j => child j ≠ 0), f(j).
This is because for j with child j = 0, f(j) = 0 (if the condition is false, it's 0; if the condition is true, 2 * delta1 * child j = 2 * delta1 * 0 = 0). Use Finset.sum_filter_of_ne or show that f(j) = 0 when child j = 0 and use Finset.sum_subset.

Step 2: Show ∑ j ∈ {j | child j ≠ 0}, f(j) = ∑ idx : Fin nz_count, f(nz_list[idx]).
This is a reindexing via the bijection given by h_inv, h_distinct, h_valid.

Actually, a cleaner approach: Show both sums are equal by showing:
∑ j : Fin d, f(j) = ∑ idx : Fin nz_count, f(⟨nz_list[idx], h_valid idx⟩)

Use Finset.sum_nbij with:
- The injection i : Fin nz_count → Fin d given by i(idx) = ⟨nz_list[idx], h_valid idx⟩
- Injectivity from h_distinct
- The range is {j | child j ≠ 0} (from h_inv)
- f is 0 outside the range (when child j = 0)

Actually, the simplest approach might be:

For each j : Fin d with child j = 0, f(j) = 0 (whether the if-condition is true or false: if true, 2*delta1*child j = 2*delta1*0 = 0; if false, 0).

So ∑ j, f(j) = ∑ j with child j ≠ 0, f(j).

The map idx ↦ ⟨nz_list[idx], h_valid idx⟩ is an injection from Fin nz_count to {j : Fin d | child j ≠ 0} (by h_distinct), and it's surjective onto {j | child j ≠ 0} (by h_inv: if child j ≠ 0 then ∃ k, nz_list[k] = j).

So ∑ j with child j ≠ 0, f(j) = ∑ idx : Fin nz_count, f(nz_list[idx]).

Use Finset.sum_nbij or Fintype.sum_bijective or similar. The function is the injection idx ↦ ⟨nz_list ⟨idx.1, by omega⟩, h_valid idx⟩.
-/
theorem sparse_cross_term_eq_dense
    {d : ℕ} (child : Fin d → ℤ)
    (k1 k2 : Fin d) (hk : k2.1 = k1.1 + 1)
    (delta1 delta2 : ℤ)
    (nz_list : Fin d → ℕ) (nz_count : ℕ)
    (hnz_count : nz_count ≤ d)
    (h_valid : ∀ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ < d)
    (h_distinct : ∀ k₁ k₂ : Fin nz_count,
      nz_list ⟨k₁.1, by omega⟩ = nz_list ⟨k₂.1, by omega⟩ → k₁ = k₂)
    (h_inv : ∀ i : Fin d, child i ≠ 0 ↔
      ∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = i.1) :
    ∀ t : ℕ,
    -- k1 cross-term contribution: dense = sparse
    (∑ j : Fin d,
      if j ≠ k1 ∧ j ≠ k2 ∧ k1.1 + j.1 = t
      then 2 * delta1 * child j else 0) =
    (∑ idx : Fin nz_count,
      let j : Fin d := ⟨nz_list ⟨idx.1, by omega⟩, h_valid idx⟩
      if j ≠ k1 ∧ j ≠ k2 ∧ k1.1 + j.1 = t
      then 2 * delta1 * child j else 0)
    ∧
    -- k2 cross-term contribution: dense = sparse
    (∑ j : Fin d,
      if j ≠ k1 ∧ j ≠ k2 ∧ k2.1 + j.1 = t
      then 2 * delta2 * child j else 0) =
    (∑ idx : Fin nz_count,
      let j : Fin d := ⟨nz_list ⟨idx.1, by omega⟩, h_valid idx⟩
      if j ≠ k1 ∧ j ≠ k2 ∧ k2.1 + j.1 = t
      then 2 * delta2 * child j else 0) := by
  intro t
  generalize_proofs at *; (
  constructor <;> rw [ ← Finset.sum_subset ( Finset.subset_univ ( Finset.image ( fun k : Fin nz_count => ⟨ nz_list ⟨ k, by linarith [ Fin.is_lt k ] ⟩, h_valid k ⟩ : Fin nz_count → Fin d ) Finset.univ ) ) ];
  · rw [ Finset.sum_image ];
    exact fun a _ b _ hab => h_distinct a b <| by simpa [ Fin.ext_iff ] using hab;
  · intro x hx hx'; specialize h_inv x; contrapose! hx'; aesop;
  · rw [ Finset.sum_image ];
    exact fun a _ b _ hab => h_distinct a b <| by simpa [ Fin.ext_iff ] using hab;
  · intro x hx hx'; specialize h_inv x; contrapose! hx'; aesop)

/-
PROBLEM
For each component, the proof has two steps:
Step 1 (zero-filtering): ∑_{j : Fin d} f(j) = ∑_{j : Fin d, child j ≠ 0} f(j)
because child j = 0 ⟹ f(j) = 2·δ·child j = 2·δ·0 = 0.
Step 2 (bijection): ∑_{j ∈ nonzero set} f(j) = ∑_{idx : Fin nz_count} f(nz_list[idx])
because h_inv + h_distinct + h_valid give a bijection between
Fin nz_count and {j : Fin d | child j ≠ 0}.

Claim 4.32: The raw_conv array after the sparse cross-term update
    is identical to the raw_conv array after the dense cross-term update.

    Both paths start from the same intermediate state raw_conv_after_self
    (after applying self-terms and mutual term, which are identical —
    run_cascade.py:1296-1300). The only difference is the cross-term
    computation. By Claim 4.31, the cross-term deltas are equal, so the
    final raw_conv arrays are equal.

    This is the master equivalence theorem: it guarantees that the
    pruning test (which reads raw_conv) sees identical values regardless
    of whether sparse or dense cross-terms were used.

    Depends on: Claim 4.31 (sparse_cross_term_eq_dense),
                IncrementalAutoconv.delta_three_way_split (S = {k1,k2}).

PROVIDED SOLUTION
Apply funext to introduce t : Fin (2 * d - 1). Rewrite using h_dense t and h_sparse t. Then use sparse_cross_term_eq_dense to show the cross-term sums are equal.

Specifically:
1. funext t
2. rw [h_dense t, h_sparse t]  -- Now the goal is:
   raw_conv_after_self t + (dense k1 sum) + (dense k2 sum) = raw_conv_after_self t + (sparse k1 sum) + (sparse k2 sum)
3. obtain ⟨h₁, h₂⟩ := sparse_cross_term_eq_dense child k1 k2 hk delta1 delta2 nz_list nz_count hnz_count h_valid h_distinct h_inv t.1
4. rw [h₁, h₂]  -- or congr and use h₁ and h₂
-/
theorem raw_conv_sparse_eq_dense
    {d : ℕ} (child : Fin d → ℤ)
    (k1 k2 : Fin d) (hk : k2.1 = k1.1 + 1)
    (delta1 delta2 : ℤ)
    (nz_list : Fin d → ℕ) (nz_count : ℕ)
    (hnz_count : nz_count ≤ d)
    (h_valid : ∀ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ < d)
    (h_distinct : ∀ k₁ k₂ : Fin nz_count,
      nz_list ⟨k₁.1, by omega⟩ = nz_list ⟨k₂.1, by omega⟩ → k₁ = k₂)
    (h_inv : ∀ i : Fin d, child i ≠ 0 ↔
      ∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = i.1)
    -- Shared intermediate state (after self-terms + mutual term)
    (raw_conv_after_self : Fin (2 * d - 1) → ℤ)
    -- Dense cross-term update: iterate all j ≠ k1, k2
    (raw_conv_dense : Fin (2 * d - 1) → ℤ)
    (h_dense : ∀ t : Fin (2 * d - 1), raw_conv_dense t = raw_conv_after_self t
      + (∑ j : Fin d, if j ≠ k1 ∧ j ≠ k2 ∧ k1.1 + j.1 = t.1
          then 2 * delta1 * child j else 0)
      + (∑ j : Fin d, if j ≠ k1 ∧ j ≠ k2 ∧ k2.1 + j.1 = t.1
          then 2 * delta2 * child j else 0))
    -- Sparse cross-term update: iterate nz_list entries ≠ k1, k2
    (raw_conv_sparse : Fin (2 * d - 1) → ℤ)
    (h_sparse : ∀ t : Fin (2 * d - 1), raw_conv_sparse t = raw_conv_after_self t
      + (∑ idx : Fin nz_count,
          let j : Fin d := ⟨nz_list ⟨idx.1, by omega⟩, h_valid idx⟩
          if j ≠ k1 ∧ j ≠ k2 ∧ k1.1 + j.1 = t.1
          then 2 * delta1 * child j else 0)
      + (∑ idx : Fin nz_count,
          let j : Fin d := ⟨nz_list ⟨idx.1, by omega⟩, h_valid idx⟩
          if j ≠ k1 ∧ j ≠ k2 ∧ k2.1 + j.1 = t.1
          then 2 * delta2 * child j else 0)) :
    raw_conv_dense = raw_conv_sparse := by
  -- Apply the equality from Claim 4.31 to each term in the sums.
  have h_sum_eq : ∀ t : Fin (2 * d - 1), (∑ j : Fin d, if j ≠ k1 ∧ j ≠ k2 ∧ k1.1 + j.1 = t.1 then 2 * delta1 * child j else 0) = (∑ idx : Fin nz_count, let j : Fin d := ⟨nz_list ⟨idx.1, by omega⟩, h_valid idx⟩; if j ≠ k1 ∧ j ≠ k2 ∧ k1.1 + j.1 = t.1 then 2 * delta1 * child j else 0) ∧ (∑ j : Fin d, if j ≠ k1 ∧ j ≠ k2 ∧ k2.1 + j.1 = t.1 then 2 * delta2 * child j else 0) = (∑ idx : Fin nz_count, let j : Fin d := ⟨nz_list ⟨idx.1, by omega⟩, h_valid idx⟩; if j ≠ k1 ∧ j ≠ k2 ∧ k2.1 + j.1 = t.1 then 2 * delta2 * child j else 0) := by
    intro t
    exact sparse_cross_term_eq_dense child k1 k2 hk delta1 delta2
      nz_list nz_count hnz_count h_valid h_distinct h_inv t.1
  exact funext fun t => by rw [ h_dense, h_sparse, h_sum_eq t |>.1, h_sum_eq t |>.2 ] ;

-- funext t; rw [h_dense, h_sparse];
  -- obtain ⟨h₁, h₂⟩ := sparse_cross_term_eq_dense ... t
  -- rw [h₁, h₂]

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART D: Subtree Prune Rebuild (Claim 4.33)
--
-- After a subtree prune, child bins are reset and raw_conv is fully
-- recomputed. The nz_list must be rebuilt from scratch. We must prove
-- that the rebuild produces a valid nz_list for the new child state.
--
-- Code reference: run_cascade.py:1478-1487
--   nz_count = 0
--   for ii in range(d_child):
--       if child[ii] != 0:
--           nz_list[nz_count] = ii; nz_pos[ii] = nz_count; nz_count += 1
--       else:
--           nz_pos[ii] = -1
--
-- This is identical to the initial nz_list construction (lines 1091-1096),
-- so Claim 4.33 also covers initialization correctness.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.33: Rebuilding nz_list from scratch by iterating all d_child
    bins and collecting nonzero indices produces a valid nz_list satisfying
    the invariant of Claim 4.26.

    This covers both the post-subtree-prune rebuild AND the initial
    construction (same procedure). The proof is identical to Claim 4.26
    instantiated with the rebuild postconditions. -/
theorem rebuild_nz_list_correct
    {d : ℕ} (child : Fin d → ℤ)
    (nz_list : Fin d → ℕ) (nz_count : ℕ)
    (hnz_count : nz_count ≤ d)
    (h_valid : ∀ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ < d)
    (_h_distinct : ∀ k₁ k₂ : Fin nz_count,
      nz_list ⟨k₁.1, by omega⟩ = nz_list ⟨k₂.1, by omega⟩ → k₁ = k₂)
    (h_nonzero : ∀ k : Fin nz_count,
      child ⟨nz_list ⟨k.1, by omega⟩, h_valid k⟩ ≠ 0)
    (h_complete : ∀ i : Fin d, child i ≠ 0 →
      ∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = i.1) :
    -- The biconditional invariant holds
    ∀ i : Fin d, child i ≠ 0 ↔
      ∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = i.1 := by
  intro i
  constructor
  · exact h_complete i
  · rintro ⟨k, hk⟩
    have heq : (⟨nz_list ⟨k.1, by omega⟩, h_valid k⟩ : Fin d) = i := Fin.ext hk
    rw [← heq]
    exact h_nonzero k

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART E: Gating Correctness (Claim 4.34)
--
-- The optimization is gated on d_child ≥ 32. We must prove that both
-- code paths (sparse and dense) produce identical results, so the gate
-- only affects performance, not correctness.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.34: The use_sparse gate does not affect the survivor set.

    For d_child < 32, the original dense cross-term loop is used.
    For d_child ≥ 32, the sparse cross-term loop is used.
    By Claim 4.32, both produce identical raw_conv arrays, so the
    pruning test produces identical results, and the survivor set
    is the same in both cases.

    This theorem states that the gate is purely a performance decision
    with no effect on the mathematical output. -/
theorem sparse_gate_correctness
    {d : ℕ} (_child : Fin d → ℤ)
    (raw_conv_dense raw_conv_sparse : Fin (2 * d - 1) → ℤ)
    (h_eq : raw_conv_dense = raw_conv_sparse)
    -- Same pruning test applied to both
    (pruned : (Fin (2 * d - 1) → ℤ) → Prop)
    (h_deterministic : ∀ r₁ r₂ : Fin (2 * d - 1) → ℤ,
      r₁ = r₂ → pruned r₁ = pruned r₂) :
    pruned raw_conv_dense = pruned raw_conv_sparse := by
  exact h_deterministic _ _ h_eq

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART F: End-to-End Soundness (Claim 4.35)
--
-- The final theorem: the Gray code kernel with sparse cross-term
-- optimization produces the identical set of canonical survivors as
-- the Gray code kernel without it.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.35 (Master Soundness Theorem): For any parent composition,
    the set of canonical survivors produced by the Gray code kernel with
    sparse cross-term optimization is identical to the set produced by
    the Gray code kernel without sparse optimization.

    Proof sketch:
    1. Both kernels enumerate the same Cartesian product of children
       (the Gray code traversal is unchanged — Claims 4.9, 4.22).
    2. For each child, the incremental autoconvolution update produces
       identical raw_conv arrays (Claim 4.32):
       - Self-terms and mutual term: computed identically (before the branch).
       - Cross-terms: identical by Claims 4.26, 4.31.
       - After subtree prune rebuild: invariant restored (Claim 4.33).
    3. The pruning test reads only raw_conv and child, both identical,
       so pruning decisions are identical (Claim 4.34).
    4. The quick-check, canonicalization, and survivor storage are
       unchanged, so the output sets are identical.

    The hypotheses formalize this chain: both survivor sets are defined
    as the same enumeration filtered by the same test, and the test
    produces the same result for each child because raw_conv is identical.

    Depends on: GrayCode (4.9), IncrementalAutoconv (4.2),
                GrayCodeSubtreePruning (4.22), Claims 4.26–4.34. -/
theorem sparse_cross_term_sound
    {d_parent : ℕ} (_parent : Fin d_parent → ℕ)
    (_lo _hi : Fin d_parent → ℕ)
    (_m : ℕ) (_c_target : ℝ) (_n_half_child : ℕ)
    -- The Cartesian product of all children of this parent
    (children : Finset (Fin (2 * d_parent) → ℕ))
    -- Pruning decision: does this child survive? (depends on raw_conv)
    (survives_dense survives_sparse : (Fin (2 * d_parent) → ℕ) → Prop)
    -- Key: both paths make the same decision for each child.
    -- Justified by Claim 4.32 (raw_conv identical) + Claim 4.34 (gate).
    (h_same : ∀ child ∈ children, (survives_sparse child ↔ survives_dense child))
    -- Survivor sets defined by membership
    (S_sparse S_dense : Finset (Fin (2 * d_parent) → ℕ))
    (h_S_dense : ∀ c, c ∈ S_dense ↔ c ∈ children ∧ survives_dense c)
    (h_S_sparse : ∀ c, c ∈ S_sparse ↔ c ∈ children ∧ survives_sparse c) :
    S_sparse = S_dense := by
  ext c
  simp only [h_S_sparse, h_S_dense]
  exact ⟨fun ⟨hc, hp⟩ => ⟨hc, (h_same c hc).mp hp⟩,
         fun ⟨hc, hp⟩ => ⟨hc, (h_same c hc).mpr hp⟩⟩

end