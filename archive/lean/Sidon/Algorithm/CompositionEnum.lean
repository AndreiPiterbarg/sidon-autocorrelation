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
-- Composition Enumeration (Claims 3.1, 3.2a)
-- Source: 31103b4c-cf4c-4f19-abf6-fe75cd7e9ee4-output.lean
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 3.1: Stars-and-bars — compositions of m into d parts = C(m+d-1, d-1). -/
theorem composition_count (m d : ℕ) (hd : d > 0) :
    Finset.card (Finset.filter (fun c : Fin d → Fin (m + 1) =>
      ∑ i, (c i : ℕ) = m) Finset.univ) = Nat.choose (m + d - 1) (d - 1) := by
  have h_stars_and_bars : ∀ m d : ℕ, d > 0 → Finset.card (Finset.filter (fun (c : Fin d → ℕ) => (∑ i, c i) = m) (Finset.Iic (fun _ => m))) = Nat.choose (m + d - 1) (d - 1) := by
    intro m d hd
    induction' d with d ih generalizing m;
    · contradiction;
    · have h_split : Finset.filter (fun (c : Fin (d + 1) → ℕ) => (∑ i, c i) = m) (Finset.Iic (fun _ => m)) = Finset.biUnion (Finset.range (m + 1)) (fun k => Finset.image (fun (c : Fin d → ℕ) => Fin.cons k c) (Finset.filter (fun (c : Fin d → ℕ) => (∑ i, c i) = m - k) (Finset.Iic (fun _ => m - k)))) := by
        ext c; simp [Finset.mem_biUnion, Finset.mem_image];
        constructor <;> intro h;
        · refine' ⟨ c 0, _, Fin.tail c, _, _ ⟩ <;> simp_all +decide [ Fin.sum_univ_succ ];
          · linarith [ h.1 0, Nat.zero_le ( ∑ i : Fin d, c i.succ ) ];
          · exact ⟨ fun i => Nat.le_sub_of_add_le <| by linarith! [ h.1 i.succ, Finset.single_le_sum ( fun a _ => Nat.zero_le ( c ( Fin.succ a ) ) ) ( Finset.mem_univ i ) ], eq_tsub_of_add_eq <| by linarith! ⟩;
        · rcases h with ⟨ a, ha, b, ⟨ hb₁, hb₂ ⟩, rfl ⟩ ; simp_all +decide [ Fin.sum_univ_succ ];
          exact ⟨ fun i => by cases i using Fin.inductionOn <;> [ exact Nat.le_of_lt_succ ha; exact le_trans ( hb₁ _ ) ( Nat.sub_le _ _ ) ], Nat.add_sub_of_le ( Nat.le_of_lt_succ ha ) ⟩;
      rw [ h_split, Finset.card_biUnion ];
      · rcases d with ( _ | d ) <;> simp_all +decide [ Finset.card_image_of_injective, Function.Injective ];
        · rw [ Finset.sum_eq_single m ] <;> simp +decide;
          intros; omega;
        · exact Nat.recOn m ( by simp +arith +decide ) fun n ih => by simp +arith +decide [ Nat.choose, Finset.sum_range_succ' ] at * ; linarith;
      · intro k hk l hl hkl; simp_all +decide [ Finset.disjoint_left ];
        intro a x hx₁ hx₂ hx₃ y hy₁ hy₂ hy₃; contrapose! hkl; aesop;
  convert h_stars_and_bars m d hd using 1;
  refine' Finset.card_bij ( fun c hc => fun i => c i ) _ _ _ <;> simp +decide [ funext_iff ];
  · exact fun a ha => ⟨ fun i => Nat.le_of_lt_succ <| Fin.is_lt _, ha ⟩;
  · exact fun a₁ ha₁ a₂ ha₂ h x => Fin.ext <| h x;
  · exact fun b hb hm => ⟨ fun i => ⟨ b i, Nat.lt_succ_of_le ( hb i ) ⟩, hm, fun i => rfl ⟩

/-- Claim 3.2a: Per-bin choice count for child generation. -/
theorem per_bin_choices (c_i x_cap : ℕ) (h : c_i ≤ 2 * x_cap) :
    Finset.card (Finset.Icc (Nat.max 0 (c_i - x_cap)) (Nat.min c_i x_cap)) =
    Nat.min c_i x_cap - Nat.max 0 (c_i - x_cap) + 1 := by
  simp +zetaDelta at *;
  grind +ring

end -- noncomputable section
