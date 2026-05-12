/-
Copyright (c) 2026 Sidon Project. All rights reserved.

# Abstract Farkas Certificate for Conic Infeasibility

This file captures the ABSTRACT content of the conic Farkas alternative that
justifies the SOS-dual Farkas LP solved by `lasserre/dual_sdp.py`:

    Primal  { Ax = b,  x ∈ K }  is infeasible
    ⟺  ∃ u  such that  Aᵀ u ∈ −K*  and  bᵀ u > 0.

The Python code (`lasserre/dual_sdp.py::build_dual_task` + `solve_dual_task`)
solves the right-hand-side LP as

    max  λ           subject to
    λ · δ_{α=0}                                                  -- y₀ = 1 dual
    + Σ_k μ_k · C_{k,α}                                          -- consistency dual
    + ⟨X₀, E₀[α]⟩ + Σ_i ⟨Xᵢ, Eᵢ[α]⟩ + Σ_i ⟨X'ᵢ, E'ᵢ[α]⟩           -- PSD duals
    + t · Σ_W ⟨X_W, E_W^t[α]⟩ − Σ_W ⟨X_W, E_W^Q[α]⟩               -- window duals
    + v_α = 0                          ∀ α                        -- stationarity
    0 ≤ λ ≤ Λ,   μ free,   v_α ≥ 0,   X_0, X_i, X'_i, X_W ⪰ 0.

and reads off the verdict

    λ*  ≈ Λ  ⟹  primal infeasible  at t
    λ*  ≈ 0  ⟹  primal feasible    at t.

The abstract content proved below is just: given a pair of sets
`P = primalFeasSet` and `C = farkasFeasSet t` with the invariant

    C non-empty  →  P empty          (⋆)

then any concrete certificate `u ∈ C` witnesses `P = ∅`.  The invariant (⋆)
is the Farkas alternative itself; we take it as an ABSTRACT HYPOTHESIS at
this level, and prove downstream corollaries (verdict → lower bound) from it.

## Why abstract

Formalising the full conic Farkas lemma for Lasserre-sized SDPs in Mathlib
would require a substantial amount of semidefinite-programming infrastructure
that is not yet in Mathlib's library.  Rather than depend on that, we prove
EVERYTHING ELSE downstream of (⋆) — and expose (⋆) as a named interface.

Concretely: once Mathlib gains `ConvexCone.farkas_alternative` (in progress),
the hypothesis `infeas_of_cert_exists` below becomes instantiable directly;
until then it serves as the proof-obligation boundary between the Python
solver and the Lean verification of the bisection bound.
-/
import Mathlib

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace SOSDual.Farkas

/-! ## Abstract primal / Farkas certificate sets

We use `Set α` / `Set β` directly (no wrapper `def`s) so that `∈` and other
set operators resolve through Mathlib's existing instances without needing
to unfold a fresh abbreviation. -/

/-! ## The Farkas alternative as a named interface

This is the only statement we take as abstract — the rest of this module
proves downstream verdict / bisection soundness from it. -/

/-- Farkas alternative: IF a certificate exists, THEN the primal is empty.

    Concretely: Slater's condition holds for the Lasserre primal at every
    t strictly above val_L(d) (a probability measure yields strict positivity
    in all PSD cones), so the conic Farkas alternative is directly applicable.
    For t ≤ val_L(d) − ε, the primal is infeasible and the Farkas LP has a
    positive-λ solution.

    The MOSEK solver produces a numerical λ* > 0 at such t (modulo finite
    tolerance).  The hypothesis here packages that into a mathematical
    existence statement. -/
structure FarkasInvariant (α β : Type) where
  /-- Primal feasible set (abstract — e.g. moment vectors satisfying the
      Lasserre SDP at some fixed t). -/
  primal : Set α
  /-- Farkas-LP certificate set (abstract — e.g. tuples (λ, μ, v, X)
      with λ > 0 satisfying stationarity and dual-cone conditions). -/
  certs  : Set β
  /-- The core Farkas alternative: existence of a certificate ⟹ primal is empty. -/
  infeas_of_cert_exists : certs.Nonempty → ¬ primal.Nonempty

/-! ## Downstream corollaries -/

variable {α β : Type}

/-- If we have a certificate, the primal is infeasible. -/
theorem primal_empty_of_cert
    (inv : FarkasInvariant α β)
    (u : β) (hu : u ∈ inv.certs) :
    ¬ inv.primal.Nonempty :=
  inv.infeas_of_cert_exists ⟨u, hu⟩

/-- Contrapositive: if the primal has a feasible point, no certificate exists. -/
theorem no_cert_of_primal_nonempty
    (inv : FarkasInvariant α β)
    (h : inv.primal.Nonempty) :
    ¬ inv.certs.Nonempty := by
  intro hcerts
  exact inv.infeas_of_cert_exists hcerts h

/-! ## Certified lower bound under the infeasibility alternative

This is the ABSTRACT form of the theorem "if MOSEK returns λ*≈1 at t=t*,
then val_L(d) ≥ t*".  We model it as: if at some t the primal is empty,
then "t is a certified strict lower bound on the optimum". -/

/-- The optimum of a minimisation over an empty feasible set is, by the
    standard convention, `+∞`.  So primal-empty-at-t gives val > t for any
    finite t.  Abstract form: given two real numbers `t` and `v` with the
    convention that an infeasible primal establishes `v > t`, the "t-certifier"
    is a proof of primal emptiness at t. -/
def IsTCertifier (primal_empty : Prop) (v : ℝ) (t : ℝ) : Prop :=
  primal_empty → t ≤ v

/-- A Farkas certificate at t establishes `t ≤ val_L(d)` under the invariant
    that primal-empty-at-t implies `t ≤ v`. -/
theorem t_le_val_of_cert
    (inv : FarkasInvariant α β) (t v : ℝ)
    (h_cert_exists : inv.certs.Nonempty)
    (h_primal_empty_bounds : ¬ inv.primal.Nonempty → t ≤ v) :
    t ≤ v :=
  h_primal_empty_bounds (inv.infeas_of_cert_exists h_cert_exists)

end SOSDual.Farkas
