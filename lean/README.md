# Sidon Autocorrelation: Lean 4 Formalisation

A Lean 4 / Mathlib formalisation of the **Piterbarg-Bajaj-Vincent
Bound**, the rigorous lower bound

```
    C_{1a} = inf { ‖f * f‖_∞ / (∫ f)² :
                   f >= 0, supp(f) ⊆ [-1/4, 1/4], 0 < ∫ f < ∞ }
           ≥ 1292/1000  =  1.292,
```

established in *Improving the Bounds on the Supremum of
Autoconvolutions*.  The proof builds a 3-scale arcsine kernel and
applies the Matolcsi–Vinuesa (2010) master inequality.  All numerical
anchors are discharged by a `flint.arb` certifier at 256-bit precision;
the Python implementation lives in
`../delsarte_dual/grid_bound_alt_kernel/`.

The formalisation has **no `sorry`**.  Exactly **one** user axiom is
declared in `Sidon/MultiScale.lean` and appears in the dependency
closure of the headline theorem; the algebraic inversion and the five
slack-soundness statements are Lean theorems (see "Axiom inventory"
below).

## Build

```bash
lake build                       # default target: Sidon (the proof chain)
lake env lean AxiomCheck.lean    # print axiom dependencies of the headline
```

`lake build` should report `Build completed successfully` with no `sorry`
warnings.

## Layout

```
Sidon.lean                root, re-exports the proof
Sidon/Defs.lean           shared definitions (autoconvolution_ratio, ...)
Sidon/MultiScale.lean     headline theorem and one user axiom
AxiomCheck.lean           axiom-inventory printer
lakefile.lean             Lake build configuration
lake-manifest.json        Mathlib dependency lock
lean-toolchain            pinned Lean toolchain
```

Earlier exploratory modules (legacy monolithic proof, Algorithm /
CoarseCascade drafts, single-scale and 2-scale Cascade variants,
alternative-kernel directions) have been moved to `../archive/lean/`.

## Headline theorem

```lean
theorem Sidon.MultiScale.C1a_ge_1292
    (f : ℝ → ℝ)
    (hf_nonneg  : ∀ x, 0 ≤ f x)
    (hf_supp    : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm
        (MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
        ⊤ MeasureTheory.volume ≠ ⊤) :
    (1292 : ℝ) / 1000 ≤ autoconvolution_ratio f
```

Equivalent restatements `autoconvolution_ratio_ge_1292_1000` and
`autoconvolution_ratio_ge_1_292` are exported from the same namespace.

## Axiom inventory

`AxiomCheck.lean` prints the axiom closure of the headline theorem.
Beyond Lean's three core axioms (`Classical.choice`, `propext`,
`Quot.sound`) the proof reaches exactly one user axiom, declared in
`Sidon/MultiScale.lean`:

| Axiom                                       | Content                                                              |
|---------------------------------------------|----------------------------------------------------------------------|
| `MV_master_inequality_for_extremiser`       | 3-scale Fourier reduction on `ℝ/uℤ` (MV 2010, Lemma 3.1) at the slack rationals `K2UpperQ`, `gainLowerQ` |

The substitution of the rational slacks `K2UpperQ = 47897/10000` for
`K_2` and `gainLowerQ = 20925/100000` for `a` is justified by the
certifier's bounds on the analytic functionals (paper Lemmas 4.1-4.5),
which are discharged externally by `flint.arb` at 256-bit precision in
`../delsarte_dual/grid_bound_alt_kernel/`.

The algebraic inversion is a Lean *theorem*:

| Theorem                                     | Content                                                              |
|---------------------------------------------|----------------------------------------------------------------------|
| `master_inequality_M_lower`                 | quadratic-in-`M` inversion of the master inequality at the anchors   |

Five further slack-soundness statements are also Lean *theorems*,
recording that the rational slacks chosen above are on the correct side
of the certifier-reported decimals (paper Lemmas 4.1-4.5):

| Theorem                                     | Content                                                              |
|---------------------------------------------|----------------------------------------------------------------------|
| `K_two_upper_bound`                         | `K2UpperQ ≥ 4788906/1000000`                                         |
| `k_one_lower_bound`                         | `K1LowerQ ≤ 92124658/100000000`                                      |
| `S_one_upper_bound`                         | `S1UpperQ ≥ 2984091/100000`                                          |
| `min_G_lower_bound`                         | `minGLowerQ ≤ 9999798/10000000`                                      |
| `gain_lower_bound`                          | `gainLowerQ ≤ 21009214/100000000`                                    |

## Construction

3-scale arcsine kernel

```
    K(x)  =  λ₁ K_arc(δ₁)(x) + λ₂ K_arc(δ₂)(x) + λ₃ K_arc(δ₃)(x),
    K̂(ξ)  =  λ₁ J₀(πδ₁ξ)² + λ₂ J₀(πδ₂ξ)² + λ₃ J₀(πδ₃ξ)²,
```

at the rational anchors

```
    δ₁ = 138/1000,    λ₁ = 85/100,
    δ₂ =  55/1000,    λ₂ = 10/100,
    δ₃ =  25/1000,    λ₃ =  5/100,
    u  = 638/1000  (= 1/2 + δ₁),
```

with a 200-coefficient cosine `G` re-optimised at this kernel.  Bochner
admissibility of `K̂` is automatic: each `J₀(πδᵢξ)²` is the square of a
real Bessel function, and a convex combination preserves
positive-semi-definiteness.

## References

* Matolcsi, M., Vinuesa, C.  *Improved bounds on the supremum of
  autoconvolutions.*  J. Math. Anal. Appl. **372** (2010), 439-447,
  arXiv:0907.1379.
* Cloninger, A., Steinerberger, S.  *On suprema of autoconvolutions
  with an application to Sidon sets.*  Proc. Amer. Math. Soc.
  **145** (2017), 3191-3200, arXiv:1403.7988.
* Cohn, H., Elkies, N.  *New upper bounds on sphere packings I.*
  Annals of Mathematics (2) 157 (2003), 689–714.
