# Computer-verified pathway for the Sidon autoconvolution constant

Date: 2026-04-20
Scope: Publication viability of a rigorous, machine-checkable lower bound on
$C_{1a}$ via our existing `lean/` + `certified_lasserre/` stack, even if the numeric
value does not exceed the current 1.2802 benchmark.

---

## 1. State of the art in computer-verified extremal constants

1. **Flyspeck / Kepler** (Hales et al.; arXiv:1501.02155, Forum of Math Pi 2017).
   HOL Light + Isabelle formalization of ~1000 nonlinear inequalities plus
   thousands of LPs. Established the template: heavy numerical search +
   exact-arithmetic LP/SDP certificate + proof-assistant wrapper.
2. **Schur S(5) = 160** (Heule 2017). SAT/ACL2 formally verified combinatorial
   extremal constant. Published in a mainline journal despite being entirely
   computational.
3. **Rational/exact SOS certificates.** Powers, Kaltofen-Li-Yang-Zhi, and
   Davis-Papp (SIOPT 2021, "Dual Certificates and Efficient Rational SOS
   Decompositions over Compact Sets") give algorithms that turn numerical SDP
   outputs into exact rational sum-of-squares certificates. Davis-Papp is the
   current best-practice reference: dual certificates, no rounding/projection.
4. **Certified SDP / SOS verification.** Roux-Voronin-Sankaranarayanan
   (SAS 2016), Magron-Safey El Din-Schweighofer (ACM TOMS 2017) show how to
   post-process SDP solver outputs into Coq-checkable bounds with Arb/MPFI.
5. **Cohn-de Laat-Salmon** (2022, arXiv:2206.09876): LP duality for Cohn-Elkies
   sphere packing has no duality gap; this is the same structural fact we use
   for Lasserre val(d) duality. They do not formalize in Lean, leaving the gap.
6. **Autoconvolution state (Aug 2025).** arXiv:2508.02803 ("Further
   Improvements...") pushes the *dual* autoconvolution bound to 0.94136 with
   2399-interval step functions; uses simulated annealing + gradient, no formal
   verification. Boyer-Li arXiv:2506.16750 (cited in our CLAUDE.md) is likewise
   informal. **No Lean/Coq/Isabelle formalization of any Sidon or
   autoconvolution constant exists.**

## 2. What is already in-repo we can leverage

- `certified_lasserre/farkas_certify.py`, `safe_certify_flint.py`: exact rational
  Farkas-style certificates for val(d) LBs, using python-flint ball/rational
  arithmetic. MEMORY notes val(4) > 1.0963 already certified.
- `lean/lasserre/`: 15 Lean 4 files covering soundness chain, PSD-submatrix,
  simplex moments, clique sparsity, bridge lemmas. Scaffolding exists; the
  numerical certificate check is the missing external input.
- `lean/Sidon/Proof/Proof.lean`: statement of the autocorrelation inequality.
- `interval_bnb/` Phase B: rigorous integer-arithmetic branch-and-bound on
  discretized mass distributions.

The pipeline is: Python (`certified_lasserre`) emits a rational certificate
(PSD Gram matrix + Lagrange multipliers as exact rationals, or a Farkas
infeasibility vector). Lean checks linear algebra identities over `Rat` and
concludes val(d) >= lambda via `ProofSoundnessChain.lean`.

## 3. Proposed verification pathway (minimum publishable unit)

### Phase V1. Lean-certified val(d) lower bound, d small

Target: a Lean 4 theorem of the form

    theorem valLB_d8 : valSidon 8 >= (10963 : Rat) / 10000

with a machine-checkable proof closed by `#print axioms` showing only
`propext, Classical.choice, Quot.sound`.

Components:

1. **Exact Gram extraction.** Run `certified_lasserre/safe_certify_flint.py`
   with MOSEK/Clarabel at high precision; project onto the PSD cone over `Rat`
   via Davis-Papp dual-certificate construction (SIOPT 2021). Emit Gram matrix
   $G \in \mathbb{Q}^{N \times N}$ plus moment-constraint multipliers in `Rat`.
2. **Lean ingestion.** Serialize $G$ as a Lean `def` of type `Matrix (Fin N)
   (Fin N) Rat`. Prove PSD via Cholesky over `Rat` (entries whose squares sum
   to diagonal) — mathlib has `Matrix.PosSemidef`.
3. **Algebraic identity.** Lean-check that $p(x) - \lambda = \sigma_0(x) +
   \sum_i \sigma_i(x) g_i(x)$ as a polynomial identity in
   `MvPolynomial (Fin d) Rat`. Pure equation checking; no real analysis.
4. **Bridge to val(d).** Use existing `ProofSoundnessChain.lean` +
   `ProofSimplexMoments.lean` to conclude
   `forall mu in simplex d, max W (mu^T M_W mu) >= lambda`.

Estimate: 6-8 weeks of focused Lean work. The mathlib prerequisites
(`MvPolynomial`, `Matrix.PosSemidef`, `Simplex`) all exist.

### Phase V2. Interval / Cloninger-Steinerberger rigorous verification

Target: a Lean theorem certifying that the discrete cascade prover's output
at resolution $S$ rigorously excludes $f$ with autocorrelation $< \lambda$.

Re-use `interval_bnb/`'s exact rational arithmetic, export the pruning tree
as a Lean term (each node a finite record), and discharge the leaf
infeasibilities by `decide` over `Rat` linear inequalities. This is the
SAT/Schur-style proof structure. Farkas-certified leaves from
`certified_lasserre/farkas_cg.py` plug in directly.

### Phase V3. Transfer discretized bound to the continuous constant

The load-bearing lemma (already sketched in `cloninger-steinerberger/` and
`mv_lemma217.py`) is the rigorous transfer

    val_discrete(d; S) >= lambda  =>  C_{1a} >= lambda - O(1/S)

with an explicit, computable O(1/S) constant. Formalize Lemma 2.17 of
Matolcsi-Vinuesa in Lean via `mathlib`'s `MeasureTheory.Convolution` and the
fine-grid correction already debugged in `project_fine_grid_fix.md`.

## 4. Publication assessment, even if lambda < 1.2802

### Why this is publishable as-is

1. **First formal verification of any Sidon/autoconvolution constant.** None
   exist. A Lean-certified lower bound of, say, 1.26 would be the first
   machine-checked constant in this problem family, which is itself a result.
   Compare: Schur S(5) was accepted because of formalization, not novelty of
   value.
2. **Methodology contribution.** Davis-Papp rational SOS + clique-sparse
   Lasserre + Lean has not been demonstrated end-to-end in the literature.
   This is a reusable template for any polynomial-optimization extremal
   constant (Delsarte LP bounds, Cohn-Elkies, energy minimization).
3. **Closes a verification gap.** Cohn-de Laat-Salmon proved LP duality but
   did not formalize; Boyer-Li and 2508.02803 improve numerics without
   certification. A rigorous Lean 4 bound removes the "informal numerics"
   caveat that all current results carry.
4. **Venue fit.** Journal of Automated Reasoning, ITP, CPP, Experimental
   Mathematics, and Mathematics of Computation all publish
   formalization-as-contribution papers. A Notices article in the style of
   Tao's "Machine assisted proof" (Feb 2024) is realistic as a companion.

### What NOT to claim

- Do not claim improvement of the state of the art bound unless lambda >
  1.2802. Position the paper as "first formal verification" + "scalable
  certified hierarchy", not "better constant".
- The Phase V3 transfer lemma's constant must be explicit and tight; otherwise
  reviewers will (correctly) note the gap from discrete to continuous is
  unconstrained.

## 5. Concrete next actions (2-3 week sprint)

1. Finalize `safe_certify_flint.py` to emit a JSON certificate: Gram matrix
   numerator/denominator, multipliers, and the polynomial identity witness.
2. Write `lean/lasserre/Certificate.lean` that imports the JSON (via
   `elab_command` or a generated `.lean` file) and proves `valLB_d4`.
3. Run `#print axioms valLB_d4` — target zero `sorryAx`.
4. Extend to d=6, d=8 once the pipeline closes at d=4.
5. Parallel track: formalize Matolcsi-Vinuesa Lemma 2.17 transfer in
   `lean/Sidon/Proof/Transfer.lean`.

---

## 100-word summary

No Sidon or autoconvolution constant has ever been formally verified; current
best bounds (Cloninger-Steinerberger 1.28, Boyer-Li 2506.16750, Schlage-Puchta
2508.02803) are informal numerics. Our repo already has the two pieces needed:
`certified_lasserre/` produces exact rational Farkas/SOS certificates
(val(4) > 1.0963 certified), and `lean/lasserre/` has the soundness-chain
scaffolding. Using Davis-Papp rational SOS (SIOPT 2021) as the bridge, we can
produce a Lean 4 theorem `valSidon d >= lambda` closed under `#print axioms`.
Even at lambda < 1.2802, this is the first machine-checked bound in the
problem family — publishable in JAR, ITP, Math. Comp., or Experimental Math.
