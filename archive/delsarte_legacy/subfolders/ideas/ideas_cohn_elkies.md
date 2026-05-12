# Cohn-Elkies LP Duality Transfer to C_{1a}

**Scope.** Research note investigating whether the Cohn-Elkies (CE) linear-programming framework — the engine behind sphere-packing bounds in dimensions 8, 24 (Viazovska 2017; CKMRV 2022) — can be transferred to produce a rigorous lower bound on the Sidon autocorrelation constant C_{1a}. This complements the Delsarte-style dual already implemented in `delsarte_dual/theory.md`.

---

## 0. Key references

- Cohn & Elkies, *New upper bounds on sphere packings I*, Annals of Math. 157 (2003), 689–714. arXiv:math/0110009.
- Cohn, *New upper bounds on sphere packings II*, Geom. & Topol. 6 (2002), 329–353.
- Viazovska, *The sphere packing problem in dimension 8*, Annals of Math. 185 (2017), 991–1015.
- Cohn, Kumar, Miller, Radchenko, Viazovska, *Universal optimality of the E8 and Leech lattices*, Annals of Math. 196 (2022).
- Carruth, Elkies, Gonçalves, Kelly, *The Beurling-Selberg box minorant problem via LP bounds*, arXiv:1702.04579 (2017).
- Cohn & Gonçalves, *An optimal uncertainty principle in twelve dimensions via modular forms*, Inventiones (2019).
- Li & Cohn, *Dual linear programming bounds for sphere packing via discrete reductions*, Adv. Math. (2024).
- Chandee & Soundararajan / Carneiro–Milinovich–Soundararajan, *Fourier optimization and prime gaps*, arXiv:1708.04122 — CE-style LP for analytic number theory extremal problems.
- Matolcsi & Vinuesa, *Improved bounds on the supremum of autoconvolutions*, arXiv:0907.1379 (used for the current 1.2802 lower bound).

## 1. The Cohn-Elkies primal LP (sphere packing)

Cohn-Elkies 2003, Thm 3.2. Let `f : R^n -> R` be Schwartz, not identically zero, with
- (C1) `f(x) <= 0` for `|x| >= r`,
- (C2) `f̂(ξ) >= 0` for all `ξ ∈ R^n`,
- (C3) `f(0) > 0, f̂(0) > 0`.

Then the sphere-packing density in `R^n` with minimum distance `r` satisfies

    Δ_n ≤ (f(0) / f̂(0)) · vol(B_{r/2}).

The derivation is Poisson summation over a packing lattice; positive-definiteness of `f̂` makes the dual sum nonneg, and `f <= 0` on pair-differences makes the primal sum ≤ `f(0)`. The **dual LP** (Cohn II, 2002 / Li-Cohn 2024) uses a positive measure `μ` on `R^n` and gives density lower bounds by certifying infeasibility via Lagrange multipliers. Strong duality holds under mild regularity (Li-Cohn 2024 prove zero duality gap for the discretized problem).

**Template for transfer.** Any extremal problem of the form

> *minimize a linear functional of a nonnegative measure subject to support / Fourier constraints*

admits a CE-style LP relaxation. The abstract ingredients are:
1. A **primal object** (sphere centers; autoconvolution mass distribution; prime-gap function) expressible as a nonnegative measure.
2. A **Fourier-side positivity** witness (positive-definite test function whose FT is explicitly nonneg).
3. A **support-side sign** witness (the test function sits below/above a target on a specified set).

## 2. Precise CE-LP formulation for C_{1a}

**Primal problem.** Define

    C_{1a} = inf { ||f*f||_∞ : f ≥ 0, supp f ⊂ [-1/4,1/4], ∫f = 1 }.

Equivalently, writing `ρ := f*f` with `supp ρ ⊂ [-1/2,1/2]`, `ρ ≥ 0`, `∫ρ = 1`, `ρ̂ = (f̂)^2 ≥ 0` (so `ρ` is **positive-definite**) and `ρ̂` is entire of exponential type `π` (Paley-Wiener). The problem becomes

    C_{1a} = inf_ρ ||ρ||_∞,  subject to   ρ ≥ 0 on R, supp ρ ⊂ [-1/2,1/2],
                                          ρ̂ ≥ 0, ρ̂(0) = 1, ρ̂ of exp. type π,
                                          ρ̂ = (f̂)^2 for some real f̂ ∈ PW_{π/2}.

The **square-root constraint** `ρ̂ = (f̂)^2` is what makes this *not* a pure LP — it is a cone program (the cone of squares of PW functions is non-polyhedral). Two relaxations are available.

### 2a. Weak relaxation: drop the square-root

Drop `ρ̂ = (f̂)^2` and keep only `ρ̂ ≥ 0, ρ̂(0)=1`, `ρ̂` entire type `π`. This is a linear program over the positive-definite cone (Bochner). The **CE-dual** becomes: find `g : R -> R` such that
- (A) `g(t) ≥ 0` for `t ∈ [-1/2, 1/2]`,
- (B) `ĝ(ξ) ≥ 0` for all `ξ ∈ R`,
- (C) `ĝ(0) > 0`.

Then
> **(CE-weak bound)** `||ρ||_∞ ≥ ĝ(0) / M_g`, where `M_g = max_{t ∈ [-1/2,1/2]} g(t)`.

This is **exactly the Delsarte-type dual already implemented in `delsarte_dual/theory.md` §2**. The CE framework recovers it as a special case; the weak relaxation throws away the square-root and the exponential-type constraint.

### 2b. Strong relaxation: keep `f̂ ∈ PW_{π/2}` with `|f̂| ≤ 1`

Keep the Paley-Wiener constraint `supp f ⊂ [-1/4,1/4]` (so `f̂` is entire of type `π/2`) and `|f̂(ξ)| ≤ f̂(0) = 1`. The Cohn-Elkies LP reformulation is:

> Find a test function `g : R -> R` with
>  (A') `g(t) ≥ 0` on `[-1/2, 1/2]`;
>  (B') `ĝ(ξ) ≥ 0` on `R`;
>  (D') `ĝ` is supported in `[-1, 1]` *or* dominated by a known lower envelope `w(ξ) ≤ |f̂(ξ)|^2`.
>
> Then `C_{1a} ≥ (∫ ĝ(ξ) w(ξ) dξ) / M_g`.

The sharp rigorous envelope (from `theory.md` §2) is

    w(ξ) = cos^2(π ξ / 2)  for |ξ| ≤ 1,    w(ξ) = 0 otherwise.

Using the weaker envelope `(1 - π|ξ|/2)_+^2` is strictly lossy. The CE-type primal gives the additional structural constraint that `f̂(ξ)^2` is a Fourier-square, which the Delsarte relaxation **ignores**. That is where extra room lives.

### 2c. The square-root constraint as a cone

The cone `K := { (f̂)^2 : f̂ ∈ PW_{π/2}, f ≥ 0, ∫f = 1 }` is *not* linear but *is* the intersection of:
- the positive-definite cone (Bochner side),
- the cone of PW_π functions (type constraint),
- the hyperplane ∫ = 1,
- a **Turán/Fejér-type** extra constraint from `f ≥ 0`.

The Fejér-Riesz theorem says a nonneg trigonometric polynomial is `|p|^2`; its continuous analogue (Krein) says a nonneg PW function of type `2T` equals `|h|^2` for `h ∈ PW_T`. So the cone `K` can be re-expressed as `{ h * h̃ : h ∈ PW_{π/2}, h ≥ 0 }` — still nonlinear but SDP-representable after discretization.

## 3. What must be positive-definite

In the CE bound for C_{1a}, positive-definiteness appears **twice**:

1. **Primal side (free / inherent).** The autoconvolution `ρ = f*f` is automatically positive-definite since `ρ̂ = f̂^2 ≥ 0` (when f real, hence f̂ hermitian, `f̂^2` real nonneg if `f̂` is real-even, else `|f̂|^2`). Equivalently, `ρ(t) = ∫ f(s) f(s+t) ds` is the autocorrelation of `f`, which is always PD. **This is the fundamental reason CE is a natural fit for C_{1a}.**

2. **Dual side (test function).** We choose `g` whose Fourier transform `ĝ ≥ 0`. This is the Bochner condition — `g` is a PD kernel on `[-1/2, 1/2]`.

**Clean restatement.** C_{1a} is the sharp constant in the inequality

    sup_{t ∈ [-1/2,1/2]} ρ(t) ≥ C_{1a} · ρ̂(0)

among PD, PW_π, nonneg-on-R, real-even measures ρ. This is **exactly** a CE-style extremal problem on the line.

## 4. Computable dual bound (proposed)

Concrete recipe, CE-strong variant:

1. Parametrize `g(t) = h(t)^2` with `h` a finite trigonometric/Gaussian/Vaaler combination (guarantees `g ≥ 0` globally, hence on `[-1/2,1/2]`).
2. Compute `ĝ = ĥ * ĥ^*` and enforce `ĝ ≥ 0` automatically (Wiener-Khinchin: autoconvolution of real ĥ has nonneg-cos Fourier-pair; actually `ĝ = |ĥ|^2` is automatic).
3. Maximize over parameters of `h`:

        L(h) = ( ∫_{-1}^{1} |ĥ(ξ)|^2 · cos^2(πξ/2) dξ ) / M_g.

4. Rigorize with mpmath interval arithmetic as in current `rigorous_max.py`.

**Why this might beat 1.2802.** The MV construction uses a finite-dim SDP with positive-definiteness on a grid. CE-strong, parametrized as `g = h^2`, enforces positivity *globally* by construction and removes the floor constraints that limit MV. Viazovska's dim-8 proof shows that the *right* modular/Laplace-transform parametrization of `h` can make the dual LP sharp.

**Warning.** Section 2 of our `theory.md` already implements (A)+(B) (this is the weak relaxation). The *additional* CE leverage here is the `|f̂|^2 ≥ cos^2(πξ/2)` envelope — which IS exploited. What remains unexploited is the **square-root cone** (§2c): we integrate `ĝ` against the weight but we do not yet constrain `ĝ` to be itself the Fourier transform of a square. Tightening to the square-root cone is the main theoretical upgrade available.

## 5. Feasibility assessment

| Aspect | Verdict |
|---|---|
| CE framework applies in principle? | **Yes** — C_{1a} is a Fourier extremal problem of exactly the CE type. `f*f` is auto-PD. |
| Delsarte (weak CE) already implemented? | **Yes** (`theory.md` §2). Hit a ceiling around 1.199 in `postmortem.md`. |
| Does strong CE give strictly more? | **Conjecturally yes**: the square-root cone (Fejér-Riesz/Krein) is strictly smaller than the raw PD cone. Empirically, MV achieved 1.2802 by using precisely this extra information in a discretized SDP. |
| Viazovska-style magic function? | **Unlikely in present form**. Modular forms in her proof exploit the specific lattice (E8) structure and the "Fourier eigenfunction with prescribed zeros" condition. The C_{1a} problem lacks a lattice; its analogue would be a Laplace transform of a cusp form of the right weight tuned to the PW_{π/2} type. No published construction. |
| Pure CE LP from first principles beats 1.2802? | **Not obviously**. MV (2010) already solves a discretized CE-strong-style QP; their 1.2802 likely reflects something close to the honest CE-strong optimum. |
| Hybrid CE + Lasserre? | **Most promising**. Use CE dual certificate as an outer bound; use Lasserre to certify positivity of `h` polynomially. `certified_lasserre/` already does half of this. |
| Applications of CE outside packing to R extremal problems? | Carruth-Elkies-Gonçalves-Kelly (2017) for box minorant; Carneiro-Milinovich-Soundararajan for prime gaps; Cohn-Gonçalves (2019) sign uncertainty. **Template is well-established**; transferring to C_{1a} is *not documented in the literature as of 2026-04*. |

**Bottom line.** CE reformulates C_{1a} cleanly but reduces (in its weak form) to the Delsarte dual we already have. The strong form (square-root cone) is SDP-representable after discretization — and that discretization is essentially what Matolcsi-Vinuesa do. Net new theoretical gain: small. Net new computational gain: tight connection to the `certified_lasserre/` pipeline, by interpreting the Lasserre hierarchy as a finite-dimensional CE primal.

## 6. Concrete next-step candidates (ranked)

1. **Fejér-Riesz square parametrization of `g`.** Replace `F1/F2` families in `delsarte_dual/` with `g = h * h̃`, `h ∈ PW_T`, automatically `ĝ ≥ 0`. Optimize with free `h`. Expected: modest improvement over 1.199 current floor; still unlikely to crack 1.2802 without also improving `w`.
2. **Sharpen `w(ξ)`.** Replace `cos^2(πξ/2)` with a Lasserre-certified tighter envelope on `|f̂(ξ)|^2` using moment constraints `∫ x^{2k} f(x) dx ≤ (1/4)^{2k}/(2k+1)`. This is a *free* upgrade inside the existing pipeline and is under-explored.
3. **Modular-form ansatz (research gamble).** Try `h(t) = L[φ](t)` where `φ` is a cusp form of weight tuned to the PW_{π/2} type — following Viazovska's playbook. No published template; open problem.
4. **Hybrid CE + Lasserre.** Use CE dual to certify `val(d) ≤ upper` on the polynomial relaxation, combined with the current `certified_lasserre/` lower bounds, to sandwich `val(d)` and extrapolate to C_{1a} as `d → ∞`.

## 7. 100-word summary

The Cohn-Elkies LP framework naturally encodes C_{1a} because `f*f` is automatically positive-definite, matching CE's Fourier-positivity hypothesis. The **weak** CE dual — require `g ≥ 0` on `[-1/2,1/2]` and `ĝ ≥ 0` on R — reduces exactly to the Delsarte bound already implemented in `delsarte_dual/`. The **strong** CE dual, which exploits the square-root cone `ρ̂ = f̂^2` (Fejér-Riesz / Krein), is SDP-representable and essentially what Matolcsi-Vinuesa computed to get 1.2802. Viazovska-style modular-form magic functions have no documented transfer; the problem has no lattice structure. Immediate practical gain: Fejér-Riesz parametrization `g = h*h̃` and a sharper Lasserre-certified envelope on `|f̂|^2`. Breakthrough beyond 1.2802 unlikely without new ideas.

---

## Sources

- [Cohn & Elkies 2003, arXiv:math/0110009](https://arxiv.org/abs/math/0110009)
- [Cohn-Elkies, Annals of Math PDF](https://annals.math.princeton.edu/wp-content/uploads/annals-v157-n2-p09.pdf)
- [Cohn 2002, New upper bounds II, Geom. Topol.](https://projecteuclid.org/journals/geometry-and-topology/volume-6/issue-1/New-upper-bounds-on-sphere-packings-II/10.2140/gt.2002.6.329.full)
- [Viazovska, Sphere packing in dim 8](https://annals.math.princeton.edu/2017/185-3/p07)
- [CKMRV, Universal optimality of E8 and Leech](https://gilkalai.wordpress.com/2019/02/15/henry-cohn-abhinav-kumar-stephen-d-miller-danylo-radchenko-and-maryna-viazovska-universal-optimality-of-the-e8-and-leech-lattices-and-interpolation-formulas/)
- [Carruth-Elkies-Gonçalves-Kelly, Beurling-Selberg box minorant](https://arxiv.org/abs/1702.04579)
- [Li-Cohn, Dual LP bounds for sphere packing via discrete reductions](https://www.sciencedirect.com/science/article/abs/pii/S0001870824005590)
- [Carneiro-Milinovich-Soundararajan, Fourier optimization and prime gaps](https://arxiv.org/pdf/1708.04122)
- [Matolcsi-Vinuesa, Improved autoconvolution bounds](https://arxiv.org/abs/0907.1379)
- [Cohn Sphere Packing page, MIT](https://cohn.mit.edu/sphere-packing/)
