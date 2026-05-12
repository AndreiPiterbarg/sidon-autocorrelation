# PSWFs for a Dual Bound on C_{1a}: Idea Note

Date: 2026-04-20
Author: Claude (research note)

## 1. Setup and why PSWFs are even a candidate

**C_{1a} problem.** For f ≥ 0 on [−1/4, 1/4] with ∫f = 1,

    ||f ∗ f||_∞ ≥ C_{1a},    1.2802 ≤ C_{1a} ≤ 1.5029.

Since f is supported on [−1/4, 1/4], its Fourier transform f̂ is entire of exponential type π/2: i.e., f̂ ∈ PW_{π/2}. The autoconvolution f ∗ f is supported on [−1/2, 1/2], and (f ∗ f)^(ξ) = f̂(ξ)². The sup-norm over t ∈ [−1/2, 1/2] of f ∗ f controls the problem.

**PSWFs.** The Slepian-Pollak-Landau prolate spheroidal wave functions ψ_n^{(c)} are the eigenfunctions of the time-and-band-limiting operator

    (P_T B_W P_T g)(x) = ∫_{-T}^{T} (sin W(x-y))/(π(x-y)) g(y) dy

with c = TW. They are simultaneously: (i) the orthonormal basis for L²([−T,T]) that diagonalizes band-limiting, (ii) eigenfunctions of B_W with eigenvalues λ_n(c) ∈ (0,1) that cluster near 1 then drop sharply past n ≈ 2c/π, (iii) orthogonal on ℝ as elements of PW_W. This *double orthogonality* is the structural feature that makes them extremal for the Slepian concentration problem.

For our problem the natural scaling is T = 1/4, W = π/2, so **c = π/8 ≈ 0.3927** — a *very small* Slepian parameter. In this regime only ψ_0 has eigenvalue close to 1 (numerically λ_0(π/8) ≈ 0.55; higher n drop off rapidly). So the low-rank "Slepian space" is essentially 1-dimensional here.

## 2. What the literature says (short survey)

- **Slepian-Pollak-Landau (1961-62)** [Bell Syst. Tech. J.]: foundational; PSWFs solve max ||P_T g||² / ||g||² over g ∈ B_W. Extremal ONLY for L² concentration, not L^∞.
- **Hogan-Lakey (2005, 2012)**: *Time-Frequency and Time-Scale Methods* and survey "An Overview of Time and Multiband Limiting". Focus on frames, sampling, dual-generator tradeoffs for PW_{π}. No direct sup-norm extremal result on convolutions.
- **Karnik et al. (arXiv:2409.16584)**: supremum bound on PSWFs outside the concentration interval — relevant if we need rigorous pointwise control, but it bounds ||ψ_n||_∞, not ||ψ_n ∗ ψ_n||_∞.
- **Carneiro-Littmann / Vaaler Beurling-Selberg machinery**: the *correct* extremal family for bandlimited sup-norm / majorant / one-sided Fourier inequalities. Much more literature on using these for LP / dual bounds than on PSWFs.
- **Matolcsi-Vinuesa (2010, arXiv:0907.1379)**: improved *upper* bound on C_{1a} via carefully chosen f (not PSWF — they use a perturbation of arcsine-like densities).
- **Boyer-Li (2025, arXiv:2506.16750)** and **Cilleruelo-Ruzsa-Vinuesa**: recent numeric work with step-functions (up to 2399 steps, c ≥ 0.94136) — again, non-PSWF constructions.
- **Searches for "PSWF + C_{1a}"**, **"PSWF + Sidon"**, **"PSWF + autoconvolution sup-norm"** returned nothing. PSWFs have *not* been used for this family of problems.

## 3. Why PSWFs are probably NOT tight for C_{1a}

Three concrete obstructions:

**(a) Wrong extremal principle.** The Slepian problem is an L² concentration problem. C_{1a} is an L^∞ problem on f ∗ f (equivalently: a Chebyshev / LP problem on f̂²). PSWFs extremize ⟨P_T B_W g, g⟩, which is a ratio of L² norms. There is no direct reason the L²-optimizer is L^∞-optimal for autoconvolution.

**(b) Sign / positivity mismatch.** ψ_0^{(c)} > 0 on [−T,T] for small c (this is the "ground state"), so one *could* take f = ψ_0 / ∫ψ_0 as a feasible candidate. But numerically at c = π/8 this gives an almost-flat bump (since c is small), and (ψ_0 ∗ ψ_0)(0) ≈ ∫ψ_0² / (∫ψ_0)² ≈ 1/|support| * (concentration factor). Comparing to the Matolcsi-Vinuesa upper bound 1.5029, the ψ_0 candidate numerically gives ||f∗f||_∞ ≈ 2 (close to box), much worse than known constructions.

**(c) The Paley-Wiener role is a red herring.** Although f̂ ∈ PW_{π/2}, the constraint driving C_{1a} is *nonnegativity of f*, not bandlimit on f̂. PSWFs are optimal among L² functions in a band + interval, with *no sign constraint*. The C_{1a} cone is f ≥ 0, ∫f = 1, supp f ⊂ [−1/4, 1/4]: this is a moment-cone / measure-theoretic constraint, which is exactly where Lasserre/Delsarte LP duality and arcsine-type extremals (Matolcsi-Vinuesa) live, not PSWFs.

## 4. Where PSWFs COULD still help (specific proposal)

Even if PSWFs are not the primal extremizer, they may be useful as a **dual test family** or a **basis for a finite-rank dual relaxation**. Concrete proposals, ordered by plausibility:

### Proposal A (preferred): PSWF basis for a finite-rank dual SOS certificate on f̂²

The Lasserre dual for C_{1a} requires nonnegative trigonometric / entire polynomials h on [−1/2, 1/2] with ĥ dominating a certain moment functional. Expand the dual multiplier in the **prolate basis {ψ_n^{(c)}} on [−1/2, 1/2] with c = π/2**: these diagonalize the band-limiting operator on PW_{π}, so SOS conditions factor diagonally. Pragmatic payoff: potentially *smaller dual cert than trigonometric polynomial bases* because the PSWF basis adapts to the support+bandlimit geometry.

Action: in `delsarte_dual/`, implement `pswf_dual_basis.py` that
  1. Computes ψ_n^{(π/2)} via the Bouwkamp / Legendre expansion,
  2. Builds Gram matrices ⟨ψ_i ψ_j, ψ_k ψ_l⟩ on [−1/2, 1/2],
  3. Feeds into existing Mosek/Clarabel SDP as an alternate basis,
  4. Compares bound vs. Chebyshev/Legendre basis at same rank.

### Proposal B: PSWF as warm-start / primal candidate for cascade

Use ψ_0^{(c)} (c = π/8) and low-order perturbations f = ψ_0 + Σ α_n ψ_n (α_n chosen to preserve nonnegativity) as initial seeds for the Cloninger-Steinerberger QP cascade. Unlikely to beat arcsine-type seeds, but cheap to try.

### Proposal C: PSWF eigenvalue bound as a soft structural constraint

The fact that λ_0(π/8) < 0.55 implies that *any* f supported on [−1/4, 1/4] has f̂ which is at most 55 % concentrated on [−π/2, π/2]. This gives a quantitative "energy leakage" statement that might feed into a dual inequality, but the numerology (c small → λ_0 small but not tiny) suggests this will only yield weak bounds.

## 5. Recommended next step

Spend **2-3 days prototyping Proposal A** (PSWF dual basis). Success criterion: at a fixed Lasserre order d, does swapping Chebyshev basis for PSWF basis reduce the certificate rank needed to hit a given bound on val(d)? If yes, scale up; if no (expected outcome), document and move on to Beurling-Selberg / arcsine-majorant families, which the literature strongly suggests are closer to the true extremal.

**Expected outcome:** PSWFs do not give a tight bound on C_{1a}. The problem is not an L² concentration problem and its extremal (by Matolcsi-Vinuesa / Boyer-Li evidence) has sharp singular structure (spikes + comb) that is antithetical to the smooth bandlimited extremals of Slepian. But PSWFs might still serve as an efficient basis inside the existing dual SDP pipeline.

## 6. 100-word summary

**PSWFs extremize L² energy concentration of bandlimited signals on an interval (Slepian 1961). C_{1a} is an L^∞ autoconvolution-sup problem under a nonnegativity + compact-support constraint on f; its extremizer (per Matolcsi-Vinuesa 2010 upper 1.5029 and Boyer-Li 2025 step-function lower 0.94+) is sharply non-smooth, unlike the smooth bandlimited PSWFs. A literature scan found no PSWF-based bound on C_{1a} or cognate Sidon-autocorrelation problems. PSWFs are unlikely to be primal-tight, but the prolate basis at c = π/2 is a plausible *dual basis* for the Lasserre SDP: implement `pswf_dual_basis.py` and A/B-test against the Chebyshev basis at fixed rank.**

## Sources

- [Slepian-Pollak: PSWFs, Fourier Analysis and Uncertainty I, Bell Syst. Tech. J. 1961](https://ieeexplore.ieee.org/document/6773659)
- [Landau-Pollak II/III (1961-62)](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1962.tb03279.x)
- [Slepian-IV: many-dimensional generalization](https://ieeexplore.ieee.org/document/6773515)
- [Karnik et al. (2024), PSWF accuracy/dimensionality, arXiv:2409.16584](https://arxiv.org/abs/2409.16584)
- [Hogan-Lakey, Time-Frequency and Time-Scale Methods (Birkhäuser)](https://link.springer.com/chapter/10.1007/978-0-8176-8376-4_5)
- [Hogan-Lakey, Prolate Shift Frames and Sampling of Bandlimited Functions](https://link.springer.com/chapter/10.1007/978-3-030-36291-1_5)
- [Cloninger-Steinerberger (2014), arXiv:1403.7988](https://arxiv.org/abs/1403.7988)
- [Matolcsi-Vinuesa (2010), arXiv:0907.1379](https://arxiv.org/abs/0907.1379)
- [Boyer-Li (2025), arXiv:2506.16750](https://arxiv.org/abs/2506.16750)
- [Further improvements (2025), arXiv:2508.02803](https://arxiv.org/abs/2508.02803)
- [Carneiro-Littmann, Gaussian Subordination, arXiv:1008.4969](https://arxiv.org/abs/1008.4969)
- [Optimal L² autoconvolution inequality (Canadian Math. Bull.)](https://www.cambridge.org/core/journals/canadian-mathematical-bulletin/article/an-optimal-l2-autoconvolution-inequality/8D109D51F271CC78EBDA2C99FB35612D)
