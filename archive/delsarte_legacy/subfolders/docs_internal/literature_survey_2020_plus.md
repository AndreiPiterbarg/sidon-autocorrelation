# Literature Survey: Sidon Autoconvolution Constant C_{1a} (2020-2026)

**Problem.** C_{1a} = inf { ||f*f||_inf : f >= 0, supp f subset [-1/4, 1/4], int f = 1 }.
**Current status.** 1.28 (Cloninger-Steinerberger 2017) <= C_{1a} <= 1.5029 (Yuksekgonul et al. 2026).
**Unpublished:** 1.2802 (Xie 2026, "Grok chat", on Tao's repo, non-rigorous).

---

## A. Lower bounds on C_{1a}

**No peer-reviewed paper since 2017 improves 1.28.** The only claim above 1.28 is the unpublished `1.2802` attributed to Xie (2026, Grok chat) on
https://teorth.github.io/optimizationproblems/constants/1a.html -- no rigor / method described.

Related but not directly improving C_{1a}:

1. **J. Madrid, J. P. G. Ramos — "On optimal autocorrelation inequalities on the real line"** (CPAA 2020, arXiv:2003.06962). Improves sharp constants in Barnard-Steinerberger *autocorrelation* inequalities (max_t int f(x)f(x+t)dx type) via Hausdorff-Young duality. Does NOT touch C_{1a}. Transferable idea: Fourier duality + extremizer existence via compactness; could give a dual certificate for C_{1a}.

2. **J. Barnard, S. Steinerberger — extensions** (arXiv:2001.02326, Madrid-coauth extensions). Local-perturbation + matrix parameterization of autocorrelation inequalities. Does not improve C_{1a}; technique is gradient-style, already subsumed by our cascade.

3. **J. de Dios Pont, J. Madrid — "On classical inequalities for autocorrelations and autoconvolutions"** (arXiv:2106.13873, 2021). Proves existence of extremizers for weighted autocorrelation inequalities (Gaussian / indicator weights) + discretization+numerics. Does NOT improve 1.28. Transferable: rigorous existence arguments for extremizer in the [-1/4,1/4] problem.

4. **J. Gaitan, J. Madrid — "On suprema of convolutions on discrete cubes"** (arXiv:2512.18188, Dec 2025). Finds *exact* C_{k,1} for k-fold convolution on {0,1} hypercube:
   C_{2,1} = 1/2 (autoconvolution k=2 on the line {0,1}).
   Gives relation C_k = lim_m k(m+1) C_{k,m} and *upper* bound C_k <= 2k C_{k,1}. **Explicitly does NOT improve the 1.28 < C_{1a} < 1.51 range.** Technique: precise combinatorial identity on the cube; transfer is a *discrete lower-bound lemma* for our Lasserre relaxation (finite hypercube analog gives a pointwise lower bound).

5. **A. Jaech, A. Joseph — "Further Improvements to the Lower Bound for an Autoconvolution Inequality"** (arXiv:2508.02803, Aug 2025). Achieves ||f*f||_2^2 / (||f*f||_inf ||f*f||_1) >= 0.94136 with a 2399-interval step function + 4x upsampling. This is the *ratio* version of a different (but closely related) autoconvolution inequality — NOT a direct bound on C_{1a}. Method: step-function optimization + upsampling. Transferable: the refinement-by-upsampling idea could boost our cascade warm-starts.

6. **C. Boyer, Z. K. Li — "An improved example for an autoconvolution inequality"** (arXiv:2506.16750, Jun 2025). Same ratio problem; 575-interval step function giving 0.901564 (beat AlphaEvolve's 0.8962). Method: coarse-to-fine gradient ascent + simulated annealing.

7. **A. Rechnitzer — "The first 128 digits of an autoconvolution inequality"** (arXiv:2602.07292, 2026). Computes the L^2 (not L^inf) autoconvolution constant nu_2^2 to 128 digits via rigorous high-precision floating-point arithmetic. Different problem (L^2 norm), but method (interval arithmetic + high-precision SDP/LP) is a possible paradigm for the last-mile rigor on our SDP bounds.

8. **E. P. White — "An almost-tight L^2 autoconvolution inequality"** (arXiv:2210.16437, 2022). Determines inf ||f*f||_2 on [-1/2,1/2] with int f = 1 up to 0.0014% error. L^2 problem, not L^inf. Yields B_h[g] set size improvements. Not directly applicable to C_{1a}; the L^2 problem is nicer (strictly convex) and admits unique minimizer, unlike L^inf.

## B. Upper bounds on C_{1a}

Heavy activity 2025-2026 driven by AI:

| Year | Bound | Reference | Method |
|------|-------|-----------|--------|
| 2010 | 1.50992 | Matolcsi-Vinuesa (arXiv:0907.1379) | analytic construction |
| 2025-05 | 1.5053 | Georgiev-Gomez-Serrano-Wagner-Tao (arXiv:2511.02864, with DeepMind AlphaEvolve) | 600-interval step function, LLM-evolved code |
| 2025-12 | 1.503164 | ibid. (refined) | AlphaEvolve V2 |
| 2025-11 | 1.503133 | Wang et al. "ThetaEvolve" (arXiv:2511.23473) | RL test-time training on AlphaEvolve |
| 2026-01 | 1.50286 | Yuksekgonul et al. "Learning to Discover at Test Time / TTT-Discover" (arXiv:2601.16175) | 30,000-piece step function, trained-from-scratch |

Upshot: all upper-bound progress since 2010 is from AI-driven evolutionary search over step functions. No new analytic constructions. All methods operate on the *primal* (finding an explicit f), not the dual.

## C. New techniques since 2017 directly applicable

- **Step-function upsampling** (Boyer-Li 2025, Jaech-Joseph 2025) — refine a coarse step-function minimizer by 4x. Plug straight into `cloninger-steinerberger/` warm starts.
- **Evolutionary/LLM search over step functions** (AlphaEvolve, ThetaEvolve, TTT-Discover) — only gave upper bounds so far; applying them to the *dual* (dual certificate for lower bound) is open.
- **Correlative-sparsity Lasserre hierarchy** (Waki-Kim-Kojima-Muramatsu, 2006) — not new, but never applied to C_{1a} until now. Recent Lasserre convergence-rate work (arXiv:2505.00544 2025 on hypercube) could help dimension the `d=64-128` range for `lasserre/`.
- **Interval arithmetic for rigorous SDP bounds** (Rechnitzer 2026) — pattern for converting Mosek/Clarabel numerical output to a proof.

## D. Related problems worth stealing from

- **Sphere packing via Cohn-Elkies / Viazovska** — Delsarte-style dual functions now well understood; Cohn's 2024 survey (arXiv:2407.14999) on sphere-packing-to-Fourier-interpolation is the modern reference. Dual function = "magic function" analog of our Delsarte dual in `delsarte_dual/`. No direct C_{1a} attack yet.
- **B_h[g] set size bounds** — White (2022) gives new L^2 bounds; these *do* transfer to continuous constants but via L^2 not L^inf.
- **Madrid-Ramos Fourier duality (2020)** — the dual-side existence argument is a template for a rigorous dual certificate on the [-1/4,1/4] L^inf problem.

---

## Bottom line

Lower bound 1.28 has NOT been improved in any rigorous publication since Cloninger-Steinerberger 2017. The only claimed improvement (1.2802, Xie 2026) is an unpublished Grok-chat transcript on Tao's page. All AI-era work (AlphaEvolve, ThetaEvolve, TTT-Discover) has moved the *upper* bound down from 1.5099 to 1.5029 via step-function construction; no one has used these tools on the dual side. The Gaitan-Madrid (2025) hypercube result is the most conceptually relevant recent paper -- it gives exact discrete constants that could seed our SDP relaxations -- but explicitly does NOT improve 1.28.

**Opportunity:** applying Lasserre/SoS (as in `lasserre/`) or LLM-evolved dual certificates to the *lower bound* is unexplored territory. Methods from (1), (3), (4), (5), (7) above are each isolated levers that could plausibly yield the first rigorous improvement beyond 1.28.

---
## Sources

- [arXiv:1403.7988 Cloninger-Steinerberger (2017)](https://arxiv.org/abs/1403.7988)
- [arXiv:0907.1379 Matolcsi-Vinuesa (2010)](https://arxiv.org/abs/0907.1379)
- [teorth.github.io/optimizationproblems/constants/1a.html](https://teorth.github.io/optimizationproblems/constants/1a.html)
- [arXiv:2003.06962 Madrid-Ramos (2020)](https://arxiv.org/abs/2003.06962)
- [arXiv:2001.02326 Barnard-Steinerberger extension (2020)](https://arxiv.org/abs/2001.02326)
- [arXiv:2106.13873 de Dios Pont-Madrid (2021)](https://arxiv.org/abs/2106.13873)
- [arXiv:2210.16437 White (2022)](https://arxiv.org/abs/2210.16437)
- [arXiv:2506.16750 Boyer-Li (2025)](https://arxiv.org/abs/2506.16750)
- [arXiv:2508.02803 Jaech-Joseph (2025)](https://arxiv.org/abs/2508.02803)
- [arXiv:2511.02864 Georgiev-Gomez-Serrano-Tao-Wagner / AlphaEvolve (2025)](https://arxiv.org/abs/2511.02864)
- [arXiv:2511.23473 Wang et al. / ThetaEvolve (2025)](https://arxiv.org/abs/2511.23473)
- [arXiv:2512.18188 Gaitan-Madrid (2025)](https://arxiv.org/abs/2512.18188)
- [arXiv:2601.16175 Yuksekgonul et al. / TTT-Discover (2026)](https://arxiv.org/abs/2601.16175)
- [arXiv:2602.07292 Rechnitzer (2026)](https://arxiv.org/abs/2602.07292)
- [Terence Tao blog: Mathematical exploration and discovery at scale (2025-11-05)](https://terrytao.wordpress.com/2025/11/05/mathematical-exploration-and-discovery-at-scale/)
- [github.com/google-deepmind/alphaevolve_repository_of_problems](https://github.com/google-deepmind/alphaevolve_repository_of_problems)
