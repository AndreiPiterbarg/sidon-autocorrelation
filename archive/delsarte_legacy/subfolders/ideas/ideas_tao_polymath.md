# Ideas from Tao / Polymath / MathOverflow — C_{1a} Lower-Bound Leads

**Date compiled:** 2026-04-20
**Budget used:** 11 fetches / 4 searches, ~15 min
**Target:** push lower bound above 1.2802 on C_{1a} (Sidon autocorrelation constant).

---

## 100-Word Summary

The Tao optimization-problems page for C_{1a} (constants/1a.html) is the primary clearinghouse. As of April 2026 the LB is 1.2802, credited to an **unpublished Grok session by Xinyuan Xie (XX2026)** — it is literally a grok.com share link, not a paper, and no construction is publicly described. The UB side has moved aggressively (AlphaEvolve -> ThetaEvolve -> TTT-Discover: 1.5053 -> 1.503164 -> 1.503133 -> **1.5029**), all in late 2025 / early 2026. **No LB improvement by LLM-evolution agents has been reported.** Main open leads: (i) re-run CS-2017 code with modern step-counts (2399 intervals is now standard for the MV constant, per Jaech-Joseph 2508.02803); (ii) Lasserre/SDP has not been tried on this specific constant anywhere outside our repo.

---

## Current Status Snapshot (teorth.github.io/optimizationproblems/constants/1a.html)

**Statement.** C_{1a} is the largest constant with
$$\max_{|t|\le 1/2} (f*f)(t) \ge C_{1a}\,\Big(\int_{-1/4}^{1/4} f\Big)^2$$
for nonneg f on [-1/4, 1/4].

**Upper bounds (all recent):**
| Value | Ref | Method |
|---|---|---|
| π/2 = 1.57059 | SS2002 | classical |
| 1.50992 | MV2009 | LP dual |
| 1.5053 | GGSWT2025 (May) | AlphaEvolve |
| 1.503164 | GGSWT2025 (Dec) | AlphaEvolve |
| 1.503133 | WSZXRYHHMPCHCWDS2025 | **ThetaEvolve** (arxiv:2511.23473) |
| **1.5029** | YKLBMWKCZGS2026 | **TTT-Discover** (arxiv:2601.16175) |

**Lower bounds:**
| Value | Ref | Method |
|---|---|---|
| 1.182778 | MO2004 | Martin-O'Bryant |
| 1.262 | MO2009 | MO |
| 1.2748 | MV2009 | Matolcsi-Vinuesa LP |
| 1.28 | CS2017 | Cloninger-Steinerberger, non-convex opt |
| **1.2802** | **XX2026** | **Xie, Grok-assisted, unpublished** |

**No "methods" or "proof strategies" section on the constant page.** No open-questions block.

---

## Concrete Leads

### 1. XX2026 — Xie Grok share (the only thing blocking our bound)

- URL: `https://grok.com/share/c2hhcmQtNQ_f4d17f80-4582-4679-b931-06277fd4cfd4?rid=a60436ae-eaba-4638-a0fd-47b231f19cd0`
- My fetch returned HTTP 403 (Grok blocks scraping). **The listing on Tao's page is literally a chat-share link, not a paper.** No construction is public.
- Claim is only 1.2802 - 1.28 = 0.0002 above CS2017 — i.e. a marginal refinement of CS-style non-convex optimisation. Likely a modest increase in discretisation (more bins in the mass distribution).
- **Action:** try to open this link manually from a browser (our fetch fails). If Grok's reasoning is visible, extract the step-count / support / bound it claims. Even if not replicable, it sets the floor we must beat.

### 2. Jaech-Joseph 2025 (arXiv:2508.02803) — relevant methodology

Not our constant (they bound the MV ν-constant on [-1/2, 1/2] from below, pushing 0.90156 -> 0.94136), **but the technique transfers directly**:
- Step function with **2,399 equally spaced intervals** (finite-dim non-convex opt).
- "**4x upsampling procedure**" on a 559-interval optimiser — i.e. refine the discretisation by bisecting each bin, warm-start.
- Closed ~40% of gap-to-conjecture in one paper.
**Action:** our `cloninger-steinerberger/cpu/` pipeline already does non-convex opt over mass-distributions; adapt Jaech-Joseph's "upsample-then-refine" strategy to the C_{1a} problem. This is the single most promising concrete technique.

### 3. Boyer-Li 2025 (arXiv:2506.16750) — simulated annealing

- 575 intervals, simulated annealing + gradient-based local search.
- Confirms **annealing is competitive with AlphaEvolve** for this class of problems.
**Action:** augment our cascade with SA escape moves on interval widths (we currently only evolve mass).

### 4. GGSWT2025 Tao et al. — "Mathematical exploration and discovery at scale"

- Blog post: terrytao.wordpress.com/2025/11/05/
- arXiv: 2511.02864
- Repository: `github.com/google-deepmind/alphaevolve_repository_of_problems` (67 problems, prompts, outputs).
- **Sidon autocorrelation / C_{1a} is Problem 2 in AlphaEvolve's repo.**
- **Action:** clone the alphaevolve repo, read the **exact prompts + output traces** for Problem 2 — this shows what construction families (support splits, step-functions, smoothed Gaussians) they explored and which failed.

### 5. ThetaEvolve (arXiv:2511.23473) — open-source AlphaEvolve

- Yiping Wang et al. (Nov 2025). Open-source, RL at test time, works with a single open-weights LLM.
- GitHub: `github.com/ypwang61/ThetaEvolve`.
- Advertises improving C_{1a} **upper** bound (1.503133) but NOT lower.
- **Action:** ThetaEvolve is cheap enough to self-host; point it at the LB side with our inner-product objective. Nobody has tried this — Xie used a generic Grok chat, which is strictly weaker.

### 6. TTT-Discover (arXiv:2601.16175) — current UB record

- Yuksekgonul, Koceja, ..., Sun (Stanford/Nvidia, Jan 2026). Holds UB 1.5029.
- Project page: `test-time-training.github.io/discover.pdf`, GitHub: `github.com/test-time-training/discover`.
- Explicitly lists C_{1a} ("first autocorrelation inequality AC1") among its target benchmarks.
- **Action:** mirror their AC1 problem-file, flip the objective sign, run for LB. Same argument as ThetaEvolve.

### 7. Cloninger-Steinerberger (arXiv:1403.7988) — the bottleneck

- CS explicitly say "our approach should be able to prove bounds arbitrarily close to the sharp result, but currently the bottleneck is **runtime** on a non-convex program."
- Our `lasserre/` hierarchy is precisely the tool to replace that non-convex program with a convex SDP relaxation.
- **Action:** CS method has never been scaled with modern SDP solvers (Mosek, Clarabel) + correlative sparsity. This is already our plan in `lasserre/preelim.py` — validated direction.

### 8. Tao's blog (terrytao.wordpress.com) — no dedicated C_{1a} post

Reviewed:
- 2025-11-05 post "Mathematical exploration and discovery at scale" — discusses Sidon-set upper bound and AlphaEvolve but **does not mention C_{1a} specifically**.
- Polymath blog: no thread on Sidon autoconvolution (Polymath 1, 5, 12, 13, 16 are on unrelated problems).
- Damek Davis tweet (linked from the constant page) — notes optimiser for C_{1a} is "surprisingly nasty". Suggests the extremizer is not a simple function family (Selberg / Vaaler / arcsine all strictly sub-optimal).

### 9. MathOverflow / research community

- No dedicated MO thread for C_{1a} surfaced in searches.
- The only deep discussion thread of the quantity is in CS2017 + MV2009 directly.
- **MO post opportunity:** asking about dual-certificate / extremizer structure could surface unpublished ideas. (Our `mo_2004.txt`, `mo_2009.txt` in this directory already cache those old threads.)

---

## Techniques Worth Testing Immediately

1. **Upsample-and-refine** (Jaech-Joseph): 4× bisection warm-start from current best. **Highest expected value.**
2. **LLM-evolution on LB side**: TTT-Discover / ThetaEvolve / AlphaEvolve open-repo — nobody has pointed these at the LB yet. Asymmetry = opportunity.
3. **Lasserre SDP with correlative sparsity** (our `lasserre/`): directly replaces CS2017 non-convex opt. Already being pursued.
4. **Simulated annealing over interval widths** (Boyer-Li): adds escape moves to our cascade.
5. **Extract XX2026 construction** from the Grok share link (needs browser).

## Explicit Unknowns After This Survey

- What construction does Xie use? (Grok chat, 403 on fetch.)
- Did GGSWT2025 / ThetaEvolve / TTT-Discover try the LB direction and fail quietly? Check alphaevolve_repository_of_problems commit history.
- Is there an extremizer structure result (symmetry, support, number of peaks)? Damek Davis's tweet hints at non-trivial shape but no paper.

---

## References (all URLs)

- Tao optim. page: https://teorth.github.io/optimizationproblems/constants/1a.html
- CS2017: https://arxiv.org/abs/1403.7988
- MV2009: https://arxiv.org/abs/0907.1379
- Boyer-Li 2025: https://arxiv.org/abs/2506.16750
- Jaech-Joseph 2025: https://arxiv.org/abs/2508.02803
- GGSWT Tao et al. 2025 (blog): https://terrytao.wordpress.com/2025/11/05/mathematical-exploration-and-discovery-at-scale/
- GGSWT arXiv: https://arxiv.org/abs/2511.02864
- AlphaEvolve repo: https://github.com/google-deepmind/alphaevolve_repository_of_problems
- ThetaEvolve: https://arxiv.org/abs/2511.23473 / https://github.com/ypwang61/ThetaEvolve
- TTT-Discover: https://arxiv.org/abs/2601.16175 / https://github.com/test-time-training/discover
- Xie Grok share (XX2026): https://grok.com/share/c2hhcmQtNQ_f4d17f80-4582-4679-b931-06277fd4cfd4?rid=a60436ae-eaba-4638-a0fd-47b231f19cd0
- Rechnitzer 128-digit (ν₂², different constant): https://arxiv.org/abs/2602.07292
