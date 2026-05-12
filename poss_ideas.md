# Possible Ideas — Pushing $C_{1a} > 1.2802$

Ideas that build on previously-investigated frameworks but introduce a new analytic/computational mechanism with mathematically-defensible chance of producing a non-trivial lift on the rigorous bound.

Conventions: $f \ge 0$ on $[-1/4, 1/4]$, $\int f = 1$. $C_{1a} = \inf_f \|f * f\|_\infty$. Current rigorous LB $\approx 1.2748$ (MV) / $1.28$ (CS).

---

## Idea 1 — MV multi-moment + Martin–O'Bryant Lemma 2.17 + 2D BnB

- **Builds on:** MV Cauchy-Schwarz framework (1.2748).
- **New mechanism:** MO 2.17 ($\Re \hat f(2)$ vs $\Re \hat f(1)$ polynomial constraint) strictly shrinks the $(z_1, z_2)$ feasibility region in the MV master inequality. Any monotone master inequality minimum **strictly increases** on a smaller feasible set — this is mathematically forced, not heuristic.
- **Math valid:** MO 2.17 is a proven nonneg-measure inequality (Martin–O'Bryant 2004, arXiv:math/0408207).
- **Feasibility:** ~80% coded in [delsarte_dual/mv_lemma217.py](delsarte_dual/mv_lemma217.py), [forbidden_region.py](delsarte_dual/forbidden_region.py), [mo_cuts.py](certified_lasserre/mo_cuts.py). 1–2 weeks integration. 2D BnB at $10^6$ cells = trivial compute.
- **Hard evidence:** Monotone-min-on-smaller-set is forced. Magnitude of lift is the open question; `delsarte_dual/improvements.md` projects 1.27481 → [1.28, 1.30].
- **Risk:** Actual lift may only be 1.275 → 1.278.
- **Tier:** 1 (strong)
- **P(beat 1.2802):** 25–35%
- **Effort:** 1–2 weeks

---

## Idea 2 — Mossinghoff–Vaaler 2019 moment-constrained Fourier minorant + Cohn–Elkies LP

- **Builds on:** Delsarte / Cohn–Elkies dual.
- **New mechanism:** MV-2019 (arXiv:1903.08731) and Madrid–Ramos (arXiv:2003.06962) provide strictly tighter Fourier minorants $|\hat f(\xi)|^2 \ge w_{\text{MV}}(\xi; m_0, m_1, \ldots, m_J)$ than the $\cos^2(\pi\xi/2)$ baseline used in the existing `delsarte_dual/`.
- **Math valid:** MV-2019 minorants are individually proven inequalities.
- **Feasibility:** 2–3 weeks SDP build + Magron rational rounding (already in repo).
- **Hard evidence:** Directly attacks the documented 33% "rigor factor gap" in [delsarte_dual/postmortem.md](delsarte_dual/postmortem.md): "A Chebyshev-type moment inequality could push the rigor factor from ~0.75 closer to 1."
- **Risk:** Even with full rigor-factor closure, the underlying class may cap at 1.10–1.20, not 1.28+.
- **Tier:** 2 (good)
- **P(beat 1.2802):** 15–25%
- **Effort:** 2–3 weeks

---

## Idea 3 — Algorithmic large-scale dual-kernel search with rigorous Bochner certification

- **Builds on:** 14-kernel Bochner-admissible sweep.
- **New mechanism:** Parametrize $K$ as a 50–100-dimensional Chebyshev/spline expansion, run CMA-ES / basin-hopping over $10^4$–$10^5$ kernels, verify each via a small Bochner-PSD SDP feasibility check (NOT a val(d) Lasserre).
- **Math valid:** any $K$ with Bochner positivity yields a rigorous LB via the MV master inequality.
- **Feasibility:** 1 week setup + few CPU-days.
- **Hard evidence:** 14-kernel sweep showed arcsine **locally optimal in tested family**, not provably global in the full Bochner cone. A 100-dim search has dramatically more headroom.
- **Risk:** arcsine may in fact be globally optimal across the entire Bochner cone (this is *consistent with* but not *implied by* the local-max result).
- **Tier:** 2 (good)
- **P(beat 1.2802):** 15–20%
- **Effort:** 1–2 weeks

---

## Idea 4 — Cauchy–Schwarz stability in MV proof — uniform slack analysis

- **Builds on:** MV's Cauchy-Schwarz step in 1.2748 derivation.
- **New mechanism:** Quantitative C-S stability (Maligranda 2006, J. Math. Anal. Appl.) bounds the slack in $|\langle u,v\rangle|^2 \le \|u\|^2\|v\|^2$ by $\|u - \lambda v\|^2$. Apply uniformly over all admissible $f$ — the slack at the MV-extremizer is small but slack at OTHER admissible $f$ (which is what we need to bound the inf) can be larger.
- **Math valid:** C-S stability is classical and explicit.
- **Feasibility:** 2–3 weeks of analytic work + numerical verification.
- **Hard evidence:** structural — MV's bound becomes equality only on a 1-parameter family; off this family, slack is positive.
- **Risk:** uniform slack may be $O(10^{-4})$ rather than the $\sim 5 \times 10^{-3}$ needed.
- **Tier:** 2 (good)
- **P(beat 1.2802):** 15–20%
- **Effort:** 2–3 weeks

---

## Idea 5 — Higher-order Carneiro–Gonçalves extremals

- **Builds on:** Beurling–Selberg / Carneiro extremal-majorant theory.
- **New mechanism:** Higher-order Carneiro–Gonçalves sign-uncertainty / Hilbert-type extremals (arXiv:1311.1157, arXiv:2003.06317, arXiv:2210.01437) NOT covered by the 14-kernel Bochner-admissible sweep.
- **Math valid:** Carneiro–Gonçalves results are rigorously proven sharp extremals.
- **Feasibility:** 2–3 weeks (1 week derivation + 1–2 weeks verification).
- **Hard evidence:** Distinct extremal class from arcsine; theoretical possibility of higher value.
- **Risk:** Per operator/spectral analysis: "MV-arcsine bound is structurally a Bohr-Favard-type result; Carneiro variants likely land at the same value."
- **Tier:** 3 (speculative but valid)
- **P(beat 1.2802):** 10–15%
- **Effort:** 2–3 weeks

---

## Idea 6 — 3-point SDP relaxation in Lasserre hierarchy *(empirically tested 2026-04-28; PARTIALLY ALIVE — formulation-dependent)*

- **Builds on:** Lasserre SDP hierarchy + delsarte_dual two-measure design.
- **New mechanism:** Add 3-point joint moments $y_{abc}$ on the 3-cube
  $[-1/4, 1/4]^3$ with full PSD on the 3D moment matrix and three box
  localizers, marginal-coupled to the existing 2-point moments
  $g_{ab}$ via $y_{ab0} = g_{ab}$.
- **Math valid:** yes — the 3-point cone is a valid relaxation tightening,
  monotone in constraints.
- **Empirical verdict (3 sessions of pilot work):** mixed.
  - **V1 ([lasserre/threepoint_sdp.py](lasserre/threepoint_sdp.py))**: positive-bump
    kernel $q^*_N$, no $\nu$-block. $\Delta$ ~ $3\times10^{-4}$ at $(7,7)$
    — later identified as monomial-basis conditioning artifact in V3.
  - **V2 ([lasserre/threepoint_full.py](lasserre/threepoint_full.py))**: full pilot
    design with Christoffel-Darboux $\nu$-side dualized to Putinar SOS Gram
    matrices. **$\lambda = 1$ exactly, $\Delta = 0$ structurally** (relaxation
    finds pseudo-moments making higher Legendre coefficients of $f*f$ vanish;
    3-point cone extends them). See [lasserre/THREEPOINT_FULL_REPORT.md](lasserre/THREEPOINT_FULL_REPORT.md).
  - **V3 ([lasserre/threepoint_alternatives.py](lasserre/threepoint_alternatives.py))**:
    bump-kernel objective + **$L^\infty$ density bound** on $\rho^{(2)}, \rho^{(3)}$.
    The diagonal pseudo-measures that gave V2's trivial $\lambda = 1$ are excluded.
    **3pt cone IS strictly tighter than 2pt cone**: basic 3D PSD+localizers add
    0.4–4% lift, additional 3D L^∞ adds 10–33%. At $(k,N) = (8,8), \|f\|_\infty \le 2.10$:
    $\lambda^{3\mathrm{pt}} = 1.357 > 1.2802$. **But this is a LB on
    $C_{1a}^{(\|f\|_\infty \le M)}$, not on $C_{1a}$ directly.** See
    [lasserre/THREEPOINT_ALTERNATIVES_REPORT.md](lasserre/THREEPOINT_ALTERNATIVES_REPORT.md).
- **Track 1 (2026-04-29) — $\|f^*\|_\infty$ question SETTLED in negative.**
  Three convergent indicators:
  (i) Analytical truncation/mollification gives $\|f'\|_\infty \ge 4$ minimum
  (Young's inequality $\|f\|_2 \ge \sqrt 2$ barrier).
  (ii) Numerical val(d) optima have $\|f^{(d)}\|_\infty$ growing monotonically:
  $d=4 \to 2.88$, $d=12 \to 5.21$. Linear-on-log fit suggests $\to \infty$.
  (iii) Literature near-optimizers (MV, White, Schinzel-Schmidt) are integrably-
  singular at endpoints. No paper proves a-priori UB on $\|f^*\|_\infty$.
  **The L^∞ bridge cannot lift $\lambda^{3pt}(M=2.15) > 1.2802$ to a $C_{1a}$ LB.**
  See [lasserre/TRACK1_FINF_FINAL_REPORT.md](lasserre/TRACK1_FINF_FINAL_REPORT.md).
- **Tier:** 4 (closed; continuous-formulation line dead)
- **P(beat 1.2802):** **~3%** (revised down from 15-25%). The 3pt cone IS
  strictly tighter than 2pt cone (this empirical finding stands), but the
  bound it gives is on $C_{1a}^{(M)}$, strictly larger than $C_{1a}$ for any
  $M < \infty$, and the Sidon optimum requires $\|f^*\|_\infty = \infty$.
- **Effort spent:** 4 sessions.
- **Live successor:** discrete val(d) with 3-point lift; 3pt within `lasserre/dual_sdp.py`
  framework would have implicit $L^\infty$ structure (per-bin density bounded).

---

## Summary

| # | Idea | Tier | P(beat 1.2802) | Effort | Builds on |
|---|------|------|----------------|--------|-----------|
| 1 | MV + MO 2.17 + 2D BnB | 1 | 25–35% | 1–2 wk | MV |
| 2 | MV-2019 / Madrid-Ramos minorant + Cohn-Elkies | 2 | 15–25% | 2–3 wk | Delsarte |
| 3 | Large-scale kernel search + Bochner cert | 2 | 15–20% | 1–2 wk | 14-kernel sweep |
| 4 | C-S stability uniform slack in MV | 2 | 15–20% | 2–3 wk | MV |
| 5 | Higher-order Carneiro-Gonçalves extremals | 3 | 10–15% | 2–3 wk | Beurling-Selberg |
| 6 | 3-point SDP in Lasserre (closed) | 4 | ~3% | done | Lasserre + delsarte_dual |

**Independent-portfolio probability of at least one breakthrough:** $1 - \prod_i (1 - p_i)$ with optimistic midpoints gives $\approx 65\%$. Honest correlated estimate (all rely on existing mathematical structure being not-yet-fully-exhausted): **35–50%**.

### Recommended sequencing

- **Week 1:** Idea 1 (MV+MO 2.17) — fastest to deploy, code mostly there.
- **Week 2:** If Idea 1 stalls, Ideas 2 + 3 in parallel (different infrastructure).
- **Week 3:** Idea 4 if any moment-related leverage emerged in week 1.
- **Backup:** Idea 5 only if Tier 1+2 all stall and you want a structurally-different attempt.

---

# Full Audit (2026-04-28) — Independent multi-agent review of Ideas 1–5

This audit was performed after a four-agent independent review (Idea 1 deep
audit, Idea 2 deep audit, Idea 3 deep audit, repo prior-art across all 5).
Findings are **substantially less optimistic than the body above**; for each
idea, evidence already in the repository contradicts or constrains the body's
probability estimates. The body's individual probabilities are overstated by
3–10×; the recommended sequencing is replaced.

**Note on Idea 6**: between drafting and audit, Idea 6 (3-point SDP) was
empirically tested and is now closed (DEAD by structural mathematics). This
removes the "fallback to 3-point" option and tightens the picture for Ideas
1–5: there is no obvious next-step in the SDP direction.

## A. Verdicts at a glance

| # | Idea | Body P | **Audited P** | Status / fatal issue |
|---|------|--------|---------------|----------------------|
| 1 | MV + MO 2.17 + 2D BnB | 25–35% | **3–6%** | **MO 2.17 constraint is INACTIVE at the MV extremizer**, derivation already in [delsarte_dual/forbidden_region.py:71-125](delsarte_dual/forbidden_region.py); repo's own [delsarte_dual/MASTER_PLAN.md:17](delsarte_dual/MASTER_PLAN.md) admits "multi-moment extension caps at ~1.2755". |
| 2 | MV-2019 / Madrid-Ramos minorant + CE-LP | 15–25% | **3–7%** | **Citation misattributed**: arXiv:1903.08731 is Barnard-Steinerberger, not Mossinghoff-Vaaler; **no 2019 MV paper exists**. Per [delsarte_dual/postmortem.md](delsarte_dual/postmortem.md), full closure of the "33% rigor factor gap" still leaves the framework below 1.28; per [ideas_cohn_elkies.md](delsarte_dual/ideas_cohn_elkies.md) §5–7, strong-CE = MV-2010 = the existing 1.2802 record. |
| 3 | Large-scale kernel search + Bochner cert | 15–20% | **3–5%** | **Already done.** [delsarte_dual/grid_bound_alt_kernel/REPORT.md](delsarte_dual/grid_bound_alt_kernel/REPORT.md) documents 5 phases: 31 named kernels (arcsine wins), full 2D $(\delta,\beta)$ grid (MV's choice = global max at 1.27481), AND a 9-D Chebyshev variational global search via `variational_pod.py` (DE, popsize 60, 200 iters). Body's "100-dim search" is a scale-up of code that already exists into a basin where arcsine has been local-max in every test. Theoretical ceiling per [mv_construction_detailed.md](delsarte_dual/mv_construction_detailed.md) §3 is **≈ 1.276**, below 1.2802. |
| 4 | C-S stability uniform slack | 15–20% | **2–8%** | **Logic flaw**: MV's bound is equality on a 1-parameter family of $f$'s; therefore $\inf_f \mathrm{slack}(f) = 0$, and uniform-slack analysis cannot lift the bound. The body itself states "MV's bound becomes equality only on a 1-parameter family" without resolving the implication. **0% code in repo.** |
| 5 | Higher-order Carneiro-Gonçalves | 10–15% | **3–8%** | **Already analyzed and rejected** in [delsarte_dual/ideas_beurling_selberg.md:99-111](delsarte_dual/ideas_beurling_selberg.md): "BS extremal properties optimize $L^1$ errors, not the $L^2$-ratio needed for $C_{1a}$"; F1 Fejér family already subsumes BS. Body's own risk ("Carneiro variants likely land at the same value") is the right read. |
| 6 | 3-point SDP in Lasserre | n/a | **Closed** | **Empirically dead** ($\lambda^{2\mathrm{pt}} = \lambda^{3\mathrm{pt}} = 1$ structurally at every reachable $(k,N)$). |

**Reconciled portfolio probability** of at least one Idea 1–5 beating 1.2802:
$1 - \prod_i (1 - p_i^{\text{audited}}) \approx 14$–$28\%$ at audited midpoints,
vs. the body's $\approx 65\%$ optimistic / $35$–$50\%$ honest correlated estimate.
**Body overstates by 2–4×**. Combined with Idea 6 closure (which removed the
strongest SDP-direction candidate from `new_ideas.md`), the realistic outlook
is: **no clear high-probability path to 1.2802 from any single idea.**

## B. Specific corrections per idea

### Idea 1 — MO 2.17 is INACTIVE at the MV extremizer (most important finding)

[delsarte_dual/forbidden_region.py:71-104](delsarte_dual/forbidden_region.py)
contains a derivation **already in the repo** that destroys the strict-lift
premise:

- MO Lemma 2.17 constrains real parts: $\Re\hat f(2) \le 2\Re\hat f(1) - 1$.
- The MV master inequality RHS is **phase-free** — depends only on $|\hat f(n)|$.
- Eliminating the unobserved phase variables collapses MO 2.17's
  magnitude-shadow constraint to: $2 r_1 + r_2 \ge 1$.
- At the MV extremizer, $r_1, r_2 \approx 0.5$, so $2 r_1 + r_2 \approx 1.5 \gg 1$.
  **Constraint inactive.**

The body's "monotone-min-on-smaller-set ⇒ strict lift" argument requires the
extremizer to be EXCLUDED from the smaller set. It isn't.

Independent corroborating evidence already in repo:
- [delsarte_dual/MASTER_PLAN.md:17](delsarte_dual/MASTER_PLAN.md): "Multi-moment extension caps at ~1.2755 — my earlier optimism was wrong."
- [delsarte_dual/creative_audit.md:9-11](delsarte_dual/creative_audit.md): scan at $n=1000$ asymptotes to **1.2764**, not 1.28+.
- [delsarte_dual/mo_framework_detailed.md:324](delsarte_dual/mo_framework_detailed.md): the "[1.28, 1.30]" projection is a hand-wave with no numerical support.

Code state: **~95% complete, runnable end-to-end** (`mv_lemma217.py` 540L,
`forbidden_region.py` 337L, `mo_cuts.py` 361L). Completeness is not the
bottleneck; the math is.

**Cheap test** before any 1–2 week commitment: run `python -m
delsarte_dual.mv_lemma217` and `python -m certified_lasserre.mo_cuts`. Both
will print lift ≤ 1e-3 per the existing analysis. **Cost: 30 minutes.**

### Idea 2 — Citation misattribution + framework ceiling below 1.28

**Misattribution.** arXiv:1903.08731 is **Barnard & Steinerberger** ("Three
Convolution Inequalities on the Real Line"), not Mossinghoff-Vaaler. arXiv
search for `au:Mossinghoff AND au:Vaaler` returns one paper
(Borwein-Mossinghoff-Vaaler 2005, math/0501163, on $L^p$ norms of polynomials).
**No 2019 Mossinghoff-Vaaler paper exists.** The "moment-constrained Fourier
minorant" object the idea presumes does not exist in the literature under that
attribution.

**Madrid-Ramos** ([arXiv:2003.06962](https://arxiv.org/abs/2003.06962)) is real
but extends Barnard-Steinerberger 2019 on the **0.42 dual-side** inequality
and on regularity of extremizers. **It does NOT produce a tighter pointwise
minorant on $|\hat f(\xi)|^2$.**

**Framework ceiling.**
[delsarte_dual/postmortem.md:24-29](delsarte_dual/postmortem.md) computes that
beating 1.2802 needs an *idealised* ratio of $\hat g(0)/M_g \ge 1.2802 / 0.75 \approx 1.7$,
"above even MV's idealised value, which is exactly the reason their record
stands." **The framework cannot reach 1.28 by minorant-tightening alone.**

[delsarte_dual/ideas_cohn_elkies.md](delsarte_dual/ideas_cohn_elkies.md) §6:
"strong-CE = essentially what Matolcsi-Vinuesa did to get 1.2802." So strong-CE
IS the existing record, not a new direction.

### Idea 3 — "100-dim kernel search" duplicates work already done

[delsarte_dual/grid_bound_alt_kernel/REPORT.md](delsarte_dual/grid_bound_alt_kernel/REPORT.md)
documents 5 phases:

- **Phase 1-2** (v1, v2): 31 named kernels (arcsine, triangular, Gaussian, Jackson, Selberg, Riesz, Chebyshev-beta {0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 1.00}, Epanechnikov, Hann, B-spline, Askey). Arcsine = Chebyshev-beta=0.5 is **strict local maximum**: β=0.45 → 1.2715, β=0.5 → 1.2745, β=0.55 → 1.2700.
- **Phase 3**: joint $(\delta,\beta)$ scan on 8×5 grid. Global maximum = MV's $(0.138, 0.5)$ at $M_{\mathrm{cert}} = 1.27481$.
- **Phase 5**: `variational_pod.py` runs a 9-D Chebyshev variational global search (popsize 60, 200 iters, multi-seed differential evolution). Sanity-checks recover arcsine. The parametrization is exactly what the body proposes.

**The MV functional is not a clean ratio** — it's a transcendental implicit
equation involving $k_1 = \hat\phi(1)^2$, $K_2 = \int|\hat\phi|^4$, and a
moment-sum $S_1$. Arcsine sits at a special corner because its
$1/\sqrt{1-4x^2}$ boundary singularity gives minimal $L^4$ for given
$\hat\phi(1)$.

**Theoretical ceiling** per [delsarte_dual/mv_construction_detailed.md](delsarte_dual/mv_construction_detailed.md)
§3: under MV's master inequality, the limit even with optimal $\|K\|_2^2$
surrogate is **≈ 1.276** — below 1.2802.

Genuinely fresh untested angles (none guaranteed):
- **PSWFs** (prolate spheroidal wavefunctions) — flagged in [delsarte_dual/ideas_pswf.md](delsarte_dual/ideas_pswf.md), never implemented.
- **Re-derive MO's $\|K\|_2^2 < 0.5747/\delta$ surrogate** with a tighter constant — would lift ALL kernels including arcsine.
- **Non-symmetric $\phi$** — every sweep assumes even.
- **Joint $(\phi, G)$ outer-loop** — every prior sweep fixed one or the other.

### Idea 4 — Logic flaw in uniform-slack argument

The body says: "slack at the MV-extremizer is small but slack at OTHER
admissible $f$ can be larger." Then concedes: "MV's bound becomes equality
only on a 1-parameter family."

But the bound is $\inf_f \|f * f\|_\infty$. To LIFT the lower bound, we need
$\inf_f \mathrm{slack}(f) > 0$. **If the bound is equality on a 1-parameter
family of admissible $f$, then $\inf_f \mathrm{slack}(f) = 0$**, and
uniform-slack analysis adds nothing.

A salvageable form would require excluding the 1-parameter family by a SEPARATE
constraint, but the body does not propose one. The repo has **0% code on this
idea** — no "Maligranda", no "C-S stability", no "uniform slack" anywhere.

### Idea 5 — Already analyzed and rejected in repo

[delsarte_dual/ideas_beurling_selberg.md:99-111](delsarte_dual/ideas_beurling_selberg.md):

1. F1 (Fejér family) subsumes BS via trigonometric polynomial optimization.
2. BS extremal properties optimize $L^1$ errors, not the $L^2$-ratio needed for $C_{1a}$.
3. Tight dual requires the unknown sharp extremal $f$.

The body's own risk ("MV-arcsine is structurally Bohr-Favard; Carneiro variants
likely land at same value") is correct. The cited papers — arXiv:1311.1157
(Carneiro-Chandee-Milinovich on Riemann zeta pair correlation),
arXiv:2003.06317, arXiv:2210.01437 — are about different extremal problems
(zero-counting, sign-uncertainty), not autoconvolution lower bounds on nonneg
compactly-supported $f$. **Directionality is wrong**: BS gives upper bounds on
counting / large-sieve quantities, not lower bounds on $\sup_t (f*f)(t)$.

## C. Cross-cutting: framework ceiling

A pattern emerges across Ideas 1–3: each works **within the MV /
Cohn-Elkies / Bochner framework**, and that framework has a **documented
ceiling at ≈ 1.276** per [delsarte_dual/mv_construction_detailed.md](delsarte_dual/mv_construction_detailed.md)
§3 and [delsarte_dual/postmortem.md](delsarte_dual/postmortem.md). Tightening
minorants, adding moment cuts, or searching for better kernels all hit this
ceiling.

The body assumed that escaping to a richer relaxation (k-point / 3-cube
SDP) would unlock 1.28+. **Idea 6 closure (2026-04-28) refutes that
assumption**: the 3-point Lasserre relaxation finds $\lambda = 1$
structurally because pseudo-moments make higher Legendre coefficients
vanish, and the 3-point cone EXTENDS rather than constrains those
pseudo-moments. To get useful tightening, one needs **diagonal-pseudo-measure
exclusion** — equivalent to an $L^\infty$ bound on $f$ or discretization,
both of which converge back to the val(d) pipeline that already exists.

## D. Surviving directions and replaced sequencing

The **only** directions with non-trivial probability of beating 1.2802 are:

| Direction | Audited P | Why |
|---|---|---|
| **PSWF kernel family** (subset of Idea 3) | 5–10% | Genuinely untested; Slepian-related extremals historically match arcsine in similar problems but not certified. |
| **Re-derive 0.5747 MO surrogate constant** (subset of Idea 3) | 5–15% | Pure analysis; lifts ALL kernels including arcsine; no compute risk. |
| **Joint $(\phi, G)$ outer-loop** (subset of Idea 3) | 3–8% | Every prior sweep fixed one or the other. |
| **Lasserre val(d) higher d with diagonal-exclusion** (subset of Idea 6 residual) | 5–10% | Already the active pipeline in [lasserre/](lasserre/); not novel but the only direction with empirical traction. |

Replaced 4-week plan:

1. **Day 1 (free):** Run `python -m delsarte_dual.mv_lemma217` + `python -m certified_lasserre.mo_cuts` to formally close Idea 1 with numerical evidence.
2. **Week 1:** Re-derive MO's $\|K\|_2^2$ surrogate with a tighter constant. Pure analysis. Cost: $0. Even a 1% improvement in the surrogate lifts ALL kernels, including arcsine.
3. **Week 2:** Implement PSWF kernel in `delsarte_dual/grid_bound_alt_kernel/`. Test against the existing `variational_pod.py` baseline. Cost: ~$50 compute.
4. **Continue main pipeline**: higher-degree val(d) Lasserre with the existing `lasserre/` infrastructure (this is the only direction with empirical traction post-Idea-6-closure).

**Total budget**: ~$50–100. **Drop Ideas 1, 2, 4, 5 from active consideration** — they have either fatal mathematical flaws (1, 4), citation problems and framework ceilings (2), or have already been analyzed and rejected in the repo (5).

## E. Honest summary

The body of `poss_ideas.md` reads like an optimistic brainstorm. The audit
shows:

- **Idea 1**: code 95% done, but the MO 2.17 constraint is provably inactive; repo already documented this.
- **Idea 2**: cited paper doesn't exist under that attribution; framework cannot reach 1.28 even at the limit.
- **Idea 3**: a 9-D variational search and 31-kernel + 2D-grid sweep already happened, with arcsine winning; framework theoretical ceiling ≈ 1.276.
- **Idea 4**: structural logic flaw the body itself flags but does not resolve.
- **Idea 5**: already analyzed and rejected in `delsarte_dual/ideas_beurling_selberg.md`.
- **Idea 6**: closed empirically (3-point SDP saturates trivially at $\lambda = 1$).

The **honest portfolio** is closer to **14–28% chance** of beating 1.2802 from
the salvageable subsets (PSWFs + MO-surrogate re-derivation + continued
val(d) at higher degree), not the body's 35–65%. The 1.2802 record's 9-year
longevity is consistent with this audit: the easy directions are genuinely
exhausted within the MV / CE-LP / 3-point Lasserre frameworks.

**The remaining viable path is the existing val(d) Lasserre pipeline at higher
degree**, plus the two "free" analytic directions (MO surrogate re-derivation,
PSWF). No 2–3 week SDP build is justified by the audit.

## F. Sources consulted

- [delsarte_dual/forbidden_region.py:71-125](delsarte_dual/forbidden_region.py) — derivation that MO 2.17 is inactive at MV extremizer
- [delsarte_dual/MASTER_PLAN.md:17](delsarte_dual/MASTER_PLAN.md) — repo admission of multi-moment 1.2755 cap
- [delsarte_dual/creative_audit.md:9-11](delsarte_dual/creative_audit.md) — actual scan asymptote 1.2764
- [delsarte_dual/mo_framework_detailed.md:324](delsarte_dual/mo_framework_detailed.md) — origin of [1.28, 1.30] hand-wave
- [delsarte_dual/postmortem.md:24-29, 67](delsarte_dual/postmortem.md) — rigor factor calculation, framework ceiling
- [delsarte_dual/ideas_cohn_elkies.md](delsarte_dual/ideas_cohn_elkies.md) §5–7 — strong-CE = MV-2010
- [delsarte_dual/grid_bound_alt_kernel/REPORT.md](delsarte_dual/grid_bound_alt_kernel/REPORT.md) — 5-phase sweep results
- [delsarte_dual/grid_bound_alt_kernel/variational_pod.py](delsarte_dual/grid_bound_alt_kernel/variational_pod.py) — existing 9-D Chebyshev DE
- [delsarte_dual/mv_construction_detailed.md](delsarte_dual/mv_construction_detailed.md) §3 — MV theoretical ceiling ≈ 1.276
- [delsarte_dual/ideas_beurling_selberg.md:99-111](delsarte_dual/ideas_beurling_selberg.md) — BS rejection
- [delsarte_dual/ideas_pswf.md](delsarte_dual/ideas_pswf.md) — PSWF flagged untested
- [lasserre/THREEPOINT_FULL_REPORT.md](lasserre/THREEPOINT_FULL_REPORT.md) — Idea 6 empirical death
- arXiv search: no 2019 Mossinghoff-Vaaler paper exists; arXiv:1903.08731 is Barnard-Steinerberger

---

# Empirical closure (2026-04-28) — Idea 1 dead, PSWFs (Idea 3 surviving angle) dead

After the audit, two cheap deterministic tests were run locally to convert the
audit's predictions into hard numerical evidence. **Both tests were negative,
in the exact direction the audit predicted.**

## E1. Idea 1 (MO 2.17) — formally closed by direct numerical run

Ran `python delsarte_dual/mv_lemma217.py` and `python -m
certified_lasserre.mo_cuts` on 2026-04-28.

### `mv_lemma217.py` — MV master inequality two-moment + Lemma 2.17

```
M_mv_single_moment (target 1.27481)        = 1.27483925805316129039946613143
M_two_moment_b1 (no Lemma 2.17)            = 1.27483811647221449507955039316
M_two_moment_lemma217                      = 1.27483811647221449507955039316
```

**Lift from MO 2.17 = 0 to ALL printed digits** (≥ 30 decimal places).
The two-moment bound itself even slightly *regresses* from the single-moment
1.274839 → 1.274838, so the additional MO 2.17 constraint contributes
**numerically zero** above the already-saturated MV extremizer. This is
exactly what the audit predicted in §B (the magnitude shadow $2 r_1 + r_2 \ge 1$
is inactive at $r_1, r_2 \approx 0.5$).

### `certified_lasserre.mo_cuts` — Lasserre val(8) baseline + MO 2.17 cut

```
[baseline]      val = 0.317431  (in 0.4s, status=optimal)
[with MO 2.17]  val = 0.317431  (in 0.4s, status=optimal)

Lift from MO 2.17:  -0.000000  (positive = better lower bound)
```

**Lift from MO 2.17 = exactly 0.000000.** The cut adds a constraint
that does not bind. Note: val(8) here is in the Lasserre lifted form
(0.317 ≈ a normalized val); the lift result is what matters and it is zero.

**Verdict for Idea 1**: **CLOSED**. Empirically zero lift in both the MV-master
formulation and the Lasserre formulation. The audit's $P \approx 3$–$6\%$ is
revised downward to **≤ 1%** (not strictly zero only because a different
parameterization could in principle find a tiny residual). **No further
work justified.**

## E2. PSWF kernel family (Idea 3 surviving angle) — implemented, swept, decisive failure

Implemented [delsarte_dual/grid_bound_alt_kernel/pswf.py](delsarte_dual/grid_bound_alt_kernel/pswf.py)
(K12 family): leading prolate spheroidal wave function $\psi_0^c$ via Bouwkamp
tridiagonal eigenvalue solve in normalized Legendre basis (even parity, 120
basis functions); reuses the existing MV M_cert pipeline ($k_1, K_2, S_1$,
$M$-bisection) verbatim. Sanity-checked: at $c=0$, $\psi_0$ is constant, $\phi$
uniform, $K = $ triangle, $M_\mathrm{cert} = 1.2161$ (matches the existing
31-kernel-sweep triangle baseline); $\chi_0 = 0$ and $\|\beta\|_2^2 = 1$
correctly.

### Sweep results

Ran 61-point coarse sweep $c \in [0, 30]$ + 41-point fine sweep around the
best (saved to `data/pswf_sweep.json`):

| $c$ | $M_\mathrm{cert}$ | $k_1$ | $K_2$ | $\chi_0$ |
|---|---|---|---|---|
| 0.00 (= triangle) | **1.21612** | 0.9389 | 4.92 | 0.000 |
| 0.50 | 1.21506 | 0.9396 | 4.95 | 0.082 |
| 1.00 | 1.21201 | 0.9415 | 5.03 | 0.319 |
| 2.00 | 1.20071 | 0.9482 | 5.36 | 1.128 |
| 5.00 | 1.15850 | 0.9689 | 7.02 | 4.195 |
| 10.00 | 1.12074 | 0.9828 | 9.52 | 9.228 |
| 15.00+ | infeasible (S_1 blow-up: $\hat\phi$ has zeros at integer $j/u$) |

**The sweep is monotone-decreasing in $c$** until infeasibility kicks in.
Fine sweep confirmed: best PSWF is **c=0 (= triangle)** at $M_\mathrm{cert} = 1.21612$.

### Comparison

| Kernel | $M_\mathrm{cert}$ | vs arcsine baseline |
|---|---|---|
| **Best PSWF (c=0 = triangle)** | **1.21612** | **−0.05345** |
| Arcsine baseline (this pipeline) | 1.26957 | reference |
| MV published (full precision) | 1.27481 | +0.00524 |
| CS record (target) | 1.28020 | +0.01063 |

**PSWFs lose to arcsine by 5.3% and to MV by 4.6%.** No value of $c$ comes
within 0.05 of the arcsine baseline.

### Why PSWFs fail (mechanism)

As $c$ grows, $\psi_0^c$ becomes **time-concentrated near $x=0$** (it
maximizes Fourier-energy concentration in $[-c, c]$ given time support
$[-1, 1]$). For our problem the extremal $\phi$ should be **boundary-
concentrated** — arcsine's $1/\sqrt{(\delta/2)^2 - x^2}$ singularity at
$x = \pm \delta/2$ is exactly opposite to PSWF's center-concentration.

Numerically:
- $k_1$ gains modestly: 0.9389 (uniform) → 0.989 (highly concentrated), only **+5%**
- $K_2$ explodes: 4.92 → 12.04, **+145%**

The MV master inequality has $K_2$ in a way that hurts the bound when
$K_2$ grows; PSWF's center-concentration trades a small $k_1$ gain for a
large $K_2$ loss. Arcsine is on the opposite Pareto corner: max $\hat\phi$
mass at low frequency PER UNIT $K_2$ via the boundary singularity (which
is locally non-$L^2$ but the auto-conv $K = \phi*\phi$ is in $L^2$ with
the celebrated $0.5747/\delta$ MO surrogate).

PSWFs are simply on the wrong corner of the bias–energy tradeoff for this
problem.

### Verdict for PSWF (Idea 3 surviving angle)

**CLOSED**. Best PSWF is 5.3% below arcsine; family monotone in the wrong
direction. The audit's "$P = 5$–$10\%$" estimate for PSWFs is revised
downward to **< 1%**. The PSWF flag in [delsarte_dual/ideas_pswf.md](delsarte_dual/ideas_pswf.md)
should be marked tested-and-rejected.

## E3. Updated audit table

| # | Idea | Audited P | **Empirical P (post-test)** | Status |
|---|------|-----------|-----------------------------|--------|
| 1 | MV + MO 2.17 + 2D BnB | 3–6% | **≤ 1%** | CLOSED — zero lift |
| 2 | MV-2019 / Madrid-Ramos minorant | 3–7% | 3–7% (unchanged; not tested) | open but cited paper does not exist |
| 3a | Large-scale kernel search | 3–5% | 3–5% (unchanged) | already done — arcsine wins |
| 3b | **PSWFs (Idea 3 untested family)** | 5–10% | **< 1%** | **CLOSED — empirically tested** |
| 3c | Re-derive MO surrogate constant | 5–15% | 5–15% (unchanged; not tested) | most promising remaining |
| 4 | C-S stability slack | 2–8% | 2–8% (unchanged) | logic flaw unresolved |
| 5 | Carneiro-Gonçalves | 3–8% | 3–8% (unchanged) | rejected in repo |
| 6 | 3-point SDP | (closed) | (closed) | empirically dead |

## E4. The remaining viable program

After Ideas 1, 3a, 3b, 6 are now empirically dead, and Ideas 2, 4, 5 have
fatal mathematical issues, the **only remaining unrefuted directions** are:

1. **Re-derive MO's $\|K\|_2^2 < 0.5747/\delta$ surrogate constant.** Pure
   analysis (no compute), would lift ALL kernels including arcsine. **5–15% probability.**
2. **Continue val(d) Lasserre at higher d.** The existing main pipeline.
   Empirical traction confirmed by repo's main lasserre/ infrastructure.

Everything else in `poss_ideas.md` and `new_ideas.md` is either closed,
tested-and-failed, or has a fatal mathematical issue.

## E5. Files & artifacts

- [delsarte_dual/grid_bound_alt_kernel/pswf.py](delsarte_dual/grid_bound_alt_kernel/pswf.py) — PSWF kernel module (Bouwkamp + MV pipeline integration; ~340 lines)
- `data/pswf_sweep.json` — full sweep results (61 coarse + 41 fine c-points, n_basis=120)
- `data/pswf_sweep_log.txt` — stdout log of the sweep run
- mv_lemma217.py and mo_cuts.py outputs above are reproducible: rerun the
  two commands in §E1 and the numbers will match.
