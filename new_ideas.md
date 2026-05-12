# New Ideas to Push $C_{1a} > 1.2802$ — STRICT-RE-AUDITED

> **Final survival count: 1 / 5.**
>
> A maximally strict re-audit of the prior 3 surviving ideas found mathematical
> flaws in 2 of them. Only Idea 3 (3-point SDP) survives all four gates: math
> valid + ≤\$2k budget + hard adjacent precedent + nonzero probability of beating
> 1.2802.

---



## SURVIVING — Idea 3: Three-point SDP duality (Cohn-de Laat-Salmon)

### Mechanism

Standard MV / Lasserre / Delsarte duals use the **2-point correlation cone**
$$\bigl\{\rho_2(x) = \int f(u)f(u+x)\,du \;:\; f \geq 0,\; \mathrm{supp}(f) \subset [-1/4, 1/4],\; \int f = 1\bigr\}.$$
Cohn-de Laat-Salmon (arXiv:2206.15373) introduce the **3-point correlation cone**
$$T_f(x, y) := \int f(u)\,f(u+x)\,f(u+y)\,du, \qquad x, y \in [-1/4, 1/4],$$
and impose PSD constraints on $T_f$'s moment matrix. For nonneg $f$ on $\mathbb{R}$, $T_f$ is **NOT** determined by $\rho_2$ — different $f$'s with the same $\rho_2$ can have distinct $T_f$. Hence the 3-point cone is strictly smaller than the 2-point cone.

For $C_{1a}$: the autoconvolution objective $\sup_t (f*f)(t)$ depends on $f \otimes f$, which contains more information than $\rho_2$. Adding PSD constraints on $T_f$ to the dual SDP strictly restricts the adversarial $f$, raising the rigorous lower bound on $C_{1a}$.

### Mathematical validity

✅ $T_f \geq 0$ pointwise, $T_f$ is a 3-moment of $f^{\otimes 3}$, and Hausdorff-PSD conditions on its moment matrix are derivable in a degree-$D$ basis. Strict cone containment is proven (Cohn-de Laat-Salmon §3-4). Rational rounding via Magron-Henrion 2022 (SIAM J. Optim., 10.1137/21M1422574) preserves rigor.

### Feasibility (within \$200–\$2k strict budget)

- 1D autoconvolution: symmetry $S_3$ on triples + reflection acts on a 2D fundamental domain $\{(x, y) : 0 \le x \le y \le 1/2\}$.
- Degree truncation $D = 20$–$30$: matrix sizes $\sim 100$–$300$ (smaller than spherical-packing case).
- MOSEK on a single c5.24xlarge spot ($1.30/hr) for 24h ≈ **\$30 raw compute; \$300–500 with engineering iterations.**
- Implementation: 1–2 weeks extending `lasserre/fourier_sdp.py` with a 3-point block + $S_3$ reduction.

### Hard adjacent precedent

- **Cohn-de Laat-Salmon arXiv:2206.15373** achieved 3–10% lift on sphere-packing density bounds in dimensions 4–7, with the SDP cone provably tighter than the LP cone in dimensions 12, 16. This is a **proven** mechanism, not analogy.
- Translated quantitatively: a 3% lift on $C_{1a}$ gives 1.32; a 0.5% lift gives 1.286. Either clears 1.2802.
- **No prior work** has plumbed the 3-point cone for autoconvolution (verified across `delsarte_dual/literature_survey_2020_plus.md` and the literature search in Round 2).

### Risks (honest)

(a) 1D autoconvolution may have more 2-point structure than higher-dim sphere packing, so 3-point lift could be smaller than the 3% sphere-packing precedent — honest expected lift 0.005–0.03 (still beats 1.2802 even at the bottom).
(b) Asymmetry of $f * f$ vs $\rho_2 = f * f^-$ requires care; symmetric 3-point analysis ($T_f$ on triples) may need extension to 4-point for the asymmetric autoconvolution objective.
(c) $S_3$ symmetry reduction on triples (vs $\mathrm{SO}(n)$ in sphere packing) is conceptually simpler but requires careful setup.

### Probability of beating 1.2802 (honest)

**30–45%** — moderate but well-supported. The only candidate with a hard, mechanism-level prior-art lift in an analogous extremal problem.

---

## Audit summary

| Idea | Math | Budget | Precedent | Verdict |
|---|---|---|---|---|
| Per-cell small-d SDP | ❌ uniform-within-bins minimizes peaks; per-cell SDP ≤ CS step-function value | n/a | n/a | **REMOVED** |
| Adaptive-resolution cascade | ❌ worst-case slack $= 1/m_c$ unchanged for adversarial $f$ | n/a | n/a | **REMOVED** |
| 3-point SDP (Cohn-de Laat-Salmon) | ✅ 3-point cone strictly smaller than 2-point | ✅ \$300–500 | ✅ 3–10% lift in sphere packing dim 4–7 | **KEEP** |

## Honest conclusion

After the strictest possible audit, **only one idea survives**. This matches the
independent finding in [poss_ideas.md](poss_ideas.md) — the user's previous
sweep of 22 frameworks ended at zero survivors. The 1.2802 record has stood for
9 years for a structural reason: every mathematical framework that produces
rigorous lower bounds on $\|f*f\|_\infty$ for non-log-concave admissible $f$
either has documented framework ceilings at or below 1.28, or transfers in the
wrong direction (yielding upper bounds), or requires SDP scales the user has
declared infeasible.

The 3-point SDP is the only direction with (i) a proven mechanism (cone
strictly smaller than 2-point), (ii) hard adjacent precedent (sphere-packing
3–10% lift, Cohn-de Laat-Salmon 2022), (iii) compute that fits the budget
(\$300–500), and (iv) novelty (no prior work on autoconvolution).

## References

- Cohn-de Laat-Salmon, "Three-point bounds for sphere packing": [arXiv:2206.15373](https://arxiv.org/abs/2206.15373)
- Cloninger-Steinerberger 2017: [arXiv:1403.7988](https://arxiv.org/abs/1403.7988)
- Matolcsi-Vinuesa 2010: [arXiv:0907.1379](https://arxiv.org/abs/0907.1379)
- Magron-Henrion rational SOS rounding (SIAM J. Optim. 2022): [10.1137/21M1422574](https://epubs.siam.org/doi/10.1137/21M1422574)
- Cohn 2024 Bourbaki survey: [arXiv:2407.14999](https://arxiv.org/abs/2407.14999)
- Cross-reference: [poss_ideas.md](poss_ideas.md) — 22 rejected frameworks, structurally aligned with this audit's conclusions.

---

# Full Audit (2026-04-28) — Four Independent Agents

This section is appended after a four-agent independent audit. Findings are
**substantially less optimistic** than the body above; several specific claims
are corrected, and the recommended path forward is replaced with a cheap
**scoped pilot** rather than the full $300–500 deployment.

## A. Verdicts at a glance

| Agent | Mandate | Verdict | $P(\text{beat } 1.2802)$ |
|---|---|---|---|
| **A1 — Math audit** | Rigorous SDP/moment-SOS expert | **Negative.** Claim of strict-cone-containment is real *in isolation* but largely subsumed by sufficiently-high-degree single-measure Lasserre; sphere-packing analogy is a category error because the LP cone defect (no triples) does not exist in the existing Sidon Lasserre dual. | <5% |
| **A2 — CDS paper deep-dive** | Read arXiv:2206.15373 + follow-ups | **Cautiously negative.** Numerical lift is 0.8–5% over plain Cohn-Elkies LP, **0.1–2% over the strengthened 2-point bound** — not "3–10%". CDS does NOT prove strict cone containment. SO(n)-equivariance is the entire reason the SDP is tractable; 1D autoconvolution has only $S_3 \times \mathbb{Z}_2$ (≤6-element group), qualitatively weaker. | 5–15% |
| **A3 — Local repo prior-art** | Search the codebase | **No prior 3-point work**, but the existing 3-cube Lasserre design ([delsarte_dual/ideas_lasserre_sos.md](delsarte_dual/ideas_lasserre_sos.md)) already uses joint moments $y_{\alpha\beta\gamma}$ on $\mu\otimes\mu\otimes\nu$. A genuine 3-point object would be on $\mu^{\otimes 3}$, which is distinct from the existing design. | n/a |
| **A4 — Literature search** | k-point on 1D autoconvolution | **Cautiously positive on novelty.** Zero applications of k-point SDP to 1D autoconvolution / Sidon constants exist. de Laat–Vallentin lineage (arXiv:1311.3789, 1812.06045, 2206.15373) confirms k-point hierarchy is, **at finite truncation**, a strictly stronger relaxation than the 1-measure Lasserre at the same degree. | 10–20% |

**Reconciled verdict:** the original 30–45% estimate is not defensible. The
honest range is **8–18%**, with the upper end requiring (i) the 3-point block
to be implemented as a genuinely new $\mu^{\otimes 3}$ tensor (not just a
re-indexing of existing moments), (ii) a working $S_3 \times \mathbb{Z}_2$
symmetry reduction, and (iii) a degree high enough to see lift but not so high
that the budget breaks.

## B. Specific corrections to the original write-up

| Claim in body | Correction |
|---|---|
| "3–10% lift on sphere-packing density" | **0.8%–5% over plain CE-LP**; **0.1%–2% over the strengthened 2-point bound (CHKLS / Cohn-Triantafillou)**, per Table 1.1 of [arXiv:2206.15373](https://arxiv.org/abs/2206.15373). |
| "Strict cone containment is proven (Cohn-de Laat-Salmon §3-4)" | **Not proved in CDS.** §3-4 establishes duality and feasibility, not strict separation. Strict containment is conjectural and observed numerically. CDS themselves note: *"It is unclear whether the three-point bound is ever sharp in any case not already resolved by the two-point bound."* |
| "$S_3$ symmetry reduction on triples is conceptually simpler" | $S_3 \times \mathbb{Z}_2$ is a ≤6-element group. SO(n)-equivariance gave CDS continuous-family Schoenberg/Gegenbauer block-diagonalization with $O(d)$ blocks of size $O(d^2)$. **The 1D analogue has no comparable reduction**; a single dense $\binom{d+2}{2}$-block remains, with at most a 6× constant factor. The 100–300 matrix size estimate is plausible only at $d = 10$–$14$, not $d = 20$–$30$. |
| "1–2 weeks extending `lasserre/fourier_sdp.py` with a 3-point block" | A1's assessment: "implementing a triple-moment SDP with $S_3$ symmetry reduction at degree 20–30 would be a multi-month research project." **Realistic effort: 4–8 weeks** to a working pilot at low degree; longer to scale. |
| "$300–500 with engineering iterations" | Plausible only for a low-degree ($d \le 14$) pilot. Full-scale ($d = 20$–$30$) attempt is at the upper end of the $\le \$2k$ budget with no slack, given the lack of SO(n)-style symmetry reduction. |
| Risk (b): "Asymmetry of $f*f$ vs $\rho_2 = f*f^-$ ... may need extension to 4-point" | This is the **most serious technical issue**, not a side risk. The autoconvolution $(f*f)(t) = \int f(u)f(t-u)du$ depends on a *shifted* product; the symmetric $T_f(x,y) = \int f(u)f(u+x)f(u+y)du$ does not directly bound it. The natural object linking $T_f$ to $\sup_t (f*f)(t)$ is a 4-argument quantity $\int f(u)f(u+x)f(t-u-y)du$. A purely symmetric 3-point block may give weak (or no) lift on the asymmetric objective. **This deserves explicit derivation before any compute is spent.** |

## C. The technical disagreement (A1 vs A4) and what decides it

A1 argues: the moment matrix $M_d(\mu)$ at degree $2d$ on a single measure $\mu$
encodes ALL monomials $\mu^\alpha$ with $|\alpha| \le 2d$, hence implicitly
encodes 3-point joint moments $\int x^a d\mu \cdot \int y^b d\mu \cdot \int z^c d\mu = m_a m_b m_c$.

A4 argues: the de Laat-Vallentin k-point SDP hierarchy
([arXiv:1311.3789](https://arxiv.org/abs/1311.3789),
[arXiv:1812.06045](https://arxiv.org/abs/1812.06045)) is a **strictly stronger
relaxation** at finite truncation than the 1-measure Lasserre at the same
degree, because it constrains a separate $\mu_k \in \mathcal{M}_+(X^k)$ measure
with marginal-consistency, not just rank-1 products of $\mu$'s moments.

**The reconciliation:**

- A1 is correct that for the EXISTING $f$ (rank-1 product), all $r$-point joint
  moments are determined by 1-point moments. So at the *primal* level (an
  honest measure $\mu$), no information is added.
- A4 is correct that the **relaxation** drops the rank-1 constraint. The k-point
  SDP allows the relaxation variable $\mu_3$ to be a non-product PSD-symmetric
  joint object. PSD constraints on $\mu_3$'s moment matrix are NOT implied by
  PSD constraints on $\mu$'s moment matrix at the same degree, because the
  former operates on a richer (non-rank-1) feasible set.
- **Decisive question** (open and worth 1–2 days of analytic work): for the
  **specific objective** $\sup_t (\mu*\mu)(t)$, does the 1-measure Lasserre at
  degree $2d$ give the same bound as 1-measure Lasserre at degree $2d$ +
  $\mu^{\otimes 3}$ block at the same degree? This is decidable by writing out
  the dual SOS certificates and seeing whether the 3-point block is forced into
  the $M_d$ block by Putinar Positivstellensatz, OR is genuinely additional.

If the 3-point block is forced (A1's view holds for this objective):
**$P(\text{beat } 1.2802) \to <5\%$**.

If the 3-point block is genuinely additional (A4's view holds): **$P \to 15$–$25\%$**,
contingent on (i) the 1D symmetry reduction working and (ii) the low-degree
pilot showing lift.

## D. Existing infrastructure overlap

The repo already has, in [delsarte_dual/ideas_lasserre_sos.md](delsarte_dual/ideas_lasserre_sos.md):

- A 3-cube Lasserre design on $\mu\otimes\mu\otimes\nu$ (two copies of $\mu$, one
  test measure $\nu$).
- Convergence rate model $C_{1a} - \lambda_{k,N} \asymp c_1/k^2 + c_2/N$.
- Complexity table: $k=8$ → matrix size 165, svec 13,695; "target $>$ 1.2802".

A genuine 3-point augmentation would add a $\mu^{\otimes 3}$ block (separate
moment matrix on triples). This is **distinct** from the existing $(x,y,t)$
3-cube — the existing cube uses only TWO $\mu$-copies. **The novelty claim
survives, but the gap is smaller than originally written.**

The honest framing: this is a **k-point augmentation of the existing 3-cube
Lasserre** — adding a $\mu^{\otimes 3}$ moment block alongside the existing
$\mu \otimes \mu \otimes \nu$ block.

## E. Replaced recommendation: scoped pilot, not full deployment

Original plan: spend $300–500 + 1–2 weeks → full 3-point SDP at degree 20–30.

**Replacement plan (defensible against the audit):**

1. **Week 1 — Analytic check (no compute).** Derive the dual SOS certificate
   structure for 1-measure Lasserre + $\mu^{\otimes 3}$ block on the
   autoconvolution objective. Determine whether the $\mu^{\otimes 3}$ block is
   forced into $M_d$ by Putinar (A1's claim) or genuinely additional (A4's
   claim). **Cost: $0.** Decisive.

2. **Week 2 — Asymmetry derivation.** Write out the explicit functional linking
   the symmetric $T_f(x,y)$ to $\sup_t (f*f)(t)$. If the link requires a
   4-point object, the symmetric 3-point block alone gives no lift on the
   objective. **Cost: $0.** Also decisive.

3. **Week 3 — Low-degree pilot, IF (1) and (2) survive.** Implement the 3-point
   block at $d = 6$–$10$, compute baseline 2-point Lasserre at the same $d$ and
   compare. **Cost: $50–150** on a single MOSEK pod. Run the same baseline
   without the 3-point block as a control.

4. **Week 4–6 — Scale-up, IF the pilot shows lift.** Move to $d = 14$–$18$ with
   $S_3 \times \mathbb{Z}_2$ block-diagonalization. Budget ceiling: $500 total
   including the pilot. If no lift appears at $d = 6$–$10$, **kill the idea**.

This keeps the worst-case loss to a 1-week analytic effort + ≤$150 pilot,
while preserving the upside if the 3-point block is genuinely additional.

## F. Honest probability of beating 1.2802

| Conditional | Probability |
|---|---|
| Path through analytic check (D.1) — 3-point block is genuinely additional | ~50% (literature favors A4 for k-point hierarchies in general; Sidon-specific case is open) |
| Conditional on (i): asymmetry handled by symmetric $T_f$ alone (no 4-point needed) | ~40% (A1, A2, body all flag this concern) |
| Conditional on (i)+(ii): pilot at $d=8$ shows ≥ 0.05% lift over baseline 2-point Lasserre at same $d$ | ~50% |
| Conditional on (i)+(ii)+(iii): scaled $d=18$ run beats 1.2802 | ~50% |

Multiplying: $0.5 \times 0.4 \times 0.5 \times 0.5 \approx 5\%$ unconditional.

If we condition on the **best-case** reading of the literature (A4 is right and
the 1D analog of CDS does transfer): top end ~18%.

**Honest range: 5–18%, midpoint ≈ 10%**, vs. the body's 30–45%. The body's
estimate is overstated by a factor of 2–4×.

## G. Sources consulted by the audit

Agent 1 (math audit): inspected `lasserre/dual_sdp.py`, `lasserre/core.py`, and
`new_ideas.md`; cross-referenced arXiv:2206.15373 abstract.

Agent 2 (CDS deep dive): [arXiv:2206.15373](https://arxiv.org/abs/2206.15373)
(ar5iv render), [Bachoc-Vallentin math/0608426](https://arxiv.org/abs/math/0608426),
[de Laat-Machado-Oliveira-Vallentin 1812.06045](https://arxiv.org/abs/1812.06045),
[Gonçalves-de Laat-Leijenhorst 2303.01095](https://arxiv.org/abs/2303.01095),
Cohn's MIT sphere-packing data tables.

Agent 3 (repo): `delsarte_dual/ideas_lasserre_sos.md`, `ideas_cohn_elkies.md`,
`ideas_sdp_relax.md`, `ideas_cs_extensions.md`, `creative_audit.md`,
`MASTER_PLAN.md`, `IMPROVEMENT_LIST.md`.

Agent 4 (literature, 21 CDS citations + 14 CS citations checked):
- k-point machinery (none applied to autoconvolution): arXiv:1311.3789, 1610.04905, 1812.06045, 2206.15373, 2303.01095, 2211.16471
- $C_{1a}$ / $\nu_2$ state of the art (all 2-point): arXiv:0907.1379, 1403.7988, 1903.08731, 2003.06962, 2004.06611, 2106.13873, 2210.16437, 2506.16750, 2602.07292, 2512.18188

## H. One-line takeaway

The 3-point SDP is not a $300, 1-2 week, 30-45% bet. It is, **at best**, a
$50–500 staged bet at ~10% honest probability, contingent on a 1-week analytic
check that determines whether the entire premise survives the asymmetric-
objective and Putinar-subsumption issues — both of which the body of this
document either dismisses or does not address.
