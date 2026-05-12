# Master Plan — Beating C_{1a} ≥ 1.28

*Synthesised from 5 research agents: MO framework, CS 2017, post-2017
literature, SDP hierarchy design, multi-moment extension.*

---

## Synthesised reality check

| Claim | Status |
|---|---|
| Lower bound 1.28 stood untouched since 2017 | **Yes** (Agent: literature) |
| 1.2802 Xie 2026 claim has no certificate | **Yes** (Agent: literature) |
| No prior SDP/Lasserre for C_{1a} | **Yes** (Agent: SDP) |
| Kernel K = β*β/δ with $\|K\|_2^2 = 0.5747/\delta$ | **Confirmed numerically** (∫J_0^4 = 0.574541) |
| MV's 1.276 ceiling is NOT rigorous (numerical only) | **Yes** (MO agent citing MV p.5 line 258) |
| Multi-moment extension caps at ~1.2755 | **Yes** (Multi-moment agent) — **my earlier optimism was wrong** |
| MO 2004 Lemma 2.17 untouched by MV: $\text{Re}\,\hat f(2) \le 2\text{Re}\,\hat f(1) - 1$ | **Yes** (MO agent) |
| MV's Hölder step has 2.266× slack (MO Conj 2.9) | **Conjectured**, not proved |
| CS 2017 uses a RICHER dual family than MV | **Yes** (CS agent) |

---

## The three real paths to > 1.28

### Path α — **MO-Lemma-2.17-enhanced dual** highest-EV
**Idea.** MV pulled out only $z_1 = |\hat f(1)|$. MO 2004 gives an additional
LINEAR constraint (Lemma 2.17):
$$\text{Re}\,\hat f(2) \le 2\,\text{Re}\,\hat f(1) - 1.$$
MV's argument depends on $(z_1, z_2) = (|\hat f(1)|, |\hat f(2)|)$ through
$k_1 = |J_0(\pi\delta)|^2$ and $k_2 = |J_0(2\pi\delta)|^2$. Incorporating
Lemma 2.17 gives a tighter joint constraint.

**Projected lift**: MO agent says 1.27481 → **[1.28, 1.30]** (their own
projection). Margin over CS's 1.28 is small but positive.

**Rigor**: Lemma 2.17 is proved in MO 2004. No numerical leap of faith.
Only the QP on G coefficients stays the same.

**Effort**: 3–5 days to derive + implement + verify.

### Path β — **Rigorous Lasserre SDP** novel-territory
**Idea.** First-ever SDP for C_{1a}. SDP agent's design: 3-cube moment
lifting $(x, y, t) \in [-\tfrac14,\tfrac14]^2 \times [-\tfrac12,\tfrac12]$ with
bivariate $F(x,y)$, test measure $\nu(t)$, Christoffel-Darboux kernel
replacement for $\delta(x+y-t)$.

**Projected bound at level k=8, N=40**: $\lambda \approx 1.28 \pm 0.02$.
Fits on the pod. **Might** beat CS, might not.

**Rigor**: SDP + interval arithmetic produces a certified ball. Clean.

**Effort**: 1–2 weeks for first working prototype; 2–4 weeks to tune.

### Path γ — **Hybrid α + β**
**Idea.** Feed MO's moment constraints (Lemmas 2.14, 2.17) as auxiliary
linear constraints into the Lasserre SDP. Every constraint tightens the
feasible cone; the SDP bound rises.

**Projected lift**: could break 1.30 reliably.

**Effort**: Path α (1 week) THEN add to Path β (1 week) = 3 weeks total.

---

## Why the other directions are lower priority (post-research)

| Direction | Agent finding | Verdict |
|---|---|---|
| **Multi-moment (A)** | Caps at 1.2755 (corrected; my initial derivation wrong) | Insufficient |
| **Alternative K (B)** | No new insight from agents; MV explicitly say K optimal | Deprioritise |
| **Hölder instead of C-S (C)** | 2.266× slack per MO Conj 2.9 IF provable | Bonus; tricky |
| **Non-cos G basis (F)** | Not supported by agent findings; MV's cos + J_0² identity is clean | Keep cos |
| **MV + CS hybrid (E)** | CS is already a dual; merging needs careful joint formulation | Path β subsumes |
| **AI-driven search (I)** | All 2025-26 work is primal (upper bd); nobody has applied to dual | Could be parallel track, not first |

---

## Detailed attack plan

### Week 1 — Path α prototype

Day 1-2:
- Read MO 2004 Lemmas 2.14, 2.17, Prop 2.11 in detail.
- Write `delsarte_dual/mo_lemma_bounds.py` implementing the constraints.
- Unit tests for each lemma.

Day 3-4:
- Extend `mv_bound.solve_for_M_with_z1` to jointly handle $(z_1, z_2)$ with MO Lemma 2.17.
- Numerical eval: plug MV's 119 coefs in, see if the bound jumps.

Day 5-6:
- Re-solve the QP with the new $(z_1, z_2)$ framework (may need re-optimising G).
- Report bound.

Day 7: Rigorous interval-arithmetic verify. Target: certified [1.28+, 1.30-] range.

**Gate 1: If Path α certifies > 1.28 → WE HAVE A PUBLISHABLE RESULT. Stop here, write paper.**

### Week 2-3 — Path β (SDP) prototype

Day 8-12:
- Implement SDP agent's 3-cube Lasserre formulation at level k=4, N=10 for
 smoke testing (~minutes to solve).
- Verify against known val_d=8 value of 1.205.
- Scale to k=8, N=40 on pod.

Day 13-17:
- Tune. Likely produces 1.27-1.29 range.
- Rigorous interval SDP via Arb / MOSEK certification.

**Gate 2: If Path β certifies > 1.28 → second publishable route.**

### Week 4+ — Path γ (hybrid) if neither α nor β breaks 1.28

- Integrate MO's Lemma 2.14, 2.17 + Paley-Wiener Lemma 3.4 as LINEAR SDP
 constraints on the 3-cube Lasserre lift.
- Target: [1.28, 1.31] range, maximally rigorous.

---

## Fail-safes / abort conditions

- If Path α collapses to ≤ 1.2755 (contradicting MO agent's projection):
 the projection was optimistic; re-evaluate. Specifically, the 2.266×
 Hölder slack (MO Conj 2.9) is KEY — verify it separately.
- If Path β SDP at level 8 gives < 1.26: the formulation is loose; re-dualise.
- If all three paths fail to exceed 1.28: PUBLISH THE RIGOROUS PIPELINE ITSELF.
 No one has built certified SDP-based bounds for C_{1a}; even matching 1.28
 with a cleanly certified interval is novel.

---

## Publishable fallback (guaranteed): Rigorous MV certificate

We already have float/mpmath reproduction of MV's 1.2748. Upgrade to FULL
interval-arithmetic certification (`python-flint` arb) of all of:
- The 119 coefficients $a_j$ (treat as exact rationals).
- The Bessel values $J_0(\pi j \delta/u)$ for j=1..119.
- The sum $S_1 = \sum a_j^2/|J_0|^2$ as a ball.
- The constant $\|K\|_2^2 = \int J_0(\pi\xi)^4 d\xi$ as a ball.
- The Min $G$ on $[0, 1/4]$ via interval B&B.
- The quadratic solution for M.

**Output**: "The Matolcsi-Vinuesa bound 1.2748, verified with 200-digit
certified interval arithmetic" — a solid minor contribution that's 100%
safe to publish.

---

## Meta: timing and resources

- Pod: $106 budget, 192-core EPYC 9655, available.
- Budget used: $0.19 of $106 so far.
- Mathematical understanding: **complete** (all key papers read by agents).
- Implementation velocity: MV reproduction took ~2 hours from theory to
 validated. Path α should be similar.

**Expected timeline: ~3–4 weeks to definitive answer.**

---

## Closing observation

The combination of three findings is very favourable:
1. **No one has tried SDP/Lasserre on C_{1a}** — novel territory.
2. **MO has identified extra constraints (Lemma 2.17) MV never used** — concrete lever.
3. **1.28 has been static for 8 years** — due for a push.

Path α is the **minimum viable research step**: low effort, immediate
answer. Path β is the **novel contribution**: SDP approach is the modern
upgrade of the whole framework. Either provides a publishable result. Both
combined may beat the record decisively.
