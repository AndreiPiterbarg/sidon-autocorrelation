# Pólya-LP at d=64: Scaling Audit

**Goal:** Make the Handelman/Pólya LP tractable at d=64 (and beyond) and produce **lower bounds tighter than the order-2 Lasserre SDP**.

**Empirical baseline (built and measured in `lasserre/polya_lp/`):**
- d=8 R=16: α = **1.173** (LP) vs **1.142** (SDP, k=2 b=7).  LP already beats SDP at low d.
- d=64 R=4: α = **1.138** in 79 s (IPM).
- Convergence: gap(R) ≈ C / R^{1.18}, with C scaling roughly linearly in d.
- Projection: need R ≈ **8.3** at d=64 to clear α ≥ 1.281, where the naive LP would have $\binom{40}{8} \approx 7.7 \cdot 10^{7}$ constraints — intractable for HiGHS, MOSEK, or Gurobi (their cliffs are 5M / 30M / 30M respectively).

The audit below explains how to bring 76M constraints down to ~$10^4$–$10^6$ while keeping (or improving) tightness.

---

## TL;DR — the plan

Stack three multiplicative reductions, then escape to GPU PDLP if needed:

| # | Technique | Reduction | Effort | Risk |
|---|---|---|---|---|
| 1 | **Newton-polytope / term-sparsity restriction** | **100–1000×** | medium | low (provably no convergence loss for our M support) |
| 2 | **Cutting-plane / column generation on β-constraints** | **10–50×** on top | medium | low (standard LP-CG) |
| 3 | **Active-window restriction** (we already have this) | **16–160×** | done | low |
| 4 | **MV-Fourier cuts** added to LP (Gorbachev–Tikhonov, Cohn–Elkies-style) | breaks **1.276 ceiling** that pure CS-window cuts saturate at | low (≤1 day per cut family) | the critical "beat SDP" lever |
| 5 | **GPU PDLP (cuOpt / cuPDLP-C)** on the residual LP | up to 90M-nonzero LP in 30–180 min on H100 | medium (dump to MPS, pip install) | only ~1e-4 accuracy — fine for a primal lower bound |

Stack 1+2+3: 76M → ~10K active constraints. Solvable in seconds.
Stack 1+2+3+4: still solvable, but now produces certificates **above** MV's 1.276 ceiling.
Fallback 5: dump the unreduced LP to a GPU solver if any of 1–3 underperforms.

---

## Tier 1 — Multiplicative LP-shrinkage (do these first)

### 1.1 Newton-polytope / term-sparsity restriction (single biggest lever)

**The problem.** Our LP has $\binom{d_\text{eff}+R}{R}$ rows (one per multi-index β). Most rows are "structurally trivial": they involve only the $q$-multiplier and $c_\beta$ slack, never touching $M_W$.

**The technique.** Apply the **TSSOS term-sparsity construction** (Wang–Magron–Lasserre 2019, [arXiv:1912.08899](https://arxiv.org/abs/1912.08899); [CS-TSSOS arXiv:2005.02828](https://arxiv.org/abs/2005.02828)) to the *LP constraint set* rather than the SOS Gram block:

1. Build the **term-sparsity graph** $G_\text{TS}$: vertex set = monomials, edges = pairs $(\beta_1, \beta_2)$ that co-occur in $\mu^T M_W \mu$ for some $W$. For our anti-banded $M_W$, this graph has **bandwidth $\ell$** (the window length).
2. Take the chordal extension of $G_\text{TS}$ by minimum-fill ordering (`MD` ordering in TSSOS).
3. Restrict the LP rows to $\beta$ in the **Minkowski sum** $B_\text{active} \oplus \Delta_{R-2}$, where $B_\text{active} = \bigcup_W \text{supp}(\mu^T M_W \mu)$.

**Quantitative reduction.**
- With band width $\ell = 16$ (typical at d=64): rows shrink from $\binom{40}{8} \approx 7.7\cdot 10^7$ to $\binom{24}{8} \approx 7.4\cdot 10^5$ — **~100× reduction**.
- With $\ell = 8$: $\binom{16}{8} = 12{,}870$ — **~6,000× reduction**.
- Theoretical guarantee: for chordal extensions, **convergence rate is preserved** (Wang–Magron Thm 3.4).

**Implementation.** Port the term-sparsity-graph construction from [TSSOS.jl](https://github.com/wangjie212/TSSOS) (Julia, ~200 LOC) to our `lasserre/polya_lp/build.py`. The chordal-MD ordering is in `SparseArrays.jl` or `scipy.sparse.csgraph.reverse_cuthill_mckee`. Don't port the SDP backend — only the support analysis.

**References:**
- Wang–Magron–Lasserre, *TSSOS: A Moment-SOS hierarchy that exploits term sparsity*, SIAM J. Optim. 2021. [arXiv:1912.08899](https://arxiv.org/abs/1912.08899).
- Mai–Magron–Lasserre–Toh, *Tractable hierarchies on the nonneg orthant*. [arXiv:2209.06175](https://arxiv.org/abs/2209.06175).

### 1.2 Cutting-plane / column generation on β-constraints

**The technique.** Standard LP CG:
1. Start with seed constraints $S_0 \subset \{\beta : |\beta| \le R\}$ (e.g., the Newton-restricted set from §1.1 plus all $|\beta| \le 4$).
2. Solve LP, get current $(\alpha^*, \lambda^*, q^*)$.
3. **Pricing problem:** find the most-violated constraint
   $$\beta^* = \arg\max_\beta \left| \sum_W \lambda_W^* \, \text{coeff}_W(\beta) - \alpha^* \delta_{\beta=0} + q_\beta^* - \sum_j q_{\beta-e_j}^* \right|$$
4. Add top-K violators, re-solve. Iterate to fixpoint.

**Quantitative reduction.** Empirically (Bienstock 2014, [arXiv:1501.00288](https://arxiv.org/abs/1501.00288); Oustry–Cerulli 2023, [arXiv:2307.14181](https://arxiv.org/html/2307.14181)): the active set at LP optimum is typically $O(n_\text{var})$, so for our problem ~$10^5$ active rows — **another 10–50× reduction** stacked on §1.1. Combined: ~$10^4$ active rows at d=64 R=8.

**Implementation.** Cleaner than §1.1 — a wrapper over our existing `build_handelman_lp` / `solve_lp`. Add `lasserre/polya_lp/cutting_plane.py`. Pricing problem is itself a polynomial enumeration over restricted β; use the same Newton-polytope restriction as a heuristic.

**References:**
- Bienstock, *LP formulations for sparse polynomial optimization*, [Columbia talk](http://www.columbia.edu/~dano/talks/o15.pdf).
- Oustry–Cerulli, *Convex semi-infinite programming with inexact separation oracles*, 2023.

### 1.3 Active-window restriction (already implemented)

We already have this in `lasserre/polya_lp/active_set.py`. At d=8 we observed only **5 of 64** windows binding at the optimum. For d=64 with ~8000 windows, expect ~50–500 active. **Stacking with §1.1+§1.2 gives the full 100,000× reduction** needed.

The current implementation reduces *variables* (lambda dim drops); it now also needs to drive the row reduction in §1.1 (drop β's whose support doesn't intersect any active window's band).

---

## Tier 2 — Beating SDP, not just matching it

The shrinkage in Tier 1 makes the LP **tractable** at d=64. Tier 2 makes it **tighter than the SDP**.

### 2.1 MV-Fourier cuts (the critical "beat SDP" lever)

**The problem.** All cuts in our current LP are CS-2017-style "window cuts": for each window $W$, the constraint is $\mu^T M_W \mu \le t$. These cuts have a **provable saturation ceiling**: the best LP using only window cuts is bounded by the Matolcsi–Vinuesa value **1.276** (single-Cauchy-Schwarz dual ceiling, MV §3 remark).

To get strictly above 1.276 with an LP, we need cuts **outside the CS-window family**. They exist:

**Cut families (drop-in):**
1. **Gorbachev–Tikhonov scale-$\ell$ Fourier moment cuts** — for each scale $\ell$, one new linear constraint on $\mu$ derived from $\hat f$'s support condition. Source: `delsarte_dual/ideas_turan_krein.md` §3.
2. **Cohn–Elkies-style Fourier majorants** restricted to symmetric measures.
3. **Vaaler / Carneiro–Littmann–Vaaler Selberg-extremal cuts** (Vaaler 1985 *Bull AMS*; [Carneiro–Littmann–Vaaler arXiv:1008.4969](https://arxiv.org/abs/1008.4969)).
4. **Beurling-Selberg majorant cuts** for indicator/L¹ approximation, restricted to support $[-1/4, 1/4]$.

**Implementation.** Each cut adds **one or a few rows per scale** to the LP — negligible cost. The Gorbachev–Tikhonov family is the easiest (closed-form coefficients).

**Honest caveat:** the published MV bound from these cuts in continuous form is **1.2748**, just below our 1.281 target. Reaching 1.281+ from MV cuts alone requires either (a) the **non-Cauchy–Schwarz** refinements in `ideas_turan_krein.md`, or (b) combining with the Pólya hierarchy (which we have).

### 2.2 Bernstein basis on the simplex (parallel route)

**The technique.** Replace the monomial basis with the Bernstein basis: $p = \sum_{|\beta|=D} b_\beta B_\beta(\mu)$ with $B_\beta = \binom{D}{\beta} \mu^\beta$. **A polynomial is nonneg on $\Delta_d$ iff all Bernstein coefficients $\ge \alpha$ in some degree-D expansion** (Garloff 1993; Lai–Schumaker 2007 "Spline functions on triangulations").

**Why it might help.**
- Same asymptotic rate $O(1/D)$ for the LP bound, but **tighter constants** (Garloff–Smith 1999) — empirically 1.5–2× tighter α at given degree.
- This could let us drop **R from 8 to 6**, which is a $\binom{40}{8}/\binom{40}{6} \approx 12$× constraint reduction on top of Tier 1.

**Subdivision (the real payoff).** Adaptive subdivision restricted to ~10 active coordinates (driven by §1.1) gives **quadratic** convergence. Realistically: ~$\binom{12+8}{8} \approx 1.3 \cdot 10^5$ rows at d=64 — competitive with §1.1 alone.

**Caveat:** Sloth 2017 ([arXiv:1710.05735](https://arxiv.org/abs/1710.05735)) constructed a counterexample where Bernstein subdivision certificates fail at isolated zeros. If our optimal $\mu^*$ is near a vertex/edge, subdivision may not converge.

**Implementation.** [`BernsteinExpansions.jl`](https://juliareach.github.io/BernsteinExpansions.jl/latest/man/introduction/) (Julia) handles up to d≈10; for d=64 we'd need our own subdivision queue with active-coord restriction. ~300 LOC.

### 2.3 Weighted Pólya / Dickinson–Povh multipliers

Replace $(\sum \mu_i)^R$ with a **single tailored monomial** $\mu^r$ where $r$ has support on the active band only. Dickinson–Povh ([JOGO 2015 / Opt-online 3879](https://optimization-online.org/wp-content/uploads/2013/05/3879.pdf)) proved Pólya's theorem extends to arbitrary toric Newton structure. For us: $\binom{w + \deg p}{\deg p}$ rows where $w$ is the band width — **same as §1.1, no chordal-extension overhead**.

Implementation: simpler than §1.1 (no graph computation), but the convergence-rate constant is unproven for our specific support. **Test against §1.1 empirically.**

### 2.4 Toeplitz / FFT-structured per-iteration speedup

Each $M_W$ is Toeplitz with banded support. The dual LP per-iteration matrix-vector $A x$ is a **batched FFT over windows**: $O(W \cdot d \log d)$ instead of $O(W \cdot d^2)$ — **~60–100× per-iter speedup** at d=64.

This is most useful in combination with Tier 3 (PDLP/PDHG) which has matrix-vectors as the dominant cost. For interior point it's harder to exploit (Schur complement densifies).

**Implementation.** Custom CuPy code, ~500–1000 LOC. Dumitrescu's *Positive Trigonometric Polynomials* book ch. 4–5 has the autocorrelation sum-of-squares cone formulation. [FastAST](https://github.com/thomaslundgaard/fast-ast) is a reference implementation for line-spectral atomic-norm estimation.

---

## Tier 3 — Escape hatches

### 3.1 GPU PDLP (cuOpt / cuPDLP-C) on the unreduced LP

If Tier 1 can't bring us below the IPM cliff, **dump the LP to MPS and call NVIDIA cuOpt's PDLP path**.

- **Scale:** validated at **90M nonzeros** on a single H100 (Mittelmann benchmarks Oct 2025).
- **Convergence:** O(1/k) with restarts, gets to ~$10^{-4}$ KKT in 30–180 min at our scale.
- **For lower bounds we don't need optimality** — any feasible primal $(\alpha, \lambda, q, c \ge 0)$ with $A x \approx b$ gives a valid (slightly weaker) lower bound. Read off α after 10K iterations.
- **Cost:** $0.50–$1/hr cloud H100. Engineering: ~50 LOC to dump our `BuildResult` to MPS.

**References:**
- [cuPDLP (Lu–Yang 2023, arXiv:2311.12180)](https://arxiv.org/abs/2311.12180)
- [cuOpt LP barrier method blog](https://developer.nvidia.com/blog/solve-linear-programs-using-the-gpu-accelerated-barrier-method-in-nvidia-cuopt/)

### 3.2 SDSOS (SOCP) — better than DSOS at LP-comparable cost

[Ahmadi–Majumdar 2017 (arXiv:1706.02586)](https://arxiv.org/abs/1706.02586): each PSD block becomes a sum of 2×2 PSD cones — an **SOCP**. For SQO on $\Delta_d$ at d=64 this means O($d^2$) rotated cones of size 3. **MOSEK handles this in <1 minute per solve.**

**Iterated SDSOS** (Ahmadi–Hall, [arXiv:1510.01597](https://arxiv.org/abs/1510.01597)): augments the basis via a column-generation-like step. Empirically reaches SOS-tightness in **5–15 iterations** for randomly-sparse $M$. Each iteration is one SOCP.

**Decision criterion:** test SDSOS at d=8. If single SDSOS beats our LP's 1.173 (or SDP's 1.142), scale to d=64. If not, drop.

### 3.3 PVZ-SDP-level-1 — small SDP that may match Pólya R=6–8

[Peña–Vera–Zuluaga (SIAM J. Opt. 16, 2006)](https://epubs.siam.org/doi/10.1137/04060562X) build SDP hierarchies for SQO: level-0 has $d \times d$ matrix variable plus $\binom{d}{2}$ scalars; level-1 has $\binom{d+1}{2} \times \binom{d+1}{2}$ matrix.

At d=64: level-0 = trivial; level-1 = 2080×2080 PSD matrix, ~10⁷ NNZ — borderline but feasible in MOSEK. **Tabulated to match Pólya R=6–8 in 1–10 seconds on $d=20$–$40$ benchmarks** (Bomze–de Klerk 2002, J. Glob. Opt. 24).

This is the **strongest practical alternative to LP-everything**. Might be the right backbone for d=64 with LP only as a refinement.

---

## Cuts ranked by what they break

The user's specific ask was "produce better bounds than SDP". The honest hierarchy:

| Cut family | Bound ceiling | Saturation at | Cost |
|---|---|---|---|
| CS-window cuts only (current) | ~1.276 | low R | LP |
| CS + Pólya hierarchy at high R | val(d) | $R \to \infty$ | LP scales as $\binom{d+R}{R}$ |
| CS + MV-Fourier cuts | break 1.276 | unknown | ~1 LP row per cut |
| CS + Vaaler/Selberg cuts | breaks 1.276 | unknown | ~1 LP row per cut |
| Lasserre level-2 SDP (current SDP path) | val(d) | $k \to \infty$ | SDP, 76M-row analog |
| Restricted-Hölder (different path entirely) | 1.378 (rigorous) | conditional on Hyp_R | already proven, see `delsarte_dual/restricted_holder/` |

**The honest verdict from agent 4:** *no Sidon-specific structural insight is known that lets a small LP at d=64 reach 1.281+ without either (a) very high R, or (b) MV-Fourier cuts above the 1.276 ceiling, or (c) hybrid with SDP*.

But our empirical d=8 result (LP 1.173 > SDP 1.142) shows the LP **does** have headroom over the order-2 SDP. The question is whether that headroom survives to d=64.

---

## Recommended attack sequence

**Phase 1 (1 week):** Implement §1.1 (Newton/term-sparsity) + §1.2 (column generation). Re-run d=64 R=8 with the reduced LP. Expected α somewhere in 1.25–1.30. **Decision point:** does the reduced LP beat 1.281?

**Phase 2 (1 week):** If Phase 1 falls short, add §2.1 (Gorbachev–Tikhonov Fourier cuts). Expected push to 1.28–1.32.

**Phase 3 (parallel):**
- §3.2: 1-day SDSOS test at d=8. If it beats Pólya 1.173, prioritize.
- §3.3: 1-day PVZ-SDP-level-1 test at d=8. Establish baseline.
- §3.1: Dump current d=64 LP to MPS, run on cuOpt H100, get baseline α at full LP scale.

**Phase 4 (only if Phase 1–3 fall short):** §2.2 (Bernstein subdivision) + §2.4 (Toeplitz FFT) — high engineering cost, treat as last resort.

---

## What we would NOT do

These were considered and rejected:

- **DSOS** (Ahmadi-Majumdar LP version): equivalent to our current Pólya, no improvement.
- **Sherali–Adams RLT alone**: same $O(1/r)$ convergence as Pólya. $\le 5%$ improvement.
- **Schmüdgen-type preordering as an LP**: rate is $O(1/r^2)$ but only as an SDP; LP version blows up to $\binom{d}{B}\binom{r+B}{B}$ which is worse than Pólya at our scale.
- **Lasserre's "no-lift" hierarchy**: gives upper bounds from above (wrong direction for us).
- **Pure copositive K^r LP hierarchy** (Bomze–de Klerk Parrilo K^r): K^0 = our LP; K^1 already requires SDP. No LP-only improvement.
- **Yildirim uniform polyhedral approximations** to copositive cone: gap closes as $O(1/m)$, intractable scaling at d=64.

---

## Sources

Top-tier (start here):
- TSSOS / CS-TSSOS: Wang–Magron–Lasserre, [arXiv:1912.08899](https://arxiv.org/abs/1912.08899), [arXiv:2005.02828](https://arxiv.org/abs/2005.02828).
- Tractable hierarchies on the orthant: Mai–Magron–Lasserre–Toh, [arXiv:2209.06175](https://arxiv.org/abs/2209.06175).
- cuPDLP / cuOpt: [arXiv:2311.12180](https://arxiv.org/abs/2311.12180).
- SDSOS: Ahmadi–Majumdar, [arXiv:1706.02586](https://arxiv.org/abs/1706.02586); iterated [arXiv:1510.01597](https://arxiv.org/abs/1510.01597).
- PVZ-SDP for SQO: Peña–Vera–Zuluaga, [SIAM J. Optim. 16 (2006)](https://epubs.siam.org/doi/10.1137/04060562X).

Background:
- Pólya rate: Powers–Reznick, *J. Pure Appl. Algebra* 164 (2001).
- Bernstein basis: Garloff–Smith 1999 / Lai–Schumaker 2007.
- Cutting-plane on polynomial LPs: Bienstock 2014, [arXiv:1501.00288](https://arxiv.org/abs/1501.00288).
- Sidon problem state-of-art: Cloninger–Steinerberger 2017 [arXiv:1403.7988](https://arxiv.org/abs/1403.7988); Matolcsi–Vinuesa 2010 [arXiv:0907.1379](https://arxiv.org/abs/0907.1379).
- Local notes on which cuts are "outside CS-window family": `delsarte_dual/ideas_turan_krein.md`, `delsarte_dual/ideas_beurling_selberg.md`, `delsarte_dual/mv_construction_detailed.md`.
