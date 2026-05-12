# High-Impact Audit: Tightening & New Directions for $C_{1a} > 1.2802$

> Synthesized from 12 parallel research agents covering 6 tightening tracks and 6 new mathematical directions. Each finding is grounded in cited file:line references.
>
> **Standing bound**: $1.2802 \le C_{1a} \le 1.5029$. Goal: push the lower bound above 1.2802.
>
> **Headline finding**: a folklore convexity-symmetry argument (codified in `lasserre/z2_symmetry.py:22-35`) appears to give $C_{1a} = C_{1a}^{\mathrm{sym}}$ unconditionally — combined with Path A's symmetric bound $C_{1a}^{\mathrm{sym}} \ge 1.42429$, this would yield **$C_{1a} \ge 1.42$ unconditionally** with NO new analysis. But `path_a_unconditional_holder/derivation.md §5.3(b)` flags exactly this reduction as **OPEN**. The two assertions appear to contradict — **resolving this is the single highest-EV item in the entire project.**

---

## §0 Ranked Top 5 Actions (by impact × tractability × effort)

| # | Action | Effort | Impact if works | Probability | Owner |
|---|---|---|---|---|---|
| **1** | **Audit the symmetry contradiction** between `lasserre/z2_symmetry.py:22-35` and `path_a_unconditional_holder/derivation.md §5.3(b)`. Decide whether $C_{1a} = C_{1a}^{\mathrm{sym}}$ is provable. | 1-2 days | $C_{1a} \ge 1.42$ unconditionally (+0.14 over CS17) | ~85% (per Agent 6) | math |
| **2** | **Vertex-enumeration box-cert** at the final cascade level (≤16 dims) — replace water-filling/triangle with exact `qp_bound_vertex` per `tests/box_cert_tightness.py:207-238`. | 1 day | Coarse cascade prover: 1.15 → likely 1.25-1.30 | ~80% | code |
| **3** | **Run sparse-order-3 Lasserre trajectory** at $d \in \{12, 16, 20\}$ via `lasserre/trajectory/run_trajectory.py` (currently `DEFAULT_ORDER=2`, line 58). Gate-check the d=64 plan **before** any pod commit. | hours | Validates/falsifies entire Lasserre track | n/a (it's a measurement) | code |
| **4** | **ThetaEvolve fork on the dual** — evolve PD Cohn-Elkies kernels $g$ with `inner_moment_lp.py` rigorous scorer; seed with unimplemented Siegel-triangle×poly (F5) family. | 2-3 weeks, $5-10k cloud | Rigorous 1.2802 → ~1.29 | ~40-50% | infra |
| **5** | **Atomic-ν copositive single-SDP** (calculus-of-variations route) — `C_{1a} \ge \inf_f \sum w_i (f*f)(t_i)` is a SINGLE Lasserre SDP per $\nu$, bypassing the diagonal-pseudo-measure pathology that killed three-point V2/V3. | 1-2 weeks | Rigorous 1.28 → 1.30 | medium | math + code |

---

## §1 TIGHTENING THE EXISTING PROVER (`coarse_cascade_prover.py`)

### 1.1 Box-cert joint-QP upgrade — partial win

Current `_box_certify_cell` at `coarse_cascade_prover.py:631-702` uses **first-order water-filling**, picking ONE feasible point per window. The exact min-over-cell of $TV_W$ is computable via vertex enumeration in `cloninger-steinerberger/cpu/qp_bound_joint.py:40-214`.

**Gain estimate** (Agent 1):
- d=6, S=75, c=1.15 (current margin **+0.0048**) → joint-QP plausibly gives **+0.012-0.020** ⇒ unlocks c=1.16/1.17
- d=6, S=200, c=1.30 (current min-net **−0.0267**) → joint-QP shaves 30-60% of the linear term ⇒ stays at **−0.010 to −0.018** (still negative)

**Blocker**: vertex enum is $O(d \cdot 2^{d-1})$ — fine at d≤16, infeasible past. Coarse cascade's final level can be d=64+.

**Concrete diff plan** (lines from Agent 1):
1. `coarse_cascade_prover.py:631-702`: replace water-filling block (lines 656-694) with `joint_bound_vertex` call.
2. `coarse_cascade_prover.py:705-751`: stop sampling cells; iterate over actual L0 survivors (current sampling is unsound for proof purposes).
3. Add McCormick LP fallback for d>16 (move from `tests/test_joint_qp_box_cert.py:324-482` into `cpu/qp_bound.py`).

**Verdict**: pushes rigorous bound from **1.15 → ~1.17-1.18**. Cannot reach 1.20+. Can't fix the structural margin=0 issue.

### 1.2 Vertex enumeration — **the surprise win**

(Agent 2's deeper read.) For the **29 surviving cells** at d=12, S=20, c=1.30 (`cpu_run_20260413_141632.log:48`), full vertex enum on the cell `{|δᵢ|≤h, 1ᵀδ=0}` has $d \cdot 2^{d-1} = 12 \cdot 2048 = 24{,}576$ vertices/cell × 300 windows × $d^2$ pair-ops ≈ $3 \times 10^{10}$ ops total — **seconds, not hours**.

This gives the **EXACT** min-over-cell of $\max_W TV_W$ (justified by the vertex theorem of quadratic-on-polytope). The current ~−0.57 reported gap is from the *uncorrected* L0 prune (`cpu/run_cascade_coarse_v2.py:_prune_no_correction`, line 500-522), NOT from the box-cert step. Box-cert only needs to certify those 29 surviving cells.

**Verdict**: vertex enum on survivors is the **single most impactful tightening change**. Plausibly certifies **c=1.25 directly at d=12, S=20**, possibly c=1.30 (if the 29 survivors are near-extremal arcsine-like with TV ≫ 1.30). 1 day implementation.

### 1.3 Theorem 1 fractional overlap — **DEAD on inspection**

(Agent 3.) The proposed sharp inequality
$$\max(f*f) \;\ge\; \tfrac{2d}{\ell}\sum_{i,j}\alpha_{i,j,W}\, \mu_i\mu_j$$
with $\alpha_{i,j,W}=|B_i+B_j\cap W|/|B_i+B_j|$ tent-shaped is **NOT a valid lower bound from bin masses alone**.

Counterexample: place $\mu_i$ as a point mass at the FAR edge of $B_i$ and $\mu_j$ at the FAR edge of $B_j$; their convolution lands at the outer endpoint of $B_i+B_j$, **outside** $W$. Worst-case $\alpha$ is 0, not 1/2. Theorem 1's "drop the wings" step is loose by design but tight against this adversary.

**Verdict**: cannot push c via this single change. Removed from candidates.

### 1.4 B&B per-node LP relaxation — high enumeration win

(Agent 9.) The "subtree pruning" at `coarse_cascade_prover.py:417-456` is a **HEURISTIC fail-fast on the partial state**, not a true subtree bound. The mature `cpu/run_cascade.py:1991-2110, 2640-2740` already implements `min_contrib` — a rigorous lower bound on `conv` over the entire subtree using cursor-bound minimum contributions.

**Concrete diff** (~100 LOC, low-risk, soundness-preserving):
1. Port `min_contrib` from `cpu/run_cascade.py:2024-2110` verbatim into `coarse_cascade_prover.py:417`.
2. Add McCormick `hi_i · hi_j` upper bound at internal nodes (cell-size-aware).
3. Wire `qp_bound_vertex` at leaf cells (d≤16).

**Win**: 5-50× pruning at internal levels, geometric to **600× at L4**. Converts $10^9$ L4 sweep to ~$1.6 \times 10^6$ — single-CPU tractable.

**Caveat**: this fixes the *enumeration* cost. The *box-cert margin* is an orthogonal blocker (the rigor-parity barrier per `memory/project_rigor_parity_barrier.md`).

### 1.5 Cell-variation Hessian-tight bound — incremental

(Agent 2.) The window Hessian $H_W$ is the antidiagonal-band adjacency matrix of pairs $\{(i,j): s\le i+j\le s+\ell-2\}$. For narrow $\ell$, $H_W$ is **strongly indefinite** with $|\lambda_-|=4d/\ell$. The current `quad_corr` already approaches the spectral bound at d=12 (within ~10%), so the gain from Hessian-tight cell-var is **~0.02-0.05 in min-net** — useful but not transformative.

**Verdict**: subsumed by §1.2 vertex enum (which is exact, not just spectral-tight).

### 1.6 Adaptive cell decomposition — **SUBSUMED**

(Agent 12.) The margin=0 at d=4, S=20, c=1.30 is **arithmetic quantization**: $c \cdot \ell \cdot S^2/(2d) \in \mathbb{Z}$ exactly when $1.30 \cdot 8 \cdot 400/8 = 520$. Trivially cured by an **off-lattice S shift** (S=21 or S=19), far cheaper than k-d-tree adaptive cells.

**Verdict**: replace adaptive subdivision plan with "rerun the worst configs at $S \pm 1$".

---

## §2 NEW MATHEMATICAL DIRECTIONS

### 2.1 Symmetry: $C_{1a} = C_{1a}^{\mathrm{sym}}$ — **THE HEADLINE**

(Agent 6.) `lasserre/z2_symmetry.py:22-35` asserts a "Symmetric extremizer theorem (folklore; cites MV2010, CS2017, White2022)": every minimizer of $\|f*f\|_\infty$ on the convex domain $F$ has a symmetric average $(f+\sigma f)/2 \in F$ with $\|((f+\sigma f)/2)*((f+\sigma f)/2)\|_\infty \le \|f*f\|_\infty$ via convexity + Jensen.

Combined with Path A §6 (rigorous, unconditional, on the symmetric subclass): $C_{1a}^{\mathrm{sym}} \ge 1.42429$.

**Chain**:
1. z2_symmetry argument ⇒ $C_{1a} = C_{1a}^{\mathrm{sym}}$ (folklore but apparently not fully exploited).
2. Path A §6 + restricted-Hölder ⇒ $C_{1a}^{\mathrm{sym}} \ge 1.42401$.
3. Combined: $C_{1a} \ge 1.42401$ unconditionally.

**Apparent contradiction**: `path_a_unconditional_holder/derivation.md` §5.3(b) explicitly lists the "symmetric reduction in disguise" — proving $C_{1a}^{\mathrm{asym}} \ge C_{1a}^{\mathrm{sym}}$ — as **OPEN**. Either:
- (i) z2_symmetry.py is correct → +0.14 free (and the §5.3(b) entry is wrong), or
- (ii) §5.3(b) has identified a flaw the folklore claim misses → no free win.

Estimated probability $C_{1a} = C_{1a}^{\mathrm{sym}}$: **~85%** (Agent 6's calibrated estimate).

**Action item #1**: 1-2 days of careful re-audit. Verify the Jensen step on the autoconv functional, including the boundary issue (does $(f+\sigma f)/2$ stay supported in $[-1/4, 1/4]$? **Yes** — $\sigma$ is measure-preserving on this support). Verify $\int(f+\sigma f)/2 = 1$. Then either upgrade $C_{1a}$ to 1.42401 or document why the folklore claim fails.

### 2.2 Lasserre SDP — track is **stalling on the gate**

(Agent 4.) Current rigorous (Farkas-certified) bounds:

| d | k | b | $\mathrm{lb_{rig}}$ | source |
|---|---|---|---|---|
| 4 | 3 | 3 | 1.0978515625 | `lasserre/certs/d4_cert.json:8` |
| 6 | 3 | 5 | 1.05 | `lasserre/certs/d6_cert.json` |
| **8** | **2** | **7** | **1.141796875** | `lasserre/certs/d8_cert.json:8` |
| 16 | 2 | 7 | 1.05 (FLOOR) | `lasserre/certs/d16_cert.json` |
| 32, 64, 128 | — | — | NOT RUN | pod-gated |

**Best certified: 1.1418 at d=8.** Trajectory gate not met (`lasserre/trajectory/REPORT.md` does not exist; only `d4`, `d8`, `order3_d4` JSON datapoints).

**Worse**: `PATH_A_FINAL_SYNTHESIS.md:98-99` flags `core.py::val_d_known[16] = 1.319` as **stale** — multi-seed PGD finds val(16) ≈ **1.278** (under 1.2802). If val(16) is really under 1.2802, the entire d-extrapolation thesis weakens.

**Author's calibrated probabilities** (`PATH_A_FINAL_SYNTHESIS.md:122`, `TRACK1_FINF_DEEP_REPORT.md:234`):
- Path A L^{3/2}: **3%**
- 3-point Lasserre + tradeoffs: **5%**
- "Any approach": 40-60% (mostly val(d) at higher d)

**Action item #3** (highest-impact non-symmetry move):
1. Run sparse-order-3 trajectory at d ∈ {12, 16, 20} locally. Edit `lasserre/trajectory/run_trajectory.py:58` (`DEFAULT_ORDER=2 → 3`).
2. Independently re-validate `lasserre/core.py::val_d_known[16]`. If <1.2802, kill the d-extrapolation thesis.
3. Implement MOSEK Task-API warm-start (per `lasserre/trajectory/OPTIMIZATIONS.md:103-106`) only AFTER the trajectory shows promise.

**Cost**: hours (not pod-hours). **Value**: settles whether to commit pod time or pivot.

### 2.3 MV alt-kernel sweep — **dead modulo one experiment**

(Agent 7.) MV's analytic ~1.276 ceiling (`mv_construction_detailed.md:463-484`) bounds every kernel-only attack within the framework. PSWFs are admissible but worse (`data/pswf_sweep_log.txt:121-133`: PSWF best 1.21612 vs arcsine 1.26957). 14-kernel search + 31-kernel sweep + Chebyshev DE all confirm arcsine is a strict local max.

**The one untried experiment**: joint $(K, G)$ alternating-projection SDP — kernel as SDP variable with Bochner constraints, jointly with cosine coefficients. Decision rule:
- $M_{\mathrm{cert}} < 1.2760$ ⇒ avenue **DEAD** (analytic ceiling tight).
- $\ge 1.2760$ but $< 1.2802$ ⇒ document and stop.
- $\ge 1.2802$ ⇒ MV's ceiling has a loophole; new SOTA.

**Effort**: 1-2 days at $\delta \in \{0.13, 0.138, 0.15, 0.16\}$ via existing `delsarte_dual/grid_bound_alt_kernel/optimize_G.py`.

### 2.4 Three-point SDP — **PIVOT to L^∞-augmented + analytical bridge**

(Agent 8.) Numerical results:

| Formulation | Value | Status |
|---|---|---|
| V1 (bump-kernel) | sub-1.27 | kernel-floor dominated |
| V2 (CD + Putinar, no L^∞) | $\lambda = 1.0$ exactly (Δ=0) | structurally trivial |
| **V3 + L^∞ at M=2.10** | **$\lambda_{3pt} = 1.3573$** | **above 1.2802** (primal only) |
| V3 + L^∞ at M=2.15 | $\lambda_{3pt} = 1.3122$ | above 1.2802 |
| V3 + L^∞ at M=2.186 | $\approx 1.2802$ | crossover |
| L^{3/2} moment | identical to V3 baseline | encoding **vacuous** |

**The breakthrough**: `lasserre/threepoint_alternatives.py` formulation B (V2 + L^∞ density bounds) gives 10-33% lift at $M \in [2.10, 2.15]$.

**The bottleneck has moved**: the 3pt SDP gives $C_{1a}^{(M)} \ge 1.31$ rigorously — the missing piece is the **analytical bridge** from $C_{1a}^{(M)}$ to $C_{1a}$, i.e., proving $\|f^*\|_\infty \le 2.18$ for the unconstrained Sidon optimum. **Not an SDP question** — invest in analysis.

**Verdict**: stop level-3/k=10 compute (values plateau). Pure 3pt-cone-as-tightening: dead. L^∞-augmented + analytical bridge: alive, **action item**: prove $\|f^*\|_\infty \le 2.18$.

### 2.5 Cohn-Elkies dual via LLM evolution — **best untapped Fourier angle**

(Agent 10.) Modern literature surveyed (arXiv:2003.06962 Madrid-Ramos, 2106.13873 de Dios Pont-Madrid, 2210.16437 White, 2506.16750 Boyer-Li, 2602.07292 Rechnitzer, plus Carneiro-Vaaler extremal-majorant family). The Cohn-Elkies LP duality for $C_{1a}$ has been formulated (`sdp_hierarchy_design.md` §1B) but optimized only over hand-picked F1-F4 ansatzes (Selberg, Gauss-poly, Vaaler, MV-arcsine).

**Untapped**:
- (a) Siegel/Turán-extremizer family $g(t) = (1-2|t|)_+ \cdot P(t^2)$ with PSD $P$ — flagged "F5" in `ideas_fourier_ineqs.md` but **never implemented**.
- (b) PSWF/prolate-weighted dual via Logvinenko-Sereda eigenvalue.
- (c) **LLM-evolution of the dual side** — every evolution campaign (AlphaEvolve, ThetaEvolve, TTT-Discover) only optimized the PRIMAL upper bound. The lower-bound dual has identically cheap deterministic scoring (`inner_moment_lp.py` + Farkas) and is untouched.

**Action item #4**: Fork ThetaEvolve, scorer = rigorous inner LP. Seed with F5 + PSWF-weighted variant. Memo's calibrated lift: $10^{-3}$ to $10^{-2}$, taking 1.2802 → [1.281, 1.29]. Compute: $5-10k.

### 2.6 Reverse-Young / Hyp_R: the asymmetric obstruction is a **proof failure, NOT a counterexample**

(Agent 5.) The two project-relevant conjectures:

**(a) REV-3/2** (`lasserre/REVERSE_YOUNG_PROOF_ATTEMPT.md`):  for nonneg $f$ on $[-1/4, 1/4]$ with $\int f = 1$,
$$\sup_{|t|\le 1/2}(f*f)(t) \;\ge\; \tfrac{\pi}{8}\,\|f\|_{3/2}^{\,3}$$
Equality at SS extremizer $f_0(x) = (2x+1/2)^{-1/2}$.

**(b) Hyp_R(c, M_max)**: $\|f*f\|_2^2 \le c\, \|f*f\|_\infty$ for $\|f*f\|_\infty \le M_{\max}$.

**Critical reframing**: per Agent 5, the asymmetric obstruction in `path_a_unconditional_holder/derivation.md §5` is **a failure of proof technique, NOT a refutation**:
- Attack 1 yields $\|g\|_2^2 \le 1 + \mu(M)(\|f\|_2^2 - 1)$, valid for all $f$.
- Closure needs $\|f\|_2^2 \le M$, holds symmetric (since $g(0) = \|f\|_2^2 \le \|g\|_\infty$), fails asymmetric (where $\|f\|_2^2 = h(0)$ can exceed $M$).
- A 3-spike Sidon construction has $K/M = 3/2$ at $M=1.378$ — but is **infeasible at $M \le 1.378$**, so the quantitative $\alpha^* := \sup \|f\|_2^2/M$ is **unknown**.
- MV's asymmetric counterexample (eq. 4.15) has ratio 0.4744 $>$ $\pi/8$, so it does **NOT** disprove REV-3/2.
- Boyer-Li 2025 ($\|g\|_\infty \approx 1.652$) is outside the restricted class $M \le 1.378$ and does not refute Hyp_R restricted.
- **No 2023-2026 paper** found in arXiv/Semantic Scholar that proves, disproves, or directly bears on either conjecture for $p=3/2$. The exponent regime is empty in classical harmonic analysis.

**Quantitative weakening — important**: per `restricted_holder/conditional_bound.py`, beating 1.2802 needs only $c < 1$ strictly. At $c=0.99$, $M_{\mathrm{target}}(c) \approx 1.281$. **A very lossy REV-3/2 (constant $\pi/8 \to \pi/8 - 0.05 \approx 0.34$ instead of 0.39) still suffices.** The bottleneck is existence of *any* unconditional proof for asymmetric $f$, not tightness.

**Action item #6 — the cleanest 2-week math problem in the project**: prove the Hardy-type reformulation from `REVERSE_YOUNG_PROOF_ATTEMPT.md §7.2`:
$$\int_0^1 \phi(1-w)\,\phi'(w)\,dw \;\ge\; \tfrac{\pi}{8}\,\Big(\int_0^1 \phi'^{3/2}\,dw\Big)^2$$
for increasing $\phi: [0,1] \to [0,1]$. Attack via Abel / fractional integration (the constant traces to $B(1/2, 1/2) = \pi$). Explicit equality case $\phi(w) = \sqrt{w}$. **3592 numerical confirmations already in `results/path_a_stress_test.json`** (zero violations). Proving it (even with a constant slightly worse than $\pi/8$) immediately resurrects asymmetric Path A and beats 1.2802.

**Strategic implication**: the "asymmetric obstruction is open, not falsified" finding directly **supports** §2.1 (the symmetry argument). If the asymmetric obstruction were a real counterexample, the convexity-Jensen route would have to fail somewhere. Since it's just a proof gap, the convexity argument may well be the right path — they attack the same gap by different mechanisms (one symmetrizes via averaging, the other circumvents via reverse-Young).

### 2.7 Restricted-Hölder M_max optimization — three concrete proposals

(Agent 11.) The conditional theorem (`restricted_holder/derivation.md:154-160`):

$$\mathrm{Hyp}_R(c, M_{\max}) \;\Rightarrow\; C_{1a} \ge M_{\mathrm{optimal}}(c)$$

| $c$ | $M_{\mathrm{optimal}}(c)$ |
|---|---|
| 1.000 (unconditional Hölder) | 1.27484 |
| 0.990 | 1.28252 |
| 0.950 | 1.31613 |
| 0.920 | 1.34292 |
| **0.88254 ($\log 16/\pi$)** | **1.37842** (HEADLINE) |

Boyer-Li 2025 disproves the unrestricted form at $M_{\max} \ge 1.652$; the BL witness is a 575-step staircase with extreme asymmetry & high-frequency mass. Restrictions excluding BL but preserving CS-class:

1. **(M_max=1.40, c=log16/π, BV ≤ 30)** — exclude staircases; gives 1.378 conditional.
2. **(M_max=1.50, c=0.95, no extra)** — weaker conjecture far from BL's $c_{\mathrm{emp}} \approx 0.90$; gives 1.316.
3. **(M_max=1.45, c=log16/π, symmetric only)** — already subsumed by §2.1 if the symmetry argument holds.

**Action item**: investigate (1) and (2). Both are tractable analytically (BV-restricted reverse-Young is closer to standard harmonic analysis than full reverse Young).

### 2.8 Atomic-ν copositive SDP — **the 3pt-failure-bypass**

(Agent extra.) From `delsarte_dual/ideas_calculus_variations.md`: for any finite-atomic measure $\nu = \sum w_i \delta_{t_i}$ on $[-1/2, 1/2]$,

$$C_{1a} \;\ge\; \inf_{f \ge 0, \int f = 1, \mathrm{supp} \subset [-1/4, 1/4]} \sum_i w_i (f*f)(t_i).$$

The inner inf is a **single copositive Lasserre SDP** — no min-max, no Christoffel-Darboux truncation error, no dualization gap. This **bypasses** the diagonal-pseudo-measure failure that produced $\lambda=1$ in `THREEPOINT_FULL_REPORT.md:124-129`.

**Action item #5**: Write `lasserre/atomic_nu_sdp.py`. Outer bilevel optimizes $(w_i, t_i)$ over a small grid (3-5 atoms suffice from KKT); inner Lasserre uses existing infrastructure + Farkas certify. Plausible 1.28 → 1.30 at d=14-16. Effort: 1-2 weeks.

### 2.8 White 2022 Hölder-$\hat f^3$ dual

(Agent extra.) White's Lemma 8 (`ideas_white2022.md` §4.2) constructs $g$ with $\hat g \propto \hat f^3$ and Hölder $(4, 4/3)$ to bound $\mu_2^2$. **Sign-aware analogue** for $f \ge 0$: choose $\hat \nu$ matched to $\hat f^3$ via Hölder. Independent of MV's saturated 1.276 ceiling. Combine with Rechnitzer's Arb 384-digit ball arithmetic. Plausible 1.28 → 1.29. Effort: 2 weeks.

---

## §3 What is DEAD (don't revisit)

| Direction | Why dead | Citation |
|---|---|---|
| Theorem 1 fractional overlap | Tent-α is not a valid lower bound (point-mass counterexample) | Agent 3 |
| Adaptive cell decomposition | Margin=0 is arithmetic, cured by S±1 shift | Agent 12 |
| Pure 3-point SDP (no L^∞) | $\lambda=1$ structurally (diagonal pseudo-measure) | `THREEPOINT_FULL_REPORT.md:124-129` |
| L^{3/2} moment encoding | Vacuous; identical to V3 baseline at all B | `results/l32_moment_sweep.json` |
| PSWF/prolate kernel sweep | Empirically 5.3% below arcsine, monotone wrong direction | `data/pswf_sweep_log.txt:121-133` |
| MV-style 14-kernel scan | Arcsine is strict local max + analytic 1.276 ceiling | `grid_bound_alt_kernel/REPORT.md:64-100` |
| Riesz/BLL/Lieb-Young rearrangement | Wrong direction | `ideas_rearrangement.md` |
| Random $f$ probabilistic | Wrong direction | `ideas_probabilistic.md` |
| Discrete $B_2[g] \to$ continuous | Transfer is only continuous→discrete | `ideas_b2_sets.md` |
| Pure cascade prover beyond 1.20 | Structural: margin=0 at grid lattice, water-filling can't manufacture margin | `COARSE_CASCADE_PROVER_RESULTS.md:282-318` |

---

## §4 Recommended sequencing (next 6 weeks)

### Week 1
- **[1d]** Vertex-enum box-cert in coarse cascade prover (§1.2). Rerun d=12, S=20-21, c∈{1.20, 1.25, 1.30}.
- **[1d]** Audit symmetry contradiction (§2.1). Decide whether $C_{1a} = C_{1a}^{\mathrm{sym}}$.
- **[1d]** Run sparse-order-3 trajectory at d∈{12,16,20} (§2.2). Re-validate `val_d_known[16]`.
- **[1d]** Joint $(K,G)$ kernel SDP one-shot (§2.3) — DEAD-confirm or breakthrough.

### Week 2-3
- **[5d]** Port `min_contrib` from `cpu/run_cascade.py` to coarse cascade prover (§1.4). Unlocks d_0=8, S=50, L4 cascade.
- **[5d]** Atomic-ν copositive SDP prototype (§2.7).

### Week 3-4
- **[10d]** White-Hölder-$\hat f^3$ dual derivation + Arb verification (§2.8).
- **[parallel]** ThetaEvolve fork on dual kernels (§2.5) — async cloud.

### Week 5-6
- Restricted-Hölder BV-bounded variant (§2.6, item 1).
- L^∞-bridge analytical proof for 3pt SDP (§2.4): prove $\|f^*\|_\infty \le 2.18$.

---

## §5 Bottom line

The project has multiple tracks running in parallel; the audit identifies **three single-action wins** that could each produce a real headline:

1. **Symmetry resolution** ($C_{1a} = C_{1a}^{\mathrm{sym}}$ ⇒ 1.42 free) — 1-2 days, ~85% probability.
2. **Vertex enum + min_contrib in coarse cascade** — pushes 1.15 to 1.25-1.30, both at low effort.
3. **Lasserre trajectory gate measurement** — settles the central pod-commit question in hours.

The *biggest live opportunity* is **(1)**: a pure mathematical re-audit, no code, that — if the folklore convexity argument holds (as `lasserre/z2_symmetry.py` claims) — gives **the largest single jump in the lower bound since CS 2017** with no new analysis. This must be the next item investigated.

Everything else is real but smaller. **Recommendation: start with §0's action item #1 today.**
