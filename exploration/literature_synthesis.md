# Synthesis: Paths to Improving the Upper Bound on $C_{1a}$

**Current state:** $C_{1a} \in [1.2802,\; 1.5029]$. Gap $\approx 0.223$.

**Sources synthesized:**
- **[MV10]** Matolcsi & Vinuesa (2010): $C_{1a} \leq 1.5098$, step functions with $n=208$, LP iteration.
- **[AE25]** AlphaEvolve (Tao et al., 2025): $C_{1a} \leq 1.5032$, LLM-evolved cubic backtracking, $n=50$.
- **[TTT26]** TTT-Discover (Yuksekgonul et al., Jan 2026): $C_{1a} \leq 1.50286$, RL-at-test-time, 30,000 pieces.
- **[KB]** Computational notebook: $\sim$8 optimization methods (Polyak, L-BFGS, NTD, prox-linear, LP, SDP, TV surrogate, Gaussian mixture) on the discretized problem up to $P=1000$.

---

## 1. BOTTLENECK TAXONOMY

### 1.1 Shared Structural Bottlenecks

These appear across multiple independent methods and reflect fundamental mathematical barriers.

| Bottleneck | Description | Papers Hitting It | Circumvented? |
|---|---|---|---|
| **S1. Non-convexity of $\|f*f\|_\infty$** | The map $f \mapsto \max_t(f*f)(t)$ is the supremum of a family of quadratic forms $h \mapsto h^T K_m h$, hence **non-convex**. All local methods converge to Clarke stationary points, not global minima. | [MV10] §3a–b, [AE25] §3.1, [TTT26] §3a, [KB] §3.2 | **No.** [AE25] and [TTT26] use global search heuristics (evolutionary, RL) that empirically escape some local minima but carry no guarantee. The non-convexity is intrinsic to the problem formulation. |
| **S2. No structural characterization of the extremizer** | No necessary conditions on the extremal $f$ are known beyond $f \geq 0$, $\int f = 1$. No conjectured closed form, symmetry, monotonicity, or regularity. | [MV10] §3d: "we do not have any natural conjectures"; [AE25] §3.4; [TTT26] §4 | **No.** [MV10]'s analytic counterexample (eq. (12)) and the sparse irregular patterns of numerical solutions provide hints but no rigorous structural constraints. |
| **S3. No dual certificate / global optimality gap** | All upper bound methods are purely primal: they exhibit an $f$ and compute the ratio. There is no dual witness bounding the gap between the exhibited $f$ and the true infimum. | [AE25] §3.4, [TTT26] §5, [KB] §3.4 | **Partially.** [KB] Method G formulates an SDP relaxation that in principle provides a dual bound on the *discrete* problem, but it is intractable for $P > 50$ and says nothing about the continuous problem. |
| **S4. Peak-locking** | Gradient-based methods reinforce whichever $t^*$ currently achieves $\max_t(f*f)(t)$, trapping the optimizer in a basin indexed by the active constraint. | [AE25] §3.2 (identified by Jaech & Anderson 2025), [TTT26] §3a, [KB] §3.2 (implicit) | **Partially.** [TTT26] Phase 3 distributes gradients across all near-maximum positions, mitigating but not eliminating the effect. The underlying issue is non-smoothness of $\max$. |

### 1.2 Method-Specific Bottlenecks

These are unique to particular approaches and suggest the method can be swapped or extended.

| Bottleneck | Description | Paper | Circumvented? |
|---|---|---|---|
| **M1. LP iteration → local fixed point** | The Kolountzakis–Matolcsi LP linearizes around the current iterate, so it cannot "see" distant basins. Converges to a fixed point where $g_0 = f$ is already LP-optimal. | [MV10] §3a, [KB] §3.2 | **Yes**, by [AE25] and [TTT26], which use global search strategies to escape LP fixed points. |
| **M2. Softmax boundary inaccessibility** | $h = 2P\,\text{softmax}(w)$ cannot reach the boundary of the simplex (requires $w_i \to -\infty$), and creates flat directions ($w \mapsto w + c\mathbf{1}$) that degrade quasi-Newton conditioning. | [KB] §3.5 | **Not directly.** But methods that project onto the simplex ([KB] Polyak, prox-linear) avoid this entirely. |
| **M3. Prox-linear $\gamma$-tuning and cycling** | Prox-linear requires $\gamma \geq \max_i \text{Lip}(\nabla \ell_i)$ for guaranteed descent, but this makes steps too conservative. At $\gamma = 0$ the method cycles. | [KB] §3.3 | **Partially.** Adaptive $\gamma$ schedules could help, but the fundamental issue is that the subproblem's convex hull approximation is too loose. |
| **M4. Gaussian mixture limited DoF** | $K$ Gaussians give $3K$ parameters vs. $P$ step-function heights. Smooth parameterization misses the irregular, sparse structure of good solutions. | [KB] §3.6 | **Not attempted** with richer parameterizations (e.g., wavelets, splines with free knots). |
| **M5. TV surrogate looseness** | Replacing $\max(f*f)$ with $g(-1/2) + \text{TV}(g)$ optimizes a surrogate, not the true objective. Produces qualitatively similar structures but worse objective values. | [KB] §3 (Method H) | **No.** The TV bound is inherently loose; it bounds the max by the total variation, which is a strict upper bound. |
| **M6. LLM code quality / context constraints** | [AE25] and [TTT26] depend on LLM-generated Python code. Quality is bounded by model capability, context window (32,768 tokens), and finite sampling. | [AE25] §3, [TTT26] §3d | **Partially** by scaling (more rollouts, bigger models), but fundamentally limited by the code-generation paradigm. |

### 1.3 Computational Bottlenecks

Limited by compute, resolution, or search budget — not by mathematical barriers.

| Bottleneck | Description | Paper | Circumvented? |
|---|---|---|---|
| **C1. Discretization at finite $n$** | Step functions on $n$ equal intervals approximate $L^1$ functions, but convergence rate is unknown. Best values: $n=208$ [MV10], $n=50$ [AE25], $n=30{,}000$ [TTT26]. | All papers | **Partially.** [TTT26] pushed to $n=30{,}000$ but gained only $\sim 0.0003$ over $n=1{,}319$ [AE25 V2]. Diminishing returns suggest discretization is **not** the dominant slack source. |
| **C2. SDP intractability at large $P$** | The SDP relaxation [KB] §3.4 has $O(P^2)$ matrix variables, making interior-point solvers $O(P^6)$ per iteration. Only feasible at $P=50$. | [KB] §3.4 | **Not yet.** First-order SDP methods (e.g., mixing method, ADMM) could push to $P \sim 200$–$500$ but are less accurate. |
| **C3. Finite search budget** | [TTT26] uses 25,600 samples (50 steps $\times$ 512 rollouts); the 30,000-dimensional search space is vastly under-explored. | [TTT26] §3d | **In principle** by more compute, but subject to diminishing returns. |
| **C4. LP/QP per-iteration cost** | Each LP step costs $O(n^2)$ (cross-convolution has $O(n)$ terms per component). Prox-linear QP at $P=600$ takes seconds to minutes. | [MV10] §3c, [KB] §3.3 | **Partially** by active-constraint selection ([TTT26] Phase 3 uses top-$K$ constraints), reducing effective problem size. |

---

## 2. BARRIER ANALYSIS

### Are the approaches hitting the same wall?

**Yes — at the deepest level, all approaches share the same fundamental barrier: the non-convexity of $\|f*f\|_\infty$ over the cone of nonnegative functions (S1).** This is not a mere computational inconvenience but a structural feature of the problem: the objective is the pointwise supremum of concave quadratics $h^T K_m h$ restricted to a convex set (the simplex), making the problem a non-convex QCQP.

**Evidence of a common wall:**

1. **Convergence of independent methods to $\sim 1.50$.** Five fundamentally different optimization paradigms — LP iteration [MV10], Newton-type [MV10], LLM-evolved heuristics [AE25], RL-at-test-time [TTT26], and classical nonsmooth optimization (Polyak, L-BFGS, NTD, prox-linear) [KB] — all converge to the range $[1.500, 1.516]$. This is despite using different initializations, different discretizations ($n$ from 50 to 30,000), and different search strategies. The consistency strongly suggests they are all finding **near-global optima of the discretized problem**, not merely hitting method-specific ceilings.

2. **Qualitatively different local minima with nearly identical values.** [TTT26] Figure 3 shows that the TTT-Discover construction is visually distinct from the AlphaEvolve construction, yet their objectives differ by only $\sim 0.003$. [MV10] reports "several different patterns seem to arise" from different initializations. This is characteristic of a landscape with many near-degenerate local minima — the hallmark of a hard non-convex problem where the global minimum is close to the local ones.

3. **Diminishing returns across methods and resolution.** The progression $1.5098 \to 1.5053 \to 1.5032 \to 1.5029$ shows each improvement requiring dramatically more computational resources for smaller gains. This is consistent with approaching an asymptotic floor, not with being stuck in a bad local basin.

**Evidence against a common wall (alternative hypothesis):**

4. **The gap to the lower bound is enormous.** If all methods are truly near-optimal and $C_{1a} \approx 1.50$, the lower bound $1.2802$ is off by $\sim 17\%$. This is possible but would make the lower bound problem significantly harder than the upper bound problem — worth flagging because it's testable (see §5, Key Unknown K1).

5. **No method has explored non-uniform grids, free-knot splines, or singular ansätze** motivated by the known boundary singularity structure (cf. White 2023 on the $L^2$ problem). It is conceivable that the uniform-grid restriction artificially inflates the minimum for moderate $n$, and that a structured low-dimensional ansatz could break below 1.50.

**Assessment:** The weight of evidence favors the interpretation that $C_{1a} \approx 1.50$ and the upper bound is near-tight. The primary remaining uncertainty is on the **lower bound side**. However, item 5 represents a genuine untested degree of freedom.

---

## 3. SLACK MAP

The total gap is $1.5029 - 1.2802 = 0.2227$. I decompose the slack into contributions from the upper bound side and the lower bound side.

### 3.1 Upper Bound Slack (distance from 1.5029 to the true $C_{1a}$)

| Source | Estimated slack | Reasoning |
|---|---|---|
| **Local vs. global optimum** (S1) | $\leq 0.01$ | Five independent methods converge to $[1.500, 1.516]$. The global AI methods [AE25, TTT26] push below 1.503. The spread of $\sim 0.003$ between the best independent discoveries suggests the remaining local-vs-global gap is small. |
| **Discretization** (C1) | $\leq 0.005$ | Going from 1,319 pieces to 30,000 pieces gained $\sim 0.0003$. Extrapolating the trend $\Delta \sim O(1/n)$ from the data points $\{(50, 1.5053), (1319, 1.5032), (30000, 1.5029)\}$, the continuum limit is $\lesssim 1.500$. |
| **Non-uniform grids / singular ansätze** | **Unknown** (0 to 0.05?) | Never tested. If the extremizer has a boundary singularity $f(x) \sim |x \pm 1/4|^{-\alpha}$, uniform grids are systematically suboptimal. This is the largest **uncertain** contribution on the upper bound side. |
| **Total upper bound slack** | $\sim 0.003$–$0.05$ | |

### 3.2 Lower Bound Slack (distance from the true $C_{1a}$ to 1.2802)

| Source | Estimated slack | Reasoning |
|---|---|---|
| **Fourier kernel method ceiling** | $\sim 0.004$ above current | [MV10] §5 estimates the theoretical ceiling of the Matolcsi–Vinuesa kernel method at $\sim 1.276$; Cloninger–Steinerberger achieved $1.28$, essentially saturating it. |
| **Structural limitation of dual approach** | **Dominant** — likely $\sim 0.17$–$0.22$ | The lower bound methods use integral/Fourier inequalities that cannot exploit pointwise control of $f*f$. The Hausdorff–Young approach (de Dios Pont–Madrid) improves by only $\sim 0.0002$ over 2017. No new analytical technique has appeared. |
| **Total lower bound slack** | $\sim 0.17$–$0.22$ | |

### 3.3 The Single Largest Source of Slack

**The lower bound analytical barrier is overwhelmingly the largest source of slack.** The upper bound methods have converged to $\sim 1.50$ from multiple independent directions with diminishing returns, while the lower bound has been essentially stuck at $\sim 1.28$ since 2017. If $C_{1a} \approx 1.50$, approximately $\sim 0.22$ of the gap ($\sim 98\%$) comes from the lower bound side.

However, **for the specific goal of improving the upper bound**, the relevant question is: how much slack remains on the upper bound side? Here the largest source is **non-uniform discretization / singular ansätze** (unknown, potentially up to $\sim 0.05$), followed by **local-vs-global optimization** ($\leq 0.01$).

**Actionable conclusion:** The most promising attack point for the upper bound is not better optimization of step functions on uniform grids — that approach is likely within $\sim 0.01$ of its floor. Instead, the attack point is **structural**: either (a) designing ansätze informed by the regularity theory of the extremizer, or (b) developing a convex relaxation (SDP/moment problem) that provides a certified lower bound on the discrete problem, thereby bounding the local-vs-global gap.

---

## 4. UNEXPLOITED IDEAS

### 4.1 Ideas Suggested by Multiple Papers Independently (High Signal)

| Idea | Papers | Status |
|---|---|---|
| **SDP relaxation of the polynomial optimization** | [MV10] §3c ("semidefinite relaxations ... would provide certified upper bounds"), [AE25] §5 ("semidefinite programming formulations"), [KB] Method G, §5.4 | **Partially explored** by [KB] at $P=50$ only. Never scaled beyond $P=50$; never combined with branch-and-bound. The convergence of three independent sources on this idea is strong signal. |
| **Dual-informed primal search** | [MV10] §5 ("dual variables would characterize necessary conditions on any near-extremal $f$"), [AE25] §5 ("Use the structure of the lower-bound inequalities to constrain the shape of candidate extremizers"), [TTT26] §5 ("the LP approach ... is implicitly solving a discretized version of the dual problem") | **Never pursued.** All three papers note that connecting the primal (upper bound) and dual (lower bound) formulations could constrain the search space, but none attempts it. |
| **Fourier-domain reformulation** | [MV10] §5 (Fourier coefficients of extremal $f$), [KB] §5.4 ("Bochner's theorem" SDP in Fourier domain), [AE25] §5 ("Fourier analysis ... dual formulations") | **Never pursued for the upper bound.** Fourier methods have only been used on the lower bound side. A Fourier-domain SDP exploiting $(f*f)^\wedge = |\hat{f}|^2 \geq 0$ is a natural candidate for convexification. |
| **Adaptive / non-uniform discretization** | [TTT26] §3b ("non-uniform grids, neither of which is explored"), [KB] §5.4 ("place knots where the autoconvolution gradient is large") | **Never explored.** All published constructions use uniform grids. |
| **$L^p$ smooth approximation to $L^\infty$** | [TTT26] §5 ("$\|f*f\|_\infty \approx (\int (f*f)^p)^{1/p}$ for large $p$"), [KB] Method H (TV surrogate as a different smooth relaxation) | **Not explored for the $L^p$ version.** The TV surrogate was tried [KB] but optimizes the wrong objective. The $L^p$ approximation has controlled error and smoother landscape — a natural next step. |

### 4.2 Ideas Suggested but Never Pursued (Open Opportunities)

1. **Branch-and-bound with SDP bounds at moderate $P$.** [KB] §5.4 proposes this explicitly. At $P=100$–$200$, the SDP relaxation is feasible (with first-order solvers), and branch-and-bound can certifiably close the local-vs-global gap. This would either confirm $C_{1a} \approx 1.50$ at moderate resolution or discover that the discrete global optimum is significantly lower.

2. **Convexification via the $L^2$ analogy.** White (2023) showed the $L^2$ autoconvolution problem $\inf \|f*f\|_2^2$ admits a **convex QP** formulation in Fourier space, achieving bounds tight to $\sim 10^{-5}$. [AE25] §5 explicitly flags this: "Whether a similar convexification is possible for the $L^\infty$ problem remains open and would represent a major breakthrough." No one has investigated whether the $L^\infty$ problem has a tractable moment/SDP formulation.

3. **Exploiting equioscillation.** [KB] §5.3 observes that good solutions have "nearly equioscillatory autoconvolution peaks," reminiscent of Chebyshev equioscillation. If the extremizer satisfies an equioscillation condition — $(f*f)(t)$ achieves its maximum at multiple points with equal values — this is a strong structural constraint that dramatically reduces the search space. No paper formalizes or exploits this.

4. **Euler–Lagrange necessary conditions.** If the extremizer exists (proven by de Dios Pont–Madrid for general weights), it must satisfy first-order optimality conditions for the variational problem $\inf_{f \geq 0} \|f*f\|_\infty / \|f\|_1^2$. Deriving and analyzing these conditions could reveal regularity, symmetry, or monotonicity properties. [TTT26] §3b mentions this ("approximate stationarity conditions from the Euler–Lagrange equation") as a potential source of structured ansätze, but it has not been pursued.

5. **Hybrid global-local with polishing.** [KB] §5.4 proposes using AlphaEvolve/CMA-ES for global exploration, then polishing with prox-linear or LP. The notebook warm-starts Polyak from AlphaEvolve but never tries prox-linear or LP from that starting point. Given that [AE25]'s solution at $P=600$ beats [KB]'s prox-linear at $P=1000$, this warm-start strategy could push below 1.50.

### 4.3 Contradictions and Tensions

1. **Discretization: does it matter?** [MV10] found its best result at $n=208$; [AE25] used $n=50$ and got a better result. [TTT26] used $n=30{,}000$ for a marginal improvement of $\sim 0.0003$. Meanwhile, [KB]'s prox-linear at $P=1000$ achieves only 1.5160, far worse than [AE25] at $P=600$. **Resolution:** Discretization resolution and optimization quality are confounded. The evidence suggests that for $n \gtrsim 200$, further resolution gives diminishing returns **if optimization quality is held constant**, but that better optimization at lower $n$ dominates. The [TTT26] result at $n=30{,}000$ is consistent: most of the gain over [AE25] came from better optimization (RL vs. evolutionary), not from resolution (which contributed $\lesssim 0.001$).

2. **Is $C_{1a} \approx 1.50$ or could it be much lower?** [MV10] states their "numerical belief is $S \approx 1.5$" but also notes that "a hidden algebraic or number-theoretic structure could yield a much smaller $S$, the possibility of which is by no means excluded." [AE25] and [TTT26] do not challenge this assessment. **Status: Unresolved.** This is Key Unknown K1 below.

3. **Sparse/irregular vs. smooth extremizer.** [MV10] reports "sparse, irregular coefficient patterns" in optimal polynomials, suggesting combinatorial/number-theoretic structure. But [TTT26] §3b notes the $L^2$ extremizer has a boundary singularity $f(x) \sim 1/\sqrt{1/4 - x^2}$, suggesting smooth-but-singular structure. **Resolution needed:** These observations are not necessarily contradictory — the discretized optimal step function may appear "sparse" as an artifact of approximating a singular continuous function on a coarse grid. Testing at high $n$ with non-uniform grids concentrated near $\pm 1/4$ would clarify.

---

## 5. KEY UNKNOWNS

### K1. Is the true $C_{1a}$ close to 1.50, or could it be significantly lower?

**Why it matters:** If $C_{1a} \approx 1.50$, then the upper bound is essentially tight and all effort should go to the lower bound. If $C_{1a}$ could be, say, 1.40 or lower, then there exists a qualitatively different type of function not yet discovered.

**How to investigate:**
- **(Computational)** Solve the SDP relaxation [KB Method G] at $P = 50$–$200$ with RLT cuts and report the lower bound on the discrete optimum. If the SDP bound at $P=200$ is $\geq 1.49$, it certifies that no 200-step function can beat 1.49, making $C_{1a} \approx 1.50$ very likely. If the SDP bound is significantly below 1.49, there is room for better constructions.
- **(Analytical)** Derive Euler–Lagrange conditions for the extremizer and check whether they admit solutions with $\|f*f\|_\infty / \|f\|_1^2 \ll 1.50$. Even a rough asymptotic analysis of the necessary conditions would be informative.

### K2. Does the extremizer have a boundary singularity?

**Why it matters:** If $f^*(x) \sim |x \pm 1/4|^{-\alpha}$ for some $\alpha \in (0, 1)$ (so $f^* \in L^1$ but $f^* \notin L^\infty$), then uniform-grid step functions are systematically poor approximations, and the diminishing returns in §3 could be an artifact of discretization rather than proximity to the true minimum.

**How to investigate:**
- **(Computational)** Run the best available optimizer (e.g., the LP iteration of [MV10] or the RL strategy of [TTT26]) on **non-uniform grids** with geometric spacing near $\pm 1/4$. If the optimal step function coefficients grow unboundedly as the grid refines near the boundary, this supports a singularity.
- **(Analytical)** Study the Euler–Lagrange equation from K1 near $x = \pm 1/4$. The $L^2$ extremizer (White 2023) has $f(x) \sim (1/4 - |x|)^{-1/2}$; determine whether the $L^\infty$ extremizer has analogous behavior. Read White (2023) for methodology.

### K3. Is the SDP relaxation tight at small $P$?

**Why it matters:** If the SDP at $P=50$ is tight (i.e., the optimal $X^*$ has rank 1), then the SDP value equals the global minimum of the 50-step discretized problem, and the gap between the SDP bound and the best known feasible solution quantifies the local-vs-global gap exactly. If it is not tight, the SDP provides only a weak lower bound, and its utility for certifying near-optimality is limited.

**How to investigate:**
- **(Computational)** Solve the SDP from [KB Method G] at $P = 50$ with maximal RLT tightening. Check $\text{rank}(X^*)$. If rank $> 1$, try adding moment/localizing constraints from the Lasserre hierarchy (degree-4 or degree-6 relaxations). Report the SDP value and compare to the best known feasible solution at $P=50$.
- **(Analytical)** The structure $\|f*f\|_\infty = \max_m h^T K_m h / (2P)$ is a max-of-quadratics. Lasserre-hierarchy tightness results for such problems (e.g., Nie 2014) may give conditions under which finite-degree relaxations are exact.

### K4. Can the $L^\infty$ autoconvolution problem be convexified via a moment/SOS formulation?

**Why it matters:** This is the "silver bullet" question. The $L^2$ problem was convexified by White (2023) using Parseval's theorem, converting $\|f*f\|_2^2 = \int |\hat{f}|^4$ into a convex program in Fourier space. An analogous convexification for $\|f*f\|_\infty$ would transform the entire landscape from non-convex to convex, enabling certifiably optimal solutions.

**How to investigate:**
- **(Analytical)** The constraint $\|f*f\|_\infty \leq \eta$ is equivalent to $\eta - (f*f)(t) \geq 0$ for all $t$. If $f$ is represented by finitely many Fourier coefficients, this becomes a nonnegativity condition on a trigonometric polynomial, which is equivalent to an SDP via the Fejér–Riesz theorem. Formalize this and check whether the resulting SDP is tractable. The key question is whether the nonnegativity constraint $f \geq 0$ (a constraint on $\hat{f}$) and the autoconvolution bound $\eta - (f*f)(t) \geq 0$ (a constraint on $|\hat{f}|^2$) can be simultaneously encoded as LMIs of manageable size.
- **(Read)** White (2023) for the $L^2$ convexification; Dumitrescu (2007) *Positive Trigonometric Polynomials and Signal Processing Applications* for SDP formulations of trigonometric nonnegativity.

### K5. What is the exact theoretical ceiling of the Fourier kernel lower bound method?

**Why it matters:** [MV10] §5 estimates the ceiling at $\sim 1.276$ but does not prove this rigorously. If the ceiling is provably $\leq 1.30$ (say), then closing the gap requires entirely new lower bound techniques — and a concrete understanding of *why* the kernel method fails would guide the search for such techniques.

**How to investigate:**
- **(Computational)** Solve the dual of the convex QP from [MV10] Remark (p. 5): minimize $\|f_s * \beta_\delta\|_2^2$ over nonneg symmetric $f_s$ via discretized QP at high resolution. The optimal value gives the tightest bound achievable by the kernel method. Compare with the stated estimate of $\sim 1.276$.
- **(Analytical)** The kernel method bounds $\|f*f\|_\infty$ below by $\langle f*f, K \rangle$ for a suitable kernel $K$. The ceiling is $\sup_K \inf_{f \geq 0} \langle f*f, K \rangle / \|f\|_1^2$, which is itself a minimax problem. Dualizing might reveal which properties of $f*f$ the kernel method fundamentally cannot capture.

---

## Summary of Recommended Attack Priorities

Ranked by expected impact per unit effort:

1. **SDP certification at moderate $P$ (K3).** Solve [KB Method G] at $P = 100$–$200$ to bound the local-vs-global gap. If the SDP bound is $\geq 1.49$, this essentially confirms $C_{1a} \approx 1.50$ and redirects effort to the lower bound. Low technical risk, high information value.

2. **Non-uniform grid experiments (K2).** Run existing optimizers on geometrically-refined grids near $\pm 1/4$. If the extremizer is singular, this could drop the upper bound below 1.50 with modest code changes. Low cost, moderate potential payoff.

3. **Fourier-domain SDP formulation (K4).** Attempt a Fejér–Riesz / moment-based convexification. This is the highest-risk, highest-reward direction: if it works, it solves the problem; if it fails, the failure mode itself is informative. Requires analytical effort; start with small truncation orders.

4. **Euler–Lagrange analysis (K1, K2).** Derive necessary conditions for the extremizer. Even partial results (e.g., ruling out monotonicity, or establishing boundary behavior) would constrain the search space and potentially resolve the sparse-vs-singular tension in §4.3.

5. **Warm-start prox-linear / LP from [AE25] or [TTT26] solutions (§4.2, item 5).** The lowest-hanging computational fruit: take the best known 1,319-piece or 30,000-piece solution, run LP iteration or prox-linear polishing. Never attempted despite being suggested by [KB].
