# Multi-scale arcsine kernel

**Status:** WORKING. The line that produced the
**Piterbarg-Bajaj-Vincent Bound**.

## Result

$$ C_{1a} \;\ge\; \frac{1292}{1000} \;=\; 1.292 $$
via the three-scale arcsine kernel, certified through `flint.arb`
interval arithmetic at 256-bit precision and formalised in Lean 4
(`lean/Sidon/MultiScale.lean`). This improves on the previously
announced Cloninger--Steinerberger value of $1.2802$ (2017,
arXiv:1403.7988) and on the Matolcsi--Vinuesa 2010 analytic baseline
$1.27481$ (the latter by $+0.0172$).

The two-scale precursor (described below) produced $C_{1a} \ge 1651/1280
\approx 1.28984$ and was independently audited by 14 agents
(13 CONFIRM, 1 soft FLAG on QP conditioning, non-blocking).

## Construction

For $\delta \in (0, 1/4]$, let $K_{\rm arc}(\delta; \cdot)$ denote the
auto-convolution arcsine kernel on $[-\delta, \delta]$ with
$\widehat{K_{\rm arc}(\delta; \cdot)}(\xi) = J_0(\pi \delta \xi)^2$
(Sonine identity). For weights $\lambda_i \ge 0$ summing to $1$ and
half-widths $\delta_i \in (0, 1/4]$, the *multi-scale arcsine kernel* is
$$ K(x) \;=\; \sum_i \lambda_i \, K_{\rm arc}(\delta_i; x). $$

Each property of Bochner-admissibility is closed under convex
combinations: $K \ge 0$,
$\operatorname{supp} K \subseteq [-\max_i \delta_i, \max_i \delta_i]$,
$\int K = 1$, and $\widehat K(\xi) = \sum_i \lambda_i J_0(\pi \delta_i \xi)^2
\ge 0$ (a sum of squares of real Bessel functions). The MV master
inequality
$$ M + 1 + \sqrt{(M-1)(K_2 - 1)} \;\ge\; \frac{2}{u} + a, \qquad a = \frac{4}{u} \cdot \frac{m_G^2}{S_1}, $$
applies verbatim, with $u = 1/2 + \delta_1$, $S_1 = \sum_{j=1}^N a_j^2
/ \widetilde K(j)$, $K_2 = \|K\|_{L^2}^2$, $m_G = \min_{[0,1/4]} G$.

## Anchors

### Three-scale current (publication)

| Parameter | Value |
|---|---|
| $(\delta_1, \delta_2, \delta_3)$ | $(138, 55, 25)/1000$ |
| $(\lambda_1, \lambda_2, \lambda_3)$ | $(85, 10, 5)/100$ |
| $u$ | $638/1000$ |
| $N$ (cosine modes in $G$) | $200$ |
| $k_1 = \widehat K(1)$ | $\ge 0.92124658$ |
| $K_2$ | $\le 4.788906$ |
| $S_1$ | $\le 29.840907$ |
| $m_G$ | $\ge 0.99997987$ |
| gain $a$ | $\ge 0.21009214$ |
| $M_{\rm cert}$ | $\ge 1292/1000$ |

### Two-scale precursor (history)

| Parameter | Value |
|---|---|
| $(\delta_1, \delta_2)$ | $(138, 45)/1000$ |
| $(\lambda_1, \lambda_2)$ | $(85, 15)/100$ |
| $N$ | $119$ |
| $k_1$ | $\in [0.92139, 0.92139]$ |
| $K_2$ | $\in [4.7588, 4.7621]$ |
| $S_1$ | $= 31.44196$ |
| $m_G$ | $\ge 0.99996$ |
| gain $a$ | $= 0.19939$ |
| $M_{\rm cert}$ | $\ge 1651/1280$ |

## Why the lift works: Bessel-zero rescue

The QP denominator sum $S_1 = \sum_{j=1}^N a_j^2 / \widetilde K(j)$ is
dominated by indices $j$ where $\widetilde K(j)$ is small. For the
single-scale arcsine ($\delta = 138/1000$), $\widehat K_{\rm arc}(\xi)
= J_0(\pi \delta \xi)^2$ *vanishes* near the first Bessel zero $z_1
\approx 2.4048$, i.e. at $\xi = z_1/(\pi \delta) \approx 5.55$. The
corresponding $\widetilde K(j)$ touch $\sim 10^{-6}$ and the reciprocals
blow up.

Adding a second (and third) scale with smaller $\delta_i$ shifts the
zeros of $J_0(\pi \delta_i \xi)^2$ away from the dual frequencies
$\{j/u\}_{j=1}^N$. The minimum of $\widetilde K_{\rm ms}(j)$ over $j \in
\{1, \dots, N\}$ rises to $\sim 10^{-4}$, and $S_1$ drops from $\sim
87.4$ (single-scale) to $\le 29.84$ (three-scale). The gain $a = (4/u)
m_G^2 / S_1$ rises from $\sim 0.072$ to $\ge 0.20958$.

Decomposition of the two-scale lift, varying one anchor at a time
relative to the single-scale MV baseline:

| Varied | $M_{\rm cert}$ | $\Delta M$ |
|---|---:|---:|
| $k_1$ alone | $1.27510$ | $+0.00011$ |
| $K_2$ alone | $1.26455$ | $-0.01044$ |
| $S_1$ alone | $1.29091$ | $+0.01592$ |
| all three | $1.28013$ | $+0.00514$ |

$K_2$ is a *headwind*: the small-scale component inflates $\|K\|_{L^2}^2$
and enters the radicand with a positive sign. The $S_1$ engine defeats
the $K_2$ headwind by $\approx 3{:}2$. The QP must be re-optimised for
each new $\widehat K$; using MV's fixed $G$ at multi-scale gives only
$+0.005$.

## The "MV ceiling" finding

MV (2010, p. 4 line 220) write: "We are quite convinced that the choice
of $K$ in [MO] is optimal, and we will not change it." This is informal,
and its quantifier scope is *single-scale* arcsine kernels. Their derivation
also uses the structural identity $\widetilde K(j) = u \,\widetilde\eta(j)^2$
(a single-square representation), which the multi-scale $\widetilde K_{\rm ms}$
does not satisfy (a sum of squares is not a single square). Hence the
informal "ceiling" does not apply to multi-scale $K$. The 31-kernel
single-family sweep ([`single_kernel_sweep.md`](single_kernel_sweep.md))
confirmed no non-arcsine *single* kernel beats MV.

## Pipeline

| Stage | Module |
|---|---|
| 1 | `delsarte_dual/grid_bound_alt_kernel/kernels.py:MultiScaleArcsineKernel` (arb-rigorous $\widehat K$, $K_2$ via `acb.integral` + Watson tail). |
| 2 | `delsarte_dual/grid_bound_alt_kernel/optimize_G.py` (QP for $G$, denominator $10^8$). |
| 3 | `delsarte_dual/grid_bound/G_min.py` (Taylor-2 B&B for $m_G$). |
| 4 | `delsarte_dual/grid_bound/phi.py` (master inequality $\Phi(M)$). |
| 5 | `delsarte_dual/grid_bound/cell_search.py` (cell bisection on $[0, \mu(M)]$). |
| 6 | `delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py` (bisection on $M$). |
| 7 | `lean/Sidon/MultiScale.lean` (Lean 4, 0 sorries). |

## Audit of the two-scale precursor

| ID | Topic | Verdict |
|---|---|---|
| V1 | Degenerate sanity (single-scale $\to$ MV baseline) | CONFIRM |
| V2 | $K_2$ via mpmath at 50 digits | CONFIRM |
| V3 | Watson Bessel tail | CONFIRM |
| V4 | Martin-O'Bryant surrogate $0.5747/\delta$ | CONFIRM |
| V5 | MV inequality universality | CONFIRM |
| V6 | `CERTIFIED_FORBIDDEN` logic | CONFIRM |
| V7 | QP $S_1$ independent reproduction | CONFIRM |
| V8 | $m_G$ Taylor B&B | CONFIRM |
| V9 | $k_1$ enclosure | CONFIRM |
| V10 | Lean (0 sorries) | CONFIRM |
| V11 | SHA-256 + reproducibility | CONFIRM |
| V12 | Bochner at QP frequencies | CONFIRM (soft: $\min_j \widetilde K(j) = 3.3 \times 10^{-5}$, non-blocking) |
| V13 | MV master-inequality re-derivation | CONFIRM |
| V14 | Cross-Bessel retraction | CONFIRM |

## Alternative master inequalities and K28 sign-uncertainty

Multi-moment MM-$N$, Martin-O'Bryant Prop 2.11, and the V2 3-point SDP
all reduce to the same $M_{\rm cert}$ as MV's eq. (10) to within
$5 \times 10^{-5}$ ([`master_attacks.md`](master_attacks.md)); the lift
is driven by the kernel-side $S_1$ collapse, not downstream
inequalities. The Cohn-Goncalves (CG) and Bourgain-Clozel-Kahane sign-
uncertainty principles do not apply to the primal $\rho = f * f$: it is
doubly non-negative, incompatible with the CG hypothesis $\widehat\rho
\le 0$ outside $[-\rho_2, \rho_2]$ (Paley-Wiener forces $\rho \equiv 0$).
On the dual test function $g$, CG only restricts the feasible set.
Viazovska-style modular ansatz adapted to $C_{1a}$'s Paley-Wiener type
remains unexplored.

## Open extensions

1. *Four or more scales.* Numerical sweep at three scales saturates near
   $1.292$; additional scales add $\sim 10^{-3}$ at best before $K_2$
   inflation dominates.
2. *Cross-family mixes.* Arcsine + Chebyshev-$\beta$ / B-spline / Wendland
   convex combinations remain admissible. Whether they beat pure-arcsine
   mixes is open.
3. *Joint $(K, G)$ optimisation.* Currently $\delta_i, \lambda_i$ are
   grid-tuned before the QP; a joint variational approach is mechanical.
4. *Lean closure.* Two pending mathlib-side axioms
   (`bessel_J0_squared_nonneg`, `K_multi_fourier_eq`) become theorems
   once the Bessel API lands; see [`../formalization.md`](../formalization.md).

## References

- M. Matolcsi, C. Vinuesa (2010), [arXiv:0907.1379](https://arxiv.org/abs/0907.1379).
- G. Martin, K. O'Bryant (2009), [arXiv:0807.5121](https://arxiv.org/abs/0807.5121).
- D. Bailey, J. Borwein, D. Broadhurst, M. Glasser (2008), [arXiv:0801.0891](https://arxiv.org/abs/0801.0891).
- G. N. Watson, *A Treatise on the Theory of Bessel Functions* (2nd ed., 1944), §13.46, §7.21.
- See [`../proof_outline.md`](../proof_outline.md), [`../formalization.md`](../formalization.md), [`../reproducibility.md`](../reproducibility.md).
