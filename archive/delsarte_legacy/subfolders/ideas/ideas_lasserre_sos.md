# Lasserre / SOS Ideas for a Lower Bound on $C_{1a}$

**Date:** 2026-04-20.
**Context.** MV (2010) $1.2748$ and CS (2017) $1.28$ are the last rigorous lower
bounds; Xie's $1.2802$ is unpublished. MV's Cauchy–Schwarz dual is saturated —
any rigorous improvement needs a **non-Cauchy–Schwarz** certificate. Moment-SOS
is the natural candidate because it produces certificates strictly stronger than
any single Cauchy–Schwarz inequality (those are the degree-2 part of the
hierarchy).
**Companion:** `delsarte_dual/sdp_hierarchy_design.md` (deeper derivation) and
`delsarte_dual/literature_survey_2020_plus.md`.

---

## 1. Cleanest SDP formulation for a LOWER bound on $C_{1a}$

We want $\lambda$ with $\lambda \le C_{1a}$. Take the primal

$$
C_{1a} \;=\; \inf_{\mu\in\mathcal P([-\tfrac14,\tfrac14])}\ \sup_{t\in[-\tfrac12,\tfrac12]}\ (\mu*\mu)(t),
$$

dualise the inner $\sup$ by a second probability measure $\nu$ on $[-\tfrac12,\tfrac12]$
so $\sup_t (\mu*\mu)(t) \ge \int(\mu*\mu)\,d\nu$, and push the Dirac out of the
integrand by smoothing it with a Christoffel–Darboux kernel
$K_N(u)=\sum_{j=0}^N p_j(u)^2$, $\{p_j\}$ Legendre-orthonormal on
$[-\tfrac12,\tfrac12]$. Introducing joint moments
$y_{\alpha\beta\gamma}=\int x^\alpha y^\beta t^\gamma\,d\mu(x)d\mu(y)d\nu(t)$,
the **Lasserre level-$k$ / kernel-$N$ relaxation** is

$$
\begin{aligned}
\lambda_{k,N} \;=\;\min_{y}\quad & \sum_{j=0}^{N} \Big\langle\, p_j(x+y)\,p_j(t),\ y\,\Big\rangle \\
\text{s.t.}\quad & y_{000}=1, \\
                 & M_k(y)\succeq 0\quad(\text{size } \binom{3+k}{k}),\\
                 & (\tfrac1{16}-x^2)\cdot M_{k-1}(y)\succeq 0,\\
                 & (\tfrac1{16}-y^2)\cdot M_{k-1}(y)\succeq 0,\\
                 & (\tfrac14-t^2)\cdot M_{k-1}(y)\succeq 0,\\
                 & y_{\alpha\beta\gamma}=y_{\beta\alpha\gamma}\quad(\text{exchange symmetry}),\\
                 & \mathbb Z/2\text{ reflection } x,y,t\mapsto -x,-y,-t.
\end{aligned}
$$

Then $\lambda_{k,N}\le C_{1a}+\varepsilon_N$ with a **rigorous, explicit**
kernel-approximation error $\varepsilon_N=O(1/N)$ (Christoffel–Darboux /
Jackson-type bound, Slot–Laurent 2022). Adding back $\varepsilon_N$ yields a
**certified lower bound** $\lambda_{k,N}-\varepsilon_N \le C_{1a}$.

**Why this is the right formulation.**
- One convex SDP, no bilevel/saddle solve.
- The two $\mu$ factors are naturally handled by the fact that $y_{\alpha\beta\cdot}$
  is a joint moment matrix of a $\mu\!\otimes\!\mu$ measure on
  $[-\tfrac14,\tfrac14]^2$; dropping the rank-1 constraint $\mu\otimes\mu$ is
  precisely the Lasserre relaxation.
- Every constraint is a localiser of a compact basic semialgebraic set →
  Putinar's Positivstellensatz guarantees $\lambda_{k,N}\to C_{1a}$ as
  $k,N\to\infty$.
- Block-diagonalises under $\mathbb Z/2$ (reuse `lasserre/z2_blockdiag.py`).
- **Strictly dominates** the MV Cauchy–Schwarz dual: that dual is recovered at
  $k=1$, $N=0$; any $k\ge 2$ can in principle beat it.

**Variant worth trying.** Work in the *trigonometric* basis on the Fourier side
($\hat f$ is entire of exponential type, $|\hat f|\le 1$, $\hat f(0)=1$). The
trigonometric Lasserre hierarchy has **exponential** convergence
(Bach–Rudi 2022, Slot 2024) rather than $O(1/k^2)$ — potentially a giant win.

---

## 2. Expected convergence rate

| Domain | Best known rate | Reference |
|---|---|---|
| Hypercube $[-1,1]^n$ | $O(1/r)$ (Baldi–Slot) or $O(1/r^2)$ via polynomial kernel | arXiv:2505.00544 (Gribling–de Klerk–Vera 2025) |
| Sphere $S^{n-1}$ | $O(1/r^2)$, near-tight | de Klerk–Laurent 2020 |
| Generic compact basic semialgebraic | $O(1/\log\log r)$ (Putinar generic) to $O(1/r)$ (Łojasiewicz) | Baldi–Mourrain 2022 |
| Trigonometric / torus | **exponential** $O(\rho^{-r})$ | Bach–Rudi, Slot 2024 |
| Kernel smoothing error $\varepsilon_N$ | $O(1/N)$ (Fejér); $O(e^{-cN})$ (CD in analytic regime) | Lasserre–Pauwels 2022 |

For **our** 3-cube at level $k$ with kernel $N$, a realistic model is

$$
C_{1a}-\lambda_{k,N} \;\asymp\; \frac{c_1}{k^2}+\frac{c_2}{N},
$$

with constants we can calibrate once we match a known baseline (e.g. the
MV $1.2748$ bound at $k=2$, $N=10$). To *beat* $1.2802$ by $10^{-4}$
(the published-vs-unpublished gap) requires $k^2\cdot N\gtrsim 10^4$ — tight
but potentially feasible with $k=8,\,N=160$.

**Warning: finite convergence is not guaranteed.** For bilevel/min-max problems
with non-unique optima the hierarchy can be slow (Baldi–Slot showed the sphere
rate is tight). For $C_{1a}$ the MV extremiser is almost certainly not unique
(cf. Schinzel–Schmidt conjecture **disproved** by MV), which is *bad* news for
finite convergence.

---

## 3. Level-$d$ complexity

Three variables $(x,y,t)$ on a compact cube; moment matrix size

$$
s(k)\;=\;\binom{3+k}{k}\;=\;\{4,10,20,35,56,84,120,165,220,\ldots\}\text{ for }k=1,\ldots,9.
$$

| $k$ | $s(k)$ | svec(dense) | SDP variables (joint) | Target |
|---|---|---|---|---|
| 2 | 10 | 55 | 84 | smoke test |
| 4 | 35 | 630 | 495 | match MV 1.2748 |
| 6 | 84 | 3570 | 1716 | push near 1.28 |
| 8 | 165 | 13 695 | 4 495 | target $>$ 1.2802 |
| 10 | 286 | 41 041 | 9 996 | stretch |

Plus **3 localiser blocks** of size $s(k-1)$ each, plus the kernel structure
(multiplies the objective by $N$ terms but does *not* grow the PSD blocks).

**MOSEK runtime** scales as $s(k)^{3}\approx k^9$; the 192-core pod handles
$k\le 10$ comfortably (svec $\le 4\!\cdot\!10^4$, within the range
`lasserre/dual_sdp.py` already solves with clique sparsity).

With $\mathbb Z/2$ block-diagonalisation the PSD blocks halve (factor-4 speed-up).
With **correlative sparsity** (3 overlapping cliques $\{x\},\{y\},\{t\}$ plus
the coupling $\{x,y,t\}$) we can reduce further, but with only 3 variables
total the gain is modest.

---

## 4. Transferable code / tools

**Ready-made libraries.**

1. **TSSOS.jl** (Wang et al., arXiv:2103.00915; CS-TSSOS: ACM TOMS 2023).
   Julia, Mosek backend, implements correlative + term sparsity moment-SOS.
   *Directly applicable*: our 3-cube POP fits its input format (polynomial
   objective + box constraints).
   https://github.com/wangjie212/TSSOS
2. **MomentSOS.jl** (Le Franc). JuMP models for moment-SOS hierarchy.
   https://github.com/adrien-le-franc/MomentSOS.jl
3. **SumOfSquares.jl** (JuMP-dev). The general-purpose Julia SOS frontend;
   more flexible, less sparsity-aware.
4. **GloptiPoly 3** (Henrion–Lasserre–Löfberg, MATLAB).
   Solid for reference / validation.
5. **mompy** (Wang, Python). Small, pedagogical; good for prototyping
   level $k\le 4$.
   https://github.com/sidaw/mompy
6. **POEM** (TU Berlin, Python). Polynomial optimisation with SONC/SDSOS.

**In-repo reuse.**
- `lasserre/dual_sdp.py` — dual SDP infra with clique sparsity, Mosek.
- `lasserre/z2_blockdiag.py` — $\mathbb Z/2$ reflection block-diagonalisation.
- `certified_lasserre/` — exact rational arithmetic on SDP certificates
  (Farkas-style, already produced val(4) > 1.0963).

**Recommended prototype path.**
- Build the model in **TSSOS.jl** (fastest path to $k=6$–$8$ with sparsity).
- Validate against `lasserre/dual_sdp.py` at $k=2$.
- Final certificate through `certified_lasserre/`.

---

## 5. 100-word feasibility assessment

Moderate. A 3-cube Lasserre relaxation with Christoffel-smoothed objective is
mathematically clean, rigorous, and novel for $C_{1a}$; at $k=8,\,N\!\sim\!40$
it fits on existing hardware and would strictly dominate MV's saturated
Cauchy–Schwarz dual. The risk is rate: on the hypercube the known rate is
$O(1/k^2)$ plus an $O(1/N)$ kernel error, so closing the
$1.28\!\to\!1.2802\!+\!\varepsilon$ gap demands $k^2N\gtrsim 10^4$ — feasible
but unforgiving of constant factors. A trigonometric (Fourier) variant may
enjoy exponential convergence and is the most promising line. Recommend
prototyping in TSSOS.jl with a Chebyshev/Fourier basis at $k=4$.

---

## Sources

- [Lasserre (2001), Global optimization with polynomials and the problem of moments](https://epubs.siam.org/doi/10.1137/S1052623400366802)
- [Lasserre (2008), SDP approach to the generalized problem of moments](https://link.springer.com/article/10.1007/s10107-006-0085-1)
- [Lasserre (2024), The Moment-SOS hierarchy: Applications and related topics, Acta Numerica 33](https://www.cambridge.org/core/journals/acta-numerica/article/momentsos-hierarchy-applications-and-related-topics/83B5483C595660346E598010DDC69200)
- [Henrion, Korda, Lasserre (2020), The Moment-SOS Hierarchy (book)](https://www.worldscientific.com/worldscibooks/10.1142/q0252)
- [Slot, Laurent (2022), SOS hierarchies and the Christoffel–Darboux kernel, SIOPT](https://epubs.siam.org/doi/10.1137/21M1458338)
- [Bach, Rudi (2022), Exponential convergence of SOS for trigonometric polynomials, SIOPT](https://epubs.siam.org/doi/10.1137/22M1540818)
- [de Klerk, Laurent (2019), Survey of SDP approaches to the generalized moment problem, arXiv:1811.05439](https://arxiv.org/abs/1811.05439)
- [de Klerk, Laurent (2024), Overview of convergence rates for SOS hierarchies, arXiv:2408.04417](https://arxiv.org/abs/2408.04417)
- [Gribling, de Klerk, Vera (2025), Revisiting Lasserre hierarchy convergence on hypercube, arXiv:2505.00544](https://arxiv.org/abs/2505.00544)
- [Matolcsi, Vinuesa (2010), Improved bounds on the supremum of autoconvolutions, arXiv:0907.1379](https://arxiv.org/abs/0907.1379)
- [Cloninger, Steinerberger (2017), arXiv:1403.7988](https://arxiv.org/abs/1403.7988)
- [Wang, Magron, Lasserre (2021), TSSOS Julia library, arXiv:2103.00915](https://arxiv.org/abs/2103.00915)
- [Wang, Magron, Lasserre, Mai (2023), CS-TSSOS, ACM TOMS](https://dl.acm.org/doi/10.1145/3569709)
- [TSSOS.jl github](https://github.com/wangjie212/TSSOS)
- [MomentSOS.jl github](https://github.com/adrien-le-franc/MomentSOS.jl)
- [mompy github](https://github.com/sidaw/mompy)
- [Lasserre (2011), An algorithm for semi-infinite polynomial optimization, arXiv:1101.4122](https://arxiv.org/abs/1101.4122)
- [Nie (2022), Moment and Polynomial Optimization, SIAM](https://my.siam.org/Store/Product/viewproduct/?ProductId=40626017)
