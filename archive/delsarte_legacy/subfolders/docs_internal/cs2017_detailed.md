# Cloninger–Steinerberger (2017): Detailed Notes on "On Suprema of Autoconvolutions with an Application to Sidon Sets"

**arXiv:** [1403.7988](https://arxiv.org/abs/1403.7988) (v3, 22 Apr 2016; Proc. AMS 145(8), 2017)
**Authors:** Alexander Cloninger, Stefan Steinerberger (Yale)

## 0. Setting and Main Result

Let $f : \mathbb{R} \to \mathbb{R}_{\ge 0}$ be supported on $[-1/4, 1/4]$. The paper works with the constant
$$c := 2/\sigma^2,\qquad c = \inf_{f\ne 0}\frac{\|f*f\|_{L^\infty(\mathbb{R})}}{\left(\int_{-1/4}^{1/4} f\right)^2} \cdot \tfrac{1}{2},$$
which is the Sidon autoconvolution constant $C_{1a}$ (up to the factor 2 convention — the paper proves $\|f*f\|_\infty \ge 1.28 \cdot (\int f)^2$ which, normalizing $\int f = 1$, equals our $C_{1a}$).

**Main Theorem (p. 2):**
$$\sup_{x\in\mathbb{R}}\int_{\mathbb{R}} f(t)\,f(x-t)\,dt \;\ge\; 1.28 \left(\int_{-1/4}^{1/4} f(x)\,dx\right)^2.$$

Predecessor bounds listed in §1.2: trivial $\ge 1$; Cilleruelo–Ruzsa–Trujillo $\ge 1.151$; Green $\ge 1.178$; Martin–O'Bryant $\ge 1.183$, later $\ge 1.263$; Yu $\ge 1.251$; Matolcsi–Vinuesa $\ge 1.274$ (later refined to 1.2749). Upper bound 1.50992 (Matolcsi–Vinuesa). MV claim their Fourier approach cannot exceed ~1.276.

**Stronger form (Full Theorem, §2):** there exists $J\subset(-1/2,1/2)$ with $|J|\ge 1/48$ such that
$$\int_J (f*f)(x)\,dx \;\ge\; 1.28\,|J|\,\left(\int_{-1/4}^{1/4} f\right)^2.$$

## 1. Nature of the Method: Dual (Averaged) Not Pointwise

Conceptually this is a **dual / averaging / pigeonhole** argument — not an explicit primal $f$. The key identity is
$$\|f*f\|_{L^\infty(\mathbb{R})}\;\ge\;\|f*f\|_{L^\infty(I)}\;\ge\;\frac{1}{|I|}\|f*f\|_{L^1(I)}\qquad(\text{eq., §2}).$$
Instead of proving $(f*f)(x_0)$ is large at some specific $x_0$, they prove the **window average** $\frac{1}{|I|}\int_I f*f$ must be large for every $f$ with $\|f*f\|_\infty \le 1.28$.

There is no single dual test function $k(x)$ in the MV/Delsarte sense. Instead, they produce a **finite list of linear inequalities** (one per sub-interval pair) that any $f$ with $\|f*f\|_\infty \le 1.28$ must satisfy; then they prove, by exhaustive computer enumeration over discrete mass distributions, that no such $f$ exists.

This is why CS can pass MV's 1.276 ceiling: MV's 1.276 is only the ceiling of **one specific** analytic dual (an arcsine-kernel SOS polynomial with a specific support). CS implicitly uses a much richer class of inequalities — all window averages on all Minkowski-sum blocks — then certifies infeasibility numerically.

## 2. The Core Inequality (pigeonhole on Minkowski sums)

Partition $(-1/4, 1/4)$ into $2n$ equal intervals $I_j = \bigl(\frac{j}{4n},\frac{j+1}{4n}\bigr)$, $j=-n,\ldots,n-1$. Let
$$a_j := 4n\int_{I_j} f(x)\,dx,\qquad \sum_{j=-n}^{n-1} a_j = 4n\int f = 4n.$$
The bin $a$-vector lives in the simplex
$$A_n=\left\{(a_{-n},\ldots,a_{n-1})\in\mathbb{R}_+^{2n} : \sum a_i = 4n\right\}.$$

**Lemma 1 (p. 4):** For any integers $2 \le \ell \le 2n$ and $-n \le k \le n-\ell$:
$$\frac{1}{4n\ell}\sum_{k\le i+j\le k+\ell-2} a_i a_j \;\le\; \|f*f\|_{L^\infty}.$$

Proof: the Minkowski sum $I_i + I_j \subseteq [k/(4n),(k+\ell)/(4n)]$ whenever $k\le i+j\le k+\ell-2$ (of length $\ell/(4n)$). Fubini + averaging gives
$$\tfrac{4n}{\ell}\sum_{k\le i+j\le k+\ell-2}(\int_{I_i}f)(\int_{I_j}f) = \tfrac{4n}{\ell}\int_{k/(4n)}^{(k+\ell)/(4n)} (f*f)(x)\,dx\;\le\;\|f*f\|_\infty.$$
Defining
$$a_n := \min_{a\in A_n}\max_{2\le\ell\le 2n}\;\max_{-n\le k\le n-\ell}\;\frac{1}{4n\ell}\sum_{k\le i+j\le k+\ell-2} a_i a_j,$$
Lemma 1 gives $c \ge a_n$. So a lower bound on $a_n$ is a lower bound on $C_{1a}$.

Thus the continuous problem is replaced by a **finite-dimensional polynomial minimax** on a simplex. This is the crux: **no shape of $f$ within bins is used, only bin masses**. An independent re-derivation of this "no-correction" form is in `proof/coarse_cascade_method.md` Theorem 1 (identical statement).

## 3. Discretization of the Simplex

**Lemma 2 (§3.2, p. 5):** For $n,m\in\mathbb{N}$ the rational lattice
$$B_{n,m}=\left\{(b_{-n},\ldots,b_{n-1})\in\bigl(\tfrac{1}{m}\mathbb{N}\bigr)^{2n}:\sum b_i=4n\right\}$$
is a $1/m$-net of $A_n$ in the $\ell^\infty$ norm. (Proof: explicit round-down/round-up rule with running residual $t_i\in(-1/m,0]$.) So every real bin vector $a$ is $1/m$-close in $\ell^\infty$ to some integer-mass vector $b$.

**Lemma 3 (Discretization, §3.2 p. 6):** Define
$$b_{n,m}:=\min_{b\in B_{n,m}}\max_{2\le\ell\le 2n}\max_{-n\le k\le n-\ell}\frac{1}{4n\ell}\sum_{k\le i+j\le k+\ell-2} b_i b_j.$$
Then
$$\boxed{\;c\;\ge\;b_{n,m}-\frac{2}{m}-\frac{1}{m^2}\;}.$$

Proof sketch: write $f=g+\varepsilon$ with $g$ the step function associated to $b\in B_{n,m}$ and $\|\varepsilon\|_\infty\le 1/m$. Then $g*g=f*f-2(f*\varepsilon)+\varepsilon*\varepsilon$, and on $(-1/2,1/2)$
$$|\varepsilon*\varepsilon|\le\tfrac{1}{m}\int|\varepsilon|\le\tfrac{1}{m^2},\qquad |f*\varepsilon|\le\tfrac{1}{m}\int f=\tfrac{1}{m}.$$
Hence any upper bound on $\|f*f\|_\infty$ propagates to $g*g$ with additive slack $2/m+1/m^2$.

**Refinement (eq. 1, §3.3 p. 7):** they actually use the tighter, position-dependent correction
$$|(f*\varepsilon)(x)|\le\tfrac{1}{m}\int_{-1/4+\max(0,x)}^{1/4+\min(0,x)}f(x-y)\,dy,$$
giving per-window corrections bounded by $W_{\text{int}}/(2n m)$ where $W_{\text{int}}$ is the mass in the window's interior — cf. the `correction` and `corr_w` terms in `cloninger-steinerberger/pruning.py` and `solvers.py`.

## 4. Asymmetry (starter) argument

Before enumerating, §3.3 notes the free bound: if $\int_{-1/4}^0 f\ge\alpha$ then by Minkowski-sum containment of $[-1/4,0]+[-1/4,0]\subseteq[-1/2,0]$ and averaging,
$$\sup_{-1/2\le x\le 0}(f*f)(x)\;\ge\;\frac{\alpha^2}{1/2}=2\alpha^2.$$
Taking $\alpha=0.8$ gives $1.28$ outright. Hence one can **restrict the search** to distributions with $0.2\le\int_{-1/4}^0 f\le 0.8$. This is the `asymmetry_prune_mask` / `asym_val = 2*(dom-margin)**2` in `cloninger-steinerberger/pruning.py:42-64`.

## 5. The Cascade (multiscale enumeration)

From §3.3 pp. 7–8, the actual algorithm:

- Fix $m=50$.
- Start at $n=3$: enumerate $B_{3,50}$. For each $b$, test whether any Lemma 3 inequality already rules it out (test value $>1.28+2/m+1/m^2$).
- Surviving vectors are **dyadically refined**: each bin is split in half, promoting $b\in B_{n,m}$ to candidates in $B_{2n,m}$ consistent with parent masses. Children inherit constraints from parents.
- Repeat: $B_{3,50}\to B_{6,50}\to B_{12,50}\to B_{24,50}$.
- End state: no $b\in B_{24,50}$ is consistent with $\|f*f\|_\infty\le 1.28$.

By Lemma 3 this rules out all $f$ with $\|f*f\|_\infty\le 1.28$.

**Size:** $|B_{n,m}|=\binom{4nm+2n-1}{2n-1}$. At $n=24, m=50$ this is $\binom{4999}{47}\approx 10^{99}$ — gigantic. The cascade is only tractable because (i) pruning kills the vast majority of configurations early, and (ii) parents imply constraints on whole families of children.

## 6. Implementation and Hardware

§3.4 (p. 8): each configuration's pruning check is phrased as a **matrix-vector product**. For each $(n,\ell,k)$ there is a precomputed $C_k\in\mathbb{R}^{(2n)^2\times 4n-k+1}$; for a candidate $b$ with refinement vector $c_b=[c_\gamma]\in\mathbb{R}^{N\times 2n}$, $N=\prod(1+m b_i)$, compute $F[\gamma,n(i-1)+j]=c_\gamma[i]c_\gamma[j]$ and check
$$F[\gamma,\cdot]\,C_k > 1.28 + 2/m + 1/m^2.$$
This is a GPU-friendly dense matmul; parent bins are independent, so trivially parallel.

**Machine:** Yale Omega, 7 CPU nodes with dedicated GPUs. **Total compute: ~20,000 CPU hours** for the 1.28 certificate.

## 7. Rigor

The certificate is **floating-point numerical**, not interval-arithmetic. §3.3–3.4 use ordinary FP arithmetic with a fixed numerical headroom $2/m + 1/m^2 = 0.0404$. There is no explicit discussion of round-off error bounds, no certified SDP solver, no interval arithmetic. In practice the headroom (4 × 10⁻² vs. FP epsilon 10⁻¹⁶) is immense, so the result is morally rigorous but not formally verified.

The companion codebase (`cloninger-steinerberger/pruning.py:23`) implements the identical correction $2/m+1/m^2$ in double precision with an added $10^{-9}$ `fp_margin`.

## 8. Why CS Beats MV's 1.276

MV's 1.2749 is the Lagrangian dual value of one specific Delsarte-style SDP over a polynomial cone (see `delsarte_dual/mv_construction_detailed.md`). Its 1.276 "ceiling" is a property of that one-parameter SOS family. CS's test functions $\mathbf{1}_{[k/(4n),(k+\ell)/(4n)]}(x+y)$ — indicator of a Minkowski-sum interval — form a much richer test class. For large $n$ and all $\ell$, this family can separate any continuous $f$ from the $\|f*f\|_\infty\le c$ slab down to $c=C_{1a}$ (§3, "Our method seems to only be limited by our ability to do large-scale computations"). CS is thus a strictly stronger dual than MV on a discretized space.

## 9. Concrete Parameters Used

| Parameter | Value | Meaning |
|---|---|---|
| $n_{\text{start}}$ | 3 | Starting half-dimension ($d=2n=6$ bins) |
| $n_{\text{end}}$ | 24 | Final half-dimension ($d=48$ bins) |
| $m$ | 50 | Height quantum $1/50$ |
| Correction | $2/m+1/m^2=0.0404$ | Additive slack per window |
| Asymmetry pre-filter | $\alpha\in[0.2,0.8]$ | Reduces search space |
| Hardware | 7-node Omega cluster + GPUs | ≈20,000 CPU-hrs |
| Achieved bound | $C_{1a}\ge 1.28$ | vs. MV's 1.2749 |

## 10. Could it Push Higher? Extensions.

1. **Higher $n,m$:** at fixed $c=1.29$, the threshold $c+2/m+1/m^2$ needs $m\gtrsim 200$ and $n\gtrsim 60$ (because the continuous optimum $\max_\ell \cdot$-TV at finite $n$ plateaus around $1.38$ at $n=32$, see `proof/coarse_cascade_method.md` §5.3). The fine grid $B_{60,200}$ has $\binom{48000+119}{119}\sim 10^{430}$ configurations — intractable by brute force.

2. **No-correction variant:** (`coarse_cascade_method.md` §3). The correction $2/m+1/m^2$ can be dropped entirely if one proves **refinement monotonicity** (the max-TV is non-decreasing under dyadic splits). This decouples grid resolution from bound quality: at $d=128$ the min-max TV $\approx 1.420$, so $c_{\text{target}}=1.40$ becomes achievable with a fixed coarse mass quantum $S=50$. Refinement monotonicity is empirically verified (183K tests, 0 violations) but unproven.

3. **Combine with MV:** CS's windowed Minkowski-sum test functions and MV's arcsine-kernel SOS test function are not redundant — they separate different parts of the simplex. A joint SDP that adjoins MV's dual cuts to the CS inequalities might achieve $\sim 1.29{-}1.30$. Empirically (see `delsarte_dual/family_f4_mv_arcsine.py`), the MV dual at the CS survivors of `B_{24,50}` gives ~0.007 additional margin, i.e. a combined $\sim 1.287$ bound.

4. **Lasserre SDP (current project focus):** The CS method's inequalities are the $\ell=2$ level of Lasserre for the polynomial minimax $\min_{\mu\in\Delta_{2n}}\max_W \mu^TM_W\mu$. Higher Lasserre levels ($t=2,3,\ldots$) automatically imply all CS-style linear cuts and more. With correlative sparsity (Waki et al. 2006) the level-2 Lasserre for $d=64$ should be tractable on a single machine and is the route pursued in `lasserre/`. See `CLAUDE.md`.

5. **Interval arithmetic certificate:** a formally verified CS certificate would fit in Lean via `kernels` — the existing `lean/Sidon/Proof/DiscretizationError.lean` formalizes exactly Lemma 3, and `lean/Sidon/Algorithm/GpuKernelSoundness.lean` wraps the pruning kernel. The numerical gap $0.04$ dwarfs FP epsilon, so this is low-risk.

## References (from the paper's bibliography)

[1] Bonze, Gollowitzer, Yildirim — rounding on the simplex (used in Lemma 2)
[2] Cilleruelo, Ruzsa, Vinuesa — equivalent continuous problem
[3] Cilleruelo, Ruzsa, Trujillo — $c\ge 1.151$
[5] Green — $c\ge 1.178$
[7,8] Martin, O'Bryant — 1.183, 1.263
[9] Martin, O'Bryant — Illinois J. Math. 2009 (supremum of autoconvolutions)
[10] Matolcsi, Vinuesa — 1.2749 via Fourier / arcsine SOS
[11] Schinzel, Schmidt — $L^1/L^\infty$ comparison (origin of problem)
[12] Yu — $c\ge 1.251$

## Cross-reference to local code

- `cloninger-steinerberger/pruning.py:11` — `correction(m) = 2/m + 1/m²` (paper Lemma 3)
- `cloninger-steinerberger/pruning.py:42` — asymmetry pre-filter (paper §3.3 starter)
- `cloninger-steinerberger/test_values.py:10` — window-TV kernel (paper Lemma 1)
- `cloninger-steinerberger/solvers.py:1201` — `run_single_level` (one cascade stage)
- `cloninger-steinerberger/compositions.py` — enumeration of $B_{n,m}$
- `proof/cs-proof/lower_bound_proof.tex` — LaTeX writeup of the CS-style argument adapted to this codebase
- `proof/coarse_cascade_method.md` — no-correction variant (Theorem 1)
- `lean/Sidon/Proof/DiscretizationError.lean` — Lean formalization of Lemma 3
