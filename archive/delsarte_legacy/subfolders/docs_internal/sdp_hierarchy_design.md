# SDP Moment Hierarchy for the Sidon Autoconvolution Constant $C_{1a}$

**Author:** design note, 2026-04-19.
**Scope:** systematic Lasserre-type hierarchies for
$C_{1a}=\inf\{\lVert f*f\rVert_\infty : f\ge 0,\ \operatorname{supp} f\subset[-\tfrac14,\tfrac14],\ \int f=1\}$.
**Current record:** $1.2802\le C_{1a}\le 1.5029$ (lower: Cloninger–Steinerberger; upper: explicit $f$).

---

## 0. Executive summary

The clean SDP formulation is a **bilevel / semi-infinite** Lasserre relaxation on
**two** coupled moment sequences, one for the unknown measure $\mu=f\,dx$ on
$[-1/4,1/4]$ and one for the dual measure $\nu$ on $[-1/2,1/2]$ that witnesses the
max in $\lVert f*f\rVert_\infty$. The right primal reads
$$
C_{1a}\;=\;\inf_{\mu\in\mathcal P([-\tfrac14,\tfrac14])}\;\sup_{t\in[-\tfrac12,\tfrac12]}\;\int\!\!\int \mathbf{1}\!\{x+y=t\}\,d\mu(x)\,d\mu(y),
$$
which is **quadratic** in $\mu$ and **semi-infinite** in $t$. A direct Lasserre
lift on $\mu$ alone does not close because the objective is non-polynomial
(Dirac-in-$t$). The cleanest way to get a **single SDP** is to **dualise $t$ via
a Dirac pairing** and relax the Dirac to a unit-mass moment sequence of a second
measure $\nu$ on $[-1/2,1/2]$. This yields a **Lasserre hierarchy of lower
bounds** on $C_{1a}$ whose convergence we can argue on the hypercube
$[-1/4,1/4]^2\times[-1/2,1/2]$.

**Bottom line (§5).** The quartic-in-moments structure gives an SDP of size
$O(k^6)$ per level $k$, too small to beat the cascade at the levels we can solve
today, but qualitatively different: the same SDP produces a **rigorous dual
certificate** and generalises to $d=\infty$ (no discretisation error).

---

## 1. Formulation candidates

### 1A. Primal moment lift on $f$ (natural but non-polynomial)

Let $m_k:=\int x^k f(x)\,dx$, $k=0,1,2,\dots$ Hausdorff moments of a positive
measure on $[-1/4,1/4]$. The constraints
$$
m_0=1,\qquad H_k\!:=\!\bigl[m_{i+j}\bigr]_{i,j=0}^{k}\succeq 0,\qquad L_k\!:=\!\bigl[(\tfrac14)^2 m_{i+j}-m_{i+j+2}\bigr]\succeq 0
$$
are the standard truncated Hausdorff PSD conditions (localising matrices for
$\tfrac14-x$, $\tfrac14+x$, equivalently $\tfrac1{16}-x^2\ge 0$).
The objective $\lVert f*f\rVert_\infty$ is, however, **not** a polynomial in the
$m_k$: it is
$$
\sup_{t} \int f(x)f(t-x)\,dx \;=\; \sup_t \sum_{k\ge 0} c_k(t)\,m_k\otimes m_k\quad(\text{in a suitable basis}),
$$
which is simultaneously a max over $t$ **and** a product of two moment
vectors. So the pure $m_k$-lift is not enough.

### 1B. Fourier lift on $\hat f$ (Delsarte-like)

Lift to $\hat f$ via Paley–Wiener: $\hat f$ is entire of exponential type $\pi/2$,
$\hat f(0)=1$, $|\hat f|\le 1$. Then $\lVert f*f\rVert_\infty\ge\int \hat g\,|\hat f|^2$
for admissible $g$ (§`theory.md`). The **dual** LP in the Selberg/Beurling
basis is well-developed (Matolcsi–Vinuesa 2010, arXiv:0907.1379, $1.2748$;
Cohn–Elkies-type). It is a **linear program** in the space of admissible test
functions $g$, not an SDP. SOS-lifting to an SDP over $\hat g\ge 0$ is possible
but gives only marginal improvements because the sharp inequality $|\hat f|^2\ge
\cos^2(\pi\xi/2)\mathbf{1}_{|\xi|\le 1}$ is already tight at $\xi=0$. **Verdict:**
this is the correct dual to combine *with* an SDP primal; alone it is an LP.

### 1C. Two-measure lift (recommended)

Introduce a second positive measure $\nu$ on $[-\tfrac12,\tfrac12]$ with
$\int d\nu=1$ and define
$$
J(\mu,\nu)\;:=\;\int\!\!\int\!\!\int \delta(x+y-t)\,d\mu(x)\,d\mu(y)\,d\nu(t)
\;=\;\int (f*f)(t)\,d\nu(t).
$$
Then for any admissible $\nu$, $J(\mu,\nu)\le\lVert f*f\rVert_\infty$; equality
is attained by $\nu=\delta_{t^\star}$. Hence
$$
\boxed{\;C_{1a}\;=\;\inf_{\mu}\;\sup_{\nu}\;J(\mu,\nu)\;=\;\inf_\mu\;\lVert f*f\rVert_\infty.\;}
$$
Swapping $\inf$ and $\sup$ gives a **lower bound**
$\sup_\nu\inf_\mu J(\mu,\nu)\le C_{1a}$ — usually with a duality gap, but a
useful **certified** lower bound.

---

## 2. The SDP (recommended: 1C after smoothing)

The Dirac $\delta(x+y-t)$ is not a polynomial. Replace it by a **Fejér/Dirichlet
kernel** $K_N(x+y-t)=\tfrac{1}{2N+1}\bigl(\tfrac{\sin((2N+1)\pi u)}{\sin(\pi u)}\bigr)^2$
or by the polynomial Christoffel–Darboux kernel
$$
K_N(u)\;=\;\sum_{j=0}^{N}p_j(u)^2,\qquad \{p_j\}\text{ orthonormal on }[-\tfrac12,\tfrac12],
$$
so that $K_N\to\delta$ weakly as $N\to\infty$. Because $K_N$ is a polynomial, the
coupled functional
$$
J_N(\mu,\nu)\;=\;\int\!\!\int\!\!\int K_N(x+y-t)\,d\mu(x)d\mu(y)d\nu(t)
\;=\;\sum_{j=0}^N\left(\int p_j(x+y)\,d\mu(x)d\mu(y)\right)\!\cdot\!\left(\int p_j(t)\,d\nu(t)\right)
$$
is **polynomial of total degree $N$** in the joint $(\mu,\mu,\nu)$ moments.

**Lasserre level-$k$ relaxation (dual / SOS side).** Find the largest
$\lambda$ and SOS certificates $\sigma_{\mu,0},\sigma_{\mu,1},\sigma_{\nu,0},\sigma_{\nu,1}$
and a polynomial $q(t)$ with $\deg\le 2k$ such that, for all $(x,y,t)\in\mathbb R^3$,
$$
\begin{aligned}
&K_N(x+y-t)-\lambda\\
&\quad=\sigma_{\mu,0}(x)+\sigma_{\mu,1}(x)(\tfrac1{16}-x^2)\\
&\qquad+\sigma_{\mu,0}'(y)+\sigma_{\mu,1}'(y)(\tfrac1{16}-y^2)\\
&\qquad+\sigma_{\nu,0}(t)+\sigma_{\nu,1}(t)(\tfrac14-t^2)\\
&\qquad+(\text{$x,y,t$-mixed SOS of degree }\le 2k)\\
&\qquad+q(t)(\int d\nu-1)+\cdots\;(\text{normalisation mults})
\end{aligned}
$$
with $\deg\sigma\le 2k-2$. The maximum such $\lambda$ is a rigorous lower bound
on $J_N(\mu,\nu)$ uniformly over admissible $(\mu,\nu)$. Recovering the
Lasserre lower bound on $C_{1a}$ then uses $K_N\to\delta$:
$$
C_{1a}\;\ge\;\lambda_{N,k}\;-\;\epsilon_N,\qquad \epsilon_N\to 0.
$$

**Primal / moment side.** Variables are joint moments
$y_{\alpha,\beta,\gamma}=\int x^\alpha y^\beta t^\gamma\,d\mu(x)d\mu(y)d\nu(t)$
with $\alpha,\beta,\gamma\ge 0$, $\alpha+\beta+\gamma\le 2k$, constrained so that
the $(x,y)$ marginal equals the tensor square of a single $\mu$-moment
sequence. **This coupling is a rank-1 constraint** and therefore **non-convex**.

This is the key obstruction: the two $\mu$ factors in $J_N$ make the primal
**quartic** in $\mu$-moments, so the natural Lasserre lift requires
$y_{\alpha+\alpha',\beta+\beta'}=m_{\alpha+\beta}m_{\alpha'+\beta'}$ type
bilinear identities, which is a **rank-one tensor factorisation** — precisely
what forces a hierarchy (each level only requires $y$ to be a valid joint
moment, relaxing rank-one).

---

## 3. Cleanest usable variant: quartic POP on $[-1/4,1/4]^2$

The hack that makes a single SDP work is to **square out** the tensor:
work with $F(x,y):=f(x)f(y)$ as a single bivariate density on
$[-1/4,1/4]^2$ with moments $M_{\alpha\beta}=\int x^\alpha y^\beta F$. The
marginal constraints $\int F(\cdot,y)\,dy=f(\cdot)$ and symmetry
$F(x,y)=F(y,x)$ are linear in $M_{\alpha\beta}$ provided we also keep the
$m_k$ of $f$. The objective becomes
$$
\lVert f*f\rVert_\infty\;=\;\sup_t\!\!\int_{x+y=t}\!F(x,y)\,ds
\;\ge\; \sup_t\!\sum_\ell K_N(x+y-t)\,M_{\alpha\beta}(\text{coeffs}).
$$
The **rank-one constraint** $F=f\otimes f$ is again only enforceable via SOS
hierarchy; dropping it gives a **convex** Lasserre SDP whose value is a lower
bound on $C_{1a}$:
$$
C_{1a}^{\text{Las}(k,N)}\;:=\;\min_{M\succeq 0}\ \sup_t\ \langle K_N(\cdot+\cdot-t),\,M\rangle
\quad\text{s.t.}\quad M\text{ is a valid deg-$2k$ moment matrix on }[-\tfrac14,\tfrac14]^2,\ \int M=1.
$$
The $\sup_t$ is handled by **one more Lasserre block** in the $t$-variable on
$[-\tfrac12,\tfrac12]$ (localising $\tfrac14-t^2$), producing a standard min–max
SDP of the form "$\min_y \max_z c(y,z)$" with $c$ bilinear — solvable by
saddle-point techniques or by dualising the inner sup.

---

## 4. Feasibility estimate

Let $k$ be the Lasserre level and $N$ the kernel degree. The moment matrix
$M_k(y)$ over the 3-variable cube has size
$$
s(k)=\binom{3+k}{k}\approx \tfrac{k^3}{6}.
$$
Localising matrices for the 3 box constraints add $O(k^3)$ rows each. The
primal SDP has:
- **PSD block size** $\approx\tfrac{k^3}{6}$, so vectorised $\approx k^6/72$ variables;
- **Equalities** $\binom{3+2k}{2k}\approx\tfrac{(2k)^3}{6}$;
- **Mosek runtime** $\sim (\text{svec})^{1.5}\approx k^9$.

With our existing pipeline (`lasserre/dual_sdp.py` already implements dual
Lasserre with clique sparsity up to $\approx 37\mathrm k \times 37\mathrm k$
Schur), realistic levels are
$$
k\le 8\ \Rightarrow\ s(k)\le 165,\ \text{svec}\le 1.4\cdot 10^4\quad\text{(trivial)}.
$$
$k=10$ yields svec $\approx 10^4$, easily fits the 192-core pod.
**Kernel degree** $N$ is the real bottleneck for **bound quality**: to
approximate the Dirac with ratio $\epsilon$, one needs $N\sim 1/\epsilon^2$ for
Fejér kernels. Matching $1.28$ to $3$-digit accuracy requires $N\approx
10^3$–$10^4$, which **blows** the SDP.

**Convergence rate.** On the cube $[-1/4,1/4]^2\times[-1/2,1/2]$, the best known
Lasserre rate is $O(1/k^2)$ (de Klerk–Laurent; Revisiting the convergence
rate, arXiv:2505.00544). With $\lambda_{N,k}\to C_{1a}$ monotone in $k$ and $N$,
one expects
$$
C_{1a}-\lambda_{N,k}\;\asymp\;\frac{1}{k^2}+\frac{1}{N},
$$
so to close the gap $1.5029-1.2802=0.2227$ to $10^{-3}$ we would need $k\ge 30$
and $N\ge 10^3$ — **well out of reach**. To beat $1.2802$ by $10^{-3}$
(what we actually need) we need the Lasserre level where the **floor** of
the relaxation crosses $1.2803$: empirically, **$k=4,N=20$** should already
produce $\approx 1.20$ (comparable to $\mathrm{val}(d=10)$ in `core.py`:
`val_d_known[10]=1.241`).

---

## 5. Concrete recommendation

**Pick variant 1C in the form of §3.** Implement a Lasserre hierarchy on the
3-variable cube $(x,y,t)\in[-\tfrac14,\tfrac14]^2\times[-\tfrac12,\tfrac12]$ with
the bivariate measure $F(x,y)$ and the test measure $\nu(t)$, using the
Christoffel–Darboux kernel
$K_N(u)=\sum_{j=0}^N p_j(u)^2$ in Legendre basis rescaled to $[-1,1]$.

### Gray-box equations

1. **Variables.** $y_{\alpha\beta\gamma}$ with $|\alpha|+|\beta|+|\gamma|\le 2k$,
   $\alpha,\beta\le k$, $\gamma\le k$. Normalisation $y_{000}=1$.
2. **Symmetry.** $y_{\alpha\beta\gamma}=y_{\beta\alpha\gamma}$ (exchange $x\leftrightarrow y$).
3. **Moment matrix PSD:** $M_k(y)\succeq 0$, size $\binom{3+k}{k}$.
4. **Localisers:**
   $(\tfrac1{16}-x^2)\cdot M_{k-1}(y)\succeq 0$,
   $(\tfrac1{16}-y^2)\cdot M_{k-1}(y)\succeq 0$,
   $(\tfrac14-t^2)\cdot M_{k-1}(y)\succeq 0$.
5. **Marginal consistency:** $\int F(x,y)\,dy$ must have the $f$-moment structure,
   enforced via auxiliary moments $m_k$ and the linear identities
   $m_k=\sum_{\beta\le k} y_{(k-\beta),\beta,0}$ (monomial marginals).
6. **Outer objective:**
   $$
   \text{minimise}\ \ \sum_{j=0}^N\Bigl(\sum_{\alpha+\beta=\gamma_j}c_{\alpha\beta}^{(j)}\,y_{\alpha\beta 0}\Bigr)\!\cdot\!\Bigl(\sum_\gamma d_\gamma^{(j)}\,y_{00\gamma}\Bigr).
   $$
   The product is replaced by a **joint moment** of the tensor measure — already
   in $y_{\alpha\beta\gamma}$. The objective is thus **linear** in $y$.
7. **Solve** with MOSEK dual form (reuse `lasserre/dual_sdp.py` infra).

### Why this should work

- **Rigorous.** Any feasible $y$ gives a valid lower bound $\ge\lambda_{N,k}$.
- **Closed under symmetry.** The exchange $x\leftrightarrow y$ and the
  $\mathbb Z/2$ reflection $x\mapsto -x$, $t\mapsto -t$ block-diagonalise the
  moment matrix (reuse `lasserre/z2_blockdiag.py`).
- **Scales polynomially** in $k$, $N$.
- **Not subject to discretisation error** in the primal — the measure $\mu$ is
  continuous, unlike cascade.

### Why this probably **will not beat 1.2802** at feasible level

Three-cube Lasserre at $k=8$, $N=40$ fits, but `val_d_known[16]=1.319` already
beats $1.28$ at $d=16$; the SDP needs to match the information content of a
$d\sim 16$ cascade to be competitive. The current approach (`lasserre/` clique
sparsity, $d=8\ldots 16$, $k=2,3$) already does this via a completely different
(discrete) lift and gets to $1.284$ at $d=14$. The continuum-SDP of §5 at
level $k=8$ carries roughly **the same information** as a $d\approx 2k=16$
discrete cascade. **So the continuum-SDP is a drop-in alternative, not a
leapfrog.**

### When the continuum-SDP *is* preferable

1. Producing a **final rigorous certificate** once cascade finds the optimal
   $f$: the SOS dual gives a machine-checkable proof of $C_{1a}\ge\lambda$.
2. **Asymptotic analysis** ($d\to\infty$): SDP directly treats the continuous
   problem; cascade gets exponential blow-up.
3. When combined with the Fourier / Paley–Wiener structure of §1B it produces
   **hybrid certificates** (LP-SDP) that are sharper than either alone.

---

## 6. Is there direct prior work?

**No direct SDP for this problem exists in the literature**, to my reading
after the searches below. The closest references are:

- Matolcsi–Vinuesa (arXiv:0907.1379, 2010): a **quadratic program**
  (not SDP) in step-function heights on a fine discretisation, with Fourier
  dual certificate. No Lasserre.
- White (arXiv:2210.16437, 2022) and Boyer–Li (arXiv:2506.16750, 2025):
  **simulated annealing + Adam** over step-function heights; numerical only.
- Jaech–Joseph (arXiv:2508.02803, 2025, "Further Improvements"):
  Adam with decreasing Gaussian-noise exploration, 559-interval step function,
  $L^2$ ratio bound $0.94136$ — **no SDP, no certificate**.
- Lasserre (2008 — *A semidefinite programming approach to the generalised
  problem of moments*, Math. Prog. 112, arXiv counterpart): the general
  template we follow, but with **polynomial** objective. Our contribution is
  the kernel-smoothing trick (§2) to polynomialise the Dirac.
- de Klerk–Laurent (*Sum-of-squares hierarchies convergence rates*, 2024,
  arXiv:2408.04417): the $O(1/k^2)$ rate we quote.
- Nie, *Moment and Polynomial Optimization*, SIAM 2022: general reference for
  semi-infinite/bilevel polynomial POPs that we implicitly use.

So the design above is **novel** in applying Lasserre to $C_{1a}$, and cleanly
packaged: kernel-smoothed polynomial objective on a 3-cube, standard Lasserre
machinery, reuses existing MOSEK dual infra.

---

## 7. Action items

1. **Prototype** the 3-cube Lasserre at $k=4$, $N=10$ (hours).
2. **Benchmark** $\lambda_{4,10}$ against `val_d_known[8]=1.205` — this validates
   the kernel-smoothing trick.
3. **Scale** to $k=8$, $N=40$ on the pod using `dual_sdp.py` infra; expected
   $\lambda\approx 1.28\pm 0.02$.
4. **Hybrid.** Add the Fourier Cauchy–Schwarz bound $|\hat f|^2\ge
   \cos^2(\pi\xi/2)\mathbf 1_{|\xi|\le 1}$ as an extra linear constraint on the
   $(x,y)$-moments (via its Taylor expansion).
5. **Certified arithmetic.** Reuse `certified_lasserre/` for exact rational
   bounds on the final $\lambda_{N,k}$.

---

### References

- Lasserre (2001), *Global optimisation with polynomials and the problem of moments*, SIAM J. Optim.
- Lasserre (2008), *A semidefinite programming approach to the generalised problem of moments*, Math. Prog.
- de Klerk & Laurent (2024), *Sum-of-squares hierarchies convergence rates*, arXiv:2408.04417.
- Henrion, Lasserre, Löfberg, *GloptiPoly 3*, 2009.
- Matolcsi & Vinuesa (2010), arXiv:0907.1379.
- Cloninger & Steinerberger (2017), arXiv:1403.7988.
- White (2022), arXiv:2210.16437.
- Boyer & Li (2025), arXiv:2506.16750.
- Jaech & Joseph (2025), arXiv:2508.02803.
- Nie (2022), *Moment and Polynomial Optimization*, SIAM.
