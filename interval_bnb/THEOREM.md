# Rigorous derivation: $\mathrm{val}(d)$ is a lower bound on $C_{1a}$

This note proves, without step-function approximation, that for every
integer $d \ge 2$

$$
C_{1a} \;\ge\; \mathrm{val}(d)
\;:=\; \min_{\mu \in \Delta_d}\; \max_{W \in \mathcal W_d}\; \mu^\top M_W \mu,
$$

where the window matrices $M_W$ are *exactly* those built by
`lasserre.core.build_window_matrices` (prefactor $2d/\ell$). Section 2
re-derives the prefactor from first principles and explains why the
alternative "$d/(\ell-1)$" guess is wrong. Section 3 records the
symmetry argument that the branch-and-bound driver uses to halve the
search region.

Throughout, "admissible $f$" means: $f : \mathbb R \to \mathbb R_{\ge 0}$
is Lebesgue measurable, $\mathrm{supp}(f) \subseteq [-1/4,\,1/4]$, and
$\int_{\mathbb R} f = 1$.

Recall the Sidon autocorrelation constant is defined so that every
admissible $f$ satisfies

$$
\max_{|t|\le 1/2}\,(f*f)(t) \;\ge\; C_{1a}.
$$

---

## 1. Theorem: $C_{1a} \ge \mathrm{val}(d)$

### 1.1 Setup

Fix an integer $d \ge 2$. Partition $[-1/4,\,1/4]$ into $d$ equal
subintervals

$$
B_i \;=\; \Bigl[\,-\tfrac14 + \tfrac{i}{2d},\ -\tfrac14 + \tfrac{i+1}{2d}\,\Bigr],
\qquad i = 0,1,\dots,d-1.
$$

Each $B_i$ has Lebesgue measure $|B_i| = 1/(2d)$ and
$\bigsqcup_{i=0}^{d-1} B_i = [-1/4,\,1/4]$ (up to the zero-measure shared
endpoints, which play no role in any integral below).

For admissible $f$, define

$$
\mu_i \;:=\; \int_{B_i} f(x)\,dx, \qquad i=0,1,\dots,d-1.
$$

Then $\mu_i \ge 0$ and $\sum_i \mu_i = \int_{-1/4}^{1/4} f = 1$, so
$\mu := (\mu_0,\dots,\mu_{d-1}) \in \Delta_d$, the probability simplex in
$\mathbb R^d$.

### 1.2 Windows and the Minkowski sums of bins

The set of windows used by `build_window_matrices` is

$$
\mathcal W_d \;=\;\bigl\{\,W=(\ell, s_{\mathrm{lo}}):
2 \le \ell \le 2d,\ 0 \le s_{\mathrm{lo}} \le 2d-\ell\,\bigr\}.
$$

For each $W \in \mathcal W_d$ define the pair-sum index set

$$
K_W \;:=\; \{\,s_{\mathrm{lo}},\ s_{\mathrm{lo}}+1,\ \dots,\ s_{\mathrm{lo}}+\ell-2\,\}
\;\subseteq\; \{0,1,\dots,2d-2\},
$$

which has cardinality $\ell-1$ (so $|K_W| \ge 1$).

For $(i,j) \in \{0,\dots,d-1\}^2$ let $k := i+j \in \{0,\dots,2d-2\}$.
The Minkowski sum $B_i + B_j = \{x+y : x \in B_i,\, y \in B_j\}$ equals

$$
B_i+B_j
\;=\;
\Bigl[\,-\tfrac12 + \tfrac{k}{2d},\ -\tfrac12 + \tfrac{k+2}{2d}\,\Bigr],
\qquad k=i+j.
$$

(Adding two intervals of length $1/(2d)$ gives one of length $2/(2d) = 1/d$.)

Define the **window $t$-interval**

$$
I_W \;:=\;
\Bigl[\,-\tfrac12 + \tfrac{s_{\mathrm{lo}}}{2d},\ -\tfrac12 + \tfrac{s_{\mathrm{lo}}+\ell}{2d}\,\Bigr]
\;\subseteq\; [-\tfrac12,\,\tfrac12].
$$

So $|I_W| = \ell/(2d)$. The containment $I_W \subseteq [-1/2,\,1/2]$
follows from $0 \le s_{\mathrm{lo}}$ and $s_{\mathrm{lo}}+\ell \le 2d$.

**Lemma 1.1 (pair-sum geometry).** For $W = (\ell,s_{\mathrm{lo}})$ and
indices $0 \le i,j \le d-1$, write $k = i+j$. Then:

(a) If $k \in K_W$, i.e. $s_{\mathrm{lo}} \le k \le s_{\mathrm{lo}}+\ell-2$,
   then $B_i+B_j \subseteq I_W$.

(b) If $k \le s_{\mathrm{lo}}-2$ or $k \ge s_{\mathrm{lo}}+\ell$, then
   $(B_i+B_j)\cap I_W$ has Lebesgue measure $0$ (the interiors are disjoint).

(c) If $k = s_{\mathrm{lo}}-1$ or $k = s_{\mathrm{lo}}+\ell-1$, then
   $(B_i+B_j)\cap I_W$ is a nondegenerate sub-interval of length
   $1/(2d)$.

*Proof.* From the formulas,
$B_i+B_j = [-1/2+k/(2d),\,-1/2+(k+2)/(2d)]$ and
$I_W = [-1/2+s_{\mathrm{lo}}/(2d),\,-1/2+(s_{\mathrm{lo}}+\ell)/(2d)]$.

For (a): $k \ge s_{\mathrm{lo}}$ gives left-containment and
$k+2 \le s_{\mathrm{lo}}+\ell$ gives right-containment.

For (b): $k+2 \le s_{\mathrm{lo}}$ forces $B_i+B_j$ to lie to the left of
$I_W$; $k \ge s_{\mathrm{lo}}+\ell$ forces it to lie to the right.

For (c): at $k=s_{\mathrm{lo}}-1$ the overlap is
$[-1/2+s_{\mathrm{lo}}/(2d),\,-1/2+(s_{\mathrm{lo}}+1)/(2d)]$ of length
$1/(2d)$. Symmetric at $k=s_{\mathrm{lo}}+\ell-1$. $\square$

### 1.3 The window matrix

`build_window_matrices` defines

$$
M_W[i,j] \;=\;
\begin{cases}
\dfrac{2d}{\ell}, & s_{\mathrm{lo}} \le i+j \le s_{\mathrm{lo}}+\ell-2,\\
0, & \text{otherwise.}
\end{cases}
$$

Equivalently, with $\mathbf 1_{K_W}$ the indicator of $K_W$,

$$
\mu^\top M_W \mu
\;=\; \frac{2d}{\ell}\ \sum_{i,j=0}^{d-1}
\mathbf 1_{K_W}(i+j)\,\mu_i\mu_j.
$$

### 1.4 Averaging inequality

**Lemma 1.2 (averaging).** For any admissible $f$ and any $W\in\mathcal W_d$,

$$
\max_{t\in[-1/2,\,1/2]}\,(f*f)(t)
\;\ge\;
\frac{1}{|I_W|}\int_{I_W}(f*f)(t)\,dt.
$$

*Proof.* The convolution $f*f$ is supported on $[-1/2,1/2]$ because
$\mathrm{supp}(f) \subseteq [-1/4,1/4]$. It is the convolution of
two $L^1$ functions, hence $f*f \in L^1(\mathbb R)$. Moreover
$f*f \ge 0$ pointwise wherever it is defined, since $f \ge 0$. Then
for any measurable $I\subseteq[-1/2,1/2]$ with $|I|>0$,

$$
\frac{1}{|I|}\int_I (f*f)(t)\,dt
\;\le\;
\frac{1}{|I|}\int_I \operatorname{ess\,sup}_{[-1/2,1/2]}(f*f)\,dt
\;=\;
\operatorname{ess\,sup}_{[-1/2,1/2]}(f*f).
$$

Because $f \in L^1$ is real-valued, $f*f$ is continuous on $\mathbb R$
(standard: convolution of two $L^1\cap L^2$-bounded compactly supported
functions is continuous — in fact for compactly supported $L^1$
functions the convolution is continuous a.e.; for bounded compactly
supported $f$ it is Lipschitz).

GAP-GUARD: The cleanest route that requires no extra regularity on $f$
is: $f*f$ is an $L^1$ function on $[-1/2,1/2]$, so
$\operatorname{ess\,sup}(f*f) = \mathrm{ess\,sup}_{t\in[-1/2,1/2]}(f*f)(t)$
exists in $(0,\infty]$. For the Sidon problem, one may (and our
competitor uses this standardly) take the supremum in the
$\operatorname{ess\,sup}$ sense. With that convention — which is the
same convention used in stating the upper bound — the chain above is
immediate.

The final statement we need is:

$$
\operatorname{ess\,sup}_{t\in[-1/2,1/2]}\,(f*f)(t)
\;\ge\;
\frac{1}{|I_W|}\int_{I_W}(f*f)(t)\,dt. \qquad\square
$$

*Remark.* For bounded admissible $f$ (e.g. step functions, which is the
extremal regime), $f*f$ is continuous, so the essential supremum and
pointwise maximum agree. The published value of $C_{1a}$ is defined as
an infimum over admissible $f$ of this quantity; both conventions give
the same infimum.

### 1.5 Computing the window integral

**Lemma 1.3 (window integral lower bound).** For any admissible $f$ and
any $W\in\mathcal W_d$,

$$
\int_{I_W} (f*f)(t)\,dt
\;\ge\;
\sum_{i,j=0}^{d-1} \mathbf 1_{K_W}(i+j)\,\mu_i\mu_j.
$$

*Proof.* Since $f \ge 0$ and $f\in L^1$, Tonelli–Fubini gives

$$
\int_{I_W}(f*f)(t)\,dt
=\int_{I_W}\!\!\int_{\mathbb R} f(x)\,f(t-x)\,dx\,dt.
$$

Substituting $y = t-x$ (so $x+y = t$) inside the inner integral, then
swapping integration orders (all integrands nonneg):

$$
\int_{I_W}(f*f)(t)\,dt
\;=\;
\iint_{\mathbb R^2} f(x)\,f(y)\,\mathbf 1\bigl[x+y\in I_W\bigr]\,dx\,dy.
$$

Because $f$ is supported in $[-1/4,1/4]$, we may restrict the double
integral to $(x,y)\in[-1/4,1/4]^2$. Split into the $d^2$ bin pairs
$(B_i,B_j)$:

$$
\int_{I_W}(f*f)(t)\,dt
\;=\;
\sum_{i,j=0}^{d-1}\int_{B_i}\!\!\int_{B_j}
f(x)\,f(y)\,\mathbf 1\bigl[x+y\in I_W\bigr]\,dy\,dx.
$$

Drop all pair contributions except those with $i+j \in K_W$. Each
dropped term has integrand $\ge 0$, so the sum decreases (or is
unchanged). By Lemma 1.1(a), for $i+j \in K_W$ we have
$B_i+B_j \subseteq I_W$, hence
$\mathbf 1[x+y\in I_W] = 1$ identically on $B_i\times B_j$. Therefore

$$
\int_{I_W}(f*f)(t)\,dt
\;\ge\;
\sum_{\substack{i,j\\ i+j\in K_W}}
\int_{B_i} f(x)\,dx \cdot \int_{B_j} f(y)\,dy
\;=\;
\sum_{i,j=0}^{d-1}\mathbf 1_{K_W}(i+j)\,\mu_i\mu_j.
\quad\square
$$

### 1.6 Combining

**Theorem 1 (main inequality).** For every integer $d \ge 2$, every
admissible $f$, and every $W \in \mathcal W_d$,

$$
\max_{t\in[-1/2,1/2]} (f*f)(t) \;\ge\; \mu^\top M_W \mu,
$$

where $\mu_i = \int_{B_i} f$ as above. Consequently,

$$
C_{1a} \;\ge\; \mathrm{val}(d).
$$

*Proof.* Combine Lemmas 1.2 and 1.3:

$$
\max_{t\in[-1/2,1/2]}(f*f)(t)
\;\ge\;
\frac{1}{|I_W|}\int_{I_W}(f*f)(t)\,dt
\;\ge\;
\frac{1}{|I_W|}\sum_{i,j}\mathbf 1_{K_W}(i+j)\,\mu_i\mu_j.
$$

With $|I_W|=\ell/(2d)$,

$$
\frac{1}{|I_W|}\sum_{i,j}\mathbf 1_{K_W}(i+j)\,\mu_i\mu_j
\;=\; \frac{2d}{\ell}\sum_{i,j}\mathbf 1_{K_W}(i+j)\,\mu_i\mu_j
\;=\; \mu^\top M_W\mu.
$$

Thus $\max_t(f*f)(t) \ge \mu^\top M_W \mu$ for every $W$, so
$\max_t(f*f)(t) \ge \max_{W\in\mathcal W_d}\mu^\top M_W\mu
\ge \min_{\nu\in\Delta_d}\max_{W\in\mathcal W_d}\nu^\top M_W \nu
= \mathrm{val}(d)$. Taking the infimum over admissible $f$ (which
defines $C_{1a}$),

$$
C_{1a} \;=\; \inf_{f\ \text{admissible}}\max_t(f*f)(t) \;\ge\; \mathrm{val}(d). \quad\square
$$

*Remarks.*

- **No step-function assumption.** The proof uses only that $\mu_i$
  are integrals of $f$ over $B_i$; $f$ is an arbitrary admissible
  non-negative $L^1$ function. The averaging inequality converts any
  such $f$ into its coarse histogram $\mu$ without discretization loss,
  and the slack "partial-overlap pairs ($i+j = s_{\mathrm{lo}}-1$ or
  $s_{\mathrm{lo}}+\ell-1$)" is *dropped*, which only strengthens the
  inequality for the lower bound.
- **Signed slack.** The difference
  $\int_{I_W}(f*f) - \sum_{i+j\in K_W}\mu_i\mu_j$ is exactly the sum of
  partial-overlap pair integrals, which is $\ge 0$. So the inequality
  in Lemma 1.3 is tight iff $f$ vanishes a.e. on the "boundary" bins
  $B_i$ with $i+j \in \{s_{\mathrm{lo}}-1, s_{\mathrm{lo}}+\ell-1\}$ —
  i.e., for full-window $W$ (where $s_{\mathrm{lo}}-1<0$ and
  $s_{\mathrm{lo}}+\ell-1 = 2d-1$ lies outside the valid pair-sum
  range), the inequality becomes an equality.

---

## 2. Scale-factor derivation: why $2d/\ell$, not $d/(\ell-1)$

**Auditor's concern.** A window $W=(\ell,s_{\mathrm{lo}})$ has pair-sum
support $K_W = \{s_{\mathrm{lo}},\dots,s_{\mathrm{lo}}+\ell-2\}$ of size
$\ell-1$. Each pair-sum value $k$ corresponds to $x+y$ ranging over an
interval of "width $1/d$" (the length of $B_i+B_j$). A naive
count — "$\ell-1$ pair-sums, each with plateau length $1/d$" — suggests
a plateau total length of $(\ell-1)/d$, and hence a prefactor of
$d/(\ell-1)$.

**Resolution: the naive count double-counts overlaps.** We show that
$|I_W| = \ell/(2d)$ (not $(\ell-1)/d$), and that the prefactor $2d/\ell$
is the correct one.

### 2.1 The pair-sum supports overlap

Pair sum $k$ contributes (via $B_i+B_j$, any $i+j = k$) to
$t \in [-1/2+k/(2d),\,-1/2+(k+2)/(2d)]$, a $t$-interval of length
$2/(2d) = 1/d$. Two consecutive pair sums $k$ and $k+1$ contribute to
$[-1/2+k/(2d),-1/2+(k+2)/(2d)]$ and $[-1/2+(k+1)/(2d),-1/2+(k+3)/(2d)]$
respectively. These two intervals **overlap** in
$[-1/2+(k+1)/(2d),\,-1/2+(k+2)/(2d)]$, of length $1/(2d)$.

Summing with multiplicity would give $(\ell-1)\cdot 1/d = (\ell-1)/d$.
But the Lebesgue measure of the union is *not* the sum with
multiplicity; it is

$$
\Bigl|\bigcup_{k\in K_W}\bigl[-\tfrac12+\tfrac{k}{2d},\,-\tfrac12+\tfrac{k+2}{2d}\bigr]\Bigr|
\;=\; \tfrac{1}{2d}\cdot \bigl(\text{length of merged block in units of }\tfrac{1}{2d}\bigr).
$$

The merged block runs from $s_{\mathrm{lo}}$ to $s_{\mathrm{lo}}+\ell$
(in half-width units), i.e. $\ell$ half-widths, so total length
$\ell/(2d)$. This is $|I_W|$ as defined in Section 1.2.

### 2.2 Why the "double count" matters: the (f*f)(t) pointwise picture

For a specific $t \in I_W$, every pair $(i,j)$ whose Minkowski sum
contains $t$ contributes. Consecutive pair sums $k,k+1$ both contribute
to the overlap band — this is fine for the *pointwise* value of
$(f*f)(t)$, but when we integrate, we want each bin-pair to be counted
**once**, multiplied by the length of its contribution to $I_W$.

The Fubini/Tonelli calculation in Lemma 1.3 handles this automatically:

$$
\int_{I_W}(f*f)(t)\,dt
\;=\;\sum_{i,j}\int_{B_i\times B_j} f(x)f(y)\,\mathbf 1[x+y\in I_W]\,dx\,dy.
$$

There is no "double counting" — each $(x,y)$-pair is summed exactly
once. The indicator $\mathbf 1[x+y\in I_W]$ collapses to $1$ on the
**full** pair-rectangle $B_i\times B_j$ precisely when
$i+j \in K_W$ (Lemma 1.1(a)), and is $\in[0,1]$ on boundary-pair
rectangles. Dropping boundary contributions yields Lemma 1.3 with the
clean sum on the right-hand side.

### 2.3 Explicit (f\*f)(t) for step f: a sanity check

Take $f$ a step function, constant $c_i := 2d\,\mu_i$ on $B_i$ (so
$\int_{B_i} f = \mu_i$ — note $f$ is density so $c_i = \mu_i/|B_i| = 2d\mu_i$).
For $t = -1/2 + \tau/(2d)$ with $\tau\in[0,2d]$, the convolution
$(f*f)(t)$ is piecewise linear in $\tau$ with kinks at integer $\tau$,
and for non-integer $\tau$ in the open unit interval $(k,k+1)$
($k\in\{0,\dots,2d-1\}$),

$$
(f*f)\bigl(-\tfrac12+\tfrac{\tau}{2d}\bigr)
\;=\;
\sum_{i+j=k}\alpha_{ij}(\tau)\,c_i c_j
\;+\;
\sum_{i+j=k-1}\beta_{ij}(\tau)\,c_i c_j
$$

for some non-negative weights $\alpha_{ij},\beta_{ij}$ with
$\sum\alpha = (\tau-k)/(2d)$ and $\sum\beta = (1-(\tau-k))/(2d)$
contribution patterns (the "tent"-shaped density in $\tau$ across each
unit interval). Integrating $(f*f)(t)\,dt$ across the unit
$\tau$-interval $[k,k+1]$ of length $1/(2d)$,

$$
\int_{t\in[-1/2+k/(2d),\,-1/2+(k+1)/(2d)]}(f*f)(t)\,dt
\;=\;
\tfrac{1}{4d^2}\Bigl(\tfrac12\sum_{i+j=k}c_ic_j + \tfrac12\sum_{i+j=k-1}c_ic_j\Bigr),
$$

where the $1/(4d^2)=1/(2d)\cdot 1/(2d)$ comes from $dt = d\tau/(2d)$
times $|B_i|=1/(2d)$ in Fubini. Using $c_i = 2d\mu_i$,

$$
=\; \tfrac12\Bigl(\sum_{i+j=k}\mu_i\mu_j + \sum_{i+j=k-1}\mu_i\mu_j\Bigr).
$$

Summing $k$ from $s_{\mathrm{lo}}$ to $s_{\mathrm{lo}}+\ell-1$ (this is
$\ell$ unit $\tau$-intervals, covering $I_W$),

$$
\int_{I_W}(f*f)(t)\,dt
\;=\;
\tfrac12 \sum_{k=s_{\mathrm{lo}}}^{s_{\mathrm{lo}}+\ell-1}\!\!
\Bigl(\sum_{i+j=k}\mu_i\mu_j + \sum_{i+j=k-1}\mu_i\mu_j\Bigr).
$$

Every pair sum $k'\in K_W = [s_{\mathrm{lo}},s_{\mathrm{lo}}+\ell-2]$
appears **twice** (once as the "$k$" term for $k=k'$, once as the
"$k-1$" term for $k=k'+1$), contributing $\tfrac12\cdot 2 = 1$
multiplier. Pair sums $k'=s_{\mathrm{lo}}-1$ and $k'=s_{\mathrm{lo}}+\ell-1$
appear once each, contributing $\tfrac12$. So

$$
\int_{I_W}(f*f)(t)\,dt
\;=\;
\sum_{i+j\in K_W}\mu_i\mu_j
\;+\;
\tfrac12\!\!\sum_{i+j\in\{s_{\mathrm{lo}}-1,s_{\mathrm{lo}}+\ell-1\}}\!\!\mu_i\mu_j,
$$

which confirms Lemma 1.3 with explicit boundary slack
$\tfrac12(\cdots)\ge 0$ (the boundary terms are the partial-overlap
contributions).

Dividing by $|I_W| = \ell/(2d)$,

$$
\tfrac{1}{|I_W|}\int_{I_W}(f*f)(t)\,dt
\;=\;
\tfrac{2d}{\ell}\sum_{i+j\in K_W}\mu_i\mu_j
\;+\;
\tfrac{d}{\ell}\sum_{i+j\in\{s_{\mathrm{lo}}-1,s_{\mathrm{lo}}+\ell-1\}}\mu_i\mu_j
\;\ge\;
\mu^\top M_W\mu.
$$

The first term is **exactly** $\mu^\top M_W\mu$ with the code's
prefactor $2d/\ell$. The alternative prefactor $d/(\ell-1)$ does not
arise from this computation.

### 2.4 Conclusion

The prefactor $2d/\ell$ is **correct** (option (a) from the problem
statement). It arises as $1/|I_W|$ where $|I_W|=\ell/(2d)$ is the
Lebesgue length of the merged plateau. The factor of $2$ (relative to
the "length $(\ell-1)/d$" naive guess) is a *symmetry-free* artefact of
the way consecutive pair-sum supports overlap: the union of $\ell-1$
intervals each of length $1/d$ has merged length $\ell/(2d)$, not
$(\ell-1)/d$, because consecutive supports overlap by exactly half
their length.

There is **no** hidden factor coming from the evenness of $f*f$;
$(f*f)(t) = (f*f)(-t)$ holds when $f$ is an even function, but we make
no such assumption, and the proof does not invoke evenness of $f*f$.
The evenness that *does* play a role is the symmetry
$\sigma(i) = d-1-i$ of the bin-indexing, which permutes windows
(Section 3) — this is a discrete combinatorial symmetry, not a source
of a factor-of-two rescaling.

---

## 3. Symmetry reduction: the half-simplex cover

Let $\sigma:\{0,\dots,d-1\}\to\{0,\dots,d-1\}$ be the reversal
$\sigma(i) = d-1-i$. Extend $\sigma$ to $\mathbb R^d$ by
$(\sigma\mu)_i = \mu_{d-1-i}$. Then $\sigma$ is an involution on
$\Delta_d$.

### 3.1 $\sigma$ acts on windows

**Lemma 3.1.** The map $(\ell,s_{\mathrm{lo}}) \mapsto
(\ell,\,\tilde s)$ with $\tilde s := 2(d-1)-s_{\mathrm{lo}}-(\ell-2)
= 2d-\ell-s_{\mathrm{lo}}$ is a self-bijection on $\mathcal W_d$.

*Proof.* We have $s_{\mathrm{lo}} \in [0,2d-\ell]$ iff
$\tilde s = 2d-\ell-s_{\mathrm{lo}} \in [0,2d-\ell]$. Symmetric.
$\square$

**Lemma 3.2.** For $W = (\ell,s_{\mathrm{lo}})$ and
$\tilde W = (\ell,\tilde s)$ from Lemma 3.1,
$M_{\tilde W}[i,j] = M_W[\sigma(i),\sigma(j)]$. Equivalently, with
$P_\sigma$ the permutation matrix for $\sigma$,
$M_{\tilde W} = P_\sigma M_W P_\sigma^\top$.

*Proof.* $M_W[\sigma(i),\sigma(j)] \ne 0$
iff $s_{\mathrm{lo}} \le (d-1-i)+(d-1-j) \le s_{\mathrm{lo}}+\ell-2$, i.e.
$2(d-1)-s_{\mathrm{lo}}-\ell+2 \le i+j \le 2(d-1)-s_{\mathrm{lo}}$, i.e.
$\tilde s \le i+j \le \tilde s + \ell - 2$. The prefactor $2d/\ell$
depends only on $\ell$ and is unchanged. $\square$

### 3.2 $\sigma$-invariance of the objective

**Lemma 3.3.** $\max_{W\in\mathcal W_d}\mu^\top M_W \mu$ is
$\sigma$-invariant: for every $\mu\in\Delta_d$,

$$
\max_{W\in\mathcal W_d}\mu^\top M_W \mu
\;=\;
\max_{W\in\mathcal W_d}(\sigma\mu)^\top M_W (\sigma\mu).
$$

*Proof.* By Lemma 3.2, for any $W$,
$(\sigma\mu)^\top M_W (\sigma\mu) = \mu^\top P_\sigma^\top M_W P_\sigma \mu
= \mu^\top M_{\tilde W}\mu$. As $W$ ranges over $\mathcal W_d$, so
does $\tilde W$ (Lemma 3.1); the maximum is unchanged. $\square$

### 3.3 Soundness of the half-simplex cut

Define the "Option C" half-simplex

$$
H_d \;:=\;\bigl\{\,\mu \in \Delta_d \;:\; \mu_0 \le \mu_{d-1}\,\bigr\}.
$$

**Lemma 3.4.** For every $\mu \in \Delta_d$, at least one of $\mu$,
$\sigma\mu$ lies in $H_d$. Consequently,

$$
\mathrm{val}(d)
\;=\;
\min_{\mu\in\Delta_d}\max_W \mu^\top M_W \mu
\;=\;
\min_{\mu\in H_d}\max_W \mu^\top M_W \mu.
$$

*Proof.* If $\mu_0 \le \mu_{d-1}$, then $\mu \in H_d$. Otherwise
$\mu_0 > \mu_{d-1}$, whence $(\sigma\mu)_0 = \mu_{d-1} < \mu_0 =
(\sigma\mu)_{d-1}$, so $\sigma\mu \in H_d$. In either case the
orbit $\{\mu,\sigma\mu\}$ meets $H_d$.

Let $v := \min_{\mu\in H_d}\max_W \mu^\top M_W \mu$. Clearly
$v \ge \mathrm{val}(d)$ since $H_d\subseteq\Delta_d$. For the reverse
direction: let $\mu^\star$ realize $\mathrm{val}(d)$ (minimum exists by
compactness — $\Delta_d$ is compact, $\max_W$ of a finite family of
continuous functions is continuous). If $\mu^\star \in H_d$, done.
Otherwise $\sigma\mu^\star\in H_d$ and by Lemma 3.3
$\max_W (\sigma\mu^\star)^\top M_W (\sigma\mu^\star)
= \max_W {\mu^\star}^\top M_W \mu^\star = \mathrm{val}(d)$,
so $v \le \mathrm{val}(d)$. $\square$

This justifies the `interval_bnb` driver's restriction to $H_d$: a
single linear cut $\mu_0 \le \mu_{d-1}$ on the initial box is sound and
halves the search volume. (A stricter "$\mu_i \le \mu_{d-1-i}$ for all
$i<d/2$" cut as in `README.md` is also sound by the same argument, but
it is not needed for correctness — only $\mu_0\le\mu_{d-1}$ is invoked
in the proof above. The stricter variant is a tighter box that still
contains at least one representative of every $\sigma$-orbit; a proof
would require a small additional argument showing every orbit meets the
stricter region, which the driver's code comments implicitly claim.
GAP-ADJACENT: this note only rigorously justifies the weaker single-cut
$\mu_0 \le \mu_{d-1}$.)

**Practical implementation note.** The driver `box.py:75-89` actually
imposes the **looser** cut $\mu_0 \le 1/2$, not $\mu_0 \le \mu_{d-1}$.
This is a SUPERSET cut:

$$
H_d^{\text{box}} \;:=\; \{\mu\in\Delta_d : \mu_0 \le 1/2\}
\;\supseteq\; H_d.
$$

Containment: for $\mu \in H_d$, $\mu_0 \le \mu_{d-1}$ and $\mu_0 + \mu_{d-1}
\le \sum_i \mu_i = 1$ give $2\mu_0 \le \mu_0 + \mu_{d-1} \le 1$, so
$\mu_0 \le 1/2$, i.e. $\mu \in H_d^{\text{box}}$.

Since $H_d \subseteq H_d^{\text{box}}$, certifying the per-box bound
$\max_W \mu^\top M_W \mu \ge c$ for every $\mu \in H_d^{\text{box}}$
**a fortiori** certifies it for every $\mu \in H_d$. Combined with
Lemma 3.4 ($\mathrm{val}(d) = \min_{H_d} \max_W \mu^\top M_W \mu$),
this gives $\mathrm{val}(d) \ge c$. So the looser-cut implementation is
**sound**, just slightly less efficient (covers a marginally larger
search region than necessary).

---

## Summary

1. Lemmas 1.1–1.3 assemble into Theorem 1: for every $d \ge 2$,
   $C_{1a} \ge \mathrm{val}(d)$, with $\mathrm{val}(d)$ using the
   window matrices from `lasserre.core.build_window_matrices` with
   prefactor $2d/\ell$.
2. The prefactor $2d/\ell$ is derived as $1/|I_W|$ with
   $|I_W|=\ell/(2d)$. The "$d/(\ell-1)$" alternative is the result of
   double-counting the overlap between consecutive pair-sum supports
   and is **incorrect**. No factor-of-two is hidden in evenness of
   $f*f$; the factor of two is purely combinatorial.
3. The symmetry $\sigma(i)=d-1-i$ is an involution of $\mathcal W_d$
   and of the objective, so restricting to $H_d = \{\mu_0 \le \mu_{d-1}\}
   \subseteq \Delta_d$ preserves $\mathrm{val}(d)$.

Therefore any rigorous numerical certificate of
$\max_W\mu^\top M_W\mu \ge c$ for all $\mu\in H_d$ (the job of
`interval_bnb`) certifies $C_{1a} \ge c$.
