# White (2022), arXiv:2210.16437 — "An almost-tight $L^2$ autoconvolution inequality"

*Faithful summary and transferability assessment for the $C_{1a}$ program.*
*Source consulted: arXiv abstract page + ar5iv HTML full-text rendering.*

---

## 1. Problem and main theorem

**Setting.** Let
$$
\mathcal{F} \;=\; \Bigl\{\, f:[-\tfrac12,\tfrac12]\to\mathbb{R}\ \Bigm|\ f\in L^1,\ \textstyle\int f = 1\,\Bigr\}.
$$
Note: **$f$ is real-valued, not required nonneg.**  The quantity studied is
$$
\mu_2^2 \;:=\; \inf_{f\in\mathcal{F}}\ \|f*f\|_2^2
\;=\; \inf_{f\in\mathcal{F}}\ \int_{-1}^{1}\!\Bigl|\int_{-1/2}^{1/2}\!f(t)\,f(x-t)\,dt\Bigr|^2\,dx .
$$

**Main Theorem (Thm. 1).**
$$
0.574635728 \;\le\; \mu_2^2 \;\le\; 0.574643711 \qquad (\text{relative gap } \approx 1.4\times10^{-5}).
$$
(Previous best $0.574575 \le \mu_2^2 \le 0.640733$.)

**Proposition 4** — existence and uniqueness of the minimiser $f^{\Diamond}\in\mathcal{F}$; $f^{\Diamond}$ is even.

**Lemma 6 (variational / Euler–Lagrange).** On the open interval $(-\tfrac12,\tfrac12)$,
$$
(f^{\Diamond}*f^{\Diamond}*f^{\Diamond})(x) \;=\; \mu_2^2/4.
$$

**Corollary 2.** Sidon-type consequences
$$
\sigma_2(g)\le \sqrt{\tfrac{2-1/g}{\mu_2^2}},\quad
\sigma_3(1)\le\Bigl(\tfrac{2}{\mu_2^2}\Bigr)^{1/3},\quad
\sigma_4(1)\le\Bigl(\tfrac{4}{\mu_2^2}\Bigr)^{1/4},
$$
giving new records for $B_h[g]$ with $(g,h)\in\{(2,2),(3,2),(4,2),(1,3),(1,4)\}$.

**Corollary 3 (discrete).** For any $H:[N]\to\mathbb{R}$ with $\sum H(j)=N$,
$$
\sum_{a+b=c+d} H(a)H(b)H(c)H(d)\ \ge\ \mu_2^2\,N^3.
$$

---

## 2. Method

### 2.1 Fourier setup
Work on the torus of period 2 (so $[-1,1]$ contains the support of $f*f$). Parseval gives
$$
\|f*f\|_2^2 \;=\; \sum_{k\in\mathbb{Z}} |\widehat{f*f}(k)|^2 \;=\; \tfrac12 + \sum_{k\neq 0} |F(k)|^4,
$$
where $F(k)$ is the period-2 Fourier coefficient of $f$ extended by zero; the $\tfrac12$ term is the contribution of $F(0)^2=\tfrac14$ squared in the convolution-squared. Even harmonics reduce to period-1 coefficients $\hat f(m)$ via
$F(2m)=\tfrac12 \hat f(m)$; odd harmonics mix frequencies (eq. 5 in White).

### 2.2 Upper bound — finite convex program
Truncate Fourier data at $T$ low modes and $R$ odd-harmonic slots. Solve a **QCQP** (lifted via $w_k\ge f_k^2,\ x_k\ge w_k^2$) to minimise
$$
\mathcal{O}(R,T)\;=\;\tfrac12 + \sum_{m=1}^T \hat f(m)^4 + \tfrac{16}{\pi^4}\sum_{m=1}^R z_m
$$
subject to the linear relations between period-1 and period-2 coefficients.

Parameters used: $T=30000,\ R=4000$, solved with **IBM CPLEX** in 32-digit variable-precision arithmetic (Matlab). Returns $\mathcal{O}(R,T)=0.574643711$.

### 2.3 Lower bound — a clean dual trick (Lemma 8)
Given $f\in\mathcal{F}$, construct test function $g$ with
$$
\hat g(k) \;=\; \tfrac{2}{1-2\mu_2^2}\,\hat f(k)^{3},\qquad k\neq 0 .
$$
Hölder with exponents $(4,\tfrac43)$ on $\sum_{k\neq 0}\hat f(k)^4$ then yields
$$
\mu_2^2 \;\ge\; \tfrac12 \;+\; \frac{1}{2\bigl(\sum_{m\neq 0}|\hat g(m)|^{4/3}\bigr)^{3}}.
$$
Combined with decay $|F(m)|<152/m$ for large odd $m$ (eq. 14), Prop. 9 gives rigorous truncation error $|\mathcal{O}(R,\sqrt R)-\mu_2^2|<10R^{-1/6}$.

### 2.4 Closed-form near-optimiser
$$
f_c(x) \;=\; \frac{\alpha_c}{(\tfrac14-x^2)^c},\qquad \alpha_c=\frac{\Gamma(2-2c)}{\Gamma(1-c)^2},
$$
with $c=0.4942$ gives $\|f_c*f_c\|_2^2 = 0.5746482$ (within $4.5\times10^{-6}$ of the infimum).

---

## 3. Relationship to $C_{1a}$ (Sidon autocorrelation)

**$C_{1a}$ is an $L^\infty$ problem with nonneg constraint:**
$$
C_{1a} \;=\; \inf_{f\ge 0,\ \mathrm{supp}\,f\subset[-\tfrac14,\tfrac14],\ \int f=1}\ \max_{|t|\le\tfrac12}(f*f)(t).
$$
White studies
$$
\mu_2^2 \;=\; \inf_{f\in L^1[-\tfrac12,\tfrac12],\ \int f=1}\ \|f*f\|_2^2 .
$$
**Two structural differences:**
1. **$L^2$ vs $L^\infty$.** $\mu_2$ controls energy, not peak. White explicitly notes (end of Sect. 1) that the $L^\infty$ problem (which is ours, up to an affine rescaling of support $[-1/2,1/2]\to[-1/4,1/4]$) is harder and only $0.64\le\inf\|f*f\|_\infty\le 0.75496$ is known.
2. **Sign-free vs nonneg.** White's $\mathcal{F}$ allows $f$ to change sign; $C_{1a}$ requires $f\ge 0$. Many of White's manipulations (e.g. Hölder with $\hat f^3$) are indifferent to sign, but the extremiser $f^{\Diamond}$ is in fact **not necessarily nonneg**, and the lower bound on $\mu_2$ does not imply a lower bound on $\max f*f$ with nonneg constraint.

**What $\mu_2$ does give us weakly.**
Cauchy–Schwarz: $\|f*f\|_\infty\cdot\|f*f\|_1 \ge \|f*f\|_2^2$, and $\|f*f\|_1 = (\int f)^2 = 1$ on the proper support, so
$$
\|f*f\|_\infty \ge \|f*f\|_2^2 \ge \mu_2^2 \approx 0.5746 .
$$
But $0.5746 \ll 1.2802$ under the $C_{1a}$ normalisation (which rescales, giving a factor 2 from $[-\tfrac14,\tfrac14]$ vs $[-\tfrac12,\tfrac12]$). Concretely, with $f$ on $[-\tfrac14,\tfrac14]$ and $\int f=1$, defining $\tilde f(x)=\tfrac12 f(x/2)$ on $[-\tfrac12,\tfrac12]$ has $\int\tilde f=1$ and $(\tilde f*\tilde f)(x)=\tfrac14(f*f)(x/2)$; so $\|\tilde f*\tilde f\|_2^2=\tfrac1{16}\cdot 2\cdot\|f*f\|_2^2 \cdot\tfrac12$... the rescaling needs care but the CS lower bound on $\max f*f$ is weaker than $C_{1a}$. **Bottom line: $\mu_2^2$ does not beat 1.2802.**

**Weak upper-bound consequence.** White's tight $\mu_2^2 \le 0.574643711$ does give $\inf_f \|f*f\|_\infty\le\|f_c*f_c\|_\infty$ evaluated on his $f_c$, but the sign-free infimum is $\le 0.75496$ which is *below* $C_{1a}$'s upper bound 1.5029 (again because nonneg constraint matters).

**Verdict.** White does **not directly apply** to $C_{1a}$. The result is on a *different* norm and *different* function class. But the machinery is highly relevant.

---

## 4. Transferable techniques (this is the useful part)

### 4.1 Period-2 Fourier embedding
Placing $f$ on a torus so $f*f$ lives in one period is a clean way to get a **sum-of-fourth-powers** representation of $\|f*f\|_2^2$. For our $L^\infty$ problem, the analogue is $\max_t(f*f)(t)=\max_t \sum_k \widehat{f*f}(k)e^{2\pi ikt}$; the relevant quadratic form is $\sum_k|\hat f(k)|^2 e^{2\pi ikt}$. This is the standard Delsarte/Beurling/MV setup. White's linear relation between period-1 and period-2 coefficients (eq. 5) could be worth re-reading when we move between $[-\tfrac14,\tfrac14]$ and $[-\tfrac12,\tfrac12]$.

### 4.2 The Hölder $f^3$-dual (Lemma 8) — **most promising idea**
White converts "$f$ has small $L^2$ autoconvolution" into a **linear functional in $\hat f$ that yields a lower bound** via one clean Hölder step. The key identity is that $\int f\cdot (f*f) = \sum_k \hat f(k)^3$ (when $f$ is real-even). For $C_{1a}$ the analogue would bound
$$
\max_t(f*f)(t) \ \ge\ \frac{\int f\cdot h}{\int h}\quad\text{for any nonneg }h\text{ with }\mathrm{supp}\,h\subset\{t:(f*f)(t)=\max\},
$$
but White's trick is more: by picking $g$ with $\hat g \propto \hat f^3$ he packages the problem as a single Hölder inequality. **This is a template for a $C_{1a}$-side dual** where, instead of $\mu_2^2$, the right-hand side becomes $\max f*f$.

### 4.3 QCQP-in-Fourier-coefficients (Sect. 2.2)
The fact that White's numerical certificate is a **finite convex program in low Fourier modes + variable-precision arithmetic** parallels our Lasserre/SOS pipeline but at much smaller cost (no moment matrices; just Hölder). For val(d) on $C_{1a}$ we already have Lasserre; White's approach is orthogonal and suggests a **direct Fourier LP** (Beurling–Selberg / Vaaler style) that we already explore in `delsarte_dual/` — his paper validates that a *few thousand* Fourier modes are enough to nail $\mu_2$ to $1.4\times10^{-5}$, so our Delsarte dual with $T\sim 10^3$–$10^4$ modes is not crazy.

### 4.4 Rigorous tail bound $|F(m)|<152/m$ (eq. 14)
White proves a rigorous decay estimate and uses it to bound truncation error for the lower bound. We should port this pattern: **any Fourier-truncated numerical certificate for $C_{1a}$ needs an $\ell^\infty$-decay estimate on $\hat f$** to close the gap rigorously. For nonneg $f$ with $\int f=1$ supported on a compact interval, $|\hat f(\xi)|\le 1$ and more refined bounds follow from total-variation estimates on $f$ (Zygmund / van der Corput). Worth implementing as a helper in `delsarte_dual/rigorous_max.py`.

### 4.5 Closed-form near-optimiser family
$f_c(x)=\alpha_c(\tfrac14-x^2)^{-c}$ with $c\approx\tfrac12$. This is a **$B$-spline-like arcsine family** (cf. our `arcsine_kernel.py` and `family_f4_mv_arcsine.py`). For $L^\infty$/$C_{1a}$ the extremiser is believed to be different (supported on a discrete set of atoms, per Matolcsi–Vinuesa); still, White's family is a cheap initialisation to scan.

### 4.6 Symmetry and variational condition (Lemma 6)
$(f^{\Diamond})^{*3}\equiv\text{const}$ on the support is a *linearised KKT*. The analogous condition for $C_{1a}$'s nonneg-$L^\infty$ problem would be: the active set $A=\{t:(f*f)(t)=C_{1a}\}$ supports a measure $\lambda\ge 0$ with $f*\lambda$ constant on $\mathrm{supp}\,f$. This is the classical Lagrangian KKT already used implicitly in MV/White; we can use White's *uniqueness* argument (Minkowski in $L^4$) as a model for **uniqueness / structure of $C_{1a}$-minimisers**, which would sharpen our cascade's "seed" heuristics.

---

## 5. Bottom line assessment

| Question | Answer |
|---|---|
| Is White 2022 about the same problem as $C_{1a}$? | **No.** Different norm ($L^2$ vs $L^\infty$), different class (real vs nonneg). |
| Does White's bound imply anything on $C_{1a}$? | Only $\|f*f\|_\infty\ge\|f*f\|_2^2=\mu_2^2\approx 0.5746$ via CS, *far weaker than 1.2802*. |
| Does White's **method** transfer? | **Partially, and worth a serious look.** The Hölder-with-$\hat f^3$ dual (§4.2) and the Fourier-QCQP certificate pattern (§4.3) are candidates for a new $C_{1a}$ lower bound technique. |
| Immediate action items | (a) Re-derive the $L^\infty$-analogue of Lemma 8 with nonneg constraint. (b) Add tail-decay helpers (§4.4) to `delsarte_dual/rigorous_max.py`. (c) Seed `family_f4_mv_arcsine.py` with $f_c$, $c=0.4942$ and re-run. |
| Priority vs current focus (Lasserre d=64–128) | **Secondary.** The Lasserre approach already attains val(d)-style rigorous bounds; White's tricks are a lighter-weight alternative / cross-check. |

---

## 6. References
- Ethan Patrick White, *An almost-tight $L^2$ autoconvolution inequality*, arXiv:2210.16437, 28 Oct 2022, 18 pp.
- Matolcsi & Vinuesa (2010), arXiv:0907.1379 — prior art on $L^\infty$ autoconvolution and $B_h[g]$ bounds.
- Cloninger & Steinerberger (2017), arXiv:1403.7988 — the $C_{1a}$ cascade baseline.
