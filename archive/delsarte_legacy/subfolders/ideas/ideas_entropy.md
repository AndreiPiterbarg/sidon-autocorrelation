# Information-Theoretic / Entropy Approaches to $\|f*f\|_\infty$

**Question.** For $f \ge 0$, $\mathrm{supp}(f)\subseteq[-1/4,1/4]$, $\int f = 1$: can information-theoretic inequalities yield a non-trivial lower bound on $\sup_{|t|\le 1/2}(f*f)(t) = \|f*f\|_\infty$ that matches or beats $C_{1a}\ge 1.2802$?

## 1. Dictionary: sup-norm $\leftrightarrow$ min-entropy

For a probability density $f$, the (differential) min-entropy is
$$H_\infty(f) := -\log \|f\|_\infty, \qquad N_\infty(f) := e^{2H_\infty(f)} = \|f\|_\infty^{-2}.$$
So $\|f*f\|_\infty = e^{-H_\infty(f*f)}$. Lower-bounding $\|f*f\|_\infty$ is equivalent to **upper-bounding** $H_\infty(f*f)$. This is the *wrong* direction for the classical Entropy Power Inequality (EPI), which gives *lower* bounds on $H_\infty(f*f)$. Thus we need **reverse** entropy/min-entropy inequalities, or inequalities leveraging compact support.

## 2. Three promising directions

### 2a. Rogozin's max-density inequality (the right tool, wrong direction)
Rogozin (1987) and its modern refinements (Bobkov–Chistyakov 2014, Madiman–Melbourne–Xu 2017, arXiv:1705.00642) prove
$$\|f*g\|_\infty \;\le\; \frac{C}{\sqrt{\sigma_f^{-2} + \sigma_g^{-2}}}\cdot \|f\|_\infty \|g\|_\infty \cdot (\text{correction}),$$
and more precisely the $\infty$-Rényi EPI
$$N_\infty(X+Y)\ge N_\infty(X)+N_\infty(Y),\quad\text{i.e. } \|f*g\|_\infty^{-2}\ge \|f\|_\infty^{-2}+\|g\|_\infty^{-2}.$$
For $f=g$: $\|f*f\|_\infty\le \|f\|_\infty/\sqrt 2$ — an **upper bound**. Not useful as written, but the equality case (uniform on an interval of length $1/\|f\|_\infty$) is informative: it identifies extremal $f$ for the sup-norm of the convolution.

### 2b. Reverse EPI via compact support / log-concavity (Bobkov–Madiman, Cover–Zhang)
For log-concave $f$ with $\mathrm{Var}(X)=\sigma^2$, one has $\|f\|_\infty \ge c/\sigma$ for universal $c>0$. For $f$ supported on $[-1/4,1/4]$, $\sigma^2\le 1/16$, giving $\|f\|_\infty\ge 4c$ and by convolution $\|f*f\|_\infty\ge 2\sqrt{2}c$. For the arcsine extremizer, $c\approx 1/\pi$, giving a bound $\approx 0.9$ — **below** 1.2802. This route cannot beat the current record without additional structure, because log-concavity is too restrictive (the near-optimal Cloninger–Steinerberger profiles are *not* log-concave).

### 2c. Entropic uncertainty + Plancherel (most promising)
$\|f*f\|_\infty = \|\hat f\|_2^2$-type identity is **false** (that's $L^2$); but
$$\|f*f\|_\infty = \sup_t \int \hat f(\xi)^2 e^{2\pi i\xi t}\,d\xi \ge \int |\hat f(\xi)|^2\,d\xi = \|f\|_2^2$$
only when $\hat f^2$ is nonneg-real, which it is if $f$ is symmetric. Combine with the **Hirschman–Beckner entropic uncertainty**
$$h(f) + h(\hat f^2) \ge \log(e/2),$$
and the support constraint $\mathrm{supp}(f)\subseteq[-1/4,1/4]$ which forces $h(f)\le \log(1/2)$. This *bounds $h(\hat f^2)$ from below*, which via Jensen/max-entropy gives bounds on $\|\hat f\|_4^4 = \|f*f\|_2^2$ — an $L^2$ object, **not** $L^\infty$. Same wall as Boyer–Li.

## 3. New idea: conditional min-entropy + kernel positivity

The Sidon problem has hidden structure the entropy literature ignores: $\mathrm{supp}(f*f)\subseteq[-1/2,1/2]$ with total mass 1, so the *average* of $f*f$ over its support is $\ge 1$. Partition $[-1/2,1/2]$ into atoms and apply a **Shearer-type entropy inequality** to the joint law of $(X_1,X_2)$ conditioned on $X_1+X_2\in A$. This yields, for any sub-interval $A$ of length $|A|$,
$$\mathrm{Pr}(X_1+X_2\in A)\ge |A|\cdot \mathrm{ess\,inf}_{t\in A}(f*f)(t),$$
and combined with the Cloninger–Steinerberger forbidden-mass lemmas, restricts which atoms can simultaneously have low density — potentially producing a feasibility LP whose dual gives a pointwise lower bound on $\max_t (f*f)(t)$.

## 4. Verdict

Pure entropy methods give the *wrong* direction (all classical EPIs *lower*-bound $H(f*f)$, i.e. *upper*-bound $\|f*f\|_\infty$). The Rogozin/$\infty$-Rényi-EPI family, reverse EPI (log-concave only), and Hirschman uncertainty all fail to cross 1.2802 without auxiliary structure. **Section 3** (Shearer + support atoms) is the only novel thread; it reduces to a finite LP similar to (but dual to) the cascade-prover, so expected gain is marginal over existing Cloninger–Steinerberger methods. **Recommendation:** low priority versus the active Lasserre/Farkas pipeline.

## 5. Key references

- Bobkov, Chistyakov (2014). *Bounds on the maximum of the density for sums* — Rogozin-type bound.
- Madiman, Melbourne, Xu (arXiv:1705.00642). *Rogozin's inequality for locally compact groups* — unifies $\infty$-Rényi EPI with Rudelson–Vershynin marginals.
- Bobkov, Chistyakov (2015). *EPI for the Rényi entropy*.
- Hochman (2014, Annals). *Self-similar sets, entropy, additive combinatorics* — inverse theorems for entropy growth under convolution.
- Hirschman (1957); Beckner (1975). *Entropic uncertainty principle*.
- Bobkov, Madiman (2011). *Reverse EPI for log-concave*.

---

## 100-word summary

The Sidon $C_{1a}$ lower bound asks to minorize $\|f*f\|_\infty$ for $f\ge 0$ supported on $[-1/4,1/4]$, $\int f=1$. In entropic language this means *upper*-bounding the min-entropy $H_\infty(f*f)$, opposite to classical EPIs. The $\infty$-Rényi EPI (Rogozin–Bobkov–Chistyakov) gives $\|f*f\|_\infty\le\|f\|_\infty/\sqrt 2$ — wrong direction. Reverse EPI for log-concave $f$ (Bobkov–Madiman) yields only $\|f*f\|_\infty\gtrsim 0.9$. Hirschman–Beckner uncertainty reduces to $L^2$ targets (as in Boyer–Li). The one novel angle — Shearer-type entropy on $(X_1,X_2)\mid X_1+X_2\in A$ combined with forbidden-mass LPs — reduces to variants of the Cloninger–Steinerberger cascade. **Verdict: low priority** versus ongoing Lasserre/Farkas work.
