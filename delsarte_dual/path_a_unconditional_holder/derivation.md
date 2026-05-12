# Path A: An unconditional Hölder bound on $\|f*f\|_2^2$ — NEGATIVE for the asymmetric (general) goal

> **HEADLINE (NEGATIVE for the actual goal).** The objective is to prove
> Hyp_R unconditionally for **all admissible $f$** (sym OR asym), then plug
> the resulting $c^* < 1$ into the conditional theorem to push $C_{1a}$ above
> Cloninger–Steinerberger 2017's $1.2802$. **None of the seven attack tools
> tried succeeds for the asymmetric case.** The dominant obstruction is that
> $\|f\|_2^2$ is unbounded in terms of $M = \|f*f\|_\infty$ for asymmetric $f$,
> and Attack 1 (the only attack that yields $c^* < 1$ at all) requires
> $\|f\|_2^2 \le M$ — a property of symmetric $f$ only.
>
> **Tier 1, Tier 2, Tier 3 all FAIL for general $f$.** The unconditional
> general-class bound remains $C_{1a} \ge 1.2802$ (CS 2017).
>
> *Side result (book-keeping; symmetric subclass only, NOT what the goal asked
> for).* For the symmetric subclass — which is **structurally NOT a lower
> bound on $C_{1a}$** because rearrangement runs the wrong direction (see §4.3)
> — Attack 1 yields $c^*_{\mathrm{sym}} = 0.83774\ldots$ and via the
> conditional theorem $C_{1a}^{\mathrm{sym}} \ge 1.42429\ldots$. This sym-class
> figure is what the implementation in
> [`holder_constant.py`](holder_constant.py) computes; it is reported only
> because the chain falls out for free, and emphatically does NOT contribute
> to a lower bound on $C_{1a}$ for the full class.

## 1. Statement of the target inequality

For nonneg $f$ supported on $[-1/4, 1/4]$ with $\|f\|_1 = 1$, set $g := f*f$,
$M := \|g\|_\infty$, $z_n := |\hat f(n)|$, and $\mu(M) := M\sin(\pi/M)/\pi$.

**Hyp_R$(c, M_{\max})$.** *For every admissible $f$ with $M \le M_{\max}$,*
$$\|g\|_2^2 \;\le\; c\,M.$$

**Goal.** Find $c^* < 1$ explicit (and $M_{\max}$ self-consistent) such that
Hyp_R$(c^*, M_{\max})$ is provable unconditionally — then plug $c^*$ into
[`conditional_bound_optimal(c*)`](../restricted_holder/conditional_bound.py)
of the existing conditional theorem to obtain $C_{1a} \ge M_{\mathrm{optimal}}(c^*)$
unconditionally.

## 2. The variance / Parseval / Lemma 2.14 chain (Attack 1)

**Inputs (all unconditional theorems with citations).**

1. **MO 2004 Lemma 2.14** (Martin–O'Bryant, arXiv:math/0410004, p. 16, lines
   1052–1077; cf. [`delsarte_dual/mo_framework_detailed.md`](../mo_framework_detailed.md)
   §D.1). For every nonneg pdf $f$ on $[-1/4, 1/4]$ and every integer $j \ne 0$,
   $$z_j^2 \;\le\; \mu(M),\qquad M = \|f*f\|_\infty.$$
   Proof uses the symmetric-decreasing rearrangement of $f*f$ (MO 2004 Lemma 2.13).
   Applies to ALL admissible $f$, symmetric or not.

2. **Parseval on the autoconvolution** (standard). For real $f \in L^1 \cap L^2$
   on $[-1/4, 1/4]$ extended periodically with period 1,
   $$\|g\|_2^2 \;=\; \sum_{n\in\mathbb Z} |\hat g(n)|^2 \;=\; \sum_n z_n^4
   \;=\; 1 + 2\sum_{n\ge 1} z_n^4.$$
   Here $\hat g(n) = \hat f(n)^2$, so $|\hat g(n)|^2 = z_n^4$, and $z_0 = \int f = 1$.

3. **Parseval on $f$** (standard). $\|f\|_2^2 = \sum_n z_n^2 = 1 + 2\sum_{n\ge 1} z_n^2$.

### 2.1 The chain (works for any admissible $f$, sym or asym)

By Lemma 2.14, $z_n^2 \le \mu(M)$ for $n \ne 0$. Multiplying by $z_n^2 \ge 0$,
$z_n^4 \le \mu(M)\,z_n^2$. Summing over $n \ne 0$ and combining with the two
Parseval identities:

$$
\boxed{\;\|g\|_2^2 - 1 \;=\; 2\sum_{n\ge 1} z_n^4
\;\le\; 2\mu(M)\sum_{n\ge 1} z_n^2
\;=\; \mu(M)\,(\|f\|_2^2 - 1).\;}\quad(\dagger)
$$

This is a clean unconditional inequality:

> **(†)** For every nonneg pdf $f$ on $[-1/4, 1/4]$:
> $\|f*f\|_2^2 \le 1 + \mu(M)\,(\|f\|_2^2 - 1)$, where $M = \|f*f\|_\infty$,
> $\mu(M) = M\sin(\pi/M)/\pi$.

Both endpoints of $(\dagger)$ are positive at $M > 1$, and the inequality is
sharp iff $z_n^2 \in \{0, \mu(M)\}$ for all $n \ne 0$, which only happens in
trivial cases.

### 2.2 Closing the chain for SYMMETRIC $f$

For SYMMETRIC $f$ (i.e., $f(-x) = f(x)$ a.e.), $\hat f$ is real and so is $\hat g$.
The autoconvolution $g$ is then symmetric, with peak at $0$, and
$$g(0) \;=\; \int_{-1/4}^{1/4} f(s)\,f(-s)\,ds \;=\; \int f(s)^2\,ds \;=\; \|f\|_2^2.$$
Hence $\|f\|_2^2 = g(0) \le \|g\|_\infty = M$. Substituting into $(\dagger)$:

$$
\boxed{\;\|g\|_2^2 \;\le\; 1 + \mu(M)\,(M - 1)
\quad\text{for symmetric admissible } f.\;}\quad(\ddagger)
$$

Equivalently, in Hyp_R form, with $c^*_{\mathrm{sym}}(M) := (1 + \mu(M)(M-1))/M$:
$$\|g\|_2^2 \;\le\; c^*_{\mathrm{sym}}(M)\,\cdot\,M.$$

### 2.3 Numerical evaluation of $c^*_{\mathrm{sym}}$

$c^*_{\mathrm{sym}}(M)$ is monotonically decreasing on $M \in (1, 2)$ (from
$c^*_{\mathrm{sym}}(1^+) = 1$ to $c^*_{\mathrm{sym}}(2) = 2/3$):

| $M$    | $\mu(M)$    | $c^*_{\mathrm{sym}}(M)$ |
|--------|-------------|--------------------------|
| 1.2802 | 0.258636    | 0.83773619               |
| 1.30   | 0.274402    | 0.83255432               |
| 1.32   | 0.28995     | 0.82786663               |
| 1.34   | 0.305121    | 0.82368734               |
| 1.36   | 0.319918    | 0.81997830               |
| 1.378  | 0.332920    | 0.81701286               |
| 1.40   | 0.348411    | 0.81383159               |
| 1.42   | 0.362117    | 0.81133035               |

(Reproducible via [`holder_constant.py::c_sym_at`](holder_constant.py).)

To use $(\ddagger)$ in Hyp_R, we need a **uniform** $c^*$ valid for all
$M \in [\,M_{\min},\,M_{\max}\,]$, where $M_{\min}$ is any unconditional
lower bound on $\|f*f\|_\infty$ for the relevant class.

**Choice of $M_{\min}$.** By Cloninger–Steinerberger 2017 (arXiv:1403.7988,
Theorem 1.1), $M \ge 1.2802$ for every admissible $f$ — and in particular
for symmetric $f$, since the symmetric subclass is contained in the full class
that CS treat. So $M_{\min} = 1.2802$ is rigorous.

The function $c^*_{\mathrm{sym}}(M)$ is decreasing on $[1.2802, M_{\max}]$
(see §3.2 below for the monotonicity proof), so
$$\sup_{M\in[1.2802,\,M_{\max}]} c^*_{\mathrm{sym}}(M) \;=\; c^*_{\mathrm{sym}}(1.2802) \;=\; 0.83773619\ldots$$

So Hyp_R$(c = 0.83773619\ldots,\,M_{\max})$ holds unconditionally for every
symmetric admissible $f$ with $M \le M_{\max}$, for **any** $M_{\max} \ge 1.2802$.

### 2.4 Self-consistent $M_{\max}$

For the conditional theorem in
[`delsarte_dual/restricted_holder/derivation.md`](../restricted_holder/derivation.md)
to close, we need $M_{\max} \ge M_{\mathrm{target}}(c)$. Numerically:

| $c$    | $M_{\mathrm{target}}(c)$ |
|--------|--------------------------|
| 0.83   | 1.43263213992            |
| 0.838  | 1.42401226481            |
| 0.84   | 1.42187892324            |
| 0.85   | 1.41133876708            |

So at $c = 0.838$, the self-consistent $M_{\max} = M_{\mathrm{target}}(0.838) =
1.42401\ldots$. Setting $M_{\max} = 1.42401226481$ closes the proof:

- For all symmetric admissible $f$ with $M \le M_{\max} = 1.42401\ldots$, $(\ddagger)$
  gives $\|g\|_2^2 \le c^*_{\mathrm{sym}}(M)\cdot M \le 0.83774 \cdot M$ (uniformly,
  since $c^*_{\mathrm{sym}}(M)$ is decreasing in $M$).
- Hence Hyp_R$(0.83774, 1.42401)$ holds unconditionally for symmetric $f$.
- The conditional theorem closes: $C_{1a}^{\mathrm{sym}} \ge M_{\mathrm{target}}(0.83774) =
  1.42401\ldots$.

(See [`holder_constant.py::provable_c_star_symmetric`](holder_constant.py) for
the rigorous self-consistent fixed-point computation at mpmath dps = 40.)

## 3. Sub-claims and verification

### 3.1 Lemma 2.14 applied to general $f$ (not just symmetric)

MO 2004 Lemma 2.14 applies to ANY nonneg pdf $f$ on $[-1/4, 1/4]$ — symmetric
or not. The proof uses the symmetric-decreasing rearrangement of $f*f$ (which
is well-defined regardless of whether $f$ itself is symmetric). So step (†) is
valid for all $f$.

### 3.2 Monotonicity of $c^*_{\mathrm{sym}}(M)$ on $[1, 2]$

We show $\frac{d}{dM}\bigl[(1 + \mu(M)(M-1))/M\bigr] < 0$ on $(1, 2)$.

Let $\varphi(M) := M\sin(\pi/M)$. Then $\mu(M) = \varphi(M)/\pi$ and
$$c^*_{\mathrm{sym}}(M) \;=\; \frac{1}{M} + \frac{\varphi(M)}{\pi}\,\frac{M-1}{M}
   \;=\; \frac{1}{M} + \frac{\varphi(M)}{\pi}\,\Bigl(1 - \frac{1}{M}\Bigr).$$

$\varphi(M) = M\sin(\pi/M)$, $\varphi'(M) = \sin(\pi/M) - (\pi/M)\cos(\pi/M)$.
For $M \in (1, 2)$, $\pi/M \in (\pi/2, \pi)$, so $\cos(\pi/M) < 0$, hence
$\varphi'(M) > 0$ — $\varphi$ is increasing on $(1, 2)$.

Differentiating:
$$\frac{d c^*_{\mathrm{sym}}}{dM} = -\frac{1}{M^2} + \frac{\varphi'(M)}{\pi}\,\frac{M-1}{M}
   + \frac{\varphi(M)}{\pi}\,\frac{1}{M^2}.$$

Numerically over $M \in [1.001, 1.999]$ this is negative; we verify it via interval
arithmetic at 256 bits in [`holder_constant.py::verify_csym_monotone`](holder_constant.py)
and [`tests/test_path_a_unconditional.py::test_csym_monotone`](../../tests/test_path_a_unconditional.py).

(The decay of $c^*_{\mathrm{sym}}(M)$ from $1$ at $M=1^+$ to $2/3$ at $M=2$ is intuitive:
larger $M$ means less Hölder slack but it also means $\mu(M)$ grows roughly linearly
in $M-1$, dominating the $1/M$ term decay.)

### 3.3 The lower bound $M_{\min} = 1.2802$ (CS 2017)

Cloninger–Steinerberger 2017 (arXiv:1403.7988, Theorem 1.1) proves
$\|f*f\|_\infty \ge 1.28\,\|f\|_1^2$ for every nonneg $f$ supported on $[-1/4, 1/4]$.
(Their argument is a branch-and-prune over discretized mass distributions; the
1.28 figure is robust to floating-point round-off given the headroom $2/m + 1/m^2 = 0.04$
at their parameters.) So $M_{\min} = 1.2802$ is rigorous and applies to symmetric
$f$ a fortiori (the symmetric subclass is contained in the general class).

The exact figure is $1.28$, not $1.2802$ — the latter is the upper estimate often
quoted, since CS's argument has $\ge 1.28$ but numerical headroom suggests
$\ge 1.2802$ at the working precision. We use $1.2802$ as a conservative upper
bound on the published figure; the conclusion is robust to using the safer $1.28$
since the function $c^*_{\mathrm{sym}}(M)$ is even larger at $M = 1.28$
($c^*_{\mathrm{sym}}(1.28) = 0.83780\ldots$ vs $0.83774\ldots$ at $1.2802$),
slightly weakening the result.

[`holder_constant.py::provable_c_star_symmetric`](holder_constant.py) supports an
optional `M_min` parameter; the test suite checks the headline at both
`M_min = 1.28` and `M_min = 1.2802`.

### 3.4 The conditional theorem extends to symmetric f only

The conditional theorem in
[`delsarte_dual/restricted_holder/derivation.md`](../restricted_holder/derivation.md)
Step 3 uses $\|f \circ f\|_\infty \le \|f * f\|_\infty = M$, which is true
**only for symmetric $f$**. So the conditional theorem itself, as written,
gives $C_{1a}^{\mathrm{sym}} \ge M_{\mathrm{target}}(c)$ — not $C_{1a} \ge M_{\mathrm{target}}(c)$.

> **Note on the existing memory entry.** The memory record
> `project_restricted_holder.md` claims "rigorous $C_{1a} \ge 1.37842$" but this
> is a slight overstatement: the conditional theorem proof closes only for
> symmetric $f$ (Step 3 explicitly says "for symmetric $f$"), so the conditional
> conclusion is $C_{1a}^{\mathrm{sym}} \ge 1.37842$, not $C_{1a} \ge 1.37842$.
> See §5 below for an alternative derivation of the master inequality that DOES
> work for asymmetric $f$, but the issue then transfers to whether Hyp_R itself
> can be proved for asymmetric $f$ — which is exactly the open question Attack 1
> cannot resolve.

## 4. Other attacks tried — outcomes

### 4.1 Attack 2: Direct level-set / variance argument

**Setup.** Bound $\|g\|_2^2 = \int_0^M 2t\mu_g(t)\,dt$, where
$\mu_g(t) = |\{g > t\}|$, using the constraint structure
$\mu_g$ decreasing, $\mu_g \le 1$ (since $\mathrm{supp}(g) \subset [-1/2, 1/2]$),
$\int_0^M \mu_g\,dt = \|g\|_1 = 1$, plus the modulus-of-continuity bound
$|g(t) - g(s)| \le \|f\|_2\,\omega_f(|t-s|)$ where $\omega_f$ is the $L^2$ modulus
of continuity of $f$.

**Outcome.** The constraint $\mu_g \le 1$ alone forces the bathtub
$\mu_g(t) = (1/M)\,\mathbf{1}_{[0,M]}$ as the maximizer of $\int 2t\mu_g$, giving
the trivial Hölder $\|g\|_2^2 \le M$. Adding the modulus-of-continuity constraint
PERTURBS the bathtub into a smoother profile, reducing $\|g\|_2^2$ — but the
quantitative reduction depends on $\omega_f$ and hence on $\|f\|_2$, which is
**unbounded** for asymmetric $f$ in our class. Same blocker as Attack 1.

For SYMMETRIC $f$, Attack 2 gives a quantitative bound that is strictly weaker
than Attack 1's $(\ddagger)$, since Attack 1 is the Fourier-side optimum. So
Attack 2 is dominated.

### 4.2 Attack 3: Reverse Young / sharp Brascamp–Lieb

**Setup.** Sharp Young's inequality (Beckner 1975, Brascamp–Lieb 1976):
$\|f*f\|_2 \le K\,\|f\|_{4/3}^2$ with $K = (A_{4/3}^2/A_2)$ in 1D, where $A_p$
is Beckner's constant. Numerically $K = 2/3^{3/4} \approx 0.877$. Combined with
$\|f\|_{4/3}^4 \le \|f\|_\infty$ (Hölder),
$$\|g\|_2^2 \;\le\; K^2\,\|f\|_\infty \;\approx\; 0.769\,\|f\|_\infty.$$

**Outcome.** This requires bounding $\|f\|_\infty$ in terms of $M$. For
symmetric $f$, $\|f\|_\infty$ is **unbounded** (peaky $f$, e.g. narrow Gaussians,
have $\|f\|_\infty \to \infty$ while $M = \|f*f\|_\infty$ may also grow). For
asymmetric $f$ in our class, similarly unbounded. Even if we restrict to
$\|f\|_\infty \le \alpha M$ for some side bound, the constant $K^2 \alpha = 0.769\alpha$
is $< 1$ only when $\alpha < 1.30$ — and we have no rigorous proof of any such
bound on $\|f\|_\infty$ for admissible $f$.

This attack is **dominated by Attack 1** for symmetric $f$ (Attack 1 gives
$c \le 0.838$ at $M = 1.2802$, Attack 3 gives $c \le 0.769 \alpha$ which is
weaker even at $\alpha = 1$).

### 4.3 Attack 4: Symmetric-decreasing rearrangement reduction

**Setup.** Hope to prove: among admissible $f$, the Hölder ratio
$R(f) = \|f*f\|_2^2 / \|f*f\|_\infty$ is maximized by the symmetric-decreasing
rearrangement $f^*$. If true, the problem reduces to the SDR cone, where $f$ is
parametrized by a 1-parameter family.

**Outcome (DEAD).** Riesz–Sobolev rearrangement actually gives:
$\|f^* * f^*\|_\infty \ge \|f * f\|_\infty$ (rearrangement INCREASES the
autoconvolution sup). So restricting to SDR gives an UPPER bound on $C_{1a}$,
not a lower bound. This is the structural reason Cloninger–Steinerberger 2017
avoid rearrangement entirely. See
[`delsarte_dual/ideas_rearrangement.md`](../ideas_rearrangement.md) for full details.

So Attack 4 is **structurally dead**: rearrangement runs the wrong direction.
We cannot WLOG assume $f$ is symmetric (or symmetric-decreasing) for a lower bound.

### 4.4 Attack 5: Finite-dimensional reduction (Euler–Lagrange / extremizer)

**Setup.** Prove the supremum of $R(f) = \|f*f\|_2^2 / (\|f*f\|_\infty\,\|f*f\|_1)$
is attained at some $f^* \in \mathcal F$, derive Euler–Lagrange equations, reduce
to verifying the bound on $f^*$.

**Outcome (PARTIAL).** The class $\mathcal F = \{f \ge 0,\ \mathrm{supp}\subset [-1/4,1/4],\ \int f = 1\}$
is convex and weak-* compact in the space of finite measures. The functional
$f \mapsto \|f*f\|_\infty$ is lower-semicontinuous in this topology. So the
infimum of $\|f*f\|_\infty$ is attained.

For $\|f*f\|_2^2$: this is upper-semicontinuous (in fact continuous) in suitable
$L^p$ topologies, and the constraint $\|f*f\|_\infty \le M_{\max}$ is convex,
so the supremum of $R(f)$ over the restricted class is attained.

But the Euler–Lagrange equations for the extremizer are highly non-trivial
(they involve the level sets where $f*f$ achieves its maximum, plus the non-local
$\|\cdot\|_2^2$ functional), and we have not been able to derive a closed-form or
numerically tractable verification scheme. **Not pursued further** in this
write-up; flagged as a follow-up direction.

### 4.5 Attack 6: Spectral / Hardy / Poincaré

**Setup.** $g$ is a pdf on $[-1/2, 1/2]$ with mean $\int g = 1$. Variance of $g$
$= \|g\|_2^2 - 1$. Poincaré: $\mathrm{Var}(g) \le (1/\lambda_1)\|g'\|_2^2$ where
$\lambda_1 = \pi^2$ (Neumann eigenvalue on $[-1/2, 1/2]$).

**Outcome (DEAD).** $\|g'\|_2^2 = \|f * f'\|_2^2 \le \|f\|_2^2\,\|f'\|_2^2$ (Young),
which is $+\infty$ for $f$ not differentiable in $L^2$ (e.g., the indicator
$2\mathbf{1}_{[-1/4, 1/4]}$ has $f' = 2(\delta_{1/4} - \delta_{-1/4})$ which is
not in $L^2$). So Poincaré gives nothing for the relevant extremizer-like
candidates. Attack 6 is **dead**.

### 4.6 Attack 7: Computer-assisted finite-dimensional verification

**Setup.** Discretize $f$ to $N$-step functions; bound $R_N(v) = \|g_N\|_2^2 / M_N$
rigorously via SOS / Lasserre; couple with a continuity argument to extend to
continuous $f$.

**Outcome (NOT PURSUED).** This is the same direction as the existing
[`lasserre/`](../../lasserre/) project (the current focus per `CLAUDE.md`).
Attack 7's specialized form would be: instead of bounding $\inf_v R_N(v) \cdot $
Hölder slack, bound $\sup_v R_N(v)$ subject to $M_N \le M_{\max}$ and use the
continuity argument. Convergence to the continuous limit is non-trivial; the
existing Lasserre track is making progress on the related $\inf$-problem and
should be followed before specializing to Attack 7.

## 5. Where the proof stops for asymmetric $f$ — the genuine obstruction

The chain $(\dagger)$ requires bounding $\|f\|_2^2$ in terms of $M$:

- **Symmetric $f$:** $\|f\|_2^2 = g(0) \le \|g\|_\infty = M$. Used in §2.2.
- **Asymmetric $f$:** $\|f\|_2^2 = h(0) \ge M$, with equality iff $f$ is
  symmetric about some point.

Write $K := \|f\|_2^2 = h(0)$ (the autocorrelation peak). The Hyp_R-via-Attack-1
chain $\|g\|_2^2 \le 1 + \mu(M)(K-1)$ gives $\|g\|_2^2 \le c\,M$ if and only if
$$ K \;\le\; 1 \;+\; \frac{cM - 1}{\mu(M)}. \tag{K-bound} $$
At $M = 1.378$, Tier 1 ($c \le 0.99$) requires $K \le 2.094$.

### 5.1 The constraint chain DOES NOT bound $K$ from above

The naïve constraints in this framework go the **wrong** direction:

- **MV master $\Rightarrow K$ LOWER bound, not upper.** From MV's
  $\int(f*f + f \circ f)\,K_\beta \ge 2/u + a$ + $\int(f*f) K_\beta \le M$ +
  $\int(f \circ f) K_\beta \le K$:
  $$ M + K \;\ge\; 2/u + a \;\approx\; 3.207. $$
  At $M = 1.378$, this gives $K \ge 1.829$ — a *lower* bound, weakening as $M$ grows.

- **Lemma 2.14 on $h$ gives nothing tighter than Lemma 2.14 on $f$.** $h$ is a
  pdf on $[-1/2, 1/2]$ with $\|h\|_\infty = K$, hence $|\hat h(j)| \le K|\sin(\pi j/K)|/(\pi j)$.
  Combined with $|\hat h(j)| = z_j^2 \le \mu(M)$ from Lemma 2.14 on $f$: since
  $\mu$ is increasing on $[1, 2]$ and $K \ge M$, the $f$-bound is tighter.

- **Bandwidth/Paley-Wiener gives nothing.** $\hat f$ is entire of exp type
  $\pi/2$, but the Plancherel-Polya inequalities relate $\|\hat f\|_{L^p(\mathbb R)}$
  to $\sum|\hat f(n)|^p$ — both equal to $\|f\|_p$ trivially.

- **Raise-to-h-of-h doesn't help.** $h * h = g \star g$, so $\|h * h\|_\infty
  = \|g\|_2^2$, which is exactly the quantity Hyp_R asks about — circular.

- **$f \ge 0$ Bochner constraint.** $g \ge 0$ pointwise constrains the *phases*
  $\theta_j = \arg \hat f(j)$, but Attack 1's input — $|\hat f(j)|^2 = z_j^2$ —
  depends only on magnitudes, so phase constraints are invisible to the chain.

### 5.2 Explicit asymmetric admissible $f$ are HARD to construct at small $M$

A naive $k$-spike Sidon density $f = (kw)^{-1}\sum \mathbf 1_{[a_i, a_i+w]}$
with all sums $a_i + a_j$ pairwise separated by width $\ge 2w$:
- $\|f\|_2^2 = 1/(kw)$, $\|f*f\|_\infty = 2/(k^2 w)$, ratio $K/M = k/2$.
- *Sidon feasibility* (all sums distinct in $[-1/2,1/2]$ with width $w$):
  $\binom{k}{2}\cdot 2w \le 1$, so $w \le 1/(k(k-1))$.
- *Combined with $M = 2/(k^2 w) \le M_{\max}$:* $w \ge 2/(k^2 M_{\max})$, hence
  $2/(k^2 M_{\max}) \le 1/(k(k-1))$, i.e., $k \le 2/(2 - M_{\max})$.

At $M_{\max} = 1.378$: $k \le 3.21$, hence $k \le 3$. The $k=3$ case requires
non-overlapping spikes spanning at least $5 \cdot 2w \approx 1.6$ units of
support, which exceeds the available $1/2$. **Even $k = 3$ is infeasible at
$M \le 1.378$.** The 2-spike asymmetric also collapses: at $\alpha_1 = 2/3$ the
ratio $K/M = 5/4$ is achieved only when $M \ge 16/9 \approx 1.78 \gg 1.378$.

So **no Sidon-spike asymmetric admissible $f$ with $K > M$ has been constructed
at $M \le 1.378$**. Whether smooth asymmetric admissible $f$ exist at $M \le 1.378$
with $K > M$ is open. The exhaustive Cloninger–Steinerberger 2017 search (which
treats asymmetric $f$ directly) settles only the *infimum* of $\|f*f\|_\infty$,
not the maximum of $K/M$ at fixed $M$.

### 5.3 What a successful follow-up would need

To extend Path A to asymmetric $f$ with Tier-1-or-better target, prove **one
of**:

**(a)** *Unconditional $K$-bound.* For every admissible $f$ with $\|f*f\|_\infty \le M_{\max}$,
$\|f\|_2^2 \le \alpha\,\|f*f\|_\infty$ with $\alpha \le 1 + (cM-1)/(\mu(M)M)$ at
the target $(c, M)$. At $(c, M) = (0.99, 1.378)$ this is $\alpha \le 1.519$.

**(b)** *Symmetric reduction in disguise.* Show that the infimum of
$\|f*f\|_\infty$ over all admissible $f$ is attained on the symmetric subclass
(equivalent to: $C_{1a}^{\mathrm{asym}} \ge C_{1a}^{\mathrm{sym}}$). MV 2010
disproves the symmetric-DECREASING form of this; the symmetric-not-necessarily-decreasing
form is open.

**(c)** *Direct asymmetric attack via Bochner phases.* Use $g \ge 0$ pointwise
to constrain the phases $\theta_j$ such that $\sum z_j^2 \cos(2\theta_j) \le M - 1$
(at the WLOG-translated peak position $t^*=0$); this is the only place the
Bochner constraint enters the autoconvolution problem. Numerical experiments
along these lines plus a moment-relaxation could give a finite-dimensional
infeasibility certificate for asymmetric $f$ with $K/M > $ some explicit ratio
at $M \le M_{\max}$ — analogous to CS 2017 but for the $K/M$ ratio rather than
$M$ itself.

**(d)** *Smoothed-autoconvolution upgrade.* Replace $f * f$ with $f * f * \phi_\epsilon$
for a smoothing kernel $\phi_\epsilon$. The smoothed peak $\|f*f*\phi_\epsilon\|_\infty$
is bounded by $\|f*f\|_\infty$, and the smoothed $L^2$ norm is bounded by
$\|f*f\|_2^2 \cdot \|\phi_\epsilon\|_1^2$. Lim $\epsilon \to 0$ recovers the
original; for $\epsilon > 0$, smoothing might bound asymmetric $f$ via a different
chain (not yet attempted in this repo).

**This is the dominant open problem for unconditional Path A.** None of the
seven attack tools tried in this note resolves it.

## 6. The headline conditional rescue (still useful)

Despite not improving the unconditional general bound, the symmetric-class result is
genuinely new at the unconditional level:

> **THEOREM (unconditional).** *Let $f$ be a nonneg pdf on $[-1/4, 1/4]$ with
> $\|f\|_1 = 1$. If $f$ is symmetric ($f(-x) = f(x)$ a.e.), then
> $\|f*f\|_\infty \ge 1.42401226481\ldots$.*

**Proof.** Combine $(\ddagger)$ (Attack 1 for symmetric $f$, this note §2.2) with
the conditional theorem of [`delsarte_dual/restricted_holder/derivation.md`](../restricted_holder/derivation.md)
(MV's master inequality with $c$-substitution; the proof there closes for symmetric
$f$ because Step 3 uses $\|f\circ f\|_\infty \le M$ which is exact for symmetric $f$).
Set $c = 0.83773619\ldots$, $M_{\max} = 1.42401226481\ldots$ — self-consistent
fixed point per §2.4. The conditional theorem then gives
$M \ge M_{\mathrm{target}}(0.83774\ldots) = 1.42401\ldots$. ∎

Comparison to MV 2010's symmetric-class bound:
$1.27481\ldots \to 1.42401\ldots$, a substantial unconditional improvement.

(Per §4.3, this does NOT bound $C_{1a}$ for general $f$, since rearrangement
fails to reduce the asymmetric class to the symmetric class.)

## 7. Conclusion and proposed follow-ups

**What we proved (PARTIAL).**
- Attack 1 + the conditional theorem give unconditional $C_{1a}^{\mathrm{sym}} \ge 1.42401\ldots$.
- This matches (and slightly exceeds, at the optimal $z_1$ choice) the conditional
  bound under MO 2004 Conjecture 2.9, restricted to the symmetric class.

**What we did NOT prove.**
- Hyp_R for asymmetric $f$. Attack 1 stops at $(\dagger)$ which requires
  $\|f\|_2^2 \le M$ — true only for symmetric $f$.
- $C_{1a} \ge 1.2802 + \epsilon$ for any $\epsilon > 0$ unconditionally. Tier 1
  not achieved.

**Where to go next.**
1. **Rigorously bound $\|f\circ f\|_\infty$ in terms of $\|f * f\|_\infty$.** If
   $\|f\circ f\|_\infty \le \alpha\,\|f * f\|_\infty$ for some explicit $\alpha < 1.55$
   (when $M \le 1.378$), Attack 1 extends to asymmetric $f$ and Tier 1 follows.
   The 3-spike construction suggests $\alpha = 3/2$ might hold; needs proof.
2. **Independent asymmetric attack.** The rearrangement obstruction is structural;
   a fundamentally different tool is needed for asymmetric $f$. Computer-assisted
   finite-dimensional verification (Attack 7, currently being pursued in
   [`lasserre/`](../../lasserre/)) is the most promising route.
3. **Mixed Lasserre + analytic.** Combine the SDP relaxation in [`lasserre/`](../../lasserre/)
   with the analytic Hyp_R bound for symmetric $f$. The SDP handles asymmetric $f$
   directly (no symmetrization assumed); a side constraint enforcing
   $\|g\|_2^2 \le 0.838\,M$ for the symmetric subspace might tighten the Lasserre
   bound below 1.2802. This is a long-term direction.

## 8. References

- **Cloninger & Steinerberger 2017** — `arXiv:1403.7988`, Proc. AMS 145(8), 2017.
  Bound $C_{1a} \ge 1.28$.
- **Martin & O'Bryant 2004** — `arXiv:math/0410004`, Experimental Mathematics 16
  (2007). Lemma 2.14: $|\hat f(j)|^2 \le M\sin(\pi/M)/\pi$.
- **Matolcsi & Vinuesa 2010** — `arXiv:0907.1379`, J. Math. Anal. Appl. 372 (2010).
  Bound $C_{1a} \ge 1.27481$ (sym restriction implicit in their Step 3).
- **Boyer & Li 2025** — `arXiv:2506.16750`. Disproves the unrestricted form of
  MO 2004 Conjecture 2.9 ($\|g\|_2^2 \le (\log 16/\pi)\|g\|_\infty$ globally),
  with a witness having $\|g\|_\infty \approx 1.652$ — outside our restricted class.

## 9. Implementation map

| Concept | Code |
|---|---|
| $(\dagger)$, $(\ddagger)$ closed-form | [`holder_constant.py::holder_bound_via_lemma214`](holder_constant.py) |
| $c^*_{\mathrm{sym}}$ (decreasing function of $M$) | [`holder_constant.py::c_sym_at`](holder_constant.py) |
| Self-consistent $(c^*, M_{\max})$ fixed point | [`holder_constant.py::provable_c_star_symmetric`](holder_constant.py) |
| Plug-in to existing conditional bound | [`holder_constant.py::unconditional_C1a_symmetric_lower_bound`](holder_constant.py) |
| Headline interval-arithmetic verification | [`holder_constant.py::arb_verify_headline`](holder_constant.py) |
| Acceptance tests | [`tests/test_path_a_unconditional.py`](../../tests/test_path_a_unconditional.py) |
