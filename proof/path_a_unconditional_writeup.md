# Path A: Negative result on the unconditional Hölder bound for general $C_{1a}$

> **Negative result (what was actually asked for).** The goal is to prove
> Hyp_R unconditionally for **all admissible $f$** and use it to push
> $C_{1a}$ above Cloninger–Steinerberger 2017's 1.2802. Seven attack tools
> were tried; **none yields a constant $c^* < 1$ for general (asymmetric) $f$**.
> The unconditional general-class bound remains $C_{1a} \ge 1.2802$ (CS 2017),
> with no improvement.
>
> The dominant obstruction (§3 below): Attack 1 (the only attack producing
> $c^* < 1$) needs $\|f\|_2^2 \le M$, which is exact for symmetric $f$ but
> fails for asymmetric $f$ — and rearrangement cannot reduce asymmetric to
> symmetric (it runs the WRONG direction; see [`ideas_rearrangement.md`](../delsarte_dual/ideas_rearrangement.md)).
>
> *Side result that the implementation produces (NOT a bound on $C_{1a}$).*
> Restricted to the symmetric subclass, Attack 1 yields $c^* = 0.83774\ldots$
> and $C_{1a}^{\mathrm{sym}} \ge 1.42429\ldots$. **This is NOT a lower bound on
> $C_{1a}$**, since the inf of $\|f*f\|_\infty$ over admissible $f$ is not
> known to be attained on symmetric $f$ (in fact, Matolcsi–Vinuesa 2010's
> upper-bound construction is asymmetric). It is reported here only because
> it falls out of the same chain.

## 1. Background and the Sidon autoconvolution constant

For nonneg integrable $f$ supported on $[-1/4, 1/4]$ with $\int f = 1$, the
Sidon autoconvolution constant is
$$ C_{1a} := \inf_{f \in \mathcal F} \|f * f\|_\infty,
\qquad \mathcal F := \{f \ge 0 : \mathrm{supp}\,f \subset [-1/4, 1/4],\,\int f = 1\}. $$

The published bounds are $1.2802 \le C_{1a} \le 1.5029$, with the lower bound
due to Cloninger & Steinerberger 2017 [CS17] and the upper bound due to
Matolcsi & Vinuesa 2010 [MV10].

This note proves the symmetric-restricted bound $C_{1a}^{\mathrm{sym}} \ge 1.42429\ldots$
unconditionally — a substantial improvement on MV's 1.27481 for the same
restricted class.

## 2. The two ingredients

**Ingredient A (MO 2004 Lemma 2.14).** [MO04, Lemma 2.14, p. 16.] For every
nonneg pdf $f$ on $[-1/4, 1/4]$ with $\int f = 1$, and every integer $j \ne 0$:
$$ |\hat f(j)|^2 \;\le\; \mu(M),\qquad \mu(M) := \frac{M\sin(\pi/M)}{\pi},
\qquad M = \|f*f\|_\infty. $$

The proof uses symmetric-decreasing rearrangement of $f * f$ and the
observation that the cosine-integral of any nonneg pdf with $L^\infty$ norm
$\le M$ is maximized by the bathtub indicator. **The bound is independent of
the symmetry of $f$.**

**Ingredient B (the conditional theorem of [`restricted_holder/derivation.md`](../delsarte_dual/restricted_holder/derivation.md)).**
For any constant $c \in (0, 1]$ and $M_{\max} \ge 1$, define
$$\text{Hyp}_{R}(c, M_{\max}) := \forall\,f \in \mathcal F\text{ with } \|f*f\|_\infty \le M_{\max},
\quad \|f*f\|_2^2 \le c\,\|f*f\|_\infty\,\|f*f\|_1.$$

Then, modulo Hyp$_R(c, M_{\max})$ (which we will verify for the symmetric
subclass below), the conditional theorem proves
$$ C_{1a}^{\mathrm{sym}} \;\ge\; M_{\mathrm{optimal}}(c), $$
where $M_{\mathrm{optimal}}(c)$ is the infimum over admissible $z_1$ of MV's
master inequality with the Hölder slack $\|f*f\|_2^2 \le M$ replaced by
$\|f*f\|_2^2 \le c M$. The closed form is in §5 of `restricted_holder/derivation.md`,
and a numerical implementation is in `restricted_holder/conditional_bound.py::conditional_bound_optimal`.

## 3. The proof for symmetric $f$

The chain has three steps.

### Step 1: Parseval + Lemma 2.14 (universal)

By Parseval applied to $g := f * f$ (which is in $L^2$ since $f \in L^2$ — note
$\|f\|_2^2 \le 2$ if $f$ is bounded, and in general $\|f\|_2 < \infty$ by
$\int f^2 = h(0)$ with $h$ the autocorrelation having $h(0) \le \|f\|_\infty\,\|f\|_1$
when $\|f\|_\infty < \infty$; the $L^2$ assumption is mild and standard):
$$ \|g\|_2^2 = \sum_{n\in\mathbb Z} |\hat g(n)|^2 = \sum_n z_n^4
   = 1 + 2 \sum_{n\ge 1} z_n^4, $$
where $z_n := |\hat f(n)|$ and $\hat g(n) = \hat f(n)^2$, hence $|\hat g(n)|^2 = z_n^4$,
and $z_0 = \int f = 1$.

Likewise, Parseval on $f$ gives
$$ \|f\|_2^2 = \sum_n z_n^2 = 1 + 2 \sum_{n\ge 1} z_n^2. $$

By Ingredient A, $z_n^2 \le \mu(M)$ for all $n \ne 0$. Multiplying by $z_n^2 \ge 0$:
$$ z_n^4 \le \mu(M)\,z_n^2,\quad\forall n\ne 0. $$

Summing and substituting:
$$\|g\|_2^2 - 1 \;=\; 2\sum_{n\ge 1} z_n^4 \;\le\; 2\mu(M)\sum_{n\ge 1} z_n^2
   \;=\; \mu(M)\,(\|f\|_2^2 - 1). \tag{$\dagger$}$$

This step is **valid for all admissible $f$, symmetric or not**.

### Step 2: $\|f\|_2^2 \le M$ for symmetric $f$ (where symmetry enters)

If $f(-x) = f(x)$ a.e., then $g(0) = \int f(s) f(-s)\,ds = \int f(s)^2\,ds = \|f\|_2^2$,
and $g(0) \le \|g\|_\infty = M$. So
$$ \|f\|_2^2 \;\le\; M,\quad\text{for symmetric } f. \tag{$\star$}$$

(For asymmetric $f$, $\|f\|_2^2 = h(0)$, the autocorrelation peak, which can
strictly exceed $M$. This is exactly the obstruction documented in §5 below
and in [`derivation.md`](../delsarte_dual/path_a_unconditional_holder/derivation.md) §5.)

### Step 3: Combine and close

Substituting ($\star$) into ($\dagger$):
$$ \|g\|_2^2 \;\le\; 1 + \mu(M)(M - 1) \;=:\; c^*_{\mathrm{sym}}(M)\,\cdot\,M, $$
where $c^*_{\mathrm{sym}}(M) := (1 + \mu(M)(M-1))/M$.

The function $c^*_{\mathrm{sym}}(M)$ is monotonically decreasing on $(1, 2)$
(verified by direct computation of the derivative, see
[`holder_constant.py::verify_csym_monotone`](../delsarte_dual/path_a_unconditional_holder/holder_constant.py)
and `tests/test_path_a_unconditional.py::test_csym_monotone`). So for any
$M_{\max}$ and $M_{\min}$ with $1 \le M_{\min} \le M_{\max}$,
$$ \sup_{M \in [M_{\min},\,M_{\max}]} c^*_{\mathrm{sym}}(M) = c^*_{\mathrm{sym}}(M_{\min}). $$

Setting $M_{\min} = 1.2802$ (the unconditional CS 2017 bound, applied a fortiori
to the symmetric subclass) gives
$$ c^* := c^*_{\mathrm{sym}}(1.2802) \;=\; 0.83773619\ldots $$

So Hyp$_R(c^* = 0.83774\ldots, M_{\max})$ holds **unconditionally** for symmetric
$f$, for any $M_{\max} \ge 1.2802$.

Plugging into Ingredient B with $c = 0.83774\ldots$ gives
$$ M_{\mathrm{optimal}}(0.83774\ldots) \;=\; 1.42429430224\ldots, $$

and choosing $M_{\max} = M_{\mathrm{optimal}}(c) = 1.42429\ldots$ closes the proof:
$$\boxed{C_{1a}^{\mathrm{sym}} \;\ge\; 1.42429430224486499006\ldots\;}$$

(Self-consistency: at $M_{\max} = 1.42429\ldots$, $c^*_{\mathrm{sym}}(M)$ has
sup $0.83774$ at $M = M_{\min} = 1.2802$. ✓)

## 4. Verification

All numerics are reproducible to 40 decimal digits via
[`holder_constant.py`](../delsarte_dual/path_a_unconditional_holder/holder_constant.py)
and at 256 bits via the `arb_verify_headline()` function. Acceptance tests
are in [`tests/test_path_a_unconditional.py`](../tests/test_path_a_unconditional.py).

| Acceptance | Tier | Result |
|------------|------|--------|
| $c^* < 1$ strictly | (sym only) | PASS — $c^* = 0.83774$ |
| $c^* \le 0.99$ | Tier 1 (sym only) | PASS |
| $c^* \le 0.95$ | Tier 2 (sym only) | PASS |
| $c^* \le \log 16/\pi$ | Tier 3 (sym only) | PASS — $0.83774 < 0.88254$ |
| Beats CS 2017's 1.2802 (sym) | headline | PASS — $1.42429 > 1.2802$ |
| **Beats CS 2017 for general $f$** | **the actual goal** | **NOT ACHIEVED** |

## 5. The asymmetric obstruction (genuine open problem)

The whole proof above assumes $f$ is symmetric, used only in Step 2 to convert
$\|f\|_2^2$ to $M$. For asymmetric $f$:

- $\|f\|_2^2 = h(0)$, the autocorrelation peak — strictly greater than
  $\|f * f\|_\infty$ in general.
- An explicit asymmetric example (3-spike Sidon, see
  [`derivation.md`](../delsarte_dual/path_a_unconditional_holder/derivation.md) §5.1):
  $\|f\|_2^2 / \|f * f\|_\infty = 3/2$ at $M = 1.378$.
- For larger $k$-spike ($k \ge 4$), $M < 1.378$ is infeasible, but this is
  a specific construction, not a general bound.
- The supremum $\alpha^* := \sup_f \|f\|_2^2 / \|f*f\|_\infty$ over admissible $f$
  with $M$ bounded is **not known to be finite** in our class (heuristically
  it appears bounded, but no rigorous quantitative bound is known).

**Direct rearrangement does not help** — see
[`delsarte_dual/ideas_rearrangement.md`](../delsarte_dual/ideas_rearrangement.md):
symmetric-decreasing rearrangement strictly increases $\|f * f\|_\infty$,
so we cannot WLOG assume symmetry.

If a rigorous bound $\|f\|_2^2 \le \alpha^*\,\|f*f\|_\infty$ with $\alpha^* < (M-1)/(\mu(M)\,M) + 1/M$ at $M = M_{\max}$ — for
the candidate $M_{\max} = 1.378$, this is $\alpha^* < 1.549$ — could be
established, the chain ($\dagger$) would close for asymmetric $f$ as well, and
Tier 1 / Tier 2 would follow.

## 6. References

[CS17] A. Cloninger, S. Steinerberger, *On suprema of autoconvolutions, with an application to Sidon sets*, Proc. Amer. Math. Soc. **145** (2017), 3191–3200; arXiv:1403.7988.

[MO04] G. Martin, K. O'Bryant, *The symmetric subset problem in continuous Ramsey theory*, Experimental Math. **16** (2007), 145–165; arXiv:math/0410004.

[MV10] M. Matolcsi, C. Vinuesa, *Improved bounds on the supremum of autoconvolutions*, J. Math. Anal. Appl. **372** (2010), 439–447; arXiv:0907.1379.

[BL25] A. Boyer, X. Li, *On a conjecture of Martin and O'Bryant*, arXiv:2506.16750 (2025). [Disproves the unrestricted form of MO 2004 Conjecture 2.9.]

## 7. Map to repository

| File | Purpose |
|---|---|
| [`delsarte_dual/path_a_unconditional_holder/derivation.md`](../delsarte_dual/path_a_unconditional_holder/derivation.md) | Detailed derivation incl. all six attacks tried |
| [`delsarte_dual/path_a_unconditional_holder/holder_constant.py`](../delsarte_dual/path_a_unconditional_holder/holder_constant.py) | Implementation: provable $c^*$ and $M^*_{\mathrm{sym}}$ |
| [`delsarte_dual/restricted_holder/conditional_bound.py`](../delsarte_dual/restricted_holder/conditional_bound.py) | The pre-existing conditional theorem $\to$ $M_{\mathrm{optimal}}(c)$ |
| [`tests/test_path_a_unconditional.py`](../tests/test_path_a_unconditional.py) | 17 acceptance tests, all passing |
| [`delsarte_dual/ideas_rearrangement.md`](../delsarte_dual/ideas_rearrangement.md) | Why rearrangement cannot reduce asymmetric to symmetric |
