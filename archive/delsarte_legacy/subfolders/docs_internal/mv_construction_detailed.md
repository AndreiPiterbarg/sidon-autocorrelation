# Matolcsi–Vinuesa (2010) — Detailed Extraction of the Dual Construction

**Paper:** M. Matolcsi, C. Vinuesa, *Improved bounds on the supremum of autoconvolutions*,
J. Math. Anal. Appl. **372** (2010) 439–447. arXiv:0907.1379 (v2, 4 Sep 2009, 15 pp.).

**Problem:** with $\mathcal{F}=\{f\ge 0:\operatorname{supp} f\subset[-1/4,1/4],\ \int f=1\}$,

$$
S \;=\; \inf_{f\in\mathcal{F}} \|f*f\|_\infty.
$$

Their result (stated on p. 2 as the boxed improvement): $1.2748\le S\le 1.5098$; the lower
bound is Theorem 3.2.

All equation numbers below refer to the MV paper unless noted. The source file
`delsarte_dual/mv_paper.txt` (extracted via `pdftotext -layout` from arXiv:0907.1379v2)
is the authoritative reference; line numbers are given where useful.

---

## 1. The exact dual "master inequality"

The argument is copied (with sharpening) from Martin–O'Bryant [6] = arXiv:0807.5121
which itself builds on Yu [7]. MV state the ingredients as:

**Lemma 3.1 (quoting Lemmas 3.1–3.4 of [6]; MV paper p. 3, eqs. (1)–(4)).**
With $K$ a nonnegative kernel with $\operatorname{supp} K\subset[-\delta,\delta]$, $\int K=1$,
$\widetilde K(j)\ge 0$ for every integer $j$, $0<\delta\le 1/4$, $u=1/2+\delta$, and the
period-$u$ Fourier coefficients
$\widetilde g(\xi)=\tfrac{1}{u}\int g(x)e^{-2\pi i x\xi/u}\,dx$:

- **(1)** $\displaystyle\int(f*f)(x)\,K(x)\,dx \;\le\; \|f*f\|_\infty.$
- **(2)** $\displaystyle\int(f*f)(x)\,K(x)\,dx \;\ge\; 1+\sqrt{\|f*f\|_\infty-1}\cdot\sqrt{\|K\|_2^2-1}.$
  (This is the "Parseval + Cauchy–Schwarz" identity of Yu/MO, exploiting the fact that $f*f$
  and $K$ are both probability densities and applying Cauchy–Schwarz to their mean-zero parts.)
- **(3)** $\displaystyle\int\!\bigl(f*f(x)+f\circ f(x)\bigr)K(x)\,dx
   \;=\; \frac{2}{u}\;+\;2u^2\!\sum_{j\ne 0}\bigl(\widetilde f(j)\bigr)^2\,\widetilde K(j).$
  Here $f\circ f(x)=\int f(t)f(x+t)\,dt$ is the autocorrelation.
- **(4)** *Key duality bound.* Let $G$ be an even, real-valued, $u$-periodic function,
  positive on $[-1/4,1/4]$, with $\widetilde G(0)=0$. Then

  $$
  u^2\sum_{j\ne 0}\bigl(\widetilde f(j)\bigr)^2\widetilde K(j)
  \;\ge\; \Bigl(\min_{0\le x\le 1/4} G(x)\Bigr)^2
          \cdot\Biggl(\sum_{j:\widetilde G(j)\ne 0}\frac{\widetilde G(j)^2}{\widetilde K(j)}\Biggr)^{-1}.
  $$

  (This is a Cauchy–Schwarz / duality inequality: the weighted $\ell^2$ sum of
  $\widetilde f(j)^2$ against $\widetilde K(j)$ is bounded below by pairing it with $\widetilde G$.
  See MV p. 3, lines 124–134.)

**Derivation of the master inequality (MV p. 3, eq. (6)).** Combining (1), (2), (3), (4):

$$
\boxed{\;
\|f*f\|_\infty \;+\; 1 \;+\; \sqrt{\|f*f\|_\infty-1}\cdot\sqrt{\|K\|_2^2-1}
\;\ge\; \frac{2}{u}\;+\;2\,\bigl(\min_{0\le x\le 1/4}G(x)\bigr)^2\!
\cdot\!\Biggl(\sum_{j:\widetilde G(j)\ne 0}\frac{\widetilde G(j)^2}{\widetilde K(j)}\Biggr)^{-1}.
\;}
\quad\text{(MV eq. (6))}
$$

Solving this quadratic for $\|f*f\|_\infty$ yields a numerical lower bound. MO [6] with
$\delta=0.13$, $u=0.63$, $G=$ Selberg $G_{0.63,22}$ get $S\ge 1.262$ (MV p. 3, line 160).

**MV's sharpening (eq. (7), p. 4).** With the explicit $K$ of (5) — see §2 below —
one has $\|K\|_2^2 < 0.5747/\delta$ (MV p. 3 line 141; p. 4 line 185) and
$\widetilde K(j)=\tfrac{1}{u}|J_0(\pi j\delta/u)|^2$ (MV p. 4 lines 184–187). With
$G(x)=\sum_{j=1}^n a_j\cos(2\pi jx/u)$ one has $\widetilde G(\pm j)=a_{|j|}/2$ for
$1\le|j|\le n$, zero otherwise, hence $\widetilde G(j)^2/\widetilde K(j)=
(a_{|j|}^2/4)\cdot u/|J_0(\pi j\delta/u)|^2$, and

$$
\boxed{\;
\|f*f\|_\infty + 1 + \sqrt{\|f*f\|_\infty-1}\cdot\sqrt{0.5747/\delta-1}
\;\ge\; \frac{2}{u}+\frac{4}{u}\cdot
\Bigl(\min_{0\le x\le 1/4}G(x)\Bigr)^2\cdot
\Biggl(\sum_{j=1}^n \frac{a_j^2}{|J_0(\pi j\delta/u)|^2}\Biggr)^{-1}.
\;}\quad\text{(MV eq. (7))}
$$

MV introduce the *gain-parameter* (MV p. 4 lines 206–210)

$$
a \;=\; \frac{4}{u}\,\bigl(\min_{0\le x\le 1/4}G(x)\bigr)^2\,
\Biggl(\sum_{j=1}^n \frac{a_j^2}{|J_0(\pi j\delta/u)|^2}\Biggr)^{-1},
$$

so that (7) reads $\|f*f\|_\infty+1+\sqrt{\|f*f\|_\infty-1}\sqrt{0.5747/\delta-1}
\ge 2/u + a$. Record: $a\approx 0.0342$ for MO's [6] $(\delta,G)=(0.13,\;G_{0.63,22})$
(MV p. 4 line 212), and MV achieve $a>0.0713$ at $(\delta,n)=(0.138,119)$ (MV p. 4 line 224).

**Further sharpening using $|\hat f(1)|$ (MV eqs. (9), (10), pp. 5–7).** Writing
$z_1=|\hat f(1)|$, $k_1=\hat K(1)=\hat K(-1)$ (period-1 Fourier coefficients) and using the
decomposition $\int(f*f)K = 1 + 2z_1^2 k_1 + \sum_{j\ne 0,\pm 1}|\hat f(j)|^2 \hat K(j)$
with Cauchy–Schwarz on the tail (Lemma 3.3, MV p. 5 line 269):

$$
\int(f*f)(x)K(x)\,dx \;\ge\; 1 + 2z_1^2 k_1 + \sqrt{\|f*f\|_\infty - 1 - 2z_1^4}\;\sqrt{\|K\|_2^2-1-2k_1^2}.
\quad\text{(MV eq. (9))}
$$

Lemma 3.4 (MV p. 5–6, lines 302–305): if $0\le h\le M$, $\int h=1$, supported on
$[-1/2,1/2]$, then $|\hat h(1)|\le (M/\pi)\sin(\pi/M)$. Applied to $h=f*f$ with
$M=\|f*f\|_\infty$ (assuming $M<1.2748$ for contradiction) gives $|\hat f(1)|^2=
|\widehat{f*f}(1)|\le M\sin(\pi/M)/\pi<0.50426$ (MV p. 6 line 336).

The final polished inequality (MV eq. (10), p. 7 lines 342–347):

$$
\boxed{\;
\tfrac{2}{u} + a \;\le\; \|f*f\|_\infty + 1 + 2 z_1^2 k_1 + \sqrt{\|f*f\|_\infty-1-2z_1^4}\cdot\sqrt{0.5747/\delta-1-2k_1^2}.
\;}
$$

Plugging $\delta=0.138$, $k_1=|J_0(\pi\delta)|^2$ (MV p. 7 line 348), $a=0.0713$, and
choosing $z_1=0.50426$ (the boundary of the allowed range) gives
$\|f*f\|_\infty\ge 1.27481$ (MV p. 7 line 353). QED.

---

## 2. The EXPLICIT test function — answering the questions in the brief

### The kernel $K$ (MV eq. (5), p. 3 lines 138–143)

$$
K(x)=\frac{1}{\delta}\,\eta\!\left(\frac{x}{\delta}\right),\qquad
\eta(x)=\frac{2/\pi}{\sqrt{1-4x^2}}\ \text{ on }(-\tfrac12,\tfrac12),\ \text{zero elsewhere.}
$$

$K$ is a probability density supported on $[-\delta,\delta]$. **Note:** MV call this $\eta$,
whereas the task brief called it $\beta$ — they are the same function.

**This is NOT an auto-convolution.** $K$ is the rescaled arcsine density itself,
not $\eta*\eta$. This answers question (b) partially: $\eta$ is indeed a probability
density on $(-1/2,1/2)$ (MV p. 3 line 140: $\eta(x)=2/\pi/\sqrt{1-4x^2}$ on $(-1/2,1/2)$),
and $\int\eta=1$.

**Important — the quantity $\|K\|_2^2$.** MV write "$\|K\|_2^2 < 0.5747/\delta$"
(p. 3 line 141). Direct calculation:
$\|K\|_2^2=\delta^{-1}\int_{-1/2}^{1/2}\eta(x)^2\,dx = \delta^{-1}\cdot\infty$ ... actually
$\eta(x)^2=4/(\pi^2(1-4x^2))$ has a logarithmic divergence, so $K\notin L^2$.
This means MV are using $\|K\|_2^2$ in some *regularised* sense. Reading the argument
more carefully: the "0.5747" is the authors' computed value of a *different* quadratic
form — plausibly $\sum_{j\ne 0}\widetilde K(j)^2$ or the Parseval sum — that appears in
the Cauchy–Schwarz bound (2). **I am not fully sure** whether "$\|K\|_2^2$" means
$\sum_j\hat K(j)^2$ (period-1 Fourier) or the $L^2$ integral with a subtraction of the
diagonal; the paper does not spell this out and relies on [6] for the derivation of (2).

### The Fourier transform of $\eta$ and $K$ (MV p. 4 lines 184–187)

MV state explicitly:
$$
\widetilde K(j) \;=\; \frac{1}{u}\,|J_0(\pi j\delta/u)|^2,
$$
where $J_0$ is the Bessel function of the first kind of order $0$. This is the
period-$u$ Fourier coefficient of $K$. The identity comes from the classical
formula $\int_{-1/2}^{1/2}\frac{2/\pi}{\sqrt{1-4x^2}}e^{-2\pi i\xi x}\,dx=J_0(\pi\xi)$,
then scaling $x\mapsto x/\delta$ and evaluating at frequency $j/u$. This answers
question (c).

MV also use $\hat K(j)$ with period-1 Fourier coefficients:
MV p. 5–6 Lemma 3.3 uses $k_1=\hat K(1)=\hat K(-1)$, and p. 7 line 348 uses
$k_1=|J_0(\pi\delta)|^2$. So in period-1 convention $\hat K(1)=|J_0(\pi\delta)|^2$.
**Remark:** There is a mild tension: in period-$u$ conventions $\widetilde K(j)=
(1/u)|J_0(\pi j\delta/u)|^2$ (MV p. 4), whereas MV's period-1 $\hat K(1)$ in p. 7
is $|J_0(\pi\delta)|^2$. The two are consistent only under the normalisations MV use
(the $1/u$ factor is absorbed in the definition of $\widetilde \cdot$); both statements
come from MV, so we follow them.

Also (MV p. 5 lines 247–248): $\widetilde K(j)=u(\widetilde \eta(j))^2$ where
$\tilde\eta(x)=\frac{1}{\delta}\eta(x/\delta)\cdot(\text{rescaling})$ — this is used
in the "theoretical-limit" calculation of §3 below.

### How $G$ and $K$ combine — answer to question (a)

They are **not** combined by multiplication, addition, convolution, or tensor product.
They enter the *same scalar inequality* (MV eq. (6) above) in **different slots**:

- $K$ supplies $\|K\|_2^2$, $\widetilde K(j)$, and $\hat K(1)=k_1$.
- $G$ supplies $\min_{[0,1/4]}G$ and the Fourier coefficients $\widetilde G(j)$, which
  combine with $\widetilde K(j)$ via the Cauchy–Schwarz dual in (4).

The test "object" is the pair $(K,G)$. There is no bivariate kernel $B(x,y)$ and no
product or convolution of $G$ and $K$. $G$ is a period-$u$ trigonometric polynomial on
$\mathbb R$; $K$ is a compactly supported density on $[-\delta,\delta]$. The inequality
(6) is *bilinear* in their respective Fourier coefficients.

### The explicit $G$ (MV p. 4 lines 191–194, and Appendix)

$$
G(x)\;=\;\sum_{j=1}^{n} a_j\cos(2\pi j x / u),\qquad n=119,\ u=0.638,\ \delta=0.138.
$$

Because each cosine has mean zero, $\widetilde G(0)=0$ automatically (satisfying the
hypothesis of Lemma 3.1(4)). For $j=\pm 1,\ldots,\pm n$ we get $\widetilde G(j)=a_{|j|}/2$;
all other $\widetilde G(j)=0$ (MV p. 4 lines 192–195).

### The arcsine role — answer to question (b)

- $\eta$ is a probability density on $(-1/2,1/2)$: $\int\eta=1$ (MV p. 3 line 140).
- $\eta$ is chosen because its Fourier transform is $J_0$, which is **positive-definite
  and explicit**, giving the clean $\widetilde K(j)=(1/u)|J_0(\pi j\delta/u)|^2\ge 0$
  (positivity of $\widetilde K(j)$ is required for Lemma 3.1; MV p. 2 line 100).
- MV note "We are quite convinced that the choice of $K$ in [6] is optimal, and we
  will not change it" (MV p. 2 lines 101–103) — i.e. they inherit $K$ from Martin–O'Bryant
  and **do not optimise over $K$**. The QP only optimises $G$.
- In the "theoretical-limit" argument (§3 below) they use the identity
  $\widetilde K(j)=u\widetilde\eta(j)^2$ (MV p. 5 lines 247–248) which expresses $K$ as
  related to the autocorrelation of $\eta$ in Fourier space — but $K$ itself is the
  single arcsine, not its auto-convolution. This is likely the source of confusion in
  the task brief.

### The QP and its constraint — answer to question (d) (MV p. 4 lines 213–228)

Quoting MV literally (p. 4 lines 213–215):
> "For any fixed $\delta$ we are therefore led to the problem of maximizing $a$
> (while we may as well assume that $\min_{0\le x\le 1/4}G(x)\ge 1$, as $G$ can be
> multiplied by any constant without changing the gain $a$)."

So the QP is (in one equivalent form):

$$
\boxed{\;
\min_{(a_1,\ldots,a_n)\in\mathbb R^n}\ \sum_{j=1}^n \frac{a_j^2}{|J_0(\pi j\delta/u)|^2}
\quad\text{subject to}\quad \min_{0\le x\le 1/4}\;\sum_{j=1}^n a_j\cos(2\pi j x/u)\;\ge\;1.
\;}
$$

This is a **semi-infinite QP** (finitely many variables, infinitely many linear
constraints, one per $x\in[0,1/4]$). MV solve it in Mathematica 6; presumably by
discretising $[0,1/4]$ to a fine grid. They report that at $\delta=0.138$, $n=119$ the
attained gain is $a>0.0713$.

### The role of 1.2748 — answer to question (e)

The number $1.2748$ is:
- **The achieved value** of the dual bound with the specific $(δ,u,n,\{a_j\})$ and
  with the $z_1$-improvement of Lemma 3.3/3.4. MV p. 4 line 228: using just eq. (7)
  (no $z_1$ improvement) gives $S\ge 1.2743$; then with the $z_1$ improvement (eq. (10))
  they get $S\ge 1.27481$ which they round down to $1.2748$.
- **Not a theoretical ceiling.** The theoretical ceiling is $\approx 1.276$ (see §3).

---

## 3. Positive-definiteness / admissibility certificate

The admissibility conditions MV need (p. 2 lines 100–103, p. 3 lines 124–125):

1. $K\ge 0$, $\operatorname{supp} K\subset[-\delta,\delta]$, $\int K=1$, and
   **$\widetilde K(j)\ge 0$** for every integer $j$. This is automatic for MV's choice
   because $\widetilde K(j)=(1/u)|J_0(\pi j\delta/u)|^2\ge 0$ — the Bessel-squared
   representation immediately gives non-negativity (no Bochner argument needed).
2. $G$ is even, real-valued, $u$-periodic, **positive on $[-1/4,1/4]$**, and
   $\widetilde G(0)=0$. MV's $G$ is manifestly even, real, $u$-periodic (sum of cosines);
   $\widetilde G(0)=0$ is automatic; positivity on $[-1/4,1/4]$ is the constraint
   $\min_{0\le x\le 1/4}G(x)\ge 1>0$ in the QP.

**Crucially, MV do NOT require $G$ to be positive-definite** (no Bochner / $\hat G\ge 0$
condition). They only need $G>0$ pointwise on $[-1/4,1/4]$. The non-negativity in the
dual comes from $\widetilde K(j)\ge 0$, not $\widetilde G(j)\ge 0$; the Fourier
coefficients $a_j$ of $G$ are signed.

**Why the bound is valid.** Given the admissibility above, (4) follows by Cauchy–Schwarz:
$$
\Bigl(\sum_j \widetilde f(j)\widetilde G(j)\Bigr)^2
\le \Bigl(\sum_j \widetilde f(j)^2\widetilde K(j)\Bigr)
    \Bigl(\sum_j \widetilde G(j)^2/\widetilde K(j)\Bigr),
$$
and the left side is bounded below by $(\min_{[0,1/4]}G)^2\cdot(\text{something involving }
\int f\cdot\mathbf 1_{[-1/4,1/4]})$. The precise derivation is in MO [6] (Lemma 3.4 there);
MV do not reprove it.

---

## 4. Table of the 119 coefficients — present in Appendix (online only)

The Appendix (MV pp. 12–13, lines 608–677) lists the 119 values of $a_j$ in reading order
$a_1,a_2,\ldots,a_{119}$. **I transcribe them verbatim** (8-digit floats, in scientific
notation where given). They are the coefficients of $G(x)=\sum_{j=1}^{119} a_j\cos(2\pi j x/0.638)$
at $\delta=0.138$.

```
 j          a_j
 1   +2.16620392e+00
 2   -1.87775750e+00
 3   +1.05828868e+00
 4   -7.29790538e-01
 5   +4.28008515e-01
 6   +2.17832838e-01
 7   -2.70415201e-01
 8   +2.72834790e-02
 9   -1.91721888e-01
10   +5.51862060e-02
11   +3.21662512e-01
12   -1.64478392e-01
13   +3.95478603e-02
14   -2.05402785e-01
15   -1.33758316e-02
16   +2.31873221e-01
17   -4.37967118e-02
18   +6.12456374e-02
19   -1.57361919e-01
20   -7.78036253e-02
21   +1.38714392e-01
22   -1.45201483e-04
23   +9.16539824e-02
24   -8.34020840e-02
25   -1.01919986e-01
26   +5.94915025e-02
27   -1.19336618e-02
28   +1.02155366e-01
29   -1.45929982e-02
30   -7.95205457e-02
31   +5.59733152e-03
32   -3.58987179e-02
33   +7.16132260e-02
34   +4.15425065e-02
35   -4.89180454e-02
36   +1.65425755e-03
37   -6.48251747e-02
38   +3.45951253e-02
39   +5.32122058e-02
40   -1.28435276e-02
41   +1.48814403e-02
42   -6.49404547e-02
43   -6.01344770e-03
44   +4.33784473e-02
45   -2.53362778e-04
46   +3.81674519e-02
47   -4.83816002e-02
48   -2.53878079e-02
49   +1.96933442e-02
50   -3.04861682e-03
51   +4.79203471e-02
52   -2.00930265e-02
53   -2.73895519e-02
54   +3.30183589e-03
55   -1.67380508e-02
56   +4.23917582e-02
57   +3.64690190e-03
58   -1.79916104e-02
59   +7.31661649e-05
60   -2.99875575e-02
61   +2.71842526e-02
62   +1.41806855e-02
63   -6.01781076e-03
64   +5.86806100e-03
65   -3.32350597e-02
66   +9.23347466e-03
67   +1.47071722e-02
68   -7.42858080e-04
69   +1.63414270e-02
70   -2.87265671e-02
71   -1.64287280e-03
72   +8.02601605e-03
73   -7.62613027e-04
74   +2.18735533e-02
75   -1.78816282e-02
76   -6.58341101e-03
77   +2.67706547e-03
78   -6.25261247e-03
79   +2.24942824e-02
80   -8.10756022e-03
81   -5.68160823e-03
82   +7.01871209e-05
83   -1.15294332e-02
84   +1.83608944e-02
85   -1.20567880e-03
86   -3.13147456e-03
87   +1.39083675e-03
88   -1.49312478e-02
89   +1.32106694e-02
90   +1.73474188e-03
91   -8.53469045e-04
92   +4.03211203e-03
93   -1.55352991e-02
94   +8.74711543e-03
95   +1.93998895e-03
96   -2.71357322e-05
97   +6.13179585e-03
98   -1.41983972e-02
99   +5.84710551e-03
100  +9.22578333e-04
101  -2.16583469e-04
102  +7.07919829e-03
103  -1.18488582e-02
104  +4.39698322e-03
105  -8.91346785e-05
106  -3.42086367e-04
107  +6.46355636e-03
108  -8.87555371e-03
109  +3.56799654e-03
110  -4.97335419e-04
111  -8.04560326e-04
112  +5.55076717e-03
113  -7.13560569e-03
114  +4.53679038e-03
115  -3.33261516e-03
116  +2.35463427e-03
117  +2.04023789e-04
118  -1.27746711e-03
119  +1.81247830e-04
```

Source: MV Appendix, PDF p. 12–13. The values are exactly as extracted; a machine-readable
copy is present at `delsarte_dual/mv_paper.txt` lines 619–677.

**First 20:** $a_1,\ldots,a_{20}$ as above.
**Last 20:** $a_{100},\ldots,a_{119}$ as above.

Note the coefficients are signed (no Bochner positivity required) and decay roughly
exponentially in magnitude.

---

## 5. Numerical verification formula

To reproduce the bound $1.2748$ numerically from the tabulated $a_j$:

**Step 1.** Fix $\delta=0.138$, $u=0.638$, $n=119$.

**Step 2.** Compute
$$
S_1 \;=\; \sum_{j=1}^{119}\frac{a_j^2}{|J_0(\pi j\delta/u)|^2},
\qquad
m \;=\; \min_{0\le x\le 1/4}\sum_{j=1}^{119}a_j\cos(2\pi j x/u).
$$
By construction (the QP's feasibility constraint) $m\ge 1$. MV achieve
$a=(4/u)\cdot m^2/S_1 > 0.0713$ (MV p. 4 line 224).

**Step 3.** Compute $k_1=|J_0(\pi\delta)|^2 = |J_0(\pi\cdot 0.138)|^2$ (MV p. 7 line 348).

**Step 4.** Compute the "forbidden range" for $z_1$: from Lemma 3.4, if
$M=\|f*f\|_\infty=1.2748$ then $|\hat f(1)|\le\sqrt{M\sin(\pi/M)/\pi}<0.50426$
(MV p. 6 line 336, p. 7 lines 350–352).

**Step 5.** The "master" eq. (10) becomes
$$
\frac{2}{u}+a \;\le\; M + 1 + 2z_1^2 k_1
              + \sqrt{M-1-2z_1^4}\cdot\sqrt{0.5747/\delta - 1 - 2k_1^2}.
$$
Set $l(z_1)=$ the smallest $M$ satisfying this (monotonically decreasing in $z_1$ on
$[0,0.50426]$, MV p. 7 lines 349–353). Plug in $z_1=0.50426$:

$$
M \;=\; l(0.50426) \;=\; 1.27481\ldots,
$$

and round down: $S\ge 1.2748$.

**Check / smoke test for re-implementation.** A correct transcription of the $a_j$ should
give:
- $\sum_{j=1}^{119} a_j^2/|J_0(\pi j\delta/u)|^2\ \approx\ 4m^2/(u\cdot 0.0713)
   \approx 4\cdot 1^2/(0.638\cdot 0.0713)\approx 87.9$ (if $m\approx 1$);
- $\min_{[0,1/4]}G(x)\approx 1$ (active constraint in QP);
- $(2/u)+a \approx 3.135+0.0713\approx 3.206$.

---

## 6. Potential improvements / open questions (MV's own remarks)

### The $1.276$ ceiling (MV Remark on p. 5, lines 232–261)

MV's "theoretical limit of the argument" is $\approx 1.276$. Derivation:

- Use eq. (3) rewritten (MV eq. (8), p. 5 line 245) via Parseval and the identity
  $\widetilde K(j)=u(\widetilde\eta(j))^2$ with $\eta_\delta(x)=(1/\delta)\eta(x/\delta)$:
  $$
  \int\!\bigl((f*f)(x)+(f\circ f)(x)\bigr)K(x)\,dx \;=\; 2\|f_s*\eta_\delta\|_2^2,
  $$
  where $f_s(x)=\tfrac12(f(x)+f(-x))$ is the symmetrisation.
- For fixed $\eta_\delta$, the best lower bound the argument can possibly give for the
  RHS is $\inf_{f_s}\|f_s*\eta_\delta\|_2^2$, taken over nonnegative symmetric $f_s$ with
  $\int f_s=1$.
- Discretise $\eta_\delta$ and $f_s$ into step functions; the infimum becomes a convex
  (non-negative definite) multivariate QP in the step heights of $f_s$.
- MV solved this numerically for several $\delta$; best is $\delta\approx 0.14$ giving
  $\|f*f\|_\infty\ge 1.276$.

**Moral:** even with the *optimal* cosine polynomial $G$ (not just their 119-term
approximation), the Yu / MO / MV dual framework cannot exceed $\approx 1.276$.
To break this ceiling one must change the *framework* (e.g. Cloninger–Steinerberger
branch-and-prune, or a different SDP hierarchy), not just push the QP further.

MV note (p. 5 line 258): "all this could be done rigorously, but one needs to control
the error arising from the discretization, and the sheer documentation of it is simply
not worth the effort, in view of the minimal gain."

### Further suggested directions (MV Remark on p. 7 lines 356–380)

MV identify three open directions:
1. **Sharper Lemma 3.4.** The current Lemma 3.4 only uses $h\ge 0$, $\int h=1$,
   $h\le M$. Exploiting that $h=f*f$ (and not just any nonneg function) could give a
   tighter bound on $|\hat f(1)|$.
2. **Union of forbidden sets.** Each $(\delta,K,G)$ triple produces a forbidden interval
   $F=\{x:l(x)<s_0\}$ for $z_1=|\hat f(1)|$. If two triples produce disjoint $F_1,F_2$
   covering $[0,\|f*f\|_\infty^{1/2}]$, the bound follows automatically. MV's single
   triple gives $F=(0.504433,\,0.529849)$ (MV p. 7 line 369).
3. **Higher Fourier coefficients.** Extend Lemma 3.3 to pull out $z_1,z_2,\ldots$ and
   analyse $l(z_1,z_2,\ldots)$.

### Why they stopped at $n=119$

Not stated explicitly. MV's $n=119$ gives gain $a>0.0713$, yielding $S\ge 1.27481$
after the $z_1$-improvement. Pushing $n\uparrow$ or $\delta$ differently would raise $a$
toward the ceiling $\approx 1.276$ but cannot exceed it (per the argument above).
Mathematica 6 QP performance on a ~119-dimensional semi-infinite QP at 2009 hardware is
a plausible practical ceiling.

---

## 7. Summary of remaining questions blocking a correct re-implementation

**Answered from the PDF:**
- Functional form of $G$, $K$, $\eta$: explicit (§2).
- Combination rule: neither product, sum, convolution, nor tensor — $G$ and $K$ enter
  different slots of the scalar inequality (6); see §2 "How $G$ and $K$ combine".
- QP objective: $\min \sum a_j^2/|J_0(\pi j\delta/u)|^2$. Constraint: $\min_{[0,1/4]}G\ge 1$.
- Positive-definiteness: only $\widetilde K(j)\ge 0$ (automatic from $J_0^2$), and $G>0$
  pointwise on $[-1/4,1/4]$. No Bochner condition on $G$.
- Coefficients: all 119 $a_j$ transcribed above.
- $1.2748$ vs. $1.276$: achieved vs. theoretical ceiling; MV are not tight to the ceiling.
- Verification formula: eqs. (7)+(10) together with the $z_1=0.50426$ boundary.

**Still unclear / not fully spelled out in MV:**

(i) The precise meaning of "$\|K\|_2^2 < 0.5747/\delta$" for the non-$L^2$ kernel $K$.
    $K$ is the (rescaled) arcsine density which is **not** square-integrable
    (logarithmic endpoint singularity). MV inherit this from Martin–O'Bryant [6]; the
    bound is likely a regularised Parseval sum $\sum_{j\ne 0}\widetilde K(j)^2$ or
    $\sum_j\hat K(j)^2$ truncated appropriately. For a clean re-implementation one
    must read MO [6] Lemma 3.2 carefully. **I am not fully sure** which norm is meant.

(ii) The precise reconciliation of period-$u$ vs. period-1 Fourier normalisations in
     the identity $k_1=\hat K(1)=|J_0(\pi\delta)|^2$ vs. $\widetilde K(j)=(1/u)|J_0(\pi j\delta/u)|^2$.
     Both statements are from MV (p. 4 line 185 and p. 7 line 348). This likely requires
     the explicit Poisson/Fourier-series identities in MO [6].

(iii) The discretisation grid on $[0,1/4]$ used inside the Mathematica QP (how fine,
      and whether feasibility is checked rigorously or numerically only). MV just say
      "Mathematica 6"; no grid size stated.

(iv) Whether MV's $0.0713$ lower bound on the gain is rigorous or numerical. The text
     says "we obtained … $a>0.0713$" (MV p. 4 line 224); it appears to be numerical
     (floating-point QP output), not a certified enclosure.

For a **rigorous** re-implementation producing $\ge 1.2748$ one must:
1. Resolve question (i) — the correct $\|K\|_2^2$ surrogate — by consulting MO [6].
2. Recompute $k_1$, $\|K\|_2^2$ surrogate, and $\widetilde K(j)$ in interval arithmetic.
3. Re-verify that MV's tabulated $a_j$ satisfy $\min_{[0,1/4]}G\ge 1$ (interval check
    on a fine grid, with Lipschitz bound for the remainder).
4. Evaluate $\sum a_j^2/|J_0(\pi j\delta/u)|^2$ in interval arithmetic and plug into
    (7)/(10) with $z_1=0.50426$.

These steps are mechanical once (i) is pinned down.
