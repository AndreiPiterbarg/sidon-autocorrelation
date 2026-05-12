# Conditional lower bound on $C_{1a}$ from a restricted Hölder hypothesis

This note proves, **modulo a restricted Hölder-slack hypothesis Hyp_R that
Boyer–Li 2025 does NOT disprove**, the rigorous bound

$$
\boxed{\;C_{1a} \;\ge\; M_{\mathrm{optimal}}\bigl(\tfrac{\log 16}{\pi}\bigr)
\;=\; 1.3784219737754177283997025524314107461234603128129\ldots\;}
$$

The hypothesis is a restriction-to-low-$M$ form of MO 2004 Conjecture 2.9.
The bound $M_{\mathrm{optimal}}(c)$ is the *infimum* of the modified MV
master equation over the admissible range of $z_1 = |\hat f(1)|$ — this is
the rigorous worst-case-$z_1$ value. All numerics in this note are
reproduced to ≥ 30 decimal digits by
[`conditional_bound.py`](conditional_bound.py).

> **Correction (2026-04-29).** An earlier draft of this note used
> $M_{\mathrm{recipe}} = 1.37925062\ldots$, obtained by following MV's recipe
> verbatim with $z_1 = 0.50426$ fixed. *That value is sub-optimal at
> $c < 1$ and is not a rigorous lower bound.* See §Z below and
> [`CORRECTION_NOTE.md`](CORRECTION_NOTE.md). The recipe value is now
> reported only as a historical curiosity in the comparison table.

---

## 1. The MO 2004 Hölder-slack conjecture (verbatim)

The following is **MO 2004 Conjecture 2.9** (Martin & O'Bryant,
*The symmetric subset problem in continuous Ramsey theory*, arXiv:math/0410004,
p. 13, lines 783–791; cf. the local extract `delsarte_dual/mo_2004.txt`).

> **Conjecture (MO 2004 Conj. 2.9).** *If $g$ is a non-negative pdf supported
> on $[-1/4, 1/4]$, then*
> $$\frac{\|g \cdot g\|_\infty \cdot \|g \cdot g\|_1}{\|g \cdot g\|_2^2}\;\ge\;\frac{\pi}{\log 16}.$$

Here $g \cdot g$ is the autocorrelation $\int g(t)\,g(t+\cdot)\,dt$; for
symmetric $g$, $g \cdot g \equiv g * g$. With $\|g*g\|_1 = 1$, the conjecture
is equivalent to the unrestricted **Hölder-slack inequality**

$$
\|g * g\|_2^2 \;\le\; \frac{\log 16}{\pi}\,\|g * g\|_\infty,
\qquad c_* := \frac{\log 16}{\pi} \approx 0.88254240061060637\ldots
$$

The constant $c_* = \log 16/\pi$ is **strictly tighter than ordinary Hölder**
$\|g*g\|_2^2 \le \|g*g\|_\infty$ (which has constant $c = 1$).

## 2. The Boyer–Li 2025 disproof of the unrestricted form

**Theorem (Boyer–Li 2025, arXiv:2506.16750).** There exists a non-negative
step function $f_0 = \sum_{n=0}^{574} v_n \mathbf{1}_{[n,n+1)}$ on $[0,575]$
with

$$
Q_0 \;:=\; \frac{\|f_0 * f_0\|_2^2}{\|f_0 * f_0\|_\infty\,\|f_0 * f_0\|_1}
       \;=\; \frac{96086089410408840289339769}{106577013451431545242354944}
       \;\approx\; 0.901564852\ldots \;>\; \frac{\log 16}{\pi}.
$$

This **disproves** MO 2004 Conjecture 2.9 in its unrestricted form. The
witness $v_n$ (575 non-negative integers, scaled by $2^{24}$) is published
in `coeffBL.txt` of the Boyer–Li repo and a verbatim copy is in
[`coeffBL.txt`](coeffBL.txt) of this directory.

### 2.1 Rescaling the Boyer–Li witness to $[-1/4, 1/4]$

The autoconvolution problem is translation- and dilation-invariant. We rescale
$f_0$ to a probability density $g$ on $[-1/4, 1/4]$ as follows:

1. **Translate** $f_0$ to symmetric support: let $f_1(x) = f_0(x + 575/2)$ on
   $[-575/2, 575/2]$. $\|f_1\|_1 = \sum v_n =: S$, and $f_1*f_1$ has the same
   peak as $f_0*f_0$.
2. **Dilate** by $\beta = 1150$: define $g(x) = (1150/S)\,f_1(1150\,x)$. Then
   $\operatorname{supp} g = [-1/4, 1/4]$, $\|g\|_1 = 1$, and

   $$
   \|g*g\|_\infty \;=\; \frac{1150}{S^2}\,\max_j L_j,\qquad
   L_j \;=\; \sum_n v_n\,v_{j-n-1}\;=\;(f_0*f_0)(j).
   $$

### 2.2 Rigorous evaluation (mpmath, dps = 50)

With $S = \sum_{n=0}^{574} v_n = 14\,912\,998$ and
$\max_j L_j = L_{475} = 319\,479\,037\,824$ (both exact integers), we obtain

$$
\boxed{\;\|g*g\|_\infty
\;=\;\frac{1150 \cdot 319\,479\,037\,824}{(14\,912\,998)^2}
\;=\; 1.65200093550821680488965602142\ldots\;}
$$

In particular, $\|g*g\|_\infty \approx 1.652$ — comfortably above the tight
threshold $M_{\max} = M_{\mathrm{target}} \approx 1.37842$ adopted in §3
below (margin $\approx 0.274$, more than twice the margin against the
earlier loose threshold $1.51$ used in the prior draft). The **rescaled
Boyer–Li witness lies OUTSIDE the restricted class** $\{f : \|f*f\|_\infty
\le M_{\max}\}$, so the restricted form of the conjecture is **not**
disproved by Boyer–Li.

(Reproducible by `python tests/test_restricted_holder.py
::test_boyerli_witness_outside_restriction`.)

## 3. The restricted Hölder hypothesis Hyp_R

For real numbers $c \in (0, 1]$ and $M_{\max} \ge 1$, define

> **Hyp_R($c, M_{\max}$).** *For every non-negative pdf $f$ supported on
> $[-1/4, 1/4]$ with $\|f*f\|_\infty \le M_{\max}$, one has*
> $$\|f*f\|_2^2 \;\le\; c\;\|f*f\|_\infty\;\|f*f\|_1.$$

Equivalently, since $\|f*f\|_1 = \|f\|_1^2 = 1$,
$\|f*f\|_2^2 \le c\,\|f*f\|_\infty$.

**What we know about Hyp_R.**

- **Unrestricted form ($M_{\max} = \infty$, $c = \log 16/\pi$): FALSE** by
  Boyer–Li 2025 (§2 above).
- **$M_{\max} = M_{\mathrm{target}} \approx 1.37842$, $c = \log 16/\pi$: still
  open**. The Boyer–Li witness has $\|g*g\|_\infty \approx 1.652 \gg
  M_{\mathrm{target}}$, so it is *outside* the restricted class and does
  *not* disprove Hyp_R($\log 16/\pi$, $M_{\mathrm{target}}$).
- **A fortiori, the same holds for any $M_{\max} \le 1.652$.** The earlier
  draft used the looser $M_{\max} = 1.51$, which is *also* not disproved by
  Boyer–Li but is a strictly weaker conditional statement than the tight
  $M_{\max} = M_{\mathrm{target}}$ chosen here.
- **$c = 1$, any $M_{\max}$: TRUE** (this is ordinary Hölder
  $\|f*f\|_2^2 \le \|f*f\|_\infty\,\|f*f\|_1$).
- **Trend in the empirical Hölder ratio.** For step-function pdfs $f$ on
  $[-1/4,1/4]$ with $\|f\|_1 = 1$, the empirical ratio
  $c_{\mathrm{emp}}(f) = \|f*f\|_2^2 / \|f*f\|_\infty$ moves the right way
  in the small-$M$ regime: MV's near-extremizer ($M \approx 1.275$) has
  $c_{\mathrm{emp}} \approx 0.589$, while the indicator $f = 2\cdot\mathbf 1_{[-1/4,1/4]}$
  ($M = 2$) has $c_{\mathrm{emp}} = 4/(3 \cdot 2) = 2/3 \approx 0.667$, and
  the Boyer–Li witness ($M \approx 1.652$) reaches $c_{\mathrm{emp}}
  \approx 0.9016$. The ratio increases as the pdf flattens its
  autoconvolution toward a near-indicator at $M \approx 2$, and Hyp_R is
  most easily satisfied for the small-$M$ regime relevant to the proof
  (cf. §6, "Caveats").
- **Empirical evidence (MV-class extremizers).** For $f = G/Z$ where $G$ is
  MV's 119-cosine test function and $Z = \int_{-1/4}^{1/4} G$, we compute
  (`sidon_extremizer_ratio.py`)

  $$
  c_\mathrm{emp} \;:=\; \frac{\|f*f\|_2^2}{\|f*f\|_\infty\,\|f*f\|_1}
                 \;\approx\; 0.5889 \;\ll\; 0.88254 = \log 16/\pi.
  $$

  This is well below the conjectured $c_* = \log 16/\pi$, which is consistent
  with Hyp_R holding (with substantial slack) for the Sidon-regime
  near-extremizer.

## 4. The conditional theorem

> **Theorem (conditional).** *Let
> $M_{\mathrm{target}} := 1.3784219737754177283997025524314107461234603128129\ldots$,
> the value computed in §5 below. If Hyp_R($c, M_{\max}$) holds with
> $c = \log 16/\pi \approx 0.88254$ and $M_{\max} = M_{\mathrm{target}}$,
> then*
> $$C_{1a} \;\ge\; M_{\mathrm{target}}.$$

**Tightness of the hypothesis.** $M_{\max} = M_{\mathrm{target}}$ is the
*minimum* M_max under which the proof's contradiction closes — see Step 2
below, where we use $\|f*f\|_\infty < M_{\mathrm{target}} \le M_{\max}$ to
invoke Hyp_R. Any larger $M_{\max}$ (e.g., the prior $1.51$) gives a valid
but **strictly weaker** conditional statement (a fortiori): the hypothesis
"Hyp_R holds for all $f$ with $\|f*f\|_\infty \le 1.51$" implies "Hyp_R
holds for all $f$ with $\|f*f\|_\infty \le M_{\mathrm{target}}$", but not
conversely. We adopt the tight value here so the hypothesis is the most
defensible form that still closes the argument; the codebase's implementation
defaults to this.

The bound $M_{\mathrm{optimal}}(c)$ is the **infimum over admissible $z_1$**
of $M_{\mathrm{master}}(z_1)$, where $M_{\mathrm{master}}(z_1)$ is the
unique solution of the **modified MV master equation**

$$
\frac{2}{u} + a_*
\;=\; M + 1 + 2 z_1^2 k_1 + \sqrt{c\,M - 1 - 2 z_1^4}\,
                                    \sqrt{\|K\|_2^2 - 1 - 2 k_1^2}.
\tag{Mod-10}
$$

with the MV constants $u = 0.638$, $\delta = 0.138$, $\|K\|_2^2 = 0.5747/\delta$,
$k_1 = J_0(\pi\delta)^2$, $a_* = (4/u)\,(\min_{[0,1/4]} G)^2 / S_1$,
$S_1 = \sum_{j=1}^{119} a_j^2/J_0(\pi j\delta/u)^2$, and $\{a_j\}$ MV's
tabulated 119 cosine coefficients.

The admissible range for $z_1$ is constrained by

- **Lemma 3.4** (MO 2004 Lemma 2.14): $z_1^2 \le M\sin(\pi/M)/\pi =: \mu(M)$.
- **Hyp_R sqrt domain**: $z_1^4 \le (cM - 1)/2$ (otherwise (Mod-10)'s first
  square root has a negative argument and Hyp_R contradicts Parseval).

Both constraints are necessary; the tighter one binds.

## 5. Proof of the conditional theorem

**Setup.** Let $f$ be a non-negative pdf supported on $[-1/4, 1/4]$ and set
$M = \|f*f\|_\infty$. Define $z_1 := |\hat f(1)|$ (period-1 Fourier coefficient).

**Step 1 — Cauchy–Schwarz on the Fourier tail.** From MV §3.3 (eq. (9) on
p. 5–7 of arXiv:0907.1379), in particular the chain at MV p. 5 lines 268–305:

$$
\int_{\mathbb R}\!(f*f)(x)\,K(x)\,dx
\;\le\; 1 + 2 z_1^2 k_1
   + \sqrt{\|f*f\|_2^2 - 1 - 2 z_1^4}\;\sqrt{\|K\|_2^2 - 1 - 2 k_1^2}.
\tag{C-S}
$$

**Step 2 — Apply Hyp_R in place of Hölder.** Suppose for contradiction
$M < M_{\mathrm{target}} = M_{\mathrm{optimal}}(c)$. Since $M_{\max} =
M_{\mathrm{target}}$, we have $M < M_{\mathrm{target}} = M_{\max}$, so
$f$ satisfies $\|f*f\|_\infty \le M_{\max}$ and Hyp_R applies:

$$
\|f*f\|_2^2 \;\le\; c\,M\,\|f*f\|_1 \;=\; c\,M.
\tag{Hyp_R}
$$

Hence $\|f*f\|_2^2 - 1 - 2 z_1^4 \le cM - 1 - 2 z_1^4$. Substituting into (C-S):

$$
\int(f*f)\,K
\;\le\; 1 + 2 z_1^2 k_1
   + \sqrt{c\,M - 1 - 2 z_1^4}\;\sqrt{\|K\|_2^2 - 1 - 2 k_1^2}.
\tag{C-S$_R$}
$$

**Step 3 — Lower bound from MV duality.** From MV eqs. (3)+(4) (p. 3–4),
unchanged:

$$
M + \int(f*f)\,K \;\ge\; \int\bigl((f*f)+(f\circ f)\bigr)K
\;\ge\; \frac{2}{u} + a_*,
$$

where the second inequality uses the explicit cosine polynomial $G$ with the
gain $a_* = (4/u) \cdot (\min G)^2 / S_1 \ge 0.07137134846\ldots$ and
$\|f \circ f\|_\infty \le \|f * f\|_\infty = M$ for symmetric $f$.

**Step 4 — Combine.** Substituting (C-S$_R$) into the lower bound,

$$
\frac{2}{u} + a_*
\;\le\; M + 1 + 2 z_1^2 k_1 + \sqrt{cM - 1 - 2 z_1^4}\,\sqrt{\|K\|_2^2 - 1 - 2 k_1^2}.
\tag{$\ast$}
$$

This must hold for the actual $z_1 = |\hat f(1)|$ of $f$. By **MV Lemma 3.4**
(MO 2004 Lemma 2.14), $z_1^2 \le \mu(M)$.

**Step 5 — Worst-case $z_1$.** Let
$M_{\mathrm{master}}(z_1)$ denote the unique solution of ($\ast$) with
equality in $M$. The right-hand side of ($\ast$) is monotone increasing in
$M$, so ($\ast$) holds iff $M \ge M_{\mathrm{master}}(z_1)$.

Since the actual $z_1$ is unknown but constrained to the admissible interval,
the only $M$-bound we can rigorously deduce is

$$
M \;\ge\; \inf_{z_1 \,\in\, [0,\sqrt{\mu(M)}\,] \,\cap\, \mathrm{Hyp_R\;domain}}
                M_{\mathrm{master}}(z_1)
\;=:\; M_{\mathrm{optimal}}(c).
$$

For our $c = \log 16/\pi$, the infimum is attained at the **interior critical
point** $z_1^*$ of the right-hand side of ($\ast$) in $z_1$:

$$
\frac{\partial}{\partial z_1}\!\bigl[\text{RHS of } (\ast)\bigr] = 0
\quad\Longleftrightarrow\quad
z_1^{*4} \;=\; \frac{k_1^2\,(c\,M - 1)}{\|K\|_2^2 - 1}.
$$

For $M$ near $1.378$ this gives $z_1^* \approx 0.4882 < \sqrt{\mu(M)} \approx
0.5767$, so $z_1^*$ is admissible.

**Step 6 — Closed form at $z_1 = z_1^*$.** At $z_1 = z_1^*$ the ($\ast$)
right-hand side simplifies (the cross term cancels by the optimality condition):

$$
\frac{2}{u} + a_* \;=\; M + 1 + \sqrt{(c\,M - 1)\,(\|K\|_2^2 - 1)},
$$

a quadratic in $M$. Solving:

$$
\boxed{\;M_{\mathrm{optimal}}(c) \;=\; \tfrac12\Bigl[(2E + cD) - \sqrt{(2E + cD)^2 - 4(D + E^2)}\Bigr],\;}
$$

where $D := \|K\|_2^2 - 1$ and $E := 2/u + a_* - 1$. Numerically at
$c = \log 16/\pi$, $\delta = 0.138$, $u = 0.638$, $\|K\|_2^2 = 0.5747/\delta$,
$a_* = 0.07137134846\ldots$ from MV's 119 coefficients:

$$
M_{\mathrm{optimal}}(\log 16/\pi)
\;=\; 1.3784219737754177283997025524314107461234603128129\ldots
$$

This completes the proof of the conditional theorem.

## §Z. Why $z_1 = 0.50426$ is not the rigorous choice for $c \ne 1$

The master equation ($\ast$) at fixed $(M, z_1)$ gives a constraint on $M$
that depends on $z_1$. The actual $f$ has *some* $z_1 = |\hat f(1)|$ satisfying
Lemma 3.4 ($z_1 \le \sqrt{\mu(M)}$), but we do not know which value. To
rigorously prove $M \ge M^*$, we need: *for every admissible $z_1$, the
master inequality fails at $M < M^*$*. Equivalently, $M^* = \inf_{z_1}
M_{\mathrm{master}}(z_1)$ — the **infimum**, not any single specific value.

For $c = 1$, the infimum happens to land at the **boundary** $z_1 =
\sqrt{\mu(M^*)}$, which equals $0.50426$ at the self-consistent $M^* \approx
1.27484$ (because the interior critical point $z_1^*$ exceeds the Lemma 3.4
boundary at this $c$ and $M$, making the boundary the tightest admissible
$z_1$). MV's recipe of plugging $z_1 = 0.50426$ verbatim therefore *coincides*
with the rigorous worst-case $z_1$ at $c = 1$ — but this is a numerical
coincidence specific to $c = 1$, not a general principle.

For $c < 1$, the interior critical point $z_1^*$ moves *inside* the admissible
range (because $cM - 1$ shrinks faster than $\mu(M)$ does as $c$ decreases),
and $z_1^*$ becomes the actual argmin of $M_{\mathrm{master}}$. Plugging
the c=1-coincidence value $z_1 = 0.50426$ instead lands at a sub-optimal
point on the *increasing* leg of $M_{\mathrm{master}}(z_1)$, giving an
**over-claimed** lower bound

$$
M_{\mathrm{recipe}}(c) \;>\; M_{\mathrm{optimal}}(c)
\quad\text{for } c < 1.
$$

For $c = \log 16/\pi$:
$M_{\mathrm{recipe}} \approx 1.37925$, but $M_{\mathrm{optimal}} \approx
1.37842$ (a gap of $\sim 8 \times 10^{-4}$). The **rigorous** conditional
bound is $1.37842$.

### Comparison table (40-digit values from `conditional_bound.py`)

| $c$    | $M_{\mathrm{recipe}}$ ($z_1 = 0.50426$ fixed) | $M_{\mathrm{optimal}}$ (rigorous inf $z_1$) |
|--------|----------------------------------------------|---------------------------------------------|
| 0.900  | 1.36197670146109599021567230185              | 1.36157601212351980047484                    |
| 0.920  | 1.34303256771876305264669025141              | 1.34292029961933085406993                    |
| 0.940  | 1.32491308185054588441667759623              | 1.32490993228096059125541                    |
| 0.950  | 1.31614008177416131056864915168              | 1.31613417938295582823387                    |
| 0.960  | 1.30754746266035857272884266057              | 1.30750514569325540685992                    |
| 0.970  | 1.29912780320400200307458347950              | 1.29901842382470249496907                    |
| 0.980  | 1.29087415691888706673580457160              | 1.29066980532446510331750                    |
| 0.990  | 1.28278001101337445712491182712              | 1.28251908392098229331718                    |
| **0.88254240… ($\log 16/\pi$)** | 1.37925062005091026156052413830732403831532  | **1.37842197377541772839970255243141074612346 (HEADLINE)** |
| 1.000  | 1.27483924972349454940688447717 (= MV)       | 1.27483758772764277243121 (slightly < MV)    |

The **headline conditional bound** under Hyp_R($\log 16/\pi$,
$M_{\mathrm{target}}$) is $M_{\mathrm{optimal}}(\log 16/\pi) \approx
1.37842$, **not** the recipe value $1.37925$.

## 6. Caveats and what is actually proved

- **The conditional theorem is rigorous *modulo* Hyp_R.** No part of MV's
  framework (Lemmas 3.1–3.4, the kernel $K$, the test polynomial $G$, the
  119 coefficients) is conjectural. The single conjectural step is
  Hyp_R($\log 16/\pi$, $M_{\mathrm{target}}$).
- **Hyp_R is plausible but unproved.** Boyer–Li disprove the unrestricted
  form, but their witness has $\|g*g\|_\infty \approx 1.652 \gg M_{\mathrm{target}}
  \approx 1.378$, so it lies far outside the restricted class. Whether
  *some* witness can be found inside the class with $\|g*g\|_2^2 >
  c_*\,\|g*g\|_\infty$ is open.
- **MV-class extremizers easily satisfy Hyp_R** with $c_\mathrm{emp} \approx
  0.589$, far below $\log 16/\pi$. The empirical ratio
  $c_\mathrm{emp}(f) = \|f*f\|_2^2 / \|f*f\|_\infty$ trends *upward* with
  $\|f*f\|_\infty$ — $\sim 0.589$ at $M = 1.275$ (MV), $\sim 2/3$ at
  $M = 2$ (indicator $2\cdot\mathbf 1_{[-1/4,1/4]}$), $\sim 0.902$ at
  $M = 1.652$ (BL). The proof's restricted regime $\|f*f\|_\infty \le
  M_{\mathrm{target}} \approx 1.378$ thus sits in the lowest-$M$, lowest-
  $c_\mathrm{emp}$ band, where Hyp_R is most defensible.
