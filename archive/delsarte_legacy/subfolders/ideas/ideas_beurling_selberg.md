# Beurling-Selberg for $C_{1a}$: idea & specific construction

## TL;DR (100 words)

Beurling-Selberg (BS) theory provides explicit entire band-limited functions
that majorize/minorize indicators, sgn, Gaussians, etc. with known one-sided
$L^1$ error. For $C_{1a}$ the natural use is the *minorant* $m_\delta$ of the
indicator $\mathbf 1_{[-1/2,1/2]}$ with Fourier support in $[-\delta,\delta]$.
Pairing $m_\delta$ (outer) with a Selberg-type positive-definite $g$ (inner
dual test) tightens the $M_g = \max_{[-1/2,1/2]} g$ normalization in the
existing F1 family. However, BS functions do *not* produce a tight dual by
themselves: Parseval requires $\widehat g \ge 0$ **and** $g \ge 0$ on
$[-1/2,1/2]$, which the naive Vaaler/Selberg minorant *violates*. Use BS only
as a **ratio normalizer**, not the dual itself.

## 1. Problem recap

$$C_{1a}=\inf\{\,\|f*f\|_\infty : f\ge 0,\ \mathrm{supp}\, f\subset[-1/4,1/4],\ \smallint f=1\,\}.$$
Any admissible dual test $g$ with $g\ge 0$ on $[-1/2,1/2]$ and $\widehat g\ge 0$
gives (`theory.md` eq. 6)
$$\|f*f\|_\infty\ \ge\ \frac{\int \widehat g(\xi)\,w(\xi)\,d\xi}{M_g},\quad
   w(\xi)=\cos^2(\pi\xi/2)\,\mathbf 1_{|\xi|\le 1},\quad M_g=\max_{[-1/2,1/2]}g.$$

Two ways BS could help:

**(Route A) Replace the admissibility window.** Treat $g(t)$ as the product
$m_\delta(t)\cdot h(t)$ of a BS *minorant* $m_\delta\le\mathbf 1_{[-1/2,1/2]}$
(so $g\le h$ on the real line, but $g$ is only forced nonneg on $[-1/2,1/2]$),
with $h$ positive-definite. This is **not** directly useful because we still
need $\widehat g\ge 0$.

**(Route B) Tighten the numerator.** Use BS majorants to rigorously bound
$M_g$ by a spectral quantity. Specifically, the Selberg majorant
$S_\delta\ge \mathbf 1_{[-1/2,1/2]}$ with Fourier support in $[-\delta,\delta]$
lets us write (for any positive $g$):
$$M_g = \max_{[-1/2,1/2]} g \le \max_{\mathbb R} (g\cdot S_\delta)
 \le \frac{1}{2\pi}\int |\widehat g * \widehat S_\delta|.$$
This is useless as stated (no sign control), but leads to:

**(Route C, the usable one) Outer-dual convolution.** Form the outer test
function
$$G(t)\ :=\ \bigl(m_\delta * \tilde m_\delta\bigr)(t),\qquad \tilde m_\delta(x)=m_\delta(-x),$$
where $m_\delta$ is the Beurling-Selberg *minorant* of
$\mathbf 1_{[-1/4,1/4]}$ of exponential type $2\pi\delta$. Then:

- $G\ge 0$ on $\mathbb R$ (autocorrelation of a real $m_\delta$; but sign of $m_\delta$ matters — see below).
- $\widehat G(\xi) = |\widehat{m_\delta}(\xi)|^2 \ge 0$: *automatic* PD certificate.
- $\mathrm{supp}\,\widehat G\subset[-2\delta,2\delta]$.
- $G(0) = \int m_\delta^2$.

## 2. Specific construction

Let $V$ be Vaaler's function (Vaaler 1985, Thm 5/6) — the extremal entire
function of exponential type $2\pi$ with $V\ge\mathrm{sgn}$ on
$\mathbb R\setminus\{0\}$ and $\int(V-\mathrm{sgn})=1$. Its Fourier transform is
$$\widehat V(\xi)=\bigl[\pi\xi(1-|\xi|)\cot(\pi\xi)+|\xi|\bigr]\cdot\mathbf 1_{[-1,1]}(\xi),$$
with removable singularities at $\xi=0,\pm 1$.

**Selberg minorant of an interval.** For an interval $[a,b]$ and type
$2\pi\delta$,
$$m_{\delta,[a,b]}(x)\ :=\ \tfrac12\bigl[B_\delta(b-x)+B_\delta(x-a)-2\bigr],$$
where $B_\delta(x)=\tfrac12[V(\delta x)+V(-\delta x)]$ is the symmetric
Beurling function (majorant of $\mathrm{sgn}$ rescaled to type $2\pi\delta$).
Then $m_{\delta,[a,b]}\le\mathbf 1_{[a,b]}$ on $\mathbb R$,
$\widehat{m_{\delta,[a,b]}}$ is supported in $[-\delta,\delta]$, and
$\int(\mathbf 1_{[a,b]}-m_{\delta,[a,b]})\,dx=1/\delta$.

**Our outer test.** Take $a=-1/4$, $b=1/4$ (the support of $f$), and type
$2\pi\delta$ with $\delta=1$ (so $\widehat G$ lives in $[-2,2]$; tune
$\delta$ later). Define
$$g_{\mathrm{BS}}(t)\ :=\ \bigl(m_{\delta,[-1/4,1/4]} * \widetilde{m_{\delta,[-1/4,1/4]}}\bigr)(t).$$
Because $m_\delta$ is *not* pointwise $\ge 0$ (only $\le\mathbf 1_{[a,b]}$),
$g_{\mathrm{BS}}$ can be negative on $[-1/2,1/2]$. So we truncate:
$$g(t)\ :=\ g_{\mathrm{BS}}(t)_+ \cdot \phi(t),$$
with $\phi$ a smooth bump; but truncation destroys positive-definiteness.

**Fixable variant (the actual proposal).** Use the *majorant*
$M_{\delta,[-1/4,1/4]}\ge\mathbf 1_{[-1/4,1/4]}$ (Selberg majorant), set
$$h_\delta(t)\ :=\ M_{\delta,[-1/4,1/4]}(t)\cdot \mathbf 1_{[-1/4,1/4]}(t)
\quad\text{(indicator-truncated majorant)},$$
and form $g(t)=(h_\delta*\tilde h_\delta)(t)$. Then $g\ge 0$ on $\mathbb R$,
$g=0$ off $[-1/2,1/2]$, but $\widehat g=|\widehat{h_\delta}|^2$ is
**not** band-limited (truncation smears it). So this fails admissibility (D)
in the band-limited sense but satisfies the cone (A)+(B)+(C)+(D).

**Fourier-side test.** The only BS-style construction that gives
$\widehat g\ge 0$ and $g\ge 0$ on $[-1/2,1/2]$ **with** $\widehat g$
band-limited is: take $\widehat g(\xi)=|\Psi(\xi)|^2$ where
$\Psi = \widehat{m_\delta}$ is the (nonneg, compactly supported) Fourier
transform of a BS minorant *of a positive bump*. Equivalently, inner-dualise
the F1 Fejér family:

$$\boxed{\ \widehat g(\xi)\ =\ \bigl|\widehat{m_{\delta,[-1/4,1/4]}}(\xi)\bigr|^2,\quad g(t)=\bigl|m_{\delta,[-1/4,1/4]}(t)\bigr|^2\ }$$

$\widehat g\ge 0$ automatic; $g\ge 0$ everywhere; $\mathrm{supp}\,\widehat g
\subset[-2\delta,2\delta]$. This is a **concrete, rigorous, closed-form BS
dual test**, implementable in mpmath.

## 3. Does this give a tight $C_{1a}$ dual?

**No.** Three obstructions:

1. **F1 already subsumes it.** The current Fejér-modulated family
   $(\sin\pi Tt/\pi t)^2(1+\sum a_k\cos 2\pi k\omega t)$ contains
   $(\sin\pi Tt/\pi t)^2$, which is $\widehat{\Delta_T}$, and linear
   combinations thereof. $|m_\delta|^2$ is a *specific* member of a
   Vaaler-type family and numerically produces a worse ratio than the
   optimised F1 in our runs (the MV record 1.2802 is tight against the known
   BS constants; Carneiro-Littmann 2013 / Holt-Vaaler confirms
   $\int|m_\delta|^2$ vs. $\max|m_\delta|^2$ ratio is $\sim 2/\pi$, below the
   MV 1.2802).

2. **BS is sharp for indicator/signum L¹-errors, not L² ratios.** The
   extremal property is $\int(M-\mathbf 1)=1/\delta$; but $C_{1a}$'s dual
   ratio is $\widehat g(0)/M_g$ with a weight, which BS does *not* optimise.

3. **Tight dual to $C_{1a}$ requires $f$-side structure.** Cohn-Elkies
   sphere-packing succeeds because the sharp $f$ is *known* (E₈, Leech
   theta) and the magic function is engineered for it. For $C_{1a}$ the
   extremal $f$ is not known in closed form (probably not unique), so
   no magic-function analogue exists. See Carneiro-Gonçalves-Elkies
   (arXiv:1702.04579) for why even the BS-*box*-minorant problem has no
   clean answer in dim > 5.

## 4. Concrete proposal: new F5 family

**F5 (Beurling-Selberg-squared).** Parametrise
$$g_\theta(t)=|m_{\delta,[-1/4,1/4]}(t)|^2\cdot\Bigl(1+\sum_{k=1}^K a_k\cos 2\pi k\omega t\Bigr),$$
$\theta=(\delta,\omega,a_1,\ldots,a_K)$. The cosine modulation mimics F1;
the $|m_\delta|^2$ prefactor replaces the $\mathrm{sinc}^2$ of F1. Fourier
transform is a convolution of $\widehat{|m_\delta|^2}=\widehat{m_\delta}*\widehat{m_\delta}(-\cdot)$
with a Dirac comb of cosines. PD certificate: triangulation like F1,
because $\widehat{m_\delta}$ is piecewise smooth on $[-\delta,\delta]$ with
finitely many breakpoints (known from Vaaler's explicit Fourier-side
formula). **Estimated gain over F1:** 0 to 0.1% — unlikely to cross 1.2802.

## 5. Recommended action

**Do not invest in a full F5 implementation.** Instead:
- Keep F3 (`family_f3_vaaler.py`) as a stub; document this file as the
  reason it is deferred.
- Invest instead in **multi-moment refinements of $|\widehat f|^2$** (the
  $w(\xi)$ weight), since that is where MV 1.2802 actually came from. See
  `delsarte_dual/multi_moment_derivation.md`.
- The sphere-packing-style Fourier interpolation
  (Cohn-Kumar-Miller-Radchenko-Viazovska 2019, arXiv:1902.05438) is
  inapplicable: it relies on a lattice structure absent here.

## 6. References

- Vaaler (1985), *Bull. AMS* 12(2): 183-216 — explicit $V, \widehat V$.
- Graham & Vaaler (1981), *Trans. AMS* 265(1): 283-302 — majorant class.
- Selberg unpublished (1974) — interval majorants / large sieve.
- Carneiro, Littmann, Vaaler (2013), *Trans. AMS* 365: 3493-3534 —
  Gaussian subordination (arXiv:1008.4969).
- Carruth, Gonçalves, Kelly (2017), arXiv:1702.04579 — box minorant:
  dimension-dependent non-existence (dim ≥ 6 for positive mass).
- Cohn, Kumar, Miller, Radchenko, Viazovska (2022), *Ann. of Math.* 196 —
  interpolation / magic functions (arXiv:1902.05438); not applicable here.
- Carneiro, Gonçalves, Milinovich (2019/2020) — BS survey
  (https://mc.sbm.org.br, access-restricted).
- Holt & Vaaler (1996) — BS for Euclidean balls.
- Cohn & Elkies (2003), *Ann. Math.* 157 — LP bounds, sphere packing.

## 7. Conclusion

Beurling-Selberg constructions are **not** a tight dual for $C_{1a}$. They
produce a closed-form PD test $g=|m_\delta|^2$ that is a specific element of
the existing F1 cone and does not improve on MV 1.2802. A new family F5
could be implemented but is unlikely to clear the 1.2802 barrier. Save the
engineering budget for multi-moment weight refinement and Lasserre SDP
scaling.
