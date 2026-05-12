# Turán/Krein-type PD-function Inequalities and Their Potential for Sharpening the Dual Bound on C_{1a}

Reference problem: improve the lower bound C_{1a} >= 1.2802 for the Sidon autocorrelation constant
(MV/Cloninger-Steinerberger setting). MV's dual reduces to choosing **one** even positive
definite kernel K supported on [-1/2,1/2] and an affine weight; any inequality that sharpens
the pointwise control of a PD function on [-1/2,1/2] (or its dual tempered distribution) is a
candidate for plugging into a richer dual.

Target scope: PD-kernel / Turán / Krein / Delsarte inequalities on the interval [-a,a] (a = 1/2)
and on the half-interval [-1/4,1/4] that could replace MV's single-K argument by a multi-kernel
or Krein-parametrized family.

---

## 1. Tightest known PD extremal results on intervals

### 1.1 Turán problem on [-a,a]
Definition. For an origin-symmetric convex body U (interval U=[-a,a] here),
```
T(U) = sup{ integral f  :  f PD, supp f ⊂ U, f(0)=1 }.
```
- **Siegel / interval case**. U=[-a,a] is a tiling set, hence Turán; the extremum is uniquely
  achieved by a constant multiple of the triangle
  ```
  f*(x) = (1 - |x|/a)_+,
  ```
  i.e. the self-convolution of (1/sqrt(a)) 1_{[-a/2,a/2]}. Integral value T([-a,a]) = a.
- **Strong duality (Kolountzakis-Lev-Matolcsi, arXiv:2510.10172, 2025)**. T(U) T'(U) = 1 with
  T'(U) the mass-at-origin of a tempered distribution alpha = delta_0 + beta, supp beta ⊂ U^c,
  alpha-hat a positive measure. This is the first strong-duality statement for the Euclidean
  problem; the dual variable is a **single** tempered distribution, but the mass-at-origin
  functional admits a Delsarte-style LP reformulation whose constraints are exactly the
  finite-interval PD constraint - the same primal ingredient MV uses.
- **Arestov-Berdysheva (2001-2002)**. Turán problem solved for regular hexagon, polytopes
  admitting a tiling, and finite unions of intervals (smaller-support approximation).
- **Kolountzakis-Revesz (2003, arXiv:math/0204086; arXiv:math/0302193)**. Proved every spectral
  convex domain (admits an orthonormal basis of exponentials) is a Turán domain. Gives
  **pointwise** upper bounds |f(x)| <= (volume correction) f(0) for PD f with support in U.
  The 1-D interval is spectral, so these pointwise estimates specialize sharply to [-a,a].
- **Boas-Kac / Krein factorization**. Every continuous PD f with supp f ⊂ [-2a,2a] can be
  written f(x) = integral u(y) conj(u(y+x)) dy with supp u ⊂ [-a,a] (the "convolution root").
  This is exactly the primal certificate MV uses implicitly: mu * mu >= C mu on [-1/2,1/2]
  is "f = mu*mu is PD with half-support". Ehm-Gneiting (arXiv:1703.02007) describes which
  multivariate PD functions admit a Boas-Kac root; in 1-D existence is automatic.

### 1.2 Krein extension / uniqueness on an interval
- **Krein (1940)**. A continuous f on (-a,a) extends to a PD function on R iff a certain
  infinite Hankel-type form is nonneg; for f(x) = 1-|x| the extension exists iff a <= 2 and is
  unique iff a = 2. Gives a **parametric family** of all PD extensions when nonuniqueness holds.
- **Sasvari, Bakonyi-Constantinescu**. Explicit Nevanlinna-Pick-type parametrization of the set
  of all PD extensions of a given f on [-a,a]. The parametrization is by a Schur-class function
  s in the unit disk; each choice of s yields a distinct PD kernel on R agreeing with f on
  [-a,a]. This is precisely a **Krein-type multi-parameter dual family** - not studied (to our
  knowledge) in the MV/Sidon literature.
- **Rudin / Manov (2019, Doklady)**. New extremal variants on compactly supported PD functions
  on the interval: bounds on integral |x|^k f(x) dx given f(0)=1 and f PD on [-a,a].

### 1.3 Integral inequalities specific to nonnegative PD functions
- **Gorbachev-Tikhonov (Math. Z. 2018; Wiener's problem)**. Sharp constants for
  ```
  G(l) = sup { integral_{[-l,l]} f / integral_{[-1,1]} f : f PD, f >= 0, integral f < infty }.
  ```
  Proved via reduction to Turán/Delsarte on an interval. Gives **direct integral inequalities
  on the interval scale** of the form integral_I f <= c(l) integral_{[-1,1]} f that are new
  ingredients beyond MV's pointwise PD bound.
- **Ehm-Gneiting-Richards** and **Boyadzhiev** give sharp pointwise PD inequalities
  f(x) <= f(0) * triangle(x/a) with extra constraints from subset symmetrization - a candidate
  replacement for MV's "K <= 1 on [-1/2,1/2]" pointwise constraint.
- **Cohn-Elkies (Ann. Math. 2003)** dual for sphere packing uses a **pair** (f, f-hat) with sign
  constraints separated by scale; generalizations (Cohn-Kumar-Miller-Radchenko-Viazovska) use
  modular-form constructions. Not a single PD kernel but a PD pair, and the dual LP admits
  tighter Delsarte analogs (arXiv:2405.07666, 2024) that use finite-dimensional reductions
  - directly imitable for the Sidon problem.

---

## 2. Applications to Sidon / convolution problems

- **MV (arXiv:0907.1379, 2010)** use a single even PD test function K supported on [-1,1] with
  K(0)=1 and K <= 1 on an explicit set, to get a **lower** bound on sup (f*f) via the
  inequality sum_j K(x-t_j) <= M * (f*f)(x) integrated against a clever measure. This is
  essentially a Turán-dual certificate restricted to the autoconvolution problem. The constant
  1.28 they derive is exactly T-dual-ish for a carefully chosen Fejer-like K.
- **Martin-O'Bryant (autoconvolution 2015; 2024)** and **White (arXiv:2210.16437, 2022)** push
  the MV bound marginally. Each replaces K by a single new PD function optimized by SDP - the
  class of dual certificates is unchanged in structure.
- **Boyer-Li (arXiv:2506.16750, 2025)** gave the current upper bound 1.5029 using a Vaaler-type
  majorant (a PD pair rather than a single K) - this is the only place in the Sidon literature
  we see a **Beurling-Selberg pair** instead of a single PD test.
- **Cilleruelo-Ruzsa-Vinuesa** type arguments for g-Sidon sets reuse Turán on the interval in a
  discretized form - the asymptotic of B_h[g] sets feeds back into the continuous autoconv
  problem through a Prekopa-Leindler-type convex duality.
- **Kolountzakis-Matolcsi (MUBs, arXiv:2202.13259)** demonstrates that the **dual** PD method
  for a packing-type problem is strictly improved by allowing **several** PD test functions
  with coupled mass constraints - this is the cleanest precedent in the literature for a
  multi-kernel dual in an interval-support PD setting.

---

## 3. Specific inequalities transferable to C_{1a}

Below, f is nonneg with supp f ⊂ [-1/4,1/4], integral f = 1, and g = f * f is supported on
[-1/2,1/2]. g is automatically PD and nonneg. We want max_{|t|<=1/2} g(t) >= C_{1a}.

### 3.1 Gorbachev-Tikhonov -> weighted MV dual
Their inequality
```
integral_{[-l,l]} g  <=  c(l) * integral_{[-1/2,1/2]} g       (g PD, g >= 0)
```
is a **new** linear constraint on g across scales l <= 1/2 that MV's single-K dual does not
use. Adding c(l) as extra LP constraints (indexed by a grid of l's) produces an LP relaxation
strictly tighter than MV. Cost: one new linear constraint per l. This is directly droppable
into `lasserre/preelim.py` as extra moment linearities.

### 3.2 Krein / Nevanlinna-Pick parametrization -> multi-kernel dual
Because g|_{[-1/2,1/2]} admits a 1-parameter family of PD extensions {g_s}_{s in Schur class},
the **dual** bound from testing against any single K can be averaged over s. Concretely:
```
max g  >=  sup_{s}  < K_s, g > / integral K_s,
```
with {K_s} the Krein family. Optimizing s inside the dual is a semi-infinite LP whose
restriction to a finite s-grid gives a multi-kernel SDP that strictly dominates MV. Structurally
this is the same move Cohn-Kumar did when passing from one radial function to a modular pair.

### 3.3 Kolountzakis-Revesz pointwise PD bound
For a PD h supported in [-a,a],
```
|h(x)|  <=  h(0) * (1 - |x|/a)_+  (valid exactly on spectral / tiling domains).
```
Applied to h = g restricted and translated, this is a **sharper** pointwise inequality than
MV's "K <= 1" constraint. It cuts a dual polytope strictly; any SDP primal that respects it
returns a strictly larger lower bound. Immediately usable as a cheap cut in the Lasserre LP.

### 3.4 Beurling-Selberg / PD pair (Vaaler majorant) dual
Following Boyer-Li on the **upper** side: use a Vaaler pair (M_+, M_-) with M_-(x) <= 1_{E}(x)
<= M_+(x), both PD. The Sidon lower bound becomes an LP with two coupled PD constraints, one
on g and one on g-hat. This is the closest continuous analog of Cohn-Elkies.

### 3.5 Strong-duality dual distribution alpha (Kolountzakis-Lev-Matolcsi 2025)
The strong-duality statement T(U) T'(U) = 1 hands us a tempered distribution alpha = delta_0
+ beta with supp beta ⊂ [-1/2,1/2]^c and alpha-hat >= 0. Testing g against alpha gives
```
g(0) = <alpha, g>  >=  g(0) - (tail through beta),
```
which translates the Turán dual certificate onto the Sidon autoconv max directly. Because
alpha is a **distribution** (not just a function) it can be sparser than any PD K, so in
finite-moment SDPs it potentially gives extreme rays the scalar MV dual cannot reach.

**Concrete drop-in order of difficulty** (easy -> hard):
1. Gorbachev-Tikhonov multi-scale integral cut (Section 3.1) - one-hour change to preelim.py.
2. Kolountzakis-Revesz pointwise majorant (3.3) - replaces K <= 1 with tighter triangular cut.
3. Krein parametric family of K's (3.2) - needs a parametric SDP (s-grid or Nevanlinna-Pick).
4. Vaaler PD pair (3.4) - requires Selberg minorant for interval indicator.
5. Distributional alpha dual (3.5) - most flexible, least concrete.

---

## 4. 100-word summary

MV's 1.2802 lower bound is a **single positive-definite-kernel dual** on [-1/2,1/2]. The Turán
extremal literature (Arestov-Berdysheva, Kolountzakis-Revesz, Gorbachev-Tikhonov, and the 2025
Kolountzakis-Lev-Matolcsi strong-duality paper) gives strictly richer certificates: a Krein
parametric family of PD extensions, Gorbachev-Tikhonov's cross-scale integral inequalities, the
Kolountzakis-Revesz pointwise triangular majorant on spectral domains, Vaaler/Selberg PD
pairs, and a tempered-distribution dual. Each replaces MV's one kernel by a family, adding
linear constraints to the primal LP/SDP without altering its feasibility. The Gorbachev-Tikhonov
and Kolountzakis-Revesz cuts are immediately droppable into `lasserre/preelim.py`; Krein and
Vaaler families promise the largest gains.

---

## Key references

- Kolountzakis, Lev, Matolcsi, "The Turán and Delsarte problems and their duals," arXiv:2510.10172 (2025).
- Kolountzakis, Revesz, "On a problem of Turan about positive definite functions," arXiv:math/0204086 (2002).
- Kolountzakis, Revesz, "On pointwise estimates of positive definite functions with given support," arXiv:math/0302193 (2003).
- Arestov, Berdysheva, "The Turán problem for a class of polytopes," E. J. Approx. (2001-2002).
- Gorbachev, Tikhonov, "Wiener's problem for positive definite functions," Math. Z. 289 (2018).
- Ehm, Gneiting, Richards, "Convolution roots of radial positive definite functions with compact support," Trans. AMS (2004).
- Sasvari, "Positive Definite and Definitizable Functions," Akad. Verlag (1994).
- Bakonyi, Constantinescu, "Schur's Algorithm and Several Applications," Longman (1992).
- Krein, "Sur le probleme du prolongement des fonctions hermitiennes positives continues," Dokl. AN SSSR 26 (1940).
- Matolcsi, Vinuesa, "Improved bounds on the supremum of autoconvolutions," arXiv:0907.1379 (2010).
- Martin, O'Bryant; White, "Improving MV bound" (2015, 2022); arXiv:2210.16437.
- Boyer, Li, arXiv:2506.16750 (2025) (upper bound 1.5029 via Vaaler pair).
- Cohn, Elkies, "New upper bounds on sphere packings I," Ann. Math. 157 (2003).
- Kolountzakis, Matolcsi, "Dual bounds for the positive definite functions approach to mutually unbiased bases," arXiv:2202.13259 (2022).
- New solutions to Delsarte's dual LP, arXiv:2405.07666 (2024).
