# Discrete B_2 / Sidon density and the continuous constant C_{1a}

**Question.** Can tight discrete B_2 density bounds be transferred to improve the
continuous lower bound C_{1a} ≥ 1.2802?

**Short answer.** No, the known transfer goes the **wrong direction** for us.
The classical argument (Schinzel-Schmidt / Cilleruelo-Ruzsa-Vinuesa /
Martin-O'Bryant) uses the **continuous** lower bound on sup(f*f) to derive
**upper** bounds on discrete B_2[g] densities -- not the reverse. A stronger
C_{1a} would improve the discrete constants σ_2(g), but a better discrete bound
does **not** (at least not via any published theorem) improve C_{1a}.

---

## 1. The precise transfer (continuous ⇒ discrete)

Define
- σ := inf_{f ≥ 0, supp f ⊂ [0,1], ∫f=1} ||f*f||_∞.  (σ is the Schinzel-Schmidt
  constant; by rescaling from [-1/4,1/4] to [0,1] one has σ = 2·C_{1a}, up to the
  standard convention difference.)
- F(g,n) := max |A|, A ⊂ {1,...,n} a B_2[g] set (every integer has ≤ g ordered
  representations a+a' with a,a' ∈ A).
- σ_2(g) := lim sup_{n → ∞} F(g,n)/√(gn).

**Theorem (Schinzel-Schmidt; sharpened by Yu 2007, Matolcsi-Vinuesa 2010,
Cloninger-Steinerberger 2017).**
For any nonneg f ∈ L^1 ∩ L^2 on an interval of length I,
    sup f*f  ≥  C · ||f||_1^2 / I,          (*)
with C ≥ 1.28 (CS2017). As a **corollary**, for every subset A ⊂ [1,n] with
max-multiplicity g of A+A,
    g·n  >  C · |A|^2,
i.e. σ_2(g) ≤ 1/√C for every g, and in fact
    lim_{g → ∞} σ_2(g) = 1/√σ.

The limit-g result shows the bound σ_2(g) ≤ 1/√σ is **sharp in the limit**, i.e.
σ is literally the discrete-density constant in the high-multiplicity regime.
Improving C_{1a} from 1.28 to C' would immediately give σ_2(g) ≤ 1/√(2C') for
all g.

## 2. Why the reverse direction does not work (as far as we know)

The standard proof of (*) goes: pick f, discretize it into a B_2[g] set (via
O'Bryant's "ε-cover" / Ruzsa's multiplicative-construction trick), apply the
discrete bound g·n ≥ C·|A|^2 at scale n → ∞, take g → ∞, and unwind. This
requires a **uniform-in-g** discrete bound. All known discrete upper bounds
F(g,n) ≤ √(gn)/√C already rely on continuous/Fourier input (or are weaker).

A discrete bound at a **fixed g** (say σ_2(1) for Sidon sets, where Erdős-Turán
gives |A| ≤ √n + O(n^{1/4}), now √n + 0.99703 n^{1/4} by
Balogh-Füredi-Roy/O'Bryant 2021) cannot be transferred: for Sidon sets the
constant 1 in |A|/√n → 1 is the trivial constant, far from σ_2(g)·√g for
large g. One would need a bound of the form
    F(g,n) ≤ (1/√C')·√(gn) · (1 - δ(g,n))
with δ → 0 **uniformly in g**, which is exactly the statement that C_{1a} ≥ C'.
There is no shortcut.

## 3. Recent discrete activity (2020-2026) we examined

- **Balogh-Füredi-Roy 2021, O'Bryant 2021**: tightened the Sidon (g=1) error
  term to 0.99703·n^{1/4}. Irrelevant to C_{1a} (trivial leading constant).
- **Pach-Zakharov 2025** (bi-Sidon, exponent 1/3+7/78): unrelated structure
  (additive+multiplicative).
- **Gaitan-Madrid 2025 (arXiv:2512.18188)**: exact discrete autoconvolution
  constants on {0,1}^n hypercube. Gives C_{2,1} = 1/2 on the 1D cube. Explicitly
  states it does **not** improve 1.28 < C_{1a}. Method is combinatorial, does
  not pass to the continuum at h=2.
- **Cilleruelo-Ruzsa-Vinuesa 2010 "Generalized Sidon sets"**: sharp asymptotics
  for F(g,n) in residue-class setting; depends on continuous input, not the
  reverse.
- **White 2022 (arXiv:2210.16437)**: almost-tight L^2 autoconvolution constant;
  transfers to B_h[g] via the **same** continuous-implies-discrete pattern.

## 4. Net assessment

Every known transfer is continuous → discrete. The discrete side has been mined
hard since 1969 (Erdős-Turán, Lindström, Cilleruelo, Ruzsa, Vinuesa,
Balogh-Füredi-Roy, O'Bryant, Pach-Zakharov) and the leading constant σ_2(g) is
**provably equal to 1/√σ** in the limit: our lower bound on C_{1a} is in fact
the tightest known bound on σ_2(g) for large g. Any discrete result that
actually moves 1/√σ downward would itself be a breakthrough on the continuous
problem, and none exists.

**Not feasible as a lever for lasserre/ or cloninger-steinerberger/.** Better
use of the B_2 literature: (a) cite the equivalence in the final write-up to
motivate C_{1a} improvements, (b) mine Cilleruelo-Ruzsa-Vinuesa's
construction (generalized Sidon sets in Z/NZ) for primal upper-bound ideas,
not lower bounds.

---

## 100-word summary

Transfer between discrete B_2[g] density and continuous C_{1a} is one-way:
Schinzel-Schmidt / Cilleruelo-Ruzsa-Vinuesa / Martin-O'Bryant / Cloninger-
Steinerberger prove σ_2(g) = lim F(g,n)/√(gn) ≤ 1/√σ, with equality as g→∞,
**using the continuous bound as input**. No published result lets a tighter
discrete Sidon/B_2 bound (e.g., Balogh-Füredi-Roy-O'Bryant's 0.99703·n^{1/4},
Pach-Zakharov's bi-Sidon, Gaitan-Madrid's {0,1}^n constant) feed back into a
better C_{1a}. Gaitan-Madrid 2025 explicitly disclaims any C_{1a} improvement.
Our current lower bound 1.28 is already the tightest known bound on the
large-g discrete constant. **Not feasible.**

---

## Sources

- [Martin-O'Bryant, "Supremum of autoconvolutions with applications to additive number theory", arXiv:0807.5121](https://arxiv.org/abs/0807.5121)
- [Cilleruelo-Ruzsa-Vinuesa, "B_2[g] sets and a conjecture of Schinzel and Schmidt", Comb. Prob. Comput. 2010](https://www.cambridge.org/core/journals/combinatorics-probability-and-computing/article/abs/b2g-sets-and-a-conjecture-of-schinzel-and-schmidt/C2F41C14FD382DBE9F057EB88DB4135C)
- [Matolcsi-Vinuesa, "Improved bounds on the supremum of autoconvolutions", arXiv:0907.1379](https://arxiv.org/abs/0907.1379)
- [Cloninger-Steinerberger, "On suprema of autoconvolutions with an application to Sidon sets", arXiv:1403.7988](https://arxiv.org/abs/1403.7988)
- [Cilleruelo-Ruzsa-Vinuesa, "Generalized Sidon sets", arXiv:0909.5024](https://arxiv.org/abs/0909.5024)
- [Balogh-Füredi-Roy, upper bound on Sidon sets, arXiv:2103.15850](https://arxiv.org/abs/2103.15850)
- [Pach-Zakharov, Ruzsa's Problem on Bi-Sidon sets, arXiv:2409.03128](https://arxiv.org/abs/2409.03128)
- [White, "An almost-tight L^2 autoconvolution inequality", arXiv:2210.16437](https://arxiv.org/abs/2210.16437)
- [Gaitan-Madrid, "On suprema of convolutions on discrete cubes", arXiv:2512.18188](https://arxiv.org/abs/2512.18188)
- [Tao's constant page, C_{1a}](https://teorth.github.io/optimizationproblems/constants/1a.html)
- [Vinuesa PhD thesis, "Generalized Sidon sets"](https://www.icmat.es/Thesis/CVinuesa.pdf)
