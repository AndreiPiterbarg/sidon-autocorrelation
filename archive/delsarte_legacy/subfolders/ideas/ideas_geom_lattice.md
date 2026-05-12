# Geometric / Lattice Connections for $C_{1a}$ Lower Bounds

**Question.** Does the Sidon autoconvolution sup-norm problem
$\;\inf_{f\ge 0,\,\mathrm{supp}\,f\subset[-1/4,1/4],\,\int f=1}\;\max_{|t|\le 1/2}(f*f)(t)\ge C_{1a}\;$
relate to a lattice/packing/covering problem with an LP dual we can exploit for a *lower* bound?

## 1. Direct structural links

### 1.1 Turán's extremal problem (the cleanest match)
Turán: among positive-definite $g$ supported on $[-1,1]$ with $g(0)=1$, maximize $\int g$.
Siegel (1930s): the extremal $g$ is a multiple of $\mathbf{1}_{[-1/2,1/2]} * \mathbf{1}_{[-1/2,1/2]}$.
So Turán's extremizer *is* a self-convolution of an indicator on our exact interval scale.
This is the closest classical geometric twin: our object (sup of $f*f$) and Turán's (integral of positive-definite function) are both governed by the Boas-Kac-Krein convolution-root theorem.
Consequence: Turán/Delsarte dual LP machinery (Kolountzakis-Révész 2006; Arestov; Ivanov 2010; Gabardo 2024) gives structured test functions — but these bound integrals, not $\|f*f\|_\infty$. Conversion requires a pointwise-to-integral dualization (open).

### 1.2 Cohn-Elkies LP (sphere packing)
Cohn-Elkies (2003) bound needs $h\in L^1(\mathbb R^d)$ with $h(0)=\hat h(0)$, $h(x)\le 0$ for $|x|\ge r$, $\hat h\ge 0$. Structurally parallel to the $(f*f)$-sup problem if we set $h=f*f-c$: negativity outside a "forbidden" zone and Fourier non-negativity. In dim $d=1$ the LP is essentially solved (Cohn-Kumar: scaled $\mathbb Z$ is universally optimal), so it does not directly yield new inequalities — but the *auxiliary-function* technology (Viazovska-style modular forms, Fourier interpolation, magic functions) is reusable for constructing dual certificates.

### 1.3 $B_2[g]$ sets / Sidon packing duality
Matolcsi-Vinuesa (2010, arXiv:0907.1379) explicitly tie $C_{1a}$ (they call it $c_\infty$, upper bound 0.75049 for the *reciprocal* convention) to $B_h[g]$-set density via the Schinzel-Schmidt circle of ideas. The problem can be phrased as a packing density: $\sup\{|A|^2/N\}$ over $A\subset[1,N]$ with bounded additive energy density. *Dual LP here targets upper bounds on $c_\infty$, not lower bounds on $C_{1a}$ (our regime).* Still, the dual's infeasibility certificates can translate.

### 1.4 $L^2$-autoconvolution and energy minimization
Cohn-Zhao (2012, arXiv:1212.1913) "energy-minimizing error-correcting codes" recast code-distance bounds as energy minimization over a discrete space; the continuous analogue (Cohn-Kumar 2019, E8/Leech) has dim-1 special case: integer lattice universally optimal for any completely monotone $(x\mapsto\phi(x^2))$ potential. Our cost $\max(f*f)$ is **not** such a potential, but the LP duality framework (find a test polynomial $p\ge \phi$ on supp, $\hat p\le 0$ outside 0) is structurally identical.

## 2. Covering reinterpretation

$\|f*f\|_\infty \ge C$ means: for every prob. density $f$ on $[-1/4,1/4]$, some translate $f_t(x)=f(x-t/2)$ has $\int f\cdot f_t \ge C$. Equivalently: *self-translates must overlap with integral at least $C$*. This is a **1-D covering problem** for measures (not sets). Tight covering density bounds via Voronoi reduction (Schürmann-Vallentin 2005) give lattice-covering analogues; the measure-valued version appears to be new, but the Delsarte dual LP is the same.

## 3. Lattice-free random / Klartag angle

Klartag (2025, arXiv:2504.05042): $\log d$ improvement to Rogers via non-lattice random packings. Translated to our 1D setting: randomized mass allocations (à la cascade branch in `cloninger-steinerberger/`) *are* a Klartag-style random construction; we could try to mirror his second-moment Poisson-graph argument to lower-bound $C_{1a}$ by analyzing the autocorrelation of a random non-negative density.

## 4. What is genuinely new to try

1. **Turán-dual polynomial test functions.** Reuse Kolountzakis-Révész families as ansatz in `lasserre/` SOS certificates: they satisfy Fourier-positivity by construction.
2. **Cohn-Elkies magic-function transport.** Viazovska modular-form magic functions in $d=1$ are classical theta series; constructing a $d=1$ "magic" autoconvolution kernel may yield a pointwise certificate $(f*f)(t^*) \ge C_{1a}^{\text{new}}$ via Poisson summation.
3. **Schürmann-Vallentin lattice-covering SDP** as a template for the `certified_lasserre/` Farkas pipeline — their rational certificates translate.
4. **Klartag-style random lower bound** as a non-constructive floor (independent of SDP).

## 5. Bottom line

There *is* a rigorous LP-dual connection: the object most closely analogous to $\|f*f\|_\infty$ is the Turán / Boas-Kac-Krein positive-definite extremal problem on $[-1,1]$. The dual LP infrastructure (Kolountzakis-Révész; Cohn-Elkies; Gabardo-FFT20) is mature but oriented toward *integral* functionals; bridging to the *sup-norm* side is the missing technical step. Concrete next move: build Lasserre test functions from Turán extremizers and Fourier-magic ansatze rather than monomials. This is cheap (days, not weeks) and reuses existing `lasserre/preelim.py` infrastructure.

---
### Key references
- Cohn-Elkies, *New upper bounds on sphere packings I*, Ann. Math. 157 (2003), arXiv:math/0110009.
- Cohn-Kumar-Miller-Radchenko-Viazovska, *Universal optimality of $E_8$ and Leech*, Ann. Math. 196 (2022), arXiv:1902.05438.
- Matolcsi-Vinuesa, *Improved bounds on the supremum of autoconvolutions*, arXiv:0907.1379 (2010).
- Gabardo et al., *The Turán problem and its dual*, J. Fourier Anal. Appl. 30 (2024).
- Kolountzakis-Révész, *Turán's extremal problem*, (2006).
- Klartag, *Lattice packings of spheres in high dim via dense sublattice*, arXiv:2504.05042 (2025).
- Cohn-Zhao, *Energy-minimizing error-correcting codes*, arXiv:1212.1913.
- Schürmann-Vallentin, *Computational approaches to lattice packing and covering*, (2006).

---

## 100-word summary

The Sidon autoconvolution sup-norm problem has three genuine geometric/lattice cousins: (i) Turán's extremal problem for positive-definite functions on $[-1,1]$, whose Siegel extremizer is exactly a self-convolution of an indicator on our interval scale; (ii) the Cohn-Elkies LP for sphere packing, whose 1-D magic-function machinery (scaled $\mathbb Z$ universally optimal, Cohn-Kumar) supplies reusable auxiliary-function ansatze; (iii) $B_h[g]$-set packing, explicitly linked to $C_{1a}$ by Matolcsi-Vinuesa. All three have mature Delsarte-style dual LPs, but they bound *integrals*, not sup-norms. Concrete win: seed Lasserre SOS certificates with Turán/Fourier-magic test functions rather than monomials — cheap to try, reuses `lasserre/preelim.py`.
