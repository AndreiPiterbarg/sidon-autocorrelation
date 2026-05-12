# From val(d) to C_{1a}: the discretization-error term is zero

> **Headline.** A certified lower bound `lb_rig` on `val(d)` is *automatically*
> a certified lower bound on `C_{1a}` for every `d ≥ 2`. There is **no
> discretization-error term ε(d)** to subtract. In particular,
>
> &nbsp;&nbsp;&nbsp;&nbsp;val(64) ≥ 1.281 ⟹ C_{1a} ≥ 1.281.
>
> The user prompt asked to compute ε(d) for d ∈ {64, 128}; the
> rigorous answer is **ε(d) ≡ 0** for every d, and this section
> reproduces the proof.

This file complements [d64_d128_plan.md](d64_d128_plan.md) and the
`proof/lasserre-proof/lasserre_lower_bound.tex` writeup. It states the
extrapolation lemma standalone so that the d=64 / d=128 certificates
can be cited end-to-end in [../proof/lasserre_unconditional_writeup.md](../proof/lasserre_unconditional_writeup.md).

## 1. Setup

Let
$$
  \mathcal{F} = \bigl\{ f \in L^1(\mathbb{R}) :
    f \ge 0,\ \operatorname{supp}(f) \subseteq (-\tfrac14,\tfrac14),\
    \int f > 0 \bigr\}.
$$

For $f \in \mathcal{F}$ the autoconvolution ratio is
$R(f) = \|f\!*\!f\|_{L^\infty} / (\int f)^2$, and
$C_{1a} = \inf_{f \in \mathcal{F}} R(f)$.

Fix $n \ge 1$ and $d = 2n$. Partition $[-\tfrac14,\tfrac14)$ into $d$
half-open bins of equal length $1/(4n)$:
$$
  I_i = \bigl[ -\tfrac14 + \tfrac{i}{4n},\
                -\tfrac14 + \tfrac{i+1}{4n}\bigr),
  \qquad i=0,\dots,d-1.
$$
For normalized $f$ ($\int f = 1$), the *bin masses*
$\mu_i = \int_{I_i} f$ form a probability vector
$\mu \in \Delta_d \subseteq \mathbb{R}_{\ge 0}^d$.

Cloninger and Steinerberger ([CS17, Lemma 1]) prove the *windowed
test-value inequality*: for any pair $(\ell, s_{\rm lo})$ with
$\ell \ge 2$ and $0 \le s_{\rm lo} \le 2(d-1) - (\ell-2)$,
$$
  \|f\!*\!f\|_{L^\infty}
  \;\ge\;
  \frac{2d}{\ell}
    \sum_{s = s_{\rm lo}}^{s_{\rm lo}+\ell-2}
    \sum_{\substack{0 \le i, j \le d-1 \\ i + j = s}}
      \mu_i \mu_j
  \;=:\; \mu^\top M_W\, \mu,
$$
with the window matrix
$(M_W)_{ij} = \frac{2d}{\ell}\,\mathbf{1}\!\bigl[s_{\rm lo} \le i+j \le s_{\rm lo}+\ell-2\bigr]$.

Define the *simplex polynomial value*
$$
  \operatorname{val}(d)
    \;=\; \min_{\mu \in \Delta_d}\, \max_{W \in \mathcal{W}_d}\, \mu^\top M_W\, \mu.
$$

## 2. The extrapolation lemma

**Lemma (val ≤ C_{1a}).** For every integer $n \ge 1$ with $d = 2n$,
$$
  \operatorname{val}(d) \;\le\; C_{1a}.
$$

*Proof.* Take any $f \in \mathcal{F}$ with $\int f = 1$ and let $\mu$ be
its bin-mass vector. By the windowed test-value inequality,
$\|f\!*\!f\|_{L^\infty} \ge \mu^\top M_W \mu$ holds **for every** window
$W$, hence
$$
  \|f\!*\!f\|_{L^\infty}
    \;\ge\; \max_{W \in \mathcal{W}_d} \mu^\top M_W \mu
    \;\ge\; \min_{\nu \in \Delta_d} \max_W \nu^\top M_W \nu
    \;=\; \operatorname{val}(d).
$$
Taking the infimum over $f$ on the left gives $C_{1a} \ge \operatorname{val}(d)$. $\square$

**No ε(d) term.** The proof is sharp at every step: no quadrature error,
no rounding, no asymptotic estimate. The window matrix is built so that
the inequality $\|f\!*\!f\|_{L^\infty} \ge \mu^\top M_W \mu$ is *exact*
for every admissible window — the binning of $f$ into $I_i$'s is the
discretization, and it is *applied to the test-value inequality*, not to
$f$ itself. The bin masses $\mu_i$ are the **exact** integrals
$\int_{I_i} f$.

A common point of confusion: in branch-and-prune approaches that
discretize $f$ to a finite-dimensional $f_d \approx f$ and then bound
$\|f_d * f_d\|_\infty$, a discretization error $\varepsilon(d)$ does
appear (this is the C&S Lemma 3 correction $2/m + 1/m^2$ — see memory
`project_fine_grid_fix.md` for the soundness lesson at $d, m$). **The
Lasserre route does not discretize $f$.** It bounds $C_{1a}$ via the
*true* bin masses and the *true* windowed test values; the
finite-dimensional object is the simplex polynomial $\operatorname{val}(d)$,
which is bounded below by SDP relaxations.

## 3. From the SDP to a rigorous lower bound

The SDP "ladder" (proven in `proof/lasserre-proof/lasserre_lower_bound.tex`,
Theorem `Lasserre soundness` and Lemma `clique-soundness`):
$$
  \operatorname{val}^{(k,b)}(d)
    \;\le\; \operatorname{val}^{(k)}(d)
    \;\le\; \operatorname{val}(d)
    \;\le\; C_{1a},
$$
for all $k \ge 1$, $0 \le b \le d-1$. Here $\operatorname{val}^{(k)}(d)$ is
the dense Lasserre-order-$k$ relaxation and $\operatorname{val}^{(k,b)}(d)$
is its sparsification under banded chordal cliques of bandwidth $b$.

Combining with the val-le-c1a lemma:
$$
  \boxed{\quad \operatorname{val}^{(k,b)}(d) \;\le\; C_{1a} \qquad
         \forall\, k \ge 1,\ b \le d-1,\ d \ge 2. \quad}
$$

So *any* certified lower bound `lb_rig` on the sparse Lasserre relaxation
$\operatorname{val}^{(k,b)}(d)$ — produced by the Farkas-rounding pipeline
in `lasserre/d64_farkas_cert.py` — is automatically a lower bound on
$C_{1a}$.

## 4. Numeric ε(d) for d ∈ {64, 128}

| d   | val(d) [float estimate] | ε(d) [discretization error] | Lasserre target |
|-----|-------------------------|------------------------------|------------------|
| 64  | 1.384                   | **0**                        | val^{(2,16)}(64) ≥ 1.281 |
| 128 | 1.420                   | **0**                        | val^{(2,16)}(128) ≥ 1.281 |

(Float estimates are from `lasserre/core.py::val_d_known`, populated
by historical multistart runs. They are upper bounds on `val(d)` — the
SDP lower bound is what makes the chain rigorous.)

## 5. Final claim

**Theorem.** *If the Farkas certificate file `lasserre/certs/d64_cert.json`
records `status = "CERTIFIED"` with `lb_rig ≥ Fraction(1281, 1000)`,
then $C_{1a} \ge 1.281$.*

*Proof.* The certificate is a Farkas-style dual feasibility witness for
the sparse SDP at fixed $t = $ `lb_rig`, with positive safety margin in
exact rationals (verified to `mpmath` precision dps=80). Therefore the
SDP at this `t` is rigorously infeasible, and
$$
  \operatorname{val}^{(k,b)}(64) \;>\; \mathtt{lb\_rig} \;\ge\; 1.281.
$$
By the boxed inequality of §3, $C_{1a} \ge \operatorname{val}^{(k,b)}(64) > 1.281$. $\square$

**Corollary.** $C_{1a} \ge 1.281 > 1.2802$, strictly improving on the
Cloninger–Steinerberger bound. (The SDP route does not subsume CS17 —
their argument is ε(d)-tight on a different discretization — but it
provides a *fundamentally simpler* proof certificate: a single
finite-dimensional SDP dual instead of an exhaustive lattice
enumeration.)

## 6. Why the ε(d) misconception is common

In the Cloninger-Steinerberger and cascade pipelines, $f$ is approximated
by a piecewise-constant $f_d$ on the bins, and the bound
$\|f_d * f_d\|_\infty$ is verified by *combinatorial enumeration over
allowed mass distributions*. There the discretization error is real and
non-zero: it comes from the *quadrature of the convolution integral
against piecewise-constant $f_d$*, and the C&S Lemma 3 correction
$\varepsilon(d, m) = 2/m + 1/m^2$ accounts for it.

Our pipeline avoids this. We never approximate $f$ by $f_d$. We use the
*exact* bin masses $\mu_i = \int_{I_i} f$ and the *exact* anti-diagonal
windowed quadratic form $\mu^\top M_W \mu$. These are valid lower bounds
on $\|f * f\|_\infty$ for *every* function $f \in \mathcal{F}$, with no
function-space approximation — the only relaxation is the move from the
infinite-dim $\mathcal{F}$ to the finite-dim $\Delta_d$, and that
relaxation is a valid LOWER bound, not a lossy approximation, by the
val-le-c1a lemma.

Hence ε(d) ≡ 0 in the Lasserre route. The val→C_{1a} extrapolation step
is **trivial up to verifying the SDP certificate**.
