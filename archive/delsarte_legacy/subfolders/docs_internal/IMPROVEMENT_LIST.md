# High-Impact, Low-Compute Improvements — MV Cauchy-Schwarz Dual

Every item stays inside the MV framework (arcsine-autoconvolution kernel $K$ + trig polynomial $G$, scalar Cauchy-Schwarz master inequality), is mathematically proved, and is computationally light. Removed: items requiring heavy QP at n=1000+, 200-digit interval LP, or large-matrix SDPs.

---

## B2. MO 2004 Lemma 2.17 joint constraint

Add the proved linear constraint $\text{Re}\,\hat f(2) \le 2\,\text{Re}\,\hat f(1) - 1$ to the joint $(z_1, z_2)$ optimisation inside MV's master inequality. MO agent projects **1.27481 → [1.28, 1.30]**.

**Compute**: trivial — one extra linear inequality in a 2D optimisation of the scalar master inequality. Seconds.

## B1. Multi-moment extension

Pull ALL $|\hat f(n)|^2 \hat K(n)$ out of the Parseval sum for $n = 1, \ldots, n_{\max}$; apply Cauchy-Schwarz only to the tail. Uses the proved **uniform** bound $|\hat h(n)| \le M\sin(\pi/M)/\pi$ (MO 2004 Lemma 2.14).

**Compute**: $n_{\max}$ more scalar terms in the master inequality. Milliseconds per eval.

## B1 + B2 combined

Multi-moment pull-out + MO Lemma 2.17 joint constraint. Gains act on independent parts of the master inequality; they compound. Strongest light-compute package.

**Compute**: a joint $(z_1, z_2, \ldots, z_{n_{\max}})$ optimisation with a handful of linear cuts. Well under a second per evaluation.

## B3. Union of forbidden $z_1$-intervals

Construct TWO triples $(K_1, G_1)$ and $(K_2, G_2)$ (different $\delta$) whose forbidden intervals for $z_1$ UNION to cover $[0, M^{1/2}]$. Closes MV's single-triple gap $(0.5044, 0.5298)$ explicitly flagged in their paper (open direction 2).

**Compute**: solve the master inequality twice at two different $\delta$. Linear in number of triples used.

## D2. Joint $(\delta, u)$ scan

MV chose $u = 1/2 + \delta$ "for convenience" (no proof of optimality). Relax it: scan $(\delta, u)$ on a coarse grid (e.g. 20×20) with MV's tabulated $a_j$ re-used. Only the constants $\|K\|_2^2 = \int J_0(\pi\delta\xi)^4 d\xi$ and $\tilde K(j) = |J_0(\pi j\delta/u)|^2/u$ change per point; the QP does not need to be re-run.

**Compute**: 400 scalar master-inequality evaluations. Well under a minute total.

---

## Priority

**B2** — single biggest lift; the only item the literature projection expects to cross 1.28.
**B1 + B2** — compounding; strongest light-compute combination.
**B3**, **D2** — moderate lifts that combine with the above without recomputing the QP.

The set {B1, B2, B3, D2} is the full light-compute package — every item is proved, inside MV's dual, and runs on a laptop in under a minute.
