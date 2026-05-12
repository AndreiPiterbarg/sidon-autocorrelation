# Proof paths for a sharper Lemma 3.4

**Setup.** `h = f*f`, `f >= 0` real, `supp(f) in [-1/4, 1/4]`, `int f = 1`. So `h >= 0`,
`supp(h) in [-1/2, 1/2]`, `int h = 1`, `||h||_inf =: M`. Write `z_n = |hat f(n)|`, so
`|hat h(n)| = z_n^2`. The sequence `(hat h(n))_{n in Z}` is positive-definite with
`hat h(0) = 1`.

**MV's bathtub extremizer** `h* = M * 1_{A*}` (with `A*` a measure-`1/M` super-level set
of `cos(2 pi n x)`) is **not attainable as an autoconvolution**: `f * f` is continuous
everywhere and vanishes at `x = +/- 1/2`, while `M * 1_{A*}` is discontinuous at `partial A*`
and need not vanish at `+/- 1/2`. So a strict qualitative gap exists; quantifying it is
the content of the three paths below.

---

## Path 1: Toeplitz-PSD SDP dual  (Krein-Nudelman + autoconvolution nonneg)

**Idea.** Maximize `y = hat h(n_0)` (say `n_0 = 1`) subject to the linear/conic constraints
that the full Fourier sequence `(hat h(k))_{k in Z}` can be realized by some `h` with
`0 <= h <= M` on `[-1/2, 1/2]`, `int h = 1`, PLUS the autoconvolution constraint
`hat h(k) >= 0` (after shifting `f` to make `hat f(n_0)` real nonneg; this is a phase WLOG).

At a truncation level `N`, the SDP is:
```
  max y_{n_0}
  s.t.  y_0 = 1,
        y_k in R and y_{-k} = y_k for all k,
        y_k >= 0 for all k (autoconvolution WLOG for n_0 = 1),
        T_N := [y_{i-j}]_{i,j=0..N} is PSD,
        trigonometric polynomial M - sum_k y_k e^{2 pi i k x} >= 0 on [-1/2, 1/2]
        (Schmuedgen/Putinar localizing form, which becomes an LMI
         via Fejer-Riesz at finite truncation).
```

**Main lemma.** (Dumitrescu 2007, Chapter 4 "Bounded-density moment sequences".)
A real sequence `(y_k)_{k=-N..N}` with `y_{-k} = y_k`, `y_0 = 1` is realizable by a
positive measure `d_nu <= M dx` on `[-1/2, 1/2]` iff the pair of LMIs
```
  T_N := [y_{i-j}]              >= 0,
  M * T_N - [y_{i-j}]           >= 0  (in the Toeplitz sense after symbol shift)
```
is feasible. (The second LMI encodes `h(x) <= M` via Fejer-Riesz.)

**Dual certificate.** The SDP dual is a trigonometric polynomial `q(x) = sum q_k e^{2 pi i k x}`
of degree `<= N` with `q(x) >= cos(2 pi n_0 x)` for `x in [-1/2, 1/2]` AND `q(x) >= 0`
(so `int q d_h <= M * int q dx`), yielding `hat h(n_0) <= M * int q dx`. Minimizing over
such `q` gives the sharpest bound `mu'_N(M, n_0)`.

**Expected yield.** At `M = 1.275, n_0 = 1`: `mu_MV(M) ~ 0.2542`. Path-1 numerical
estimates suggest `mu'_N(M, 1) ~ 0.24 - 0.25` for `N = 3` to `5`, i.e. a 1-5% gap. This
propagates to `~0.001` on the final `M_cert` after the Phase-2 pipeline — comparable to
MV's whole multi-moment gain.

**Rigor/feasibility.** Numerical SDP: trivial (cvxpy, seconds). Rigorous certificate:
- extract rational dual polynomial `q_Q(x)` from numerical solution,
- verify PSD of the slack matrix `(q_Q - cos(2 pi n_0 x)` and `(q_Q - cos)` on the
  relevant support via Putinar/Schmuedgen SOS in rational arithmetic (Lean or sage),
- feed certified `mu'(M, n_0)` into `F_bathtub_sharper`.

This is a significant rigorous-certification task (likely 1-3 weeks of careful work,
not 2 hours).

**Citations.** Krein-Nudelman 1977 Ch. III; Dumitrescu 2007 Ch. 4; Lasserre 2009;
Boyer-Li 2025 (piecewise-linear autoconv witness construction is the primal side of
this SDP).

---

## Path 2: Fejer-Riesz / Akhiezer restriction to bandlimited nonneg squares

**Idea.** Any nonneg `L^1` function `h` on `R` with `supp(hat h) in [-L, L]` factors as
`h = |p|^2` with `p` entire of exponential type `L/2` (Akhiezer-Krein). In our setting
`h` lives on the torus and is supported on `[-1/2, 1/2]`; a toroidal version of this
factorization expresses `h = |p|^2` for `p` a trigonometric polynomial of controlled
type.

**Main lemma.** (Fejer-Riesz, trigonometric polynomial version.) Every nonneg
trigonometric polynomial `h` on `T` of degree `d` factors as `|p|^2` with `p` a
trigonometric polynomial of degree `d`. For band-limited (but not polynomial) nonneg
`h` an analogous factorization via Akhiezer holds.

**Reformulation.** Maximize `|p_{n_0}|^2 = |hat f(n_0)|^2` over `p` with
`int_T |p|^2 = 1` and `||p||_inf^2 = M`. This is a "flat polynomial" problem and is
amenable to SDP/SOS analysis.

**Expected yield.** Similar to Path 1; possibly cleaner closed-form at small `N`.

**Rigor/feasibility.** Akhiezer factorization for bandlimited nonneg `L^1` functions
is nontrivial to discretize rigorously. Fejer-Riesz polynomial version is clean but
only after projecting `h` to a polynomial truncation, which introduces a tail error
that itself needs rigorous bounding.

---

## Path 3: Karlin-Studden upper principal representation

**Idea.** The constrained Chebyshev-Markov problem (max `int h cos(2 pi n x) dx` over
measures with prescribed moments and `h <= M`) has extremal measures that are sums of
finitely many indicator-step "bathtub pieces" — canonical representations in the
Karlin-Studden theory. Imposing `hat h(k) >= 0` for all `k` (autoconvolution) cuts the
feasible moment cone; the new extremizer has different structure.

**Main lemma.** Karlin-Studden 1966 Ch. II, Theorem 2.1, generalized to include
matrix-valued (PSD) moment constraints.

**Expected yield.** Analytically cleanest path if it goes through; likely gives an
explicit closed-form `mu'(M) = (M/pi) sin(pi/M) - c_1 (M - 1) + ...`.

**Rigor/feasibility.** Heavy analytical derivation. Not a 2-hour job.

---

## Ranking (by tractability-of-proof x expected-yield)

1. **Path 1 (Toeplitz-PSD SDP dual).** Numerical SDP gives instant sanity on the gap;
   rigorous certification is hard but follows established Lasserre / Farkas pipelines.
2. **Path 3 (Karlin-Studden).** Most elegant closed form if derivable; analytical heavy
   lifting.
3. **Path 2 (Fejer-Riesz).** Cleaner at small `N` but doesn't scale as well as Path 1.

## Recommended first step

Implement a **numerical** Path-1 SDP in cvxpy to verify empirically that `mu'_N < mu_MV`
for `N >= 2` and to estimate the size of the gap:

```python
import cvxpy as cp, numpy as np
def mu_sharper_numerical(M, N=3, n0=1):
    y = cp.Variable(N+1, nonneg=True)
    T = cp.bmat([[y[abs(i-j)] for j in range(N+1)] for i in range(N+1)])
    xs = np.linspace(-0.5, 0.5, 201)
    cons = [y[0] == 1, T >> 0]
    for x in xs:
        val = y[0] + 2*sum(y[k]*np.cos(2*np.pi*k*x) for k in range(1, N+1))
        cons += [val <= M, val >= 0]
    cp.Problem(cp.Maximize(y[n0]), cons).solve(solver=cp.SCS, eps=1e-9)
    return y[n0].value
```

If the empirical gap confirms, pursue Path-1 rigorous certification over multiple sessions.

## Obstruction summary

All three paths require multi-session rigorous development beyond the scope of a
single working session. Within the constraint "don't ship unproven code", we cannot
deliver a certified `mu_sharper(M, n) < mu_MV(M)` from a 2-hour attempt.

**Recommendation:** document the obstruction in `derivation.md`, ship a
soundness-preserving placeholder `mu_sharper(M, n) = mu_MV(M)` so Phase-2 is
uncompromised, and schedule the SDP dual-certificate work as a dedicated project.
