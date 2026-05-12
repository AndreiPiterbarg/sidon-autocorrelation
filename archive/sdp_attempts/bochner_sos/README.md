# Path B: Continuum Bochner-SOS dual certificate for C_{1a}

## What this is

A certificate-generation pipeline for proving lower bounds on the Sidon
autocorrelation constant by exhibiting a single nonneg dual function
`g(t)` on `[-1/2, 1/2]` and the associated copositivity / SOS cert.

## The Theorem (and what it actually says)

For any nonneg `g` on `[-1/2, 1/2]`, `g != 0`,

```
   C_{1a}  >=  M(g)  :=  ( inf_{ mu in P([-1/4, 1/4]) }  int int g(x+y) dmu(x) dmu(y) )
                          / int_{-1/2}^{1/2} g(t) dt
```

Cleanest case: take `mu` to be a probability *measure* (point masses
allowed in the sup-over-`mu` formulation, but for the inf over `f >= 0`
we need `f` to be a function so that `f*f` is bounded; any continuous
nonneg `f` with `int f = 1` makes the integral well-defined). For
discrete `mu` supported on a `d`-grid, the inf over the simplex is a
nonconvex QP with kernel `G[i,j] = g(x_i + x_j)`.

## What's been built

| File | Purpose |
|---|---|
| `iv_core.py` (in `cert_pipeline/`) | reused: interval arithmetic via `mpmath.iv` |
| `m_g_eval.py` | numerical evaluator for `M(g)` via Shor SDP relaxation (rigorous lower bound on the true inf, *hence* on `C_{1a}`) |
| `g_candidates.py` | candidate `g` constructors: constant, piecewise-constant, piecewise-linear, cosine, central-bump |

Self-tests pass:
- `g = 1` (constant) gives `M = 1.0` (matches closed form)
- `g = central bump` gives `M ~ 0.8` (Shor relaxation; true value likely higher)
- piecewise hats and cosines give the expected scaling

## What's NOT been built (intentionally — see honest difficulty assessment below)

- The OUTER joint SDP that *optimizes* `g` to maximize `M(g)`. This is
  the heavy lift and its difficulty is what gates whether Path B beats
  CS 2017's 1.2802.
- Rational rounding of the SDP solution.
- Exact-arithmetic SOS verifier.
- Lean 4 cert checker.

## Honest difficulty assessment (from independent math/software research agents)

**The CS17 LP rate.** Cloninger-Steinerberger 2017 used essentially
this same dual at `d = 24` (48 sub-intervals on `[-1/4, 1/4]`) with
piecewise-constant `g`, and certified `1.2802`. The *gap* `C_{1a} -
val_LP(d)` closes as `O(1/d)` for piecewise-constant `g`; for
piecewise-linear `g` it closes as `O(1/d^2)`. To certify `1.2805`, one
needs roughly `10x` the CS17 compute (about `10^5` CPU-hours, by the
math agent's estimate).

**KLM 2025 strong duality DOES NOT apply** — that framework
(`arXiv:2510.10172`) is for the Turán/Delsarte problem with the *primal*
variable being Fourier-positive (`hat f >= 0`); the autoconvolution
problem has primal `f >= 0` pointwise, structurally different. **Strike
this from the synthesis.**

**The right Positivstellensatz is COPOSITIVE, not generic SOS.** The
inner inequality `int int g(x+y) f f >= alpha int g` reduces to:

```
   for all a in R^d_{>=0}:   a^T (G - alpha * avg(g) * J) a  >=  0
```

where `G[i,j] = g(x_i+x_j)` is a symmetric Toeplitz matrix and `J` is
the all-ones matrix.  This is a *copositive* statement, certified
either by the Parrilo SDP at degree 2 or by Pólya / Handelman LP on the
simplex.  Generic SOS (no Pólya multipliers) does NOT give a converging
hierarchy here.

**The big computational win identified by software agent:**
substituting `u = x+y, v = x-y` collapses the 2D positivity constraint
on `[-1/4, 1/4]^2` to a 1D weighted-Putinar problem in `u` with the
weight being the length of the `v`-slice. **This is a 100x speedup on
the SDP**.

## What you should actually do, in order

### Stage 1: validation (1 week)

1. Run `bochner_sos.m_g_eval` on several candidate `g`'s, including the
   actual LP-dual extracted from the user's `val(d=22)` solver (see
   `lasserre/` directory). Confirm `M(g) ~ 1.28` reproduces.
   Decision point: if reproducing CS17 numerically does not hit `1.28`,
   the framework has a bug; fix before proceeding.

2. Implement `optimize_g.py`: outer SDP / LP that maximizes `M(g)` over
   a parametric class of `g`. Start with piecewise-constant `g` at
   `N = 48` (= CS17's resolution) and confirm the result matches CS17's
   `1.2802`. **This is the day the Path B viability is settled.**

### Stage 2: improvement (2-3 weeks)

3. Switch to piecewise-linear `g` at `N = 64-96`. Use the `u, v` change
   of variable. Expected: the SDP solver returns a numerical `M_num`
   somewhere in `[1.281, 1.285]`.

4. Solve at high precision (SDPA-GMP at 80 digits, slow but rigorous
   numerics).

### Stage 3: rigor (3-4 weeks)

5. Apply Magron–Safey-El-Din round-project-lift
   (`https://github.com/magronv/RealCertify`, Maple-based) to round
   numerical Gram matrices to `Q`.

6. Verify the rational SOS certificate exactly. `SageMath` recommended
   for the polynomial arithmetic at this scale.

7. (Optional, +4 weeks) Lean 4 formalization. `mmaaz-git/sostactic`
   provides the SOS-to-Lean glue, currently demo-grade.

## The Big Risk

The combination of:

- CS17 LP gap closing as `O(1/n)` (`O(1/n^2)` for piecewise-linear)
- The math agent's `10x CPU` estimate
- The fact that *17 years passed without anyone improving 1.2802*

means there is a real chance Path B *cannot* break `1.2802` without a
structural insight (e.g., the conjectural extremal `f` shape, cf.
Matolcsi-Vinuesa's `1/sqrt(2x+1/2)` ansatz that they themselves
disproved). If Stage 1 step 2 shows you cannot reproduce `1.2802` at
`N=48` with reasonable solver parameters, this is a signal to retreat
and do Path A (BnB-to-completion) instead.

## References

- Cloninger-Steinerberger 2017, arXiv:1403.7988
- Matolcsi-Vinuesa 2010, arXiv:0907.1379
- White 2022, arXiv:2210.16437 (L^2 case, sharp)
- Boyer-Li 2025, arXiv:2506.16750 (different functional)
- Parrilo PhD 2000 (copositive SDP framework)
- Lasserre 2001 (moment hierarchy)
- Magron-Safey-El-Din 2025, arXiv:2503.11119 (rational rounding)
- Wang-Magron 2022, TSSOS book (sparse SOS code)
- RealCertify: https://github.com/magronv/RealCertify
- TSSOS.jl: https://github.com/wangjie212/TSSOS

## Software dependencies (already verified in this env)

- `numpy`, `scipy`, `cvxpy` (Clarabel default), `mpmath` — all installed.

Will need (Stage 2+):
- Julia + TSSOS.jl + SumOfSquares.jl + JuMP
- MOSEK academic license (free) for high-quality SDP
- SDPA-GMP for high-precision pre-rounding
- SageMath 10 for exact-arithmetic verification
- Maple for RealCertify (or hand-roll Peyrl-Parrilo rounding via `fpylll`)
