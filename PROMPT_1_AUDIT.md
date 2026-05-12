# Prompt 1 — Full audit for publication-grade correctness

**Context — what we are doing.**

We are working to rigorously prove a lower bound on the Sidon
autocorrelation constant `C_{1a}`, where for any nonneg `f : R -> R_{>=0}`
with support `[-1/4, 1/4]` and integral 1,

```
max_{|t| <= 1/2} (f * f)(t) >= C_{1a}.
```

The current best lower bound is Cloninger-Steinerberger 2017's `1.2802`.
We are pushing to prove `C_{1a} >= 1.281` by certifying `val(d) >= 1.281`
for `d = 22` via interval branch-and-bound, where

```
val(d) := min_{mu in Delta_d} max_W TV_W(mu),
TV_W(mu) := (2d/ell) * mu^T A_W mu,
```

with windows `W = (ell, s_lo)` ranging `2 <= ell <= 2d`, `0 <= s_lo <= 2d-ell`.
KKT-correct `mu*` gives `val(22) UB = 1.3093356`, residual `3.4e-14`, so the
target margin is `+0.028`.

**What just happened.**

A 16-core local d=22 BnB run with FOUR new bound-tightening fixes drove
coverage to `99.99999849816%` (uncovered volume `4.5e-13` of state space)
in 3 hours, vs. the prior 192-core d=20 pod stall at `99.80189%` on the
SAME problem. The fixes are:

1. **Epigraph LP extra cuts** in `interval_bnb/bound_epigraph.py`:
   - column-sum RLT (`Sigma_i Y_{ij} = mu_j`)
   - Y-symmetry (`Y_{ij} = Y_{ji}`)
   - diagonal SOS (`Sigma_i Y_{ii} >= 1/d`)
   - midpoint diagonal tangent (`Y_{ii} >= 2 m_i mu_i - m_i^2`)

2. **H_d half-simplex pre-filter** (drop boxes with `lo_int[0] > hi_int[d-1]`)
   in `interval_bnb/symmetry.py`, `bnb.py`, `parallel.py`.

3. **mu\*-anchor cut** in `interval_bnb/bound_anchor.py`: supporting
   hyperplane `f(mu) >= f(mu*) + g.(mu - mu*) + curvature_concession`,
   where `g` is the subgradient at the binding window and the curvature
   concession `min(0, scale_W* * lambda_min(A_W*)) * D2(B)` accounts
   for `A_W` not being PSD in general (eigenvalues to `-1` for `ell=2`
   windows; agent caught this — the supporting-hyperplane bound is
   unsound without the concession).

4. **LP-binding-axis split heuristic** in `interval_bnb/parallel.py`:
   uses the just-solved epigraph LP's `ineqlin` marginals to pick the
   split axis whose McCormick face is most binding.

Both `Box.split`'s `D_SHIFT=60` dyadic-saturation guard and a `splittable`
mask (axes with `hi_int - lo_int >= 2`) are in place to prevent rigor
violations on deep boxes.

**The audit task.**

Please perform a complete, line-by-line audit of `[FILE]`, with the goal
of confirming the implementation is publication-rigorous. Your audit
must verify ALL of the following and report any violations:

**A. SOUNDNESS:**
- Every cut/bound is mathematically valid (write out the proof
  of validity for each one in your report).
- Every floating-point arithmetic is either exact (integer at
  denom `2^60`) or carries an explicit, rigorous safety cushion
  larger than the worst-case rounding error.
- No silent assumptions about `A_W` being PSD, `mu*` being a true
  minimizer, etc.
- Subgradients, supporting hyperplanes, and other convex-analysis
  constructs are applied only where the underlying object is
  actually convex (or with explicit concession otherwise).

**B. RIGOR PARITY WITH THE LP:**
- Where `scipy.linprog` (HiGHS) is used, identify whether the
  certified bound carries enough cushion against HiGHS's primal/
  dual feasibility tolerance (default `1e-7`) — NOT just float
  arithmetic error.
- Flag any place where `lp_val` is used with only an `n_vars *
  1e-14` cushion as POTENTIALLY UNSOUND for publication.
- Identify whether a Neumaier-Shcherbina integer dual cert is
  present, partial, or missing.

**C. INTEGER ARITHMETIC:**
- Box endpoints stored as Python ints at denom `2^60` (`D_SHIFT=60`).
- Every place where float -> int rounding occurs: confirm the
  rounding direction (floor vs ceil) is conservative for the
  resulting bound.
- Fraction arithmetic (where used) is always exact.

**D. STATE / RACE / CONCURRENCY:**
- Multi-process worker state (`mp.Value`, `mp.Queue`, donation logic)
  is correct under all interleavings.
- Volume accounting (`closed_vol`, `in_flight`, `coverage_fraction`) is
  conservation-of-mass: every box that enters the system either
  contributes to `closed_vol` or is `in_flight` at termination.
- The H_d cut and saturation continue paths update accounting
  correctly.

**E. EDGE CASES:**
- d=2 / d=3 small-d edge cases.
- boxes touching the simplex boundary.
- boxes that fully contain or exclude `mu*`.
- LP infeasibility / numerical failure.
- `mu_star_d{d}.npz` missing / wrong shape / wrong d.

**F. PUBLICATION CHECKLIST:**
- Is there any TODO, FIXME, "WARNING: NOT YET RIGOROUS", or
  similar comment indicating known gaps?
- Is every claim in the docstring backed by code?
- Are there dead branches or removed code paths that mention
  soundness claims that are no longer enforced?

For EACH issue you find:
- Cite the exact `file:line`.
- State precisely why it's a problem (mathematical or
  implementation-level).
- Classify as **BLOCKER** (kills publication rigor), **CONCERN** (works
  in practice but not provable), **NIT** (cosmetic).
- Propose a concrete fix.

If everything checks out, say so explicitly with the proof obligations
satisfied. Don't summarize approvingly — be adversarial and search
for failure modes. Assume the goal is to publish; your job is to find
the one thing that would let a reviewer reject.

Working dir: `C:/Users/andre/OneDrive - PennO365/Desktop/compact_sidon`
