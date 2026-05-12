# Correction note: M_recipe → M_optimal

**Date:** 2026-04-29.

## What changed

The headline conditional bound under Hyp_R($\log 16/\pi$, $1.51$) was
**downgraded** from

  ~~$M_{\mathrm{recipe}} = 1.37925062005091\ldots$ (MV's $z_1 = 0.50426$ fixed)~~

to the rigorous

  $M_{\mathrm{optimal}} = 1.37842197377541\ldots$ (inf over admissible $z_1$).

A gap of $\sim 8 \times 10^{-4}$.

## Why the change

To prove $C_{1a} \ge M^*$ from the modified MV master inequality

$$
\frac{2}{u} + a_*
\;\le\; M + 1 + 2 z_1^2 k_1 + \sqrt{c\,M - 1 - 2 z_1^4}\,
                                    \sqrt{\|K\|_2^2 - 1 - 2 k_1^2}
\quad(*)
$$

we need: *for every admissible $z_1 = |\hat f(1)|$, $(*)$ fails at $M < M^*$*.
Equivalently, $M^* = \inf_{z_1} M_{\mathrm{master}}(z_1)$ — the **infimum**
over the admissible $z_1$ range, not any single specific $z_1$.

For $c = 1$ (ordinary Hölder), it happens that the interior critical
point $z_1^*$ of (the right-hand side of) $(*)$ as a function of $z_1$
*exceeds* the Lemma 3.4 boundary $\sqrt{\mu(M)} = \sqrt{M\sin(\pi/M)/\pi}$,
so the boundary $z_1 = \sqrt{\mu(M^*)} \approx 0.50426$ binds and *is* the
argmin of $M_{\mathrm{master}}$. MV's recipe of plugging $z_1 = 0.50426$
verbatim therefore coincides with the rigorous worst-case at $c = 1$ —
this is what reproduces MV's $1.27481$.

For $c < 1$ (the restricted-Hölder regime, including $c = \log 16/\pi$),
the interior critical point $z_1^*$ moves *inside* the admissible range
(because $cM - 1$ shrinks faster than $\mu(M)$ as $c$ decreases). So
$z_1^*$ becomes the actual argmin of $M_{\mathrm{master}}$, and the
recipe value $z_1 = 0.50426$ — which sits *past* $z_1^*$ on the
*increasing* leg of $M_{\mathrm{master}}(z_1)$ — over-claims.

At $c = \log 16/\pi$:
- Interior $z_1^* \approx 0.4882$ (admissible: $\sqrt{\mu(M)} \approx 0.5767$).
- Recipe $z_1 = 0.50426 > z_1^*$, on the increasing leg.
- $M_{\mathrm{recipe}}(0.50426) \approx 1.37925 > M_{\mathrm{optimal}}(z_1^*) \approx 1.37842$.

The original brief specified $z_1 = 0.50426$ fixed across $c$, which is
the source of the over-claim. The corrected version computes
$M_{\mathrm{optimal}}(c) = \inf_{z_1} M_{\mathrm{master}}(z_1)$.

## What was kept (unchanged)

- The Boyer–Li dilation argument and the conclusion that the rescaled
  witness has $\|g*g\|_\infty \approx 1.652 > 1.51$ (so the restricted
  conjecture is not disproved by Boyer–Li). See `derivation.md` §2.
- The empirical Hölder-ratio computation $c_{\mathrm{emp}} \approx 0.589$
  for MV's near-extremizer in `sidon_extremizer_ratio.py`.
- The fetch of `coeffBL.txt` from the Boyer–Li repo.
- The MV 119 cosine coefficients re-parsed at high dps for 40-digit accuracy.

## What was added

- `conditional_bound_optimal(c, dps)`: closed-form rigorous bound via the
  no-$z_1$ form (the master equation at the interior critical point of
  the right-hand side w.r.t. $z_1$).
- `conditional_bound_optimal_bisection(c, dps)`: golden-section cross-check.
- `_admissible_z1_max_sq(M, c, K2, k1)`: combined Lemma 3.4 + Hyp_R sqrt
  domain bound on $z_1^2$.
- `derivation.md` §Z: detailed explanation of why $z_1 = 0.50426$ is not
  the rigorous choice for $c \ne 1$.
- New tests in `test_restricted_holder.py`:
  - `test_optimal_bound_is_rigorous_inf`: verifies the rigorous bound,
    including positive residual at recipe $z_1$ confirming sub-optimality.
  - `test_master_inequality_holds_for_all_admissible_z1`: 1000-point
    grid sweep to verify no violation across the admissible $z_1$ range.
  - `test_optimal_matches_bisection_cross_check`: closed-form vs.
    golden-section search agreement.

## What was renamed

- `conditional_bound` → `conditional_bound_recipe` (kept as a back-compat
  alias so old import paths continue to work, but new code should call
  either `conditional_bound_optimal` or `conditional_bound_recipe`
  explicitly).
- `test_reproduces_1_379` → `test_recipe_value_1_37925`.

## Headline numbers

| Quantity                | Value                                      |
|-------------------------|--------------------------------------------|
| $M_{\mathrm{optimal}}(\log 16/\pi)$ — **HEADLINE (rigorous)** | $1.3784219737754177283997025524314107461235\ldots$ |
| $M_{\mathrm{recipe}}(\log 16/\pi)$ — historical, sub-optimal | $1.3792506200509102615605241383073240383153\ldots$  |
| $M_{\mathrm{optimal}}(1)$ — ordinary Hölder (matches MV)      | $1.2748375877276427724312\ldots$            |
| $M_{\mathrm{recipe}}(1)$ — coincides with MV's published 1.27481 to 4 dps | $1.2748392497234945494069\ldots$ |
