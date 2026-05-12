# Path B — Krein-Boas-Kac SDP (DEAD)

Order-1 Lasserre / Schur-lift SDP attempt to close the Hyp_R hypothesis used by the Path A conditional 1.378 bound. Did not close Hyp_R; closed because the relaxation is too loose and the underlying obstruction is structural.

## Goal

Prove unconditionally:

> For every nonneg pdf f on [-1/4, 1/4] with `int f = 1` and `||f*f||_infty <= M_max = 1.378`,
>
> `||f*f||_2^2 <= (log 16 / pi) ||f*f||_infty ~ 0.88254 M`.

If proved, gives C_{1a} >= 1.378 unconditionally — a +0.10 jump over the Cloninger--Steinerberger value 1.2802 (arXiv:1403.7988).

## Setup

Variables: `f_hat(n) = a_n + i b_n` for n = 1, ..., N (with `f_hat(0) = 1`). Set `y_n := |f_hat(n)|^2`.

Constraints encoded:

| # | Constraint | Encoding |
|---|------------|----------|
| 1 | MO 2.14: `y_n <= mu(M) = M sin(pi/M) / pi` | Linear |
| 2 | Bochner on f (f >= 0) | Hermitian Toeplitz `T = [f_hat(i-j)]` PSD |
| 3 | Krein-Boas-Kac (supp f subset [-1/4, 1/4]) | Localizing Hermitian Toeplitz `L = [(p . f_hat)(i-j)]` PSD with `p(x) = 1/16 - x^2` |
| 4 | Parseval | `sum y_n = (K - 1) / 2` where `K = ||f||_2^2 >= 2` |

Objective: upper-bound `max sum y_n^2`. If `max sum y_n^2 <= 0.108`, then `c_emp <= 0.88254` and Hyp_R closes.

Implementation: `kbk_sdp.py` (order-1 Schur lift `V succeeq y y^T` with linear cut `V_{nn} <= mu y_n`); real-form Hermitian Toeplitz (size `2(N+1) x 2(N+1)`); localizing real-form via `p_hat(k)`.

## Numerical results at M = 1.378, mu(M) = 0.333, target 0.108

| K_upper | Path A loose | + phase-Bochner | + KBK localizing |
|---------|--------------|------------------|---------------------|
| 2.0 | 0.967 | 0.967 | 0.967 |
| 2.5 | 1.088 | 1.088 | 1.088 |
| 3.0 | 1.209 | 1.209 | 1.209 |
| 4.0 | 1.451 | 1.451 | 1.451 |

KBK adds **nothing** at this Lasserre order. SDP returns `mu * sum y_n = mu * (K-1)/2`, the loose Path A bound. Verified by direct computation at N=16, K_trunc=10: `max sum y_n = 5.33` with KBK + Bochner + MO 2.14 (vs Cauchy-Schwarz minimum 0.5). Phases can be chosen to satisfy KBK with all `y_n = mu`.

## Why it didn't close

For asymmetric admissible f, K = `||f||_2^2` has **no analytical upper bound** in terms of `M = ||f*f||_infty` — the same K-unboundedness obstruction that defeats Path A directly (CLAUDE.md F4 / MEMORY: project_path_a_unconditional_status). The order-1 Lasserre / Schur-lift relaxation only uses linear cuts on `V_{nn}` and PSD constraints on the moment matrix at order 1; these do not capture the deeper coupling between admissibility at small M and `sum y_n^2`.

## What would unlock

Order-2 Lasserre with quartic moment matrix `M_2` of size `~2N^2 x 2N^2` (e.g. ~511x511 at N=15). Implementation effort 1-2 weeks; substantial solver cost per evaluation. Whether order-2 actually closes Hyp_R is unknown — theoretical analysis suggests partial closure at best. User constraint forbids large-SDP routes, so this is not on the active path.

## References

- [../proof_outline.md](../proof_outline.md), [../formalization.md](../formalization.md)
- [path_a_holder.md](path_a_holder.md), [master_attacks.md](master_attacks.md), [lasserre.md](lasserre.md)
- `delsarte_dual/restricted_holder/derivation.md` — conditional 1.378 chain that Path B would have unlocked.
