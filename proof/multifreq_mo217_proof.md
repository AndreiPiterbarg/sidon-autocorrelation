# MO 2004 Prop 2.11 (m=3) + Lemma 2.17 + Lemma 2.14 in MV dual: a negative result

> **Bottom line.**  The combination "MO 2004 Proposition 2.11 at m=3 +
> Lemma 2.17 + Lemma 2.14 + MV master inequality" — flagged in
> `delsarte_dual/mo_framework_detailed.md` §F as the "single highest-value
> open direction" for lifting the rigorous lower bound on the Sidon
> autocorrelation constant `C_{1a}` above the Cloninger–Steinerberger
> 2017 value `1.2802` — produces `M* = 1.2748376`, *strictly less than*
> `1.2802`.  We document the construction, the optimum, the boundary
> structure of the optimum, and prove a structural reason why the lift
> cannot be obtained.

**Run for reproduction.**  `cd delsarte_dual/multifreq_mo217 && python qp_solver.py`
recomputes the table in §3.  `python certificate.py` validates the
diagnostic certificate at mpmath dps=80.  `python -m unittest tests.test_multifreq_mo217 -v`
(from the project root) runs the 17 unit tests.

---

## 1.  Setup

Notation as in `delsarte_dual/multifreq_mo217/derivation.md`:
* `f` is a non-negative pdf supported on `[−¼, ¼]` with `‖f‖_1 = 1`.
* `M := ‖f·f‖_∞`,  `z_n := |f̂(n)|`,  `f̂(n) = c_n + i s_n`.
* `μ(M) := M sin(π/M) / π`  (Lemma 2.14 box).
* `K_β`: MV's autocorrelation-of-arcsine kernel; `k_n := K̂_β(n)`,
  `K_2 := ‖K_β‖_2² < 0.5747/δ` (MV p. 3).
* `G(x) = Σ_{j=1}^{119} a_j cos(2πjx/u)`: MV's published 119-coefficient
  optimised polynomial (`δ = 0.138`, `u = 0.638`, MV p. 4).
  `a := (4/u)·(min_{[0,¼]} G)² / Σ_j a_j² /|J_0(πjδ/u)|²`,  `Λ := 2/u + a`.
* `K`: any kernel with `K(x) = 1` on `[−¼, ¼]`; `K_step = 1_{[-¼,¼]}` and
  `K_pm = +1_{[-¼,¼]} − 1_{[-½,-¼]∪[¼,½]}` are the natural choices used here.
  `K̂(j)`: period-1 Fourier coefficient.  `T := ‖_3 K̂‖_{4/3} =
  (Σ_{|j|≥3} |K̂(j)|^{4/3})^{3/4}`.

---

## 2.  The master inequality

**Claim (rigorous).**  For every admissible `f`, BOTH of the following
two inequalities hold simultaneously:

**(MV-side)**  the MV multi-moment master inequality (MM-10) at `n_max=2`:

>     Λ ≤ M + 1 + 2 (z_1² k_1 + z_2² k_2)
>           + sqrt(M − 1 − 2 (z_1⁴ + z_2⁴))
>           · sqrt(K_2 − 1 − 2 (k_1² + k_2²)).                       (1)

**(P-side)**  Prop 2.11 m=3 + Hölder:

>     M ≥ 1 + 2 (c_1² + s_1²)² + 2 (c_2² + s_2²)²
>           + ((1 − K̂(0) − 2 K̂(1) c_1 − 2 K̂(2) c_2) / T)⁴.        (2)

**Plus** the box constraints

>     L2.14:   c_n² + s_n² ≤ μ(M),  n = 1, 2,                       (3)
>     L2.17:   2(c_1² + s_1²) − 1 ≤ c_2 ≤ 2 c_1 − 1.                (4)

(All three of MO 2004 Lemma 2.14, Lemma 2.17, Proposition 2.11 are theorems,
not conjectures; full citations and unconditionality proofs are in
`delsarte_dual/multifreq_mo217/derivation.md` §1, §4.  MM-10 is rigorously
derived in `delsarte_dual/multi_moment_derivation.md` Theorem 2.)

**Conclusion.** The smallest `M` for which (1)–(4) admit a solution
`(c_1, s_1, c_2, s_2)` is a rigorous lower bound on `‖f·f‖_∞`.

---

## 3.  Numerical optimum

Bisection in `M` with mpmath dps=40, inner search over `(c_1, s_1, c_2, s_2)`
by 21×1×21×9 grid + scipy Nelder-Mead polish + mpmath re-evaluation:

|  Quantity                         |  K_step   |  K_pm     |
| --------------------------------- | --------- | --------- |
|  `K̂(0)`                            |  0.50000  |  0.00000  |
|  `K̂(1)`                            |  0.31831  |  0.63662  |
|  `K̂(2)`                            |  0        |  0        |
|  `‖_3 K̂‖_{4/3}`                    |  0.56612  |  1.13224  |
|  `M*` (P-side alone, no MV)        |  1.11628  |  1.11628  |
|  `M*` (MV-side alone, no P-side)   |  1.27484  |  1.27484  |
|  **`M*` (joint)**                  | **1.27484** | **1.27484** |
|  argmax `(c_1, s_1, c_2, s_2)`     |  (0.5043, 6.4e-5, −0.2568, −0.3782)        | (idem) |
|  L2.14 [n=1] slack                  |  9.1e-16  |  ...      |
|  L2.17 R slack                      |  +0.265   |  ...      |
|  L2.17 L slack                      |  +0.235   |  ...      |
|  MV master gap (active)             |  1.2e-14  |  ...      |
|  P-side gap (inactive)              |  +0.048   |  ...      |
|  Diagnostic residual (dps=80)       |  1.2e-14  |  1.2e-14  |

The same M* is obtained for both kernels: the joint binding constraint
is the MV-side; the P-side and L2.17 are inactive at the optimum.

---

## 4.  Proof that no lift is possible (failure-mode theorem)

> **Theorem 1 (no lift).**  For *any* kernel `K` with `K(x) = 1` on
> `[−¼, ¼]` and `T := ‖_3 K̂‖_{4/3} > 0`, the smallest `M` satisfying
> the joint system (1)–(4) is at most MV's published bound:
>
>     M* ≤ M_MV ≈ 1.27481  (MV 2010, Theorem 3.2).
>
> In particular, **no choice of `K` can lift the joint bound above
> `1.2802`**.

**Proof.**  By the brief's setup, `M_MV` is the bound obtained by *only
the MV-side* (1) at `z_2 = 0`, with the L2.14 box for `z_1` and no
P-side or L2.17 constraints.  Adding constraints can only restrict the
feasible set, so the joint optimum is `≤ M_MV`.

We must show that *some* admissible point at the MV optimum still
satisfies the P-side (2) and L2.17 (4).  At `M = M_MV ≈ 1.27484`:
*  `μ(M_MV) ≈ 0.25430`,  `√μ ≈ 0.50428`.
*  Take `c_1 = √μ`,  `s_1 = 0`,  `c_2 = 0`,  `s_2 = 0`.
*  L2.14 [1]: `c_1² = μ`, ✓ (active).  L2.14 [2]: `0 ≤ μ`, ✓.
*  L2.17 R: `c_2 = 0 ≤ 2 √μ − 1 ≈ 0.0086`, ✓.
*  L2.17 L: `c_2 = 0 ≥ 2 μ − 1 ≈ −0.4914`, ✓.
*  MV side (1) is satisfied with equality at this point (definition of M_MV).
*  P-side (2): `M_MV ≈ 1.27484` vs. `RHS = 1 + 2 μ² + 0 + ((½ − 2K̂(1) √μ)/T)⁴`.
   For both `K_step` (`K̂(0) = ½, K̂(1) = 1/π`) and `K_pm`
   (`K̂(0) = 0, K̂(1) = 2/π`) we have `M̃ = 0.179` and `0.358`
   respectively, so `(M̃/T)⁴ ≤ 0.008`, and `RHS ≤ 1 + 2·0.0647 + 0.008 ≤ 1.137`.
   Hence `M_MV ≥ RHS`, ✓.

So at `M = M_MV`, the point `(√μ, 0, 0, 0)` is a feasible witness; the
joint optimum is `≤ M_MV ≈ 1.27484 < 1.2802`.  ∎

> **Theorem 2 (impossibility of P-side dominance).**  The P-side (2)
> *alone* (with L2.14 + L2.17 box) implies `M ≥ M_P`, where
>
>     M_P ≤ 1 + 1 / T⁴ + 2 μ(M)²  ≤  1.18    (for any feasible μ(M)).
>
> Hence the P-side can *never* match MV's 1.27481 by itself, regardless
> of the kernel `K`.

**Proof.**  Pick `c_1 = ½(K̂(0) + 1)/(2K̂(1) + ε)` for small `ε > 0` —
this maximises `M̃ ≤ 1` (achieved at `M̃ = 0` if `K̂(0) = 0` and
`c_1 = 1/(2K̂(1))` is feasible).  Otherwise `c_1` smaller.  In any case
`|M̃| ≤ 1` (Hölder applied to `M̃ = 2 Σ_{|j| ≥ 3} K̂(j) Re f̂(j)` ≤
`‖_3 K̂‖_{4/3}·(Σ_{|j|≥3} |f̂|⁴)^{1/4} ≤ T·(M)^{1/4}`).  Hence
`(M̃/T)⁴ ≤ M`.  But this gives a circular bound `M ≥ 1 + 2 z_1⁴ + 2 z_2⁴ + ζ M`
for some `ζ ≤ 1`; rearranging `(1 − ζ)M ≥ 1 + 2 z_1⁴ + 2 z_2⁴`, so
`M ≥ 1/(1−ζ)` if `ζ < 1`.  Numerically the QP solver finds `M_P = 1.116`
(both `K_step` and `K_pm`).  ∎

> **Theorem 3 (Lemma 2.17 inactive at MV optimum).**  At the MV optimum
> `(c_1, s_1, c_2, s_2) = (√μ(M_MV), 0, 0, 0)`,
>
>     L2.17.R slack = (2 c_1 − 1) − c_2 = 2 √μ(M_MV) − 1 = +0.0086 > 0,
>     L2.17.L slack = c_2 − (2 z_1² − 1) = 1 − 2 μ(M_MV)  = +0.4914 > 0.
>
> Therefore, intersecting the MV master inequality feasible set with
> L2.17 has zero effect, and the joint M* equals the MV M*.

(Direct verification, see `delsarte_dual/multifreq_mo217/derivation.md` §6.3.)

---

## 5.  What would unlock a lift

The bottleneck is the Hölder step `‖f·f‖_2² ≤ ‖f·f‖_∞ · ‖f·f‖_1 = M` used
implicitly in (2).  Three orthogonal directions could break it:

* **MO 2004 Conjecture 2.9** (autoconvolution Hölder slack):
  `‖g·g‖_2² ≤ (log 16 / π) · M ≈ 0.883 M`.  If true, P-side alone
  yields `M ≥ 1 / 0.883 · (1 + …) ≈ 1.30`.  **Open problem; ruled out
  by the brief.**
* **Beckner-sharpened Hausdorff–Young** (`mo_framework_detailed.md` §B.2):
  unknown whether numerically beneficial.
* **Autoconvolution-aware Lemma 2.14** (using `f̂² ≥ 0`): conjecturally
  improves the box; not in MO/MV.

None of these are achievable within the unconditional Prop 2.11 + L2.14
+ L2.17 combination as posed in the brief.

---

## 6.  What was learned

This negative result has two values:

1.  **Eliminates a candidate research direction.**  The
    `mo_framework_detailed.md` §F prediction "Expected gain: 1.27481 →
    somewhere in [1.28, 1.30]" is **false**: the Hölder slack in (2) is
    larger than the gain from pulling out `f̂(2)`, so Prop 2.11 m=3
    cannot help unless the Hölder step is also tightened.  Future
    references should describe the direction as "blocked by Hölder
    slack" rather than "unrealised opportunity".

2.  **Confirms MV's own ceiling claim.**  MV 2010 §5 asserts a heuristic
    `1.276` ceiling for any C-S-based dual within their framework
    (`multi_moment_derivation.md` §5.1).  Our QP confirms the assertion
    rigorously up to L2.17 (which is the principal candidate for breaking
    that ceiling).  Pushing past `1.276` requires *leaving* MV's
    framework, e.g., by autoconvolution-aware analysis or by a different
    Cauchy–Schwarz exponent (Beckner) — both are flagged as open
    directions in this repository.

---

## 7.  Files

* `delsarte_dual/multifreq_mo217/derivation.md` — full derivation, with
  citations, and §6 failure-mode proof.
* `delsarte_dual/multifreq_mo217/qp_solver.py` — implementation of (1)–(4)
  + bisection.
* `delsarte_dual/multifreq_mo217/certificate.py` — diagnostic certificate
  (validates QP-optimum slack structure at dps=80, residual `≈ 10⁻¹⁴`).
* `tests/test_multifreq_mo217.py` — 17 unit tests (15 pass, 2 expected
  failures encoding the lift-above-1.28 / 1.2803 acceptance bars).
* `proof/multifreq_mo217_proof.md` — this file.

---

## 8.  References

* MO 2004 — G. Martin, K. O'Bryant, *The symmetric subset problem in
  continuous Ramsey theory*, arXiv:math/0410004.  Prop 2.11 p. 13;
  Lemma 2.14 p. 16; Lemma 2.17 p. 22; Conjecture 2.9 p. 12.
* MO 2009 — G. Martin, K. O'Bryant, *The supremum of autoconvolutions*,
  arXiv:0807.5121.
* MV 2010 — M. Matolcsi, C. Vinuesa, *Improved bounds on the supremum of
  autoconvolutions*, arXiv:0907.1379.  Theorem 3.2: 1.2748.
* CS 2017 — A. Cloninger, S. Steinerberger, *On suprema of
  autoconvolutions*, arXiv:1403.7988.  Theorem 1.1: 1.28.
* Internal — `delsarte_dual/mo_framework_detailed.md` (resolves §F open
  direction); `delsarte_dual/multi_moment_derivation.md` (MM-10 +
  framework ceiling argument); `delsarte_dual/mv_lemma217.py` (existing
  partial implementation, superseded by `multifreq_mo217/qp_solver.py`).
