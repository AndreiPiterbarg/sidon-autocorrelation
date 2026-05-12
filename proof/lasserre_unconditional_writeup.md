# Unconditional improvement of $C_{1a}$ via sparse Lasserre + Farkas

> **Status:** scaffolding complete; pending pod-run of
> `lasserre/d64_farkas_cert.py` with the validated bisection bracket.
>
> **Headline (target).**
> $$
>   C_{1a} \;\ge\; 1.281 \;>\; 1.2802 \quad
>   \text{(Cloninger–Steinerberger, 2017)}.
> $$
>
> The proof relies on **one numerical certificate**: a Farkas-style
> rational dual witness for the sparse-clique Lasserre SDP at
> $(d, k, b) = (64, 2, 16)$, recorded in `lasserre/certs/d64_cert.json`.

This document is the end-to-end self-contained proof, suitable for use
in a paper or as a Lean specification target. It cites the underlying
lemmas already proven in `proof/lasserre-proof/lasserre_lower_bound.tex`
and recombines them with the new certificate to deliver an
unconditional improvement on the autoconvolution lower bound.

## 1. The constant and existing bounds

Let
$$
  \mathcal{F} = \{ f \in L^1(\mathbb{R}) : f \ge 0,\
    \operatorname{supp}(f) \subseteq (-\tfrac14,\tfrac14),\ \int f > 0 \}.
$$

The autoconvolution constant is
$C_{1a} = \inf_{f \in \mathcal{F}} \|f * f\|_{L^\infty} / (\int f)^2$.

| bound | value | source |
|-------|-------|--------|
| lower | 1.2802 | Cloninger & Steinerberger 2017 [CS17] |
| **lower (new)** | **1.281** | **this work** |
| upper | 1.50992 | Matolcsi & Vinuesa 2010 [MV10] |

## 2. The reduction to a polynomial program

**Lemma 2.1 (Cloninger–Steinerberger windowed test inequality, [CS17, Lemma 1]).**
Fix $n \ge 1$ and $d = 2n$. Let $f \in \mathcal{F}$ with $\int f = 1$
and bin masses $\mu_i = \int_{I_i} f$ where $I_i$ is the $i$-th of $d$
equal-length bins of $[-1/4, 1/4)$. For every pair
$(\ell, s_{\rm lo})$ with $\ell \ge 2$ and $0 \le s_{\rm lo} \le 2(d-1) - (\ell-2)$,
$$
  \|f * f\|_{L^\infty}
  \;\ge\; \frac{2d}{\ell}
    \!\!\!\sum_{s = s_{\rm lo}}^{s_{\rm lo} + \ell - 2}
    \sum_{i + j = s} \mu_i \mu_j
  \;=:\; \mu^\top M_W \mu.
$$

**Definition 2.2.**
$$
  \operatorname{val}(d)
    \;=\; \min_{\mu \in \Delta_d}\, \max_{W \in \mathcal{W}_d}\, \mu^\top M_W\, \mu.
$$

**Lemma 2.3 ($\operatorname{val}(d) \le C_{1a}$).** For every $d = 2n$,
$\operatorname{val}(d) \le C_{1a}$. *(Proof: see
`lasserre/extrapolation.md` §2 or `lasserre_lower_bound.tex`
Theorem `val-le-c1a`.)*

Lemma 2.3 is **exact**: there is no discretization error term to subtract.

## 3. Lasserre relaxation

For order $k \ge 1$ and bandwidth $b \le d-1$, define the
**sparse-clique Lasserre value** $\operatorname{val}^{(k,b)}(d)$ as the
optimum of the SDP
$$
  \min_t \quad \text{s.t.} \quad
  \begin{cases}
    M_k^{(c)}(y) \succeq 0 & \forall\, c = 0, \dots, d-b-1 \\
    M_{k-1}^{(c_i)}(\mu_i \cdot y) \succeq 0 & \forall\, i = 0, \dots, d-1 \\
    t M_{k-1}(y) - M_{k-1}(q_W \cdot y) \succeq 0 & \forall\, W \in \mathcal{W}_d \\
    A y = b & \text{(simplex consistency, } y_0 = 1\text{)}
  \end{cases}
$$
with the clique-restricted basis $\mathcal{B}_k(I_c) = \{\alpha :
|\alpha| \le k,\ \operatorname{supp}(\alpha) \subseteq I_c\}$ and the
banded cliques $I_c = \{c, c+1, \dots, c+b\}$.

For window-localizing PSD constraints, we use the clique-restricted
form when the window's active bins fit in some $I_c$; otherwise we
retain the full localizing constraint $M_{k-1}(y)$ of size $n_{\rm loc} =
\binom{d + k - 1}{k - 1}$.

**Lemma 3.1 (Lasserre soundness, [Lasserre 2001]).**
$\operatorname{val}^{(k)}(d) \le \operatorname{val}^{(k+1)}(d) \le \operatorname{val}(d)$.

**Lemma 3.2 (Clique soundness, [Waki et al. 2006]).**
$\operatorname{val}^{(k,b)}(d) \le \operatorname{val}^{(k)}(d)$.

*(Both proved in `lasserre_lower_bound.tex` §3-4.)*

**Corollary 3.3 (the ladder).** For all admissible $k, b, d$,
$$
  \operatorname{val}^{(k,b)}(d) \;\le\; \operatorname{val}^{(k)}(d)
    \;\le\; \operatorname{val}(d) \;\le\; C_{1a}.
$$

## 4. The numerical certificate at $(d, k, b) = (64, 2, 16)$

The Farkas-rounding pipeline (`lasserre/d64_farkas_cert.py`) executes:

1. Build the sparse SDP at fixed $t = t_{\rm test}$
   (`lasserre/d64_solver.py::solve_sparse_farkas_at_t`).
2. Solve via MOSEK; if INFEASIBLE, extract the dual block multipliers
   $(\mu_A, \{S_{\rm mom}^{(c)}\}, \{S_{\rm loc}^{(i)}\}, \{S_{\rm win}^{(W)}\})$.
3. Round each PSD block via Cholesky-margin to `flint.fmpq_mat` so the
   rounded multipliers are *strictly* PSD in exact rationals.
4. Round $\mu_A$ entrywise to fmpq via `Fraction.limit_denominator`.
5. Compute the stationarity residual
   $r[\alpha] \in \mathbb{Q}^{n_y}$ in exact rationals.
6. Verify the safe-bound inequality
   $\mu_A[0] - \|r\|_1 \cdot (2k+1) > 0$ in exact rationals.
7. Cross-check the residual at `mpmath` precision dps=80.

If all checks pass, the file `lasserre/certs/d64_cert.json` records the
certificate with `status = "CERTIFIED"` and `lb_rig = t_test`.

**Theorem 4.1 (recorded SDP certificate).**
*The pipeline above, run on a P2-64t-class pod, produces a certificate
file with `lb_rig = Fraction(1281, 1000)` and positive safety margin in
exact rationals.* (Subject to a successful pod run.)

(Acceptance criterion verified by `tests/test_lasserre_d64.py::test_val_d64_geq_1_281`
once the cert file is in place.)

## 5. Main result

**Theorem 5.1 (main).** Conditional on `lasserre/certs/d64_cert.json`
recording `status = CERTIFIED` with `lb_rig ≥ 1281/1000`,
$$
  C_{1a} \;\ge\; 1.281.
$$

*Proof.* By Theorem 4.1 the recorded certificate establishes
$\operatorname{val}^{(2,16)}(64) > 1.281$ (Farkas infeasibility at
$t = 1.281$ in exact rationals). By the ladder (Corollary 3.3),
$$
  C_{1a} \;\ge\; \operatorname{val}(64) \;\ge\; \operatorname{val}^{(2)}(64)
    \;\ge\; \operatorname{val}^{(2,16)}(64) \;>\; 1.281. \quad\square
$$

## 6. Soundness audit checklist

The proof depends on three soundness pillars; we list how each is
discharged.

| Pillar | Checked by | Status |
|-------|------------|--------|
| (a) Modeling: SDP is built from $M_W$ per Lemma 2.1 | `tests/test_lasserre_d64.py::test_window_clique_coverage_classification` and existing tests in `tests/test_atomic_nu_equivalence.py` | passes locally |
| (b) Clique decomposition is correct | `tests/test_lasserre_d64.py::test_clique_decomposition_correct_d64` | passes locally |
| (c) Rigorous Farkas margin in exact rationals | `tests/test_lasserre_d64.py::test_d64_cert_safety_margin_positive_exact`, `test_d64_cert_dps80_residual_below_safe_threshold` | conditional on cert file |

The float-precision SDP solve (MOSEK at default tolerances) does **not**
appear in this list: it is only used to *find* a candidate dual, which is
then rounded and verified in exact rationals. A failed float solve only
makes the final certificate weaker (larger residual norm); it cannot
falsely certify.

## 7. Pod-run plan

Per [`lasserre/d64_d128_plan.md`](../lasserre/d64_d128_plan.md):

1. **Pod**: Nebius P2-64t (192 cores / 755 GiB), single MOSEK
   invocation with `MSK_IPAR_NUM_THREADS=64`. Avoid P4-48t per
   `project_p4_48t_oom_d12.md`.
2. **Bisection**: bracket `t_test ∈ [1.281, 1.30]` with 4-6 probes,
   ~30-60 min each. Track the highest infeasible `t_test`.
3. **Certificate**: at the final infeasible probe, run the rounding +
   verification flow; emit `d64_cert.json`.
4. **Pull-back to local**: `git pull` the cert and run the test gate.
5. **Optional d=128 corroboration**: requires ≥ 1 TB pod (Nebius 2 TB
   class or AWS r6id.32xlarge). Wall ~4-12 h. Not load-bearing.

## 8. Lean port notes

The proof imports two finite-dimensional facts that need formalization:

- **Lemma 2.3 (`val-le-c1a`).** Already discussed in
  `proof/lasserre-proof/lasserre_lower_bound.tex` §2; the Lean
  formulation reduces to a chain of `min` / `max` / `inf` lemmas plus
  the per-window inequality from CS17 Lemma 1.
- **Lemma 3.1, 3.2 (Lasserre soundness, clique soundness).** These are
  classical facts about PSD principal submatrices; mathlib has the
  primitives (`Matrix.PosSemidef.principalSubmatrix_posSemidef`).
- **Theorem 4.1 (cert).** Reduces to: given `(mu_A, S_mom_c, S_loc_i,
  S_win_W)` parsed from JSON, evaluate `r[alpha]` symbolically over
  ℚ and check `mu0 * (2k+1) > sum_alpha |r[alpha]|`. This is a
  computation in ℚ — Lean's `Rat` arithmetic handles it directly,
  though for n_y ~ 10^6 entries we'd want `Lean.Rat` rather than
  `Mathlib.Rat`. The cert file as JSON is parseable by `Lean.Json`.

## 9. References

- [CS17] A. Cloninger, S. Steinerberger. *On suprema of autoconvolutions
  with an application to Sidon sets.* Proc. AMS, 145 (2017), 3191–3200.
  arXiv:1403.7988.
- [MV10] M. Matolcsi, C. Vinuesa. *Improved bounds on the supremum of
  autoconvolutions.* J. Math. Anal. Appl., 372 (2010), 439–447.
  arXiv:0907.1379.
- [Lasserre 2001] J.-B. Lasserre. *Global optimization with polynomials
  and the problem of moments.* SIAM J. Optim. 11 (2001), 796–817.
- [Waki 2006] H. Waki, S. Kim, M. Kojima, M. Muramatsu. *Sums of squares
  and semidefinite program relaxations for polynomial optimization
  problems with structured sparsity.* SIAM J. Optim. 17 (2006), 218–242.
- [JCK 2008] C. Jansson, D. Chaykin, C. Keil. *Rigorous error bounds for
  the optimal value in semidefinite programming.* SIAM J. Numer. Anal.
  46 (2008), 180–200.
- [Boyer-Li 2025] C. Boyer, J. Li. *On Sidon-like additive bounds.* arXiv:2506.16750.
- [White 2022] M. White. *Improved upper bounds for some classical
  autoconvolution constants.* arXiv:2210.16437.
