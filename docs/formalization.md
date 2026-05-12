# Lean 4 Formalization

The analytic chain of the **Piterbarg--Bajaj--Vincent Bound** is
mechanised in Lean 4 under the namespace `Sidon.MultiScale`. The module
lives at
[`../lean/Sidon/MultiScale.lean`](../lean/Sidon/MultiScale.lean) and
imports the shared definitions in
[`../lean/Sidon/Defs.lean`](../lean/Sidon/Defs.lean) along with
`Mathlib`. The module builds with $0$ `sorry` tactics. Exactly **one**
user axiom is declared, `MV_master_inequality_for_extremiser` (the
three-scale MV master inequality with the slack rationals
`K2UpperQ = 47897/10000` and `gainLowerQ = 20925/100000` substituted
for the analytic `K_2` and `a`). The headline theorem's `#print
axioms` listing reaches that single user axiom together with Lean's
three core axioms (`propext`, `Classical.choice`, `Quot.sound`). The
quadratic inversion `master_inequality_M_lower` and the five
slack-soundness statements (`K_two_upper_bound`, `k_one_lower_bound`,
`S_one_upper_bound`, `min_G_lower_bound`, `gain_lower_bound`,
discharging paper Lemmas 4.1--4.5 as pure rational `norm_num` checks)
are Lean *theorems* and do not contribute axioms to the dependency
closure.

## Headline theorem

```lean
theorem autoconvolution_ratio_ge_1292_1000 (f : ℝ → ℝ)
    (hf_nonneg  : ∀ x, 0 ≤ f x)
    (hf_supp    : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm
      (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume ≠ ⊤) :
    autoconvolution_ratio f ≥ (1292 / 1000 : ℝ)
```

The four hypotheses are nonnegativity, support inside $(-1/4, 1/4)$, strict
positivity of $\int f$, and finiteness of $\|f * f\|_\infty$ (an
`ENNReal.toReal` encoding artifact; harmless when passing to the infimum).
Equivalent restatements `autoconvolution_ratio_ge_1_292` and `C1a_ge_1292`
are exported from the same namespace.

## The sole user axiom

| Axiom name | Statement | Discharged by |
|------------|-----------|---------------|
| `MV_master_inequality_for_extremiser` | For every admissible $f$, $R(f) + 1 + \sqrt{R(f)-1}\sqrt{\texttt{K2UpperQ}-1} \ge 2/u + \texttt{gainLowerQ}$, with $\texttt{K2UpperQ}=47897/10000$ and $\texttt{gainLowerQ}=20925/100000$ | The three-scale extension of Matolcsi-Vinuesa (2010, Lemma 3.1) on $\mathbb{R}/u\mathbb{Z}$. The Fourier reduction proceeds identically: each $J_0(\pi \delta_i \xi)^2$ is Bochner-positive, the convex combination preserves Bochner-positivity, and $G$ enters only through the gain parameter $a$. The substitution of the slack rationals $\texttt{K2UpperQ}$ for $K_2$ and $\texttt{gainLowerQ}$ for $a$ is valid because the master inequality is monotone in $K_2 - 1$ and $a$, and the slack rationals are true bounds on the analytic functionals (paper Lemmas 4.1--4.5). Paper Theorem 2.3 (= MV (7)). |

The rational bounds on the analytic functionals (paper Lemmas
4.1--4.5) are discharged externally by the `flint.arb` certifier at
256-bit precision; the precise enclosures are recorded in
[`reference_anchors.json`](../delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json)
and in the docstrings of `Sidon/MultiScale.lean`.

## Lean theorems backing the headline

The quadratic-in-$M$ inversion and the five slack-soundness statements
are Lean *theorems*, not axioms.

| Theorem name | Statement | Proof |
|--------------|-----------|-------|
| `master_inequality_M_lower` | If $a_{\rm lo} \ge \texttt{gainLowerQ}$ and $M + 1 + \sqrt{M-1}\sqrt{\texttt{K2UpperQ} - 1} \ge 2/u + a_{\rm lo}$ then $M \ge \texttt{MTargetQ} = 1292/1000$ | At $M = 1292/1000$ the LHS attains $\Phi(M) \le 66879/20000 = 3.34395$, strictly below $\tau = 2/u + \texttt{gainLowerQ} = 4267003/1276000 \approx 3.344046$, with margin $307/3190000 \ge 9.6 \times 10^{-5}$. Proved by case analysis on $M \le 1$ versus $M > 1$ using `Real.sqrt` monotonicity and `nlinarith`. Paper Proposition 5.1. |
| `K_two_upper_bound`   | $\texttt{K2UpperQ} \ge 4788906/1000000$       | `norm_num` rational comparison; certifier-reported $K_2 \le 4.788906$ (paper Lemma 4.2). |
| `k_one_lower_bound`   | $\texttt{K1LowerQ} \le 92124658/100000000$    | `norm_num` rational comparison; certifier-reported $k_1 \ge 0.92124658$ (paper Lemma 4.1). |
| `S_one_upper_bound`   | $\texttt{S1UpperQ} \ge 2984091/100000$        | `norm_num` rational comparison; certifier-reported $S_1 \le 29.840907$ (paper Lemma 4.3). |
| `min_G_lower_bound`   | $\texttt{minGLowerQ} \le 9999798/10000000$    | `norm_num` rational comparison; certifier-reported $\min_{[0,1/4]} G \ge 0.99997987$ (paper Lemma 4.4). |
| `gain_lower_bound`    | $\texttt{gainLowerQ} \le 21009214/100000000$  | `norm_num` rational comparison; certifier-reported $a \ge 0.21009214$ (paper Lemma 4.5). |

The five `norm_num` theorems record that the rational slacks fed into
the axiom are on the correct side of the certifier-reported decimals;
they are *not* axioms about the analytic functionals, and they do not
appear in the dependency closure of the headline theorem.

## Correspondence with the paper

The Lean module is consistent with the paper *Improving the Bounds on
the Supremum of Autoconvolutions*, which proposes the
Piterbarg--Bajaj--Vincent Bound:

- The axiom `MV_master_inequality_for_extremiser` corresponds to
  Theorem 2.3 of the paper (the three-scale instance of the
  Matolcsi-Vinuesa master inequality), specialised by the slack
  rationals justified by paper Lemmas 4.1--4.5.
- The theorem `master_inequality_M_lower` corresponds to Proposition
  5.1 of the paper (the strict-failure witness at $M = 1292/1000$).
- The five `norm_num` theorems record the soundness of the slack
  rationals against the certifier-reported decimals of paper Lemmas
  4.1--4.5.

For the analytic chain combining these into the headline statement, see
Section 5 of [`lower_bound_proof.pdf`](../lower_bound_proof.pdf).

## Build and inspection

For build instructions, the expected `lake build` output, and the
`#print axioms` invocation that prints the dependency list, see
[`reproducibility.md`](reproducibility.md).
