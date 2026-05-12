# Path A — Hölder direct attack (DEAD)

Path A's unconditional Hölder programme NEGATIVE after seven attacks (MEMORY: project_path_a_unconditional_status). The only surviving residue is the **conditional** restricted-Hölder bound C_{1a} >= 1.37842 under Hyp_R(M_max = 1.51), which is **not** unconditional.

## The target identity

Martin-O'Bryant 2004 Conjecture 2.9 (the Hölder-slack inequality) asserts for nonneg pdf g on [-1/4, 1/4] with `int g = 1`:

$$\|g*g\|_2^2 \;\le\; c_* \, \|g*g\|_\infty, \qquad c_* := \log 16 / \pi = 0.882542\ldots$$

This strengthens ordinary Hölder (`||g*g||_2^2 <= 1 * M`) by ~11.7%. If MO 2.9 held on the restricted class `M <= 1.378`, the conditional restricted-Hölder theorem in `delsarte_dual/restricted_holder/derivation.md` would close unconditionally to give C_{1a} >= 1.37842.

## Status: full class disproved (Boyer-Li 2025)

Boyer-Li (arXiv:2506.16750) construct an asymmetric 575-step witness `f_0` (rescaled g) with `||g*g||_infty = 1.6520` and `c_*(g) = 0.9016 > 0.8825`. **MO 2.9 fails on the unrestricted class.**

The BL witness sits **outside** the restricted class (`1.378 < 1.51 < 1.652`), so the restricted version is not disproved.

## Empirical scan of restricted class

`c_*(f) = ||f*f||_2^2 / ||f*f||_infty` over restricted-class candidates:

| Family | M | c_*(f) | Margin to 0.8825 |
|--------|---|--------|------------------|
| MV's 119-cosine extremizer | 2.314 | 0.5889 | -0.294 |
| Uniform `2 * 1_{[-1/4,1/4]}` | 2.000 | 0.6667 | -0.216 |
| Triangle | 2.667 | 0.7190 | -0.164 |
| Arcsine `(1/4-x^2)^{-0.4942}` | 2.78 | 0.4164 | -0.466 |
| BL 575-step (full witness) | **1.652** | **0.9016** | +0.0191 |

Every symmetric f has `M >= 2`, so the entire symmetric branch lives **outside** the restricted class. Only highly asymmetric BL-style continuous-step densities can populate `1.275 <= M <= 1.66`. Of 4000 random 50-cell step functions, zero had `M <= 1.51`. BL perturbations (convex mix, smoothing, compression, symmetrization, 8000-step Metropolis hill-climbing) all push `M` upward while pushing `c_*` toward 0.5-0.7 — no continuous path connects BL to the restricted class.

## Seven structural obstructions to direct proof

| Technique | Obstruction |
|-----------|-------------|
| Schwarz/symmetric rearrangement | Riesz inequality runs wrong way: `||f**f**||_infty >= ||f*f||_infty` |
| MO Prop 2.11 + L2.17 | `M_LB <= 1.116` (P-side ceiling) — `multifreq_mo217/derivation.md` §6.1 |
| Bochner-phase identity (Lemma A1) | Phased identity at peak cannot bound *phaseless* `||f||_2^2`; doubling `z_n` preserves identity but inflates `||g||_2^2` unboundedly |
| White-2022 dual-Hölder | Wrong direction; see [cascade_estimator.md](cascade_estimator.md) §Route C |
| EL / compactness (Theorem C) | EL admits asymmetric solutions; no symmetry-forcing argument |
| Hölder-Hausdorff-Young | Dominated by trivial Hölder applied alongside MV |
| Asymmetric Hyp_R | `||f||_2^2` unbounded by M |

## Remaining opening (narrow, untried)

The "BL gap" — `1.378 <= M <= 1.652` is unpopulated by known witnesses but unproved to be empty. A rigidity theorem `c_*(f) <= log 16 / pi` for every `f in C_{1.378}` would close MO 2.9 on the restricted class and unlock C_{1a} >= 1.37842 unconditionally. No tool in MV/MO/White/BL/Bochner toolkit suffices — would require a width-vs-Hölder-ratio coupling. The most plausible new ingredient is the Cohn-Goncalves sign-uncertainty family (MEMORY: project_cohn_elkies_status), recorded as Priority 1 in the master strategy.

## References

- [../proof_outline.md](../proof_outline.md), [../formalization.md](../formalization.md)
- [master_attacks.md](master_attacks.md), [path_b_kbk.md](path_b_kbk.md), [cloninger_steinerberger.md](cloninger_steinerberger.md), [cascade_estimator.md](cascade_estimator.md), [audits.md](audits.md)
- Martin & O'Bryant (2004), arXiv:math/0410004.
- Boyer & Li (2025), arXiv:2506.16750.
