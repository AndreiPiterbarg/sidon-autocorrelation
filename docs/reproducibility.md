# Reproducibility

Exact commands to reproduce the `flint.arb` certificate for the
**Piterbarg--Bajaj--Vincent Bound** $C_{1a} \ge 1292/1000 = 1.292$ and
to build the Lean 4 formalization.

## Prerequisites

- **Python** 3.11 or newer.
- **Python packages.** `python-flint >= 0.6` (arb / acb / fmpq backend),
  `numpy`, `mpmath`, `cvxpy`. The QP step prefers `mosek` (academic licence
  available); a `clarabel`, `scs`, or `ecos` fallback is used automatically.
- **Lean 4 toolchain.** The repository pins `leanprover/lean4:v4.29.1`
  via `lean/lean-toolchain`, with `mathlib` pinned to commit
  [`5e932f97dd25535344f80f9dd8da3aab83df0fe6`](https://github.com/leanprover-community/mathlib4/commit/5e932f97dd25535344f80f9dd8da3aab83df0fe6)
  (post-Nov 2025). The `v4.29.1` bump is required because the
  formalisation relies on the $L^2$-Plancherel API
  (`MeasureTheory.Lp.fourierTransformₗᵢ`) and convolution--Fourier
  duality (`Real.fourier_mul_convolution_eq`), both first available
  at that mathlib commit. Install
  [`elan`](https://github.com/leanprover/elan) and let `lake` pick up
  the pinned versions on first build.

## One-line install

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install python-flint numpy mpmath cvxpy mosek
```

`mosek` may be replaced by `clarabel`; the driver selects the first solver
it finds.

## Reproducing the certificate

The certifier driver lives at
[`delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py`](../delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py).
Run it as a module:

```bash
python -m delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel
```

Defaults reproduce the published bound: the three-scale arcsine kernel at
$(\delta_1, \delta_2, \delta_3) = (138, 55, 25)/1000$ with weights $(85, 10,
5)/100$, a 200-coefficient cosine multiplier $G$ re-optimized against this
kernel, all anchors in arb at 256-bit precision, cell-search bisection on
$M$ targeting `1292/1000`.

### Expected output

The driver prints the five anchors and writes a self-contained certificate
to `delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json`.
Reference values are recorded in
[`reference_anchors.json`](../delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json):

| Anchor      | Bound                                          |
|-------------|------------------------------------------------|
| $k_1$       | $\ge 0.92124658$ (radius $< 7 \times 10^{-77}$) |
| $K_2$       | $\in [4.788823,\; 4.788906]$                   |
| $S_1$       | $\le 29.840907$                                |
| $\min G$    | $\ge 0.99997987$                               |
| gain $a$    | $\ge 0.21009214$                               |
| $M_{\rm cert}$ (production) | $= 66167/51200 \approx 1.29232422$ |
| $M_{\rm cert}$ (slack-anchor) | $\ge 1.29215650$ (`reference_anchors.json`) |
| Headline rational target | $1292/1000$ |

Wall time on a modern laptop is roughly 11 s at 256-bit precision.

### Certificate hash

The emitted JSON has the form `{"sha256_of_body": <digest>, "body": {...}}`.
To re-derive the digest:

```bash
python -c "import json, hashlib; d = json.load(open('delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json')); print(hashlib.sha256(json.dumps(d['body'], indent=2, sort_keys=True).encode()).hexdigest())"
```

It must match the certificate's `sha256_of_body` field.

### Independent verifier

`delsarte_dual/grid_bound/certify.py` re-checks every quantitative claim
using only `python-flint` primitives:

```bash
python -m delsarte_dual.grid_bound.certify \
  delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json
```

Exit code `0` on success.

## Building the Lean formalization

```bash
cd lean
lake build                     # full proof chain
lake build Sidon.MultiScale    # headline module only
```

Expected result: exit code `0`, no `sorry` warnings.

### Axiom inventory

```bash
cd lean
lake env lean AxiomCheck.lean
```

This prints the axiom dependency closure of the headline theorem.
Equivalently, after `lake build`, `#print axioms
Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000` reports

```
'Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000' depends on axioms:
  [propext, Classical.choice, Quot.sound,
   Sidon.MultiScale.K2_analytic_le_K2UpperQ,
   Sidon.MultiScale.gain_analytic_ge_gainLowerQ]
```

Exactly **two** user axioms appear in the dependency closure, both
*verifiable-by-computation* (i.e. rigorously certified numerical
assertions): each is a logically decidable inequality about a specific
real number, backed by `flint.arb` at 256-bit precision via the driver
[`../delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py`](../delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py).
They appear as `axiom` rather than `theorem` only because mathlib does
not yet ship a Bessel interval-arithmetic library; the FlySpeck
formalisation of Kepler's conjecture used the same convention.

- `K2_analytic_le_K2UpperQ`: $K_2(K_{\rm ms}) := \int K_{\rm ms}^2 \le
  47897/10000$. Analogue of "Mathematica computed $K_2 \approx 4.788$"
  in MV 2010, but backed by `flint.arb` at 256-bit precision (paper
  Lemma 4.2).
- `gain_analytic_ge_gainLowerQ`: $\texttt{gain\_analytic} = (4/u) \cdot
  m_G^2 / S_G \ge 20925/100000$. Analogue of MV's Mathematica citation
  of $a$, certifier-coupled in arb (paper Lemmas 4.3--4.5).

In addition to these two axioms, the headline theorem takes an
**analytic admissibility-bundle hypothesis** `ExtremiserPrimitives f`
encoding the four MV Lemma 3.1 outputs (Eqs.(1)--(4)) for the
specific pair $(f, K_{\rm ms})$. The bundle's existence for an
arbitrary admissible $f$ is the analogue of MV invoking "by Lemma 3.1
(Martin--O'Bryant)"; closing it for general $f$ in mathlib requires
the $L^1 \cap L^2$ Plancherel + period-$u$ Parseval bridge that
`Sidon.TorusParseval` and `Sidon.FourierAux` are built around but
that is not yet a one-line mathlib call.

The previous macro axiom `MV_master_inequality_for_extremiser` is
**now a Lean theorem** (post-Wave-12 restructuring); its
content factors through the bundle hypothesis plus the two
verifiable-by-computation axioms above. The quadratic inversion
`master_inequality_M_lower`, the slack-monotonicity lift
`MV_master_via_slack_monotonicity`, the full MV-Lemmas chain
`MV_master_inequality_from_MV_lemmas`, and the five slack-soundness
statements (`K_two_upper_bound`, `k_one_lower_bound`,
`S_one_upper_bound`, `min_G_lower_bound`, `gain_lower_bound`) are also
Lean *theorems* -- none of them contributes an axiom to the dependency
closure. See [`formalization.md`](formalization.md) for the axiom
statements, the theorem statements, and the module layout (fifteen
modules totalling roughly 8650 lines). The repository additionally
exports two Schwartz-class headlines
(`autoconvolution_ratio_ge_1292_1000_schwartz` consuming
`SchwartzAtomic`, and
`autoconvolution_ratio_ge_1292_1000_schwartz_residual` consuming the
slimmer `SchwartzAtomicResidual`) under
`Sidon.MultiScaleSchwartz` and
`Sidon.SchwartzAtomicDischarge` respectively.
