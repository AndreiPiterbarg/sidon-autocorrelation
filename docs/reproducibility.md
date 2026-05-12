# Reproducibility

Exact commands to reproduce the `flint.arb` certificate for the
**Piterbarg--Bajaj--Vincent Bound** $C_{1a} \ge 1292/1000 = 1.292$ and
to build the Lean 4 formalization.

## Prerequisites

- **Python** 3.11 or newer.
- **Python packages.** `python-flint >= 0.6` (arb / acb / fmpq backend),
  `numpy`, `mpmath`, `cvxpy`. The QP step prefers `mosek` (academic licence
  available); a `clarabel`, `scs`, or `ecos` fallback is used automatically.
- **Lean 4 toolchain.** The repository pins `leanprover/lean4:v4.24.0` via
  `lean/lean-toolchain`. Install [`elan`](https://github.com/leanprover/elan)
  and let `lake` pick up the pinned version on first build.

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
   Sidon.MultiScale.MV_master_inequality_for_extremiser]
```

Exactly one user axiom appears in the dependency closure
(`MV_master_inequality_for_extremiser`, the three-scale MV master
inequality with the slack rationals `K2UpperQ = 47897/10000` and
`gainLowerQ = 20925/100000` substituted for the analytic `K_2` and
`a`).  The quadratic inversion `master_inequality_M_lower` and the
five slack-soundness statements (`K_two_upper_bound`,
`k_one_lower_bound`, `S_one_upper_bound`, `min_G_lower_bound`,
`gain_lower_bound`) are Lean *theorems* — they are not axioms and do
not appear in the dependency closure.  See
[`formalization.md`](formalization.md) for the axiom statement and
the theorem statements.
