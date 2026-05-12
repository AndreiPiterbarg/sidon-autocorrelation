# Tests

Test suite for the production `delsarte_dual` package -- the multi-scale
arcsine kernel chain that certifies the Piterbarg-Bajaj-Vincent Bound
`C_{1a} >= 1292/1000 = 1.292`.

## Running

From the repo root:

```bash
pytest tests/                                       # full suite (15 tests)
pytest tests/grid_bound_alt_kernel/ -v              # verbose
pytest tests/grid_bound_alt_kernel/test_alt_kernel.py::TestKernelValuesSanity
```

`tests/conftest.py` puts the repo root on `sys.path` so
`from delsarte_dual.X import Y` resolves regardless of where pytest is
invoked from.

## Layout

| Path | Tests for |
|---|---|
| `grid_bound_alt_kernel/test_alt_kernel.py` | The full production chain: kernel admissibility (`ArcsineKernel`, `MultiScaleArcsineKernel`), Bochner positivity, QP G optimisation, M bisection, certificate round-trip. 15 tests. |

## Legacy tests

Tests that exercised earlier exploration directions (single-moment grid
bound, Hoelder, sharper bathtub, triples union, multi-frequency MO 2.17,
restricted Hoelder, Path A unconditional Hoelder, families F1-F5,
multi-moment pipeline, Lasserre, BnB cascades, reproductions of
Cloninger and Steinerberger (2017, arXiv:1403.7988), GPU SCS, three-point
SDP, ...) live under `archive/legacy_tests/` grouped by direction.  They
are kept for provenance but are not part of the active suite.
