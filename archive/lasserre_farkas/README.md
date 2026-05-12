# lasserre_farkas/

Archived code for the Lasserre hierarchy and Farkas-certified routes to lower-bound C_{1a}.

**Status: DEAD direction.** Per project memory, large SDP/Lasserre routes were forbidden; the publishable rigorous bound (1.28984) was reached via the multi-scale arcsine kernel route, not from anything here. Kept for reproducibility and as a reference for the Farkas-certified pipeline (val(4) > 1.0963 was the high-water mark; see `project_farkas_certified_lasserre.md`).

## Layout

- `lasserre/` — Lasserre SDP hierarchy: solvers (`d64_solver.py`, `d128_solver.py`), Farkas certs (`d64_farkas_cert.py`, `d128_farkas_cert.py`), Chebyshev / Fourier / clique reductions, three-point and L^{3/2} variants, Z2 symmetry exploits. Includes `certs/` (JSON cert snapshots d=4,6,8,16), `polya_lp/` (Polya LP relaxations + tier4 / tier_dual subpipelines) and `trajectory/` (val(d) curves).
- `certified_lasserre/` — Rigorous certification layer: atomic-nu SDP, Bessel kernels, Farkas CG/certify, safe certify (incl. flint), MO cuts, parallel adjustments. Tests under `tests/`.
- `farkas_fast/` — Pod runner stub (`run_pod.py`) for fast Farkas batch jobs.
- `polya_lp_mps/` — MPS snapshots of Polya LP problems (d=8 R=4/8/12, d=16 R=4/6/8) plus `benchmark.json`. Inputs only — not intended to be re-solved here.

## Notes

- `__pycache__/` removed during cleanup.
- Empty `tests/__init__.py` retained as a Python package marker.
- Original sub-direction layout preserved; no files renamed.
