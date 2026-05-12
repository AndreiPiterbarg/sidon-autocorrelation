# archive/audits/

Independent verification probes for the C_{1a} lower-bound work.
Scripts validate (or refute) claims made elsewhere in the repo;
JSON / log / PNG siblings are saved outputs.

## Subfolders

- `probes/` — `_audit_{A,B,C,D,D2..D6,E,F,G,G2}_probe.{py,json}` and
  `_audit_prune_D.py`. The lettered probe series independently re-checks
  cell-pruning and B&B closure logic. Each `*.json` is the output of the
  matching `*.py`.
- `verify/` — `_verify_*` scripts and outputs: `_verify_125_no_b35`
  (cascade-125 without the B35 test), `_verify_K26_independent`
  (K26 multiscale audit), `_verify_minG_reopt(+_refine)` with result JSON.
- `b35_and_bl/` — `_b35_audit.py` (B35 unsoundness check) plus the
  Bourgain-Luczak witness scripts `_bl_witness_audit.py`,
  `_bl_witness_zeros.py`, and `bl_witness_audit.json`.
- `check_scripts/` — assorted `_check_*` utilities: codegen sanity
  (`_check_codegen.py`), d=2 window check (`_check_d2_windows.py`),
  L0-cell enumeration (`_check_l0_cells.py`), and the PowerShell
  process probe `_check_proc.ps1`.

## Top-level files

Single-shot audits not part of any series:
`_audit_check_unsoundness.py`, `_audit_matlab_bug_repro.py`
(reproduces the CS17 MATLAB bug), `_audit_pe_axiom.py`,
`_audit_pe_gap.py`, `_audit_threshold_algebra.py`,
`_audit_M1.json` (M1 audit result), `mv_f_check.png`
(MV-kernel visual sanity check).
