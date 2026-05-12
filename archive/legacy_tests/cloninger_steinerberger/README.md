# cloninger_steinerberger/

Reproduction / verification work around Cloninger–Steinerberger 2017
(arXiv:1403.7988). Note: per `MEMORY.md`, the 1.2802 LB was later
shown invalid (`project_cs_1.2802_invalid.md`); these scripts pre-date
that finding.

- `tests/` — `test_*.py` unit / integration / framework tests.
- `verify/` — `verify_*.py`, axiom-gap / theorem / threshold checks.
- `gpu_admm/` — ADMM + SCS GPU solver, profiling.
- `cpu_gpu_compare/` — CPU vs GPU vs MATLAB comparison harnesses.
- `vacuity_threshold/` — critical vacuity table + corrected/precise
  vacuity probes.
- `cpu_gpu_data/` — saved survivor / parent / meta `.npy` + `.json`.
