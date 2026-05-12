# archive/polya_lp/

Pólya-class LP attempts on the Sidon $C_{1a}$ lower bound. **DEAD direction** —
no improvement over MV's 1.2748 was achieved here; archived for reference only.

## Layout

- `code/` — 24 Python scripts: LP formulations, sweeps, solver benchmarks,
  cut-generation experiments. Naming hints at sub-direction:
  - `polya_lp_d{4,8,12,16,64}_*.py` — dimension-specific sweeps / build /
    solve / focused tests
  - `polya_lp_cut*.py`, `polya_lp_cuts_smoke.py` — cutting-plane attempts
  - `polya_lp_cg_smoke.py` — column-generation smoke test
  - `polya_lp_pdlp_smoke.py`, `polya_lp_mosek_ipm_bench.py`,
    `polya_lp_ts_smoke.py`, `polya_lp_cloud_cuopt.py` — solver back-ends
  - `polya_lp_full_sweep.py`, `polya_lp_sweep_analysis.py` — driver / analysis
  - `polya_lp_moment_smoke.py`, `polya_lp_project.py`,
    `polya_lp_active_test.py`, `polya_lp_smoke.py`,
    `polya_lp_laptop_benchmark.py`, `polya_lp_d8_lambda_test.py`
- `logs/` — 21 run logs (`*.log`) paired by name with the scripts above.
- `polya_lp_results.json` — consolidated sweep results (top-level, single file).

## Status

Pólya / Bochner-admissible LP relaxations capped at the MV ceiling
(~1.2748). See `MEMORY.md` entries on Cohn–Elkies status and the MV
alt-kernel sweep for context.
