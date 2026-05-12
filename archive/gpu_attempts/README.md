# gpu_attempts

GPU and large-pod scaling attempts for the cascade prover. Code, scripts,
and config only. Narrative writeups were archived prior to publication.

## Layout

- `kernel_code/` — CUDA host/device sources (`cascade_host.cu`,
  `cascade_kernel.cu/.h`), prebuilt `.exe` binaries (Windows), build
  scripts (`build.sh`, `build.bat`).
- `deploy_scripts/` — pod / cluster launch shell scripts: `run.sh`,
  `run_full_proof{,_multi}.sh`, `run_multi_gpu.sh`, `run_proof_1_40.sh`,
  `setup_prime_intellect.sh`, `spot_runner.sh`, plus benchmarking
  (`bench.sh`, `bench_hyperparams.sh`) and `profile_bottlenecks.sh`.
- `helpers/` — Python utilities used between kernel runs:
  `generate_l0.py`, `split_parents.py`, `merge_survivors.py`,
  `run_chunked.py`.
- `gpupod/` — self-contained Python package for remote pod orchestration
  (sessions, budgets, sync). Importable as `gpupod`. Includes
  `.session.json` runtime state.

## Status

All exploratory. The GPU SCS line was not used for the current bound;
see memory note `project_p4_48t_oom_d12.md` for one of the cost lessons.
