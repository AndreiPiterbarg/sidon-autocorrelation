# kernel_code

CUDA implementation of the cascade prover (host + device), with build
scripts and prebuilt Windows binaries.

- `cascade_host.cu` — host-side launcher / chunk dispatcher.
- `cascade_kernel.cu`, `cascade_kernel.h` — device kernel: per-cell F/FN/Q
  closure tests, atomic survivor compaction.
- `cascade_prover.exe`, `cascade_prover_trace.exe` — prebuilt binaries
  (release + trace). Trace variant emits per-cell decision logs.
- `build.sh`, `build.bat` — Linux / Windows build via `nvcc`.

Built against CUDA 12; pod-side rebuilds use `deploy_scripts/run.sh`.
