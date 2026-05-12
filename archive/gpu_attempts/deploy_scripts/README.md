# deploy_scripts

Shell scripts for launching the CUDA cascade prover on remote GPU pods
(primarily Prime Intellect spot instances).

## Launchers

- `run.sh` — single-pod, single-GPU smoke run.
- `run_full_proof.sh` — full d=12 proof, single pod.
- `run_full_proof_multi.sh` — sharded full proof across N pods.
- `run_multi_gpu.sh` — multi-GPU on one pod (per-device chunk split).
- `run_proof_1_40.sh` — proof of survivor parents 1..40.
- `spot_runner.sh` — spot-instance wrapper with reconnect logic.

## Setup / bench

- `setup_prime_intellect.sh` — pod bootstrap (CUDA, deps, repo sync).
- `bench.sh`, `bench_hyperparams.sh` — throughput benchmarks (chunk size,
  block size, occupancy sweep).
- `profile_bottlenecks.sh` — Nsight / nvprof wrappers for kernel hotspots.

All scripts assume the `kernel_code/` build artifacts are present.
