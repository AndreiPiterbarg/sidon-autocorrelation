# Prime Intellect Runbook (Using `uv`, Best Cost/Perf GPU)

## Summary

Use `A100 (SXM4)` as the default. At the current quoted pricing (`$1.23/hr` vs `$1.20/hr` PCIe), SXM4 is usually the better value for this CUDA-heavy workload.  
Fallback: `A100 (PCIE)` if availability is better.  
Use `H100` only if you prioritize shortest wall-clock time and are comfortable with about 2x hourly cost.

## Proof Integrity Mode (Strict Fail-Closed)

`run_proof.py` now runs in strict fail-closed mode for formal claims:

- `status=proven`: exhaustive required work completed, no survivors remain.
- `status=not_proven`: exhaustive required work completed, survivors remain.
- `status=inconclusive`: processing was incomplete (timeout, extraction truncation, streamed artifact issue, etc.).

Process exit codes:

- `0`: completed run (`proven` or `not_proven`)
- `2`: inconclusive run (no formal proof claim may be made)

Correction-term note:

- GPU prove/refine kernels currently use dynamic correction-based pruning.
- Formal validation of the correction derivation remains an open research review item.

## GPU Selection

| GPU | Price (USD/hr) | Recommendation |
|---|---:|---|
| A100 (PCIE) | 1.20 | Good fallback if SXM4 is unavailable |
| A100 (SXM4) | 1.23 | Recommended default |
| H100 (PCIE) | 2.35 | Faster, but much higher cost |
| H100 (SXM5) | 2.43 | Faster, but much higher cost |

## Step-by-Step Commands

### 1. On your local machine: configure Prime and pick an A100 offer

```bash
prime config set-api-key <YOUR_API_KEY>
prime config set-ssh-key-path ~/.ssh/id_ed25519.pub
prime availability list --gpu-type A100_80GB --regions united_states --no-group-similar
prime pods create --id <AVAILABILITY_ID> --name sidon-gpu --disk-size 500 --image ubuntu_22_cuda_12
prime pods status <POD_ID>
prime pods ssh <POD_ID>
```

### 2. On the pod: install system tools

```bash
nvidia-smi
nvcc --version
apt-get update && apt-get install -y git python3 python3-venv build-essential tmux curl
```

### 3. Clone repo and set up Python with `uv` (no direct `pip` usage)

```bash
git clone <YOUR_REPO_URL> sidon-autocorrelation
cd sidon-autocorrelation

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 4. Build CUDA kernels

```bash
python cloninger-steinerberger/gpu/build.py
```

### 5. Run a smoke test first

```bash
python run_proof.py --target 1.10 --m 20 --max-levels 1 --time-budget 1800
```

### 6. Run the real job in `tmux`

```bash
tmux new -s sidon
source .venv/bin/activate
python run_proof.py --target 1.20 --m 50 --n-start 3 --max-levels 4 --time-budget 54000 | tee data/proof.log
```

Detach with `Ctrl+b`, then `d`.

### 7. Monitor progress

```bash
tmux attach -t sidon
tail -f data/proof.log
```

### 8. Pull results to local machine after completion

```bash
# from your local machine (use host/port from `prime pods status`)
scp -P <SSH_PORT> -r root@<SSH_HOST>:/workspace/sidon-autocorrelation/data ./data
```

### 9. Terminate pod when done

```bash
prime pods terminate <POD_ID>
```

## Resume Semantics

- Checkpoints include an explicit `storage_mode`:
  - `memory`: survivors are resumed from checkpoint `.npy`.
  - `file`: survivors are resumed from `survivor_file_path` on disk.
- If a streamed survivor file is missing/corrupt at resume time, the run is treated as incomplete (inconclusive) rather than silently continuing.

## Test Scenarios

1. Smoke scenario:

```bash
python run_proof.py --target 1.10 --m 20 --max-levels 1 --time-budget 1800
```

Expected: run completes and writes logs and JSON under `data/`.

2. Production scenario:

```bash
python run_proof.py --target 1.20 --m 50 --n-start 3 --max-levels 4 --time-budget 54000
```

Expected: long-form run with incremental checkpoints/results under `data/`.

## Assumptions and Defaults

- Deployment target is Prime Intellect (not the existing RunPod automation in `gpupod/`).
- Python dependency workflow uses `uv`.
- One GPU pod is used.
- Default GPU is `A100 (SXM4)`; fallback is `A100 (PCIE)` when queue/availability is better.
