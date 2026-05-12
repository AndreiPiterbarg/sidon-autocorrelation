#!/usr/bin/env python
"""Deploy a 1x H100 pod, sync code, run GPU vs CPU benchmark.

Usage:
    python deploy_bench_gpu.py              # full: create pod + setup + run
    python deploy_bench_gpu.py --test-only  # reuse existing pod
    python deploy_bench_gpu.py --teardown   # destroy pod
"""
import json
import os
import subprocess
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from gpupod.config import (
    RUNPOD_API_KEY, DOCKER_IMAGE, CLOUD_TYPE,
    SSH_KEY_PATH, SSH_OPTIONS, SSH_OPTIONS_BASH,
    SYNC_EXCLUDES, REMOTE_WORKDIR, _to_msys_path,
)
import platform

STATE_FILE = PROJECT_ROOT / "data" / "gpu_bench_pod_state.json"
GPU_TYPE = "NVIDIA H100 80GB HBM3"
GPU_COUNT = 1


def create_pod():
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    pub_key = SSH_KEY_PATH.with_suffix(".pub").read_text().strip()

    print(f"Creating GPU pod (1x H100)...")
    pod = runpod.create_pod(
        name="sidon-gpu-bench",
        image_name=DOCKER_IMAGE,
        gpu_type_id=GPU_TYPE,
        gpu_count=GPU_COUNT,
        cloud_type=CLOUD_TYPE,
        container_disk_in_gb=50,
        volume_in_gb=0,
        env={"PUBLIC_KEY": pub_key},
        ports="22/tcp",
    )
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")

    print("Waiting for SSH...", end="", flush=True)
    for attempt in range(120):
        time.sleep(5)
        status = runpod.get_pod(pod_id)
        if status and status.get("desiredStatus") == "RUNNING":
            runtime = status.get("runtime")
            if runtime and runtime.get("ports"):
                for p in runtime["ports"]:
                    if p.get("privatePort") == 22 and p.get("ip"):
                        host = p["ip"]
                        port = p["publicPort"]
                        print(f" ready! {host}:{port}")
                        return {"pod_id": pod_id, "ssh_host": host, "ssh_port": port}
        print(".", end="", flush=True)
    raise RuntimeError("Pod did not become ready in 10 minutes")


def ssh_run(host, port, cmd, timeout=600):
    ssh_cmd = ["ssh", "-p", str(port)] + SSH_OPTIONS + [f"root@{host}", cmd]
    return subprocess.run(ssh_cmd, capture_output=True, text=True,
                          timeout=timeout, stdin=subprocess.DEVNULL)


def ssh_run_print(host, port, cmd, timeout=600):
    r = ssh_run(host, port, cmd, timeout=timeout)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0 and r.stderr.strip():
        print(f"STDERR: {r.stderr.strip()}")
    return r


def sync_code(host, port):
    print("Syncing code...")
    excludes = " ".join(f"--exclude='{e}'" for e in SYNC_EXCLUDES)
    ssh_opts = " ".join(SSH_OPTIONS_BASH)
    remote = REMOTE_WORKDIR

    if platform.system() == "Windows":
        local = _to_msys_path(PROJECT_ROOT)
        bash_cmd = (
            f"cd '{local}' && "
            f"tar cf - {excludes} . | "
            f"ssh -p {port} {ssh_opts} root@{host} "
            f"'mkdir -p {remote} && "
            f"find {remote} -maxdepth 1 ! -name data ! -path {remote} -exec rm -rf {{}} + && "
            f"tar xf - --no-same-owner -C {remote}'"
        )
        r = subprocess.run(["bash", "-c", bash_cmd], capture_output=True,
                           text=True, timeout=120)
    else:
        tar_cmd = f"cd '{PROJECT_ROOT}' && tar cf - {excludes} ."
        ssh_extract = (
            f"ssh -p {port} {' '.join(SSH_OPTIONS)} root@{host} "
            f"'mkdir -p {remote} && "
            f"find {remote} -maxdepth 1 ! -name data ! -path {remote} -exec rm -rf {{}} + && "
            f"tar xf - --no-same-owner -C {remote}'"
        )
        r = subprocess.run(f"{tar_cmd} | {ssh_extract}", shell=True,
                           capture_output=True, text=True, timeout=120)

    if r.returncode != 0:
        print(f"  WARNING: {r.stderr[:300]}")
    else:
        print("  Sync complete.")


def setup_env(host, port):
    print("Installing dependencies...")
    r = ssh_run_print(host, port,
        "pip install -q scipy scs numpy python-dotenv && "
        "python -c 'import torch; print(f\"PyTorch {torch.__version__}, "
        "CUDA {torch.cuda.is_available()}, "
        "GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")' && "
        "python -c 'import scs; print(f\"SCS {scs.__version__}\")' && "
        "python -c 'import scipy; print(f\"scipy {scipy.__version__}\")'",
        timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"Setup failed: {r.stderr[:300]}")
    print("  Environment ready.")


def run_benchmark(host, port):
    """Run the GPU vs CPU benchmark suite."""
    print("\n" + "=" * 70)
    print("PHASE 1: Sanity check (scs_gpu.py standalone test)")
    print("=" * 70)
    r = ssh_run_print(host, port,
        f"cd {REMOTE_WORKDIR} && python -u tests/scs_gpu.py 2>&1",
        timeout=120)
    if r.returncode != 0:
        print("SANITY CHECK FAILED — aborting benchmark")
        return

    print("\n" + "=" * 70)
    print("PHASE 2: GPU vs CPU benchmark (small configs)")
    print("=" * 70)
    r = ssh_run_print(host, port,
        f"cd {REMOTE_WORKDIR} && python -u tests/bench_gpu_vs_cpu.py "
        f"--configs small --save data/bench_gpu_vs_cpu_small.json 2>&1",
        timeout=3600)

    print("\n" + "=" * 70)
    print("PHASE 3: GPU vs CPU benchmark (medium configs)")
    print("=" * 70)
    r = ssh_run_print(host, port,
        f"cd {REMOTE_WORKDIR} && python -u tests/bench_gpu_vs_cpu.py "
        f"--configs medium --save data/bench_gpu_vs_cpu_medium.json 2>&1",
        timeout=7200)

    # Pull results back
    print("\n" + "=" * 70)
    print("Pulling results...")
    print("=" * 70)
    for fname in ["bench_gpu_vs_cpu_small.json", "bench_gpu_vs_cpu_medium.json"]:
        scp_cmd = ["scp", "-P", str(port)] + SSH_OPTIONS + [
            f"root@{host}:{REMOTE_WORKDIR}/data/{fname}",
            str(PROJECT_ROOT / "data" / fname)
        ]
        subprocess.run(scp_cmd, capture_output=True, timeout=30)
        if (PROJECT_ROOT / "data" / fname).exists():
            print(f"  Retrieved {fname}")
        else:
            print(f"  Failed to retrieve {fname}")


def save_state(state):
    os.makedirs(STATE_FILE.parent, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return None


def teardown():
    state = load_state()
    if not state:
        print("No active pod found.")
        return
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    print(f"Terminating pod {state['pod_id']}...")
    runpod.terminate_pod(state["pod_id"])
    STATE_FILE.unlink(missing_ok=True)
    print("Pod terminated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--teardown', action='store_true')
    parser.add_argument('--sync-only', action='store_true')
    args = parser.parse_args()

    if args.teardown:
        teardown()
        return

    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY not set in .env")
        sys.exit(1)

    state = load_state()
    if args.test_only and state:
        host, port = state["ssh_host"], state["ssh_port"]
        print(f"Reusing pod {state['pod_id']} at {host}:{port}")
        sync_code(host, port)
        run_benchmark(host, port)
        return

    if args.sync_only and state:
        host, port = state["ssh_host"], state["ssh_port"]
        sync_code(host, port)
        return

    # Full deploy
    state = create_pod()
    save_state(state)
    host, port = state["ssh_host"], state["ssh_port"]

    # Wait a bit for SSH to fully come up
    time.sleep(10)
    sync_code(host, port)
    setup_env(host, port)
    run_benchmark(host, port)

    print("\n" + "=" * 70)
    print(f"Pod still running: {state['pod_id']}")
    print(f"SSH: ssh -p {port} root@{host}")
    print(f"Teardown: python deploy_bench_gpu.py --teardown")
    print("=" * 70)


if __name__ == "__main__":
    main()
