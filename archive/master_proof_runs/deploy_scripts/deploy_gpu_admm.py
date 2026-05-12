#!/usr/bin/env python
"""Deploy a single H100 GPU pod and run ADMM solver benchmarks.

Creates a RunPod pod with 1x H100, syncs code, installs PyTorch + deps,
and runs the GPU ADMM solver tests.

Usage:
    python deploy_gpu_admm.py              # deploy + setup + run tests
    python deploy_gpu_admm.py --test-only  # just run tests (pod already up)
    python deploy_gpu_admm.py --teardown   # destroy pod
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

STATE_FILE = PROJECT_ROOT / "data" / "gpu_admm_pod_state.json"

# Single GPU is enough for ADMM testing
GPU_TYPE = "NVIDIA H100 80GB HBM3"
GPU_COUNT = 1


def create_pod():
    """Create a 1x H100 pod."""
    import runpod
    runpod.api_key = RUNPOD_API_KEY

    pub_key = SSH_KEY_PATH.with_suffix(".pub").read_text().strip()

    print(f"Creating GPU pod (1x H100)...")
    pod = runpod.create_pod(
        name="sidon-admm-test",
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

    # Wait for SSH
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
    raise RuntimeError("Pod did not become ready")


def ssh_run(host, port, cmd, timeout=600):
    """Run command on pod via SSH."""
    ssh_cmd = ["ssh", "-p", str(port)] + SSH_OPTIONS + [f"root@{host}", cmd]
    return subprocess.run(ssh_cmd, capture_output=True, text=True,
                          timeout=timeout, stdin=subprocess.DEVNULL)


def ssh_run_print(host, port, cmd, timeout=600):
    """Run command and print output."""
    r = ssh_run(host, port, cmd, timeout=timeout)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0 and r.stderr.strip():
        print(f"STDERR: {r.stderr.strip()}")
    return r


def sync_code(host, port):
    """Sync project code to pod."""
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
    """Install PyTorch, SCS, scipy on the pod."""
    print("Installing dependencies...")

    # The Docker image already has PyTorch + CUDA. Install extras.
    r = ssh_run_print(host, port,
        "pip install -q scipy scs numpy && "
        "python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")' && "
        "python -c 'import scs; print(f\"SCS {scs.__version__}\")' && "
        "python -c 'import scipy; print(f\"scipy {scipy.__version__}\")'",
        timeout=300)

    if r.returncode != 0:
        raise RuntimeError(f"Setup failed: {r.stderr[:300]}")
    print("  Environment ready.")


def run_test(host, port, test_script, timeout=1800):
    """Run a test script on the pod and print output."""
    cmd = f"cd {REMOTE_WORKDIR} && python -u {test_script} 2>&1"
    print(f"\n{'='*60}")
    print(f"Running: {test_script}")
    print(f"{'='*60}")

    r = ssh_run_print(host, port, cmd, timeout=timeout)
    return r.returncode


def run_inline(host, port, code, timeout=600):
    """Run inline Python code on the pod."""
    # Escape for shell
    escaped = code.replace("'", "'\\''")
    cmd = f"cd {REMOTE_WORKDIR} && python -u -c '{escaped}' 2>&1"
    return ssh_run_print(host, port, cmd, timeout=timeout)


def run_admm_tests(host, port):
    """Run the full ADMM benchmark suite."""

    # Test 1: Sanity check
    print("\n" + "="*60)
    print("TEST 1: ADMM sanity check")
    print("="*60)
    run_test(host, port, "tests/admm_gpu_solver.py", timeout=60)

    # Test 2: Phase-1 boundary detection
    print("\n" + "="*60)
    print("TEST 2: Phase-1 boundary detection (d=4 O2 bw=3)")
    print("="*60)
    run_inline(host, port, """
import sys, os, time
sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from lasserre_highd import _precompute_highd, _build_banded_cliques
from run_scs_direct import build_base_problem, _precompute_window_psd_decomposition, _assemble_window_psd
from admm_gpu_solver import admm_solve, augment_phase1
import numpy as np
from scipy import sparse as sp

d, order, bw = 4, 2, 3
cliques = _build_banded_cliques(d, bw)
P = _precompute_highd(d, order, cliques, verbose=False)
A_base, b_base, c_obj, cone_base, meta = build_base_problem(P, add_upper_loc=True)

all_windows = set(range(P['n_win']))
win_decomp = _precompute_window_psd_decomposition(P, all_windows)

print('Phase-1 tau boundary test (val(d=4)=1.102, SCS lb~1.099):')
for t_val in [1.0, 1.05, 1.08, 1.09, 1.095, 1.10, 1.12]:
    A_win, b_win, psd_win = _assemble_window_psd(win_decomp, t_val)
    A_full = sp.vstack([A_base, A_win], format='csc')
    b_full = np.concatenate([b_base, np.zeros(win_decomp['n_rows'])])
    cone_full = {'z': cone_base['z'], 'l': cone_base['l'],
                 's': list(cone_base['s']) + psd_win}
    A_p1, b_p1, c_p1, cone_p1, tau_col = augment_phase1(A_full, b_full, cone_full)

    t0 = time.time()
    sol = admm_solve(A_p1, b_p1, c_p1, cone_p1,
                     max_iters=10000, eps_abs=1e-7, eps_rel=1e-7,
                     device='cuda', verbose=False)
    dt = time.time() - t0
    tau = sol['x'][tau_col]
    iters = sol['info']['iter']
    feas = 'FEAS' if tau <= 1e-4 else 'INFEAS'
    print(f't={t_val:.3f}: tau={tau:+.8f} {feas:6s} ({iters} iters, {dt:.1f}s)')
""", timeout=600)

    # Test 3: Full pipeline
    print("\n" + "="*60)
    print("TEST 3: Full pipeline d=4 O2 bw=3 --gpu")
    print("="*60)
    run_test(host, port,
             "tests/run_scs_direct.py --d 4 --order 2 --bw 3 --gpu "
             "--cg-rounds 5 --bisect 12 --scs-iters 10000 --scs-eps 1e-6",
             timeout=1800)

    # Test 4: Compare with CPU SCS
    print("\n" + "="*60)
    print("TEST 4: CPU SCS baseline d=4 O2 bw=3")
    print("="*60)
    run_test(host, port,
             "tests/run_scs_direct.py --d 4 --order 2 --bw 3 "
             "--cg-rounds 5 --bisect 12 --scs-iters 50000 --scs-eps 1e-6",
             timeout=1800)


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
    parser.add_argument('--test-only', action='store_true',
                        help='Skip pod creation, run tests on existing pod')
    parser.add_argument('--teardown', action='store_true',
                        help='Destroy the pod')
    parser.add_argument('--sync-only', action='store_true',
                        help='Just sync code to existing pod')
    args = parser.parse_args()

    if args.teardown:
        teardown()
        return

    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY not set in .env")
        sys.exit(1)

    state = load_state()

    if args.test_only or args.sync_only:
        if not state:
            print("ERROR: No pod state found. Run without --test-only first.")
            sys.exit(1)
    else:
        # Create pod
        info = create_pod()
        state = info
        save_state(state)
        time.sleep(5)  # let SSH fully initialize

        # Setup environment
        setup_env(state["ssh_host"], state["ssh_port"])

    # Sync code
    sync_code(state["ssh_host"], state["ssh_port"])

    if args.sync_only:
        print("Code synced. Done.")
        return

    # Run tests
    run_admm_tests(state["ssh_host"], state["ssh_port"])

    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*60}")
    print(f"Pod still running: ssh -p {state['ssh_port']} "
          f"-i {SSH_KEY_PATH} -o StrictHostKeyChecking=no "
          f"root@{state['ssh_host']}")
    print(f"To teardown: python deploy_gpu_admm.py --teardown")


if __name__ == "__main__":
    main()
