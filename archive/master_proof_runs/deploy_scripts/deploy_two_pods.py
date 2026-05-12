#!/usr/bin/env python
"""Deploy two CPU pods and launch one Lasserre config on each.

Pod A: d=14 O3 bw=13 (full) — ~11 hours estimated
Pod B: d=16 O3 bw=12         — ~27 hours estimated

Usage:
    python deploy_two_pods.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# --- Config ---
PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))
from cpupod.config import (
    RUNPOD_API_KEY, INSTANCE_ID, TEMPLATE_ID, CLOUD_TYPE,
    CONTAINER_DISK_GB, SSH_OPTIONS, SSH_KEY_PATH, REMOTE_WORKDIR,
    SYNC_EXCLUDES,
)

PYTHON = "python3.13"
MOSEK_LIC = Path.home() / "mosek" / "mosek.lic"

JOBS = [
    {"name": "sidon-d14-full", "args": "--d 14 --order 3 --bw 13 --cg-rounds 10 --bisect 10"},
    {"name": "sidon-d16-bw12", "args": "--d 16 --order 3 --bw 12 --cg-rounds 10 --bisect 10"},
]

STATE_FILE = PROJECT_ROOT / "data" / "two_pods_state.json"


def create_pod(name):
    """Create a RunPod CPU pod and wait for SSH to be ready."""
    import runpod
    runpod.api_key = RUNPOD_API_KEY

    pub_key = SSH_KEY_PATH.with_suffix(".pub").read_text().strip()

    print(f"  Creating pod '{name}' ({INSTANCE_ID})...")
    pod = runpod.create_pod(
        name=name,
        image_name="runpod/base:0.7.0-ubuntu2004",
        instance_id=INSTANCE_ID,
        cloud_type=CLOUD_TYPE,
        template_id=None,
        container_disk_in_gb=CONTAINER_DISK_GB,
        env={"PUBLIC_KEY": pub_key},
    )
    pod_id = pod["id"]
    print(f"  Pod created: {pod_id}")

    # Wait for SSH
    for attempt in range(60):
        time.sleep(5)
        status = runpod.get_pod(pod_id)
        runtime = status.get("runtime")
        if runtime and runtime.get("ports"):
            for p in runtime["ports"]:
                if p.get("privatePort") == 22 and p.get("ip"):
                    host = p["ip"]
                    port = p["publicPort"]
                    print(f"  SSH ready: {host}:{port}")
                    return {"pod_id": pod_id, "ssh_host": host, "ssh_port": port}
        if attempt % 6 == 5:
            print(f"  Still waiting... ({attempt*5}s)")
    raise RuntimeError(f"Pod {pod_id} did not become ready in 5 minutes")


def ssh_run(host, port, cmd, timeout=300):
    """Run a command on the pod via SSH."""
    ssh_cmd = ["ssh", "-p", str(port)] + SSH_OPTIONS + [f"root@{host}", cmd]
    return subprocess.run(ssh_cmd, capture_output=True, text=True,
                          timeout=timeout, stdin=subprocess.DEVNULL)


def sync_code(host, port):
    """Sync code to pod via tar pipe."""
    print(f"  Syncing code...")
    excludes = " ".join(f"--exclude='{e}'" for e in SYNC_EXCLUDES)

    from cpupod.config import SSH_OPTIONS_BASH, _to_msys_path
    import platform

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
        r = subprocess.run(["bash", "-c", bash_cmd], capture_output=True, text=True, timeout=120)
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
        print(f"  WARNING sync stderr: {r.stderr[:200]}")
    else:
        print(f"  Sync complete.")


def install_mosek(host, port):
    """Install MOSEK + scipy and copy license."""
    print(f"  Installing MOSEK + scipy...")
    r = ssh_run(host, port,
                f"{PYTHON} -m pip install -q Mosek scipy 2>&1 | tail -3",
                timeout=300)
    if r.returncode != 0:
        print(f"  pip stderr: {r.stderr[:200]}")

    # Copy license
    print(f"  Copying MOSEK license...")
    ssh_run(host, port, "mkdir -p /root/mosek", timeout=10)
    scp_cmd = (["scp", "-P", str(port)] + SSH_OPTIONS +
               [str(MOSEK_LIC), f"root@{host}:/root/mosek/mosek.lic"])
    subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30)

    # Verify
    r = ssh_run(host, port,
                f'{PYTHON} -c "from mosek.fusion import Model; print(\'MOSEK OK\')"',
                timeout=30)
    if "MOSEK OK" in r.stdout:
        print(f"  MOSEK verified.")
    else:
        raise RuntimeError(f"MOSEK verification failed: {r.stderr}")


def install_tmux(host, port):
    """Ensure tmux is available."""
    ssh_run(host, port,
            "which tmux >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null 2>&1)",
            timeout=300)


def launch_job(host, port, args_str):
    """Launch the job in a detached tmux session."""
    log = f"{REMOTE_WORKDIR}/data/cpu_job.log"
    script = "tests/run_single_config.py"
    inner = (
        f"set -o pipefail; "
        f"cd {REMOTE_WORKDIR} && {PYTHON} -u {script} {args_str} "
        f"2>&1 | tee {log}; "
        f"echo ===JOB_EXIT_CODE=$?=== >> {log}; true"
    )
    cmd = (
        f"tmux kill-session -t job 2>/dev/null; true && "
        f"mkdir -p {REMOTE_WORKDIR}/data && "
        f"tmux new-session -d -s job bash -c '{inner}'"
    )
    r = ssh_run(host, port, cmd, timeout=30)
    return r.returncode


def main():
    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY not set in .env")
        sys.exit(1)
    if not MOSEK_LIC.exists():
        print(f"ERROR: MOSEK license not found at {MOSEK_LIC}")
        sys.exit(1)

    state = {}
    print("=" * 60)
    print("Deploying TWO CPU pods for Lasserre benchmarks")
    print("=" * 60)

    for i, job in enumerate(JOBS):
        print(f"\n--- Pod {i+1}: {job['name']} ---")
        print(f"    Config: {job['args']}")

        info = create_pod(job["name"])
        state[job["name"]] = info

        # Save state after each pod (in case of failure)
        os.makedirs(STATE_FILE.parent, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))

        sync_code(info["ssh_host"], info["ssh_port"])
        install_tmux(info["ssh_host"], info["ssh_port"])
        install_mosek(info["ssh_host"], info["ssh_port"])

        print(f"  Launching: {job['args']}")
        rc = launch_job(info["ssh_host"], info["ssh_port"], job["args"])
        if rc != 0:
            print(f"  WARNING: launch returned {rc}")
        else:
            print(f"  Job launched in tmux.")

    print(f"\n{'='*60}")
    print(f"BOTH PODS RUNNING")
    print(f"{'='*60}")
    for name, info in state.items():
        print(f"  {name}: ssh -p {info['ssh_port']} root@{info['ssh_host']}")
    print(f"\nState saved to: {STATE_FILE}")
    print(f"\nTo check logs:")
    for name, info in state.items():
        print(f"  ssh -p {info['ssh_port']} -i ~/.ssh/id_ed25519 "
              f"-o StrictHostKeyChecking=no root@{info['ssh_host']} "
              f"'tail -50 {REMOTE_WORKDIR}/data/cpu_job.log'")
    print(f"\nTo teardown all pods:")
    print(f"  python -m cpupod cleanup")


if __name__ == "__main__":
    main()
