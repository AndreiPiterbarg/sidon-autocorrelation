"""Code sync and result collection for GPU pods."""
import os
import shutil
import subprocess
import platform

from .config import PROJECT_ROOT, REMOTE_WORKDIR, SSH_OPTIONS_BASH, SYNC_EXCLUDES, _to_msys_path


def _find_bash():
    """Find Git Bash, avoiding WSL bash on Windows."""
    git_bash = r"C:\Program Files\Git\bin\bash.exe"
    if os.path.exists(git_bash):
        return git_bash
    found = shutil.which("bash")
    if found:
        return found
    raise RuntimeError("bash not found. Install Git for Windows.")


def _ssh_opts_str():
    return " ".join(SSH_OPTIONS_BASH)


def _run_bash(pipeline, cwd=None, timeout=600):
    bash = _find_bash()
    return subprocess.run(
        [bash, "-c", pipeline],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def sync_code(ssh_host, ssh_port):
    """Sync local project code to remote pod via tar-pipe."""
    print("Syncing code to pod...")
    excludes = " ".join(f"--exclude '{p}'" for p in SYNC_EXCLUDES)
    ssh_opts = _ssh_opts_str()

    # Preserve data/ (checkpoints) on the remote — only remove code files,
    # then overlay the new code on top.
    pipeline = (
        f"tar cf - {excludes} . | "
        f"ssh -p {ssh_port} {ssh_opts} root@{ssh_host} "
        f"'mkdir -p {REMOTE_WORKDIR} && "
        f"find {REMOTE_WORKDIR} -maxdepth 1 -mindepth 1 ! -name data -exec rm -rf {{}} + && "
        f"tar xf - --no-same-owner -C {REMOTE_WORKDIR}'"
    )

    result = _run_bash(pipeline, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError(f"Sync failed: {result.stderr}")

    print("Sync complete.")


def upload_checkpoint(ssh_host, ssh_port, local_path, remote_name=None):
    """Upload a single .npy checkpoint file to the remote data/ directory."""
    if remote_name is None:
        remote_name = os.path.basename(local_path)

    remote_path = f"{REMOTE_WORKDIR}/data/{remote_name}"
    print(f"Uploading {local_path} -> {remote_path}...")

    ssh_opts = _ssh_opts_str()

    # Ensure remote data/ directory exists
    pipeline = (
        f"ssh -p {ssh_port} {ssh_opts} root@{ssh_host} "
        f"'mkdir -p {REMOTE_WORKDIR}/data'"
    )
    _run_bash(pipeline)

    # Use scp via bash
    local_msys = _to_msys_path(local_path) if platform.system() == "Windows" else local_path
    pipeline = (
        f"scp -P {ssh_port} {ssh_opts} "
        f"'{local_msys}' root@{ssh_host}:{remote_path}"
    )
    result = _run_bash(pipeline, timeout=1200)

    if result.returncode != 0:
        raise RuntimeError(f"Upload failed: {result.stderr}")

    # Verify the file arrived
    verify = (
        f"ssh -p {ssh_port} {ssh_opts} root@{ssh_host} "
        f"'ls -lh {remote_path}'"
    )
    vr = _run_bash(verify, timeout=30)
    print(f"  Uploaded: {vr.stdout.strip()}")


def collect_results(ssh_host, ssh_port, remote_path=None, local_path=None):
    """Download results from remote pod to local data/ directory."""
    if remote_path is None:
        remote_path = f"{REMOTE_WORKDIR}/data"
    if local_path is None:
        local_path = str(PROJECT_ROOT / "data")

    print(f"Collecting results from {remote_path}...")

    ssh_opts = _ssh_opts_str()

    # Check if remote has files
    check_cmd = (
        f"ssh -p {ssh_port} {ssh_opts} root@{ssh_host} "
        f"'ls {remote_path}/*.npy 2>/dev/null | wc -l'"
    )
    check = _run_bash(check_cmd, timeout=30)
    nfiles = int(check.stdout.strip()) if check.stdout.strip().isdigit() else 0
    if nfiles == 0:
        print("No .npy files to collect.")
        return

    print(f"  Found {nfiles} .npy file(s). Downloading...")

    os.makedirs(local_path, exist_ok=True)
    local_msys = _to_msys_path(local_path) if platform.system() == "Windows" else local_path

    pipeline = (
        f"ssh -p {ssh_port} {ssh_opts} root@{ssh_host} "
        f"'cd {remote_path} && tar cf - *.npy *.json *.log 2>/dev/null' | "
        f"tar xf - --no-same-owner -C '{local_msys}'"
    )
    result = _run_bash(pipeline, timeout=1200)
    if result.returncode != 0:
        print(f"Warning: collect may be incomplete: {result.stderr}")
    else:
        print(f"Results saved to {local_path}")
