"""Code sync via tar-pipe over SSH."""
import os
import platform
import shutil
import subprocess

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


def _run_bash(pipeline, cwd=None, timeout=120):
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
    print(f"Syncing code to {ssh_host}:{ssh_port}...")

    excludes = " ".join(f"--exclude '{p}'" for p in SYNC_EXCLUDES)
    ssh_opts = _ssh_opts_str()

    pipeline = (
        f"tar cf - {excludes} . | "
        f"ssh -p {ssh_port} {ssh_opts} root@{ssh_host} "
        f"'rm -rf {REMOTE_WORKDIR} && mkdir -p {REMOTE_WORKDIR} && tar xf - --no-same-owner -C {REMOTE_WORKDIR}'"
    )

    result = _run_bash(pipeline, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError(f"Sync failed: {result.stderr}")

    print("Sync complete.")


def collect_results(ssh_host, ssh_port, remote_path=None, local_path=None):
    """Download results from remote pod to local data/ directory."""
    if remote_path is None:
        remote_path = f"{REMOTE_WORKDIR}/data"
    if local_path is None:
        local_path = str(PROJECT_ROOT / "data")

    print(f"Collecting results from {remote_path}...")

    ssh_opts = _ssh_opts_str()

    check_cmd = (
        f"ssh -p {ssh_port} {ssh_opts} root@{ssh_host} "
        f"'ls {remote_path} 2>/dev/null | head -5'"
    )
    result = _run_bash(check_cmd)
    if result.returncode != 0 or not result.stdout.strip():
        print("No results to collect.")
        return

    print(f"Files found: {result.stdout.strip()}")

    os.makedirs(local_path, exist_ok=True)

    tar_local_path = _to_msys_path(local_path) if platform.system() == "Windows" else local_path

    pipeline = (
        f"ssh -p {ssh_port} {ssh_opts} root@{ssh_host} "
        f"'tar cf - -C {remote_path} .' | "
        f"tar xf - -C '{tar_local_path}'"
    )

    result = _run_bash(pipeline)

    if result.returncode != 0:
        print(f"Warning: result collection may be incomplete: {result.stderr}")
    else:
        print(f"Results saved to {local_path}/")
