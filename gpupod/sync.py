"""Code sync via tar-pipe over SSH (no rsync needed).

Uses bash -c to run the full tar|ssh pipeline, ensuring compatibility
with Windows Git Bash (which provides GNU tar and ssh).
"""
import os
import shutil
import subprocess

from .config import PROJECT_ROOT, REMOTE_WORKDIR, SSH_OPTIONS_BASH, SYNC_EXCLUDES


def _find_bash():
    """Find Git Bash, avoiding WSL bash on Windows."""
    # Prefer Git Bash explicitly
    git_bash = r"C:\Program Files\Git\bin\bash.exe"
    if os.path.exists(git_bash):
        return git_bash
    # Fallback to PATH
    found = shutil.which("bash")
    if found:
        return found
    raise RuntimeError("bash not found. Install Git for Windows.")


def _ssh_opts_str():
    """Return SSH options as a shell-safe string for bash -c pipelines."""
    return " ".join(SSH_OPTIONS_BASH)


def _run_bash(pipeline, cwd=None, timeout=120):
    """Run a shell pipeline through Git Bash."""
    bash = _find_bash()
    result = subprocess.run(
        [bash, "-c", pipeline],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


def sync_code(ssh_host, ssh_port):
    """Sync local project code to remote pod via tar-pipe.

    Uses: tar cf - ... | ssh ... tar xf -
    Runs through Git Bash for Windows compatibility.
    The codebase is <1MB so full re-sync is fast.
    """
    print(f"Syncing code to {ssh_host}:{ssh_port}...")

    # Build tar exclude args
    excludes = " ".join(f"--exclude '{p}'" for p in SYNC_EXCLUDES)
    ssh_opts = _ssh_opts_str()

    # Single bash pipeline: tar | ssh tar
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

    # Check if remote data dir has files
    check_cmd = (
        f"ssh -p {ssh_port} {ssh_opts} root@{ssh_host} "
        f"'ls {remote_path} 2>/dev/null | head -5'"
    )
    result = _run_bash(check_cmd)
    if result.returncode != 0 or not result.stdout.strip():
        print("No results to collect.")
        return

    print(f"Files found: {result.stdout.strip()}")

    # Ensure local data dir exists
    os.makedirs(local_path, exist_ok=True)

    # Reverse tar-pipe: ssh tar | local tar
    pipeline = (
        f"ssh -p {ssh_port} {ssh_opts} root@{ssh_host} "
        f"'tar cf - -C {remote_path} .' | "
        f"tar xf - -C '{local_path}'"
    )

    result = _run_bash(pipeline)

    if result.returncode != 0:
        print(f"Warning: result collection may be incomplete: {result.stderr}")
    else:
        print(f"Results saved to {local_path}/")
