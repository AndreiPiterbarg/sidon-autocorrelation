"""Remote SSH command execution on RunPod pods."""
import os
import subprocess
import sys

from .config import SSH_OPTIONS, REMOTE_WORKDIR


def ssh_run(ssh_host, ssh_port, command, timeout=None, stream=False,
            log_file=None):
    """Run a command on the remote pod via SSH.

    Parameters
    ----------
    ssh_host : str
    ssh_port : int
    command : str
        Shell command to run remotely.
    timeout : int, optional
        Timeout in seconds. None = no timeout.
    stream : bool
        If True, stream stdout/stderr to terminal in real time.
    log_file : str, optional
        If set (and stream=True), tee all output to this local file.

    Returns
    -------
    subprocess.CompletedProcess or int (if stream=True, returns exit code)
    """
    ssh_cmd = (
        ["ssh", "-p", str(ssh_port)]
        + SSH_OPTIONS
        + [f"root@{ssh_host}", command]
    )

    if stream:
        if log_file:
            # Tee output to both terminal and local log file
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as lf:
                proc = subprocess.Popen(
                    ssh_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    text=True,
                )
                try:
                    for line in proc.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        lf.write(line)
                        lf.flush()
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    msg = f"\nCommand timed out after {timeout}s\n"
                    sys.stdout.write(msg)
                    lf.write(msg)
                    return 1
                except KeyboardInterrupt:
                    proc.kill()
                    lf.write("\n[INTERRUPTED by user]\n")
                    raise
                return proc.returncode
        else:
            proc = subprocess.Popen(
                ssh_cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"\nCommand timed out after {timeout}s")
                return 1
            return proc.returncode
    else:
        return subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def install_deps(ssh_host, ssh_port):
    """Install Python dependencies and system tools on the remote pod."""
    print("Installing dependencies on pod...")

    # Ensure tmux is available (needed for 'launch' detached jobs)
    tmux_cmd = (
        "which tmux > /dev/null 2>&1 || "
        "(apt-get update -qq && apt-get install -y -qq tmux > /dev/null 2>&1)"
    )
    try:
        result = ssh_run(ssh_host, ssh_port, tmux_cmd, timeout=300)
        if isinstance(result, int) or result.returncode != 0:
            print("Warning: could not verify/install tmux (non-fatal)")
        else:
            print("tmux: OK")
    except subprocess.TimeoutExpired:
        print("Warning: tmux install timed out (non-fatal, only needed for 'launch')")

    cmd = (
        f"cd {REMOTE_WORKDIR} && "
        f"pip install -q numpy numba joblib 2>&1 | tail -3"
    )
    result = ssh_run(ssh_host, ssh_port, cmd, timeout=300)
    if result.returncode != 0:
        print(f"Warning: pip install issues: {result.stderr}")
    else:
        print("Dependencies installed.")


def build_cuda(ssh_host, ssh_port):
    """Build CUDA kernels on the remote pod."""
    print("Building CUDA kernels...")
    cmd = (
        f"cd {REMOTE_WORKDIR} && "
        f"python cloninger-steinerberger/gpu/build.py"
    )
    rc = ssh_run(ssh_host, ssh_port, cmd, timeout=300, stream=True)
    if rc != 0:
        raise RuntimeError(f"CUDA build failed (exit code {rc})")
    print("CUDA build successful.")


def verify_gpu(ssh_host, ssh_port, retries=5, timeout_per_try=60):
    """Verify GPU is accessible and print device info.

    Retries with increasing wait because the pod's GPU driver may
    not be ready immediately after SSH becomes available.
    """
    import time
    cmd = "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
    for attempt in range(1, retries + 1):
        print(f"Verifying GPU (attempt {attempt}/{retries})...")
        try:
            result = ssh_run(ssh_host, ssh_port, cmd, timeout=timeout_per_try)
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip()
                print(f"GPU: {gpu_info}")
                return gpu_info
            err = result.stderr.strip() if result.stderr else "no output"
            print(f"  nvidia-smi returned code {result.returncode}: {err}")
        except subprocess.TimeoutExpired:
            print(f"  Timed out after {timeout_per_try}s")
        if attempt < retries:
            wait = 10 * attempt
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("GPU verification failed after all retries")


def run_script(ssh_host, ssh_port, script="cloninger-steinerberger/gpu/solvers.py",
               args="", timeout=None, log_dir=None):
    """Run a Python script on the remote pod, streaming output.

    If log_dir is set, all output is tee'd to a timestamped log file
    in that directory so results are preserved even if the connection drops.
    """
    from datetime import datetime

    print(f"Running: {script} {args}")

    log_file = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"gpu_run_{timestamp}.log")
        print(f"Logging output to: {log_file}")

    cmd = f"cd {REMOTE_WORKDIR} && python {script} {args}"
    rc = ssh_run(ssh_host, ssh_port, cmd, timeout=timeout, stream=True,
                 log_file=log_file)
    if rc != 0:
        print(f"Script exited with code {rc}")
    return rc


# ---------- Detached job management (survives SSH disconnect) ----------

REMOTE_LOG = f"{REMOTE_WORKDIR}/data/gpu_job.log"
TMUX_SESSION = "job"


def launch_script(ssh_host, ssh_port, script="run_proof.py", args="",
                   auto_teardown=False, api_key=None, pod_id=None):
    """Launch a script in a detached tmux session on the pod.

    The script keeps running even if SSH disconnects or the local
    machine is turned off. Output goes to REMOTE_LOG on the pod.

    If auto_teardown=True, the pod self-terminates via the RunPod API
    after the job finishes (requires api_key and pod_id).
    """
    # Safety: never pass --auto-teardown to the remote script
    args = args.replace("--auto-teardown", "").strip()
    # Kill any existing job session
    try:
        ssh_run(ssh_host, ssh_port,
                f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null; true",
                timeout=30)
    except subprocess.TimeoutExpired:
        pass  # No existing session, or SSH slow — safe to continue

    # Ensure data dir exists, then launch in tmux.
    # Single quotes protect $? from the remote SSH shell — it gets
    # expanded inside the tmux bash session instead.
    # set -o pipefail ensures $? captures python's exit code, not tee's.
    teardown_cmd = ""
    if auto_teardown and api_key and pod_id:
        # After job finishes, call RunPod API to terminate the pod.
        # Use curl to avoid needing the runpod SDK on the pod.
        teardown_cmd = (
            f"echo '=== AUTO-TEARDOWN: terminating pod ===' >> {REMOTE_LOG}; "
            f"curl -s -X POST https://api.runpod.io/graphql "
            f"-H 'Content-Type: application/json' "
            f"-H 'api-key: {api_key}' "
            f"-d '{{\"query\":\"mutation {{ podTerminate(input: {{podId: \\\"{pod_id}\\\"}}) }}\"}}' "
            f">> {REMOTE_LOG} 2>&1; "
        )

    tmux_inner = (
        f"set -o pipefail; "
        f"cd {REMOTE_WORKDIR} && python -u {script} {args} "
        f"2>&1 | tee {REMOTE_LOG}; "
        f"echo ===JOB_EXIT_CODE=$?=== >> {REMOTE_LOG}; "
        f"{teardown_cmd}"
        f"true"
    )
    cmd = (
        f"mkdir -p {REMOTE_WORKDIR}/data && "
        f"tmux new-session -d -s {TMUX_SESSION} "
        f"bash -c '{tmux_inner}'"
    )
    result = ssh_run(ssh_host, ssh_port, cmd, timeout=30)
    if isinstance(result, int):
        return result
    return result.returncode


def check_job_status(ssh_host, ssh_port):
    """Check if the detached job is still running.

    Returns 'RUNNING', 'DONE', or 'NO_SESSION'.
    """
    result = ssh_run(ssh_host, ssh_port,
                     f"tmux has-session -t {TMUX_SESSION} 2>/dev/null "
                     f"&& echo RUNNING || echo DONE",
                     timeout=10)
    return result.stdout.strip()


def tail_remote_log(ssh_host, ssh_port, follow=False, lines=80):
    """Tail the remote job log.

    If follow=True, streams live (Ctrl-C to detach safely).
    Otherwise prints the last `lines` lines.
    """
    if follow:
        print("(Ctrl-C to detach — the job keeps running)")
        cmd = f"tail -f {REMOTE_LOG}"
        try:
            ssh_run(ssh_host, ssh_port, cmd, stream=True)
        except KeyboardInterrupt:
            print("\nDetached from log. Job is still running on the pod.")
    else:
        cmd = f"tail -n {lines} {REMOTE_LOG} 2>/dev/null || echo '(no log yet)'"
        result = ssh_run(ssh_host, ssh_port, cmd, timeout=15)
        print(result.stdout)
