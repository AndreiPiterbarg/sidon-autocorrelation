"""Remote SSH command execution on RunPod CPU pods."""
import os
import subprocess
import sys

from .config import SSH_OPTIONS, REMOTE_WORKDIR

# RunPod Ubuntu base image has python→3.8 (no pip) and pip→3.13.
# Use python3.12 explicitly everywhere for consistency + numba support.
PYTHON = "python3.12"


def ssh_run(ssh_host, ssh_port, command, timeout=None, stream=False,
            log_file=None):
    """Run a command on the remote pod via SSH."""
    ssh_cmd = (
        ["ssh", "-p", str(ssh_port)]
        + SSH_OPTIONS
        + [f"root@{ssh_host}", command]
    )

    if stream:
        if log_file:
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
    """Install Python dependencies on the remote CPU pod."""
    print("Installing dependencies on pod...")

    # Ensure tmux is available
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
        print("Warning: tmux install timed out (non-fatal)")

    # Install with python3.12 -m pip (guarantees correct interpreter)
    cmd = (
        f"cd {REMOTE_WORKDIR} && "
        f"{PYTHON} -m pip install -q numpy numba joblib 2>&1 | tail -5"
    )
    result = ssh_run(ssh_host, ssh_port, cmd, timeout=300)
    if result.returncode != 0:
        print(f"Warning: pip install issues: {result.stderr}")
    else:
        print("Dependencies installed.")

    # Verify imports actually work
    verify_cmd = (
        f"{PYTHON} -c \"import numpy; import numba; "
        f"print(f'numpy={{numpy.__version__}}, numba={{numba.__version__}}')\""
    )
    result = ssh_run(ssh_host, ssh_port, verify_cmd, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            f"Dependency verification failed: {result.stderr}\n"
            f"stdout: {result.stdout}"
        )
    print(f"Verified: {result.stdout.strip()}")


def verify_cpu(ssh_host, ssh_port):
    """Print CPU info on the remote pod."""
    print("Verifying CPU...")
    cmd = "lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|MHz' | head -6"
    result = ssh_run(ssh_host, ssh_port, cmd, timeout=15)
    if result.returncode == 0 and result.stdout.strip():
        print(result.stdout.strip())
    else:
        print("(could not query CPU info — non-fatal)")


DEFAULT_SCRIPT = "cloninger-steinerberger/cpu/run_cascade.py"


def run_script(ssh_host, ssh_port, script=DEFAULT_SCRIPT,
               args="", timeout=None, log_dir=None):
    """Run a Python script on the remote pod, streaming output."""
    from datetime import datetime

    print(f"Running: {script} {args}")

    log_file = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"cpu_run_{timestamp}.log")
        print(f"Logging output to: {log_file}")

    cmd = f"cd {REMOTE_WORKDIR} && {PYTHON} {script} {args}"
    rc = ssh_run(ssh_host, ssh_port, cmd, timeout=timeout, stream=True,
                 log_file=log_file)
    if rc != 0:
        print(f"Script exited with code {rc}")
    return rc


# ---------- Detached job management ----------

REMOTE_LOG = f"{REMOTE_WORKDIR}/data/cpu_job.log"
TMUX_SESSION = "job"


def launch_script(ssh_host, ssh_port, script=DEFAULT_SCRIPT, args="",
                   auto_teardown=False, api_key=None, pod_id=None):
    """Launch a script in a detached tmux session on the pod."""
    args = args.replace("--auto-teardown", "").strip()

    try:
        ssh_run(ssh_host, ssh_port,
                f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null; true",
                timeout=30)
    except subprocess.TimeoutExpired:
        pass

    teardown_cmd = ""
    if auto_teardown and api_key and pod_id:
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
        f"cd {REMOTE_WORKDIR} && {PYTHON} -u {script} {args} "
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
    """Check if the detached job is still running."""
    result = ssh_run(ssh_host, ssh_port,
                     f"tmux has-session -t {TMUX_SESSION} 2>/dev/null "
                     f"&& echo RUNNING || echo DONE",
                     timeout=30)
    return result.stdout.strip()


def tail_remote_log(ssh_host, ssh_port, follow=False, lines=80):
    """Tail the remote job log."""
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
