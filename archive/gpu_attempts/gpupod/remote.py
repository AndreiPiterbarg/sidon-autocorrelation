"""Remote SSH command execution on RunPod GPU pods."""
import os
import subprocess
import sys
import time

from .config import SSH_OPTIONS, REMOTE_WORKDIR


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
                    encoding="utf-8",
                    errors="replace",
                )
                try:
                    for line in proc.stdout:
                        try:
                            sys.stdout.write(line)
                        except UnicodeEncodeError:
                            sys.stdout.write(line.encode("ascii", "replace").decode())
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
            stdin=subprocess.DEVNULL,
        )


def verify_gpu(ssh_host, ssh_port, retries=10, timeout_per_try=30):
    """Verify GPU is available and print GPU info."""
    print("Verifying GPU...")
    cmd = "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"

    for attempt in range(retries):
        try:
            result = ssh_run(ssh_host, ssh_port, cmd, timeout=timeout_per_try)
            if result.returncode == 0 and result.stdout.strip():
                print(f"  GPU: {result.stdout.strip()}")
                return True
        except Exception:
            pass
        if attempt < retries - 1:
            print(f"  GPU not ready (attempt {attempt+1}/{retries}), waiting...")
            time.sleep(10)

    print("WARNING: Could not verify GPU (non-fatal)")
    return False


def build_cuda(ssh_host, ssh_port):
    """Build the CUDA cascade prover on the remote pod."""
    print("Building CUDA kernel on pod...")

    cmd = (
        f"cd {REMOTE_WORKDIR}/gpu && "
        f"chmod +x build.sh && "
        f"./build.sh release 2>&1"
    )
    rc = ssh_run(ssh_host, ssh_port, cmd, timeout=300, stream=True)
    if rc != 0:
        raise RuntimeError(f"CUDA build failed (exit code {rc})")

    # Verify binary exists
    verify_cmd = f"ls -lh {REMOTE_WORKDIR}/gpu/cascade_prover"
    result = ssh_run(ssh_host, ssh_port, verify_cmd, timeout=15)
    if result.returncode != 0:
        raise RuntimeError("CUDA build produced no binary")
    print(f"  Binary: {result.stdout.strip()}")


def install_deps(ssh_host, ssh_port):
    """Install Python dependencies (numpy + numba for L0 generation)."""
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
            print("  tmux: OK")
    except subprocess.TimeoutExpired:
        print("Warning: tmux install timed out (non-fatal)")

    # numpy + numba: needed for L0 checkpoint generation (run_cascade.py uses numba JIT)
    cmd = "pip install -q numpy numba 2>&1 | tail -5"
    result = ssh_run(ssh_host, ssh_port, cmd, timeout=300)
    if isinstance(result, int):
        pass
    elif result.returncode != 0:
        print(f"Warning: pip install issues: {result.stderr}")
    else:
        print("  numpy + numba: OK")


def run_cascade(ssh_host, ssh_port, level, max_survivors=None,
                timeout=None, log_dir=None, m=None, c_target=None):
    """Run the GPU cascade prover for a specific level."""
    from datetime import datetime

    log_file = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"gpu_L{level}_{timestamp}.log")
        print(f"Logging to: {log_file}")

    env_prefix = ""
    if m is not None:
        env_prefix += f"M={m} "
    if c_target is not None:
        env_prefix += f"C_TARGET={c_target} "

    cmd = (
        f"cd {REMOTE_WORKDIR}/gpu && "
        f"chmod +x run.sh && "
        f"{env_prefix}./run.sh {level}"
    )
    if max_survivors:
        cmd += f" {max_survivors}"

    print(f"Running cascade level {level}...")
    rc = ssh_run(ssh_host, ssh_port, cmd, timeout=timeout, stream=True,
                 log_file=log_file)
    if rc != 0:
        print(f"Cascade level {level} exited with code {rc}")
    return rc


def run_full_proof(ssh_host, ssh_port, m=35, c_target=1.33, max_level=3,
                   timeout=None, log_dir=None):
    """Run the full GPU cascade proof (L0 generation + all levels)."""
    from datetime import datetime

    log_file = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"gpu_proof_m{m}_c{c_target}_{timestamp}.log")
        print(f"Logging to: {log_file}")

    cmd = (
        f"cd {REMOTE_WORKDIR}/gpu && "
        f"chmod +x run_full_proof.sh && "
        f"./run_full_proof.sh --m {m} --c_target {c_target} --max_level {max_level}"
    )

    print(f"Running full proof: m={m}, c_target={c_target}, max_level={max_level}...")
    rc = ssh_run(ssh_host, ssh_port, cmd, timeout=timeout, stream=True,
                 log_file=log_file)
    if rc != 0:
        print(f"Full proof exited with code {rc}")
    return rc


def run_custom(ssh_host, ssh_port, command, timeout=None, log_dir=None):
    """Run an arbitrary command on the pod."""
    from datetime import datetime

    log_file = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"gpu_custom_{timestamp}.log")

    cmd = f"cd {REMOTE_WORKDIR} && {command}"
    return ssh_run(ssh_host, ssh_port, cmd, timeout=timeout, stream=True,
                   log_file=log_file)


# ---------- Detached job management ----------

REMOTE_LOG = f"{REMOTE_WORKDIR}/data/gpu_job.log"
TMUX_SESSION = "gpujob"


def launch_cascade(ssh_host, ssh_port, level, max_survivors=None,
                   auto_teardown=False, api_key=None, pod_id=None,
                   m=None, c_target=None):
    """Launch cascade prover in a detached tmux session."""
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

    env_prefix = ""
    if m is not None:
        env_prefix += f"M={m} "
    if c_target is not None:
        env_prefix += f"C_TARGET={c_target} "

    run_cmd = f"cd {REMOTE_WORKDIR}/gpu && {env_prefix}./run.sh {level}"
    if max_survivors:
        run_cmd += f" {max_survivors}"

    tmux_inner = (
        f"set -o pipefail; "
        f"{run_cmd} 2>&1 | tee {REMOTE_LOG}; "
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


def launch_full_proof(ssh_host, ssh_port, m=35, c_target=1.33, max_level=3,
                      auto_teardown=False, api_key=None, pod_id=None):
    """Launch the full cascade proof in a detached tmux session."""
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

    run_cmd = (
        f"cd {REMOTE_WORKDIR}/gpu && "
        f"chmod +x run_full_proof.sh && "
        f"./run_full_proof.sh --m {m} --c_target {c_target} --max_level {max_level}"
    )

    tmux_inner = (
        f"set -o pipefail; "
        f"{run_cmd} 2>&1 | tee {REMOTE_LOG}; "
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
                     timeout=60)
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
