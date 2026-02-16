"""Remote SSH command execution on RunPod pods."""
import subprocess
import sys

from .config import SSH_OPTIONS, REMOTE_WORKDIR


def ssh_run(ssh_host, ssh_port, command, timeout=None, stream=False):
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
    """Install Python dependencies on the remote pod."""
    print("Installing dependencies on pod...")
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


def verify_gpu(ssh_host, ssh_port):
    """Verify GPU is accessible and print device info."""
    print("Verifying GPU...")
    cmd = "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
    result = ssh_run(ssh_host, ssh_port, cmd, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"GPU verification failed: {result.stderr}")
    gpu_info = result.stdout.strip()
    print(f"GPU: {gpu_info}")
    return gpu_info


def run_script(ssh_host, ssh_port, script="cloninger-steinerberger/gpu/solvers.py",
               args="", timeout=None):
    """Run a Python script on the remote pod, streaming output."""
    print(f"Running: {script} {args}")
    cmd = f"cd {REMOTE_WORKDIR} && python {script} {args}"
    rc = ssh_run(ssh_host, ssh_port, cmd, timeout=timeout, stream=True)
    if rc != 0:
        print(f"Script exited with code {rc}")
    return rc
