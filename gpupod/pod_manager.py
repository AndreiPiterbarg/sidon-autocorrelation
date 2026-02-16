"""Pod lifecycle management via the runpod SDK."""
import time

import runpod as runpod_sdk

from .config import RUNPOD_API_KEY, GPU_TYPE, DOCKER_IMAGE, CLOUD_TYPE, SSH_KEY_PATH


def _init_sdk():
    """Initialize the RunPod SDK with our API key."""
    if not RUNPOD_API_KEY:
        raise RuntimeError(
            "RUNPOD_API_KEY not set. Create a .env file with:\n"
            "  RUNPOD_API_KEY=your_key_here"
        )
    runpod_sdk.api_key = RUNPOD_API_KEY


def create_pod(name="sidon-gpu"):
    """Create a new GPU pod and wait for it to be ready.

    If RUNPOD_VOLUME_ID is set, attaches the network volume at /workspace
    so all data persists across pod restarts.

    Returns dict with pod_id, ssh_host, ssh_port.
    """
    _init_sdk()

    # Read SSH public key
    pub_key_path = SSH_KEY_PATH.with_suffix(".pub")
    if not pub_key_path.exists():
        raise RuntimeError(
            f"SSH public key not found at {pub_key_path}\n"
            "Generate one with: ssh-keygen -t ed25519"
        )
    pub_key = pub_key_path.read_text().strip()

    # Build create_pod kwargs
    # Uses the built-in volume disk (persistent at /workspace across stop/restart).
    # Results are synced back locally before teardown, so no network volume needed.
    kwargs = dict(
        name=name,
        image_name=DOCKER_IMAGE,
        gpu_type_id=GPU_TYPE,
        cloud_type=CLOUD_TYPE,
        gpu_count=1,
        volume_in_gb=75,
        container_disk_in_gb=50,
        env={"PUBLIC_KEY": pub_key},
        ports="22/tcp",
    )

    print(f"Creating pod '{name}' with {GPU_TYPE}...")

    pod = runpod_sdk.create_pod(**kwargs)

    if not pod or "id" not in pod:
        raise RuntimeError(f"Pod creation failed. API response: {pod}")

    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")

    # Wait for RUNNING status with SSH available
    print("Waiting for pod to start...", end="", flush=True)
    for _ in range(120):  # up to 10 minutes
        status = get_pod_status(pod_id)
        if status and status.get("desiredStatus") == "RUNNING":
            runtime = status.get("runtime")
            if runtime and runtime.get("ports"):
                ssh_info = _extract_ssh_info(runtime["ports"])
                if ssh_info:
                    print(" ready!")
                    return {
                        "pod_id": pod_id,
                        "ssh_host": ssh_info["host"],
                        "ssh_port": ssh_info["port"],
                    }
        print(".", end="", flush=True)
        time.sleep(5)

    raise RuntimeError(f"Pod {pod_id} did not become ready within 10 minutes")


def _extract_ssh_info(ports):
    """Extract SSH host/port from RunPod port mappings."""
    if not ports:
        return None
    for port_info in ports:
        if port_info.get("privatePort") == 22:
            ip = port_info.get("ip")
            port = port_info.get("publicPort")
            if ip and port:
                return {"host": ip, "port": port}
    return None


def get_pod_status(pod_id):
    """Get current pod status."""
    _init_sdk()
    try:
        return runpod_sdk.get_pod(pod_id)
    except Exception:
        return None


def terminate_pod(pod_id):
    """Terminate (destroy) a pod."""
    _init_sdk()
    print(f"Terminating pod {pod_id}...")
    runpod_sdk.terminate_pod(pod_id)
    print("Pod terminated.")


def list_pods():
    """List all active pods."""
    _init_sdk()
    return runpod_sdk.get_pods()


def terminate_all_pods():
    """Emergency: terminate ALL pods."""
    _init_sdk()
    pods = list_pods()
    if not pods:
        print("No active pods.")
        return
    for pod in pods:
        pid = pod.get("id", "")
        name = pod.get("name", "unknown")
        print(f"Terminating {name} ({pid})...")
        try:
            runpod_sdk.terminate_pod(pid)
        except Exception as e:
            print(f"  Warning: {e}")
    print(f"Terminated {len(pods)} pod(s).")
