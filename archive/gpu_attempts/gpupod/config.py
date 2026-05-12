"""Configuration constants for RunPod GPU pod integration."""
import os
import platform
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# RunPod API key
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")

# GPU pod configuration
# Docker image with CUDA 12.x + development tools (nvcc, headers)
DOCKER_IMAGE = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
GPU_TYPE = "NVIDIA H100 80GB HBM3"
GPU_COUNT = int(os.environ.get("GPU_COUNT", "8"))
CLOUD_TYPE = os.environ.get("CLOUD_TYPE", "ALL")  # ALL = spot+on-demand, SECURE = on-demand only
CONTAINER_DISK_GB = 100
VOLUME_IN_GB = 200

# Cost tracking (8x H100 SXM spot ~$2/GPU/hr = $16/hr)
COST_PER_HOUR = float(os.environ.get("COST_PER_HOUR", "16.0"))
BUDGET_LIMIT = float(os.environ.get("BUDGET_LIMIT", "100.0"))
BUDGET_WARN_PCT = 0.80

# Paths
SESSION_FILE = Path(__file__).resolve().parent / ".session.json"
REMOTE_WORKDIR = "/workspace/sidon-autocorrelation"

# SSH configuration
SSH_KEY_PATH = Path(os.environ.get(
    "SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_ed25519")
))


def _to_msys_path(win_path):
    """Convert Windows path to MSYS/Git Bash path."""
    s = str(win_path).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        s = "/" + s[0].lower() + s[2:]
    return s


# SSH options with Windows-native paths
SSH_OPTIONS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=10",
    "-i", str(SSH_KEY_PATH),
]

# SSH options with MSYS paths (for bash -c pipeline strings on Windows)
if platform.system() == "Windows":
    SSH_OPTIONS_BASH = [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-i", _to_msys_path(SSH_KEY_PATH),
    ]
else:
    SSH_OPTIONS_BASH = SSH_OPTIONS

# Files/dirs to exclude from sync
SYNC_EXCLUDES = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "*.pyc",
    ".env",
    "exploration",
    "cloud",
    "cloud_results",
    "data",
    ".claude",
    "cpupod/.session.json",
    "gpupod/.session.json",
    "*.dll",
    "*.so",
    "*.egg-info",
    "lean",
    ".lake",
    "proof",
    "prompts",
    "docs",
    "baseline",
    "*.npy",
    "*.dat",
    "*.pdf",
]
