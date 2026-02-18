"""Configuration constants and .env loading for RunPod integration."""
import os
import platform
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# RunPod API key
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")

# Pod configuration
GPU_TYPE = "NVIDIA A100-SXM4-80GB"
DOCKER_IMAGE = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
CLOUD_TYPE = "SECURE"  # SECURE = on-demand (not spot)

# Cost tracking
COST_PER_HOUR = 1.49  # USD/hr for A100 SXM 80GB on-demand
BUDGET_LIMIT = 25.0   # USD per session
BUDGET_WARN_PCT = 0.80  # Warn at 80% of budget

# Paths
SESSION_FILE = Path(__file__).resolve().parent / ".session.json"
REMOTE_WORKDIR = "/workspace/sidon-autocorrelation"

# SSH configuration
SSH_KEY_PATH = Path(os.environ.get(
    "SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_ed25519")
))


def _to_msys_path(win_path):
    """Convert Windows path to MSYS/Git Bash path (e.g. C:\\Users -> /c/Users)."""
    s = str(win_path).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        s = "/" + s[0].lower() + s[2:]
    return s


# SSH options with Windows-native paths (for subprocess list calls)
SSH_OPTIONS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
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
    "gpupod/.session.json",
    "*.dll",
    "*.so",
    "*.egg-info",
]
