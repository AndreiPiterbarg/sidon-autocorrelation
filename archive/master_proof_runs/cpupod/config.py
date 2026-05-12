"""Configuration constants for RunPod CPU pod integration."""
import os
import platform
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# RunPod API key
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")

# CPU pod configuration
# Available CPU3 sizes (all have same vCPU, differ in RAM):
#   cpu3c (Compute-Optimized, 2x RAM):  cpu3c-32-64   → 32 vCPU,  64 GB
#   cpu3g (General Purpose,   4x RAM):  cpu3g-32-128  → 32 vCPU, 128 GB
#   cpu3m (Memory-Optimized,  8x RAM):  cpu3m-32-256  → 32 vCPU, 256 GB
INSTANCE_ID = "cpu3m-32-256"
TEMPLATE_ID = "runpod-ubuntu"
CLOUD_TYPE = "SECURE"  # SECURE = on-demand (not spot)
CONTAINER_DISK_GB = 320  # GB container disk (RunPod max, ~$0.048/hr)

# Cost tracking
COST_PER_HOUR = 0.53   # USD/hr for CPU3 Compute-Optimized 32vCPU/64GB
BUDGET_LIMIT = 106.0   # USD per session (~100 hrs at $1.06/hr)
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
