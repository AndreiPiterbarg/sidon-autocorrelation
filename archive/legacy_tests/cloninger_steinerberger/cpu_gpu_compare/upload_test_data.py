"""Upload test data to GPU pod for comparison."""
import glob
import json
import os
import platform
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gpupod.sync import _run_bash, _ssh_opts_str, _to_msys_path
from gpupod.config import REMOTE_WORKDIR

session = json.load(open(os.path.join(os.path.dirname(__file__),
                                       '..', 'gpupod', '.session.json')))
host = session['ssh_host']
port = session['ssh_port']
ssh_opts = _ssh_opts_str()

# Create dirs on remote
cmd = (f"ssh -p {port} {ssh_opts} root@{host} "
       f"'mkdir -p {REMOTE_WORKDIR}/tests/cpu_gpu_data "
       f"{REMOTE_WORKDIR}/data'")
r = _run_bash(cmd)
print(f"mkdir: rc={r.returncode}")

# Upload test data files
test_dir = os.path.join(os.path.dirname(__file__), 'cpu_gpu_data')
test_files = glob.glob(os.path.join(test_dir, '*'))
print(f"Uploading {len(test_files)} test data files...")

for f in test_files:
    fname = os.path.basename(f)
    local = _to_msys_path(f) if platform.system() == 'Windows' else f
    remote = f"{REMOTE_WORKDIR}/tests/cpu_gpu_data/{fname}"
    scp = f"scp -P {port} {ssh_opts} '{local}' root@{host}:{remote}"
    r = _run_bash(scp, timeout=120)
    status = "OK" if r.returncode == 0 else f"FAIL: {r.stderr.strip()}"
    print(f"  {fname}: {status}")

# Upload standalone comparison scripts
for script in ['gpu_compare_standalone.py', 'compare_checkpoint.py']:
    path = os.path.join(os.path.dirname(__file__), script)
    if os.path.exists(path):
        local = _to_msys_path(path) if platform.system() == 'Windows' else path
        scp = f"scp -P {port} {ssh_opts} '{local}' root@{host}:{REMOTE_WORKDIR}/tests/{script}"
        r = _run_bash(scp, timeout=60)
        print(f"  {script}: {'OK' if r.returncode == 0 else 'FAIL'}")

# Upload checkpoint files
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
for ckpt in ['checkpoint_L0_survivors.npy', 'checkpoint_L1_survivors.npy',
             'checkpoint_L2_survivors.npy']:
    path = os.path.join(data_dir, ckpt)
    if os.path.exists(path):
        local = _to_msys_path(path) if platform.system() == 'Windows' else path
        remote = f"{REMOTE_WORKDIR}/data/{ckpt}"
        scp = f"scp -P {port} {ssh_opts} '{local}' root@{host}:{remote}"
        r = _run_bash(scp, timeout=300)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {ckpt} ({size_mb:.1f}MB): {'OK' if r.returncode == 0 else 'FAIL'}")

print("\nDone.")
