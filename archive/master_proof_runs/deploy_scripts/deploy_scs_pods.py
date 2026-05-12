#!/usr/bin/env python
"""Deploy two CPU pods running CVXPY+SCS Lasserre solver.

Pod A: d=14 O3 bw=13 (full) — uses ~1-5 GB RAM
Pod B: d=16 O3 bw=12         — uses ~5-20 GB RAM

Both fit easily in 256 GB. SCS is ~100x slower per-solve than MOSEK
but uses O(nnz) memory instead of O(n_y^2 * PSD_dim).

Usage:
    python deploy_scs_pods.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpupod.config import (
    RUNPOD_API_KEY, SSH_OPTIONS, SSH_KEY_PATH, REMOTE_WORKDIR,
)

PYTHON = "python3.13"
MOSEK_LIC = Path.home() / "mosek" / "mosek.lic"

FILES_TO_SYNC = [
    'tests/run_single_cvxpy.py', 'tests/lasserre_highd.py',
    'tests/lasserre_fusion.py', 'tests/lasserre_scalable.py',
    'tests/lasserre_enhanced.py',
    'lasserre/__init__.py', 'lasserre/core.py',
    'lasserre/precompute.py', 'lasserre/cliques.py', 'lasserre/solvers.py',
]

JOBS = [
    {'name': 'scs-d14-full', 'script': 'tests/run_single_cvxpy.py',
     'args': '--d 14 --order 3 --bw 13 --cg-rounds 10 --bisect 8'},
    {'name': 'scs-d16-bw12', 'script': 'tests/run_single_cvxpy.py',
     'args': '--d 16 --order 3 --bw 12 --cg-rounds 10 --bisect 8'},
]


def ssh(host, port, cmd, t=300):
    return subprocess.run(
        ['ssh', '-p', str(port)] + SSH_OPTIONS + [f'root@{host}', cmd],
        capture_output=True, text=True, timeout=t, stdin=subprocess.DEVNULL)


def setup_pod(name):
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    pub_key = SSH_KEY_PATH.with_suffix('.pub').read_text().strip()

    print(f'Creating pod {name}...')
    pod = runpod.create_pod(
        name=name, image_name='runpod/base:0.7.0-ubuntu2004',
        instance_id='cpu3m-32-256', cloud_type='SECURE',
        container_disk_in_gb=320, env={'PUBLIC_KEY': pub_key})
    pod_id = pod['id']

    host = port = None
    for attempt in range(60):
        time.sleep(5)
        st = runpod.get_pod(pod_id)
        rt = st.get('runtime')
        if rt and rt.get('ports'):
            for p in rt['ports']:
                if p.get('privatePort') == 22 and p.get('ip'):
                    host, port = p['ip'], p['publicPort']
                    break
            if host:
                break
        if attempt % 6 == 5:
            print(f'  Waiting... ({attempt*5}s)')

    if not host:
        raise RuntimeError(f'Pod {pod_id} SSH not ready')
    print(f'  SSH: {host}:{port}')

    # Check cgroup limit
    r = ssh(host, port, 'cat /sys/fs/cgroup/memory.max 2>/dev/null')
    limit_gb = int(r.stdout.strip()) / 1e9 if r.stdout.strip().isdigit() else 0
    print(f'  Cgroup limit: {limit_gb:.0f} GB')

    # Install deps: cvxpy, scs, scipy, numpy (NO mosek needed!)
    print(f'  Installing cvxpy + scs + scipy...')
    ssh(host, port, f'{PYTHON} -m pip install -q cvxpy scs scipy numpy 2>&1 | tail -3')

    # Verify
    r = ssh(host, port, f'{PYTHON} -c "import cvxpy, scs; print(f\'cvxpy={{cvxpy.__version__}}, scs={{scs.__version__}}\')"', t=30)
    print(f'  Verified: {r.stdout.strip()}')

    # tmux
    ssh(host, port, 'which tmux >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null 2>&1)')

    # Sync files
    print(f'  Syncing files...')
    ssh(host, port, f'mkdir -p {REMOTE_WORKDIR}/tests {REMOTE_WORKDIR}/lasserre')
    for f in FILES_TO_SYNC:
        r = subprocess.run(
            ['scp', '-P', str(port)] + SSH_OPTIONS +
            [f, f'root@{host}:{REMOTE_WORKDIR}/{os.path.dirname(f)}/'],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f'    RETRY {f}...')
            time.sleep(2)
            subprocess.run(
                ['scp', '-P', str(port)] + SSH_OPTIONS +
                [f, f'root@{host}:{REMOTE_WORKDIR}/{os.path.dirname(f)}/'],
                capture_output=True, timeout=120)

    # Verify key file
    r = ssh(host, port, f'ls {REMOTE_WORKDIR}/tests/run_single_cvxpy.py', t=10)
    if 'run_single_cvxpy' not in r.stdout:
        raise RuntimeError('File sync failed!')
    print(f'  Files OK.')

    return {'pod_id': pod_id, 'ssh_host': host, 'ssh_port': port}


def main():
    if not RUNPOD_API_KEY:
        print('ERROR: RUNPOD_API_KEY not set')
        sys.exit(1)

    state = {}
    for job in JOBS:
        print(f'\n{"="*60}')
        info = setup_pod(job['name'])
        state[job['name']] = info

        # Launch
        log = f'{REMOTE_WORKDIR}/data/cpu_job.log'
        script = job['script']
        args = job['args']
        inner = (f'set -o pipefail; cd {REMOTE_WORKDIR} && '
                 f'{PYTHON} -u {script} {args} '
                 f'2>&1 | tee {log}; echo ===JOB_EXIT_CODE=\\$?=== >> {log}; true')
        cmd = (f"tmux kill-session -t job 2>/dev/null; true && "
               f"mkdir -p {REMOTE_WORKDIR}/data && "
               f"tmux new-session -d -s job bash -c '{inner}'")
        ssh(info['ssh_host'], info['ssh_port'], cmd, t=30)
        print(f'  LAUNCHED: {args}')

    # Save state
    os.makedirs('data', exist_ok=True)
    with open('data/two_pods_state.json', 'w') as f:
        json.dump(state, f, indent=2)

    # Verify
    time.sleep(5)
    print(f'\n{"="*60}')
    print('STATUS:')
    for name, info in state.items():
        r = ssh(info['ssh_host'], info['ssh_port'],
                'tmux has-session -t job 2>/dev/null && echo RUNNING || echo STOPPED', t=10)
        print(f'  {name}: {r.stdout.strip()}')

    print(f'\nMonitor:')
    for name, info in state.items():
        print(f'  ssh -p {info["ssh_port"]} -i ~/.ssh/id_ed25519 '
              f'-o StrictHostKeyChecking=no root@{info["ssh_host"]} '
              f'"tail -30 {REMOTE_WORKDIR}/data/cpu_job.log"')
    print(f'\nTeardown: python -m cpupod cleanup')


if __name__ == '__main__':
    main()
