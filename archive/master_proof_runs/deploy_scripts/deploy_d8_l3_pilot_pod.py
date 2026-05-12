#!/usr/bin/env python
"""Deploy H200 SXM pod for the d=8 L3 FULL Lasserre pilot.

The pilot answers the critical question: does MOSEK solve the full-L3
formulation at d=8 with ~99% gap closure in reasonable time? The
previous attempt was killed at >30 min/round; we're giving it a beefy
machine and a longer budget to get the actual data point.

Usage:
    python deploy_d8_l3_pilot_pod.py              # create pod + sync + run
    python deploy_d8_l3_pilot_pod.py --sync-only  # sync files only
    python deploy_d8_l3_pilot_pod.py --run-only   # just run pilot
    python deploy_d8_l3_pilot_pod.py --log        # tail the running log
    python deploy_d8_l3_pilot_pod.py --teardown   # destroy pod
    python deploy_d8_l3_pilot_pod.py --ssh        # print SSH command
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpupod.config import (
    RUNPOD_API_KEY, SSH_OPTIONS, SSH_KEY_PATH, REMOTE_WORKDIR,
    DOCKER_IMAGE,
)

STATE_FILE = (Path(__file__).resolve().parent / 'data'
              / 'd8_l3_pilot_pod_state.json')
MOSEK_LIC_LOCAL = Path.home() / 'mosek' / 'mosek.lic'
GPU_TYPE_ID = 'NVIDIA H200'  # 141 GB HBM3e; used for CPU+RAM, MOSEK is CPU-only

FILES_TO_SYNC = [
    'tests/lasserre_enhanced.py',
    'tests/lasserre_fusion.py',
    'tests/lasserre_scalable.py',
    'tests/lasserre_highd.py',
    'tests/admm_gpu_solver.py',
    'tests/run_scs_direct.py',
    'lasserre/__init__.py',
    'lasserre/core.py',
    'lasserre/precompute.py',
    'lasserre/cliques.py',
    'lasserre/solvers.py',
    'lasserre/z2_symmetry.py',
]
# Optional files: sync only if they exist (newer repos may have these)
OPTIONAL_SYNC = [
    'lasserre/gap_accelerator.py',
]


def ssh(host, port, cmd, t=600):
    return subprocess.run(
        ['ssh', '-p', str(port)] + SSH_OPTIONS + [f'root@{host}', cmd],
        capture_output=True, text=True, timeout=t, stdin=subprocess.DEVNULL)


def scp_to(host, port, local, remote):
    return subprocess.run(
        ['scp', '-P', str(port)] + SSH_OPTIONS + [local, f'root@{host}:{remote}'],
        capture_output=True, text=True, timeout=120)


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return None


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def create_pod():
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    pub_path = SSH_KEY_PATH.with_suffix('.pub')
    if not pub_path.exists():
        raise RuntimeError(f'Missing SSH public key at {pub_path}')
    pub_key = pub_path.read_text().strip()

    print(f'Creating 1x {GPU_TYPE_ID} pod for d=8 L3 pilot...')
    pod = runpod.create_pod(
        name='sidon-d8-l3-pilot',
        image_name=DOCKER_IMAGE,
        gpu_type_id=GPU_TYPE_ID,
        gpu_count=1,
        cloud_type='ALL',
        container_disk_in_gb=60,
        volume_in_gb=0,
        env={'PUBLIC_KEY': pub_key},
        ports='22/tcp',
    )
    pod_id = pod['id']
    print(f'  Pod ID: {pod_id}')

    host = port = None
    for attempt in range(80):  # up to ~400s for pod to boot
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
            print(f'  Waiting for SSH... ({attempt*5}s)')

    if not host:
        raise RuntimeError(f'Pod {pod_id} SSH not ready after 400s')

    state = {'pod_id': pod_id, 'ssh_host': host, 'ssh_port': port,
             'created_at': time.time(), 'gpu_type': GPU_TYPE_ID}
    save_state(state)
    print(f'  SSH ready: root@{host} -p {port}')
    return state


def wait_for_ssh(host, port, timeout=180):
    """Wait until SSH actually accepts a connection (pod may still be booting)."""
    start = time.time()
    while time.time() - start < timeout:
        r = ssh(host, port, 'echo ok', t=15)
        if r.returncode == 0 and 'ok' in r.stdout:
            return True
        time.sleep(5)
    return False


def install_mosek_and_deps(host, port):
    """Install MOSEK Python + license on the pod."""
    print('Installing MOSEK + deps (pip + license copy)...')
    r = ssh(host, port,
            'pip install mosek numpy scipy cvxpy scs 2>&1 | tail -6',
            t=600)
    print(f'  pip: {r.stdout.strip()[-500:] if r.stdout else r.stderr.strip()[-500:]}')

    if not MOSEK_LIC_LOCAL.exists():
        raise RuntimeError(f'No MOSEK license at {MOSEK_LIC_LOCAL}')

    print('  Copying MOSEK academic license to pod...')
    ssh(host, port, 'mkdir -p /root/mosek', t=30)
    r = scp_to(host, port, str(MOSEK_LIC_LOCAL), '/root/mosek/mosek.lic')
    print(f'  license copy: {"ok" if r.returncode == 0 else r.stderr.strip()}')

    r = ssh(host, port,
            'python -c "import mosek; m = mosek.Task(); print(\'mosek ok, version\', '
            'mosek.Env.getversion())" 2>&1', t=60)
    print(f'  mosek import: {r.stdout.strip()}')

    r = ssh(host, port, 'nproc ; free -g | head -2', t=30)
    print(f'  host resources:\n{r.stdout}')


def sync_files(host, port):
    print('Syncing code to pod...')
    ssh(host, port,
        f'mkdir -p {REMOTE_WORKDIR}/tests {REMOTE_WORKDIR}/lasserre '
        f'{REMOTE_WORKDIR}/data',
        t=30)
    all_files = list(FILES_TO_SYNC)
    for f in OPTIONAL_SYNC:
        if (Path(__file__).resolve().parent / f).exists():
            all_files.append(f)
    for f in all_files:
        local = Path(__file__).resolve().parent / f
        if not local.exists():
            print(f'  SKIP (missing): {f}')
            continue
        remote = f'{REMOTE_WORKDIR}/{f}'
        r = scp_to(host, port, str(local), remote)
        status = 'ok' if r.returncode == 0 else f'FAIL: {r.stderr.strip()}'
        print(f'  {f} -> {status}')


def run_pilot(host, port):
    """Run d=8 L3 full pilot in background, capture log, return immediately.

    Reads the log later with --log.
    """
    print('\n=== d=8 L3 full pilot (MOSEK, CPU-bound) ===', flush=True)
    log_path = f'{REMOTE_WORKDIR}/data/d8_l3_full_pilot.log'
    # Background launch so we can poll without holding SSH.
    launch = (
        f'cd {REMOTE_WORKDIR} && '
        f'mkdir -p data && '
        f'PYTHONPATH={REMOTE_WORKDIR}:{REMOTE_WORKDIR}/tests '
        f'nohup python -u tests/lasserre_enhanced.py '
        f'  --d 8 --order 3 --psd full --c_target 1.28 '
        f'  --bisect 12 --max-rounds 20 --max-add 20 '
        f'  > {log_path} 2>&1 & '
        f'echo PID=$!'
    )
    r = ssh(host, port, launch, t=30)
    print(f'  launched: {r.stdout.strip()}')
    print(f'  log: {log_path}')
    print(f'  tail later with: python deploy_d8_l3_pilot_pod.py --log')


def tail_log(host, port, n=120):
    log_path = f'{REMOTE_WORKDIR}/data/d8_l3_full_pilot.log'
    r = ssh(host, port,
            f'ls -la {log_path} 2>&1 | head; echo "----"; '
            f'tail -n {n} {log_path} 2>&1; echo "----"; '
            f'pgrep -af lasserre_enhanced 2>&1 | head',
            t=30)
    print(r.stdout)


def fetch_result(host, port):
    """Try to pull any result JSON the solver wrote."""
    print('Fetching result artefacts...')
    r = ssh(host, port,
            f'ls -la {REMOTE_WORKDIR}/data/ 2>&1 | head -40', t=30)
    print(r.stdout)
    local_data = Path(__file__).resolve().parent / 'data' / 'pilot_d8_l3_full'
    local_data.mkdir(parents=True, exist_ok=True)
    # Pull the log and any result_*.json files
    subprocess.run(
        ['scp', '-P', str(port)] + SSH_OPTIONS +
        [f'root@{host}:{REMOTE_WORKDIR}/data/d8_l3_full_pilot.log',
         str(local_data / 'd8_l3_full_pilot.log')],
        capture_output=True, text=True, timeout=120)
    print(f'  saved log to {local_data}/')


def teardown():
    state = load_state()
    if not state:
        print('No pilot pod state.')
        return
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    print(f'Terminating pod {state["pod_id"]}...')
    runpod.terminate_pod(state['pod_id'])
    STATE_FILE.unlink(missing_ok=True)
    print('Done.')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync-only', action='store_true')
    parser.add_argument('--run-only', action='store_true')
    parser.add_argument('--log', action='store_true',
                        help='Tail the pilot log')
    parser.add_argument('--fetch', action='store_true',
                        help='Fetch log + results locally')
    parser.add_argument('--teardown', action='store_true')
    parser.add_argument('--ssh', action='store_true')
    args = parser.parse_args()

    if args.teardown:
        teardown()
        return

    state = load_state()

    if args.ssh:
        if state:
            key = str(SSH_KEY_PATH)
            print(f'ssh -p {state["ssh_port"]} -i {key} '
                  f'-o StrictHostKeyChecking=no root@{state["ssh_host"]}')
        return

    if args.log:
        if not state:
            print('No pilot pod state.')
            return
        tail_log(state['ssh_host'], state['ssh_port'])
        return

    if args.fetch:
        if not state:
            print('No pilot pod state.')
            return
        fetch_result(state['ssh_host'], state['ssh_port'])
        return

    if args.sync_only:
        if not state:
            print('No pilot pod state. Run without --sync-only first.')
            return
        sync_files(state['ssh_host'], state['ssh_port'])
        return

    if args.run_only:
        if not state:
            print('No pilot pod state. Run without --run-only first.')
            return
        run_pilot(state['ssh_host'], state['ssh_port'])
        return

    # Full deploy
    if state:
        print(f'Existing pilot pod state: {state["pod_id"]}')
        r = ssh(state['ssh_host'], state['ssh_port'], 'echo ok', t=15)
        if r.returncode == 0 and 'ok' in r.stdout:
            print('  pod alive, reusing (use --teardown to destroy and recreate).')
        else:
            print('  pod unreachable, creating new...')
            state = create_pod()
    else:
        state = create_pod()

    if not wait_for_ssh(state['ssh_host'], state['ssh_port']):
        raise RuntimeError('SSH never came up on pilot pod.')

    install_mosek_and_deps(state['ssh_host'], state['ssh_port'])
    sync_files(state['ssh_host'], state['ssh_port'])
    run_pilot(state['ssh_host'], state['ssh_port'])


if __name__ == '__main__':
    main()
