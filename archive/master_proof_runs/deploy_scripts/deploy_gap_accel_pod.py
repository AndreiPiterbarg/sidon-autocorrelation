#!/usr/bin/env python
"""Deploy a NEW GPU pod dedicated to gap_accelerator testing.

Uses a separate state file (data/gap_accel_pod_state.json) so it does
not disturb any existing bench/prod pod.

Usage:
    python deploy_gap_accel_pod.py              # new pod + sync + run tests
    python deploy_gap_accel_pod.py --sync-only  # sync files to existing gap pod
    python deploy_gap_accel_pod.py --run-only   # just run tests
    python deploy_gap_accel_pod.py --teardown   # destroy only the gap pod
    python deploy_gap_accel_pod.py --ssh        # print SSH command
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
              / 'gap_accel_pod_state.json')

FILES_TO_SYNC = [
    'tests/run_scs_direct.py',
    'tests/admm_gpu_solver.py',
    'tests/lasserre_highd.py',
    'tests/test_gap_accelerator.py',
    'lasserre/__init__.py',
    'lasserre/core.py',
    'lasserre/precompute.py',
    'lasserre/cliques.py',
    'lasserre/solvers.py',
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
    pub_key = SSH_KEY_PATH.with_suffix('.pub').read_text().strip()

    print('Creating NEW gap-accel GPU pod (1x H100)...')
    pod = runpod.create_pod(
        name='sidon-gap-accel',
        image_name=DOCKER_IMAGE,
        gpu_type_id='NVIDIA H100 80GB HBM3',
        gpu_count=1,
        cloud_type='ALL',
        container_disk_in_gb=50,
        volume_in_gb=0,
        env={'PUBLIC_KEY': pub_key},
        ports='22/tcp',
    )
    pod_id = pod['id']
    print(f'  Pod ID: {pod_id}')

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

    state = {'pod_id': pod_id, 'ssh_host': host, 'ssh_port': port}
    save_state(state)
    print(f'  SSH: root@{host} -p {port}')
    return state


def setup_deps(host, port):
    print('Installing dependencies...')
    r = ssh(host, port, 'pip install scipy numpy scs pytest 2>&1 | tail -5', t=300)
    print(f'  {r.stdout.strip()[-200:] if r.stdout else r.stderr.strip()[-200:]}')

    r = ssh(host, port,
            'python -c "import torch; print(f\'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name() if torch.cuda.is_available() else None}\')"',
            t=30)
    print(f'  {r.stdout.strip()}')


def sync_files(host, port):
    print('Syncing files...')
    ssh(host, port,
        f'mkdir -p {REMOTE_WORKDIR}/tests {REMOTE_WORKDIR}/lasserre '
        f'{REMOTE_WORKDIR}/data')
    for f in FILES_TO_SYNC:
        local = str(Path(__file__).resolve().parent / f)
        remote = f'{REMOTE_WORKDIR}/{f}'
        r = scp_to(host, port, local, remote)
        status = 'ok' if r.returncode == 0 else f'FAIL: {r.stderr.strip()}'
        print(f'  {f} -> {status}')


def run_tests(host, port):
    print('\n=== Phase 1: pytest tests/test_gap_accelerator.py ===', flush=True)
    cmd = (f'cd {REMOTE_WORKDIR} && '
           f'PYTHONPATH={REMOTE_WORKDIR}:{REMOTE_WORKDIR}/tests '
           f'python -m pytest tests/test_gap_accelerator.py -v 2>&1')
    r = ssh(host, port, cmd, t=300)
    print(r.stdout)
    if r.stderr:
        print(f'STDERR: {r.stderr[-500:]}')
    if r.returncode != 0:
        print('Unit tests FAILED; skipping GPU integration run.')
        return r.returncode

    print('\n=== Phase 2: d=8 L2 baseline vs --use-gap-accel (GPU) ===',
          flush=True)
    cmd_base = (f'cd {REMOTE_WORKDIR} && '
                f'PYTHONPATH={REMOTE_WORKDIR}:{REMOTE_WORKDIR}/tests '
                f'python tests/run_scs_direct.py --d 8 --order 2 --bw 6 '
                f'--cg-rounds 3 --bisect 6 --gpu 2>&1')
    r = ssh(host, port, cmd_base, t=900)
    print('---- BASELINE ----')
    print(r.stdout[-3000:])

    cmd_accel = (f'cd {REMOTE_WORKDIR} && '
                 f'PYTHONPATH={REMOTE_WORKDIR}:{REMOTE_WORKDIR}/tests '
                 f'python tests/run_scs_direct.py --d 8 --order 2 --bw 6 '
                 f'--cg-rounds 3 --bisect 6 --gpu --use-gap-accel 2>&1')
    r = ssh(host, port, cmd_accel, t=900)
    print('---- WITH gap_accel ----')
    print(r.stdout[-3000:])

    print('\n=== Phase 3: fetch gap_report JSON ===', flush=True)
    r = ssh(host, port,
            f'cat {REMOTE_WORKDIR}/data/gap_report_d8_o2_bw6_scs.json '
            f'2>&1 || echo "no gap report written"', t=30)
    print(r.stdout)

    return 0


def teardown():
    state = load_state()
    if not state:
        print('No gap-accel pod state found.')
        return
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    print(f'Terminating gap-accel pod {state["pod_id"]}...')
    runpod.terminate_pod(state['pod_id'])
    STATE_FILE.unlink(missing_ok=True)
    print('Done.')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync-only', action='store_true')
    parser.add_argument('--run-only', action='store_true')
    parser.add_argument('--teardown', action='store_true')
    parser.add_argument('--ssh', action='store_true')
    parser.add_argument('--run-cmd', type=str, default=None,
                        help='Run arbitrary command on the gap-accel pod')
    args = parser.parse_args()

    if args.teardown:
        teardown()
        return

    if args.ssh:
        state = load_state()
        if state:
            key = str(SSH_KEY_PATH)
            print(f'ssh -p {state["ssh_port"]} -i {key} '
                  f'-o StrictHostKeyChecking=no root@{state["ssh_host"]}')
        return

    state = load_state()

    if args.sync_only:
        if not state:
            print('No gap-accel pod state. Run without --sync-only first.')
            return
        sync_files(state['ssh_host'], state['ssh_port'])
        return

    if args.run_only:
        if not state:
            print('No gap-accel pod state. Run without --run-only first.')
            return
        if args.run_cmd:
            cmd = f'cd {REMOTE_WORKDIR} && {args.run_cmd} 2>&1'
            r = ssh(state['ssh_host'], state['ssh_port'], cmd, t=600)
            print(r.stdout)
            if r.stderr:
                print(f'STDERR: {r.stderr[-500:]}')
        else:
            run_tests(state['ssh_host'], state['ssh_port'])
        return

    # Full deploy — ALWAYS create a NEW pod on fresh run.
    if state:
        print(f'Existing gap-accel pod state found: {state["pod_id"]}')
        try:
            r = ssh(state['ssh_host'], state['ssh_port'], 'echo ok', t=10)
            if r.returncode == 0 and 'ok' in r.stdout:
                print('  gap-accel pod is alive, reusing (use --teardown to '
                      'force a fresh one).')
            else:
                print('  gap-accel pod unreachable, creating new...')
                state = create_pod()
        except Exception:
            print('  gap-accel pod unreachable, creating new...')
            state = create_pod()
    else:
        state = create_pod()

    setup_deps(state['ssh_host'], state['ssh_port'])
    sync_files(state['ssh_host'], state['ssh_port'])
    run_tests(state['ssh_host'], state['ssh_port'])


if __name__ == '__main__':
    main()
