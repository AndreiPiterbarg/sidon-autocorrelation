#!/usr/bin/env python
"""Deploy 2x H100 SXM pod (~500 GB RAM) and run the Clarabel-based
d=16 L3 feasibility probe at t=1.2802.

Clarabel (pure Rust + Python bindings, pip install clarabel) is the
final production fallback after MOSEK hung indefinitely at d=16 L3
symbolic factorisation and after MATLAB+SDPNAL+ was blocked by the
Penn licence activation policy.

USAGE
    python deploy_clarabel_d16.py             # full deploy+launch
    python deploy_clarabel_d16.py --status
    python deploy_clarabel_d16.py --log [N]
    python deploy_clarabel_d16.py --fetch
    python deploy_clarabel_d16.py --teardown
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpupod.config import (RUNPOD_API_KEY, SSH_OPTIONS, SSH_KEY_PATH,
                            REMOTE_WORKDIR, DOCKER_IMAGE)

STATE_FILE = (Path(__file__).resolve().parent / 'data'
              / 'clarabel_d16_pod_state.json')
GPU_TYPE_ID = 'NVIDIA H100 80GB HBM3'
GPU_COUNT = 2
POD_NAME = 'sidon-clarabel-d16'

RUNNER_REMOTE = f'{REMOTE_WORKDIR}/run_clarabel_d16.sh'
LOG_REMOTE = f'{REMOTE_WORKDIR}/data/clarabel_d16.log'
DONE_MARKER = f'{REMOTE_WORKDIR}/data/clarabel_d16_done'
RESULT_DIR = f'{REMOTE_WORKDIR}/data/clarabel_d16_production'

# Clarabel driver imports from these files
FILES_TO_SYNC = [
    'tests/lasserre_clarabel_driver.py',
    'tests/lasserre_clarabel.py',
    'tests/lasserre_highd.py',
    'tests/run_d16_l3.py',
    'tests/lasserre_mosek_tuned.py',
    'tests/lasserre_fusion.py',
    'tests/lasserre_scalable.py',
    'tests/lasserre_enhanced.py',
    'lasserre/__init__.py',
    'lasserre/core.py',
    'lasserre/precompute.py',
    'lasserre/cliques.py',
    'lasserre/z2_symmetry.py',
    'lasserre/z2_blockdiag.py',
    'lasserre/z2_elim.py',
    'lasserre/gap_accelerator.py',
    'lasserre/dual_sdp.py',
    'lasserre/cheby_basis.py',
    'lasserre/preelim.py',
    'lasserre/solvers.py',
]


def ssh(host, port, cmd, t=60):
    return subprocess.run(
        ['ssh', '-p', str(port)] + SSH_OPTIONS + [f'root@{host}', cmd],
        capture_output=True, text=True, timeout=t,
        stdin=subprocess.DEVNULL)


def scp_to(host, port, local, remote):
    return subprocess.run(
        ['scp', '-P', str(port)] + SSH_OPTIONS +
        [local, f'root@{host}:{remote}'],
        capture_output=True, text=True, timeout=300)


def scp_from(host, port, remote, local, recursive=False):
    args = ['scp', '-P', str(port)]
    if recursive:
        args.append('-r')
    args += SSH_OPTIONS + [f'root@{host}:{remote}', local]
    return subprocess.run(args, capture_output=True, text=True,
                           timeout=1800)


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
    pub_key = pub_path.read_text().strip()
    print(f'Creating {GPU_COUNT}x {GPU_TYPE_ID} pod ...')
    pod = runpod.create_pod(
        name=POD_NAME, image_name=DOCKER_IMAGE,
        gpu_type_id=GPU_TYPE_ID, gpu_count=GPU_COUNT,
        cloud_type='ALL', container_disk_in_gb=60, volume_in_gb=0,
        env={'PUBLIC_KEY': pub_key}, ports='22/tcp')
    pod_id = pod['id']
    print(f'  Pod ID: {pod_id}')
    host = port = None
    for attempt in range(120):
        time.sleep(5)
        st = runpod.get_pod(pod_id)
        rt = (st or {}).get('runtime')
        if rt and rt.get('ports'):
            for p in rt['ports']:
                if p.get('privatePort') == 22 and p.get('ip'):
                    host, port = p['ip'], p['publicPort']
                    break
            if host:
                break
        if attempt % 6 == 5:
            print(f'  waiting ({attempt * 5}s)')
    if not host:
        raise RuntimeError('SSH never came up')
    state = {'pod_id': pod_id, 'ssh_host': host, 'ssh_port': port,
             'created_at': time.time()}
    save_state(state)
    print(f'  SSH: root@{host} -p {port}')
    return state


def wait_for_ssh(host, port, timeout=240):
    start = time.time()
    while time.time() - start < timeout:
        r = ssh(host, port, 'echo ok', t=15)
        if r.returncode == 0 and 'ok' in r.stdout:
            return True
        time.sleep(5)
    return False


def install_deps(host, port):
    print('Installing clarabel + numpy + scipy + psutil + py-spy...')
    r = ssh(host, port,
             'pip install --quiet clarabel numpy scipy psutil py-spy '
             '2>&1 | tail -3', t=600)
    print(f'  pip: {(r.stdout or r.stderr).strip()[-400:]}')
    r = ssh(host, port,
             'python -c "import clarabel; print(\\"clarabel\\", '
             'clarabel.__version__)"', t=60)
    print(f'  clarabel: {r.stdout.strip()}')


def verify_container_limits(host, port):
    r = ssh(host, port,
             'echo CGROUP_MEM=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); '
             'echo NPROC=$(nproc); free -g | head -3', t=30)
    print(r.stdout)
    cgroup_gb = None
    for line in r.stdout.splitlines():
        if line.startswith('CGROUP_MEM='):
            try:
                cgroup_gb = int(line.split('=', 1)[1]) / 1e9
            except ValueError:
                pass
    if cgroup_gb is None or cgroup_gb < 400:
        print(f'!! WARNING cgroup={cgroup_gb} GB < 400 GB threshold')


def sync_files(host, port):
    print('Syncing code...')
    ssh(host, port,
         f'mkdir -p {REMOTE_WORKDIR}/tests {REMOTE_WORKDIR}/lasserre '
         f'{REMOTE_WORKDIR}/data', t=30)
    root = Path(__file__).resolve().parent
    missing = []
    for f in FILES_TO_SYNC:
        local = root / f
        if not local.exists():
            missing.append(f)
            print(f'  MISSING: {f}')
            continue
        r = scp_to(host, port, str(local), f'{REMOTE_WORKDIR}/{f}')
        print(f'  {f}: '
              f'{"ok" if r.returncode == 0 else "FAIL " + r.stderr.strip()[:200]}')
    if missing:
        raise RuntimeError(f'Missing files: {missing}')


def launch_clarabel(host, port, target=1.2802, d=16, order=3,
                     bandwidth=15):
    print(f'Launching Clarabel d={d} L{order} bw={bandwidth} '
          f'target={target} ...')
    runner = f"""#!/bin/bash
set -u
cd {REMOTE_WORKDIR}
export PYTHONPATH={REMOTE_WORKDIR}:{REMOTE_WORKDIR}/tests
export PYTHONUNBUFFERED=1
mkdir -p {RESULT_DIR}
echo "=== START $(date -u +%FT%TZ) ==="
echo "=== Clarabel d={d} L{order} bw={bandwidth} target={target} SCOUT (loose) ==="
python -u tests/lasserre_clarabel_driver.py \\
    --d {d} --order {order} --bw {bandwidth} \\
    --target {target} \\
    --tol-gap-abs 1e-3 --tol-gap-rel 1e-3 \\
    --tol-feas 1e-3 \\
    --tol-infeas-abs 1e-4 --tol-infeas-rel 1e-4 \\
    --max-iter 500 \\
    --time-limit-s 3600 \\
    --direct-solve-method qdldl \\
    --data-dir {RESULT_DIR} \\
    2>&1 || echo "Clarabel FAILED: $?"
echo "=== END $(date -u +%FT%TZ) ==="
touch {DONE_MARKER}
"""
    tmp = Path(__file__).resolve().parent / 'data' / '_clarabel_runner.sh'
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(runner.encode('utf-8').replace(b'\r\n', b'\n'))
    r = scp_to(host, port, str(tmp), RUNNER_REMOTE)
    if r.returncode != 0:
        raise RuntimeError(f'scp runner failed: {r.stderr.strip()}')
    r = ssh(host, port,
             f'rm -f {DONE_MARKER} {LOG_REMOTE}; '
             f'chmod +x {RUNNER_REMOTE} && '
             f'( nohup {RUNNER_REMOTE} > {LOG_REMOTE} 2>&1 < /dev/null & '
             f'  disown ) && echo LAUNCHED', t=30)
    print(f'  {r.stdout.strip()}')
    print(f'  log: {LOG_REMOTE}')
    print(f'  done: {DONE_MARKER}')
    print(f'  results: {RESULT_DIR}')


def status():
    state = load_state()
    if not state:
        return 'NO_STATE'
    host, port = state['ssh_host'], state['ssh_port']
    cmd = (
        f'if [ -f {DONE_MARKER} ]; then echo STATE=DONE; '
        f'else echo STATE=RUNNING; fi; '
        f'echo "MEM_MB=$(ps -o rss= -p $(pgrep -f lasserre_clarabel_driver '
        f'| head -1) 2>/dev/null | awk \'{{print int($1/1024)}}\')"; '
        f'echo "LAST=$(tail -n 1 {LOG_REMOTE} 2>/dev/null)"')
    r = ssh(host, port, cmd, t=30)
    if r.returncode != 0:
        return f'UNREACHABLE rc={r.returncode}'
    p = {}
    for line in r.stdout.strip().splitlines():
        if '=' in line:
            k, _, v = line.partition('=')
            p[k.strip()] = v.strip()
    return (f'{p.get("STATE", "?")} mem={p.get("MEM_MB", "?")}MB '
            f'last="{p.get("LAST", "")[-120:]}"')


def tail_log(n=200):
    state = load_state()
    host, port = state['ssh_host'], state['ssh_port']
    r = ssh(host, port, f'tail -n {n} {LOG_REMOTE}', t=30)
    print(r.stdout)


def fetch():
    state = load_state()
    host, port = state['ssh_host'], state['ssh_port']
    local = Path(__file__).resolve().parent / 'data' / 'clarabel_d16_results'
    local.mkdir(parents=True, exist_ok=True)
    for remote_name, local_name in (
        (LOG_REMOTE, 'clarabel_d16.log'),
    ):
        r = scp_from(host, port, remote_name, str(local / local_name))
        print(f'  {local_name}: '
              f'{"ok" if r.returncode == 0 else r.stderr.strip()[:200]}')
    r = scp_from(host, port, f'{RESULT_DIR}/.', str(local), recursive=True)
    print(f'  results/: '
          f'{"ok" if r.returncode == 0 else r.stderr.strip()[:200]}')


def teardown():
    state = load_state()
    if not state:
        print('No state.')
        return
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    pod_id = state['pod_id']
    print(f'Terminating {pod_id} ...')
    try:
        runpod.terminate_pod(pod_id)
        print('  terminated.')
    except Exception as e:
        print(f'  error: {e}')
    STATE_FILE.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--log', nargs='?', const=200, type=int)
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--teardown', action='store_true')
    ap.add_argument('--ssh', action='store_true')
    ap.add_argument('--target', type=float, default=1.2802)
    ap.add_argument('--d', type=int, default=16)
    ap.add_argument('--order', type=int, default=3)
    ap.add_argument('--bw', type=int, default=15)
    args = ap.parse_args()

    if args.ssh:
        state = load_state()
        if state:
            print(f'ssh -p {state["ssh_port"]} -i {SSH_KEY_PATH} '
                  f'root@{state["ssh_host"]}')
        return
    if args.status:
        print(status()); return
    if args.log is not None:
        tail_log(args.log); return
    if args.fetch:
        fetch(); return
    if args.teardown:
        teardown(); return

    if load_state():
        print(f'Existing state at {STATE_FILE}; --teardown first.')
        sys.exit(1)

    state = create_pod()
    host, port = state['ssh_host'], state['ssh_port']
    if not wait_for_ssh(host, port):
        raise RuntimeError('SSH not ready')
    verify_container_limits(host, port)
    install_deps(host, port)
    sync_files(host, port)
    launch_clarabel(host, port, target=args.target, d=args.d,
                     order=args.order, bandwidth=args.bw)
    print(f'\n=== LAUNCHED === pod={state["pod_id"]}')


if __name__ == '__main__':
    main()
