#!/usr/bin/env python
"""Deploy CPU pod and run coarse cascade v2 at S=10,000, c_target=1.285.

USAGE:
    python deploy_cascade_S10k.py              # full deploy+launch
    python deploy_cascade_S10k.py --status     # progress
    python deploy_cascade_S10k.py --log [N]    # tail remote log
    python deploy_cascade_S10k.py --fetch      # copy log home
    python deploy_cascade_S10k.py --teardown   # destroy pod
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpupod.config import (
    RUNPOD_API_KEY, SSH_OPTIONS, SSH_KEY_PATH, INSTANCE_ID,
    TEMPLATE_ID, CLOUD_TYPE, CONTAINER_DISK_GB, REMOTE_WORKDIR,
)

STATE = Path(__file__).resolve().parent / 'data' / 'cascade_S10k_state.json'
POD_NAME = 'sidon-cascade-S10k'
LOG = f'{REMOTE_WORKDIR}/data/cascade_S10k.log'
DONE = f'{REMOTE_WORKDIR}/data/cascade_S10k_done'
RESULT_DIR = f'{REMOTE_WORKDIR}/data/cascade_S10k_out'

SYNC = [
    'cloninger-steinerberger/cpu/run_cascade_coarse_v2.py',
    'cloninger-steinerberger/cpu/run_cascade_coarse.py',
]


def ssh(host, port, cmd, t=60):
    return subprocess.run(
        ['ssh', '-p', str(port)] + SSH_OPTIONS + [f'root@{host}', cmd],
        capture_output=True, text=True, timeout=t, stdin=subprocess.DEVNULL)


def scp_to(host, port, local, remote):
    return subprocess.run(
        ['scp', '-P', str(port)] + SSH_OPTIONS +
        [local, f'root@{host}:{remote}'],
        capture_output=True, text=True, timeout=180)


def scp_from(host, port, remote, local):
    return subprocess.run(
        ['scp', '-P', str(port)] + SSH_OPTIONS +
        [f'root@{host}:{remote}', local],
        capture_output=True, text=True, timeout=300)


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else None


def save(s):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def create():
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    pub_key = SSH_KEY_PATH.with_suffix('.pub').read_text().strip()
    print(f'Creating CPU pod {INSTANCE_ID} ...')
    pod = runpod.create_pod(
        name=POD_NAME, instance_id=INSTANCE_ID, template_id=TEMPLATE_ID,
        cloud_type=CLOUD_TYPE, container_disk_in_gb=CONTAINER_DISK_GB,
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
            if host: break
        if attempt % 6 == 5: print(f'  waiting ({attempt * 5}s)')
    if not host:
        raise RuntimeError('SSH never came up')
    s = {'pod_id': pod_id, 'ssh_host': host, 'ssh_port': port,
         'created_at': time.time()}
    save(s)
    print(f'  SSH: root@{host} -p {port}')
    return s


def wait_ssh(host, port, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = ssh(host, port, 'echo ok', t=15)
        if r.returncode == 0 and 'ok' in r.stdout: return True
        time.sleep(5)
    return False


def install(host, port):
    print('Installing deps ...')
    r = ssh(host, port,
        'python3.13 -m pip install -q numpy scipy numba psutil 2>&1 | tail -3',
        t=600)
    print(f'  pip: {(r.stdout or r.stderr).strip()[-300:]}')
    r = ssh(host, port,
        'python3.13 -c "import numpy, numba; print(numpy.__version__, numba.__version__)"',
        t=60)
    print(f'  versions: {r.stdout.strip()}')
    r = ssh(host, port, 'nproc; free -g | head -2', t=15)
    print(r.stdout)


def sync(host, port):
    print('Syncing cascade code ...')
    ssh(host, port,
        f'mkdir -p {REMOTE_WORKDIR}/cloninger-steinerberger/cpu '
        f'{REMOTE_WORKDIR}/data', t=30)
    root = Path(__file__).resolve().parent
    for f in SYNC:
        local = root / f
        if not local.exists():
            print(f'  MISSING {f}'); continue
        r = scp_to(host, port, str(local), f'{REMOTE_WORKDIR}/{f}')
        print(f'  {f}: '
              f'{"ok" if r.returncode == 0 else "FAIL " + r.stderr.strip()[:200]}')


def launch(host, port, c_target=1.285, S=10000, d0=2, max_levels=10):
    runner = f"""#!/bin/bash
set -u
cd {REMOTE_WORKDIR}
export PYTHONPATH={REMOTE_WORKDIR}:{REMOTE_WORKDIR}/cloninger-steinerberger/cpu
export PYTHONUNBUFFERED=1
mkdir -p {RESULT_DIR}
echo "=== START $(date -u +%FT%TZ) ==="
echo "=== coarse v2 d0={d0} S={S} c_target={c_target} max_levels={max_levels} ==="
python3.13 -u cloninger-steinerberger/cpu/run_cascade_coarse_v2.py \\
    --d0 {d0} --S {S} --c_target {c_target} --max_levels {max_levels} \\
    2>&1 || echo "cascade FAILED: $?"
echo "=== END $(date -u +%FT%TZ) ==="
touch {DONE}
"""
    tmp = Path(__file__).resolve().parent / 'data' / '_cascade_S10k_runner.sh'
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(runner.encode('utf-8').replace(b'\r\n', b'\n'))
    r = scp_to(host, port, str(tmp),
               f'{REMOTE_WORKDIR}/run_cascade_S10k.sh')
    if r.returncode != 0:
        raise RuntimeError('scp runner failed')
    cmd = (
        f'rm -f {DONE} {LOG}; '
        f'chmod +x {REMOTE_WORKDIR}/run_cascade_S10k.sh && '
        f'( nohup {REMOTE_WORKDIR}/run_cascade_S10k.sh > {LOG} 2>&1 < /dev/null & disown ) && echo LAUNCHED'
    )
    r = ssh(host, port, cmd, t=30)
    print(f'  {r.stdout.strip()}')


def status():
    s = load()
    if not s: return 'NO_STATE'
    host, port = s['ssh_host'], s['ssh_port']
    cmd = (
        f'if [ -f {DONE} ]; then echo STATE=DONE; else echo STATE=RUNNING; fi; '
        f'echo "MEM_MB=$(ps -o rss= -p $(pgrep -f run_cascade_coarse_v2 | head -1) 2>/dev/null | awk \'{{print int($1/1024)}}\')"; '
        f'echo "ETIME=$(ps -o etime= -p $(pgrep -f run_cascade_coarse_v2 | head -1) 2>/dev/null | tr -d \' \')"; '
        f'echo "LAST=$(tail -n 1 {LOG} 2>/dev/null)"')
    r = ssh(host, port, cmd, t=30)
    return r.stdout.strip() if r.returncode == 0 else f'UNREACHABLE rc={r.returncode}'


def tail_log(n=100):
    s = load()
    r = ssh(s['ssh_host'], s['ssh_port'], f'tail -n {n} {LOG}', t=30)
    print(r.stdout)


def fetch():
    s = load()
    local = Path(__file__).resolve().parent / 'data' / 'cascade_S10k_results'
    local.mkdir(parents=True, exist_ok=True)
    r = scp_from(s['ssh_host'], s['ssh_port'], LOG, str(local / 'cascade_S10k.log'))
    print(f'  log: {"ok" if r.returncode == 0 else r.stderr.strip()[:200]}')


def teardown():
    s = load()
    if not s:
        print('No state'); return
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    print(f'Terminating {s["pod_id"]} ...')
    try:
        runpod.terminate_pod(s['pod_id']); print('  terminated')
    except Exception as e:
        print(f'  err {e}')
    STATE.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--log', nargs='?', const=100, type=int)
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--teardown', action='store_true')
    ap.add_argument('--ssh', action='store_true')
    ap.add_argument('--S', type=int, default=10000)
    ap.add_argument('--c-target', type=float, default=1.285)
    ap.add_argument('--d0', type=int, default=2)
    ap.add_argument('--max-levels', type=int, default=10)
    args = ap.parse_args()

    if args.ssh:
        s = load()
        if s:
            print(f'ssh -p {s["ssh_port"]} -i {SSH_KEY_PATH} root@{s["ssh_host"]}')
        return
    if args.status: print(status()); return
    if args.log is not None: tail_log(args.log); return
    if args.fetch: fetch(); return
    if args.teardown: teardown(); return

    if load():
        print(f'State exists; --teardown first.'); sys.exit(1)
    s = create()
    host, port = s['ssh_host'], s['ssh_port']
    if not wait_ssh(host, port):
        raise RuntimeError('SSH not ready')
    install(host, port)
    sync(host, port)
    launch(host, port, c_target=args.c_target, S=args.S,
           d0=args.d0, max_levels=args.max_levels)
    print(f'\n=== LAUNCHED pod={s["pod_id"]} S={args.S} c_target={args.c_target} ===')
    print('\nProgress commands:')
    print('  python deploy_cascade_S10k.py --status')
    print('  python deploy_cascade_S10k.py --log 100')
    print('  python deploy_cascade_S10k.py --log 500')
    print('  python deploy_cascade_S10k.py --fetch')
    print('  python deploy_cascade_S10k.py --teardown')


if __name__ == '__main__':
    main()
