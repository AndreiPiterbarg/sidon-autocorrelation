#!/usr/bin/env python
"""Deploy a FRESH cpupod and run estimate_d16_l3_mosek --impl z2block.

State is tracked in `data/z2block_d16_pod_state.json` (separate from all
other pod sessions).  The pod is NAMED `sidon-z2block-d16-<timestamp>` so
every run gets a distinguishable name and a fresh pod ID.

USAGE
-----
    python deploy_z2block_d16_pod.py                   # full deploy + run
    python deploy_z2block_d16_pod.py --status          # pod + job state
    python deploy_z2block_d16_pod.py --tail            # last 120 log lines
    python deploy_z2block_d16_pod.py --report          # one 2-min report
    python deploy_z2block_d16_pod.py --fetch           # pull artefacts
    python deploy_z2block_d16_pod.py --teardown        # destroy pod
    python deploy_z2block_d16_pod.py --ssh             # print ssh command
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpupod.config import (RUNPOD_API_KEY, SSH_OPTIONS, SSH_KEY_PATH,
                           REMOTE_WORKDIR, INSTANCE_ID, TEMPLATE_ID,
                           CLOUD_TYPE, CONTAINER_DISK_GB)

HERE = Path(__file__).resolve().parent
STATE_FILE = HERE / 'data' / 'z2block_d16_pod_state.json'
MOSEK_LIC_LOCAL = Path.home() / 'mosek' / 'mosek.lic'
POD_NAME_PREFIX = 'sidon-z2block-d16'

FILES_TO_SYNC = [
    # The new block-diag solver + the original tuned file (estimate
    # imports from both; solve_mosek_tuned gets rebound based on --impl).
    'tests/lasserre_mosek_z2block.py',
    'tests/lasserre_mosek_tuned.py',
    'tests/lasserre_fusion.py',
    'tests/lasserre_scalable.py',
    'tests/estimate_d16_l3_mosek.py',
    'lasserre/__init__.py',
    'lasserre/core.py',
    'lasserre/precompute.py',
    'lasserre/cliques.py',
    'lasserre/z2_symmetry.py',
    'lasserre/z2_blockdiag.py',
    'lasserre/z2_elim.py',
]

DONE_MARKER = f'{REMOTE_WORKDIR}/data/z2block_d16_estimate_done'
LOG_PATH = f'{REMOTE_WORKDIR}/data/z2block_d16_estimate.log'
OUT_JSON = f'{REMOTE_WORKDIR}/data/z2block_d16_estimate.json'


# =====================================================================
# SSH helpers
# =====================================================================

def ssh(host, port, cmd, t=600):
    return subprocess.run(
        ['ssh', '-p', str(port)] + SSH_OPTIONS + [f'root@{host}', cmd],
        capture_output=True, text=True, timeout=t,
        stdin=subprocess.DEVNULL)


def scp_to(host, port, local, remote):
    return subprocess.run(
        ['scp', '-P', str(port)] + SSH_OPTIONS +
        [local, f'root@{host}:{remote}'],
        capture_output=True, text=True, timeout=180)


def scp_from(host, port, remote, local):
    return subprocess.run(
        ['scp', '-P', str(port)] + SSH_OPTIONS +
        [f'root@{host}:{remote}', local],
        capture_output=True, text=True, timeout=180)


# =====================================================================
# State
# =====================================================================

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return None


def save_state(s):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=2))


# =====================================================================
# Pod lifecycle — always FRESH (new pod ID, new name)
# =====================================================================

def create_fresh_pod():
    """Create a brand-new pod and record its id in our dedicated state
    file.  Never reuses any prior session.
    """
    import runpod
    runpod.api_key = RUNPOD_API_KEY

    pub_key_path = SSH_KEY_PATH.with_suffix('.pub')
    if not pub_key_path.exists():
        raise RuntimeError(f'Missing ssh public key at {pub_key_path}')
    pub_key = pub_key_path.read_text().strip()

    name = f'{POD_NAME_PREFIX}-{int(time.time())}'
    print(f"Creating NEW cpu pod '{name}' ({INSTANCE_ID}, "
          f"{CLOUD_TYPE}, disk={CONTAINER_DISK_GB}GB)...",
          flush=True)

    pod = runpod.create_pod(
        name=name,
        instance_id=INSTANCE_ID,
        template_id=TEMPLATE_ID,
        cloud_type=CLOUD_TYPE,
        container_disk_in_gb=CONTAINER_DISK_GB,
        env={'PUBLIC_KEY': pub_key},
        ports='22/tcp',
    )
    if not pod or 'id' not in pod:
        raise RuntimeError(f'create_pod failed: {pod}')
    pod_id = pod['id']
    print(f'  pod_id={pod_id}', flush=True)

    host = port = None
    for attempt in range(180):  # up to 15 min
        time.sleep(5)
        st = runpod.get_pod(pod_id)
        rt = st.get('runtime') if st else None
        if rt and rt.get('ports'):
            for p in rt['ports']:
                if p.get('privatePort') == 22 and p.get('ip'):
                    host, port = p['ip'], p['publicPort']
                    break
            if host:
                break
        if attempt % 6 == 5:
            print(f'  waiting for ssh... {5 * attempt}s', flush=True)
    if not host:
        raise RuntimeError(f'pod {pod_id} ssh never came up')

    state = {'pod_id': pod_id, 'name': name,
             'ssh_host': host, 'ssh_port': port,
             'created_at': time.time()}
    save_state(state)
    print(f'  ssh ready: root@{host} -p {port}', flush=True)
    return state


def wait_ssh(host, port, timeout=240):
    start = time.time()
    while time.time() - start < timeout:
        r = ssh(host, port, 'echo ok', t=15)
        if r.returncode == 0 and 'ok' in r.stdout:
            return True
        time.sleep(5)
    return False


def install_mosek(host, port):
    print('Installing MOSEK + deps...', flush=True)
    r = ssh(host, port,
             'pip install --quiet mosek numpy scipy psutil 2>&1 | tail -6',
             t=900)
    print(f'  pip: {(r.stdout or r.stderr).strip()[-600:]}', flush=True)

    if not MOSEK_LIC_LOCAL.exists():
        raise RuntimeError(f'No MOSEK license at {MOSEK_LIC_LOCAL}')
    ssh(host, port, 'mkdir -p /root/mosek', t=30)
    r = scp_to(host, port, str(MOSEK_LIC_LOCAL), '/root/mosek/mosek.lic')
    print(f'  license: {"ok" if r.returncode == 0 else r.stderr.strip()}',
          flush=True)

    r = ssh(host, port,
             'python -c "import mosek; print(\'mosek\', '
             'mosek.Env.getversion())" 2>&1', t=60)
    print(f'  mosek: {r.stdout.strip()}', flush=True)
    r = ssh(host, port, 'nproc; free -g | head -2', t=30)
    print(f'  resources:\n{r.stdout}', flush=True)


def sync_files(host, port):
    print('Syncing code to pod...', flush=True)
    ssh(host, port,
         f'mkdir -p {REMOTE_WORKDIR}/tests {REMOTE_WORKDIR}/lasserre '
         f'{REMOTE_WORKDIR}/data',
         t=30)
    for rel in FILES_TO_SYNC:
        src = HERE / rel
        if not src.exists():
            print(f'  MISSING: {rel}', flush=True)
            continue
        r = scp_to(host, port, str(src), f'{REMOTE_WORKDIR}/{rel}')
        tag = 'ok' if r.returncode == 0 else f'FAIL: {r.stderr.strip()}'
        print(f'  {rel} -> {tag}', flush=True)


def run_estimate(host, port):
    """Kick off estimate_d16_l3_mosek.py --impl z2block in detached mode.

    The estimate script runs d=4, 6, 8, 10 L3 calibration and extrapolates
    to d=16 size.  Total wall ≈ 1-2 h on 32 vCPU.
    """
    print('\n=== Launch: estimate_d16_l3_mosek.py --impl z2block ===',
          flush=True)

    runner = f'{REMOTE_WORKDIR}/run_z2block_d16.sh'
    script = (
        "#!/bin/bash\n"
        "set -u\n"
        f"cd {REMOTE_WORKDIR}\n"
        f"export PYTHONPATH={REMOTE_WORKDIR}:{REMOTE_WORKDIR}/tests\n"
        "export PYTHONUNBUFFERED=1\n"
        "echo \"=== START $(date -u +%FT%TZ) ===\"\n"
        "python -u tests/lasserre_mosek_z2block.py --self-test "
        f" 2>&1 | tee {LOG_PATH}.selftest\n"
        f"python -u tests/estimate_d16_l3_mosek.py "
        f"  --impl z2block --mode z2block "
        f"  --primary-tol 1e-6 --order-method forceGraphpar "
        f"  --include-d8 --include-d10 "
        f"  --calib-bisect-small 3 --calib-bisect-d8 2 "
        f"  --calib-bisect-d10 1 "
        f"  --out {OUT_JSON} "
        f"  2>&1 || echo 'estimate FAILED: '$?\n"
        "echo \"=== END $(date -u +%FT%TZ) ===\"\n"
        f"touch {DONE_MARKER}\n"
    )
    tmp = HERE / 'data' / '_z2block_d16_runner.sh'
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(script.encode('utf-8').replace(b'\r\n', b'\n'))
    r = scp_to(host, port, str(tmp), runner)
    if r.returncode != 0:
        raise RuntimeError(f'scp runner failed: {r.stderr.strip()}')

    launch = (
        f'chmod +x {runner} && '
        f'( nohup {runner} > {LOG_PATH} 2>&1 < /dev/null & '
        f'  disown ) && '
        f'echo LAUNCHED'
    )
    r = ssh(host, port, launch, t=30)
    print(f'  launched: {r.stdout.strip()}', flush=True)
    print(f'  log:  {LOG_PATH}', flush=True)
    print(f'  done: {DONE_MARKER}', flush=True)


# =====================================================================
# Status / monitor / tail
# =====================================================================

_ITER_RE = re.compile(
    r'^\[MOSEK\]\s+(\d+)\s+'
    r'([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+'
    r'(\S+)\s+'
    r'([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+'
    r'([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$'
)


def _compact_report(host, port):
    """Fetch current status as a single dict for 2-min reports."""
    cmd = (
        f'if [ -f {DONE_MARKER} ]; then echo STATE=DONE; '
        f'else echo STATE=RUNNING; fi; '
        f'echo TS=$(date -u +%FT%TZ); '
        f'echo PID=$(pgrep -f estimate_d16_l3_mosek | head -1); '
        f'if [ -f {LOG_PATH} ]; then '
        f'  echo LOG_SIZE=$(stat -c %s {LOG_PATH}); '
        f'  echo LAST="$(tail -n 1 {LOG_PATH})"; '
        f'fi; '
        f'if [ -f {OUT_JSON} ]; then echo OUT_JSON=yes; '
        f'else echo OUT_JSON=no; fi; '
        f'echo LAST_IPM="$(tac {LOG_PATH} 2>/dev/null | '
        f'  grep -E "^\\[MOSEK\\]\\s+[0-9]+\\s" | head -1)"; '
        f'echo LAST_CALIB="$(grep -E "^\\[calib\\] d=" {LOG_PATH} '
        f'  2>/dev/null | tail -1)"; '
        f'echo RSS_MB=$(ps -C python -o rss= 2>/dev/null | '
        f'  awk \'{{s+=$1}}END{{print int(s/1024)}}\')'
    )
    r = ssh(host, port, cmd, t=30)
    if r.returncode != 0:
        return {'ERROR': r.stderr.strip()}
    parsed = {}
    for line in r.stdout.splitlines():
        if '=' in line:
            k, _, v = line.partition('=')
            parsed[k.strip()] = v.strip().strip('"')
    return parsed


def _parse_ipm(line):
    m = _ITER_RE.match(line)
    if not m:
        return None
    return {
        'iter': int(m.group(1)),
        'pfeas': float(m.group(2)),
        'dfeas': float(m.group(3)),
        'gfeas': float(m.group(4)),
        'prstatus': m.group(5),
        'mu': float(m.group(8)),
        'mosek_time': float(m.group(9)),
    }


def print_report(host, port):
    p = _compact_report(host, port)
    state = p.get('STATE', '?')
    ts = p.get('TS', '?')
    size = p.get('LOG_SIZE', '?')
    rss = p.get('RSS_MB', '?')
    calib = p.get('LAST_CALIB', '')
    last_ipm = p.get('LAST_IPM', '')
    ipm = _parse_ipm(last_ipm) if last_ipm else None
    if ipm:
        ipm_str = (f"iter={ipm['iter']} mu={ipm['mu']:.2e} "
                    f"prstatus={ipm['prstatus']} "
                    f"mtime={ipm['mosek_time']:.1f}s")
    else:
        ipm_str = 'no MOSEK IPM line yet'
    print(f"[{ts}] state={state} log={size}B rss={rss}MB  "
          f"calib={calib or '-'}  |  {ipm_str}", flush=True)


def tail_log(host, port, n=120):
    r = ssh(host, port, f'tail -n {n} {LOG_PATH} 2>&1', t=30)
    print(r.stdout)


def teardown():
    state = load_state()
    if not state:
        print('No z2block-d16 pod state.', flush=True)
        return
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    pod_id = state['pod_id']
    print(f'Terminating pod {pod_id} ({state.get("name")})...',
          flush=True)
    try:
        runpod.terminate_pod(pod_id)
    except Exception as exc:
        print(f'  terminate error (maybe already gone): {exc}')
    STATE_FILE.unlink(missing_ok=True)
    print('Teardown complete.', flush=True)


def fetch_results(host, port):
    print('Fetching z2block d=16 artefacts...', flush=True)
    local = HERE / 'data'
    for rel in ('z2block_d16_estimate.log',
                'z2block_d16_estimate.log.selftest',
                'z2block_d16_estimate.json',
                'z2block_d16_estimate_done'):
        r = scp_from(host, port,
                      f'{REMOTE_WORKDIR}/data/{rel}',
                      str(local / rel))
        print(f'  {rel}: '
              f'{"ok" if r.returncode == 0 else r.stderr.strip()}',
              flush=True)


# =====================================================================
# CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--tail', action='store_true')
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--teardown', action='store_true')
    ap.add_argument('--ssh', action='store_true')
    ap.add_argument('--sync-only', action='store_true')
    ap.add_argument('--run-only', action='store_true')
    args = ap.parse_args()

    if args.teardown:
        teardown(); return

    state = load_state()
    if args.ssh:
        if state:
            print(f'ssh -p {state["ssh_port"]} -i {SSH_KEY_PATH} '
                  f'-o StrictHostKeyChecking=no root@{state["ssh_host"]}')
        return
    if args.tail:
        if not state:
            print('No pod state.'); return
        tail_log(state['ssh_host'], state['ssh_port']); return
    if args.status or args.report:
        if not state:
            print('NO_STATE'); return
        print_report(state['ssh_host'], state['ssh_port']); return
    if args.fetch:
        if not state:
            print('No pod state.'); return
        fetch_results(state['ssh_host'], state['ssh_port']); return

    # Full deploy: always create FRESH (per user requirement).
    if state:
        print(f'Existing z2block-d16 pod state found: {state["pod_id"]}',
              flush=True)
        print('  refusing to reuse — tear down first with --teardown, '
              'then rerun.', flush=True)
        sys.exit(2)

    state = create_fresh_pod()
    if not wait_ssh(state['ssh_host'], state['ssh_port']):
        raise RuntimeError('ssh never came up')

    if args.sync_only:
        sync_files(state['ssh_host'], state['ssh_port']); return
    if args.run_only:
        run_estimate(state['ssh_host'], state['ssh_port']); return

    install_mosek(state['ssh_host'], state['ssh_port'])
    sync_files(state['ssh_host'], state['ssh_port'])
    run_estimate(state['ssh_host'], state['ssh_port'])
    print('\nPod deployed + estimate launched.  Poll with:')
    print('  python deploy_z2block_d16_pod.py --report     # one line')
    print('  python deploy_z2block_d16_pod.py --tail       # last 120 lines')


if __name__ == '__main__':
    main()
