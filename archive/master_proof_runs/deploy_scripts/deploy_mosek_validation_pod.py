#!/usr/bin/env python
"""Deploy CPU-heavy pod for d=10 and d=12 L3 MOSEK validation runs.

This deploys a RunPod instance (H100 pod — MOSEK is CPU-only, we use
the box for its vCPUs + system RAM; the GPU is idle), installs MOSEK +
dependencies, syncs the tuned solver, and runs two sequential jobs:
    • d=10 L3 z2_full bisection (6 steps)
    • d=12 L3 z2_full bisection (4 steps)

At the end it writes a completion marker file that our local monitor
can poll.

USAGE
-----
    python deploy_mosek_validation_pod.py              # full deploy + run
    python deploy_mosek_validation_pod.py --log        # tail the run log
    python deploy_mosek_validation_pod.py --status     # check pid + log tail
    python deploy_mosek_validation_pod.py --fetch      # copy results back
    python deploy_mosek_validation_pod.py --teardown   # destroy pod
    python deploy_mosek_validation_pod.py --ssh        # print ssh command
"""
from __future__ import annotations

import argparse
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
              / 'mosek_validation_pod_state.json')
MOSEK_LIC_LOCAL = Path.home() / 'mosek' / 'mosek.lic'
GPU_TYPE_ID = 'NVIDIA H100 80GB HBM3'  # for its CPU/RAM, MOSEK CPU-only
POD_NAME = 'sidon-mosek-validation'

FILES_TO_SYNC = [
    'tests/lasserre_mosek_tuned.py',
    'tests/lasserre_fusion.py',
    'tests/lasserre_scalable.py',
    # `lasserre` package — __init__.py re-exports from every module,
    # so even though the tuned solver only uses z2_symmetry and
    # z2_blockdiag directly, the package import pulls them all.
    'lasserre/__init__.py',
    'lasserre/core.py',
    'lasserre/precompute.py',
    'lasserre/cliques.py',
    'lasserre/z2_symmetry.py',
    'lasserre/z2_blockdiag.py',
]

# Marker file the remote script writes when the whole batch is done.
DONE_MARKER = '/workspace/sidon-autocorrelation/data/mosek_validation_done'


# =====================================================================
# Pod primitives
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

    print(f'Creating 1x {GPU_TYPE_ID} pod for MOSEK validation...')
    pod = runpod.create_pod(
        name=POD_NAME,
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
    for attempt in range(120):  # up to ~600s
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
            print(f'  Waiting for SSH... ({attempt * 5}s)')
    if not host:
        raise RuntimeError(f'Pod {pod_id} SSH not ready after 600s')

    state = {'pod_id': pod_id, 'ssh_host': host, 'ssh_port': port,
              'created_at': time.time(), 'gpu_type': GPU_TYPE_ID}
    save_state(state)
    print(f'  SSH ready: root@{host} -p {port}')
    return state


def wait_for_ssh(host, port, timeout=240):
    start = time.time()
    while time.time() - start < timeout:
        r = ssh(host, port, 'echo ok', t=15)
        if r.returncode == 0 and 'ok' in r.stdout:
            return True
        time.sleep(5)
    return False


def install_mosek(host, port):
    print('Installing MOSEK + deps...')
    r = ssh(host, port,
             'pip install --quiet mosek numpy scipy 2>&1 | tail -4',
             t=600)
    out = (r.stdout or r.stderr).strip()
    print(f'  pip: {out[-500:]}')

    if not MOSEK_LIC_LOCAL.exists():
        raise RuntimeError(f'No MOSEK license at {MOSEK_LIC_LOCAL}')
    ssh(host, port, 'mkdir -p /root/mosek', t=30)
    r = scp_to(host, port, str(MOSEK_LIC_LOCAL),
                '/root/mosek/mosek.lic')
    print(f'  license: {"ok" if r.returncode == 0 else r.stderr.strip()}')

    r = ssh(host, port,
             'python -c "import mosek; print(\'mosek\', '
             'mosek.Env.getversion())" 2>&1', t=60)
    print(f'  mosek import: {r.stdout.strip()}')
    r = ssh(host, port, 'nproc; free -g | head -2', t=30)
    print(f'  resources:\n{r.stdout}')


def sync_files(host, port):
    print('Syncing code to pod...')
    ssh(host, port,
         f'mkdir -p {REMOTE_WORKDIR}/tests {REMOTE_WORKDIR}/lasserre '
         f'{REMOTE_WORKDIR}/data',
         t=30)
    root = Path(__file__).resolve().parent
    for f in FILES_TO_SYNC:
        local = root / f
        if not local.exists():
            print(f'  MISSING: {f}')
            continue
        remote = f'{REMOTE_WORKDIR}/{f}'
        r = scp_to(host, port, str(local), remote)
        print(f'  {f} -> '
              f'{"ok" if r.returncode == 0 else f"FAIL: {r.stderr.strip()}"}')


def run_validation(host, port, d_list=None, nbisect_list=None):
    """Kick off one or more d-values sequentially (background).

    Writes the runner script locally, scps it, then launches it — this
    avoids the SSH-heredoc-too-long problem on some sshd configurations.

    d_list       : list of int (default [10, 12])
    nbisect_list : list of int matching d_list (default [6, 4])
    """
    d_list = d_list or [10, 12]
    nbisect_list = nbisect_list or [6, 4]
    assert len(d_list) == len(nbisect_list)

    print(f"\n=== MOSEK validation: d={d_list} z2_full ===",
          flush=True)
    log_path = f'{REMOTE_WORKDIR}/data/mosek_validation.log'
    done_path = DONE_MARKER
    runner_path = f'{REMOTE_WORKDIR}/run_validation.sh'

    jobs = []
    for d, nb in zip(d_list, nbisect_list):
        jobs.append(
            f"echo '=== d={d} L3 z2_full ==='\n"
            f"python -u tests/lasserre_mosek_tuned.py --d {d} --order 3 "
            f"  --mode z2_full --n-bisect {nb} "
            f"  --json data/mosek_d{d}_l3_z2full.json "
            f"  || echo 'd={d} FAILED: '$?\n"
        )

    script = (
        "#!/bin/bash\n"
        "set -u\n"
        f"cd {REMOTE_WORKDIR}\n"
        f"export PYTHONPATH={REMOTE_WORKDIR}:{REMOTE_WORKDIR}/tests\n"
        "export PYTHONUNBUFFERED=1\n"
        "echo \"=== START $(date -u +%FT%TZ) ===\"\n"
        + "".join(jobs) +
        "echo \"=== END $(date -u +%FT%TZ) ===\"\n"
        f"touch {done_path}\n"
    )

    # Write locally and scp.
    tmp = (Path(__file__).resolve().parent / 'data'
            / '_run_validation_local.sh')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    # Use unix line endings for the remote bash.
    tmp.write_bytes(script.encode('utf-8').replace(b'\r\n', b'\n'))
    r = scp_to(host, port, str(tmp), runner_path)
    if r.returncode != 0:
        raise RuntimeError(f'scp runner failed: {r.stderr.strip()}')
    print(f'  runner uploaded: {runner_path}')

    # Full detach: stdin /dev/null, stdout/stderr redirected, disown.
    # Without `< /dev/null` SSH can keep the channel open until the
    # nohup'd child closes its file descriptors, causing hangs.
    launch_cmd = (
        f'chmod +x {runner_path} && '
        f'( nohup {runner_path} > {log_path} 2>&1 < /dev/null & '
        f'  disown ) && '
        f'echo LAUNCHED'
    )
    r = ssh(host, port, launch_cmd, t=30)
    print(f'  launched: {r.stdout.strip()}')
    if r.stderr:
        print(f'  stderr: {r.stderr.strip()}')
    print(f'  log:  {log_path}')
    print(f'  done: {done_path}')


def tail_log(host, port, n=120):
    log_path = f'{REMOTE_WORKDIR}/data/mosek_validation.log'
    r = ssh(host, port,
             f'ls -la {log_path} 2>&1 | head; echo "----"; '
             f'tail -n {n} {log_path} 2>&1; echo "----"; '
             f'pgrep -af lasserre_mosek_tuned 2>&1 | head',
             t=30)
    print(r.stdout)


def status(host, port):
    """Compact, machine-friendly status line for the 15-min monitor.

    Emits one of:
       RUNNING pid=<N> log_tail=<last line>
       DONE    d10=ok|fail  d12=ok|fail
       UNREACHABLE <reason>
    """
    log_path = f'{REMOTE_WORKDIR}/data/mosek_validation.log'
    done_path = DONE_MARKER
    # Bundle checks into one SSH so the monitor stays cheap.
    cmd = (
        f'if [ -f {done_path} ]; then echo "STATE=DONE"; '
        f'else echo "STATE=RUNNING"; fi; '
        f'echo "PID=$(pgrep -f lasserre_mosek_tuned | head -1)"; '
        f'echo "LAST=$(tail -n 1 {log_path} 2>/dev/null)"; '
        f'echo "D10_JSON=$(ls {REMOTE_WORKDIR}/data/'
        f'mosek_d10_l3_z2full.json 2>/dev/null)"; '
        f'echo "D12_JSON=$(ls {REMOTE_WORKDIR}/data/'
        f'mosek_d12_l3_z2full.json 2>/dev/null)"'
    )
    r = ssh(host, port, cmd, t=30)
    if r.returncode != 0:
        return f'UNREACHABLE rc={r.returncode} stderr={r.stderr.strip()}'
    parsed = {}
    for line in r.stdout.strip().splitlines():
        if '=' in line:
            k, _, v = line.partition('=')
            parsed[k.strip()] = v.strip()
    state = parsed.get('STATE', 'UNKNOWN')
    pid = parsed.get('PID', '')
    last = parsed.get('LAST', '')
    d10 = 'ok' if parsed.get('D10_JSON') else 'none'
    d12 = 'ok' if parsed.get('D12_JSON') else 'none'
    if state == 'DONE':
        return f'DONE d10={d10} d12={d12}'
    return f'RUNNING pid={pid or "?"} d10={d10} d12={d12} last="{last[-120:]}"'


def fetch_results(host, port):
    print('Fetching result artefacts...')
    local_dir = (Path(__file__).resolve().parent / 'data'
                  / 'mosek_validation_results')
    local_dir.mkdir(parents=True, exist_ok=True)
    for fname in ('mosek_validation.log',
                   'mosek_d10_l3_z2full.json',
                   'mosek_d12_l3_z2full.json'):
        remote = f'{REMOTE_WORKDIR}/data/{fname}'
        local = local_dir / fname
        r = scp_from(host, port, remote, str(local))
        print(f'  {fname}: '
              f'{"ok" if r.returncode == 0 else r.stderr.strip()}')
    print(f'Saved to {local_dir}')


def teardown():
    state = load_state()
    if not state:
        print('No validation pod state.')
        return
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    print(f'Terminating pod {state["pod_id"]}...')
    try:
        runpod.terminate_pod(state['pod_id'])
    except Exception as exc:
        print(f'  terminate failed (maybe already gone): {exc}')
    STATE_FILE.unlink(missing_ok=True)
    print('Done.')


# =====================================================================
# CLI
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log', action='store_true')
    p.add_argument('--status', action='store_true')
    p.add_argument('--fetch', action='store_true')
    p.add_argument('--teardown', action='store_true')
    p.add_argument('--ssh', action='store_true')
    p.add_argument('--sync-only', action='store_true')
    p.add_argument('--run-only', action='store_true')
    args = p.parse_args()

    if args.teardown:
        teardown()
        return

    state = load_state()
    if args.ssh:
        if state:
            print(f'ssh -p {state["ssh_port"]} -i {SSH_KEY_PATH} '
                  f'-o StrictHostKeyChecking=no '
                  f'root@{state["ssh_host"]}')
        return
    if args.log:
        if not state:
            print('No validation pod state.')
            return
        tail_log(state['ssh_host'], state['ssh_port'])
        return
    if args.status:
        if not state:
            print('NO_STATE')
            return
        print(status(state['ssh_host'], state['ssh_port']))
        return
    if args.fetch:
        if not state:
            print('No validation pod state.')
            return
        fetch_results(state['ssh_host'], state['ssh_port'])
        return

    # Full deploy
    if state:
        print(f'Existing validation pod state: {state["pod_id"]}')
        r = ssh(state['ssh_host'], state['ssh_port'], 'echo ok', t=15)
        if r.returncode == 0 and 'ok' in r.stdout:
            print('  pod alive; reusing.')
        else:
            print('  pod unreachable; creating fresh.')
            state = create_pod()
    else:
        state = create_pod()

    if not wait_for_ssh(state['ssh_host'], state['ssh_port']):
        raise RuntimeError('SSH never came up on validation pod.')

    if args.sync_only:
        sync_files(state['ssh_host'], state['ssh_port'])
        return

    if args.run_only:
        run_validation(state['ssh_host'], state['ssh_port'])
        return

    install_mosek(state['ssh_host'], state['ssh_port'])
    sync_files(state['ssh_host'], state['ssh_port'])
    run_validation(state['ssh_host'], state['ssh_port'])


if __name__ == '__main__':
    main()
