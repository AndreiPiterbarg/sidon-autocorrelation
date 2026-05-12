#!/usr/bin/env python
"""Deploy 2x H100 SXM pod (~500 GB RAM) to run the d=16 L3 SUBLEVEL
solver with Z/2 pre-elim and target lb > 1.2802.

The cpu3m-32-256 CPU pod OOMed at 238 GB during MOSEK's Schur-complement
factorisation because the container cgroup caps at 256 GB.  The 2x H100
SXM SKU provides ~500 GB cgroup memory, which matches the historical
d=16 L3 run's 342 GB peak RSS with headroom.

USAGE
    python deploy_sublevel_d16_pod.py             # full deploy+launch
    python deploy_sublevel_d16_pod.py --status    # progress snapshot
    python deploy_sublevel_d16_pod.py --log [N]   # tail remote log
    python deploy_sublevel_d16_pod.py --fetch     # collect results
    python deploy_sublevel_d16_pod.py --teardown  # destroy pod
    python deploy_sublevel_d16_pod.py --ssh       # print ssh command
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
              / 'sublevel_d16_pod_state.json')
MOSEK_LIC_LOCAL = Path.home() / 'mosek' / 'mosek.lic'
GPU_TYPE_ID = 'NVIDIA H100 80GB HBM3'
GPU_COUNT = 2
POD_NAME = 'sidon-sublevel-d16'

RUNNER_REMOTE = f'{REMOTE_WORKDIR}/run_sublevel_d16.sh'
LOG_REMOTE = f'{REMOTE_WORKDIR}/data/sublevel_d16.log'
DONE_MARKER = f'{REMOTE_WORKDIR}/data/sublevel_d16_done'
RESULT_JSON = f'{REMOTE_WORKDIR}/data/sublevel_d16_final.json'
PROGRESS_JSON = f'{REMOTE_WORKDIR}/data/sublevel_d16_progress.json'
PROOF_DIR = f'{REMOTE_WORKDIR}/data/mosek_d16_sublevel_proof'

FILES_TO_SYNC = [
    'tests/lasserre_mosek_sublevel.py',
    'tests/lasserre_mosek_tuned.py',
    'tests/lasserre_fusion.py',
    'tests/lasserre_scalable.py',
    'lasserre/__init__.py',
    'lasserre/core.py',
    'lasserre/precompute.py',
    'lasserre/cliques.py',
    'lasserre/z2_symmetry.py',
    'lasserre/z2_blockdiag.py',
    'lasserre/z2_elim.py',
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
    if not pub_path.exists():
        raise RuntimeError(f'Missing SSH public key at {pub_path}')
    pub_key = pub_path.read_text().strip()

    print(f'Creating {GPU_COUNT}x {GPU_TYPE_ID} pod for d=16 L3 '
          f'SUBLEVEL run (target lb > 1.2802)...')
    pod = runpod.create_pod(
        name=POD_NAME,
        image_name=DOCKER_IMAGE,
        gpu_type_id=GPU_TYPE_ID,
        gpu_count=GPU_COUNT,
        cloud_type='ALL',
        container_disk_in_gb=60,
        volume_in_gb=0,
        env={'PUBLIC_KEY': pub_key},
        ports='22/tcp',
    )
    pod_id = pod['id']
    print(f'  Pod ID: {pod_id}')

    host = port = None
    for attempt in range(120):
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

    state = {
        'pod_id': pod_id, 'ssh_host': host, 'ssh_port': port,
        'created_at': time.time(), 'gpu_type': GPU_TYPE_ID,
        'gpu_count': GPU_COUNT,
    }
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


def verify_container_limits(host, port):
    r = ssh(host, port,
             'echo CGROUP_MEM=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); '
             'echo NPROC=$(nproc); '
             'free -g | head -3', t=30)
    print(r.stdout)
    cgroup_gb = None
    for line in r.stdout.splitlines():
        if line.startswith('CGROUP_MEM='):
            val = line.split('=', 1)[1].strip()
            try:
                cgroup_gb = int(val) / 1e9
            except ValueError:
                cgroup_gb = None
    if cgroup_gb is None or cgroup_gb < 400:
        print(f'\n!!! WARNING: container cgroup limit = {cgroup_gb} GB, '
              '< 400 GB threshold.  d=16 L3 sublevel may OOM again.')
        return False
    print(f'\nOK: container RAM = {cgroup_gb:.0f} GB (>= 400 GB).')
    return True


def install_mosek(host, port):
    print('Installing MOSEK + deps on pod...')
    r = ssh(host, port,
             'pip install --quiet mosek numpy scipy psutil py-spy '
             '2>&1 | tail -3', t=600)
    print(f'  pip: {(r.stdout or r.stderr).strip()[-400:]}')

    if not MOSEK_LIC_LOCAL.exists():
        raise RuntimeError(f'No MOSEK license at {MOSEK_LIC_LOCAL}')
    ssh(host, port, 'mkdir -p /root/mosek', t=30)
    r = scp_to(host, port, str(MOSEK_LIC_LOCAL),
                '/root/mosek/mosek.lic')
    print(f'  license: {"ok" if r.returncode == 0 else r.stderr.strip()}')
    r = ssh(host, port,
             'python -c "import mosek; print(mosek.Env.getversion())"',
             t=60)
    print(f'  mosek: {r.stdout.strip()}')


def sync_files(host, port):
    print('Syncing code to pod...')
    ssh(host, port,
         f'mkdir -p {REMOTE_WORKDIR}/tests {REMOTE_WORKDIR}/lasserre '
         f'{REMOTE_WORKDIR}/data', t=30)
    root = Path(__file__).resolve().parent
    for f in FILES_TO_SYNC:
        local = root / f
        if not local.exists():
            print(f'  MISSING: {f}')
            continue
        r = scp_to(host, port, str(local), f'{REMOTE_WORKDIR}/{f}')
        print(f'  {f} -> '
              f'{"ok" if r.returncode == 0 else f"FAIL: {r.stderr.strip()[:200]}"}')


def launch_sublevel(host, port, time_budget_s=28800,
                    add_per_step=50, bisect_per_step=3,
                    target_lb=1.2802):
    """Fire the sublevel solver detached on the pod.

    Parameters match the ones validated locally.  Default wall budget
    is 8 h — 2x H100 SXM on RunPod spot is ~$4-6/hr so this is ~$32-48
    worst case.
    """
    script = (
        "#!/bin/bash\n"
        "set -u\n"
        f"cd {REMOTE_WORKDIR}\n"
        f"export PYTHONPATH={REMOTE_WORKDIR}:{REMOTE_WORKDIR}/tests\n"
        "export PYTHONUNBUFFERED=1\n"
        f"mkdir -p {PROOF_DIR}\n"
        "echo \"=== START $(date -u +%FT%TZ) ===\"\n"
        f"echo \"=== d=16 L3 SUBLEVEL + z2 pre-elim, target lb > "
        f"{target_lb} ===\"\n"
        "python -u tests/lasserre_mosek_sublevel.py "
        f"  --d 16 --order 3 --sublevel "
        f"  --target-lb {target_lb} "
        f"  --add-per-step {add_per_step} "
        f"  --bisect-per-step {bisect_per_step} "
        f"  --time-budget-s {time_budget_s} "
        f"  --watcher-interval 60 "
        f"  --z2-pre-elim "
        f"  --progress {PROGRESS_JSON} "
        f"  --json {RESULT_JSON} "
        f"  --proof-dir {PROOF_DIR} "
        "  || echo 'sublevel FAILED:' $?\n"
        "echo \"=== END $(date -u +%FT%TZ) ===\"\n"
        f"touch {DONE_MARKER}\n"
    )
    tmp = (Path(__file__).resolve().parent / 'data'
            / '_sublevel_d16_runner.sh')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(script.encode('utf-8').replace(b'\r\n', b'\n'))
    r = scp_to(host, port, str(tmp), RUNNER_REMOTE)
    if r.returncode != 0:
        raise RuntimeError(f'scp runner failed: {r.stderr.strip()}')

    launch_cmd = (
        f'rm -f {DONE_MARKER} {LOG_REMOTE} {RESULT_JSON} '
        f'  {PROGRESS_JSON}; '
        f'chmod +x {RUNNER_REMOTE} && '
        f'( nohup {RUNNER_REMOTE} > {LOG_REMOTE} 2>&1 < /dev/null & '
        f'  disown ) && echo LAUNCHED'
    )
    r = ssh(host, port, launch_cmd, t=30)
    print(f'  {r.stdout.strip()}')
    print(f'  log      : {LOG_REMOTE}')
    print(f'  done     : {DONE_MARKER}')
    print(f'  result   : {RESULT_JSON}')
    print(f'  progress : {PROGRESS_JSON}')
    print(f'  proof    : {PROOF_DIR}')


def status():
    state = load_state()
    if not state:
        return 'NO_STATE'
    host, port = state['ssh_host'], state['ssh_port']
    cmd = (
        f'if [ -f {DONE_MARKER} ]; then echo "STATE=DONE"; '
        f'else echo "STATE=RUNNING"; fi; '
        f'echo "MEM_MB=$(ps -o rss= -p $(pgrep -f '
        f'lasserre_mosek_sublevel | head -1) 2>/dev/null '
        f'| awk \'{{print int($1/1024)}}\')"; '
        f'echo "LAST=$(tail -n 1 {LOG_REMOTE} 2>/dev/null)"; '
        f'if [ -f {PROGRESS_JSON} ]; then '
        f'  echo "PROGRESS=$(python3 -c "import json; '
        f'd=json.load(open(\\"{PROGRESS_JSON}\\")); '
        f'print(f\\"step={{d.get(\\\'step\\\', \\\'?\\\')}}| '
        f'n_active={{d.get(\\\'n_active\\\', \\\'?\\\')}}| '
        f'current_lb={{d.get(\\\'current_lb\\\', \\\'?\\\')}}| '
        f'final_status={{d.get(\\\'final_status\\\', \\\'?\\\')}}| '
        f'elapsed={{int(d.get(\\\'elapsed_s\\\', 0))}}\\")" '
        f'2>/dev/null)"; fi'
    )
    r = ssh(host, port, cmd, t=30)
    if r.returncode != 0:
        return f'UNREACHABLE rc={r.returncode}'
    p = {}
    for line in r.stdout.strip().splitlines():
        if '=' in line:
            k, _, v = line.partition('=')
            p[k.strip()] = v.strip()
    mem = p.get('MEM_MB', '')
    progress = p.get('PROGRESS', '')
    if p.get('STATE') == 'DONE':
        return f'DONE progress="{progress}" last="{p.get("LAST", "")[-120:]}"'
    return (f'RUNNING mem={mem}MB progress="{progress}" '
            f'last="{p.get("LAST", "")[-120:]}"')


def tail_log(n=150):
    state = load_state()
    host, port = state['ssh_host'], state['ssh_port']
    r = ssh(host, port,
             f'ls -la {LOG_REMOTE} 2>&1 | head; echo "----"; '
             f'tail -n {n} {LOG_REMOTE} 2>&1', t=30)
    print(r.stdout)


def fetch():
    state = load_state()
    host, port = state['ssh_host'], state['ssh_port']
    local_dir = (Path(__file__).resolve().parent / 'data'
                  / 'sublevel_d16_results')
    local_dir.mkdir(parents=True, exist_ok=True)
    for remote, name in (
        (LOG_REMOTE, 'sublevel_d16.log'),
        (RESULT_JSON, 'sublevel_d16_final.json'),
        (PROGRESS_JSON, 'sublevel_d16_progress.json'),
    ):
        r = scp_from(host, port, remote, str(local_dir / name))
        print(f'  {name}: '
              f'{"ok" if r.returncode == 0 else r.stderr.strip()[:200]}')
    # Proof bundle directory
    proof_local = local_dir / 'proof'
    proof_local.mkdir(exist_ok=True)
    r = scp_from(host, port, f'{PROOF_DIR}/.',
                  str(proof_local), recursive=True)
    print(f'  proof: '
          f'{"ok" if r.returncode == 0 else r.stderr.strip()[:200]}')


def teardown():
    state = load_state()
    if not state:
        print('No state; nothing to tear down.')
        return
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    pod_id = state['pod_id']
    print(f'Terminating pod {pod_id}...')
    try:
        runpod.terminate_pod(pod_id)
        print('  terminated.')
    except Exception as exc:
        print(f'  termination error: {exc}')
    STATE_FILE.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--log', nargs='?', const=150, type=int)
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--teardown', action='store_true')
    ap.add_argument('--ssh', action='store_true')
    ap.add_argument('--target-lb', type=float, default=1.2802)
    ap.add_argument('--add-per-step', type=int, default=50)
    ap.add_argument('--bisect-per-step', type=int, default=3)
    ap.add_argument('--time-budget-s', type=float, default=28800.0,
                     help='Default 8 hours.')
    args = ap.parse_args()

    if args.ssh:
        state = load_state()
        if state:
            print(f'ssh -p {state["ssh_port"]} '
                  f'-i {SSH_KEY_PATH} root@{state["ssh_host"]}')
        else:
            print('No state.')
        return
    if args.status:
        print(status())
        return
    if args.log is not None:
        tail_log(args.log)
        return
    if args.fetch:
        fetch()
        return
    if args.teardown:
        teardown()
        return

    # Full deploy pipeline.
    state = load_state()
    if state:
        print(f'Existing state found: pod {state["pod_id"]}.  '
              'Teardown first with --teardown, or remove '
              f'{STATE_FILE}.')
        sys.exit(1)

    state = create_pod()
    host, port = state['ssh_host'], state['ssh_port']
    print('\nWaiting for SSH...')
    if not wait_for_ssh(host, port):
        raise RuntimeError('SSH never came up')
    print('\nVerifying container limits...')
    verify_container_limits(host, port)
    print('\nInstalling MOSEK...')
    install_mosek(host, port)
    print('\nSyncing files...')
    sync_files(host, port)
    print('\nLaunching sublevel solver...')
    launch_sublevel(host, port,
                    time_budget_s=args.time_budget_s,
                    add_per_step=args.add_per_step,
                    bisect_per_step=args.bisect_per_step,
                    target_lb=args.target_lb)
    print('\n=== LAUNCHED ===')
    print(f'pod_id = {state["pod_id"]}')
    print('\nTo follow progress:')
    print('  python deploy_sublevel_d16_pod.py --status')
    print('  python deploy_sublevel_d16_pod.py --log')
    print('  python deploy_sublevel_d16_pod.py --fetch')
    print('  python deploy_sublevel_d16_pod.py --teardown')


if __name__ == '__main__':
    main()
