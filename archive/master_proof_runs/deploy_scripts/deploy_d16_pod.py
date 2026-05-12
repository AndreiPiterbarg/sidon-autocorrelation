#!/usr/bin/env python
"""Deploy 4× H100 SXM RunPod (1004 GB RAM) for the d=16 L3 z2_full
MOSEK feasibility probe at t = 1.2802.

The single H100 SKU capped container RAM at 251 GB which OOM-killed
d=12 on its second bisection step.  The 4× H100 SXM SKU on RunPod
advertises 1004 GB RAM + 112 vCPU — enough headroom for the full
d=16 L3 z2_full IPM (expected 500 GB – 1 TB peak).

CRITICAL: on boot the deploy script prints the container cgroup
memory limit (`/sys/fs/cgroup/memory.max`).  If this shows ~251 GB
instead of ~1 TB, RunPod is throttling the container and the run
must be cancelled before OOM.

USAGE
-----
    python deploy_d16_pod.py                  # full deploy (create+install+run)
    python deploy_d16_pod.py --ssh            # print ssh command
    python deploy_d16_pod.py --status         # compact status
    python deploy_d16_pod.py --log            # tail remote log
    python deploy_d16_pod.py --fetch          # copy results + proof locally
    python deploy_d16_pod.py --teardown       # destroy pod

The probe solves exactly one SDP at t = 1.2802; a `Dual=Certificate`
result CERTIFIES lb ≥ 1.2802 at d=16 L3 full Lasserre — the
existing world-record bound.  Proof bundle (MOSEK task + vectors +
meta) is dumped to data/mosek_d16_proof/ on the pod and fetched by
--fetch.
"""
from __future__ import annotations

import argparse
import json
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
              / 'd16_pod_state.json')
MOSEK_LIC_LOCAL = Path.home() / 'mosek' / 'mosek.lic'

# 2× H100 SXM ≈ 500 GB RAM, 56 vCPU, ~$6/hr.  Sized from the prior
# d=16 run's measured peak RSS (342 GB) plus 90 GB safety — easily
# under 500 GB.  Halves the per-hour cost vs 4× H100.  If the 2×
# SKU is unavailable RunPod will fall back; we verify cgroup memory
# on boot and abort if < 450 GB.
GPU_TYPE_ID = 'NVIDIA H100 80GB HBM3'
GPU_COUNT = 2
POD_NAME = 'sidon-d16-mosek'

RUNNER_REMOTE = f'{REMOTE_WORKDIR}/run_d16.sh'
LOG_REMOTE = f'{REMOTE_WORKDIR}/data/mosek_d16_l3.log'
DONE_MARKER = f'{REMOTE_WORKDIR}/data/mosek_d16_done'
RESULT_JSON = f'{REMOTE_WORKDIR}/data/mosek_d16_l3_z2full.json'
PROOF_DIR = f'{REMOTE_WORKDIR}/data/mosek_d16_proof'

FILES_TO_SYNC = [
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


# =====================================================================
# Helpers
# =====================================================================

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

    print(f'Creating {GPU_COUNT}x {GPU_TYPE_ID} pod for d=16 L3 run...')
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


def verify_container_limits(host, port) -> dict:
    """CRITICAL: confirm the cgroup memory limit is >= 500 GB.

    If it comes back as 251 GB (the single-H100 limit), abort — the
    run will OOM just like d=12 did.
    """
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
    # 450 GB minimum: observed peak RSS was 342 GB with ~90 GB headroom
    # needed for numeric IPM scratch.  A 500 GB 2× H100 pod (the new
    # default SKU) clears this threshold.
    if cgroup_gb is None or cgroup_gb < 450:
        print(f'\n!!! WARNING: container cgroup limit = '
              f'{cgroup_gb} GB, < 450 GB threshold.')
        print('!!! d=16 L3 full MOSEK may OOM.  Abort and '
              'pick a different SKU (try 4× H100).')
    else:
        print(f'\nOK: container RAM = {cgroup_gb:.0f} GB '
              '(>= 450 GB threshold).')
    return {'cgroup_gb': cgroup_gb}


def install_mosek(host, port):
    print('Installing MOSEK + deps...')
    # psutil: powers the Python watcher thread (RSS / CPU / thread sampling).
    # py-spy : lets us attach to the running process from SSH and dump the
    #          native stack if MOSEK stalls — diagnoses which internal
    #          routine is single-threaded.
    r = ssh(host, port,
             'pip install --quiet mosek numpy scipy psutil py-spy '
             '2>&1 | tail -3',
             t=600)
    out = (r.stdout or r.stderr).strip()
    print(f'  pip: {out[-400:]}')

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
    print('Syncing code...')
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
        r = scp_to(host, port, str(local), f'{REMOTE_WORKDIR}/{f}')
        print(f'  {f} -> '
              f'{"ok" if r.returncode == 0 else f"FAIL: {r.stderr.strip()[:200]}"}')


def launch_probe(host, port, t_val: float = 1.2802) -> None:
    """Fire one MOSEK feasibility probe at t = t_val.

    We use n-bisect = 0 so the only SDP solve is the "hi probe" at
    t = t_val.  A `Dual=Certificate` verdict at that t certifies
    lb >= t_val (since the SDP is primal-infeasible).
    """
    script = (
        "#!/bin/bash\n"
        "set -u\n"
        f"cd {REMOTE_WORKDIR}\n"
        f"export PYTHONPATH={REMOTE_WORKDIR}:{REMOTE_WORKDIR}/tests\n"
        "export PYTHONUNBUFFERED=1\n"
        f"mkdir -p {PROOF_DIR}\n"
        "echo \"=== START $(date -u +%FT%TZ) ===\"\n"
        "echo \"=== d=16 L3 z2_full + pre_elim at t=\"" +
        f"{t_val}\" ===\"\n"
        "python -u tests/lasserre_mosek_tuned.py "
        f"  --d 16 --order 3 --mode z2_full --n-bisect 0 "
        f"  --t-lo {t_val} --t-hi {t_val} "
        f"  --pre-elim "
        f"  --json {RESULT_JSON} --proof-dir {PROOF_DIR} "
        "  || echo 'd=16 FAILED:' $?\n"
        "echo \"=== END $(date -u +%FT%TZ) ===\"\n"
        f"touch {DONE_MARKER}\n"
    )
    tmp = (Path(__file__).resolve().parent / 'data' / '_d16_runner.sh')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(script.encode('utf-8').replace(b'\r\n', b'\n'))
    r = scp_to(host, port, str(tmp), RUNNER_REMOTE)
    if r.returncode != 0:
        raise RuntimeError(f'scp runner failed: {r.stderr.strip()}')

    launch_cmd = (
        f'rm -f {DONE_MARKER} {LOG_REMOTE} {RESULT_JSON}; '
        f'chmod +x {RUNNER_REMOTE} && '
        f'( nohup {RUNNER_REMOTE} > {LOG_REMOTE} 2>&1 < /dev/null & '
        f'  disown ) && echo LAUNCHED'
    )
    r = ssh(host, port, launch_cmd, t=30)
    print(f'  {r.stdout.strip()}')
    print(f'  log:   {LOG_REMOTE}')
    print(f'  done:  {DONE_MARKER}')
    print(f'  json:  {RESULT_JSON}')
    print(f'  proof: {PROOF_DIR}')


def status() -> str:
    state = load_state()
    if not state:
        return 'NO_STATE'
    host, port = state['ssh_host'], state['ssh_port']
    cmd = (
        f'if [ -f {DONE_MARKER} ]; then echo "STATE=DONE"; '
        f'else echo "STATE=RUNNING"; fi; '
        f'echo "MEM_MB=$(ps -o rss= -p $(pgrep -f lasserre_mosek_tuned '
        f'| head -1) 2>/dev/null | awk \'{{print int($1/1024)}}\')"; '
        f'echo "LAST=$(tail -n 1 {LOG_REMOTE} 2>/dev/null)"; '
        f'echo "JSON=$(ls {RESULT_JSON} 2>/dev/null)"; '
        f'echo "PROOF=$(ls -1 {PROOF_DIR} 2>/dev/null | wc -l)"'
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
    jflag = 'ok' if p.get('JSON') else 'none'
    nproof = p.get('PROOF', '0')
    if p.get('STATE') == 'DONE':
        return f'DONE json={jflag} proof_bundles={nproof}'
    return (f'RUNNING mem={mem}MB proof={nproof} json={jflag} '
            f'last="{p.get("LAST", "")[-120:]}"')


def tail_log(n=200):
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
                  / 'mosek_d16_results')
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in ('mosek_d16_l3.log', 'mosek_d16_l3_z2full.json'):
        r = scp_from(host, port, f'{REMOTE_WORKDIR}/data/{name}',
                      str(local_dir / name))
        print(f'  {name}: '
              f'{"ok" if r.returncode == 0 else r.stderr.strip()[:200]}')
    # Proof bundle directory.
    proof_local = local_dir / 'mosek_d16_proof'
    proof_local.mkdir(parents=True, exist_ok=True)
    r = scp_from(host, port, f'{PROOF_DIR}/.',
                  str(proof_local), recursive=True)
    print(f'  proof/: '
          f'{"ok" if r.returncode == 0 else r.stderr.strip()[:200]}')
    print(f'Saved to {local_dir}')


def teardown():
    state = load_state()
    if not state:
        print('No d=16 pod state.')
        return
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    print(f'Terminating pod {state["pod_id"]}...')
    try:
        runpod.terminate_pod(state['pod_id'])
    except Exception as exc:
        print(f'  terminate failed: {exc}')
    STATE_FILE.unlink(missing_ok=True)
    print('Done.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--status', action='store_true')
    p.add_argument('--log', action='store_true')
    p.add_argument('--fetch', action='store_true')
    p.add_argument('--teardown', action='store_true')
    p.add_argument('--ssh', action='store_true')
    p.add_argument('--sync-only', action='store_true')
    p.add_argument('--launch-only', action='store_true',
                    help='Only launch probe (requires existing pod)')
    p.add_argument('--t', type=float, default=1.2802,
                    help='Target t for feasibility probe (default 1.2802)')
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
    if args.status:
        print(status())
        return
    if args.log:
        tail_log()
        return
    if args.fetch:
        fetch()
        return

    if state:
        print(f'Existing d=16 pod state: {state["pod_id"]}')
        r = ssh(state['ssh_host'], state['ssh_port'], 'echo ok', t=15)
        if r.returncode == 0 and 'ok' in r.stdout:
            print('  pod alive; reusing.')
        else:
            print('  pod unreachable; creating fresh.')
            state = create_pod()
    else:
        state = create_pod()

    if not wait_for_ssh(state['ssh_host'], state['ssh_port']):
        raise RuntimeError('SSH never came up.')

    # ALWAYS verify cgroup memory before launching.
    limits = verify_container_limits(
        state['ssh_host'], state['ssh_port'])
    if limits.get('cgroup_gb') is None or limits['cgroup_gb'] < 500:
        print('\nABORT: container RAM < 500 GB. '
              'Run --teardown and pick a larger SKU.')
        return

    if args.sync_only:
        sync_files(state['ssh_host'], state['ssh_port'])
        return

    if args.launch_only:
        launch_probe(state['ssh_host'], state['ssh_port'], args.t)
        return

    install_mosek(state['ssh_host'], state['ssh_port'])
    sync_files(state['ssh_host'], state['ssh_port'])
    launch_probe(state['ssh_host'], state['ssh_port'], args.t)


if __name__ == '__main__':
    main()
