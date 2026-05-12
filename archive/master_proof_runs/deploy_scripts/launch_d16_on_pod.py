#!/usr/bin/env python
"""Launch the d=16 L3 z2_full MOSEK job on the existing validation pod.

Reuses the same pod already deployed for d=10/d=12 validation (2 TB RAM,
224 vCPUs is more than enough for d=16).  Uploads a dedicated runner and
fires it in the background so the local monitor can poll every 15 min.

Writes DONE_MARKER_D16 on completion to signal the local monitor.

USAGE
-----
    python launch_d16_on_pod.py                 # fire d=16 run
    python launch_d16_on_pod.py --status        # compact status
    python launch_d16_on_pod.py --log           # tail remote log
    python launch_d16_on_pod.py --fetch         # pull results locally
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpupod.config import SSH_OPTIONS, SSH_KEY_PATH, REMOTE_WORKDIR

STATE_FILE = (Path(__file__).resolve().parent / 'data'
              / 'mosek_validation_pod_state.json')
RUNNER_REMOTE = f'{REMOTE_WORKDIR}/run_d16.sh'
LOG_REMOTE = f'{REMOTE_WORKDIR}/data/mosek_d16_l3.log'
DONE_MARKER_D16 = f'{REMOTE_WORKDIR}/data/mosek_d16_done'
RESULT_JSON = f'{REMOTE_WORKDIR}/data/mosek_d16_l3_z2full.json'


def ssh(host, port, cmd, t=60):
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
        capture_output=True, text=True, timeout=300)


def _load_state():
    if not STATE_FILE.exists():
        raise RuntimeError('No pod state — deploy the validation pod '
                            'first (deploy_mosek_validation_pod.py).')
    return json.loads(STATE_FILE.read_text())


def launch(n_bisect: int = 3, t_lo: float = 1.2802,
            t_hi: float = 1.32) -> None:
    """Upload runner and launch d=16 L3 z2_full in the background.

    Defaults bracket [1.2802, 1.32] because the scientific objective is
    to certify lb > 1.2802 at d=16 L3 — a short bisection below that
    floor is sufficient for a world-record proof.
    """
    state = _load_state()
    host, port = state['ssh_host'], state['ssh_port']

    proof_dir = f'{REMOTE_WORKDIR}/data/mosek_d16_proof'
    script = (
        "#!/bin/bash\n"
        "set -u\n"
        f"cd {REMOTE_WORKDIR}\n"
        f"export PYTHONPATH={REMOTE_WORKDIR}:{REMOTE_WORKDIR}/tests\n"
        "export PYTHONUNBUFFERED=1\n"
        f"mkdir -p {proof_dir}\n"
        "echo \"=== START $(date -u +%FT%TZ) ===\"\n"
        "echo \"=== d=16 L3 z2_full ===\"\n"
        "python -u tests/lasserre_mosek_tuned.py "
        "  --d 16 --order 3 --mode z2_full "
        f"  --n-bisect {n_bisect} --t-lo {t_lo} --t-hi {t_hi} "
        f"  --json {RESULT_JSON} "
        f"  --proof-dir {proof_dir} "
        "  || echo 'd=16 FAILED:' $?\n"
        "echo \"=== END $(date -u +%FT%TZ) ===\"\n"
        f"touch {DONE_MARKER_D16}\n"
    )
    tmp = (Path(__file__).resolve().parent / 'data' / '_run_d16_local.sh')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(script.encode('utf-8').replace(b'\r\n', b'\n'))

    r = scp_to(host, port, str(tmp), RUNNER_REMOTE)
    if r.returncode != 0:
        raise RuntimeError(f'scp runner failed: {r.stderr.strip()}')
    print(f'  runner uploaded: {RUNNER_REMOTE}')

    launch_cmd = (
        f'rm -f {DONE_MARKER_D16} {LOG_REMOTE} {RESULT_JSON}; '
        f'chmod +x {RUNNER_REMOTE} && '
        f'( nohup {RUNNER_REMOTE} > {LOG_REMOTE} 2>&1 < /dev/null & '
        f'  disown ) && echo LAUNCHED'
    )
    r = ssh(host, port, launch_cmd, t=30)
    print(f'  {r.stdout.strip()}')
    if r.stderr:
        print(f'  stderr: {r.stderr.strip()}')
    print(f'  log:  {LOG_REMOTE}')
    print(f'  done: {DONE_MARKER_D16}')
    print(f'  json: {RESULT_JSON}')


def status() -> str:
    state = _load_state()
    host, port = state['ssh_host'], state['ssh_port']
    cmd = (
        f'if [ -f {DONE_MARKER_D16} ]; then echo "STATE=DONE"; '
        f'else echo "STATE=RUNNING"; fi; '
        f'echo "LAST=$(tail -n 1 {LOG_REMOTE} 2>/dev/null)"; '
        f'echo "JSON=$(ls {RESULT_JSON} 2>/dev/null)"'
    )
    r = ssh(host, port, cmd, t=30)
    if r.returncode != 0:
        return f'UNREACHABLE rc={r.returncode}'
    p = {}
    for line in r.stdout.strip().splitlines():
        if '=' in line:
            k, _, v = line.partition('=')
            p[k.strip()] = v.strip()
    jflag = 'ok' if p.get('JSON') else 'none'
    if p.get('STATE') == 'DONE':
        return f'DONE json={jflag}'
    return f'RUNNING json={jflag} last="{p.get("LAST", "")[-120:]}"'


def tail_log(n: int = 80) -> None:
    state = _load_state()
    host, port = state['ssh_host'], state['ssh_port']
    r = ssh(host, port,
             f'ls -la {LOG_REMOTE} 2>&1 | head; echo "----"; '
             f'tail -n {n} {LOG_REMOTE} 2>&1', t=30)
    print(r.stdout)


def fetch() -> None:
    state = _load_state()
    host, port = state['ssh_host'], state['ssh_port']
    local_dir = (Path(__file__).resolve().parent / 'data'
                  / 'mosek_d16_results')
    local_dir.mkdir(parents=True, exist_ok=True)
    # 1. Log + JSON result.
    for name in ('mosek_d16_l3.log', 'mosek_d16_l3_z2full.json'):
        r = scp_from(host, port, f'{REMOTE_WORKDIR}/data/{name}',
                      str(local_dir / name))
        print(f'  {name}: '
              f'{"ok" if r.returncode == 0 else r.stderr.strip()}')
    # 2. Proof bundle directory (recursive).
    proof_remote = f'{REMOTE_WORKDIR}/data/mosek_d16_proof'
    proof_local = local_dir / 'mosek_d16_proof'
    proof_local.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ['scp', '-P', str(port), '-r'] + SSH_OPTIONS +
        [f'root@{host}:{proof_remote}/.', str(proof_local)],
        capture_output=True, text=True, timeout=1800)
    print(f'  proof/: '
          f'{"ok" if r.returncode == 0 else r.stderr.strip()[:200]}')
    print(f'Saved to {local_dir}')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--status', action='store_true')
    p.add_argument('--log', action='store_true')
    p.add_argument('--fetch', action='store_true')
    p.add_argument('--n-bisect', type=int, default=3)
    p.add_argument('--t-lo', type=float, default=1.2802)
    p.add_argument('--t-hi', type=float, default=1.32)
    args = p.parse_args()

    if args.status:
        print(status())
        return
    if args.log:
        tail_log()
        return
    if args.fetch:
        fetch()
        return

    launch(n_bisect=args.n_bisect, t_lo=args.t_lo, t_hi=args.t_hi)


if __name__ == '__main__':
    main()
