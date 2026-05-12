#!/usr/bin/env python
"""Deploy a SECOND CPU pod dedicated to interval_bnb runs.

This sidesteps cpupod's single-session-file model so the existing
Lasserre sweep pod is not disturbed. State lives in
cpupod/.session_interval.json. Mirrors enough of cpupod.session.Session
to create + sync + launch + teardown.

Usage
-----
    python deploy_interval_bnb_pod.py start
    python deploy_interval_bnb_pod.py sync
    python deploy_interval_bnb_pod.py launch -- d=10 target=1.24
    python deploy_interval_bnb_pod.py launch -- d=14 target=1.2802
    python deploy_interval_bnb_pod.py status
    python deploy_interval_bnb_pod.py logs [-f]
    python deploy_interval_bnb_pod.py fetch
    python deploy_interval_bnb_pod.py teardown

The launched script is `interval_bnb/run_pod.py` — a one-shot driver
that parses d and target arguments and writes
    data/interval_bnb_d{D}_t{T}.log
    data/interval_bnb_d{D}_t{T}.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cpupod import config as cp_cfg  # noqa: E402
from cpupod import pod_manager  # noqa: E402
from cpupod.budget import BudgetTracker  # noqa: E402
from cpupod.sync import sync_code, collect_results  # noqa: E402
from cpupod.remote import install_deps, verify_cpu, launch_script, tail_remote_log, check_job_status  # noqa: E402


SESSION_FILE = _HERE / "cpupod" / ".session_interval.json"


def _load():
    if SESSION_FILE.exists():
        try:
            return json.loads(SESSION_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save(data):
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SESSION_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(SESSION_FILE)


def _clear():
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()


def cmd_start(args):
    data = _load()
    if data.get("pod_id"):
        st = pod_manager.get_pod_status(data["pod_id"])
        if st and st.get("desiredStatus") == "RUNNING":
            print(f"Existing interval_bnb pod: {data['pod_id']} "
                  f"ssh -p {data['ssh_port']} root@{data['ssh_host']}")
            return
    info = pod_manager.create_pod(name="sidon-interval-bnb")
    data = {
        "pod_id": info["pod_id"],
        "ssh_host": info["ssh_host"],
        "ssh_port": info["ssh_port"],
        "start_time": time.time(),
    }
    _save(data)
    print(f"\nSSH: ssh -p {data['ssh_port']} root@{data['ssh_host']}")
    try:
        sync_code(data["ssh_host"], data["ssh_port"])
        install_deps(data["ssh_host"], data["ssh_port"])
        verify_cpu(data["ssh_host"], data["ssh_port"])
    except Exception as e:
        print(f"Setup failed: {e}")
        raise
    print("Pod ready.")


def cmd_sync(args):
    data = _load()
    if not data.get("pod_id"):
        print("No session. Run start first."); sys.exit(1)
    sync_code(data["ssh_host"], data["ssh_port"])


def cmd_launch(args):
    data = _load()
    if not data.get("pod_id"):
        print("No session."); sys.exit(1)
    script = args.script or "interval_bnb/run_pod.py"
    # argparse REMAINDER may include a leading "--" from the CLI; drop it.
    raw = [a for a in args.args if a != "--"]
    script_args = " ".join(raw)
    launch_script(
        data["ssh_host"], data["ssh_port"],
        script=script, args=script_args,
        auto_teardown=False, api_key=None, pod_id=None,
    )
    print(f"Launched {script} {script_args!r} on pod.")


def cmd_status(args):
    data = _load()
    if not data.get("pod_id"):
        print("No session."); return
    print(f"pod_id={data['pod_id']} ssh -p {data['ssh_port']} root@{data['ssh_host']}")
    st = pod_manager.get_pod_status(data["pod_id"])
    if st:
        print(f"  Pod: {st.get('desiredStatus')} uptime={st.get('uptimeSeconds', 0)}s")
    try:
        job = check_job_status(data["ssh_host"], data["ssh_port"])
        print(f"  Job: {job}")
    except Exception as e:
        print(f"  Job: (unreachable: {e})")


def cmd_logs(args):
    data = _load()
    if not data.get("pod_id"):
        print("No session."); sys.exit(1)
    tail_remote_log(data["ssh_host"], data["ssh_port"],
                    follow=args.follow, lines=args.lines)


def cmd_fetch(args):
    data = _load()
    if not data.get("pod_id"):
        print("No session."); sys.exit(1)
    collect_results(data["ssh_host"], data["ssh_port"])


def cmd_teardown(args):
    data = _load()
    if not data.get("pod_id"):
        print("No session."); return
    try:
        collect_results(data["ssh_host"], data["ssh_port"])
    except Exception as e:
        print(f"warn: fetch failed: {e}")
    try:
        pod_manager.terminate_pod(data["pod_id"])
    except Exception as e:
        print(f"warn: terminate failed: {e}")
    _clear()
    print("interval_bnb pod torn down.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start")
    sub.add_parser("sync")
    p_l = sub.add_parser("launch")
    p_l.add_argument("--script", default=None)
    p_l.add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    p_lg = sub.add_parser("logs")
    p_lg.add_argument("-f", "--follow", action="store_true")
    p_lg.add_argument("-n", "--lines", type=int, default=80)
    sub.add_parser("fetch")
    sub.add_parser("teardown")
    args = ap.parse_args()
    dispatch = {
        "start": cmd_start, "sync": cmd_sync, "launch": cmd_launch,
        "status": cmd_status, "logs": cmd_logs, "fetch": cmd_fetch,
        "teardown": cmd_teardown,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
