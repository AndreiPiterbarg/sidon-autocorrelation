#!/usr/bin/env python
"""Deploy a FRESH CPU pod for Farkas-fast val(d) runs at d=12, d=16.

Uses a dedicated session file (cpupod/.session_farkas_fast.json) distinct
from both .session.json (old Lasserre sweep) and .session_interval2.json
(interval BnB).  `start` creates a brand-new pod via runpod SDK; if a
session file already exists and its pod is still RUNNING, the command
REFUSES to reuse unless --force-reuse is passed, and with --fresh it
tears down the existing pod first.

Usage
-----
    # Create NEW pod (fails if a pod from this session is already running
    # unless --fresh is given):
    python deploy_farkas_fast_pod.py start --fresh

    # Verify the pod_id is different from the other two session files:
    python deploy_farkas_fast_pod.py verify_new

    # Launch a single d/order probe (streams log to stdout via tail):
    python deploy_farkas_fast_pod.py launch -- --d 12 --order 3 --t_hi 1.30
    python deploy_farkas_fast_pod.py launch -- --d 16 --order 3 --t_hi 1.33

    python deploy_farkas_fast_pod.py status
    python deploy_farkas_fast_pod.py logs [-f]
    python deploy_farkas_fast_pod.py fetch
    python deploy_farkas_fast_pod.py teardown
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cpupod import pod_manager  # noqa: E402
from cpupod.sync import sync_code, collect_results  # noqa: E402
from cpupod.remote import (  # noqa: E402
    install_deps, verify_cpu, launch_script, tail_remote_log, check_job_status,
)


SESSION_FILE = _HERE / "cpupod" / ".session_farkas_fast.json"
OTHER_SESSION_FILES = [
    _HERE / "cpupod" / ".session.json",
    _HERE / "cpupod" / ".session_interval2.json",
    _HERE / "cpupod" / ".session_old.json",
]


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


def _other_pod_ids() -> list:
    out = []
    for p in OTHER_SESSION_FILES:
        if p.exists():
            try:
                d = json.loads(p.read_text())
                if d.get("pod_id"):
                    out.append((p.name, d["pod_id"]))
            except Exception:
                pass
    return out


def cmd_start(args):
    data = _load()
    if data.get("pod_id") and not args.fresh:
        st = pod_manager.get_pod_status(data["pod_id"])
        if st and st.get("desiredStatus") == "RUNNING":
            if args.force_reuse:
                print(f"Reusing existing farkas-fast pod: {data['pod_id']} "
                      f"ssh -p {data['ssh_port']} root@{data['ssh_host']}")
                return
            print(f"REFUSING to reuse running pod {data['pod_id']} "
                  f"(host={data['ssh_host']}:{data['ssh_port']}).\n"
                  f"  Pass --fresh to terminate and create a new one, "
                  f"--force-reuse to reuse it, or run `teardown` first.",
                  file=sys.stderr)
            sys.exit(2)
    if data.get("pod_id") and args.fresh:
        print(f"--fresh: tearing down previous pod {data['pod_id']}...")
        try:
            pod_manager.terminate_pod(data["pod_id"])
        except Exception as e:
            print(f"  warn: terminate failed: {e}")
        _clear()
        data = {}

    # Sanity: print other pod ids so user can verify the new one is distinct.
    others = _other_pod_ids()
    if others:
        print("Other session files (for distinctness check):")
        for (name, pid) in others:
            print(f"  {name}: {pid}")

    info = pod_manager.create_pod(name="sidon-farkas-fast")
    new_pid = info["pod_id"]
    for (_, pid) in others:
        if pid == new_pid:
            raise RuntimeError(
                f"NEWLY CREATED pod {new_pid} collides with existing "
                f"session — something is wrong. Aborting.")
    data = {
        "pod_id": new_pid,
        "ssh_host": info["ssh_host"],
        "ssh_port": info["ssh_port"],
        "start_time": time.time(),
    }
    _save(data)
    print(f"\nNEW POD: {new_pid}")
    print(f"SSH: ssh -p {data['ssh_port']} root@{data['ssh_host']}")
    try:
        sync_code(data["ssh_host"], data["ssh_port"])
        install_deps(data["ssh_host"], data["ssh_port"])
        verify_cpu(data["ssh_host"], data["ssh_port"])
    except Exception as e:
        print(f"Setup failed: {e}")
        raise
    print("Pod ready.")


def cmd_verify_new(args):
    data = _load()
    if not data.get("pod_id"):
        print("No session."); sys.exit(1)
    others = _other_pod_ids()
    print(f"current (farkas_fast) pod_id = {data['pod_id']}")
    for (name, pid) in others:
        tag = "DISTINCT" if pid != data["pod_id"] else "COLLIDES"
        print(f"  {name}: {pid}  [{tag}]")
    for (_, pid) in others:
        if pid == data["pod_id"]:
            print("VERIFICATION FAILED — pod is not new.", file=sys.stderr)
            sys.exit(3)
    print("VERIFICATION OK — pod is new.")


def cmd_sync(args):
    data = _load()
    if not data.get("pod_id"):
        print("No session. Run start first."); sys.exit(1)
    sync_code(data["ssh_host"], data["ssh_port"])


def cmd_launch(args):
    data = _load()
    if not data.get("pod_id"):
        print("No session."); sys.exit(1)
    script = args.script or "farkas_fast/run_pod.py"
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
    print("farkas_fast pod torn down.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_start = sub.add_parser("start")
    p_start.add_argument("--fresh", action="store_true",
                         help="Tear down any existing farkas_fast pod first.")
    p_start.add_argument("--force-reuse", action="store_true",
                         help="Reuse existing running pod without creating new.")
    sub.add_parser("verify_new")
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
        "start": cmd_start, "verify_new": cmd_verify_new,
        "sync": cmd_sync, "launch": cmd_launch,
        "status": cmd_status, "logs": cmd_logs, "fetch": cmd_fetch,
        "teardown": cmd_teardown,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
