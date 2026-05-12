"""Persistent local backup for a running pod BnB job.

Uses paramiko (native Python SSH) to avoid Git-bash MSYS path mangling.

Pulls the pod's log every POLL_SECONDS into data/d16_salvage/,
writes a heartbeat, and takes a timestamped snapshot every ~20 min.
Runs forever until killed.

Usage (from project root):
    python scripts/pod_log_watcher.py POD_HOST POD_PORT REMOTE_LOG_PATH

Example:
    python scripts/pod_log_watcher.py 103.196.86.88 33611 \\
        /workspace/sidon-autocorrelation/data/d16_t1p285_32w.log
"""
from __future__ import annotations

import datetime as _dt
import os
import sys
import time
from pathlib import Path

import paramiko

POLL_SECONDS = 120
SSH_IDENTITY = str(Path.home() / ".ssh" / "id_ed25519")


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def fetch_once(host: str, port: int, remote: str, local: Path) -> tuple[bool, str]:
    """SSH to host, read remote file via SFTP, write to local."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(
            hostname=host, port=port, username="root",
            key_filename=SSH_IDENTITY, timeout=15,
        )
        sftp = ssh.open_sftp()
        try:
            with sftp.file(remote, "rb") as fh:
                data = fh.read()
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            return True, f"{len(data)} bytes"
        finally:
            sftp.close()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        ssh.close()


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    host = sys.argv[1]
    port = int(sys.argv[2])
    remote = sys.argv[3]

    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "data" / "d16_salvage"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_name = Path(remote).name
    current_local = out_dir / f"{log_name}.current"
    heartbeat = out_dir / "heartbeat.txt"
    error_log = out_dir / "errors.log"
    snapshot_dir = out_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    print(f"[watcher] host={host}:{port} remote={remote}")
    print(f"[watcher] local copy: {current_local}")
    print(f"[watcher] poll every {POLL_SECONDS}s")

    iteration = 0
    while True:
        iteration += 1
        ok, msg = fetch_once(host, port, remote, current_local)
        now = _now()
        heartbeat.write_text(f"{now}  iter={iteration}  ok={ok}\n")
        if ok:
            size = current_local.stat().st_size if current_local.exists() else 0
            print(f"[{now}] iter {iteration}: ok, {size} bytes", flush=True)
            if iteration % 10 == 1:
                snap_name = snapshot_dir / f"{log_name}.{now.replace(':', '-')}"
                try:
                    snap_name.write_bytes(current_local.read_bytes())
                except Exception as e:
                    print(f"[{now}] snapshot failed: {e}", flush=True)
        else:
            with error_log.open("a") as fh:
                fh.write(f"{now}  iter={iteration}  fail: {msg}\n")
            print(f"[{now}] iter {iteration}: FAIL ({msg})", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
