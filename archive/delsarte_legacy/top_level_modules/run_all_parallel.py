"""Run the 4 MV-dual improvement scripts (B1, B2, B3, D2) IN PARALLEL.

Each script runs in its own subprocess so they make use of separate CPU cores
(the pod has 192).  Outputs stream to separate log files; a final summary is
printed.

Usage
-----
    python -m delsarte_dual.run_all_parallel
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    ("B1", "mv_multimoment.py"),
    ("B2", "mv_lemma217.py"),
    ("B3", "mv_union_forbidden.py"),
    ("D2", "mv_delta_u_scan.py"),
]


def main(log_dir: str = "/tmp/mv_improvements"):
    log_dir_p = Path(log_dir)
    log_dir_p.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).parent
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    # Limit BLAS threads per subprocess since we fan out at the subprocess level;
    # joblib inside individual scripts uses its own pools.
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")

    t0 = time.time()
    procs = []
    for tag, name in SCRIPTS:
        script_path = here / name
        log_path = log_dir_p / f"{tag}.log"
        print(f"[launch] {tag:3s} -> {name}  log: {log_path}")
        f = open(log_path, "w", encoding="utf-8")
        p = subprocess.Popen(
            [sys.executable, "-u", str(script_path)],
            stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(here),
        )
        procs.append((tag, name, p, f, log_path))

    print()
    print(f"[wait ] {len(procs)} processes launched; waiting for completion...")

    # Poll: print status every 20 s
    finished = {tag: False for tag, *_ in procs}
    while not all(finished.values()):
        time.sleep(20)
        elapsed = time.time() - t0
        line = f"[{elapsed:6.1f}s] "
        for tag, name, p, f, log_path in procs:
            if p.poll() is None:
                status = "running"
            else:
                status = f"done(rc={p.returncode})"
                finished[tag] = True
            line += f"{tag}={status}  "
        print(line, flush=True)

    # Close log file handles
    for tag, name, p, f, log_path in procs:
        f.close()

    total = time.time() - t0
    print()
    print(f"[done ] total wall time: {total:.1f}s")
    print()

    # Tail each log
    for tag, name, p, f, log_path in procs:
        print("=" * 72)
        print(f"Tail of {tag} ({name}):  {log_path}")
        print("=" * 72)
        try:
            text = log_path.read_text(encoding="utf-8")
        except Exception as err:
            print(f"  [could not read log: {err}]")
            continue
        lines = text.rstrip("\n").splitlines()
        for line in lines[-40:]:
            print("  " + line)
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="/tmp/mv_improvements")
    args = parser.parse_args()
    main(log_dir=args.log_dir)
