"""S-sweep bench for the local v5 coarse cascade.

Goal:  identify the smallest S at which the coarse cascade closes
(survivors -> 0) at a cheap level (L1/L2/L3/L4) WITHOUT touching L3+
heavy work, for d0=2 and a target c.

For each S in a small grid the script invokes
`cloninger-steinerberger/cpu/run_cascade_coarse_v5.py` as a subprocess
with `--no_sdp`, an adaptive mode and a hard wall-clock cap (default
3 minutes).  Per-level survivor counts and timings are parsed from
the cascade's own log lines, plus the global wall time.

The sweep results are written to a JSON file
(default `_S_sweep_results.json`).  A short table is printed at the
end.

Usage:
    python _S_sweep_bench.py --c_target 1.281
    python _S_sweep_bench.py --c_target 1.281 --S_grid 20,25,30,35,40,45,50 --per_test_timeout 180
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

# -----------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------
HERE = os.path.abspath(os.path.dirname(__file__))
V5_PATH = os.path.join(HERE, "cloninger-steinerberger", "cpu",
                       "run_cascade_coarse_v5.py")

# Lines we drop from the subprocess output so the runtime parsing
# is robust (they're cvxpy / ortools warnings spammed at import time).
NOISE_RE = re.compile(r"(CVXPY|ortools|cvxpy)", re.IGNORECASE)


# -----------------------------------------------------------------
# Parsing helpers — extract per-level info from v5's stdout.
# -----------------------------------------------------------------
RE_L0_DONE = re.compile(
    r"L0:?\s*.*?\bsurv(?:ivors)?\s*=\s*([\d,]+)",
    re.IGNORECASE)
RE_L0_GENERIC_SURV = re.compile(
    r"L0\b.*?(\d[\d,]*)\s+survivor", re.IGNORECASE)

# v5 prints e.g. "  L1 done in 1.2s: 12,345 children -> 234 survivors (1.8901%)"
RE_LEVEL_DONE = re.compile(
    r"L(\d+)\s+done in\s+([\dhms\. ]+):\s+([\d,]+)\s+children\s+->\s+([\d,]+)\s+survivors")

# Closure announcement
RE_PROVEN = re.compile(r"GRID-POINT PROOF COMPLETE at L(\d+)")
RE_PROVEN_L0 = re.compile(r"PROVEN AT L0")
RE_NOT_PROVEN = re.compile(r"Cascade did not converge", re.IGNORECASE)

# v5 prints L0 stats slightly differently (handled by run_level0 from v4).
# Fall back: count survivors via the line "n_survivors=" or "X survivors at d="
RE_FINAL_SURV_AT_D = re.compile(
    r"\(([\d,]+)\s+survivors at d=", re.IGNORECASE)


def _strip_noise(line: str) -> str:
    return "" if NOISE_RE.search(line) else line


def _parse_int(s: str) -> int:
    return int(s.replace(",", ""))


# -----------------------------------------------------------------
# One run.
# -----------------------------------------------------------------
def run_one(S: int, c_target: float, d0: int, max_levels: int,
            n_workers: int, mode: str, tight_max_level: int,
            timeout_sec: float, python_exe: str) -> dict[str, Any]:
    """Spawn a single v5 cascade subprocess for this S; return parsed dict."""
    cmd = [
        python_exe, V5_PATH,
        "--d0", str(d0),
        "--S", str(S),
        "--c_target", str(c_target),
        "--max_levels", str(max_levels),
        "--mode", mode,
        "--tight_max_level", str(tight_max_level),
        "--n_workers", str(n_workers),
        "--no_sdp",
        "--no_checkpoint",
    ]

    record: dict[str, Any] = {
        "S": S,
        "c_target": c_target,
        "d0": d0,
        "mode": mode,
        "tight_max_level": tight_max_level,
        "max_levels": max_levels,
        "n_workers": n_workers,
        "timeout_sec": timeout_sec,
        "cmd": " ".join(cmd),
        "levels": [],          # list of {level, children, survivors, ...}
        "L0_survivors": None,  # filled when parseable
        "closed_at": None,     # "L<k>" or None
        "status": "started",
        "wall_time_sec": None,
        "stdout_tail": "",
    }

    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=HERE,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    captured: list[str] = []
    last_l0_surv: int | None = None
    closed_at: str | None = None
    timed_out = False

    try:
        while True:
            if proc.stdout is None:
                break
            line = proc.stdout.readline()
            if not line:
                # Process likely exited.
                if proc.poll() is not None:
                    break
                # Otherwise short sleep waiting for data.
                if time.time() - t0 > timeout_sec:
                    timed_out = True
                    break
                continue

            line = line.rstrip("\n")
            stripped = _strip_noise(line)
            if stripped:
                captured.append(stripped)

                # L0 survivors — try multiple patterns.
                m = RE_L0_DONE.search(stripped)
                if m:
                    last_l0_surv = _parse_int(m.group(1))
                else:
                    m = RE_L0_GENERIC_SURV.search(stripped)
                    if m:
                        last_l0_surv = _parse_int(m.group(1))

                # Per-level "L done in" lines.
                m = RE_LEVEL_DONE.search(stripped)
                if m:
                    L = int(m.group(1))
                    children = _parse_int(m.group(3))
                    survivors = _parse_int(m.group(4))
                    record["levels"].append({
                        "level": L,
                        "children": children,
                        "survivors": survivors,
                    })

                # Closure events.
                if RE_PROVEN_L0.search(stripped):
                    closed_at = "L0"
                m = RE_PROVEN.search(stripped)
                if m:
                    closed_at = f"L{m.group(1)}"
                if RE_NOT_PROVEN.search(stripped):
                    # Captures the survivor count remaining
                    pass

            # Check overall timeout
            if time.time() - t0 > timeout_sec:
                timed_out = True
                break
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass

    record["wall_time_sec"] = round(time.time() - t0, 2)
    record["L0_survivors"] = last_l0_surv
    record["closed_at"] = closed_at

    if timed_out:
        # Determine "stuck level" = first level after L0 that started but didn't finish.
        # If we have N completed levels (record["levels"] of length N) then the next
        # level (L_{N+1}) was the stuck one.
        completed_levels = {lv["level"] for lv in record["levels"]}
        # The first not-yet-completed level after L0.
        stuck_at = 1
        while stuck_at in completed_levels:
            stuck_at += 1
        last_surv = (record["levels"][-1]["survivors"]
                     if record["levels"] else last_l0_surv)
        record["status"] = "timeout"
        record["stuck_at"] = f"L{stuck_at}"
        record["stuck_with_survivors"] = last_surv
    elif closed_at is not None:
        record["status"] = "closed"
    else:
        # Process finished without closure (rare unless max_levels too small).
        record["status"] = "didnotclose"

    record["stdout_tail"] = "\n".join(captured[-30:])
    return record


# -----------------------------------------------------------------
# Driver
# -----------------------------------------------------------------
def format_table(results: list[dict[str, Any]]) -> str:
    rows = []
    rows.append(
        f"{'S':>4}  {'L0':>6}  {'L1':>8}  {'L2':>10}  {'L3':>10}  "
        f"{'L4':>10}  {'closes':>8}  {'wall(s)':>9}  status"
    )
    rows.append("-" * 80)
    for r in sorted(results, key=lambda x: x["S"]):
        S = r["S"]
        l0 = r.get("L0_survivors")
        per_L = {lv["level"]: lv["survivors"] for lv in r["levels"]}
        l1 = per_L.get(1, "-")
        l2 = per_L.get(2, "-")
        l3 = per_L.get(3, "-")
        l4 = per_L.get(4, "-")
        closes = r.get("closed_at") or "-"
        if r["status"] == "timeout":
            closes = f"TO@{r.get('stuck_at')}"
        wall = r["wall_time_sec"]
        status = r["status"]
        rows.append(
            f"{S:>4}  {str(l0):>6}  {str(l1):>8}  {str(l2):>10}  "
            f"{str(l3):>10}  {str(l4):>10}  {closes:>8}  {wall:>9.1f}  {status}"
        )
    return "\n".join(rows)


def recommend_S(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the smallest S that closed cheaply."""
    candidates = []
    for r in sorted(results, key=lambda x: x["S"]):
        if r["status"] != "closed":
            continue
        ca = r["closed_at"] or ""
        # Rank: lowest closure level wins; tiebreak on smaller S.
        try:
            level = int(ca.replace("L", ""))
        except Exception:
            level = 99
        candidates.append((level, r["S"], r))
    if not candidates:
        # Fallback: prefer the run with the fewest L2 survivors (most progress).
        best = None
        for r in sorted(results, key=lambda x: x["S"]):
            per_L = {lv["level"]: lv["survivors"] for lv in r["levels"]}
            l2 = per_L.get(2)
            if l2 is None:
                continue
            if best is None or l2 < best[0]:
                best = (l2, r)
        if best:
            return {"S": best[1]["S"],
                    "rationale": f"no closure; smallest L2 survivors ({best[0]})"}
        return {"S": None, "rationale": "no usable result"}
    candidates.sort(key=lambda x: (x[0], x[1]))
    level, S, r = candidates[0]
    return {"S": S, "closes_at": f"L{level}",
            "rationale": "smallest S that closes; lowest closure level"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_target", type=float, default=1.281)
    ap.add_argument("--d0", type=int, default=2)
    ap.add_argument("--max_levels", type=int, default=6)
    ap.add_argument("--n_workers", type=int, default=4)
    ap.add_argument("--mode", default="adaptive",
                    choices=["adaptive", "tight", "fast"])
    ap.add_argument("--tight_max_level", type=int, default=2)
    ap.add_argument("--per_test_timeout", type=float, default=180.0,
                    help="hard cap per S (seconds); default 3 min")
    ap.add_argument("--total_budget_sec", type=float, default=300.0,
                    help="overall sweep wall-clock budget; default 5 min")
    ap.add_argument("--S_grid", default="20,25,30,35,40,45,50",
                    help="comma list of S values to sweep")
    ap.add_argument("--out", default=os.path.join(HERE,
                                                  "_S_sweep_results.json"))
    ap.add_argument("--python", default=sys.executable,
                    help="python interpreter for the subprocess")
    args = ap.parse_args()

    S_grid = [int(s.strip()) for s in args.S_grid.split(",") if s.strip()]
    print(f"[sweep] S_grid={S_grid} c_target={args.c_target} "
          f"d0={args.d0} mode={args.mode} max_levels={args.max_levels}")
    print(f"[sweep] per-test timeout: {args.per_test_timeout:.0f}s   "
          f"total budget: {args.total_budget_sec:.0f}s")
    print(f"[sweep] using v5 at: {V5_PATH}")
    print(f"[sweep] writing results to: {args.out}")

    results = []
    t_start = time.time()
    for S in S_grid:
        elapsed = time.time() - t_start
        remaining = args.total_budget_sec - elapsed
        if remaining <= 5.0:
            print(f"[sweep] global budget exhausted "
                  f"({elapsed:.0f}s / {args.total_budget_sec:.0f}s); stopping early")
            break
        per_timeout = min(args.per_test_timeout, max(15.0, remaining))
        print(f"\n[sweep] === S={S}  per_test_timeout={per_timeout:.0f}s "
              f"(global remaining {remaining:.0f}s) ===")
        rec = run_one(
            S=S, c_target=args.c_target, d0=args.d0,
            max_levels=args.max_levels, n_workers=args.n_workers,
            mode=args.mode, tight_max_level=args.tight_max_level,
            timeout_sec=per_timeout, python_exe=args.python,
        )
        per_L = {lv["level"]: lv["survivors"] for lv in rec["levels"]}
        print(f"[sweep] S={S}  status={rec['status']}  "
              f"closed_at={rec.get('closed_at')}  "
              f"L0={rec.get('L0_survivors')}  L1={per_L.get(1)}  "
              f"L2={per_L.get(2)}  L3={per_L.get(3)}  "
              f"wall={rec['wall_time_sec']:.1f}s")
        results.append(rec)
        # Persist incrementally so a kill won't lose everything.
        try:
            with open(args.out, "w") as f:
                json.dump({
                    "results": results,
                    "params": {
                        "c_target": args.c_target, "d0": args.d0,
                        "mode": args.mode,
                        "tight_max_level": args.tight_max_level,
                        "max_levels": args.max_levels,
                        "n_workers": args.n_workers,
                        "per_test_timeout": args.per_test_timeout,
                    },
                }, f, indent=2)
        except Exception as e:
            print(f"[sweep] WARN: failed to persist: {e}")

    print("\n" + "=" * 80)
    print("S-SWEEP RESULTS")
    print("=" * 80)
    print(format_table(results))

    rec = recommend_S(results)
    print("\n" + "=" * 80)
    print(f"RECOMMENDATION: S={rec.get('S')}  "
          f"closes_at={rec.get('closes_at', '?')}  "
          f"({rec.get('rationale', '?')})")
    print("=" * 80)
    print(f"\n[sweep] full results JSON: {args.out}")


if __name__ == "__main__":
    main()
