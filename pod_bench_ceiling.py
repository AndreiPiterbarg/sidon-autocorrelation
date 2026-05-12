"""Benchmark sweep on pod to determine the ceiling for c=1.28 proof.

Runs a sequence of cascades at increasing (d, c, S) with HARD timeout per case.
Reports time + success for each.
"""
import os
import sys
import time
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coarse_cascade_prover import run_cascade


class TimeoutError(Exception):
    pass


def _to_handler(signum, frame):
    raise TimeoutError("Hit timeout")


def time_cascade(c_target, S, d_start, timeout_s=600, max_levels=0):
    """Run cascade with timeout. Returns (ok, elapsed, did_timeout)."""
    signal.signal(signal.SIGALRM, _to_handler)
    signal.alarm(int(timeout_s))
    t = time.perf_counter()
    try:
        ok = run_cascade(c_target=c_target, S=S, d_start=d_start,
                         max_levels=max_levels, verbose=False)
        signal.alarm(0)
        return bool(ok), time.perf_counter() - t, False
    except TimeoutError:
        return False, time.perf_counter() - t, True
    except Exception as e:
        signal.alarm(0)
        return False, time.perf_counter() - t, False


def main():
    # Each entry: (label, c_target, S, d_start, timeout_seconds)
    suite = [
        # Anchor (laptop-validated)
        ("d=8 c=1.16 S=101 (laptop:19s)", 1.16, 101, 6, 120),
        ("d=8 c=1.17 S=51 (laptop:68s)", 1.17, 51, 8, 300),
        # Push past laptop ceiling
        ("d=8 c=1.18 S=76 (laptop:1156s)", 1.18, 76, 8, 1800),
        # Targets toward val(8)=1.205
        ("d=8 c=1.19 S=101", 1.19, 101, 8, 3600),
        # Higher d
        ("d=10 c=1.20 S=21", 1.20, 21, 10, 1800),
        ("d=10 c=1.22 S=24", 1.22, 24, 10, 1800),
        # Toward val(12)=1.271
        ("d=12 c=1.20 S=18", 1.20, 18, 12, 1800),
        ("d=12 c=1.22 S=18", 1.22, 18, 12, 1800),
        ("d=12 c=1.25 S=21", 1.25, 21, 12, 3600),
        # Approach val(14)=1.284 (where 1.28 lives)
        ("d=14 c=1.20 S=18", 1.20, 18, 14, 3600),
        ("d=14 c=1.25 S=21", 1.25, 21, 14, 3600),
        ("d=14 c=1.28 S=21", 1.28, 21, 14, 7200),
    ]
    results = []
    for label, c, S, d, timeout in suite:
        print(f"\n=== {label} ===", flush=True)
        ok, elapsed, did_to = time_cascade(c, S, d, timeout_s=timeout)
        status = "TIMEOUT" if did_to else ("PASS" if ok else "FAIL")
        results.append((label, c, S, d, status, elapsed))
        print(f"    {status} time={elapsed:.2f}s", flush=True)
        # If we timeout or hit a wall, no point trying harder cases at same d
        if did_to:
            print(f"    -- timeout reached, skipping further harder tests at d={d}",
                  flush=True)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, c, S, d, status, elapsed in results:
        print(f"  {label:35s} {status:8s} {elapsed:8.1f}s")


if __name__ == "__main__":
    main()
