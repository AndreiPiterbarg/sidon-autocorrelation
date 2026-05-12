"""Pod benchmark v2 — optimized batch size, focused d=10/12/14 wall cases.

Skips long d=8 c=1.18+ cases (already-known to work). Focuses on cases that
inform whether c=1.28 at d=14 is reachable on the 64-core pod.
"""
import os
import sys
import time
import signal
import numba

# Force max threads
numba.set_num_threads(numba.config.NUMBA_DEFAULT_NUM_THREADS)
print(f"Numba threads: {numba.get_num_threads()}", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coarse_cascade_prover import run_cascade


class TimeoutError(Exception):
    pass


def _to_handler(signum, frame):
    raise TimeoutError("Hit timeout")


def time_cascade(c, S, d, timeout_s=600):
    signal.signal(signal.SIGALRM, _to_handler)
    signal.alarm(int(timeout_s))
    t = time.perf_counter()
    try:
        ok = run_cascade(c_target=c, S=S, d_start=d, max_levels=0, verbose=False)
        signal.alarm(0)
        return bool(ok), time.perf_counter() - t, False
    except TimeoutError:
        return False, time.perf_counter() - t, True
    except Exception as e:
        signal.alarm(0)
        print(f"  EXCEPTION: {e}")
        return False, time.perf_counter() - t, False


def main():
    # Each entry: (label, c, S, d, timeout_seconds)
    # We FOCUS on cases that test the d=14 wall:
    suite = [
        # Short anchor: confirm pod with optimized batch size
        ("d=8 c=1.16 S=101 (was 11s)", 1.16, 101, 8, 60),
        # d=10 frontier (val(10)=1.241, so c=1.20 has margin 0.04, 1.22 has 0.02)
        ("d=10 c=1.20 S=24", 1.20, 24, 10, 600),
        ("d=10 c=1.22 S=24", 1.22, 24, 10, 1800),
        # d=12 (val(12)=1.271; c=1.20 has margin 0.07, c=1.25 has 0.02)
        ("d=12 c=1.20 S=21", 1.20, 21, 12, 600),
        ("d=12 c=1.25 S=21", 1.25, 21, 12, 1800),
        # d=14 (val(14)=1.284; c=1.28 has margin 0.004 — VERY TIGHT)
        # Use small S to keep cell count manageable.
        # Cell count C(d+S-1, d-1)/2:
        ("d=14 c=1.20 S=18", 1.20, 18, 14, 1800),
        ("d=14 c=1.25 S=18", 1.25, 18, 14, 1800),
        # The actual goal: c=1.28 at d=14
        ("d=14 c=1.28 S=18 (target!)", 1.28, 18, 14, 3600),
        ("d=14 c=1.28 S=21 (target+)", 1.28, 21, 14, 7200),
    ]
    results = []
    print("Starting benchmark sweep...", flush=True)
    for label, c, S, d, timeout in suite:
        # Cell count estimate
        from math import comb
        try:
            cells = comb(S + d - 1, d - 1) // 2
        except Exception:
            cells = -1
        print(f"\n=== {label}  ({cells:,} canonical cells est.) ===", flush=True)
        ok, elapsed, did_to = time_cascade(c, S, d, timeout_s=timeout)
        status = "TIMEOUT" if did_to else ("PASS" if ok else "FAIL")
        results.append((label, c, S, d, status, elapsed))
        print(f"    {status} time={elapsed:.2f}s", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for label, c, S, d, status, elapsed in results:
        print(f"  {label:35s} {status:8s} {elapsed:8.1f}s", flush=True)


if __name__ == "__main__":
    main()
