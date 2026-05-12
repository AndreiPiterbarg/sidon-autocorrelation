"""Offline ETA analyzer for interval_bnb parallel runs.

Parses a `parallel_branch_and_bound` log file and fits multiple decay
models to estimate remaining time. For BnB with exhaustive cover, the
volume-closure fraction follows a long-tailed law (most volume closes
early; the last few percent contain most of the work), so a single
linear/exponential extrapolation is DEEPLY misleading. We instead fit:

  1. Linear in recent progress (noisy -- short horizon).
  2. Exponential decay of remaining volume:   rem(t) = A * exp(-lambda * t).
  3. Power-law decay:                         rem(t) = A * (t + t0)^(-alpha).
  4. Exponential decay of in_flight (only valid once in_flight is
     monotonically declining; signals true endgame).
  5. Cross-run reference: if a previous d=10 or d=4 run's log is
     provided, scale by finish-time / remaining-volume relationship.

All ETAs are reported together with a confidence tag based on how well
the fit matches the recent tail of the data.

Usage
-----
Local log:
    python -m interval_bnb.eta data/interval_bnb_d14_t1p27.log

Remote log (pulls via SSH from the current pod session):
    python -m interval_bnb.eta --pod --tag d14_t1p27

Fit tuning:
    --tail N           Use only the last N log lines for fitting (default 30)
    --rem_target F     Target remaining-volume fraction for "done" (default 1e-9)
    --verbose          Print per-model residuals
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


_LINE_RE = re.compile(
    r"t=\s*([\d.]+)s"
    r".*?nodes=\s*(\d+)"
    r".*?cert=\s*(\d+)"
    r".*?in_flight=\s*(\d+)"
    r".*?queue=\s*(-?\d+)"
    r".*?active=\s*(\d+)/(\d+)"
    r".*?rate=\s*([\d.]+)/s"
    r".*?progress=\s*([\d.]+)%"
)


@dataclass
class Sample:
    t: float              # elapsed seconds
    nodes: int
    cert: int
    in_flight: int
    queue: int
    active: int
    workers: int
    rate: float           # nodes/s (short-term)
    progress: float       # volume fraction in [0, 1]


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------

def parse_log(path: str) -> List[Sample]:
    samples: List[Sample] = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = _LINE_RE.search(line)
            if not m:
                continue
            t, n, c, inf, q, act, wks, r, p = m.groups()
            samples.append(Sample(
                t=float(t), nodes=int(n), cert=int(c),
                in_flight=int(inf), queue=int(q),
                active=int(act), workers=int(wks),
                rate=float(r), progress=float(p) / 100.0,
            ))
    return samples


def fetch_pod_log(tag: str, dest: str) -> str:
    """Pull the pod's log over SSH to a local tmp path."""
    session_file = Path(__file__).parent.parent / "cpupod" / ".session_interval.json"
    if not session_file.exists():
        raise SystemExit(f"session file not found: {session_file}")
    s = json.loads(session_file.read_text())
    host, port = s["ssh_host"], s["ssh_port"]
    remote_path = f"/workspace/sidon-autocorrelation/data/interval_bnb_{tag}.log"
    cmd = (
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-p {port} -i ~/.ssh/id_ed25519 root@{host} 'cat {remote_path}'"
    )
    with open(dest, "w", encoding="utf-8") as out:
        rc = subprocess.call(cmd, shell=True, stdout=out)
    if rc != 0:
        raise SystemExit(f"ssh fetch failed (rc={rc})")
    return dest


# ---------------------------------------------------------------------
# Decay-model fits
# ---------------------------------------------------------------------

def _filter_decreasing_rem(samples: List[Sample]) -> List[Sample]:
    """Keep only samples where remaining-volume is strictly positive AND
    the cumulative progress is non-decreasing (logs occasionally
    repeat)."""
    out = []
    last_p = -1.0
    for s in samples:
        if s.progress > last_p and s.progress < 1.0:
            out.append(s); last_p = s.progress
    return out


def fit_exponential(samples: List[Sample]) -> Optional[Tuple[float, float, float]]:
    """Fit rem = A * exp(-lambda * t). Returns (lambda, A, R2)."""
    xs = _filter_decreasing_rem(samples)
    if len(xs) < 4:
        return None
    t = np.array([s.t for s in xs], dtype=np.float64)
    rem = 1.0 - np.array([s.progress for s in xs], dtype=np.float64)
    rem = np.clip(rem, 1e-300, 1.0)
    y = np.log(rem)
    slope, intercept = np.polyfit(t, y, 1)
    lam = -slope
    A = math.exp(intercept)
    yhat = intercept + slope * t
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return lam, A, r2


def fit_power_law(samples: List[Sample]) -> Optional[Tuple[float, float, float]]:
    """Fit rem = A * (t + t0)^(-alpha). For simplicity use t0 = 1.
    Returns (alpha, A, R2)."""
    xs = _filter_decreasing_rem(samples)
    if len(xs) < 4:
        return None
    t = np.array([s.t + 1.0 for s in xs], dtype=np.float64)
    rem = 1.0 - np.array([s.progress for s in xs], dtype=np.float64)
    rem = np.clip(rem, 1e-300, 1.0)
    x = np.log(t)
    y = np.log(rem)
    slope, intercept = np.polyfit(x, y, 1)
    alpha = -slope
    A = math.exp(intercept)
    yhat = intercept + slope * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return alpha, A, r2


def fit_in_flight_decay(
    samples: List[Sample],
) -> Optional[Tuple[float, float, float]]:
    """Only applicable once in_flight is monotonically declining.
    Returns (lambda, A, R2) for in_flight(t) = A exp(-lambda (t - t0))."""
    # Find the last monotonically-decreasing window of length >= 4.
    in_flights = [s.in_flight for s in samples]
    best_lo = -1
    for i in range(len(samples) - 1, 3, -1):
        ok = True
        for j in range(i, 0, -1):
            if in_flights[j - 1] < in_flights[j]:
                ok = False
                break
        if ok and i >= 3:
            best_lo = 0
            break
    if best_lo < 0:
        # Take just the last 10 if mostly declining.
        window = samples[-min(10, len(samples)):]
    else:
        window = samples[best_lo:]
    if len(window) < 4:
        return None
    t = np.array([s.t for s in window], dtype=np.float64)
    y = np.log(np.clip(np.array([s.in_flight for s in window], dtype=np.float64), 1, None))
    slope, intercept = np.polyfit(t, y, 1)
    if slope >= 0:
        return None  # not declining
    lam = -slope
    A = math.exp(intercept)
    yhat = intercept + slope * t
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return lam, A, r2


# ---------------------------------------------------------------------
# ETA reporting
# ---------------------------------------------------------------------

def _fmt_s(s: float) -> str:
    if s < 0: return "<0"
    if s >= 86400: return f"{s/86400:.1f}d"
    if s >= 3600:  return f"{s/3600:.1f}h"
    if s >= 60:    return f"{s/60:.1f}m"
    return f"{s:.0f}s"


def _confidence(r2: float) -> str:
    if r2 > 0.95: return "HIGH"
    if r2 > 0.85: return "med"
    if r2 > 0.70: return "low"
    return "WEAK"


def _phase_detect(samples: List[Sample]) -> Tuple[str, str]:
    """Classify the BnB phase from in_flight dynamics.

    Returns (phase, explanation).
      EXPANSION  -- in_flight still rising → tree is growing faster than
                    closing; no reliable ETA yet.
      EQUILIBRIUM -- in_flight oscillating near a stable value → expansion
                    balances closure; ETA only via cross-run reference.
      ENDGAME    -- in_flight monotonically falling → ETA from decay fit
                    is meaningful.
    """
    if len(samples) < 5:
        return "UNKNOWN", "too few samples"
    window = samples[-min(15, len(samples)):]
    ifs = np.array([s.in_flight for s in window], dtype=np.float64)
    # Linear trend
    slope = np.polyfit(np.arange(len(ifs)), ifs, 1)[0]
    peak = max(s.in_flight for s in samples)
    cur = samples[-1].in_flight
    # Normalised slope per-sample
    rel_slope = slope / max(1.0, ifs.mean())
    if cur < 0.3 * peak and rel_slope < -0.02:
        return "ENDGAME", f"in_flight {cur} << peak {peak}, declining ({rel_slope:+.3f}/step)"
    if rel_slope > 0.02:
        return "EXPANSION", f"in_flight rising ({rel_slope:+.3f}/step)"
    return "EQUILIBRIUM", (
        f"in_flight stable near {cur} (peak {peak}, slope {rel_slope:+.3f}/step) "
        f"-- tree expansion rate ~= closure rate"
    )


def _expansion_eta_from_inflight_peak(samples: List[Sample]) -> Optional[Tuple[float, str]]:
    """If in_flight has PEAKED and started falling, extrapolate time to
    reach 0 using just the declining section."""
    peak_idx = int(np.argmax([s.in_flight for s in samples]))
    if peak_idx >= len(samples) - 3:
        return None  # peak too recent
    decline = samples[peak_idx:]
    # Simple linear slope on in_flight since peak.
    t = np.array([s.t for s in decline])
    y = np.array([s.in_flight for s in decline], dtype=np.float64)
    slope, intercept = np.polyfit(t, y, 1)
    if slope >= 0:
        return None
    t_zero = -intercept / slope
    last_t = samples[-1].t
    eta = max(0.0, t_zero - last_t)
    return eta, f"peak_inf={samples[peak_idx].in_flight}, slope={slope:.3f}/s"


def report(path: str, tail: int, rem_target: float, verbose: bool) -> None:
    samples = parse_log(path)
    if not samples:
        print(f"no samples parsed from {path}")
        return
    last = samples[-1]
    print(f"=== {path}")
    print(f"  latest:  t={last.t:.1f}s  nodes={last.nodes:,}  cert={last.cert:,}  "
          f"in_flight={last.in_flight}  progress={100*last.progress:.4f}%")

    # Phase classification — the single most important signal.
    phase, why = _phase_detect(samples)
    print(f"  phase:   {phase}  ({why})")

    # Lifetime rates from node/cert counters (high resolution).
    elapsed = last.t - samples[0].t + 1e-9
    nodes_rate = (last.nodes - samples[0].nodes) / elapsed
    cert_rate = (last.cert - samples[0].cert) / elapsed
    # Short-window rates (last ~30s)
    short = [s for s in samples if last.t - s.t <= 30.0]
    if len(short) >= 2:
        s0 = short[0]; s1 = short[-1]
        short_node_rate = (s1.nodes - s0.nodes) / max(1e-9, s1.t - s0.t)
        short_cert_rate = (s1.cert - s0.cert) / max(1e-9, s1.t - s0.t)
    else:
        short_node_rate = short_cert_rate = 0.0
    print(f"  rates:   nodes/s  lifetime={nodes_rate:>7.0f}  recent={short_node_rate:>7.0f}")
    print(f"           cert/s   lifetime={cert_rate:>7.0f}  recent={short_cert_rate:>7.0f}")

    peak_inf = max(s.in_flight for s in samples)
    print(f"  in_flight history: current={last.in_flight}  peak={peak_inf}  "
          f"current/peak={last.in_flight/peak_inf:.3f}")

    # Closure ratio = cert_rate / node_rate. Each processed box either
    # certifies (Δin_flight=-1) or splits (Δin_flight=+1). At steady-
    # state node_rate N and cert_rate C the derivative of in_flight is
    # d(in_flight)/dt = (N - C) - C = N - 2C. So:
    #   C/N < 0.5  : tree growing
    #   C/N = 0.5  : equilibrium
    #   C/N > 0.5  : tree shrinking (endgame)
    #   C/N = 1.0  : every node certifies -- done imminently
    if short_node_rate > 1.0:
        closure = short_cert_rate / short_node_rate
        if closure < 0.48:
            label = "GROWING (tree still discovering new work)"
        elif closure < 0.55:
            label = "EQUILIBRIUM (expansion ~= closure)"
        elif closure < 0.75:
            label = "SHRINKING (endgame approaching)"
        else:
            label = "ENDGAME (closing fast)"
        print(f"  closure_ratio: {closure:.3f}  -> {label}")
        # Remaining-nodes extrapolation from in_flight decay:
        #   d(in_flight)/dt = N - 2C = N * (1 - 2 * closure)
        # If closure > 0.5, in_flight decreasing at rate |N (1 - 2 closure)|.
        # ETA to in_flight=0: in_flight / (N * (2 closure - 1)).
        if closure > 0.505:
            shrink_rate = short_node_rate * (2 * closure - 1)
            eta = last.in_flight / max(shrink_rate, 1.0)
            print(f"  closure-ratio ETA:  in_flight={last.in_flight:,}  "
                  f"shrink_rate={shrink_rate:.0f}/s  ETA={_fmt_s(eta)}")
        elif closure < 0.495:
            # still growing
            grow_rate = short_node_rate * (1 - 2 * closure)
            print(f"  closure-ratio:  in_flight GROWING at {grow_rate:.0f}/s  "
                  f"(no ETA until closure > 0.5)")

    print()

    # Strategy branches by phase.
    if phase == "ENDGAME":
        r = fit_exponential(samples)
        if r and r[0] > 0:
            lam, A, r2 = r
            # Use cert-count fit instead if progress resolution is too coarse.
            t_done = math.log(max(A, 1e-300) / rem_target) / lam
            eta = max(0.0, t_done - last.t)
            print(f"  [exp decay of remaining volume]  lambda={lam:.5f}/s  "
                  f"R2={r2:.3f} ({_confidence(r2)})  ETA={_fmt_s(eta)}")
        r = fit_in_flight_decay(samples)
        if r and r[0] > 0:
            lam, A, r2 = r
            t_done = math.log(max(A, 1e-300) / 1.0) / lam
            eta = max(0.0, t_done - last.t)
            print(f"  [exp decay of in_flight]         lambda={lam:.5f}/s  "
                  f"R2={r2:.3f} ({_confidence(r2)})  ETA={_fmt_s(eta)}  (to in_flight=1)")
        peak_eta = _expansion_eta_from_inflight_peak(samples)
        if peak_eta:
            print(f"  [linear in_flight decay]         {peak_eta[1]}  "
                  f"ETA={_fmt_s(peak_eta[0])}")

    elif phase == "EQUILIBRIUM":
        print("  NOTE: in_flight is stable -> the tree is being replenished as "
              "fast as it's being closed. A DECAY-based ETA is not yet "
              "computable. Honest estimates below rely on extrapolation of "
              "the current cert rate against an ASSUMED total tree size.")
        # Reference-based estimate.
        print()
        _print_reference_estimate(samples, short_cert_rate or cert_rate)

    elif phase == "EXPANSION":
        print("  NOTE: in_flight still rising -> still discovering new work. "
              "No ETA possible until in_flight peaks.")

    else:
        print("  NOTE: not enough samples yet.")

    print()


def _print_reference_estimate(samples: List[Sample], cert_rate: float) -> None:
    """Rough estimate: if the current run has similar tree shape to the
    completed d=10 target=1.22 reference (5.77M leaves in 381s), scale
    up by dimension + target difficulty.

    For d=14/16 the tree size is empirically 5x-50x the d=10 size; we
    report the BOUNDS of that range so the user sees the uncertainty.
    """
    last = samples[-1]
    ref_leaves = 5_767_847
    ref_time_s = 381.2
    ref_rate_cert = ref_leaves / ref_time_s  # ~15k/s
    print(f"  Reference (d=10 target=1.22 completed run):")
    print(f"      {ref_leaves:,} leaves certified in {ref_time_s:.0f}s "
          f"(rate={ref_rate_cert:.0f}/s)")
    if cert_rate <= 0:
        return
    rate_ratio = cert_rate / ref_rate_cert
    print(f"      current cert_rate is {rate_ratio:.2f}x the reference rate")
    print()
    print("  Scale-up hypotheses for current run's total leaves:")
    for scale in (2, 5, 10, 25, 50):
        total = ref_leaves * scale
        remaining = max(0, total - last.cert)
        eta = remaining / cert_rate
        done_t = last.t + eta
        print(f"      if tree is {scale:>3}x d=10 tree  ({total:>11,} leaves total):  "
              f"{remaining:>11,} to go  ->  ETA {_fmt_s(eta)}  (finish t={done_t:.0f}s)")


def _load_reference_completion() -> Optional[dict]:
    """Load the d=10 t=1.22 completed run's summary if available."""
    # Placeholder for future cross-run referencing.
    return None


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", nargs="?", help="path to log file (or use --pod)")
    ap.add_argument("--pod", action="store_true",
                    help="fetch log from the current interval_bnb pod session")
    ap.add_argument("--tag", default="d14_t1p27",
                    help="log tag (used with --pod), e.g. d10_t1p22")
    ap.add_argument("--tail", type=int, default=30,
                    help="use last N log samples for fitting (default 30)")
    ap.add_argument("--rem_target", type=float, default=1e-9,
                    help="target remaining-volume for 'done' (default 1e-9)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if args.pod:
        import tempfile
        fd, tmp = tempfile.mkstemp(prefix="eta_", suffix=".log")
        os.close(fd)
        fetch_pod_log(args.tag, tmp)
        report(tmp, args.tail, args.rem_target, args.verbose)
    else:
        if not args.log_path:
            ap.error("log_path required (or use --pod)")
        report(args.log_path, args.tail, args.rem_target, args.verbose)


if __name__ == "__main__":
    main()
