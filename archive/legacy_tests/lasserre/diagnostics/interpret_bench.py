#!/usr/bin/env python
"""Read bench_results.mat from sdpnal_bench.m and extrapolate to d=16.

Fits wall-time scaling from (d=4, d=6) data points, projects d=8 L3 and
d=16 L3 full on the same laptop. Prints a GO / NO-GO verdict for running
d=16 L3 full overnight.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.io as sio


# Problem sizes per config (A: rows, cols, nnz; n_psd_max).
# These come from the Python build log — hard-coded here to avoid
# re-running the precompute during interpretation.
_SIZES = {
    'd4': {
        'n_psd_max': 35,       'n_y': 70,       'm': 5475,
        'ncols': 5459,         'nnz': 27_537,
    },
    'd6': {
        'n_psd_max': 84,       'n_y': 924,      'm': 37_213,
        'ncols': 37_223,       'nnz': 329_883,
    },
    'd8': {   # predicted (not run yet)
        'n_psd_max': 165,      'n_y': 3_003,    'm': 150_000,  # rough
        'ncols': 150_000,      'nnz': 1_600_000,
    },
    'd16': {  # predicted (not run yet)
        'n_psd_max': 969,      'n_y': 74_613,   'm': 975_000,
        'ncols': 975_000,      'nnz': 17_000_000,
    },
}


def main():
    bench_path = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'data', 'sdpnal_bench', 'bench_results.mat'))

    if not os.path.exists(bench_path):
        print(f"ERROR: {bench_path} not found.")
        print("Run the MATLAB script first: sdpnal_bench (inside MATLAB)")
        return 1

    data = sio.loadmat(bench_path, squeeze_me=True, struct_as_record=False)
    results = data['results']

    # MATLAB struct fields are top-level in `results` after squeeze_me.
    # Pull the two cases.
    d4 = getattr(results, 'bench_d4_l3_bw3', None)
    d6 = getattr(results, 'bench_d6_l3_bw5', None)
    if d4 is None or d6 is None:
        print("ERROR: bench_results.mat is missing expected fields.")
        return 1

    t4, t6 = float(d4.wall_s), float(d6.wall_s)
    mem4_mb = float(d4.mem_matlab_MB)
    mem6_mb = float(d6.mem_matlab_MB)
    phys_total_gb = float(d6.phys_total_GB)
    phys_avail_gb = float(d6.phys_avail_GB)
    iter4 = int(d4.iter_main)
    iter6 = int(d6.iter_main)
    term4 = int(d4.termcode)
    term6 = int(d6.termcode)
    kkt4 = float(d4.kkt)
    kkt6 = float(d6.kkt)

    print("=" * 66)
    print("SDPNAL+ BENCHMARK RESULTS")
    print("=" * 66)
    print(f"{'config':<18} {'t_solve':>10} {'iters':>7} {'KKT':>10} {'term':>6} {'mem_MB':>9}")
    print("-" * 66)
    print(f"{'d=4 L3 bw=3':<18} {t4:>9.2f}s {iter4:>7d} {kkt4:>10.2e} {term4:>6d} {mem4_mb:>9.1f}")
    print(f"{'d=6 L3 bw=5':<18} {t6:>9.2f}s {iter6:>7d} {kkt6:>10.2e} {term6:>6d} {mem6_mb:>9.1f}")
    print(f"\nPhysical memory: {phys_avail_gb:.1f} / {phys_total_gb:.1f} GB available")

    # ------------------------------------------------------------
    # Fit scaling: assume wall_time = C * (n_psd_max ** p)
    # p is typically 2.5 to 3.0 for IPM-class solvers on SDPs where
    # the dominant block is close to dense.
    # ------------------------------------------------------------
    npsd4 = _SIZES['d4']['n_psd_max']
    npsd6 = _SIZES['d6']['n_psd_max']
    npsd8 = _SIZES['d8']['n_psd_max']
    npsd16 = _SIZES['d16']['n_psd_max']

    # Regress on log scale: log t = log C + p * log n_psd
    logs = np.log([npsd4, npsd6])
    logt = np.log([max(t4, 0.01), max(t6, 0.01)])
    p = (logt[1] - logt[0]) / (logs[1] - logs[0])
    logC = logt[0] - p * logs[0]
    C = np.exp(logC)
    print()
    print(f"Fitted scaling: wall = {C:.2e} * n_psd_max ^ {p:.2f}")

    # Also fit vs nnz (alternative scaling — linear algebra cost).
    nnz4 = _SIZES['d4']['nnz']
    nnz6 = _SIZES['d6']['nnz']
    nnz_logs = np.log([nnz4, nnz6])
    p_nnz = (logt[1] - logt[0]) / (nnz_logs[1] - nnz_logs[0])
    print(f"Alternative:    wall = C * nnz ^ {p_nnz:.2f}")

    # Project wall times.
    t8_proj = C * (npsd8 ** p)
    t16_proj = C * (npsd16 ** p)

    # Bound the projection using nnz-scaling too (usually more conservative).
    t8_nnz_proj = np.exp(logt[1]) * (nnz_logs_ratio_safe(
        _SIZES['d8']['nnz'], _SIZES['d6']['nnz']) ** p_nnz) \
        if p_nnz > 0 else t8_proj
    t16_nnz_proj = np.exp(logt[1]) * (nnz_logs_ratio_safe(
        _SIZES['d16']['nnz'], _SIZES['d6']['nnz']) ** p_nnz) \
        if p_nnz > 0 else t16_proj

    # Memory projection: scale linearly with n_psd_max^2 (dominant block
    # storage) and with nnz (sparse structures).
    mem16_psd = mem6_mb * (npsd16 / npsd6) ** 2
    mem16_nnz = mem6_mb * (_SIZES['d16']['nnz'] / nnz6)
    mem16_est = max(mem16_psd, mem16_nnz) / 1024.0  # to GB

    print()
    print("PROJECTIONS (high uncertainty with only 2 data points):")
    print(f"  d=8 L3 full:   ~{_fmt_time(t8_proj)}  (n_psd scaling)")
    print(f"                 ~{_fmt_time(t8_nnz_proj)}  (nnz scaling)")
    print(f"  d=16 L3 full:  ~{_fmt_time(t16_proj)}  (n_psd scaling)")
    print(f"                 ~{_fmt_time(t16_nnz_proj)}  (nnz scaling)")
    print(f"  d=16 memory:   ~{mem16_est:.1f} GB estimated peak")

    # Verdict for running d=16 locally overnight.
    wall_estimate_s = max(t16_proj, t16_nnz_proj)
    wall_h = wall_estimate_s / 3600.0

    print()
    print("=" * 66)
    print("VERDICT for d=16 L3 full on this laptop:")
    print("=" * 66)
    if mem16_est > 0.85 * phys_avail_gb:
        print(f"  MEMORY RISK: projected {mem16_est:.1f} GB vs "
              f"{phys_avail_gb:.1f} GB available.")
        print("  → likely OOM on this machine. Use a pod or SEAS server.")
    elif wall_h > 24:
        print(f"  TIME: ~{wall_h:.1f} hours — longer than one overnight run.")
        print("  → feasible but consider pod for faster turnaround.")
    elif wall_h > 8:
        print(f"  TIME: ~{wall_h:.1f} hours — one overnight run.")
        print("  → feasible locally. Start before bed, check in the morning.")
    elif wall_h > 2:
        print(f"  TIME: ~{wall_h:.1f} hours — afternoon run.")
        print("  → comfortable locally. Run it.")
    else:
        print(f"  TIME: ~{wall_h*60:.0f} minutes — trivial.")
        print("  → just run it.")
    print()
    print("Before the d=16 run, still do d=8 L3 full pilot (~%.1f min)"
          % (t8_proj / 60.0))
    print("  — binary gate on whether L3 retains tightness at d=8.")

    return 0


def nnz_logs_ratio_safe(a, b):
    if b <= 0:
        return 1.0
    return a / b


def _fmt_time(sec):
    if sec < 60:
        return f"{sec:.1f} s"
    if sec < 3600:
        return f"{sec/60:.1f} min"
    if sec < 86400:
        return f"{sec/3600:.1f} hr"
    return f"{sec/86400:.1f} day"


if __name__ == '__main__':
    sys.exit(main())
