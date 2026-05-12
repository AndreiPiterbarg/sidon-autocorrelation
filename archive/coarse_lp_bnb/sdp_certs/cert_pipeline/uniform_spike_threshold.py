"""Step 1 of "Active-Window Saddle-KKT Exclusion": uniform-K-spike enumeration.

For a box B = [0, h]^d cap Delta_d with h = 1/N, the *vertices* of B are
exactly the uniform-K-spike measures mu_S = (1/K) * 1_S with S subset [d],
|S| = K, K >= N. (For h = 1/N exact, K in {N-1, N} both reduce to uniform-N
on N axes; for h = 1/2^k boxes the same vertex characterisation holds.)

This script enumerates ALL uniform-K-spikes for K = 1..d and computes

    val(mu_S) := max_W TV_W(mu_S)
              = max_W (2d/ell) * mu_S^T A_W mu_S
              = max_W (2d/ell) * count_W(S) / K^2

where count_W(S) = |{(i, j) in S x S : s_lo <= i + j <= s_lo + ell - 2}|.

For each K it tracks the *minimum* over S of val(mu_S) (the worst case, since
val_B for the box hosting K-spikes is at most this minimum).  It then finds

    N* := smallest K such that for ALL K' in [K, d],
          ALL uniform-K'-spikes have val >= TARGET + MARGIN.

If N* exists, then any box B = [0, 1/N]^d cap Delta_d with N >= N* is
*certified* against uniform-K-spike vertices: every vertex of B has
val >= target + margin.  The remaining task (handled by Step 2 of the
algorithm, i.e. Krawczyk on the saddle-KKT system) is only the non-vertex
KKT points.

Verification is done in EXACT integer arithmetic:
    val(mu_S, W) = (2d * count_W) / (ell * K^2)
    val(mu_S, W) >= target + margin
        iff 2d * count_W * margin_den >= ell * K^2 * threshold_num
where threshold = (target + margin) = threshold_num / margin_den.

Usage:
    python -m cert_pipeline.uniform_spike_threshold --d 22 --target 1.2805 \
        --margin 1e-6 --output runs/uniform_spike_d22.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------
# Window index
# ---------------------------------------------------------------------

def build_window_index(d: int):
    """Return (ells, slos, lower_idx, upper_idx) as int64 arrays.

    Window W = (ell, s_lo) with 2 <= ell <= 2d, 0 <= s_lo <= 2d - ell.
    Pair-sums in window W are s in [s_lo, s_lo + ell - 2] (length ell - 1).
    Using prefix sum P[k] = sum_{s < k} H[s], count_W = P[s_lo + ell - 1]
    - P[s_lo].
    """
    ells = []
    slos = []
    for ell in range(2, 2 * d + 1):
        for s in range(2 * d - ell + 1):
            ells.append(ell)
            slos.append(s)
    ells_a = np.array(ells, dtype=np.int64)
    slos_a = np.array(slos, dtype=np.int64)
    upper_idx = slos_a + ells_a - 1
    lower_idx = slos_a
    return ells_a, slos_a, lower_idx, upper_idx


# ---------------------------------------------------------------------
# Floating-point fast path (used for argmin search per K)
# ---------------------------------------------------------------------

def val_max_float(S, K: int, d: int,
                  ells: np.ndarray, lower_idx: np.ndarray,
                  upper_idx: np.ndarray) -> float:
    """val(mu_S) = max_W TV_W(mu_S) computed in float64. Exact for our
    integer-ratio values when count <= 2d^2 and ell*K^2 < 2^53."""
    v = np.zeros(d, dtype=np.int64)
    v[list(S)] = 1
    H = np.convolve(v, v)
    P = np.zeros(2 * d, dtype=np.int64)
    P[1:] = np.cumsum(H)
    counts = P[upper_idx] - P[lower_idx]
    vals = (2.0 * d) * counts / (ells * (K * K))
    return float(vals.max())


# ---------------------------------------------------------------------
# Exact integer arithmetic for verification
# ---------------------------------------------------------------------

def counts_for_S(S, d: int, lower_idx: np.ndarray,
                 upper_idx: np.ndarray) -> np.ndarray:
    """Return int64 array of count_W(S) for every window."""
    v = np.zeros(d, dtype=np.int64)
    v[list(S)] = 1
    H = np.convolve(v, v)
    P = np.zeros(2 * d, dtype=np.int64)
    P[1:] = np.cumsum(H)
    return P[upper_idx] - P[lower_idx]


def val_max_exact(S, K: int, d: int,
                  ells: np.ndarray, lower_idx: np.ndarray,
                  upper_idx: np.ndarray) -> Tuple[Fraction, int, int, int]:
    """Return (max val as Fraction, argmax window index, count, ell) using
    cross-multiplication: argmax of count/ell. Exact integer comparison."""
    counts = counts_for_S(S, d, lower_idx, upper_idx)
    # argmax of count / ell via integer comparison: we compare
    # count_a * ell_b vs count_b * ell_a, both fit in int64.
    n = counts.shape[0]
    best_idx = 0
    best_c = int(counts[0])
    best_e = int(ells[0])
    for k in range(1, n):
        c = int(counts[k])
        e = int(ells[k])
        # c/e > best_c/best_e iff c * best_e > best_c * e
        if c * best_e > best_c * e:
            best_idx = k
            best_c = c
            best_e = e
    val = Fraction(2 * d * best_c, best_e * K * K)
    return val, best_idx, best_c, best_e


def certify_S_above_threshold(S, K: int, d: int,
                              ells: np.ndarray, lower_idx: np.ndarray,
                              upper_idx: np.ndarray,
                              num: int, den: int) -> Tuple[bool, int, int, int]:
    """Return (certified, best_idx, best_count, best_ell) where certified
    is true iff there exists W with (2d * count_W) * den >= num * (ell * K^2),
    i.e., val(mu_S) >= num/den.  Uses pure integer arithmetic.

    Search greedily by best count/ell first; once a window meets the
    threshold we early-exit.
    """
    counts = counts_for_S(S, d, lower_idx, upper_idx)
    n = counts.shape[0]
    K2 = K * K
    # Quick pass: any single W satisfies threshold?
    # val_W >= num/den
    #   iff (2d * count) * den >= num * (ell * K^2)
    twoD = 2 * d
    rhs_factor = num * K2
    found_idx = -1
    found_c = 0
    found_e = 0
    # Track the largest count/ell as fallback for reporting.
    best_idx = 0
    best_c = int(counts[0])
    best_e = int(ells[0])
    for k in range(n):
        c = int(counts[k])
        e = int(ells[k])
        # Check threshold
        if twoD * c * den >= rhs_factor * e:
            found_idx = k
            found_c = c
            found_e = e
            break
        # Update best (for reporting on failure)
        if c * best_e > best_c * e:
            best_idx = k
            best_c = c
            best_e = e
    if found_idx >= 0:
        return True, found_idx, found_c, found_e
    return False, best_idx, best_c, best_e


# ---------------------------------------------------------------------
# Threshold parsing
# ---------------------------------------------------------------------

def parse_decimal(s: str) -> Fraction:
    """Parse a decimal string like '1.2805' or '1e-6' into an exact Fraction."""
    s = s.strip().lower()
    if "e" in s:
        m, ex = s.split("e")
        mant = parse_decimal(m) if "." in m or m.startswith("-") else Fraction(int(m))
        if "." in m and not m.startswith("-"):
            i, f = m.split(".")
            mant = Fraction(int((i if i else "0") + f), 10 ** len(f))
        elif "." in m:
            sign = -1 if m.startswith("-") else 1
            mm = m.lstrip("-")
            i, f = mm.split(".")
            mant = sign * Fraction(int((i if i else "0") + f), 10 ** len(f))
        else:
            mant = Fraction(int(m))
        ex_i = int(ex)
        return mant * (Fraction(10) ** ex_i)
    if "." in s:
        sign = -1 if s.startswith("-") else 1
        ss = s.lstrip("-")
        i, f = ss.split(".")
        return sign * Fraction(int((i if i else "0") + f), 10 ** len(f))
    return Fraction(int(s))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=22)
    p.add_argument("--target", type=str, default="1.2805")
    p.add_argument("--margin", type=str, default="1e-6")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--quick", action="store_true",
                   help="Skip K with C(d,K) > 1e6 (testing).")
    args = p.parse_args(argv)

    D = args.d
    target_rat = parse_decimal(args.target)
    margin_rat = parse_decimal(args.margin)
    threshold = target_rat + margin_rat
    th_num = threshold.numerator
    th_den = threshold.denominator

    print(f"=== uniform-K-spike threshold enumeration ===")
    print(f"d        = {D}")
    print(f"target   = {target_rat} = {float(target_rat):.10f}")
    print(f"margin   = {margin_rat} = {float(margin_rat):.2e}")
    print(f"threshold= {threshold} = {float(threshold):.10f}")
    print(f"         = {th_num} / {th_den}")
    print()

    ells, slos, lower_idx, upper_idx = build_window_index(D)
    n_W = len(ells)
    print(f"Windows: {n_W}")
    print()

    print(f"{'K':>3} | {'subsets':>10} | {'min val':>14} | {'cert':>4} | "
          f"{'wall (s)':>9} | worst-S")
    print("-" * 90)

    results = {}
    total_t0 = time.time()

    for K in range(1, D + 1):
        # C(D, K)
        n_subs = 1
        for k in range(K):
            n_subs = n_subs * (D - k) // (k + 1)
        if args.quick and n_subs > 1_000_000:
            print(f"{K:>3} | {n_subs:>10d} | (skipped --quick)")
            results[K] = None
            continue

        t0 = time.time()
        min_val_f = float("inf")
        worst_S = None

        # Enumerate
        for S in combinations(range(D), K):
            v = np.zeros(D, dtype=np.int64)
            v[list(S)] = 1
            H = np.convolve(v, v)
            P = np.zeros(2 * D, dtype=np.int64)
            P[1:] = np.cumsum(H)
            counts = P[upper_idx] - P[lower_idx]
            vals = (2.0 * D) * counts / (ells * (K * K))
            mv = float(vals.max())
            if mv < min_val_f:
                min_val_f = mv
                worst_S = S

        # Exact verification of worst_S against the threshold
        cert_ok, _, c_, e_ = certify_S_above_threshold(
            worst_S, K, D, ells, lower_idx, upper_idx,
            th_num, th_den,
        )
        # Also compute exact value for reporting
        exact_val, _, _, _ = val_max_exact(worst_S, K, D, ells, lower_idx, upper_idx)

        elapsed = time.time() - t0

        results[K] = {
            "min_val_float": min_val_f,
            "exact_val": exact_val,
            "worst_S": worst_S,
            "certified": cert_ok,
            "n_subsets": n_subs,
            "wall_s": elapsed,
        }
        ver = "PASS" if cert_ok else "FAIL"
        ws_str = str(list(worst_S)) if K <= 8 else f"len={K}, ..."
        print(f"{K:>3} | {n_subs:>10d} | {float(exact_val):>14.10f} | "
              f"{ver:>4} | {elapsed:>9.2f} | {ws_str}")

    total_wall = time.time() - total_t0
    print()
    print(f"Total wall: {total_wall:.1f}s")
    print()

    # Find N*: smallest K such that for all K' in [K, D], certified.
    N_star = None
    for K in range(D, 0, -1):
        if results.get(K) is None:
            continue
        if results[K]["certified"]:
            N_star = K
        else:
            break

    if N_star is None:
        print("=== N* DOES NOT EXIST ===")
        print("Some uniform-K-spike for K' >= 1 fails the threshold.")
        # The smallest K that fails dominates the problem.
        for K in sorted(results):
            r = results[K]
            if r is None:
                continue
            if not r["certified"]:
                print(f"First failing K: {K}, exact val = {float(r['exact_val']):.10f} "
                      f"= {r['exact_val']}, worst_S = {r['worst_S']}")
                break
    else:
        print(f"=== N* = {N_star} ===")
        if N_star == 1:
            print("All uniform-K-spikes for K = 1..d certify val >= threshold.")
        else:
            print(f"All uniform-K-spikes with K in [{N_star}, {D}] certify "
                  f"val >= threshold = {float(threshold):.10f}.")
            print(f"Boxes B = [0, h]^d cap Delta_d with h <= 1/{N_star} are vertex-certified.")
            # Show first failing K (just below N_star)
            if N_star > 1 and results.get(N_star - 1) is not None and not results[N_star - 1]["certified"]:
                r = results[N_star - 1]
                print(f"\nFirst failing K below N*: K = {N_star - 1}")
                print(f"  worst S      = {r['worst_S']}")
                print(f"  exact val    = {r['exact_val']} = {float(r['exact_val']):.10f}")
                print(f"  margin to thr= {float(r['exact_val'] - threshold):+.2e}")

    if args.output:
        out = {
            "d": D,
            "target": str(target_rat),
            "margin": str(margin_rat),
            "threshold": str(threshold),
            "threshold_float": float(threshold),
            "N_star": N_star,
            "n_windows": int(n_W),
            "total_wall_s": total_wall,
            "per_K": {
                str(K): None if r is None else {
                    "min_val_float": r["min_val_float"],
                    "exact_val": str(r["exact_val"]),
                    "exact_val_float": float(r["exact_val"]),
                    "worst_S": list(r["worst_S"]) if r["worst_S"] is not None else None,
                    "certified": bool(r["certified"]),
                    "n_subsets": int(r["n_subsets"]),
                    "wall_s": float(r["wall_s"]),
                }
                for K, r in results.items()
            },
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
