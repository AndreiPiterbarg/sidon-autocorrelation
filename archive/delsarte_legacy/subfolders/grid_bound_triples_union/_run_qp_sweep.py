"""Agent A: MV QP sweep over delta values for grid_bound_triples_union.

Solves the Matolcsi-Vinuesa semi-infinite QP (via delsarte_dual.mv_qp_optimize.solve_mv_qp)
for a fixed sweep of delta values. Writes one JSON per delta into
``delsarte_dual/grid_bound_triples_union/G_by_delta/delta_<str>.json`` with the
optimum coefficients and diagnostics at lossless float precision.

Run from the project root via:
    python -m delsarte_dual.grid_bound_triples_union._run_qp_sweep
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from delsarte_dual.mv_qp_optimize import solve_mv_qp


# Exact-rational delta mapping (as required by the spec)
DELTA_SWEEP = [
    ("0.10",  10,  100),
    ("0.115", 115, 1000),
    ("0.13",  13,  100),
    ("0.138", 138, 1000),
    ("0.15",  15,  100),
    ("0.165", 165, 1000),
    ("0.18",  18,  100),
]

N_COEFFS = 119
N_GRID = 5001
SOLVER = "cvxpy/CLARABEL"


def _lossless_floats(xs):
    """Convert an iterable of floats into a list where each element is the exact
    lossless float literal (via repr). JSON-serialising these as Python floats
    preserves full 17-digit precision (json uses repr for floats)."""
    return [float(x) for x in xs]


def _u_fraction(delta_p: int, delta_q: int):
    """u = 1/2 + delta  =  (delta_q + 2*delta_p) / (2*delta_q)."""
    num = delta_q + 2 * delta_p
    den = 2 * delta_q
    return num, den


def run_one(delta_str: str, delta_p: int, delta_q: int, out_dir: Path) -> dict:
    delta = delta_p / delta_q
    u_p, u_q = _u_fraction(delta_p, delta_q)
    u = u_p / u_q

    entry = {
        "delta_str": delta_str,
        "delta_p": int(delta_p),
        "delta_q": int(delta_q),
        "u_p": int(u_p),
        "u_q": int(u_q),
        "n_coeffs": N_COEFFS,
        "n_grid": N_GRID,
        "solver": SOLVER,
    }

    try:
        res = solve_mv_qp(n=N_COEFFS, delta=delta, u=u, n_grid=N_GRID,
                          solver="cvxpy", verbose=False)
        entry["a_opt"] = _lossless_floats(res["a_opt"].tolist())
        entry["S1"] = float(res["S1"])
        entry["min_G_numeric"] = float(res["min_G"])
        entry["gain"] = float(res["gain"])
        entry["M_no_z1_numeric"] = float(res["M_no_z1"])
        entry["M_with_z1_numeric"] = float(res["M_with_z1"])
        entry["status"] = "OK"
    except Exception as exc:  # pragma: no cover - defensive
        tb = traceback.format_exc(limit=2)
        reason = f"{type(exc).__name__}: {exc}".strip()
        # Keep a short, single-line failure reason for the JSON
        entry["a_opt"] = []
        entry["S1"] = float("nan")
        entry["min_G_numeric"] = float("nan")
        entry["gain"] = float("nan")
        entry["M_no_z1_numeric"] = float("nan")
        entry["M_with_z1_numeric"] = float("nan")
        entry["status"] = f"FAILED: {reason}"
        print(f"[delta={delta_str}] FAILED: {reason}")
        print(tb)

    out_path = out_dir / f"delta_{delta_str}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        # json.dump writes floats via repr, preserving full precision
        json.dump(entry, fh, indent=2, allow_nan=True)
        fh.write("\n")
    return entry


def main() -> int:
    here = Path(__file__).resolve().parent
    out_dir = here / "G_by_delta"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for delta_str, dp, dq in DELTA_SWEEP:
        print(f"[sweep] solving delta={delta_str} ({dp}/{dq}) ...", flush=True)
        entry = run_one(delta_str, dp, dq, out_dir)
        rows.append(entry)
        if entry["status"] == "OK":
            print(f"  -> OK  gain={entry['gain']:.6f}  "
                  f"min_G={entry['min_G_numeric']:.6f}  "
                  f"M_with_z1={entry['M_with_z1_numeric']:.6f}")
        else:
            print(f"  -> {entry['status']}")

    # Summary table
    print()
    print("=" * 88)
    print(f"{'delta':>7s}  {'status':<8s}  {'S1':>14s}  {'min_G':>10s}  "
          f"{'gain':>12s}  {'M_with_z1':>12s}")
    print("-" * 88)
    for e in rows:
        status = "OK" if e["status"] == "OK" else "FAIL"
        print(f"{e['delta_str']:>7s}  {status:<8s}  "
              f"{e['S1']:>14.8f}  {e['min_G_numeric']:>10.6f}  "
              f"{e['gain']:>12.8f}  {e['M_with_z1_numeric']:>12.8f}")
    print("=" * 88)
    print(f"Wrote {len(rows)} JSON files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
