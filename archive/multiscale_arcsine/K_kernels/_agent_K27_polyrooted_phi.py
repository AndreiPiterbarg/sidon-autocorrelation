"""Agent K27: polynomial-rooted phi sweep for MV C_{1a} lower bound.

Tests phi(x) = (1 - u^2)^p * (a_0 + a_2 u^2 + a_4 u^4 + a_6 u^6)
where u = 2 x / DELTA, on x in [-DELTA/2, DELTA/2].
K = phi * phi has K_hat = (phi_hat)^2 >= 0 automatically.

Two-pass strategy:
  Pass 1: pure envelope (a_0=1, a_2=a_4=a_6=0) sweep over p.
  Pass 2: for each p, DE over (a_2, a_4, a_6) with phi >= 0 enforced
          (return -inf if phi negative on grid).
  Pass 3 (optional): if best M_cert > 1.272, DE all 5 params.

Output: _agent_K27_polyrooted_phi_result.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Sequence

import numpy as np
from scipy.optimize import differential_evolution

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import DELTA, evaluate_phi  # noqa: E402

EPS_END = 1e-10  # clip u to avoid 0^p at endpoints when p < 0


def phi_polyrooted(x: np.ndarray, p: float,
                   a0: float, a2: float, a4: float, a6: float) -> np.ndarray:
    """phi(x) = (1 - u^2)^p * (a0 + a2 u^2 + a4 u^4 + a6 u^6),  u = 2 x / DELTA."""
    u = 2.0 * x / DELTA
    u2 = u * u
    # Clip 1-u^2 to be strictly positive to avoid 0^p singularity for p < 0
    one_minus_u2 = np.clip(1.0 - u2, EPS_END, 1.0)
    env = one_minus_u2 ** p
    poly = a0 + a2 * u2 + a4 * (u2 * u2) + a6 * (u2 * u2 * u2)
    return env * poly


# A dense grid in u for the phi-positivity check (used inside DE objective).
N_UCHECK = 401
_U_CHECK = np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, N_UCHECK)
_U2_CHECK = _U_CHECK * _U_CHECK


def poly_nonneg_on_unit(a0: float, a2: float, a4: float, a6: float) -> bool:
    """Check poly(u^2) = a0 + a2 u^2 + a4 u^4 + a6 u^6 >= 0 on u in [-1, 1]."""
    s = _U2_CHECK   # s in [0,1)
    vals = a0 + a2 * s + a4 * s * s + a6 * s * s * s
    return bool(np.min(vals) >= -1e-12)


def eval_params(p: float, a0: float, a2: float, a4: float, a6: float,
                label: str, verbose: bool = False) -> dict:
    """Evaluate M_cert for given parameters.  Returns dict including M_cert."""
    if not poly_nonneg_on_unit(a0, a2, a4, a6):
        return {"label": label, "p": p, "a0": a0, "a2": a2, "a4": a4, "a6": a6,
                "M_cert": None, "reason": "poly negative on [0,1]"}
    try:
        r = evaluate_phi(
            lambda x: phi_polyrooted(x, p, a0, a2, a4, a6),
            label=label, verbose=verbose,
        )
    except Exception as e:
        r = {"label": label, "M_cert": None, "reason": f"exception: {e}"}
    r["p"] = float(p)
    r["a0"] = float(a0)
    r["a2"] = float(a2)
    r["a4"] = float(a4)
    r["a6"] = float(a6)
    return r


# ============================================================
# Pass 1: envelope-only sweep over p
# ============================================================

P_SWEEP = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0]


def pass1_envelope_sweep() -> list[dict]:
    print("=" * 60)
    print("PASS 1: envelope-only sweep over p (a0=1, a2=a4=a6=0)")
    print("=" * 60)
    results = []
    for p in P_SWEEP:
        r = eval_params(p, 1.0, 0.0, 0.0, 0.0,
                        label=f"K27_env_p={p:+.2f}",
                        verbose=True)
        results.append(r)
    return results


# ============================================================
# Pass 2: DE over (a2, a4, a6) for each p
# ============================================================

def de_objective_3p(theta: Sequence[float], p: float) -> float:
    """DE objective: -M_cert (since DE minimises) with a0=1 fixed."""
    a2, a4, a6 = theta
    if not poly_nonneg_on_unit(1.0, a2, a4, a6):
        return 1.0  # infeasible -> high cost
    try:
        r = evaluate_phi(
            lambda x: phi_polyrooted(x, p, 1.0, a2, a4, a6),
            label="DE",
            verbose=False,
        )
    except Exception:
        return 1.0
    M = r.get("M_cert")
    if M is None or not np.isfinite(M):
        return 1.0
    return -float(M)


def pass2_de_per_p(p_values: Sequence[float], maxiter: int = 8,
                   popsize: int = 6, seed: int = 17) -> list[dict]:
    print()
    print("=" * 60)
    print("PASS 2: DE over (a2, a4, a6) in [-1,1]^3 for each p")
    print(f"  maxiter={maxiter}  popsize={popsize}")
    print("=" * 60)
    results = []
    bounds = [(-1.0, 1.0)] * 3
    for p in p_values:
        print(f"\n--- DE for p={p:+.2f} ---", flush=True)
        t0 = time.time()
        try:
            res = differential_evolution(
                lambda th: de_objective_3p(th, p),
                bounds=bounds,
                maxiter=maxiter, popsize=popsize, seed=seed,
                tol=1e-4, polish=False, workers=1, updating='deferred',
            )
            a2_s, a4_s, a6_s = float(res.x[0]), float(res.x[1]), float(res.x[2])
            M_best = -float(res.fun) if res.fun < 0 else None
            dt = time.time() - t0
            print(f"  DE done in {dt:.1f}s.  "
                  f"M_best={M_best}  a2={a2_s:.4f} a4={a4_s:.4f} a6={a6_s:.4f}",
                  flush=True)
            # Evaluate with verbose=True at optimum to record full record
            r = eval_params(p, 1.0, a2_s, a4_s, a6_s,
                            label=f"K27_DE_p={p:+.2f}", verbose=True)
        except Exception as e:
            r = {"label": f"K27_DE_p={p:+.2f}", "p": p,
                 "M_cert": None, "reason": f"DE exception: {e}"}
        r["pass"] = 2
        results.append(r)
    return results


# ============================================================
# Pass 3: DE over (p, a2, a4, a6) [a0=1 fixed]
# ============================================================

def de_objective_4p(theta: Sequence[float]) -> float:
    p, a2, a4, a6 = theta
    if not poly_nonneg_on_unit(1.0, a2, a4, a6):
        return 1.0
    try:
        r = evaluate_phi(
            lambda x: phi_polyrooted(x, p, 1.0, a2, a4, a6),
            label="DE4",
            verbose=False,
        )
    except Exception:
        return 1.0
    M = r.get("M_cert")
    if M is None or not np.isfinite(M):
        return 1.0
    return -float(M)


def pass3_de_all_params(maxiter: int = 30, popsize: int = 18,
                        seed: int = 23) -> dict:
    print()
    print("=" * 60)
    print("PASS 3: DE over (p, a2, a4, a6) (a0=1 fixed)")
    print("=" * 60)
    bounds = [(-0.5, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    t0 = time.time()
    try:
        res = differential_evolution(
            de_objective_4p,
            bounds=bounds,
            maxiter=maxiter, popsize=popsize, seed=seed,
            tol=1e-6, polish=False, workers=1, updating='deferred',
        )
        p_s, a2_s, a4_s, a6_s = (float(res.x[i]) for i in range(4))
        M_best = -float(res.fun) if res.fun < 0 else None
        print(f"DE done in {time.time()-t0:.1f}s.  "
              f"M_best={M_best}  p={p_s:.4f} a2={a2_s:.4f} a4={a4_s:.4f} a6={a6_s:.4f}")
        r = eval_params(p_s, 1.0, a2_s, a4_s, a6_s,
                        label="K27_DE4_optimum", verbose=True)
    except Exception as e:
        r = {"label": "K27_DE4_optimum", "M_cert": None,
             "reason": f"DE4 exception: {e}"}
    r["pass"] = 3
    return r


# ============================================================
# Main
# ============================================================

def main():
    pass1 = pass1_envelope_sweep()
    # Pass 2 focuses on p values whose envelope-only baseline is within 0.06
    # of the best (i.e. plausibly improvable via DE over the polynomial mod).
    # Pass-1 results show this restricts to p in [-0.5, -0.1] (M_env >= 1.222).
    p_focus = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.3]
    pass2 = pass2_de_per_p(p_focus, maxiter=8, popsize=6, seed=17)

    # Best across pass1 and pass2
    all_results = list(pass1) + list(pass2)
    valid = [r for r in all_results if r.get("M_cert") is not None]
    valid.sort(key=lambda r: r["M_cert"], reverse=True)
    best12 = valid[0] if valid else None

    pass3 = None
    if best12 is not None and best12.get("M_cert", 0) > 1.272:
        print()
        print(f"Best so far M_cert={best12['M_cert']:.5f} > 1.272 "
              f"-> running PASS 3 (full 4-param DE).")
        pass3 = pass3_de_all_params(maxiter=30, popsize=18, seed=23)
        all_results.append(pass3)
        valid = [r for r in all_results if r.get("M_cert") is not None]
        valid.sort(key=lambda r: r["M_cert"], reverse=True)
    else:
        if best12 is None:
            print("\nNo valid M_cert in passes 1-2.")
        else:
            print(
                f"\nBest M_cert={best12['M_cert']:.5f} <= 1.272 "
                "-> SKIPPING PASS 3 (no improvement direction)."
            )

    print()
    print("=" * 60)
    print("TOP 10 across all passes")
    print("=" * 60)
    for r in valid[:10]:
        pas = r.get("pass", 1)
        print(f"  pass={pas}  p={r['p']:+.3f}  a0={r.get('a0',1):.3f}  "
              f"a2={r.get('a2',0):+.4f}  a4={r.get('a4',0):+.4f}  "
              f"a6={r.get('a6',0):+.4f}  "
              f"M_cert={r['M_cert']:.5f}  beats_MV={r.get('beats_MV')}")

    payload = {
        "delta": DELTA,
        "p_sweep": P_SWEEP,
        "pass1_envelope": pass1,
        "pass2_DE_per_p": pass2,
        "pass3_DE_all": pass3,
        "top10": valid[:10],
        "best": valid[0] if valid else None,
    }
    out_path = os.path.join(REPO, "_agent_K27_polyrooted_phi_result.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
