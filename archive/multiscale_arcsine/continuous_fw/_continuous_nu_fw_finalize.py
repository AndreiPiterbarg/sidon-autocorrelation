"""Finalize the continuous-nu Frank-Wolfe pipeline from a checkpoint.

Loads `_continuous_nu_fw_ckpt_N{N}.npz`, runs a constrained CD polish
(top-K active atoms only) and a final arb-rigorous certification.

Usage:
    python _continuous_nu_fw_finalize.py --N 100 --max_active 12 \
        --cd_sweeps 3 --xi_max_rigor 10000 --prune_thresh 5e-4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
_p = REPO
for _ in range(5):
    if (_p / "delsarte_dual").is_dir():
        sys.path.insert(0, str(_p))
        break
    _p = _p.parent

from _continuous_nu_fw import (
    eval_nu_optG, pairwise_cd_optG, support_summary, rigorous_verify,
)
from _master_k26_continuous import precompute_atoms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--max_active", type=int, default=12)
    p.add_argument("--cd_sweeps", type=int, default=3)
    p.add_argument("--xi_max_rigor", type=int, default=10000)
    p.add_argument("--prune_thresh", type=float, default=5e-4)
    p.add_argument("--out", default="_continuous_nu_fw_final.json")
    args = p.parse_args()

    deltas, C, Kh_qp, Kh_1 = precompute_atoms(args.N, verbose=False,
                                              cache_dir=str(REPO))
    ckpt_path = REPO / f"_continuous_nu_fw_ckpt_N{args.N}.npz"
    d = np.load(ckpt_path)
    lam = d["lam"].copy()
    M_curr_init = float(d["M_curr"])
    print(f"Loaded checkpoint: iter={int(d['it'])}, M={M_curr_init:.7f}, "
          f"support={int((lam > 1e-8).sum())}")
    print("Top atoms:")
    for dd, ll in support_summary(lam, deltas, thresh=1e-3)[:15]:
        print(f"  delta = {dd:.6f}   lambda = {ll:.6f}")

    # CD polish (limited to top-K active)
    print(f"\nCD polish: max_active={args.max_active}, "
          f"n_sweeps={args.cd_sweeps}")
    t0 = time.time()
    lam_cd, M_cd = pairwise_cd_optG(
        lam, C, Kh_qp, Kh_1, n_sweeps=args.cd_sweeps,
        verbose=True, max_active=args.max_active)
    print(f"  CD done: M={M_cd:.7f}  in {time.time()-t0:.1f}s")
    print("After CD, top atoms:")
    final_atoms = support_summary(lam_cd, deltas, thresh=1e-5)
    for dd, ll in final_atoms[:15]:
        print(f"  delta = {dd:.6f}   lambda = {ll:.6f}")
    eff_atoms = [(dd, ll) for dd, ll in final_atoms if ll > 1e-4]
    print(f"\nEffective Caratheodory dim (lambda>1e-4): {len(eff_atoms)}")

    # Rigorous arb verification
    pruned = [(dd, ll) for dd, ll in final_atoms if ll > args.prune_thresh]
    print(f"\nRigorous arb verification: support={len(pruned)} "
          f"(prune_thresh={args.prune_thresh})")
    t0 = time.time()
    r = rigorous_verify(pruned, xi_max=args.xi_max_rigor, verbose=True)
    print(f"\n  arb cert: M_cert >= {r['M_cert_lower']:.8f}  "
          f"in {time.time()-t0:.1f}s")

    # Also do a quick comparison with v4 3-scale baseline (1.2922 vs current)
    out = {
        "N_grid": args.N,
        "checkpoint": str(ckpt_path),
        "ckpt_iter": int(d["it"]),
        "ckpt_M_cert_numeric": M_curr_init,
        "cd_polish_M_cert_numeric": float(M_cd),
        "max_active_in_cd": args.max_active,
        "cd_sweeps": args.cd_sweeps,
        "final_atoms": final_atoms,
        "eff_caratheodory_dim": len(eff_atoms),
        "support_size_full": int((lam_cd > 1e-8).sum()),
        "pruned_atoms_for_rigor": pruned,
        "prune_thresh": args.prune_thresh,
        "rigorous": r,
        "comparison_v4_3scale_arb": 1.29216,
        "rigorous_beats_v4_3scale": r["M_cert_lower"] > 1.29216,
    }
    with open(REPO / args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {REPO / args.out}")


if __name__ == "__main__":
    main()
