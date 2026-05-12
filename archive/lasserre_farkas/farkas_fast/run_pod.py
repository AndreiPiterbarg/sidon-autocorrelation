"""Pod-side driver: run farkas_certify_bisect with the fast-bignum residual.

Invoked by deploy_farkas_fast_pod.py. CLI args:
    --d           int   number of bins
    --order       int   Lasserre order k (≥2)
    --t_lo        float starting lower bracket (default 1.0)
    --t_hi        float starting upper bracket (default 1.5)
    --tol         float bisection tolerance (default 1e-5)
    --max_bisect  int   max bisection steps (default 30)
    --fast_D_L    int   rounding denominator (default 10^9)
    --use_bignum  flag  force bignum path (default: true when D_L >= 10^8)
    --nthreads    int   MOSEK threads (default 16)
    --log_dir     str   output directory (default data/)

Writes:
    {log_dir}/farkas_fast_d{D}_k{K}.log      — streaming solver output
    {log_dir}/farkas_fast_d{D}_k{K}.json     — final result dict
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from certified_lasserre.farkas_certify import farkas_certify_bisect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--t_lo", type=float, default=1.0)
    ap.add_argument("--t_hi", type=float, default=1.5)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max_bisect", type=int, default=30)
    ap.add_argument("--fast_D_L", type=int, default=10**9)
    ap.add_argument("--use_bignum", action="store_true")
    ap.add_argument("--nthreads", type=int, default=16)
    ap.add_argument("--log_dir", default=os.path.join(_REPO, "data"))
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    tag = f"farkas_fast_d{args.d}_k{args.order}"
    json_path = os.path.join(args.log_dir, f"{tag}.json")

    use_bignum = args.use_bignum or (args.fast_D_L >= 10**8)

    header = {
        "d": args.d, "order": args.order,
        "t_lo": args.t_lo, "t_hi": args.t_hi, "tol": args.tol,
        "max_bisect": args.max_bisect,
        "fast_D_L": args.fast_D_L, "use_bignum": use_bignum,
        "nthreads": args.nthreads,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_epoch": time.time(),
    }
    print(f"[farkas_fast] header = {json.dumps(header, indent=2)}", flush=True)

    t0 = time.time()
    try:
        res = farkas_certify_bisect(
            d=args.d, order=args.order,
            t_lo=args.t_lo, t_hi=args.t_hi,
            tol=args.tol, max_bisect=args.max_bisect,
            max_denom_S=10**9, max_denom_mu=10**10,
            eig_margin=1e-9, nthreads=args.nthreads,
            use_fast_residual=True, fast_D_L=args.fast_D_L,
            fast_use_bignum=use_bignum,
            verbose=True,
        )
        wall = time.time() - t0
        out = {
            **header,
            "status": res.status,
            "lb_rig_frac": f"{int(res.lb_rig.numerator)}/{int(res.lb_rig.denominator)}",
            "lb_rig_float": float(res.lb_rig),
            "lb_rig_decimal": res.lb_rig_decimal,
            "mu0_float": res.mu0_float,
            "residual_l1_float": res.residual_l1_float,
            "safety_margin_float": res.safety_margin_float,
            "solver_time": res.solver_time,
            "round_time": res.round_time,
            "total_wall_s": wall,
        }
    except Exception as e:
        wall = time.time() - t0
        out = {
            **header,
            "status": "ERROR",
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "total_wall_s": wall,
        }
        print(f"[farkas_fast] ERROR: {type(e).__name__}: {e}", flush=True)

    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[farkas_fast] RESULT written to {json_path}", flush=True)
    print(f"[farkas_fast] result = {json.dumps(out, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
