"""Cloud H100 runner: solve a Pólya LP MPS file via cuOpt.

Usage on a fresh GPU instance (e.g., Lambda H100 or vast.ai):

    pip install cuopt-server-cu12 cuopt-mps-parser
    python polya_lp_cloud_cuopt.py --mps polya_d64_R8.mps \\
                                   --tol 1e-7 --maxtime 7200 --out out.json

Output: a JSON with primal/dual obj, residuals, and the variable-vector
written to a separate .npy file for offline rationalization.

This is the "PDLP numerical solve" half of the rigorous-cert pipeline.
The exact rationalization step (Jansson-style perturbation + gmpy2 verify)
runs separately offline on the dumped solution.
"""
import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mps", required=True, help="Path to MPS input file")
    ap.add_argument("--out", default="cuopt_result.json", help="JSON output")
    ap.add_argument("--xnpy", default="cuopt_x.npy",
                    help="Numpy file for primal solution (for rationalization)")
    ap.add_argument("--ynpy", default="cuopt_y.npy",
                    help="Numpy file for dual solution")
    ap.add_argument("--tol", type=float, default=1e-7,
                    help="Target KKT tolerance")
    ap.add_argument("--maxtime", type=float, default=7200.0,
                    help="Max solve time (seconds)")
    args = ap.parse_args()

    # Try cuOpt first (NVIDIA GPU PDLP)
    print(f"Loading MPS: {args.mps}")
    try:
        import cuopt
        from cuopt.linear_programming import Solver
        # cuOpt API: load MPS via the parser, set tolerances, solve
        solver = Solver()
        solver.read_mps(args.mps)
        solver.set_optimality_tolerance(args.tol)
        solver.set_time_limit(args.maxtime)
        t0 = time.time()
        sol = solver.solve()
        wall = time.time() - t0
        out = {
            "solver": "cuopt",
            "status": str(sol.get_status()),
            "obj_primal": sol.get_primal_objective(),
            "obj_dual": sol.get_dual_objective(),
            "primal_res": sol.get_primal_residual(),
            "dual_res": sol.get_dual_residual(),
            "wall_s": wall,
            "n_vars": sol.n_vars(),
            "n_eq": sol.n_constraints(),
        }
        # Dump solution vectors
        import numpy as np
        x = sol.get_primal_solution()
        y = sol.get_dual_solution()
        np.save(args.xnpy, np.asarray(x, dtype=np.float64))
        np.save(args.ynpy, np.asarray(y, dtype=np.float64))
        print(f"\nResult: {out}")
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        return
    except (ImportError, ModuleNotFoundError):
        print("cuOpt not available; trying OR-Tools PDLP")
    except Exception as e:
        print(f"cuOpt failed: {e}; trying OR-Tools PDLP")

    # Fallback: OR-Tools PDLP (CPU multithreaded)
    try:
        from ortools.pdlp import solvers_pb2
        from ortools.pdlp.python import pdlp
        # Pdlp can read MPS via the standard MathOpt interface, but the
        # simplest approach: parse with scipy and pass directly.
        from scipy.io import mmread  # not the right format; need MPS reader
        # Use HiGHS as MPS reader (any LP solver can do this)
        import highspy
        h = highspy.Highs()
        h.silent()
        h.readModel(args.mps)
        # Extract LP
        lp = h.getLp()
        # ... build OR-Tools problem (skipped — would need protobuf)
        print("OR-Tools PDLP path not fully implemented in this script.")
        print("Use cuOpt or write a custom HiGHS adapter.")
        return
    except (ImportError, ModuleNotFoundError):
        print("OR-Tools PDLP not available either.")

    # Final fallback: HiGHS IPM (CPU only, but rigorous)
    try:
        import highspy
        print("Falling back to HiGHS IPM (CPU)")
        h = highspy.Highs()
        if False:
            h.silent()
        h.setOptionValue("solver", "ipm")
        h.setOptionValue("primal_feasibility_tolerance", args.tol)
        h.setOptionValue("dual_feasibility_tolerance", args.tol)
        h.setOptionValue("time_limit", args.maxtime)
        h.readModel(args.mps)
        t0 = time.time()
        h.run()
        wall = time.time() - t0
        info = h.getInfo()
        sol = h.getSolution()
        import numpy as np
        x = np.asarray(sol.col_value)
        y = np.asarray(sol.row_dual)
        np.save(args.xnpy, x)
        np.save(args.ynpy, y)
        out = {
            "solver": "highs_ipm",
            "status": str(h.getModelStatus()),
            "obj_primal": info.objective_function_value,
            "primal_res": info.max_primal_infeasibility,
            "dual_res": info.max_dual_infeasibility,
            "wall_s": wall,
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Result: {out}")
    except Exception as e:
        print(f"HiGHS IPM failed: {e}")
        raise


if __name__ == "__main__":
    main()
