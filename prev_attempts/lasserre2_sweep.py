#!/usr/bin/env python3
"""
Run a reproducible Lasserre level-2 sweep for discretized C_{1a}.

This script extracts the core level-2 formulation from lasserre_level2.ipynb,
but packages it as a standalone CLI with:
- CLARABEL-first solve path (with optional cross-check solver),
- binary search on eta feasibility,
- optional primal upper-bound comparison,
- JSON artifact output.

Important caveat:
These are lower bounds for the discretized P-bin relaxation, not automatic
rigorous bounds for the continuous C_{1a}.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize


def monomial_basis(P: int, max_deg: int) -> List[Tuple[int, ...]]:
    basis: List[Tuple[int, ...]] = [()]
    for deg in range(1, max_deg + 1):
        for combo in combinations_with_replacement(range(P), deg):
            basis.append(combo)
    return basis


def combine(*multi_indices: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(sorted(sum(multi_indices, ())))


def parse_p_values(raw: str) -> List[int]:
    vals = [int(s.strip()) for s in raw.split(",") if s.strip()]
    if not vals:
        raise ValueError("p-values cannot be empty")
    if any(v < 2 for v in vals):
        raise ValueError("all P values must be >= 2")
    return vals


def project_simplex(x: np.ndarray) -> np.ndarray:
    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
    tau = cssv[rho] / (rho + 1.0)
    return np.maximum(x - tau, 0.0)


def solve_primal_discrete(P: int, n_restarts: int, seed: int) -> float:
    rng = np.random.default_rng(seed)

    def softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z)
        ez = np.exp(z)
        return ez / np.sum(ez)

    def objective(z: np.ndarray) -> float:
        x = softmax(z)
        conv = np.convolve(x, x, mode="full")
        return float(2.0 * P * np.max(conv))

    best_val = np.inf
    for _ in range(n_restarts):
        z0 = rng.standard_normal(P) * 0.5
        res = minimize(
            objective,
            z0,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
        )
        best_val = min(best_val, float(res.fun))
    return float(best_val)


def _solver_kwargs(solver: str) -> Dict[str, float]:
    solver = solver.upper()
    if solver == "CLARABEL":
        return {
            "tol_gap_abs": 1e-8,
            "tol_gap_rel": 1e-8,
            "tol_feas": 1e-8,
        }
    if solver == "SCS":
        return {
            "max_iters": 25000,
            "eps": 1e-7,
        }
    return {}


def _is_feasible_status(status: str) -> bool:
    return status in ("optimal", "optimal_inaccurate")


@dataclass
class SolverResult:
    solver: str
    status: str
    bound: Optional[float]
    eta_lo: Optional[float]
    eta_hi: Optional[float]
    n_bisection: int
    elapsed_sec: float
    n_moments: int
    moment_rank: Optional[int]
    d: int
    loc_d: int


class Lasserre2Model:
    def __init__(self, P: int, moment_box: bool = True) -> None:
        self.P = P
        self.moment_box = moment_box

        basis_2 = monomial_basis(P, 2)
        basis_1 = monomial_basis(P, 1)
        all_beta = monomial_basis(P, 3)

        self.d = len(basis_2)
        self.loc_d = len(basis_1)

        moments_set = set()
        for i in range(self.d):
            for j in range(i, self.d):
                moments_set.add(combine(basis_2[i], basis_2[j]))
        for k in range(P):
            for a in range(self.loc_d):
                for b in range(a, self.loc_d):
                    moments_set.add(combine((k,), basis_1[a], basis_1[b]))
        for beta in all_beta:
            moments_set.add(beta)
            for i in range(P):
                moments_set.add(combine((i,), beta))
        for k_conv in range(2 * P - 1):
            for a in range(self.loc_d):
                for b in range(a, self.loc_d):
                    moments_set.add(combine(basis_1[a], basis_1[b]))
                    for i in range(P):
                        j_val = k_conv - i
                        if 0 <= j_val < P:
                            moments_set.add(
                                combine(basis_1[a], basis_1[b], (i,), (j_val,))
                            )

        moments_list = sorted(moments_set, key=lambda m: (len(m), m))
        self.moment_idx = {m: idx for idx, m in enumerate(moments_list)}
        self.n_moments = len(moments_list)

        B_M: Dict[int, np.ndarray] = {}
        for i in range(self.d):
            for j in range(i, self.d):
                mu = combine(basis_2[i], basis_2[j])
                idx = self.moment_idx[mu]
                if idx not in B_M:
                    B_M[idx] = np.zeros((self.d, self.d))
                B_M[idx][i, j] = 1.0
                if i != j:
                    B_M[idx][j, i] = 1.0
        self.B_M = B_M

        B_locs: List[Dict[int, np.ndarray]] = []
        for k in range(P):
            B_L: Dict[int, np.ndarray] = {}
            for a in range(self.loc_d):
                for b in range(a, self.loc_d):
                    mu = combine((k,), basis_1[a], basis_1[b])
                    idx = self.moment_idx[mu]
                    if idx not in B_L:
                        B_L[idx] = np.zeros((self.loc_d, self.loc_d))
                    B_L[idx][a, b] = 1.0
                    if a != b:
                        B_L[idx][b, a] = 1.0
            B_locs.append(B_L)
        self.B_locs = B_locs

        simplex_data = []
        for beta in all_beta:
            lhs_indices = [self.moment_idx[combine((i,), beta)] for i in range(P)]
            rhs_idx = self.moment_idx[beta]
            simplex_data.append((lhs_indices, rhs_idx))
        self.simplex_data = simplex_data

        B_M1: Dict[int, np.ndarray] = {}
        for a in range(self.loc_d):
            for b in range(a, self.loc_d):
                mu = combine(basis_1[a], basis_1[b])
                idx = self.moment_idx[mu]
                if idx not in B_M1:
                    B_M1[idx] = np.zeros((self.loc_d, self.loc_d))
                B_M1[idx][a, b] = 1.0
                if a != b:
                    B_M1[idx][b, a] = 1.0
        self.B_M1 = B_M1

        B_pks: List[Dict[int, np.ndarray]] = []
        for k_conv in range(2 * P - 1):
            B_pk: Dict[int, np.ndarray] = {}
            for a in range(self.loc_d):
                for b in range(a, self.loc_d):
                    for i in range(P):
                        j_val = k_conv - i
                        if 0 <= j_val < P:
                            mu = combine(basis_1[a], basis_1[b], (i,), (j_val,))
                            idx = self.moment_idx[mu]
                            if idx not in B_pk:
                                B_pk[idx] = np.zeros((self.loc_d, self.loc_d))
                            B_pk[idx][a, b] += 1.0
                            if a != b:
                                B_pk[idx][b, a] += 1.0
            B_pks.append(B_pk)
        self.B_pks = B_pks

        y = cp.Variable(self.n_moments, name="y")
        eta_param = cp.Parameter(nonneg=True, name="eta")
        constraints = []

        M_expr = sum(y[idx] * mat for idx, mat in self.B_M.items())
        constraints.append(M_expr >> 0)
        constraints.append(y[self.moment_idx[()]] == 1.0)

        if self.moment_box:
            constraints.append(y >= 0.0)
            constraints.append(y <= 1.0)

        for B_L in self.B_locs:
            L_k = sum(y[idx] * mat for idx, mat in B_L.items())
            constraints.append(L_k >> 0)

        for lhs_indices, rhs_idx in self.simplex_data:
            constraints.append(sum(y[i] for i in lhs_indices) == y[rhs_idx])

        M1_expr = sum(y[idx] * mat for idx, mat in self.B_M1.items())
        for B_pk in self.B_pks:
            pk_expr = sum(y[idx] * mat for idx, mat in B_pk.items())
            L_gk = eta_param * M1_expr - 2.0 * P * pk_expr
            constraints.append(L_gk >> 0)

        self.y = y
        self.eta_param = eta_param
        self.prob = cp.Problem(cp.Minimize(0), constraints)

    def solve_bound(
        self,
        solver: str,
        eta_tol: float,
        eta_lo: float,
        eta_hi: float,
        max_expand: int = 12,
    ) -> SolverResult:
        t0 = time.time()
        solver = solver.upper()
        kwargs = _solver_kwargs(solver)

        def solve_at(eta: float) -> Tuple[str, Optional[np.ndarray]]:
            self.eta_param.value = float(eta)
            try:
                self.prob.solve(solver=solver, warm_start=True, verbose=False, **kwargs)
            except Exception:
                return "solver_error", None
            if _is_feasible_status(self.prob.status):
                return self.prob.status, self.y.value.copy()
            return self.prob.status, None

        status, best_y = solve_at(eta_hi)
        n_expand = 0
        while (not _is_feasible_status(status)) and n_expand < max_expand:
            eta_hi *= 1.5
            status, best_y = solve_at(eta_hi)
            n_expand += 1

        if not _is_feasible_status(status):
            return SolverResult(
                solver=solver,
                status=status,
                bound=None,
                eta_lo=None,
                eta_hi=None,
                n_bisection=0,
                elapsed_sec=time.time() - t0,
                n_moments=self.n_moments,
                moment_rank=None,
                d=self.d,
                loc_d=self.loc_d,
            )

        n_bisect = 0
        while eta_hi - eta_lo > eta_tol:
            eta_mid = 0.5 * (eta_lo + eta_hi)
            mid_status, y_mid = solve_at(eta_mid)
            n_bisect += 1
            if _is_feasible_status(mid_status):
                eta_hi = eta_mid
                best_y = y_mid
                status = mid_status
            else:
                eta_lo = eta_mid

        rank = None
        if best_y is not None:
            M_val = sum(best_y[idx] * mat for idx, mat in self.B_M.items())
            eigvals = np.linalg.eigvalsh(M_val)
            cutoff = 1e-6 * max(float(np.max(eigvals)), 1e-12)
            rank = int(np.sum(eigvals > cutoff))

        return SolverResult(
            solver=solver,
            status=status,
            bound=float(eta_hi),
            eta_lo=float(eta_lo),
            eta_hi=float(eta_hi),
            n_bisection=n_bisect,
            elapsed_sec=time.time() - t0,
            n_moments=self.n_moments,
            moment_rank=rank,
            d=self.d,
            loc_d=self.loc_d,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep level-2 Lasserre lower bounds for discretized C_{1a}.",
    )
    parser.add_argument(
        "--p-values",
        type=str,
        default="6,7,8,9,10,11,12",
        help="Comma-separated P values to sweep.",
    )
    parser.add_argument(
        "--primary-solver",
        type=str,
        default="CLARABEL",
        help="Primary CVXPY solver (e.g., CLARABEL, SCS).",
    )
    parser.add_argument(
        "--crosscheck-solver",
        type=str,
        default="SCS",
        help="Optional secondary solver for consistency checks. Use 'none' to disable.",
    )
    parser.add_argument(
        "--crosscheck-max-p",
        type=int,
        default=8,
        help="Run cross-check only for P <= this value.",
    )
    parser.add_argument(
        "--eta-tol",
        type=float,
        default=1e-3,
        help="Binary-search tolerance for eta.",
    )
    parser.add_argument(
        "--moment-box",
        action="store_true",
        help="Enable valid box constraints 0<=moments<=1 for conditioning.",
    )
    parser.add_argument(
        "--primal-restarts",
        type=int,
        default=20,
        help="Number of restarts for primal comparison (0 to disable).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for primal restarts.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="prev_attempts/lasserre2_sweep_results.json",
        help="Path to output JSON report.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    p_values = parse_p_values(args.p_values)

    crosscheck_solver = args.crosscheck_solver.strip().upper()
    if crosscheck_solver in ("NONE", ""):
        crosscheck_solver = ""

    print("=" * 88)
    print("Lasserre-2 Sweep (Discretized C_{1a})")
    print("=" * 88)
    print(
        f"P values={p_values}, primary={args.primary_solver.upper()}, "
        f"crosscheck={crosscheck_solver or 'off'}, eta_tol={args.eta_tol}"
    )
    print(
        f"moment_box={'on' if args.moment_box else 'off'}, "
        f"primal_restarts={args.primal_restarts}"
    )
    print("=" * 88)

    rows: List[Dict[str, object]] = []
    prev_primary_bound = 2.0

    for P in p_values:
        print(f"\n[P={P}] Building level-2 model...")
        model = Lasserre2Model(P, moment_box=args.moment_box)
        print(
            f"       d={model.d}, loc_d={model.loc_d}, n_moments={model.n_moments}"
        )

        eta_hi = max(1.05, min(2.0 * P, prev_primary_bound + 0.08))
        eta_lo = 1.0
        primary = model.solve_bound(
            solver=args.primary_solver,
            eta_tol=args.eta_tol,
            eta_lo=eta_lo,
            eta_hi=eta_hi,
        )
        print(
            f"       primary[{primary.solver}] status={primary.status} "
            f"bound={primary.bound} time={primary.elapsed_sec:.1f}s"
        )

        cross = None
        if crosscheck_solver and P <= args.crosscheck_max_p:
            cross = model.solve_bound(
                solver=crosscheck_solver,
                eta_tol=args.eta_tol,
                eta_lo=eta_lo,
                eta_hi=max(eta_hi, (primary.bound or eta_hi) + 0.1),
            )
            print(
                f"       cross  [{cross.solver}] status={cross.status} "
                f"bound={cross.bound} time={cross.elapsed_sec:.1f}s"
            )

        primal = None
        if args.primal_restarts > 0:
            t_primal = time.time()
            primal = solve_primal_discrete(
                P=P,
                n_restarts=args.primal_restarts,
                seed=args.seed + P,
            )
            print(
                f"       primal  [L-BFGS-B] ub={primal:.6f} "
                f"time={time.time() - t_primal:.1f}s"
            )

        row: Dict[str, object] = {
            "P": P,
            "primary": asdict(primary),
            "crosscheck": asdict(cross) if cross is not None else None,
            "primal_ub": float(primal) if primal is not None else None,
            "primary_to_primal_gap": (
                float(primal - primary.bound)
                if (primal is not None and primary.bound is not None)
                else None
            ),
            "note": (
                "discretized lower bound for P-bin relaxation; not automatically "
                "a rigorous continuous C_{1a} lower bound"
            ),
        }
        rows.append(row)

        if primary.bound is not None:
            prev_primary_bound = primary.bound

    print("\n" + "=" * 88)
    print(
        f"{'P':>4} | {'LB(primary)':>12} | {'LB(cross)':>12} | "
        f"{'Primal UB':>10} | {'Gap':>10}"
    )
    print("-" * 88)
    for row in rows:
        P = row["P"]
        pb = row["primary"]["bound"] if row["primary"] else None
        cb = row["crosscheck"]["bound"] if row["crosscheck"] else None
        ub = row["primal_ub"]
        gap = row["primary_to_primal_gap"]
        pb_s = f"{pb:.6f}" if pb is not None else "None"
        cb_s = f"{cb:.6f}" if cb is not None else "n/a"
        ub_s = f"{ub:.6f}" if ub is not None else "n/a"
        gap_s = f"{gap:.6f}" if gap is not None else "n/a"
        print(f"{P:>4} | {pb_s:>12} | {cb_s:>12} | {ub_s:>10} | {gap_s:>10}")
    print("=" * 88)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "method": "lasserre_level2_sweep",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {
            "p_values": p_values,
            "primary_solver": args.primary_solver.upper(),
            "crosscheck_solver": crosscheck_solver or None,
            "crosscheck_max_p": args.crosscheck_max_p,
            "eta_tol": args.eta_tol,
            "moment_box": bool(args.moment_box),
            "primal_restarts": args.primal_restarts,
            "seed": args.seed,
        },
        "rows": rows,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved JSON report to: {out_path}")


if __name__ == "__main__":
    main()
