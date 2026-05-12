#!/usr/bin/env python
"""Test script for proposed Lasserre improvements.

Tests ideas at small d values to verify they bring genuine improvement.
Uses the existing solve_enhanced infrastructure and injects new cuts.
"""

import numpy as np
from mosek.fusion import (Model, Domain, Expr, Matrix,
                          ObjectiveSense, SolutionStatus)
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_fusion import (
    enum_monomials, _make_hash_bases, _hash_monos,
    _build_hash_table, _hash_lookup,
    build_window_matrices, collect_moments,
)
from lasserre_scalable import (
    _precompute, _build_base_constraints, _add_psd_window,
)

val_d_known = {
    4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
    12: 1.271, 14: 1.284, 16: 1.319,
}


# =====================================================================
# CUT GENERATORS
# =====================================================================

def add_mccormick_cuts(mdl, y, P):
    """McCormick/RLT cuts: y_{e_i+e_j} <= y_{e_i} for i != j."""
    d = P['d']
    bases = P['bases']
    sorted_h, sort_o = P['sorted_h'], P['sort_o']
    n_cuts = 0

    for i in range(d):
        ei = np.zeros(d, dtype=np.int64)
        ei[i] = 1
        idx_ei = int(_hash_lookup(
            _hash_monos(ei.reshape(1, -1), bases), sorted_h, sort_o)[0])
        if idx_ei < 0:
            continue
        for j in range(d):
            if j == i:
                continue
            eij = np.zeros(d, dtype=np.int64)
            eij[i] = 1
            eij[j] = 1
            idx_eij = int(_hash_lookup(
                _hash_monos(eij.reshape(1, -1), bases), sorted_h, sort_o)[0])
            if idx_eij < 0:
                continue
            # y_{e_i + e_j} <= y_{e_i}
            mdl.constraint(f"mc_{i}_{j}",
                           Expr.sub(y.index(idx_ei), y.index(idx_eij)),
                           Domain.greaterThan(0.0))
            n_cuts += 1
    return n_cuts


def add_simplex_cuts(mdl, y, P):
    """Simplex valid inequalities from algebraic structure."""
    d = P['d']
    bases = P['bases']
    sorted_h, sort_o = P['sorted_h'], P['sort_o']
    n_cuts = 0

    # Get degree-2 diagonal indices y_{2*e_i}
    diag_idx = []
    for i in range(d):
        ei2 = np.zeros(d, dtype=np.int64)
        ei2[i] = 2
        idx = int(_hash_lookup(
            _hash_monos(ei2.reshape(1, -1), bases), sorted_h, sort_o)[0])
        if idx >= 0:
            diag_idx.append(idx)

    if len(diag_idx) == d:
        # Cauchy-Schwarz: sum mu_i^2 >= 1/d
        mdl.constraint("cs_lower", Expr.sum(y.pick(diag_idx)),
                       Domain.greaterThan(1.0 / d))
        n_cuts += 1

    # Higher-degree cuts if available: y_{3e_i} <= y_{2e_i}
    for i in range(d):
        ei3 = np.zeros(d, dtype=np.int64)
        ei3[i] = 3
        idx3 = int(_hash_lookup(
            _hash_monos(ei3.reshape(1, -1), bases), sorted_h, sort_o)[0])
        ei2 = np.zeros(d, dtype=np.int64)
        ei2[i] = 2
        idx2 = int(_hash_lookup(
            _hash_monos(ei2.reshape(1, -1), bases), sorted_h, sort_o)[0])
        if idx3 >= 0 and idx2 >= 0:
            mdl.constraint(f"cube_{i}",
                           Expr.sub(y.index(idx2), y.index(idx3)),
                           Domain.greaterThan(0.0))
            n_cuts += 1

    return n_cuts


# =====================================================================
# Modified solver that injects cuts into existing framework
# =====================================================================

def solve_with_cuts(d, order=2, add_mc=False, add_si=False,
                    n_bisect=12, max_cg_rounds=10, verbose=True):
    """Lasserre solver + optional McCormick and simplex cuts.

    Uses _precompute + _build_base_constraints from lasserre_scalable.
    """
    t_total = time.time()
    P = _precompute(d, order, verbose=False)
    n_y = P['n_y']
    n_win = P['n_win']
    n_loc = P['n_loc']
    ab_eiej_idx = P['ab_eiej_idx']
    t_pick = P['t_pick']
    windows = P['windows']

    if verbose:
        print(f"  d={d} order={order} n_y={n_y} n_win={n_win} "
              f"basis={P['n_basis']} loc={n_loc}")

    # ---- Phase 1: min t (scalar windows only) ----
    mdl = Model("p1")
    y = mdl.variable("y", n_y, Domain.greaterThan(0.0))
    t_var = mdl.variable("t")

    _build_base_constraints(mdl, y, P, add_upper_loc=True, verbose=False)

    mc_n = 0
    si_n = 0
    if add_mc:
        mc_n = add_mccormick_cuts(mdl, y, P)
    if add_si:
        si_n = add_simplex_cuts(mdl, y, P)

    F = Matrix.sparse(n_win, n_y, P['f_r'], P['f_c'], P['f_v'])
    f_all = Expr.mul(F, y)
    ones = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep = Expr.flatten(Expr.mul(ones, t_var))
    mdl.constraint("ws", Expr.sub(t_rep, f_all), Domain.greaterThan(0.0))
    mdl.objective(ObjectiveSense.Minimize, t_var)
    mdl.solve()
    lb_lin = t_var.level()[0]
    y_star = np.array(y.level())
    mdl.dispose()

    if verbose:
        print(f"  Phase1 lb={lb_lin:.10f} mc={mc_n} si={si_n}")

    # ---- Check violations ----
    def check_violations(yv, tv, active):
        viols = []
        T_vals = np.array([yv[t_pick[a * n_loc + b]]
                           for a in range(n_loc) for b in range(n_loc)]
                          ).reshape(n_loc, n_loc)
        for w in range(n_win):
            if w in active:
                continue
            ell, s = windows[w]
            scale = 2.0 * d / ell
            Q = np.zeros((n_loc, n_loc))
            for a in range(n_loc):
                for b in range(n_loc):
                    val = 0.0
                    for i in range(d):
                        for j in range(d):
                            if s <= i + j <= s + ell - 2:
                                idx = ab_eiej_idx[a, b, i, j]
                                if idx >= 0:
                                    val += yv[idx]
                    Q[a, b] = val * scale
            L = tv * T_vals - Q
            eigs = np.linalg.eigvalsh(L)
            if eigs[0] < -1e-6:
                viols.append((w, eigs[0]))
        viols.sort(key=lambda x: x[1])
        return viols

    violations = check_violations(y_star, lb_lin, set())
    if not violations:
        elapsed = time.time() - t_total
        return {'lb': lb_lin, 'elapsed': elapsed, 'n_active': 0,
                'mc': mc_n, 'si': si_n}

    # ---- Phase 2: CG loop ----
    mdl2 = Model("p2")
    mdl2.setSolverParam("intpntCoTolRelGap", 1e-7)
    y2 = mdl2.variable("y", n_y, Domain.greaterThan(0.0))
    t_p = mdl2.parameter("t")

    _build_base_constraints(mdl2, y2, P, add_upper_loc=True, verbose=False)
    if add_mc:
        add_mccormick_cuts(mdl2, y2, P)
    if add_si:
        add_simplex_cuts(mdl2, y2, P)

    F2 = Matrix.sparse(n_win, n_y, P['f_r'], P['f_c'], P['f_v'])
    f2 = Expr.mul(F2, y2)
    ones2 = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep2 = Expr.flatten(Expr.mul(ones2, Expr.reshape(t_p, 1, 1)))
    mdl2.constraint("ws", Expr.sub(t_rep2, f2), Domain.greaterThan(0.0))
    mdl2.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    active = set()
    best_lb = 0.0

    def feasible(tv):
        t_p.setValue(tv)
        try:
            mdl2.solve()
            ps = mdl2.getPrimalSolutionStatus()
            return ps in [SolutionStatus.Optimal, SolutionStatus.Feasible]
        except:
            return False

    # Add initial violations
    for w, eig in violations[:20]:
        _add_psd_window(mdl2, y2, t_p, w, P)
        active.add(w)

    for rnd in range(1, max_cg_rounds + 1):
        lo = max(0.5, best_lb - 0.01)
        hi = best_lb * 1.05 + 0.15 if best_lb > 0.5 else 5.0
        while not feasible(hi):
            hi *= 1.5
            if hi > 100:
                break

        for step in range(n_bisect):
            mid = lo + 0.4 * (hi - lo) if step >= 2 else (lo + hi) / 2
            if feasible(mid):
                hi = mid
            else:
                lo = mid

        lb = lo
        imp = lb - best_lb
        best_lb = max(best_lb, lb)

        # Extract y for violation check
        t_p.setValue(hi)
        mdl2.solve()
        yv = np.array(y2.level())
        viols = check_violations(yv, hi, active)

        if verbose:
            v = val_d_known.get(d, 0)
            gc = (best_lb - 1) / (v - 1) * 100 if v > 1 else 0
            print(f"    R{rnd}: lb={lb:.10f} +{imp:.2e} "
                  f"gc={gc:.1f}% viols={len(viols)}")

        if not viols:
            break
        if imp < 1e-7 and rnd >= 3:
            break
        for w, eig in viols[:20]:
            _add_psd_window(mdl2, y2, t_p, w, P)
            active.add(w)

    mdl2.dispose()
    elapsed = time.time() - t_total
    return {'lb': best_lb, 'elapsed': elapsed,
            'n_active': len(active), 'mc': mc_n, 'si': si_n}


def main():
    print("=" * 70)
    print("LASSERRE IMPROVEMENT TESTS")
    print("=" * 70)

    results = {}

    for d in [6, 8]:
        v = val_d_known[d]
        print(f"\n{'='*60}")
        print(f"d={d}, val(d)={v}")
        print(f"{'='*60}")

        configs = [
            ("Baseline",          False, False),
            ("+ McCormick",       True,  False),
            ("+ Simplex",         False, True),
            ("+ Both",            True,  True),
        ]

        for name, mc, si in configs:
            print(f"\n--- {name} ---")
            r = solve_with_cuts(d, order=2, add_mc=mc, add_si=si,
                                n_bisect=12, max_cg_rounds=8)
            gc = (r['lb'] - 1) / (v - 1) * 100
            results[(d, name)] = r
            print(f"  RESULT: lb={r['lb']:.10f} gc={gc:.1f}% "
                  f"t={r['elapsed']:.1f}s active={r['n_active']}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for d in [6, 8]:
        v = val_d_known[d]
        base_lb = results[(d, "Baseline")]['lb']
        for name in ["Baseline", "+ McCormick", "+ Simplex", "+ Both"]:
            r = results[(d, name)]
            gc = (r['lb'] - 1) / (v - 1) * 100
            delta = r['lb'] - base_lb
            print(f"  d={d} {name:15s}: lb={r['lb']:.10f} "
                  f"gc={gc:.1f}% delta={delta:+.2e}")


if __name__ == '__main__':
    main()
