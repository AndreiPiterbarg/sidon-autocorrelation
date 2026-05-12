"""Sparse-clique Lasserre feasibility SDP solver for high-d Farkas certification.

Builds the order-k Lasserre relaxation of val(d) at fixed t = t_test using
correlative sparsity (banded chordal cliques, bandwidth b). Calls MOSEK
Fusion and returns primal y plus the dual PSD multipliers needed by
``d64_farkas_cert.py`` to assemble a rigorous Farkas certificate.

Soundness statement (proven in ``proof/lasserre-proof/lasserre_lower_bound.tex``,
Lemma "clique-soundness"):

    val^{(k,b)}(d) := min t s.t. M_k^{(c)}(y) ⪰ 0  ∀ clique c
                                M_{k-1}^{(c)}(μ_i y) ⪰ 0  ∀ i, c covering {i}
                                t M_{k-1}(y) - M_{k-1}(q_W y) ⪰ 0  ∀ W
                                  (clique-restricted when W is covered;
                                   full localizing otherwise)
                                Ay = b  (y_0 = 1, simplex consistency)

    val^{(k,b)}(d) ≤ val^{(k)}(d) ≤ val(d) ≤ C_{1a}.

Infeasibility of the fixed-t feasibility problem at t = t_test certifies
val^{(k,b)}(d) > t_test, hence C_{1a} > t_test.

The dual at infeasibility is a Farkas certificate; we extract it,
rationalize, and verify the stationarity identity in
``d64_farkas_cert.py``.

The solver is dimension-agnostic; ``d128_solver.py`` is a thin re-export
with a different default bandwidth.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from lasserre.precompute import _precompute
from lasserre.cliques import _build_banded_cliques, _build_clique_basis
from lasserre.core import _hash_monos, _hash_lookup, _hash_add


# ---------------------------------------------------------------------
# Sparse SDP build description (dual-extraction friendly)
# ---------------------------------------------------------------------

@dataclass
class CliqueBlock:
    """One per clique. Holds the picked-moment indices needed to extract
    the primal block M_k^{(c)}(y) and the dual PSD multiplier S_mom^{(c)}.
    """
    c_idx: int
    clique: List[int]
    cb_arr: np.ndarray           # (n_cb, d) clique basis monomials embedded in N^d
    moment_pick: np.ndarray      # (n_cb*n_cb,) hash-table indices into y
    n_cb: int


@dataclass
class LocBlock:
    """One per coordinate i (assigned to its nearest clique).
    Holds picks for M_{k-1}^{(c_i)}(μ_i y).
    """
    i_var: int
    c_idx: int
    n_cb_loc: int
    loc_pick: np.ndarray         # (n_cb_loc*n_cb_loc,) hash-table indices into y
    cb_arr_loc: np.ndarray


@dataclass
class WinBlock:
    """One per active window. Either clique-restricted or full-localizing."""
    w_idx: int
    sparse: bool                 # True if covered by a clique
    n_loc: int                   # block size used (n_cb_loc when sparse, full n_loc otherwise)
    t_pick: np.ndarray           # (n_loc**2,) for the t * M_{k-1}(y) part
    coeff_rows: np.ndarray       # COO rows of the q_W operator (flat n_loc**2 indices)
    coeff_cols: np.ndarray       # COO cols (n_y indices)
    coeff_vals: np.ndarray       # COO values (M_W[i,j] entries)


@dataclass
class SparseSolveResult:
    d: int
    order: int
    bandwidth: int
    t_test: float
    status: str                  # 'INFEASIBLE', 'FEASIBLE', 'UNKNOWN', 'ERROR'
    n_y: int
    n_eq: int
    n_clique: int
    primal_y: Optional[np.ndarray]
    mu_A: Optional[np.ndarray]                 # equality duals
    S_mom_blocks: Optional[List[np.ndarray]]   # per-clique moment dual
    S_loc_blocks: Optional[List[np.ndarray]]   # per-coordinate localizing dual
    S_win_blocks: Optional[List[np.ndarray]]   # per-window-localizing dual
    primal_obj: float
    dual_obj: float
    gap: float
    solver_time: float
    build_time: float
    blocks_meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Builders for sparse moment / localizing / window blocks (dual-aware)
# ---------------------------------------------------------------------

def _build_clique_moment_blocks(P: dict, cliques: List[List[int]]) -> List[CliqueBlock]:
    """For each clique I_c, build the picked-moment array for M_k^{(c)}(y).

    Returns a list of CliqueBlock (one per clique that has all moments
    available; otherwise the clique is skipped — should not happen at
    bandwidth ≤ d-1 for the standard precompute).
    """
    bases = P['bases']
    prime = P.get('prime')
    sorted_h, sort_o = P['sorted_h'], P['sort_o']
    order = P['order']
    d = P['d']

    blocks = []
    for c_idx, clique in enumerate(cliques):
        cb_arr = _build_clique_basis(clique, order, d)
        n_cb = len(cb_arr)
        AB_hash = _hash_monos(
            cb_arr[:, np.newaxis, :] + cb_arr[np.newaxis, :, :], bases, prime)
        pick = _hash_lookup(AB_hash, sorted_h, sort_o).ravel()
        if np.any(pick < 0):
            raise RuntimeError(
                f"Missing moments in clique {c_idx} — precompute order is "
                f"insufficient (clique {clique} at order {order})."
            )
        blocks.append(CliqueBlock(
            c_idx=c_idx, clique=clique, cb_arr=cb_arr,
            moment_pick=pick, n_cb=n_cb,
        ))
    return blocks


def _build_clique_loc_blocks(P: dict, cliques: List[List[int]]) -> List[LocBlock]:
    """For each variable i, find its nearest clique (by midpoint distance)
    and build the clique-restricted localizing pick array.

    Returns one LocBlock per i in 0..d-1 that has all moments available.
    """
    if P['order'] < 2:
        return []

    d = P['d']
    bases = P['bases']
    prime = P.get('prime')
    sorted_h, sort_o = P['sorted_h'], P['sort_o']
    order = P['order']

    bin_to_clique = {}
    for c_idx, clique in enumerate(cliques):
        mid = (clique[0] + clique[-1]) / 2.0
        for i in clique:
            cur = bin_to_clique.get(i)
            new_dist = abs(i - mid)
            if cur is None or new_dist < cur[1]:
                bin_to_clique[i] = (c_idx, new_dist)
    bin_to_clique = {i: v[0] for i, v in bin_to_clique.items()}

    blocks = []
    for i_var in range(d):
        c_idx = bin_to_clique.get(i_var)
        if c_idx is None:
            continue
        clique = cliques[c_idx]
        cb_arr_loc = _build_clique_basis(clique, order - 1, d)
        n_cb_loc = len(cb_arr_loc)
        e_i = np.zeros(d, dtype=np.int64)
        e_i[i_var] = 1
        AB_ei = _hash_monos(
            cb_arr_loc[:, np.newaxis, :] + cb_arr_loc[np.newaxis, :, :] +
            e_i[np.newaxis, np.newaxis, :], bases, prime)
        pick = _hash_lookup(AB_ei, sorted_h, sort_o).ravel()
        if np.any(pick < 0):
            continue
        blocks.append(LocBlock(
            i_var=i_var, c_idx=c_idx, n_cb_loc=n_cb_loc,
            loc_pick=pick, cb_arr_loc=cb_arr_loc,
        ))
    return blocks


def _build_window_blocks(P: dict, cliques: List[List[int]]) -> List[WinBlock]:
    """For each window W, build the COO data for the q_W localizing tensor.

    If the active bins of M_W are contained in a single clique, use the
    clique-restricted (small) localizing matrix; otherwise fall back to
    the full M_{k-1}(y) localizing of size n_loc.
    """
    if P['order'] < 2:
        return []

    d = P['d']
    bases = P['bases']
    prime = P.get('prime')
    sorted_h, sort_o = P['sorted_h'], P['sort_o']
    order = P['order']
    n_loc_full = P['n_loc']
    # P['t_pick'] is a Python list (cached for MOSEK pick); we need the
    # numpy array for hash arithmetic and ``.tolist()`` later.
    t_pick_full_np = P.get('t_pick_np')
    if t_pick_full_np is None:
        t_pick_full_np = np.asarray(P['t_pick'], dtype=np.int64)
    t_pick_full = t_pick_full_np

    # Build EE_hash once for the full e_i+e_j tensor
    E_arr = np.eye(d, dtype=np.int64)
    EE_deg2 = E_arr[:, np.newaxis, :] + E_arr[np.newaxis, :, :]
    EE_hash = _hash_monos(EE_deg2, bases, prime)

    blocks = []
    for w in range(P['n_win']):
        Mw = P['M_mats'][w]
        nz_i, nz_j = np.nonzero(Mw)
        if len(nz_i) == 0:
            continue
        active_bins = sorted(set(nz_i.tolist()) | set(nz_j.tolist()))

        covering_clique = None
        cov_idx = None
        for c_idx, clique in enumerate(cliques):
            cl_set = set(clique)
            if all(b in cl_set for b in active_bins):
                covering_clique = clique
                cov_idx = c_idx
                break

        if covering_clique is not None:
            cb = _build_clique_basis(covering_clique, order - 1, d)
            n_cb = len(cb)
            AB_hash = _hash_monos(
                cb[:, np.newaxis, :] + cb[np.newaxis, :, :], bases, prime)
            t_pick = _hash_lookup(AB_hash, sorted_h, sort_o).ravel()
            if np.any(t_pick < 0):
                covering_clique = None  # fall back

        if covering_clique is None:
            n_cb = n_loc_full
            t_pick = t_pick_full
            cb = None  # marker for "use the full LB_arr from P"

        # Build the COO data for sum_{i,j} M_W[i,j] * y_{α(a,b) + e_i + e_j}
        # where α(a,b) is the basis monomial sum at row a, col b.
        if cb is not None:
            AB_hash = _hash_monos(
                cb[:, np.newaxis, :] + cb[np.newaxis, :, :], bases, prime)
        else:
            LB_arr = np.array(P.get('loc_basis') or [], dtype=np.int64)
            if LB_arr.size == 0:
                LB_arr = np.zeros((1, d), dtype=np.int64)
            AB_hash = _hash_monos(
                LB_arr[:, np.newaxis, :] + LB_arr[np.newaxis, :, :], bases, prime)

        # ABIJ_hash[a, b, i, j] = α(a,b) + e_i + e_j. We only need entries
        # for (i,j) in (nz_i, nz_j).
        # Compute on the fly to save memory.
        rows = []
        cols = []
        vals = []
        # Precompute α(a,b) hashes once.
        ab_flat = (np.arange(n_cb)[:, None] * n_cb +
                   np.arange(n_cb)[None, :])  # (n_cb, n_cb)
        for k_ij in range(len(nz_i)):
            i = int(nz_i[k_ij])
            j = int(nz_j[k_ij])
            mij = float(Mw[i, j])
            # Hash of α(a,b) + e_i + e_j for all (a,b)
            shifted = _hash_add(AB_hash, EE_hash[i, j], prime)
            picks = _hash_lookup(shifted, sorted_h, sort_o)
            valid = picks >= 0
            if not np.any(valid):
                continue
            r = ab_flat[valid]
            c = picks[valid]
            rows.append(r.ravel())
            cols.append(c.ravel())
            vals.append(np.full(r.size, mij, dtype=np.float64))

        if not rows:
            continue
        rows_a = np.concatenate(rows)
        cols_a = np.concatenate(cols)
        vals_a = np.concatenate(vals)

        blocks.append(WinBlock(
            w_idx=w,
            sparse=(covering_clique is not None),
            n_loc=n_cb,
            t_pick=t_pick,
            coeff_rows=rows_a,
            coeff_cols=cols_a,
            coeff_vals=vals_a,
        ))
    return blocks


# ---------------------------------------------------------------------
# Top-level: solve the sparse Farkas feasibility SDP
# ---------------------------------------------------------------------

def solve_sparse_farkas_at_t(
    d: int,
    order: int = 2,
    bandwidth: int = 16,
    t_test: float = 1.281,
    *,
    n_threads: int = 0,
    mosek_tol: float = 1e-9,
    verbose: bool = True,
    _P=None,
    _mom_blocks=None,
    _loc_blocks=None,
    _win_blocks=None,
) -> SparseSolveResult:
    """Build and solve the sparse-clique Lasserre feasibility SDP at fixed t.

    Returns a SparseSolveResult with the primal y, the equality duals
    mu_A, and the per-block PSD duals (S_mom for each clique, S_loc for
    each coordinate, S_win for each active window).

    The MOSEK status is mapped to:
        - 'INFEASIBLE' (Farkas certificate exists) -> val^{(k,b)}(d) > t_test
        - 'FEASIBLE' (primal point found)          -> val^{(k,b)}(d) ≤ t_test
        - 'UNKNOWN' / 'ERROR' otherwise
    """
    try:
        from mosek.fusion import (
            Model, Domain, Expr, Matrix, ObjectiveSense, AccSolutionStatus,
        )
    except ImportError as e:
        raise RuntimeError(
            "MOSEK Fusion not available. Install mosek and a license."
        ) from e

    t_build_0 = time.time()

    if _P is None:
        P = _precompute(d, order, verbose=verbose, lazy_ab_eiej=True)
    else:
        P = _P
    cliques = _build_banded_cliques(d, bandwidth)

    mom_blocks = _mom_blocks if _mom_blocks is not None \
        else _build_clique_moment_blocks(P, cliques)
    loc_blocks = _loc_blocks if _loc_blocks is not None \
        else _build_clique_loc_blocks(P, cliques)
    win_blocks = _win_blocks if _win_blocks is not None \
        else _build_window_blocks(P, cliques)

    if verbose:
        n_sparse_w = sum(1 for w in win_blocks if w.sparse)
        n_full_w = len(win_blocks) - n_sparse_w
        print(f"  d={d} k={order} b={bandwidth} t_test={t_test}", flush=True)
        print(f"  cliques={len(mom_blocks)} loc_blocks={len(loc_blocks)} "
              f"win_blocks={len(win_blocks)} (sparse={n_sparse_w} "
              f"full={n_full_w})", flush=True)

    # Build the MOSEK Fusion model
    n_y = P['n_y']
    M = Model('d{}_sparse_farkas'.format(d))
    if n_threads > 0:
        M.setSolverParam('numThreads', n_threads)
    M.setSolverParam('intpntCoTolPfeas', mosek_tol)
    M.setSolverParam('intpntCoTolDfeas', mosek_tol)
    M.setSolverParam('intpntCoTolRelGap', mosek_tol)

    y = M.variable('y', n_y, Domain.unbounded())

    # y_0 = 1
    idx = P['idx']
    zero = tuple(0 for _ in range(d))
    y0_con = M.constraint('y0', y.index(idx[zero]), Domain.equalsTo(1.0))

    # Simplex consistency: y_α = sum_i y_{α + e_i}
    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']
    c_rows, c_cols, c_vals = [], [], []
    n_cons = 0
    for r in range(len(P['consist_mono'])):
        ai = int(consist_idx[r])
        if ai < 0:
            continue
        child_idx = consist_ei_idx[r]
        has_child = False
        for ci in range(d):
            ch = int(child_idx[ci])
            if ch >= 0:
                c_rows.append(n_cons)
                c_cols.append(ch)
                c_vals.append(1.0)
                has_child = True
        if has_child:
            c_rows.append(n_cons)
            c_cols.append(ai)
            c_vals.append(-1.0)
            n_cons += 1

    cons_con = None
    if n_cons > 0:
        Cmat = Matrix.sparse(n_cons, n_y, c_rows, c_cols, c_vals)
        cons_con = M.constraint('consist', Expr.mul(Cmat, y),
                                 Domain.equalsTo([0.0] * n_cons))

    # Per-clique moment PSD: M_k^{(c)}(y) ⪰ 0
    mom_psd_cons = []
    for blk in mom_blocks:
        Mc = Expr.reshape(y.pick(blk.moment_pick.tolist()), blk.n_cb, blk.n_cb)
        mom_psd_cons.append(
            M.constraint(f'mom_{blk.c_idx}', Mc, Domain.inPSDCone(blk.n_cb)))

    # Per-coord localizing PSD: M_{k-1}^{(c_i)}(μ_i y) ⪰ 0
    loc_psd_cons = []
    for blk in loc_blocks:
        Lm = Expr.reshape(y.pick(blk.loc_pick.tolist()), blk.n_cb_loc, blk.n_cb_loc)
        loc_psd_cons.append(
            M.constraint(f'loc_{blk.i_var}', Lm, Domain.inPSDCone(blk.n_cb_loc)))

    # Per-window localizing PSD: t M_{k-1}(y) - M_{k-1}(q_W y) ⪰ 0
    win_psd_cons = []
    for blk in win_blocks:
        n_cb = blk.n_loc
        t_term = Expr.mul(t_test, y.pick(blk.t_pick.tolist()))
        Cw = Matrix.sparse(
            n_cb * n_cb, n_y,
            blk.coeff_rows.tolist(), blk.coeff_cols.tolist(), blk.coeff_vals.tolist(),
        )
        Lflat = Expr.sub(t_term, Expr.mul(Cw, y))
        Lmat = Expr.reshape(Lflat, n_cb, n_cb)
        win_psd_cons.append(
            M.constraint(f'wpsd_{blk.w_idx}', Lmat, Domain.inPSDCone(n_cb)))

    # Pure feasibility — minimize 0
    M.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    M.acceptedSolutionStatus(AccSolutionStatus.Anything)
    t_build_1 = time.time()

    if verbose:
        print(f"  build: {t_build_1 - t_build_0:.2f}s — solving...", flush=True)

    t_solve_0 = time.time()
    M.solve()
    t_solve_1 = time.time()

    pstatus = M.getPrimalSolutionStatus()
    dstatus = M.getDualSolutionStatus()
    pobj = M.primalObjValue() if str(pstatus) != 'Undefined' else float('nan')
    dobj = M.dualObjValue() if str(dstatus) != 'Undefined' else float('nan')

    if verbose:
        print(f"  solve: {t_solve_1 - t_solve_0:.2f}s "
              f"primal={pstatus} dual={dstatus} pobj={pobj} dobj={dobj}",
              flush=True)

    # Status mapping. The model is min 0 s.t. {SDP feasibility constraints}.
    # MOSEK Fusion semantics:
    #   primal Optimal + dual Optimal      -> FEASIBLE point exists
    #   primal Undefined + dual Certificate -> primal INFEASIBLE (Farkas dual)
    #   primal Certificate + dual Undefined -> dual unbounded (n/a here)
    pstr = str(pstatus)
    dstr = str(dstatus)
    is_infeas = ('Certificate' in pstr) or ('Certificate' in dstr)
    is_feas = (pstr.endswith('Optimal') and dstr.endswith('Optimal'))

    if is_infeas:
        status = 'INFEASIBLE'
    elif is_feas:
        status = 'FEASIBLE'
    else:
        status = f'UNKNOWN({pstatus}/{dstatus})'
        # Always log unknown status so the trajectory can diagnose
        print(f'  WARN: solver returned UNKNOWN: primal={pstatus} dual={dstatus} '
              f'pobj={pobj} dobj={dobj}', flush=True)

    # Extract primal y and duals (always — even at infeasibility they
    # carry the Farkas direction). The Fusion duals on PSD constraints
    # are stored in vech form via getDual on the constraint object.
    primal_y = None
    mu_A = None
    S_mom_blocks = None
    S_loc_blocks = None
    S_win_blocks = None
    try:
        primal_y = np.array(y.level())
    except Exception:
        primal_y = None

    def _dual_psd(con, n):
        """Pull a length-n*n dual vector and reshape to (n,n)."""
        vec = np.array(con.dual())
        return vec.reshape(n, n)

    try:
        mu_eq = []
        mu_eq.append(float(np.array(y0_con.dual())[0]))
        if cons_con is not None:
            mu_eq.extend(np.array(cons_con.dual()).tolist())
        mu_A = np.array(mu_eq, dtype=np.float64)

        S_mom_blocks = [_dual_psd(c, blk.n_cb)
                         for c, blk in zip(mom_psd_cons, mom_blocks)]
        S_loc_blocks = [_dual_psd(c, blk.n_cb_loc)
                         for c, blk in zip(loc_psd_cons, loc_blocks)]
        S_win_blocks = [_dual_psd(c, blk.n_loc)
                         for c, blk in zip(win_psd_cons, win_blocks)]
    except Exception as e:
        if verbose:
            print(f"  WARNING: dual extraction failed: {e}", flush=True)

    gap = abs(pobj - dobj) if not (np.isnan(pobj) or np.isnan(dobj)) else float('nan')

    result = SparseSolveResult(
        d=d, order=order, bandwidth=bandwidth, t_test=t_test, status=status,
        n_y=n_y, n_eq=1 + n_cons, n_clique=len(mom_blocks),
        primal_y=primal_y, mu_A=mu_A,
        S_mom_blocks=S_mom_blocks, S_loc_blocks=S_loc_blocks,
        S_win_blocks=S_win_blocks,
        primal_obj=float(pobj), dual_obj=float(dobj), gap=float(gap),
        solver_time=t_solve_1 - t_solve_0,
        build_time=t_build_1 - t_build_0,
        blocks_meta={
            'mom_blocks': [(b.c_idx, b.clique, b.n_cb) for b in mom_blocks],
            'loc_blocks': [(b.i_var, b.c_idx, b.n_cb_loc) for b in loc_blocks],
            'win_blocks': [(b.w_idx, b.sparse, b.n_loc) for b in win_blocks],
        },
    )

    # Dispose MOSEK Fusion model — duals already copied to numpy arrays.
    # Without this, repeated probes leak ~2 GB per call at d=16.
    try:
        M.dispose()
    except Exception:
        pass
    import gc
    gc.collect()

    return result


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--order', type=int, default=2)
    ap.add_argument('--bandwidth', type=int, default=16)
    ap.add_argument('--t_test', type=float, default=1.281)
    ap.add_argument('--n_threads', type=int, default=0)
    args = ap.parse_args()

    res = solve_sparse_farkas_at_t(
        d=args.d, order=args.order, bandwidth=args.bandwidth,
        t_test=args.t_test, n_threads=args.n_threads, verbose=True,
    )
    print(f"\n=== status: {res.status} ===")
    print(f"  build_time:  {res.build_time:.2f} s")
    print(f"  solver_time: {res.solver_time:.2f} s")
    print(f"  primal_obj:  {res.primal_obj}")
    print(f"  dual_obj:    {res.dual_obj}")
    print(f"  gap:         {res.gap}")
    if res.status == 'INFEASIBLE':
        print(f"  ⇒ val^({args.order},{args.bandwidth})({args.d}) > {args.t_test}")
    elif res.status == 'FEASIBLE':
        print(f"  ⇒ val^({args.order},{args.bandwidth})({args.d}) ≤ {args.t_test}")


if __name__ == '__main__':
    main()
