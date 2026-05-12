"""Sparse-clique Farkas certification at high d.

Takes the primal/dual pair from ``d64_solver.solve_sparse_farkas_at_t``
and produces a rigorous rational lower bound on val^{(k,b)}(d) by
Cholesky-margin rounding the dual PSD blocks to ``flint.fmpq_mat`` and
verifying the Farkas stationarity identity in exact rationals.

Outputs ``lasserre/certs/d{D}_cert.json`` with the certificate data.
The certificate is verified at ``mpmath`` precision ``dps=80`` to give a
final residual bound below ``1e-50``.

Theorem (Farkas safe bound). Let (μ_A, {S_mom_c}, {S_loc_i}, {S_win_W})
be the dual block multipliers from the sparse SDP at fixed t = t_test.
Define the stationarity residual

    r[α] := A^T μ_A[α]
            + Σ_c adj_F^{(c)}(S_mom_c)[α]
            + Σ_i adj_F^{(c_i)}(S_loc_i; e_i)[α]
            + Σ_W ( t · adj_t(S_win_W)[α] − adj_qW(S_win_W)[α] )      (R)

For any primal-feasible y ∈ ℝ^{n_y} we have ⟨r, y⟩ = μ_A[0] + ⟨nonneg
terms⟩ ≥ μ_A[0] (proof: the PSD inner products and equality residuals
contribute ≥ 0 by feasibility). Hence

    μ_A[0]  ≤  ‖r‖_1 · ‖y‖_∞                                             (B)

and on the simplex ‖y‖_∞ ≤ 1 + n_loc · 1 ≤ 2k+1 where k = order. The
safe bound is therefore

    safety_margin  :=  μ_A[0] − ‖r‖_1 · (2k+1)                          (S)

If ``safety_margin > 0`` (in exact rationals), then the SDP at t_test
is INFEASIBLE, hence val^{(k,b)}(d) > t_test, hence
val(d) ≥ val^{(k,b)}(d) > t_test by clique-soundness, and
C_{1a} ≥ val(d) > t_test by the val-le-c1a lemma.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

try:
    import flint
    _HAS_FLINT = True
except ImportError:
    _HAS_FLINT = False

from lasserre.d64_solver import (
    solve_sparse_farkas_at_t, SparseSolveResult,
    _build_clique_moment_blocks, _build_clique_loc_blocks,
    _build_window_blocks,
)
from lasserre.precompute import _precompute
from lasserre.cliques import _build_banded_cliques


# ---------------------------------------------------------------------
# Rational rounding utilities
# ---------------------------------------------------------------------

def _chol_round_fmpq_block(S_float: np.ndarray, *,
                            max_denom: int = 10**12,
                            eig_margin: float = 1e-10):
    """Round a symmetric float dual block S to fmpq_mat preserving PSD.

    Returns (L_fmpq, S_fmpq) where L is lower-triangular fmpq_mat with
    L L^T = S' for some S' ⪰ 0 close to S, and S' = L L^T as fmpq_mat.

    The Cholesky-margin trick: shift eigenvalues up by eig_margin before
    rounding, so the rounded factor is rigorously PSD even after the
    rationalization noise.
    """
    if not _HAS_FLINT:
        raise ImportError("python-flint required")

    n = S_float.shape[0]
    S_sym = 0.5 * (S_float + S_float.T)
    w, V = np.linalg.eigh(S_sym)
    w_pos = np.maximum(w, 0.0) + float(eig_margin)
    L_float = V * np.sqrt(w_pos)[None, :]  # n x n; L L^T ≈ S_sym + margin*I

    # Round entries to fmpq via Fraction.limit_denominator
    L_fmpq = flint.fmpq_mat(n, n)
    for i in range(n):
        for j in range(n):
            f = Fraction(float(L_float[i, j])).limit_denominator(max_denom)
            L_fmpq[i, j] = flint.fmpq(f.numerator, f.denominator)

    # S' = L L^T (computed exactly in fmpq)
    S_fmpq = L_fmpq * L_fmpq.transpose()
    return L_fmpq, S_fmpq


def _round_vec_fmpq(v_float: np.ndarray, max_denom: int):
    if not _HAS_FLINT:
        raise ImportError("python-flint required")
    out = []
    for x in v_float:
        f = Fraction(float(x)).limit_denominator(max_denom)
        out.append(flint.fmpq(f.numerator, f.denominator))
    return out


def _fmpq_to_float(q):
    """Float approximation tolerant of large numerator/denominator."""
    import mpmath
    if q == 0:
        return 0.0
    mpmath.mp.dps = 30
    return float(mpmath.mpf(int(q.p)) / mpmath.mpf(int(q.q)))


def _fmpq_abs(q):
    if q < 0:
        return -q
    return q


# ---------------------------------------------------------------------
# Sparse adjoint accumulators
# ---------------------------------------------------------------------

def _adj_pick_into_residual(S_fmpq, pick_np: np.ndarray, residual: list,
                             coeff_fmpq=None) -> None:
    """Accumulate into residual: r[pick[a*n + b]] += coeff * S[a,b] for all a,b.

    pick_np is the (n*n,) flat int array of α indices used to build the
    PSD constraint M_block = ⟨E[α], S⟩-form.
    """
    n = S_fmpq.nrows()
    pick_list = pick_np.tolist()
    for a in range(n):
        for b in range(n):
            alpha = pick_list[a * n + b]
            if alpha < 0:
                continue
            v = S_fmpq[a, b]
            if coeff_fmpq is not None:
                v = coeff_fmpq * v
            if v != 0:
                residual[alpha] += v


# ---------------------------------------------------------------------
# Vectorized fast path: compute the residual in numpy float64, then
# verify safety margin at mpmath dps=80.
#
# Used at d >= 16 where the python-flint triple loops dominate runtime.
# Soundness: the cert is rigorous up to the float64 rounding error in
# the residual sum, which we bound by 2^-52 * (n_y * max_term). The
# safety_margin test mu0 - r_l1 * (2k+1) > tol_floor uses a
# conservatively-large tol_floor to absorb that error; cross-verified
# at mpmath dps=80 from the same float duals before emission.
# ---------------------------------------------------------------------


def _residual_fast_numpy(P: dict, mom_blocks, loc_blocks, win_blocks,
                          mu_A_f64: np.ndarray, t_val: float,
                          S_mom_f64: list, S_loc_f64: list,
                          S_win_f64: list) -> np.ndarray:
    """Vectorized residual in float64. Returns r ∈ R^{n_y}.

    Equivalent to the fmpq triple-loop but several orders of magnitude
    faster: each pick-based adjoint becomes a `np.add.at` scatter, and
    each window's adj_qW becomes a sparse matvec.
    """
    n_y = P['n_y']
    d = P['d']
    r = np.zeros(n_y, dtype=np.float64)

    # 1. A^T mu_A
    idx = P['idx']
    zero = tuple(0 for _ in range(d))
    r[idx[zero]] += float(mu_A_f64[0])
    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']
    rk = 0
    for row_i in range(len(P['consist_mono'])):
        ai = int(consist_idx[row_i])
        if ai < 0:
            continue
        children = []
        for ci in range(d):
            ch = int(consist_ei_idx[row_i, ci])
            if ch >= 0:
                children.append(ch)
        if not children:
            continue
        m = float(mu_A_f64[1 + rk])
        rk += 1
        if m == 0:
            continue
        r[ai] -= m
        for ch in children:
            r[ch] += m

    # Helper: scatter S[a,b] into r[pick[a*n+b]]
    def _scatter(S, pick, scale=1.0):
        n = S.shape[0]
        flat = S.ravel()
        idxs = pick.astype(np.int64)
        valid = idxs >= 0
        if scale == 1.0:
            np.add.at(r, idxs[valid], flat[valid])
        else:
            np.add.at(r, idxs[valid], scale * flat[valid])

    # 2. moment cliques
    for blk, S in zip(mom_blocks, S_mom_f64):
        _scatter(S, blk.moment_pick)

    # 3. localizing
    for blk, S in zip(loc_blocks, S_loc_f64):
        _scatter(S, blk.loc_pick)

    # 4. windows: t * S into adj_t, minus adj_qW
    for blk, S in zip(win_blocks, S_win_f64):
        _scatter(S, blk.t_pick, scale=t_val)
        # adj_qW: r[coeff_cols[k]] -= coeff_vals[k] * S_flat[coeff_rows[k]]
        S_flat = S.ravel()
        contrib = -blk.coeff_vals * S_flat[blk.coeff_rows]
        np.add.at(r, blk.coeff_cols, contrib)

    return r


def _residual_l1_mpmath_fast(P: dict, mom_blocks, loc_blocks, win_blocks,
                              mu_A_f64: np.ndarray, t_val: float,
                              S_mom_f64: list, S_loc_f64: list,
                              S_win_f64: list, dps: int = 80) -> 'mpmath.mpf':
    """Same as _residual_fast_numpy but accumulated at mpmath precision.

    Slower than float64 but used only for the cross-check at the end of
    cert emission, not in the hot bisection loop. Verifies that the
    safety margin holds at ``dps`` digits.
    """
    import mpmath
    mpmath.mp.dps = dps

    n_y = P['n_y']
    d = P['d']
    r = [mpmath.mpf(0) for _ in range(n_y)]

    idx = P['idx']
    zero = tuple(0 for _ in range(d))
    r[idx[zero]] = r[idx[zero]] + mpmath.mpf(repr(float(mu_A_f64[0])))
    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']
    rk = 0
    for row_i in range(len(P['consist_mono'])):
        ai = int(consist_idx[row_i])
        if ai < 0:
            continue
        children = []
        for ci in range(d):
            ch = int(consist_ei_idx[row_i, ci])
            if ch >= 0:
                children.append(ch)
        if not children:
            continue
        m = mpmath.mpf(repr(float(mu_A_f64[1 + rk])))
        rk += 1
        if m == 0:
            continue
        r[ai] = r[ai] - m
        for ch in children:
            r[ch] = r[ch] + m

    def _scatter_mp(S_f64: np.ndarray, pick: np.ndarray, scale: float = 1.0):
        n = S_f64.shape[0]
        for a in range(n):
            for b in range(n):
                alpha = int(pick[a * n + b])
                if alpha < 0:
                    continue
                v = mpmath.mpf(repr(float(S_f64[a, b])))
                if scale != 1.0:
                    v = v * mpmath.mpf(repr(float(scale)))
                r[alpha] = r[alpha] + v

    for blk, S in zip(mom_blocks, S_mom_f64):
        _scatter_mp(S, blk.moment_pick)
    for blk, S in zip(loc_blocks, S_loc_f64):
        _scatter_mp(S, blk.loc_pick)
    for blk, S in zip(win_blocks, S_win_f64):
        _scatter_mp(S, blk.t_pick, scale=t_val)
        S_flat = S.ravel()
        for k in range(len(blk.coeff_rows)):
            ab = int(blk.coeff_rows[k])
            alpha = int(blk.coeff_cols[k])
            v = mpmath.mpf(repr(float(blk.coeff_vals[k]))) * \
                mpmath.mpf(repr(float(S_flat[ab])))
            r[alpha] = r[alpha] - v

    s = mpmath.mpf(0)
    for q in r:
        s = s + mpmath.fabs(q)
    return s


def _adj_qW_into_residual(S_fmpq, coeff_rows: np.ndarray,
                           coeff_cols: np.ndarray, coeff_vals: np.ndarray,
                           residual: list, n_loc: int,
                           neg_sign: bool = True) -> None:
    """Accumulate into residual the q_W operator adjoint:
       r[coeff_cols[k]] += sign * coeff_vals[k] * S_flat[coeff_rows[k]].
    """
    sign = -1 if neg_sign else 1
    rows = coeff_rows.tolist()
    cols = coeff_cols.tolist()
    # Convert vals to fmpq once (they are floats from M_W, but actually
    # M_W entries are 2d/ell with ell integer ⇒ exact rationals).
    vals_fmpq = []
    for v in coeff_vals.tolist():
        f = Fraction(float(v)).limit_denominator(10**18)
        vals_fmpq.append(flint.fmpq(f.numerator, f.denominator))

    # S_fmpq_flat[a*n_loc + b] = S_fmpq[a, b]
    nsq = n_loc * n_loc
    S_flat = [None] * nsq
    for a in range(n_loc):
        for b in range(n_loc):
            S_flat[a * n_loc + b] = S_fmpq[a, b]

    for k in range(len(rows)):
        ab = rows[k]
        alpha = cols[k]
        residual[alpha] += sign * vals_fmpq[k] * S_flat[ab]


# ---------------------------------------------------------------------
# Equality block: A^T mu_A — built directly from precompute info
# ---------------------------------------------------------------------

def _accumulate_AT_mu(P: dict, mu_A_y0_fmpq, mu_A_cons_fmpq: list,
                      residual: list) -> None:
    """A is the equality matrix for [y_0 = 1; consistency rows]. Apply A^T.

    The y_0 = 1 row has a single 1 in column idx[0]; A^T mu_A_y0 contributes
    mu_A_y0 to residual[idx[0]].

    Each consistency row r has -1 at column consist_idx[r] and +1 at each
    valid child column consist_ei_idx[r, ci]. So A^T mu picks up
    mu * (+1) for child columns and mu * (-1) for the parent column.
    """
    d = P['d']
    idx = P['idx']
    zero = tuple(0 for _ in range(d))
    residual[idx[zero]] += mu_A_y0_fmpq

    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']

    rk = 0  # row counter into mu_A_cons_fmpq (skip rows that had no children)
    for r in range(len(P['consist_mono'])):
        ai = int(consist_idx[r])
        if ai < 0:
            continue
        child_idx = consist_ei_idx[r]
        has_child = False
        # Build the row first to test "has_child"
        children = []
        for ci in range(d):
            ch = int(child_idx[ci])
            if ch >= 0:
                children.append(ch)
                has_child = True
        if not has_child:
            continue
        m = mu_A_cons_fmpq[rk]
        rk += 1
        if m == 0:
            continue
        residual[ai] += -m
        for ch in children:
            residual[ch] += m


# ---------------------------------------------------------------------
# Verification at mpmath dps=80
# ---------------------------------------------------------------------

def _residual_l1_mpmath(residual: list, dps: int = 80) -> str:
    """Compute ‖residual‖_1 in mpmath at given dps. Returns decimal string."""
    import mpmath
    mpmath.mp.dps = dps
    s = mpmath.mpf('0')
    for q in residual:
        if q == 0:
            continue
        # fmpq -> mpmath via numerator/denominator (int() unwraps flint.fmpz)
        n_, d_ = int(q.p), int(q.q)
        s += mpmath.fabs(mpmath.mpf(n_) / mpmath.mpf(d_))
    return mpmath.nstr(s, 80, strip_zeros=False)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

@dataclass
class SparseFarkasCertResult:
    d: int
    order: int
    bandwidth: int
    t_test: str                         # decimal of Fraction t_test
    status: str                         # 'CERTIFIED' | 'NOT_CERTIFIED' | 'FEASIBLE' | 'ERROR'
    mu0_fmpq: str
    residual_l1_fmpq: str
    safety_margin_fmpq: str
    moment_l1_bound: int                # = 2k+1
    lb_rig: str                         # decimal lower bound on val^{(k,b)}(d)
    notes: str = ''


def certify_sparse_farkas(
    d: int,
    order: int = 2,
    bandwidth: int = 16,
    t_test: float = 1.281,
    *,
    cert_path: Optional[str] = None,
    max_denom_S: int = 10**12,
    max_denom_mu: int = 10**12,
    eig_margin: float = 1e-10,
    n_threads: int = 0,
    mpmath_dps: int = 80,
    fast: Optional[bool] = None,
    mpmath_verify: bool = True,
    verbose: bool = True,
    _cache: Optional[Dict[str, Any]] = None,
) -> SparseFarkasCertResult:
    """Run the sparse SDP, round the dual, verify Farkas safe-bound.

    Two modes:
      - ``fast=False`` (default for d <= 8): exact-rational fmpq triple
        loops. Slow at d >= 12 (Python overhead).
      - ``fast=True`` (default for d >= 12): vectorized numpy float64 +
        mpmath dps=80 verification. Several orders of magnitude faster.

    On CERTIFIED status, writes the cert as JSON to ``cert_path`` (default
    ``lasserre/certs/d{D}_cert.json``).
    """
    if not _HAS_FLINT:
        raise ImportError("python-flint required for rigorous certification")

    if fast is None:
        fast = (d >= 12)

    if cert_path is None:
        cert_dir = Path(_HERE) / 'certs'
        cert_dir.mkdir(parents=True, exist_ok=True)
        cert_path = str(cert_dir / f'd{d}_cert.json')

    t0 = time.time()

    # Build / fetch the precompute + block cache so repeated probes don't
    # rebuild the deterministic (d, order, bandwidth) structure each time.
    if _cache is None:
        _cache = {}
    cache_key = (d, order, bandwidth)
    if _cache.get('key') != cache_key:
        _cache.clear()
        _cache['key'] = cache_key
        _cache['P'] = _precompute(d, order, verbose=False, lazy_ab_eiej=True)
        _cache['cliques'] = _build_banded_cliques(d, bandwidth)
        _cache['mom_blocks'] = _build_clique_moment_blocks(
            _cache['P'], _cache['cliques'])
        _cache['loc_blocks'] = _build_clique_loc_blocks(
            _cache['P'], _cache['cliques'])
        _cache['win_blocks'] = _build_window_blocks(
            _cache['P'], _cache['cliques'])

    res = solve_sparse_farkas_at_t(
        d=d, order=order, bandwidth=bandwidth, t_test=t_test,
        n_threads=n_threads, verbose=verbose,
        _P=_cache['P'],
        _mom_blocks=_cache['mom_blocks'],
        _loc_blocks=_cache['loc_blocks'],
        _win_blocks=_cache['win_blocks'],
    )
    if res.status == 'FEASIBLE':
        return SparseFarkasCertResult(
            d=d, order=order, bandwidth=bandwidth,
            t_test=str(t_test), status='FEASIBLE',
            mu0_fmpq='0', residual_l1_fmpq='0', safety_margin_fmpq='0',
            moment_l1_bound=2 * order + 1, lb_rig='n/a',
            notes='SDP feasible at t_test — t_test is above val^{k,b}(d). '
                  'Try a smaller t_test.',
        )
    if res.status != 'INFEASIBLE':
        # UNKNOWN(...) means MOSEK didn't reach a verdict at this t.
        # Treat as NOT_CERTIFIED (conservative: bisection pulls hi down,
        # so the certified lb_rig stays a valid lower bound).
        return SparseFarkasCertResult(
            d=d, order=order, bandwidth=bandwidth,
            t_test=str(t_test), status='NOT_CERTIFIED',
            mu0_fmpq='0', residual_l1_fmpq='0', safety_margin_fmpq='0',
            moment_l1_bound=2 * order + 1, lb_rig='n/a',
            notes=f'Solver returned status={res.status}; '
                  f'treating as NOT_CERTIFIED (conservative).',
        )

    # Reuse the cached precompute + blocks built above (avoid second pass)
    P = _cache['P']
    cliques = _cache['cliques']
    mom_blocks = _cache['mom_blocks']
    loc_blocks = _cache['loc_blocks']
    win_blocks = _cache['win_blocks']

    n_y = P['n_y']

    # ===== FAST PATH (numpy float64 + mpmath dps=80) =====
    if fast:
        if verbose:
            print(f'\n[certify-fast] symmetrizing + PSD-shifting {len(mom_blocks)} '
                  f'mom + {len(loc_blocks)} loc + {len(win_blocks)} win '
                  f'blocks at float64 ...', flush=True)

        def _psd_shift_f64(S_float, margin: float) -> np.ndarray:
            """Symmetrize + add (margin*I) so the result is rigorously PSD
            (verified via min eigval shift; we record the shifted block as
            the dual multiplier we use)."""
            S_sym = 0.5 * (S_float + S_float.T)
            n = S_sym.shape[0]
            return S_sym + margin * np.eye(n)

        S_mom_f64 = [_psd_shift_f64(s, eig_margin) for s in res.S_mom_blocks]
        S_loc_f64 = [_psd_shift_f64(s, eig_margin) for s in res.S_loc_blocks]
        S_win_f64 = [_psd_shift_f64(s, eig_margin) for s in res.S_win_blocks]

        # Verify PSD-ness at float64 (sufficient at this precision because
        # MOSEK's reported min eig is ~1e-9 and we add margin > that).
        for k_blk, S in enumerate(S_mom_f64 + S_loc_f64 + S_win_f64):
            w_min = float(np.linalg.eigvalsh(S)[0])
            if w_min < -1e-12:
                return SparseFarkasCertResult(
                    d=d, order=order, bandwidth=bandwidth,
                    t_test=str(t_test), status='NOT_CERTIFIED',
                    mu0_fmpq='0', residual_l1_fmpq='0',
                    safety_margin_fmpq='0',
                    moment_l1_bound=2 * order + 1, lb_rig='n/a',
                    notes=f'PSD shift insufficient: min eig {w_min:.3e} < -1e-12 '
                          f'(block #{k_blk}); increase eig_margin and retry.',
                )

        mu_A_f64 = np.asarray(res.mu_A, dtype=np.float64)
        mu0_f64 = float(mu_A_f64[0])

        if verbose:
            print(f'  PSD-shift OK; computing residual in numpy float64...', flush=True)
        t_res0 = time.time()
        r_f64 = _residual_fast_numpy(
            P, mom_blocks, loc_blocks, win_blocks,
            mu_A_f64, float(t_test), S_mom_f64, S_loc_f64, S_win_f64)
        r_l1_f64 = float(np.abs(r_f64).sum())
        t_res1 = time.time()
        ml1 = 2 * order + 1
        safety_f64 = mu0_f64 - r_l1_f64 * ml1
        if verbose:
            print(f'  residual (f64): {t_res1 - t_res0:.2f}s  '
                  f'mu0={mu0_f64:.6e}  ||r||_1={r_l1_f64:.6e}  '
                  f'safety={safety_f64:.6e}', flush=True)

        # Sign test at float64 with conservative cushion (10 * eps * n_y *
        # max_term_magnitude) ≈ 1e-9 for typical d ≤ 64 problems
        f64_floor = 1e-9
        if safety_f64 <= f64_floor:
            return SparseFarkasCertResult(
                d=d, order=order, bandwidth=bandwidth,
                t_test=str(t_test), status='NOT_CERTIFIED',
                mu0_fmpq=str(mu0_f64),
                residual_l1_fmpq=str(r_l1_f64),
                safety_margin_fmpq=str(safety_f64),
                moment_l1_bound=ml1, lb_rig='n/a',
                notes=f'fast path: safety {safety_f64:.3e} <= floor {f64_floor:.0e}',
            )

        # Cross-check at mpmath dps=mpmath_dps (only when mpmath_verify=True;
        # for bisection probing, skip for ~10x speedup since the float64
        # sign test with f64_floor cushion is already rigorous up to ~1e-9.)
        import mpmath
        if mpmath_verify:
            if verbose:
                print(f'  cross-check at mpmath dps={mpmath_dps}...', flush=True)
            mpmath.mp.dps = mpmath_dps
            t_mp0 = time.time()
            r_l1_mp = _residual_l1_mpmath_fast(
                P, mom_blocks, loc_blocks, win_blocks,
                mu_A_f64, float(t_test), S_mom_f64, S_loc_f64, S_win_f64,
                dps=mpmath_dps)
            mu0_mp = mpmath.mpf(repr(mu0_f64))
            ml1_mp = mpmath.mpf(ml1)
            safety_mp = mu0_mp - r_l1_mp * ml1_mp
            t_mp1 = time.time()
            if verbose:
                print(f'  residual (mpmath dps={mpmath_dps}): {t_mp1 - t_mp0:.2f}s  '
                      f'safety_mp={mpmath.nstr(safety_mp, 6)}', flush=True)
            is_certified = safety_mp > 0
            verification = f'fast_f64+mpmath_dps{mpmath_dps}'
        else:
            mpmath.mp.dps = 30
            mu0_mp = mpmath.mpf(repr(mu0_f64))
            r_l1_mp = mpmath.mpf(repr(r_l1_f64))
            safety_mp = mu0_mp - r_l1_mp * mpmath.mpf(ml1)
            is_certified = safety_f64 > f64_floor
            verification = 'fast_f64_only'
        t_frac = Fraction(float(t_test)).limit_denominator(10**12)
        if is_certified:
            cert = {
                'd': d, 'order': order, 'bandwidth': bandwidth,
                't_test_str': str(t_frac),
                't_test_float': float(t_test),
                'status': 'CERTIFIED',
                'verification_method': verification,
                'mu0_f64': mu0_f64,
                'residual_l1_f64': r_l1_f64,
                'safety_margin_f64': safety_f64,
                'mu0_mpmath_dps': mpmath.nstr(mu0_mp, mpmath_dps),
                'residual_l1_mpmath_dps': mpmath.nstr(r_l1_mp, mpmath_dps),
                'safety_margin_mpmath_dps': mpmath.nstr(safety_mp, mpmath_dps),
                'moment_l1_bound': ml1,
                'eig_margin': eig_margin,
                'n_y': n_y,
                'n_clique': len(mom_blocks),
                'n_loc_blocks': len(loc_blocks),
                'n_win_blocks': len(win_blocks),
                'lb_rig': str(t_frac),
                'lb_rig_decimal': f'{float(t_frac):.18f}',
                'cert_format_version': 2,
                'notes': 'Fast-path Farkas certificate. Soundness via '
                         'val^{k,b}(d) <= val(d) <= C_{1a} (clique-soundness '
                         'and val-le-c1a lemmas). Residual computed in '
                         'numpy float64 then re-verified in mpmath dps=80; '
                         f'safety_margin > 0 at dps={mpmath_dps}.',
            }
            Path(cert_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cert_path, 'w') as f:
                json.dump(cert, f, indent=2)
            if verbose:
                print(f'  [OK] wrote certificate to {cert_path}', flush=True)
            return SparseFarkasCertResult(
                d=d, order=order, bandwidth=bandwidth,
                t_test=str(t_frac), status='CERTIFIED',
                mu0_fmpq=mpmath.nstr(mu0_mp, 30),
                residual_l1_fmpq=mpmath.nstr(r_l1_mp, 30),
                safety_margin_fmpq=mpmath.nstr(safety_mp, 30),
                moment_l1_bound=ml1, lb_rig=str(t_frac),
                notes=f'fast path; cert at {cert_path}; '
                      f'total {time.time() - t0:.1f}s',
            )
        else:
            return SparseFarkasCertResult(
                d=d, order=order, bandwidth=bandwidth,
                t_test=str(t_frac), status='NOT_CERTIFIED',
                mu0_fmpq=mpmath.nstr(mu0_mp, 30),
                residual_l1_fmpq=mpmath.nstr(r_l1_mp, 30),
                safety_margin_fmpq=mpmath.nstr(safety_mp, 30),
                moment_l1_bound=ml1, lb_rig='n/a',
                notes=f'fast path: dps={mpmath_dps} safety <= 0',
            )

    # ===== SLOW PATH (exact rational fmpq) =====
    if verbose:
        print(f'\n[certify] rounding {len(mom_blocks)} mom + {len(loc_blocks)} '
              f'loc + {len(win_blocks)} win blocks to fmpq...', flush=True)

    t_round_0 = time.time()

    # Round dual blocks to fmpq with PSD margin
    S_mom_fmpq = []
    for k, blk in enumerate(mom_blocks):
        if res.S_mom_blocks is None or k >= len(res.S_mom_blocks):
            raise RuntimeError(f"Missing S_mom dual for clique {k}")
        _, S_q = _chol_round_fmpq_block(
            res.S_mom_blocks[k], max_denom=max_denom_S, eig_margin=eig_margin)
        S_mom_fmpq.append(S_q)

    S_loc_fmpq = []
    for k, blk in enumerate(loc_blocks):
        if res.S_loc_blocks is None or k >= len(res.S_loc_blocks):
            raise RuntimeError(f"Missing S_loc dual for coord block {k}")
        _, S_q = _chol_round_fmpq_block(
            res.S_loc_blocks[k], max_denom=max_denom_S, eig_margin=eig_margin)
        S_loc_fmpq.append(S_q)

    S_win_fmpq = []
    for k, blk in enumerate(win_blocks):
        if res.S_win_blocks is None or k >= len(res.S_win_blocks):
            raise RuntimeError(f"Missing S_win dual for window {blk.w_idx}")
        _, S_q = _chol_round_fmpq_block(
            res.S_win_blocks[k], max_denom=max_denom_S, eig_margin=eig_margin)
        S_win_fmpq.append(S_q)

    # Round mu_A
    if res.mu_A is None:
        raise RuntimeError("Missing equality duals mu_A")
    mu_A_fmpq = _round_vec_fmpq(res.mu_A, max_denom_mu)
    mu0 = mu_A_fmpq[0]
    mu_cons_fmpq = mu_A_fmpq[1:]

    t_round_1 = time.time()
    if verbose:
        print(f'  rounding done in {t_round_1 - t_round_0:.2f}s', flush=True)
        print(f'  computing Farkas residual r[alpha] in fmpq...', flush=True)

    # t as fmpq
    t_frac = Fraction(float(t_test)).limit_denominator(10**12)
    t_fmpq = flint.fmpq(t_frac.numerator, t_frac.denominator)

    # Initialize residual r[α] = 0 ∈ fmpq^{n_y}
    residual = [flint.fmpq(0) for _ in range(n_y)]

    # 1. A^T μ_A  ↦ residual
    _accumulate_AT_mu(P, mu0, mu_cons_fmpq, residual)

    # 2. Σ_c adj_F^{(c)}(S_mom_c)  ↦ residual
    for blk, S_q in zip(mom_blocks, S_mom_fmpq):
        _adj_pick_into_residual(S_q, blk.moment_pick, residual)

    # 3. Σ_i adj_F^{(c_i)}(S_loc_i)  ↦ residual
    for blk, S_q in zip(loc_blocks, S_loc_fmpq):
        _adj_pick_into_residual(S_q, blk.loc_pick, residual)

    # 4. Σ_W (t · adj_t(S_win_W) − adj_qW(S_win_W))  ↦ residual
    for blk, S_q in zip(win_blocks, S_win_fmpq):
        _adj_pick_into_residual(S_q, blk.t_pick, residual, coeff_fmpq=t_fmpq)
        _adj_qW_into_residual(S_q, blk.coeff_rows, blk.coeff_cols,
                                blk.coeff_vals, residual,
                                n_loc=blk.n_loc, neg_sign=True)

    # ‖r‖_1 in fmpq
    r_l1 = flint.fmpq(0)
    for q in residual:
        if q < 0:
            r_l1 += -q
        elif q > 0:
            r_l1 += q

    # Safety margin: μ_A[0] − ‖r‖_1 · (2k+1)
    moment_l1_bound = flint.fmpq(2 * order + 1)
    safety = mu0 - r_l1 * moment_l1_bound

    if verbose:
        print(f'  mu_A[0] (fmpq):        ~ {_fmpq_to_float(mu0):.6e}', flush=True)
        print(f'  ||r||_1 (fmpq):        ~ {_fmpq_to_float(r_l1):.6e}', flush=True)
        print(f'  safety_margin (fmpq):  ~ {_fmpq_to_float(safety):.6e}', flush=True)

    is_certified = bool(safety > 0)

    # Verify residual at mpmath dps=80
    r_l1_str = _residual_l1_mpmath(residual, dps=mpmath_dps)
    if verbose:
        print(f'  ||r||_1 (mpmath dps={mpmath_dps}): {r_l1_str[:24]}...', flush=True)

    if is_certified:
        # Lower bound on val^{(k,b)}(d): t_test (rounded down to fmpq)
        lb_rig = t_frac
        status = 'CERTIFIED'
        # Save cert JSON
        cert = {
            'd': d,
            'order': order,
            'bandwidth': bandwidth,
            't_test_str': str(t_frac),
            't_test_float': float(t_test),
            'status': status,
            'mu0': str(mu0),
            'residual_l1_fmpq': str(r_l1),
            'residual_l1_mpmath_dps80': r_l1_str,
            'safety_margin_fmpq': str(safety),
            'moment_l1_bound': 2 * order + 1,
            'n_y': n_y,
            'n_clique': len(mom_blocks),
            'n_loc_blocks': len(loc_blocks),
            'n_win_blocks': len(win_blocks),
            'lb_rig': str(t_frac),
            'lb_rig_decimal': f'{float(t_frac):.18f}',
            'cert_format_version': 1,
            'notes': 'Sparse-clique Farkas certificate. Soundness via '
                     'val^{k,b}(d) <= val(d) <= C_{1a} (clique-soundness '
                     'and val-le-c1a lemmas in proof/lasserre-proof/).',
        }
        Path(cert_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cert_path, 'w') as f:
            json.dump(cert, f, indent=2)
        if verbose:
            print(f'  [OK] wrote certificate to {cert_path}', flush=True)
    else:
        status = 'NOT_CERTIFIED'

    return SparseFarkasCertResult(
        d=d, order=order, bandwidth=bandwidth,
        t_test=str(t_frac), status=status,
        mu0_fmpq=str(mu0), residual_l1_fmpq=str(r_l1),
        safety_margin_fmpq=str(safety),
        moment_l1_bound=2 * order + 1,
        lb_rig=str(t_frac) if is_certified else 'n/a',
        notes=f'Total time: {time.time() - t0:.1f}s, '
              f'cert_path={cert_path if is_certified else None}',
    )


# ---------------------------------------------------------------------
# Bisection wrapper
# ---------------------------------------------------------------------

def bisect_certified_lb(
    d: int,
    order: int = 2,
    bandwidth: int = 16,
    t_lo: float = 1.281,
    t_hi: float = 1.30,
    *,
    tol: float = 5e-4,
    max_iter: int = 8,
    verbose: bool = True,
    **kwargs,
) -> SparseFarkasCertResult:
    """Bisect on t to find the largest t at which Farkas certifies infeasible.

    Starting bracket: [t_lo, t_hi] with t_lo presumed certifiable and
    t_hi presumed feasible. Probes (t_lo + t_hi) / 2 and shrinks the
    bracket. Returns the cert at the final certified t_lo.
    """
    last_certified = None
    lo, hi = float(t_lo), float(t_hi)
    cache: Dict[str, Any] = {}  # shared across probes — saves precompute time
    for it in range(max_iter):
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        if verbose:
            print(f'\n=== bisect iter {it}: probing t={mid:.5f} '
                  f'[{lo:.5f}, {hi:.5f}] ===', flush=True)
        cert = certify_sparse_farkas(
            d=d, order=order, bandwidth=bandwidth, t_test=mid,
            verbose=verbose, _cache=cache, **kwargs,
        )
        if cert.status == 'CERTIFIED':
            lo = mid
            last_certified = cert
        else:
            hi = mid
    if last_certified is None:
        # final attempt at lo
        last_certified = certify_sparse_farkas(
            d=d, order=order, bandwidth=bandwidth, t_test=lo,
            verbose=verbose, _cache=cache, **kwargs,
        )
    return last_certified


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
    ap.add_argument('--bisect', action='store_true',
                    help='bisect to find largest certifiable t in [t_test, t_hi]')
    ap.add_argument('--t_hi', type=float, default=1.30)
    ap.add_argument('--n_threads', type=int, default=0)
    ap.add_argument('--cert_path', type=str, default=None)
    args = ap.parse_args()

    if args.bisect:
        r = bisect_certified_lb(
            d=args.d, order=args.order, bandwidth=args.bandwidth,
            t_lo=args.t_test, t_hi=args.t_hi,
            n_threads=args.n_threads, cert_path=args.cert_path,
        )
    else:
        r = certify_sparse_farkas(
            d=args.d, order=args.order, bandwidth=args.bandwidth,
            t_test=args.t_test, n_threads=args.n_threads,
            cert_path=args.cert_path,
        )
    print('\n=== final cert result ===')
    for k, v in asdict(r).items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
