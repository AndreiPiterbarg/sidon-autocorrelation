"""Tests for certified_lasserre/atomic_nu_sdp.py.

The guarantees we test:

  (G1) Geometry: window_shift / project_to_windows are self-consistent on
       known anchor points and preserve symmetry.
  (G2) Soundness: the SDP's lb_numerical is nonneg, finite, and does not
       exceed the known val(d) numerically (up to solver slack).
  (G3) Order monotonicity: at fixed ν, raising Lasserre order k cannot
       decrease the bound (modulo solver tolerance).
  (G4) Concentration sense-check: ν concentrated at t ≈ 0 gives a STRICTLY
       larger bound than ν concentrated at t near ±1/2 (where (f*f) is
       forced small for any f supported on [-1/4, 1/4] — the autoconv
       support tapers).
  (G5) Error handling: empty ν / wrong-shape inputs raise loudly; solver
       failure statuses raise instead of silently returning nonsense.

We do NOT test an equivalence λ(ν_uniform) ≈ val(d) — this is false in
general (val(d) = sup_ν λ(ν), and uniform ν is not the optimal ν).  The
plan's original Dirac-δ₀ sanity check was mathematically incorrect
((f*f)(0) = ∫f(x)f(-x)dx, not ∫f², so inf_f (f*f)(0) = 0 for asymmetric
f) — we replace it with (G4).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Ensure repo root is on sys.path so `lasserre.*` and `certified_lasserre.*`
# import cleanly when pytest is run from the repo root or from tests/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lasserre.core import build_window_matrices, val_d_known
from certified_lasserre.atomic_nu_sdp import (
    window_shift, list_windows, windows_of_length,
    project_to_windows, uniform_grid_nu, peak_concentrated_nu,
    uniform_over_all_windows, seed_from_joint_bisect,
    solve_atomic_nu_sdp, AtomicNuResult,
)


# =====================================================================
# (G1) Geometry / projection correctness
# =====================================================================

@pytest.mark.parametrize("d", [4, 6, 8])
def test_window_shift_center_is_zero(d):
    """The length-2 window centered at conv-index (d-1) has shift 0."""
    # shift(ell=2, s_lo=d-1, d) = ((d-1) + 0) + 1 - d) / (2d) = 0
    assert window_shift(2, d - 1, d) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("d", [4, 6, 8])
def test_window_shift_range(d):
    """Length-2 window shifts span (-1/2, 1/2) symmetrically around 0."""
    shifts = np.array([window_shift(2, s, d) for s in range(2 * d - 1)])
    assert shifts[0] < 0
    assert shifts[-1] > 0
    assert shifts[0] == pytest.approx(-shifts[-1], abs=1e-12)
    # Strictly increasing
    assert np.all(np.diff(shifts) > 0)


@pytest.mark.parametrize("d", [4, 6])
def test_project_uniform_on_length2_windows(d):
    """Projecting ν-points placed exactly at length-2 window shifts with
    uniform weights gives uniform lam over those windows (zero on others)."""
    ell_wins = windows_of_length(d, ell=2)
    shifts = [window_shift(e, s, d) for (_, (e, s)) in ell_wins]
    N = len(shifts)
    weights = [1.0 / N] * N
    lam = project_to_windows(shifts, weights, d, ell=2)
    # Mass on exactly the length-2 window indices, each = 1/N
    total_win = len(list_windows(d))
    mass_len2 = sum(lam[gi] for (gi, _) in ell_wins)
    assert mass_len2 == pytest.approx(1.0, abs=1e-12)
    for (gi, _) in ell_wins:
        assert lam[gi] == pytest.approx(1.0 / N, abs=1e-12)
    # Other windows untouched
    other_mass = lam.sum() - mass_len2
    assert other_mass == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("d", [4, 6, 8])
def test_project_preserves_symmetry(d):
    """ν symmetric about 0 → lam symmetric across the matching length-ell windows."""
    pts = [-0.3, -0.1, 0.0, 0.1, 0.3]
    wts = [0.2] * 5
    lam = project_to_windows(pts, wts, d, ell=2)
    ell_wins = windows_of_length(d, ell=2)
    # For each length-2 window, its symmetric partner has s_lo -> (2d-2) - s_lo
    for (gi, (e, s)) in ell_wins:
        s_sym = (2 * d - 2) - s
        # Find the symmetric partner's global index
        partner_gi = next(gj for (gj, (ee, ss)) in ell_wins if ee == e and ss == s_sym)
        assert lam[gi] == pytest.approx(lam[partner_gi], abs=1e-12)


def test_project_empty_raises():
    with pytest.raises(ValueError):
        project_to_windows([], [], d=4, ell=2)


def test_project_negative_weight_raises():
    with pytest.raises(ValueError):
        project_to_windows([0.0, 0.1], [0.5, -0.1], d=4, ell=2)


def test_project_shape_mismatch_raises():
    with pytest.raises(ValueError):
        project_to_windows([0.0, 0.1], [1.0], d=4, ell=2)


# =====================================================================
# (G2) Solver soundness
# =====================================================================

def _solver_available():
    """MOSEK or Clarabel must be importable for the solve tests to run."""
    try:
        import mosek.fusion  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import cvxpy  # noqa: F401
        import clarabel  # noqa: F401
        return True
    except Exception:
        return False


requires_solver = pytest.mark.skipif(
    not _solver_available(),
    reason="neither MOSEK Fusion nor cvxpy+Clarabel available",
)


@requires_solver
def test_solve_uniform_d4_order2_is_sound():
    """Uniform ν over length-2 windows at d=4, order=2 gives a valid bound."""
    d, order = 4, 2
    lam = uniform_grid_nu(d, K=2 * d - 1, ell=2)
    res = solve_atomic_nu_sdp(lam, d=d, order=order, solver="auto",
                              verbose=False, compute_window_values=True)
    assert isinstance(res, AtomicNuResult)
    assert res.lb_numerical > 0.0
    assert np.isfinite(res.lb_numerical)
    # λ(ν) ≤ val(d), with solver slack.  val(4) ≈ 1.10233.
    assert res.lb_numerical <= val_d_known[d] + 5e-3, (
        f"lb_numerical={res.lb_numerical:.6f} exceeds val({d})={val_d_known[d]:.6f}"
    )
    # Primal y is sane: y_{0,...,0} ≈ 1 (the y_0 = 1 equality).
    from certified_lasserre.build_sdp import build_sdp_data as _bsd
    _sdp_for_idx = _bsd(d=d, order=order, lam=res.lam_windows, verbose=False)
    zero = tuple(0 for _ in range(d))
    assert res.y[_sdp_for_idx.mono_idx[zero]] == pytest.approx(1.0, abs=1e-6)
    # Window values computed and sane: sum over active-ν windows weighted
    # by lam gives back lb_numerical (since c^T y == Σ lam_W f_W(y)).
    wv = res.window_values
    assert wv is not None
    implied = float(res.lam_windows @ wv)
    assert implied == pytest.approx(res.lb_numerical, rel=1e-5, abs=1e-6), (
        f"c^T y = {res.lb_numerical:.8f} vs Σ lam_W f_W(y) = {implied:.8f}"
    )


# =====================================================================
# (G3) Order monotonicity
# =====================================================================

@requires_solver
def test_order_monotone_d4():
    """At fixed ν, Lasserre order 3 bound ≥ order 2 bound (up to solver slack)."""
    d = 4
    lam = uniform_grid_nu(d, K=2 * d - 1, ell=2)
    r2 = solve_atomic_nu_sdp(lam, d=d, order=2, solver="auto",
                             compute_window_values=False)
    r3 = solve_atomic_nu_sdp(lam, d=d, order=3, solver="auto",
                             compute_window_values=False)
    # order 3 should be no looser; allow 1e-4 solver slack
    assert r3.lb_numerical >= r2.lb_numerical - 1e-4, (
        f"order 2 lb={r2.lb_numerical:.6f} > order 3 lb={r3.lb_numerical:.6f}"
    )


# =====================================================================
# (G4) Concentration sense-check (replaces the broken δ₀ test)
# =====================================================================

@requires_solver
def test_center_peak_beats_tail_peak_d4():
    """A peak at t=0 should give a larger bound than a peak at t ≈ -1/2.

    Rationale: for any admissible f on [-1/4,1/4], (f*f) is supported on
    [-1/2, 1/2] and tapers at the endpoints.  Concentrating ν at ±1/2
    measures (f*f) where it is necessarily small, giving a weak bound;
    concentrating at 0 measures it where it is forced to be big.
    """
    d = 4
    order = 2
    lam_center = peak_concentrated_nu(d, t_star=0.0, width=0.05, K=12, ell=2)
    lam_tail = peak_concentrated_nu(d, t_star=-0.45, width=0.05, K=12, ell=2)
    r_c = solve_atomic_nu_sdp(lam_center, d=d, order=order, solver="auto",
                              compute_window_values=False)
    r_t = solve_atomic_nu_sdp(lam_tail, d=d, order=order, solver="auto",
                              compute_window_values=False)
    assert r_c.lb_numerical > r_t.lb_numerical, (
        f"center peak {r_c.lb_numerical:.6f} !> tail peak {r_t.lb_numerical:.6f}"
    )


# =====================================================================
# (G5) Error handling
# =====================================================================

def test_solve_rejects_all_zero_lam():
    with pytest.raises(ValueError):
        solve_atomic_nu_sdp(np.zeros(len(list_windows(4))), d=4, order=2,
                            solver="auto")


def test_solve_rejects_negative_lam():
    n = len(list_windows(4))
    bad = np.zeros(n)
    bad[0] = 1.0
    bad[1] = -0.1
    with pytest.raises(ValueError):
        solve_atomic_nu_sdp(bad, d=4, order=2, solver="auto")


# =====================================================================
# (G2′) c^T y identity: the primal objective equals the ν-weighted
# window-value sum, used by adaptive_from_solution.  Covered inside
# test_solve_uniform_d4_order2_is_sound but keep a standalone copy for
# clarity at a different (d, ν) setting.
# =====================================================================

@requires_solver
def test_obj_equals_nu_weighted_window_values_d6():
    d, order = 6, 2
    lam = uniform_grid_nu(d, K=5, ell=2)
    res = solve_atomic_nu_sdp(lam, d=d, order=order, solver="auto",
                              compute_window_values=True)
    implied = float(res.lam_windows @ res.window_values)
    assert implied == pytest.approx(res.lb_numerical, rel=1e-5, abs=1e-6)


# =====================================================================
# Multi-ell projection
# =====================================================================

@pytest.mark.parametrize("d", [4, 6])
def test_project_ell_all_covers_every_length(d):
    """ell='all' projection distributes mass across windows of every length."""
    lam = project_to_windows([0.0], [1.0], d, ell="all")
    # For each ell in [2, 2d], exactly one window of that length gets +1/(2d-1).
    n_ell = 2 * d - 1  # |{2, 3, ..., 2d}|
    share = 1.0 / n_ell
    counts_by_ell: Dict[int, int] = {}
    for (gi, (e, _)) in [(i, w) for i, w in enumerate(list_windows(d))]:
        if lam[gi] > 0:
            counts_by_ell[e] = counts_by_ell.get(e, 0) + 1
    assert set(counts_by_ell.keys()) == set(range(2, 2 * d + 1))
    for e, ct in counts_by_ell.items():
        assert ct == 1, f"ell={e} should have 1 hit, got {ct}"
    # Total mass = 1
    assert float(lam.sum()) == pytest.approx(1.0, abs=1e-12)


def test_project_ell_list():
    """Explicit list of ells distributes w_i/|list| to each."""
    d = 4
    lam = project_to_windows([0.0], [1.0], d, ell=[2, 4])
    # Exactly two nonzero entries, each = 0.5
    nz = np.flatnonzero(lam > 0)
    assert len(nz) == 2, nz
    assert float(lam.sum()) == pytest.approx(1.0, abs=1e-12)
    for k in nz:
        assert lam[k] == pytest.approx(0.5, abs=1e-12)


def test_project_ell_bad_string_raises():
    with pytest.raises(ValueError):
        project_to_windows([0.0], [1.0], d=4, ell="not-a-valid-mode")


def test_project_empty_ell_list_raises():
    with pytest.raises(ValueError):
        project_to_windows([0.0], [1.0], d=4, ell=[])


# =====================================================================
# uniform_over_all_windows
# =====================================================================

@pytest.mark.parametrize("d", [4, 6, 8])
def test_uniform_over_all_windows_shape_and_sum(d):
    lam = uniform_over_all_windows(d)
    assert lam.shape == (len(list_windows(d)),)
    assert float(lam.sum()) == pytest.approx(1.0, abs=1e-12)
    assert np.all(lam > 0)


# =====================================================================
# Typing import needed only inside tests (keeps top of file minimal).
# =====================================================================

from typing import Dict  # noqa: E402  — for test_project_ell_all_covers_every_length


# =====================================================================
# seed_from_joint_bisect — the primary equivalence check
# =====================================================================

def _mosek_available():
    try:
        import mosek.fusion  # noqa: F401
        return True
    except Exception:
        return False


requires_mosek = pytest.mark.skipif(
    not _mosek_available(),
    reason="MOSEK Fusion not available (joint_bisect requires it directly)",
)


@requires_mosek
@pytest.mark.parametrize("d,order", [(4, 2), (4, 3), (6, 2)])
def test_seed_from_joint_bisect_is_valid_lower_bound(d, order):
    """λ_win from joint_bisect, fed as atomic-ν, yields lb ≤ t_hi ≤ val(d).

    NOTE on structure: the scalar atomic-ν SDP built by solve_atomic_nu_sdp
    does NOT include window-localizing PSD constraints, so it is strictly
    weaker than joint_bisect's min-max (L_k^strong).  Concretely, at d=4
    order=3 the gap is ≈ 0.62 (joint_bisect t_hi ≈ 1.10 vs atomic-ν
    lb ≈ 0.49).  The aggregated constraint
        t · M_{k-1}(y) − M_{k-1}(q_λ y) ⪰ 0
    is a nonneg combination of the per-W constraints with weights λ_W, so
    removing it (as scalar atomic-ν does) is a relaxation.  The LP
    duality min_μ max_W q_W = max_λ min_μ Σ λ_W q_W holds on the CONTINUOUS
    simplex but NOT on the Lasserre moment hierarchy, because the
    hierarchy's window-localizing PSD constraints are matrix-valued and
    strictly stronger than their scalar (0,0) entries.

    What we test here is the weaker-but-correct statement: the returned
    lb_numerical is a valid lower bound (nonneg, ≤ t_hi up to solver slack).
    Matching val(d) requires a strong atomic-ν SDP with the aggregated
    window-localizing block and bisection on t — not yet implemented.
    """
    lam_win, t_hi = seed_from_joint_bisect(d=d, order=order,
                                           t_lo=1.0, t_hi=1.5, tol=1e-5)
    assert lam_win.shape == (len(list_windows(d)),)
    assert float(lam_win.sum()) == pytest.approx(1.0, abs=1e-6)

    res = solve_atomic_nu_sdp(lam_win, d=d, order=order, solver="auto",
                              compute_window_values=False)
    # Validity: nonneg, finite, ≤ t_hi (up to solver slack).
    assert res.lb_numerical >= 0.0
    assert np.isfinite(res.lb_numerical)
    assert res.lb_numerical <= t_hi + 1e-5, (
        f"scalar atomic-ν exceeds joint_bisect t_hi: {res.lb_numerical:.6f} "
        f"> {t_hi:.6f}"
    )


@requires_mosek
def test_seed_beats_uniform_d4_order2():
    """At d=4 order=2, the joint_bisect seed gives a strictly larger
    scalar atomic-ν bound than uniform ell=2 ν.  Confirms that ν-design
    matters even inside the weaker scalar relaxation."""
    d, order = 4, 2
    lam_uniform = uniform_grid_nu(d, K=2 * d - 1, ell=2)
    lam_seed, _ = seed_from_joint_bisect(d=d, order=order,
                                          t_lo=1.0, t_hi=1.5, tol=1e-5)
    r_u = solve_atomic_nu_sdp(lam_uniform, d=d, order=order, solver="auto",
                              compute_window_values=False)
    r_s = solve_atomic_nu_sdp(lam_seed, d=d, order=order, solver="auto",
                              compute_window_values=False)
    assert r_s.lb_numerical > r_u.lb_numerical + 1e-3, (
        f"seed {r_s.lb_numerical:.6f} should strictly dominate uniform "
        f"{r_u.lb_numerical:.6f} in the scalar relaxation"
    )
