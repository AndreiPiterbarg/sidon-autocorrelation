"""CCTR setup: extract KKT-correct alpha and build aggregate M_int once
per BnB run.

Workflow:
  1. Get mu_star (a μ near the global minimum of max_W TV_W). May be:
     (a) loaded from `mu_star_d{d}.npz` if available, or
     (b) computed fresh via `kkt_correct_mu_star.find_kkt_correct_mu_star`.
  2. Solve QP for KKT alpha at mu_star (via kkt_residual_qp).
  3. Identify active windows (those with TV close to max).
  4. Build M_int aggregate via build_cctr_aggregate_int.

The α is computed once at startup. CCTR uses this fixed α for ALL boxes.
This is sound for any valid α ∈ Δ_active (need not be exactly KKT).
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np

from .bound_cctr import build_cctr_aggregate_int, D_M_DEFAULT
from .windows import build_windows


def _try_load_mu_star(d: int, search_paths) -> Optional[np.ndarray]:
    fname = f"mu_star_d{d}.npz"
    for p in search_paths:
        path = os.path.join(p, fname)
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            keys = list(data.keys())
            for k in ("mu_star", "mu"):
                if k in keys:
                    return np.asarray(data[k], dtype=np.float64)
    return None


def _compute_mu_star_fresh(d: int, n_starts=None,
                             n_workers: int = 4) -> np.ndarray:
    """Run kkt_correct_mu_star pipeline to get a μ* ∈ Δ_d.

    Auto-scales n_starts and worker count by d to avoid expensive runs at
    small d (where val(d) is well known anyway).
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kkt_correct_mu_star import find_kkt_correct_mu_star
    if n_starts is None:
        if d <= 6:
            n_starts = 200
        elif d <= 10:
            n_starts = 400
        elif d <= 14:
            n_starts = 800
        else:
            n_starts = 1600
    n_workers = max(1, min(n_workers, n_starts // 50))
    res = find_kkt_correct_mu_star(
        d=d, x_cap=1.0, n_starts=n_starts, n_workers=n_workers,
        top_K_phase2=50, top_K_phase3=10, target_residual=1e-6, verbose=False,
    )
    if res['mu_star'] is None:
        raise RuntimeError(f"kkt_correct_mu_star pipeline failed at d={d}")
    return np.asarray(res['mu_star'], dtype=np.float64)


def _load_full_mu_data(d: int, search_paths) -> Optional[dict]:
    """Load saved npz with mu_star, alpha_star, active_idx if all present."""
    fname = f"mu_star_d{d}.npz"
    for p in search_paths:
        path = os.path.join(p, fname)
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            keys = list(data.keys())
            mu_key = "mu_star" if "mu_star" in keys else (
                "mu" if "mu" in keys else None
            )
            if mu_key is None:
                continue
            out = {'mu_star': np.asarray(data[mu_key], dtype=np.float64)}
            if 'alpha_star' in keys:
                out['alpha_star'] = np.asarray(data['alpha_star'], dtype=np.float64)
            if 'active_idx' in keys:
                out['active_idx'] = np.asarray(data['active_idx']).tolist()
            return out
    return None


def setup_cctr(
    d: int, *,
    mu_star: Optional[np.ndarray] = None,
    tol_active: float = 1e-3,
    D_M: int = D_M_DEFAULT,
    search_paths=None,
):
    """Compute CCTR aggregate M_int for use by rigor cert at d.

    Returns dict with:
      'M_int': (d, d) integer matrix at denom D_M.
      'D_M': denom.
      'alpha': float array of length n_active (normalized).
      'active_idx': list of int (indices into windows).
      'mu_star': float array (the μ used to define active set).

    If `mu_star` is None, attempts to load from mu_star_d{d}.npz under
    `search_paths`; if not found, computes fresh.

    PRIORITY ORDER for α / active set:
      1. Loaded from npz if alpha_star and active_idx are saved.
      2. Else: re-compute via kkt_residual_qp at mu_star.
    """
    if search_paths is None:
        here = os.path.dirname(os.path.abspath(__file__))
        search_paths = [
            os.path.dirname(here),  # repo root
            os.path.expanduser("~"),
            ".",
        ]
    saved_full = None
    if mu_star is None:
        saved_full = _load_full_mu_data(d, search_paths)
        if saved_full is not None:
            mu_star = saved_full['mu_star']
    if mu_star is None:
        mu_star = _compute_mu_star_fresh(d)
    mu_star = np.asarray(mu_star, dtype=np.float64)
    assert mu_star.ndim == 1 and mu_star.shape[0] == d, (
        f"mu_star shape {mu_star.shape}, expected ({d},)")
    assert abs(mu_star.sum() - 1.0) < 1e-6, f"mu_star sum {mu_star.sum()} != 1"

    # Use saved (alpha, active_idx) if available; else recompute.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kkt_correct_mu_star import build_window_data, kkt_residual_qp
    A_stack, c_W_arr = build_window_data(d)
    if (saved_full is not None and 'alpha_star' in saved_full
            and 'active_idx' in saved_full):
        alpha_kkt = saved_full['alpha_star']
        active_in_kkt = saved_full['active_idx']
        # Recompute residual diagnostically
        res_dict = kkt_residual_qp(mu_star, A_stack, c_W_arr,
                                    x_cap=1.0, tol_active=tol_active)
    else:
        res_dict = kkt_residual_qp(mu_star, A_stack, c_W_arr,
                                      x_cap=1.0, tol_active=tol_active)
        alpha_kkt = res_dict['alpha']
        active_in_kkt = res_dict['active_idx']
    if alpha_kkt is None or len(active_in_kkt) == 0:
        raise RuntimeError(f"kkt_residual_qp returned no active windows at d={d}")

    # Map kkt's window indices to interval_bnb's window list.
    # Both enumerate windows with the same (ell, s) order:
    # interval_bnb.windows.build_windows uses lasserre.core which enumerates
    # in some order; kkt_correct_mu_star.build_window_data uses
    # for ell in range(2, 2d+1): for s in range(2d-ell+1).
    # The interval_bnb WindowMeta has fields ell and s_lo. Match by (ell, s).
    bnb_windows = build_windows(d)
    bnb_idx_by_key = {(w.ell, w.s_lo): i for i, w in enumerate(bnb_windows)}
    # Reconstruct (ell, s) for each kkt window index.
    kkt_window_ells_ss = []
    for ell in range(2, 2 * d + 1):
        for s in range(2 * d - ell + 1):
            kkt_window_ells_ss.append((ell, s))

    active_bnb_idx = []
    alpha_for_bnb = []
    for k_local, kkt_w_idx in enumerate(active_in_kkt):
        ell, s = kkt_window_ells_ss[int(kkt_w_idx)]
        bnb_idx = bnb_idx_by_key.get((ell, s))
        if bnb_idx is None:
            raise RuntimeError(
                f"kkt window (ell={ell}, s={s}) not found in interval_bnb "
                f"window list; window enumerations diverged"
            )
        active_bnb_idx.append(bnb_idx)
        alpha_for_bnb.append(float(alpha_kkt[k_local]))

    # Build M_int aggregate using interval_bnb's WindowMeta objects.
    active_bnb_windows = [bnb_windows[i] for i in active_bnb_idx]
    M_int, D_M_actual = build_cctr_aggregate_int(
        alpha_for_bnb, active_bnb_windows, d, D_M=D_M,
    )
    return {
        'M_int': M_int,
        'D_M': D_M_actual,
        'alpha': np.asarray(alpha_for_bnb, dtype=np.float64),
        'active_idx': active_bnb_idx,
        'mu_star': mu_star,
        'kkt_residual': float(res_dict['residual']),
        'val_max': float(res_dict['val_max']),
        'name': 'kkt',
    }


def setup_multi_cctr(
    d: int, *,
    mu_star: Optional[np.ndarray] = None,
    tol_active: float = 1e-3,
    D_M: int = D_M_DEFAULT,
    search_paths=None,
    include: tuple = ('kkt', 'uniform_active', 'all_windows'),
):
    """Build MULTIPLE CCTR aggregates with different α weightings.

    Each aggregate is independently sound (α >= 0, sum α = 1). The BnB
    can take MAX over their LBs, certifying a box if ANY aggregate's LB
    exceeds the target. Different aggregates tighten different regions:

      'kkt'             — KKT-correct α at μ*. Tight near μ*.
      'uniform_active'  — uniform α over active windows at μ*.
                          Tight in a wider neighborhood of μ*.
      'all_windows'     — uniform α over all 2d(2d−1)/2 windows.
                          Tight near simplex centroid; loosest at boundary.

    Returns: list of context dicts (one per aggregate). Each has the same
    schema as `setup_cctr`'s output, plus a 'name' field identifying the
    aggregate strategy. Always at least 1 aggregate (the kkt one).
    """
    out = []
    base_ctx = setup_cctr(
        d, mu_star=mu_star, tol_active=tol_active, D_M=D_M,
        search_paths=search_paths,
    )
    if 'kkt' in include:
        out.append(base_ctx)

    if 'uniform_active' in include:
        bnb_windows = build_windows(d)
        active_bnb_w = [bnb_windows[i] for i in base_ctx['active_idx']]
        n_a = len(active_bnb_w)
        if n_a > 0:
            alpha_uniform = np.ones(n_a, dtype=np.float64) / n_a
            M_int_u, _ = build_cctr_aggregate_int(
                alpha_uniform, active_bnb_w, d, D_M=D_M,
            )
            out.append({
                'M_int': M_int_u, 'D_M': D_M, 'alpha': alpha_uniform,
                'active_idx': list(base_ctx['active_idx']),
                'mu_star': base_ctx['mu_star'],
                'kkt_residual': float('nan'),
                'val_max': base_ctx['val_max'],
                'name': 'uniform_active',
            })

    if 'all_windows' in include:
        bnb_windows = build_windows(d)
        n_w = len(bnb_windows)
        alpha_all = np.ones(n_w, dtype=np.float64) / n_w
        M_int_all, _ = build_cctr_aggregate_int(
            alpha_all, bnb_windows, d, D_M=D_M,
        )
        out.append({
            'M_int': M_int_all, 'D_M': D_M, 'alpha': alpha_all,
            'active_idx': list(range(n_w)),
            'mu_star': base_ctx['mu_star'],
            'kkt_residual': float('nan'),
            'val_max': float('nan'),
            'name': 'all_windows',
        })

    # Boundary-face anchors: aggregates for regions where some μ_i ≈ 0.
    # For each index i with mu_star[i] BELOW some threshold (i.e., already
    # near-zero at μ*), the BnB will produce many boxes near that face.
    # Building an aggregate at the face midpoint (μ with μ_i = 0 and other
    # entries spread proportionally to mu_star) tightens those boxes.
    if 'boundary_faces' in include:
        bnb_windows = build_windows(d)
        bnb_idx_by_key = {(w.ell, w.s_lo): i for i, w in enumerate(bnb_windows)}
        kkt_window_ells_ss = []
        for ell in range(2, 2 * d + 1):
            for s in range(2 * d - ell + 1):
                kkt_window_ells_ss.append((ell, s))
        # Use top-K LARGEST mu_star indices and zero them (force "drop"
        # of the dominant index — explore a different region).
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from kkt_correct_mu_star import build_window_data, kkt_residual_qp
        A_stack, c_W_arr = build_window_data(d)
        mu_star_arr = base_ctx['mu_star']
        # Pick the top-2 (by magnitude) and bottom-2 (smallest non-zero)
        # indices to anchor face aggregates.
        sort_idx = np.argsort(mu_star_arr)
        anchors_idx = list(sort_idx[-2:]) + [
            i for i in sort_idx[:2] if mu_star_arr[i] > 1e-6
        ]
        for face_i in anchors_idx:
            face_mu = mu_star_arr.copy()
            face_mu[face_i] = 0.0
            s = face_mu.sum()
            if s < 1e-9:
                continue
            face_mu = face_mu / s  # renormalize to simplex
            try:
                res_face = kkt_residual_qp(
                    face_mu, A_stack, c_W_arr,
                    x_cap=1.0, tol_active=1e-2,  # looser to grab more windows
                )
            except Exception:
                continue
            if res_face['alpha'] is None or len(res_face['active_idx']) == 0:
                continue
            face_alpha = res_face['alpha']
            face_active = []
            for kkt_w_idx in res_face['active_idx']:
                ell, s_lo = kkt_window_ells_ss[int(kkt_w_idx)]
                bnb_idx = bnb_idx_by_key.get((ell, s_lo))
                if bnb_idx is not None:
                    face_active.append(bnb_idx)
            if not face_active:
                continue
            face_w = [bnb_windows[i] for i in face_active]
            M_face_int, _ = build_cctr_aggregate_int(
                face_alpha[:len(face_active)], face_w, d, D_M=D_M,
            )
            out.append({
                'M_int': M_face_int, 'D_M': D_M,
                'alpha': face_alpha[:len(face_active)],
                'active_idx': face_active,
                'mu_star': face_mu,
                'kkt_residual': float(res_face['residual']),
                'val_max': float(res_face['val_max']),
                'name': f'face_{int(face_i)}',
            })

    return out
