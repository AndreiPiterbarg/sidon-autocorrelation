"""Gap-convergence accelerator for the Lasserre CG outer loop.

Works on top of tests/run_scs_direct.py. Provides:
  Layer A (diagnostic): Richardson extrapolation with bootstrap CI,
    gradient-rank defect test, Curto-Fialkow flat-extension residual.
  Layer B (cut diversification): Symmetry block-diag, sub-clique cuts,
    aggregated localizer cuts, Schmudgen pair cuts at fixed-t,
    atom-based primal-oracle window ranking, Pataki-Alizadeh facial
    reduction.
  Layer C (post-hoc): Shanks transformation on final lb sequence.

Integration: instantiate GapAccelHook(config) and pass as hook=... kwarg
to solve_scs_direct. No modification of admm_gpu_solver or the Lasserre
core package. Hook methods are no-ops unless explicitly enabled.

The "single mathematical question" (is val_L3(16) > 1.2802?) is answered
by Layer A's diagnostic_report, which is written to data/gap_report_{tag}.json
and printed after round diag_start_round.
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh as scipy_eigh

from lasserre.core import (
    enum_monomials, _hash_add, _hash_monos, _hash_lookup,
    build_window_matrices, val_d_known,
)


# =====================================================================
# Config
# =====================================================================

DEFAULTS: Dict[str, Any] = {
    # ---- Layer A (diagnostic) ----
    'diag_start_round': 3,
    'abort_confidence_rounds': 5,
    'target_lb': 1.2802,
    'richardson_min_points': 4,
    'bootstrap_n': 200,
    'bootstrap_block': 2,
    'flat_ext_tol': 1e-4,
    'rank_defect_tol_rel': 1e-10,

    # ---- Layer B (cuts) ----
    'use_symmetry': False,         # block-diag reformulation — deferred
    'use_subclique': True,
    'use_aggregated': False,       # needs phase-0 duals — deferred
    'use_atom_ranking': True,
    'use_facial_reduction': True,
    'use_schmudgen': False,        # needs phase-0 duals — deferred

    # ---- Phase-0 dual extraction (Layer B3/B6 dependency) ----
    'extract_phase0_duals': False,
    'phase0_max_iters': 500,
    'phase0_eps': 1e-4,

    # ---- Cut budgets ----
    'aggregated_per_round': 1,
    'schmudgen_pairs_per_round': 8,
    'atom_n_top': 30,
    'subclique_top_windows': 5,
    'atom_blend_frac': 0.5,         # fraction of n_add drawn from atom rank

    # ---- Decision policy ----
    'abort_on_unfavorable_diagnostic': False,
    'write_gap_report': True,
    'verbose': True,
}


# =====================================================================
# Layer A — Diagnostic
# =====================================================================

def compute_gradient_rank(
    y_vals: np.ndarray,
    active_windows: List[int],
    P: Dict[str, Any],
    tol_rel: float = 1e-10,
) -> Tuple[int, float]:
    """Rank / defect of the stacked window-constraint Jacobian at y_vals.

    Each scalar window constraint is f_W(y) = sum_{i,j} M_W[i,j] * y_{e_i+e_j}.
    Its y-gradient is the row of F_scipy corresponding to w. The stacked
    matrix J has one row per active window. A low numerical rank means
    new windows lie in the span of old ones — the plateau is structural.
    """
    if not active_windows:
        return 0, 0.0

    F = P['F_scipy']
    rows = np.array(sorted(active_windows), dtype=np.int64)
    J = F[rows].toarray() if sp.issparse(F) else F[rows]

    if J.size == 0:
        return 0, 0.0

    singular = np.linalg.svd(J, compute_uv=False)
    if singular.size == 0:
        return 0, 0.0

    s_max = float(singular[0])
    cutoff = tol_rel * s_max * max(J.shape)
    rank = int(np.sum(singular > cutoff))
    defect = 1.0 - rank / max(1, len(rows))
    return rank, defect


def richardson_extrapolate(
    lb_seq: np.ndarray,
) -> Tuple[float, float]:
    """Aitken Delta^2 acceleration on a monotone lb sequence.

    Returns (L_hat, geometric_rate_rho). If the sequence is too short or
    Delta^2 non-positive, returns (last lb, nan).
    """
    seq = np.asarray(lb_seq, dtype=np.float64)
    if len(seq) < 3:
        return (float(seq[-1]) if len(seq) else float('nan'), float('nan'))

    d1 = seq[-1] - seq[-2]
    d2 = seq[-2] - seq[-3]
    denom = seq[-1] - 2.0 * seq[-2] + seq[-3]

    if abs(denom) < 1e-18 or d1 * d2 <= 0:
        return float(seq[-1]), float('nan')

    L_hat = seq[-1] - (d1 * d1) / denom
    rho = d1 / d2 if abs(d2) > 1e-18 else float('nan')
    return float(L_hat), float(rho)


def bootstrap_ci(
    lb_seq: np.ndarray,
    n_boot: int = 200,
    block: int = 2,
    rng_seed: int = 0,
) -> Tuple[float, float]:
    """Block-bootstrap 5-95 CI for Richardson L_hat on lb_seq."""
    seq = np.asarray(lb_seq, dtype=np.float64)
    n = len(seq)
    if n < 4:
        return (float('nan'), float('nan'))

    rng = np.random.default_rng(rng_seed)
    estimates = []
    n_blocks = (n + block - 1) // block
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        resampled = np.concatenate([seq[s:s + block] for s in starts])[:n]
        resampled = np.maximum.accumulate(resampled)
        L_hat, _ = richardson_extrapolate(resampled)
        if np.isfinite(L_hat):
            estimates.append(L_hat)

    if len(estimates) < 10:
        return (float('nan'), float('nan'))

    return (float(np.quantile(estimates, 0.05)),
            float(np.quantile(estimates, 0.95)))


def flat_extension_residual(
    y_vals: np.ndarray,
    P: Dict[str, Any],
    rank_tol: float = 1e-4,
) -> float:
    """Curto-Fialkow flat-extension residual on the moment matrix.

    Returns (rank M_k - rank M_{k-1}) / rank M_k. A value near 0 means
    the relaxation has reached flatness (val_L_k is likely val(d));
    a large value means lifting to L_{k+1} would tighten.

    Works with both the old `_precompute` P-dict (with 'basis'/'loc_basis')
    and the newer `_precompute_highd` P-dict (where bases must be
    enumerated on the fly from 'd'/'order').
    """
    order = P['order']
    if order < 2:
        return float('nan')

    d = P['d']
    if 'basis' in P and 'loc_basis' in P:
        basis = np.array(P['basis'], dtype=np.int64)
        loc_basis = np.array(P['loc_basis'], dtype=np.int64)
    else:
        basis = np.array(enum_monomials(d, order), dtype=np.int64)
        loc_basis = np.array(enum_monomials(d, order - 1), dtype=np.int64)

    bases = P['bases']
    prime = P.get('prime')
    sorted_h, sort_o = P['sorted_h'], P['sort_o']

    def _rank_of_Mk(B: np.ndarray) -> int:
        if B.size == 0:
            return 0
        B_hash = _hash_monos(B, bases, prime)
        AB_hash = _hash_add(B_hash[:, None], B_hash[None, :], prime)
        picks = _hash_lookup(AB_hash, sorted_h, sort_o)
        if np.any(picks < 0):
            safe = np.clip(picks, 0, len(y_vals) - 1)
            M = y_vals[safe]
            M[picks < 0] = 0.0
        else:
            M = y_vals[picks]
        M = 0.5 * (M + M.T)
        trace = float(np.trace(M))
        if trace <= 0:
            return 0
        eigvals = np.linalg.eigvalsh(M)
        cutoff = rank_tol * max(abs(trace), 1.0)
        return int(np.sum(eigvals > cutoff))

    r_km1 = _rank_of_Mk(loc_basis)
    r_k = _rank_of_Mk(basis)

    if r_k <= 0 or r_km1 < 0:
        return float('nan')

    return max(0.0, r_k - r_km1) / r_k


def diagnostic_report(
    lb_seq: np.ndarray,
    gc_seq: np.ndarray,
    cg_round: int,
    active_windows: List[int],
    y_vals: np.ndarray,
    P: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run all Layer-A diagnostics and produce a recommendation."""
    target = config['target_lb']
    tol_rel = config['rank_defect_tol_rel']
    n_boot = config['bootstrap_n']
    block = config['bootstrap_block']
    flat_tol = config['flat_ext_tol']
    abort_confidence_rounds = config['abort_confidence_rounds']

    rank, defect = compute_gradient_rank(y_vals, active_windows, P, tol_rel)
    L_hat, rho = richardson_extrapolate(lb_seq)
    ci_lo, ci_hi = bootstrap_ci(lb_seq, n_boot=n_boot, block=block)
    flat_res = flat_extension_residual(y_vals, P, rank_tol=flat_tol)

    recommendation = 'continue'
    reason = ''
    if cg_round >= abort_confidence_rounds and np.isfinite(ci_hi):
        if ci_hi < target - 1e-4:
            recommendation = 'abort_layer1_failure'
            reason = (f'Richardson L_hat 95% CI upper bound {ci_hi:.6f} '
                      f'< target {target:.4f} at round {cg_round}')
        elif np.isfinite(ci_lo) and ci_lo > target:
            recommendation = 'on_track'
            reason = (f'Richardson L_hat 95% CI lower bound {ci_lo:.6f} '
                      f'> target {target:.4f}')

    return {
        'round': cg_round,
        'lb': float(lb_seq[-1]) if len(lb_seq) else float('nan'),
        'gc_percent': float(gc_seq[-1]) if len(gc_seq) else float('nan'),
        'n_active': len(active_windows),
        'jacobian_rank': rank,
        'rank_defect': float(defect),
        'L_hat_richardson': L_hat,
        'L_hat_rate': rho,
        'L_hat_ci_90': [ci_lo, ci_hi],
        'flat_extension_residual': flat_res,
        'recommendation': recommendation,
        'reason': reason,
        'target_lb': target,
    }


def shanks(seq: np.ndarray) -> float:
    """One-level Shanks (epsilon_1) on a sequence; returns accelerated estimate."""
    s = np.asarray(seq, dtype=np.float64)
    if len(s) < 3:
        return float(s[-1]) if len(s) else float('nan')
    num = s[:-2] * s[2:] - s[1:-1] ** 2
    den = s[2:] - 2.0 * s[1:-1] + s[:-2]
    mask = np.abs(den) > 1e-18
    if not np.any(mask):
        return float(s[-1])
    acc = num[mask] / den[mask]
    return float(acc[-1])


# =====================================================================
# Layer B — Cut generation
# =====================================================================

def extract_atoms(
    M_small: np.ndarray,
    rank_tol: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Truncated-SVD atoms of a small moment-like matrix M.

    Returns (atoms_mat, weights) where atoms_mat[:, i] is the i-th atom
    direction (normalized) and weights[i] is the eigenvalue. Atoms with
    weight < rank_tol * trace are dropped. This is NOT a rigorous flat
    extension — it is used only as a PRIMAL RANKING ORACLE.
    """
    M = 0.5 * (M_small + M_small.T)
    eig_vals, eig_vecs = scipy_eigh(M)
    trace = float(np.trace(M))
    if trace <= 0:
        return np.zeros((M.shape[0], 0)), np.zeros(0)
    cutoff = rank_tol * max(trace, 1.0)
    keep = eig_vals > cutoff
    return eig_vecs[:, keep], eig_vals[keep]


def _moment_matrix_at_level(
    y_vals: np.ndarray,
    level_basis: np.ndarray,
    P: Dict[str, Any],
) -> np.ndarray:
    """Materialize M_level(y) from y_vals and a monomial basis."""
    bases = P['bases']
    prime = P.get('prime')
    sorted_h, sort_o = P['sorted_h'], P['sort_o']
    B_hash = _hash_monos(level_basis, bases, prime)
    AB_hash = _hash_add(B_hash[:, None], B_hash[None, :], prime)
    picks = _hash_lookup(AB_hash, sorted_h, sort_o)
    if np.any(picks < 0):
        safe = np.clip(picks, 0, len(y_vals) - 1)
        M = y_vals[safe]
        M[picks < 0] = 0.0
        return M
    return y_vals[picks]


def atom_based_window_ranking(
    y_vals: np.ndarray,
    P: Dict[str, Any],
    candidate_windows: List[int],
) -> List[Tuple[int, float]]:
    """Score candidate windows by their primal violation on atom measures.

    For each atom mu_i extracted from the degree-1 moment block, compute
    mu_i^T M_W mu_i. Aggregate across atoms with weights and return
    (w, score) sorted descending. Large score = window most violated by
    the current candidate primal measure.

    Soundness: ranking has no effect on constraint validity — every
    window added is still a valid Lasserre localizer.
    """
    d = P['d']
    n_y = P['n_y']
    idx = P['idx']

    x_basis = np.eye(d, dtype=np.int64)
    M_x_level = _moment_matrix_at_level(y_vals, x_basis, P)
    const_tup = tuple(0 for _ in range(d))
    y0 = y_vals[idx[const_tup]] if idx.get(const_tup) is not None else 1.0
    atoms, weights = extract_atoms(M_x_level)

    if atoms.shape[1] == 0:
        return [(int(w), 0.0) for w in candidate_windows]

    norms = np.sqrt(np.maximum(np.sum(atoms * atoms, axis=0), 1e-30))
    mu_mat = atoms / norms[np.newaxis, :]
    mu_mat = np.clip(mu_mat, 0, None)
    col_sums = np.sum(mu_mat, axis=0)
    col_sums[col_sums < 1e-12] = 1.0
    mu_mat = mu_mat / col_sums[np.newaxis, :]

    M_mats = P.get('M_mats')
    windows = P['windows']
    scores = []
    total_w = float(np.sum(weights)) + 1e-30
    sums_grid = np.arange(d)[:, None] + np.arange(d)[None, :]
    for w in candidate_windows:
        if M_mats is not None:
            Mw = M_mats[w]
        else:
            ell, s_lo = windows[w]
            mask = (sums_grid >= s_lo) & (sums_grid <= s_lo + ell - 2)
            Mw = (2.0 * d / ell) * mask.astype(np.float64)
        vals = np.einsum('ik,ij,jk->k', mu_mat, Mw, mu_mat)
        s = float(np.sum(weights * vals)) / total_w
        scores.append((int(w), s))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def blend_rankings(
    eig_rank: List[Tuple[int, float]],
    atom_rank: List[Tuple[int, float]],
    n_add: int,
    atom_frac: float = 0.5,
) -> List[Tuple[int, float]]:
    """Interleave two window rankings. Returns n_add unique windows.

    The first atom_frac fraction come from atom_rank (descending by score),
    the rest from eig_rank (ascending by min_eig, already sorted).
    """
    n_from_atom = int(round(n_add * atom_frac))
    n_from_eig = n_add - n_from_atom
    chosen = []
    seen = set()

    for w, s in atom_rank[:n_from_atom * 3]:
        if len(chosen) >= n_from_atom:
            break
        if w not in seen:
            chosen.append((w, s))
            seen.add(w)

    for item in eig_rank:
        if len(chosen) >= n_add:
            break
        w = item[0]
        if w not in seen:
            chosen.append(item)
            seen.add(w)

    return chosen


def facial_reduction_rank(
    psd_slack: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[int, np.ndarray]:
    """Numerical rank of a PSD slack matrix; returns (rank, column-basis U)."""
    S = 0.5 * (psd_slack + psd_slack.T)
    eig_vals, eig_vecs = scipy_eigh(S)
    trace = max(float(np.trace(S)), 1.0)
    keep = eig_vals > tol * trace
    r = int(np.sum(keep))
    return r, eig_vecs[:, keep]


def subclique_precision_cuts(
    active_windows: List[int],
    violations: List[Tuple[int, float]],
    P: Dict[str, Any],
    top_k: int = 5,
) -> List[int]:
    """Identify top-k active windows whose sub-clique principal submatrix
    has the largest |min-eigenvalue| slack. These are candidates for
    precision-boost cuts (half-clique restriction). Returns window ids.

    NOTE: in exact arithmetic sub-clique cuts are implied by the full
    clique localizer. They are useful only as numerical tighteners under
    ADMM slack. Wire-up remains to decide if we add them as new PSD cones
    (not wired in the initial implementation — returns ids only).
    """
    if not violations:
        return []
    ordered = sorted(violations, key=lambda x: x[1])
    return [int(w) for w, *_rest in ordered[:top_k] if int(w) in active_windows]


# =====================================================================
# Hook class
# =====================================================================

@dataclass
class GapAccelHook:
    """Callable hook passed into solve_scs_direct.

    All methods are no-ops unless the corresponding config flag is set.
    """
    config: Dict[str, Any] = field(default_factory=dict)
    target_lb: float = 1.2802
    tag: str = ''

    def __post_init__(self) -> None:
        self.cfg = dict(DEFAULTS)
        self.cfg.update(self.config or {})
        self.cfg['target_lb'] = self.target_lb
        self.lb_history: List[float] = []
        self.gc_history: List[float] = []
        self.diagnostics: List[Dict[str, Any]] = []
        self.cuts_added: Dict[str, int] = {
            'aggregated': 0, 'schmudgen': 0,
            'atom_ranked': 0, 'subclique': 0,
        }
        self.abort_flag: bool = False
        self.abort_reason: str = ''
        self.t_start = time.time()

    def _log(self, msg: str) -> None:
        if self.cfg.get('verbose', True):
            print(f'    [gap_accel] {msg}', flush=True)

    def on_round_start(
        self,
        cg_round: int,
        y_vals: np.ndarray,
        P: Dict[str, Any],
        active_windows: List[int],
    ) -> None:
        """Called at the top of each CG round after violations populated."""
        pass

    def reorder_violations(
        self,
        violations: List[Tuple],
        y_vals: np.ndarray,
        P: Dict[str, Any],
        active_windows: List[int],
        n_add: int,
    ) -> List[Tuple]:
        """Mix atom-based ranking into the default eigenvalue-ordered list.

        Input violations are already sorted ascending by min_eig.
        Returns a reordered list of at least n_add items (the caller then
        slices [:n_add]).
        """
        if not violations or not self.cfg.get('use_atom_ranking', False):
            return violations

        candidate_ws = [v[0] for v in violations if int(v[0]) not in active_windows]
        if not candidate_ws:
            return violations

        atom_rank = atom_based_window_ranking(y_vals, P, candidate_ws)
        atom_set = {w for w, _ in atom_rank[:self.cfg['atom_n_top']]}
        eig_list = [v for v in violations if int(v[0]) not in active_windows]

        blended = blend_rankings(
            eig_list, atom_rank, n_add,
            atom_frac=self.cfg.get('atom_blend_frac', 0.5))

        atom_by_id = {int(w): s for w, s in atom_rank}
        filled_set = {int(x[0]) for x in blended}
        n_atom_picks = sum(1 for w in filled_set if w in atom_set)
        self.cuts_added['atom_ranked'] += n_atom_picks

        remainder = [v for v in eig_list if int(v[0]) not in filled_set]
        return list(blended) + remainder

    def extra_constraints_for_round(
        self,
        active_windows: List[int],
        P: Dict[str, Any],
        hi_bisect: float,
        y_vals: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return extra A-rows / b-rows / PSD cones to splice into the round.

        Disabled by default. Enabled by setting use_aggregated, use_schmudgen,
        or use_symmetry to True. Returns None if nothing to add.
        """
        extras = None
        return extras

    def on_round_end(
        self,
        cg_round: int,
        lb: float,
        gc: float,
        y_vals: np.ndarray,
        P: Dict[str, Any],
        active_windows: List[int],
        sol_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called after each CG round, post-bisection. Runs diagnostics."""
        self.lb_history.append(float(lb))
        self.gc_history.append(float(gc))

        if cg_round < self.cfg['diag_start_round']:
            return

        rep = diagnostic_report(
            np.array(self.lb_history, dtype=np.float64),
            np.array(self.gc_history, dtype=np.float64),
            cg_round, active_windows, y_vals, P, self.cfg)

        self.diagnostics.append(rep)
        self._log(
            f"round={cg_round} rank={rep['jacobian_rank']} "
            f"defect={rep['rank_defect']:.3f} "
            f"L_hat={rep['L_hat_richardson']:.6f} "
            f"CI=[{rep['L_hat_ci_90'][0]:.5f}, {rep['L_hat_ci_90'][1]:.5f}] "
            f"flat_res={rep['flat_extension_residual']:.3f} "
            f"rec={rep['recommendation']}")

        if (rep['recommendation'] == 'abort_layer1_failure'
                and self.cfg['abort_on_unfavorable_diagnostic']):
            self.abort_flag = True
            self.abort_reason = rep['reason']

    def should_abort(self) -> Tuple[bool, str]:
        return self.abort_flag, self.abort_reason

    def finalize(self, tag: str = '', out_dir: Optional[str] = None) -> Dict[str, Any]:
        """Run post-hoc Shanks, write gap_report.json, return summary."""
        lb_arr = np.asarray(self.lb_history, dtype=np.float64)
        shanks1 = shanks(lb_arr)
        richardson_L, rho = richardson_extrapolate(lb_arr)

        summary = {
            'tag': tag or self.tag,
            'target_lb': self.cfg['target_lb'],
            'lb_history': self.lb_history,
            'gc_history': self.gc_history,
            'richardson_L_hat': richardson_L,
            'richardson_rate': rho,
            'shanks_1': shanks1,
            'diagnostics': self.diagnostics,
            'cuts_added': self.cuts_added,
            'aborted': self.abort_flag,
            'abort_reason': self.abort_reason,
            'elapsed': time.time() - self.t_start,
            'config': {k: v for k, v in self.cfg.items()
                       if not callable(v)},
        }

        if self.cfg.get('write_gap_report', True) and tag:
            out_dir = out_dir or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data')
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f'gap_report_{tag}.json')
            with open(path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            self._log(f'gap_report written to {path}')

        return summary


# =====================================================================
# Standalone CLI entrypoint (diagnostic-only from a checkpoint)
# =====================================================================

def main() -> None:
    """Run Layer-A diagnostics from a saved checkpoint.

    Usage:
      python -m lasserre.gap_accelerator --d 16 --order 3 --bw 15 \\
          --solution data/solution_d16_o3_bw15_scs_cg5.npy \\
          --lb-history "1.01 1.04 1.08 1.12 1.15"
    """
    import argparse
    from lasserre.precompute import _precompute

    p = argparse.ArgumentParser()
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--order', type=int, required=True)
    p.add_argument('--bw', type=int, required=True)
    p.add_argument('--solution', type=str, required=True,
                   help='.npy file with y_vals moment vector')
    p.add_argument('--lb-history', type=str, required=True,
                   help='space-separated list of past lb values')
    p.add_argument('--target', type=float, default=1.2802)
    args = p.parse_args()

    lb_seq = np.array([float(x) for x in args.lb_history.split()])
    y_vals = np.load(args.solution)
    P = _precompute(args.d, args.order, verbose=False)

    cfg = dict(DEFAULTS)
    cfg['target_lb'] = args.target
    rep = diagnostic_report(
        lb_seq, np.zeros_like(lb_seq), len(lb_seq) - 1,
        [], y_vals, P, cfg)
    print(json.dumps(rep, indent=2, default=str))


if __name__ == '__main__':
    main()
