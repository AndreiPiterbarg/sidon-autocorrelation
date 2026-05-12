"""Newton-polytope SEED for the Pólya-Handelman LP, and helpers.

Provides a starting Sigma_R (constraint multi-indices) and B_R (q-poly
indices) that the column-generation solver in cutting_plane.py uses as
its initial restricted set. The naive Newton-polytope restriction is
NOT formally sound for our inhomogeneous LP (the q-multiplier breaks
the agent's lemma — see AUDIT.md discussion); soundness must be
recovered empirically via violation checking and CG expansion.

Definitions (notation matches lasserre/polya_lp/build.py):
  A      = union over windows W of {beta = e_i + e_j : (i,j) in supp(M_W)}
  Sigma_R^{(0)} = (A oplus Delta_{R-2}) cap {beta : |beta| <= R}
                  cup {beta : |beta| <= 1}     [minimal seed; the linear
                                               degree-1 indices are needed
                                               to absorb the q multiplier]
  B_R^{(0)} = Sigma_R^{(0)} cap {gamma : |gamma| <= R-1}

After solving the restricted LP, cutting_plane.find_violators() computes
the *full* LHS at every dropped beta, and any beta with c_beta < -tol is
added to Sigma_R^{(k+1)}. Iterate until no violations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple, FrozenSet, Optional
import time
import numpy as np

from lasserre.polya_lp.poly import enum_monomials_le, multinomial


@dataclass
class TermSparsitySupport:
    """Output of build_term_sparsity_support."""
    Sigma_R: List[Tuple[int, ...]]   # constraint multi-indices (|beta| <= R)
    B_R: List[Tuple[int, ...]]       # q-polynomial multi-indices (|K| <= R-1)
    A: List[Tuple[int, ...]]         # polynomial support (|beta| = 2)
    n_var_dim: int                   # ambient dimension d
    R: int
    seed_iter: int                   # which TS iteration this is
    n_constraints: int
    n_q_vars: int
    full_n_constraints: int          # what the unrestricted count would be
    full_n_q_vars: int
    notes: str = ""

    def __repr__(self) -> str:
        return (f"TermSparsitySupport(d={self.n_var_dim}, R={self.R}, "
                f"|Sigma_R|={self.n_constraints}/{self.full_n_constraints}, "
                f"|B_R|={self.n_q_vars}/{self.full_n_q_vars}, "
                f"|A|={len(self.A)})")


def polynomial_support_from_M_mats(
    M_mats: Sequence[np.ndarray],
    eps: float = 0.0,
) -> List[Tuple[int, ...]]:
    """Compute A = union over W of {e_i + e_j : (i,j) in supp(M_W)}.

    These are exactly the |beta|=2 multi-indices where the polynomial
    p(mu) = mu^T M(lambda) mu has a (potentially) nonzero coefficient.
    """
    if not M_mats:
        return []
    d = M_mats[0].shape[0]
    # Stack all M_mats and find any-nonzero positions.
    M_stack = np.stack([np.abs(M) for M in M_mats], axis=0)
    any_nonzero = (M_stack > eps).any(axis=0)  # (d, d) bool
    # Symmetrize (in case some M_mats aren't perfectly symmetric numerically).
    any_nonzero = any_nonzero | any_nonzero.T

    A_set: Set[Tuple[int, ...]] = set()
    nz_i, nz_j = np.where(any_nonzero)
    for i, j in zip(nz_i.tolist(), nz_j.tolist()):
        beta = [0] * d
        if i == j:
            beta[i] = 2
        else:
            # only include i<=j to avoid duplicates
            if i > j:
                continue
            beta[i] = 1
            beta[j] = 1
        A_set.add(tuple(beta))
    return sorted(A_set)


def minkowski_with_simplex(
    A: Sequence[Tuple[int, ...]],
    R_minus_2: int,
    d: int,
) -> List[Tuple[int, ...]]:
    """Compute (A oplus Delta_{R-2}) cap {beta in N^d : |beta| <= R}.

    Delta_{R-2} = {gamma in N^d : |gamma| <= R-2}.
    The Minkowski sum of A with Delta_{R-2} is { a + g : a in A, |g| <= R-2 }.
    Each a in A has |a|=2, so |a+g| <= R.
    """
    if R_minus_2 < 0:
        return [tuple(a) for a in A]
    Delta = enum_monomials_le(d, R_minus_2)
    sums: Set[Tuple[int, ...]] = set()
    for a in A:
        for g in Delta:
            sums.add(tuple(ai + gi for ai, gi in zip(a, g)))
    return sorted(sums)


def compute_B_R(
    Sigma_R: Iterable[Tuple[int, ...]],
    d: int,
    R: int,
) -> List[Tuple[int, ...]]:
    """B_R = {gamma : |gamma| <= R-1, gamma + e_j in Sigma_R for some j}
              cup {gamma in Sigma_R, |gamma| <= R-1}."""
    Sigma_set = set(Sigma_R)
    B_set: Set[Tuple[int, ...]] = set()
    for beta in Sigma_set:
        if sum(beta) <= R - 1:
            B_set.add(beta)
        # Predecessors: beta - e_j for each j with beta_j > 0
        for j, b in enumerate(beta):
            if b > 0:
                pred = list(beta)
                pred[j] -= 1
                if sum(pred) <= R - 1:
                    B_set.add(tuple(pred))
    return sorted(B_set)


def build_term_sparsity_support(
    M_mats: Sequence[np.ndarray],
    R: int,
    eps: float = 0.0,
    include_low_degree: bool = True,
) -> TermSparsitySupport:
    """Compute Sigma_R^{(0)} and B_R^{(0)} (initial seed for column generation).

    include_low_degree: if True, append all |beta| <= 1 to Sigma_R. The
    degree-0 index (0,...,0) is needed for alpha; the degree-1 indices
    are needed to absorb the q multiplier (otherwise the LP can satisfy
    constraints at degree-1 only via free q values, which then propagate
    to violate dropped constraints in the c_beta = q_beta - sum q_{beta-e_j}
    structure). Empirically without these, the LP often has dropped
    constraints with c_beta = q_K - q_0 < 0 in the extension.
    """
    if not M_mats:
        raise ValueError("Need at least one window matrix.")
    d = M_mats[0].shape[0]
    A = polynomial_support_from_M_mats(M_mats, eps=eps)
    Sigma_R = minkowski_with_simplex(A, R - 2, d)
    Sigma_set = set(Sigma_R)
    # Always include the zero index (alpha lives in row beta=0).
    zero = tuple([0] * d)
    Sigma_set.add(zero)
    # Optionally include degree-1 indices (recommended; cheap, helps CG).
    if include_low_degree:
        for k in range(d):
            ek = tuple(1 if j == k else 0 for j in range(d))
            Sigma_set.add(ek)
    Sigma_R = sorted(Sigma_set)
    B_R = compute_B_R(Sigma_R, d, R)

    from math import comb
    full_n_eq = comb(d + R, R)
    full_n_q = comb(d + R - 1, R - 1)

    return TermSparsitySupport(
        Sigma_R=Sigma_R,
        B_R=B_R,
        A=A,
        n_var_dim=d,
        R=R,
        seed_iter=0,
        n_constraints=len(Sigma_R),
        n_q_vars=len(B_R),
        full_n_constraints=full_n_eq,
        full_n_q_vars=full_n_q,
        notes=f"Seed (Newton + |beta|<=1)" if include_low_degree else "Seed (Newton only)",
    )


def expand_support(
    support: TermSparsitySupport,
    binding_betas: Iterable[Tuple[int, ...]],
    expand_radius: int = 1,
) -> TermSparsitySupport:
    """Iterative TS step: expand Sigma_R to include lattice neighbors of
    binding constraints.

    expand_radius=1: add beta + e_j - e_{j'} and beta + e_j (within |.|<=R).
    expand_radius=2: also add 2-step neighbors (rarely needed).
    """
    d = support.n_var_dim
    R = support.R
    Sigma_set: Set[Tuple[int, ...]] = set(support.Sigma_R)
    bind = list(binding_betas)
    for beta in bind:
        if sum(beta) > R:
            continue
        # Up-shifts: add beta + e_j (within degree R)
        for j in range(d):
            shift = list(beta)
            shift[j] += 1
            if sum(shift) <= R:
                Sigma_set.add(tuple(shift))
        # Lateral: beta + e_j - e_{j'} for valid j, j'
        for j in range(d):
            for jp in range(d):
                if j == jp:
                    continue
                if beta[jp] == 0:
                    continue
                lat = list(beta)
                lat[j] += 1
                lat[jp] -= 1
                if sum(lat) <= R:
                    Sigma_set.add(tuple(lat))
        if expand_radius >= 2:
            # Two-step neighbors via repeated expansion (omitted for now;
            # second TS round usually suffices).
            pass

    # Recompute B_R from new Sigma_R
    Sigma_list = sorted(Sigma_set)
    B_list = compute_B_R(Sigma_list, d, R)
    return TermSparsitySupport(
        Sigma_R=Sigma_list,
        B_R=B_list,
        A=support.A,
        n_var_dim=d,
        R=R,
        seed_iter=support.seed_iter + 1,
        n_constraints=len(Sigma_list),
        n_q_vars=len(B_list),
        full_n_constraints=support.full_n_constraints,
        full_n_q_vars=support.full_n_q_vars,
        notes=f"TS iteration {support.seed_iter + 1}",
    )


def support_intersection_count(
    M: np.ndarray,
    support_set: Set[Tuple[int, ...]],
    eps: float = 0.0,
) -> int:
    """Count how many entries of M have their corresponding |beta|=2
    multi-index in support_set. Used for diagnostics."""
    d = M.shape[0]
    count = 0
    for i in range(d):
        for j in range(i, d):
            if abs(M[i, j]) <= eps:
                continue
            if i == j:
                beta = tuple(2 if k == i else 0 for k in range(d))
            else:
                beta = tuple(1 if (k == i or k == j) else 0 for k in range(d))
            if beta in support_set:
                count += 1
    return count
