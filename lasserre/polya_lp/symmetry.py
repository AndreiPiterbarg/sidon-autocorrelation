"""Z/2 symmetry reduction for the Sidon problem.

The Sidon problem at d points is invariant under the involution
sigma(i) = d - 1 - i (reflection of the discretization grid). The
optimal mu is symmetric: mu_i = mu_{d-1-i}. We reduce the LP by
restricting to the symmetric subspace.

Variable change for d even:
  nu_i = mu_i = mu_{d-1-i},   i = 0, ..., d/2 - 1
  sum mu_i = 2 sum nu_i
For d odd:
  nu_i = mu_i = mu_{d-1-i},   i = 0, ..., (d-1)/2 - 1
  nu_{(d-1)/2} = mu_{(d-1)/2}
  sum mu_i = 2 sum_{i < (d-1)/2} nu_i + nu_{(d-1)/2}

Window matrices project as:
  M_W^sym[i][j] = (M_W[i][j] + M_W[d-1-i][d-1-j]) restricted to symmetric basis
"""
from __future__ import annotations
from typing import Tuple, List
import numpy as np


def z2_dim(d: int) -> int:
    """Effective dimension after Z/2 reduction."""
    return (d + 1) // 2  # ceil(d/2)


def z2_index_map(d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (orbit_id, orbit_size) for each i in 0..d-1.

    orbit_id[i] in 0..z2_dim(d)-1 maps i -> its symmetric coordinate.
    orbit_size[orbit_id] in {1, 2}: 2 if the orbit has 2 elements,
    1 only for the central index when d is odd.
    """
    n = z2_dim(d)
    orbit_id = np.zeros(d, dtype=np.int64)
    orbit_size = np.ones(n, dtype=np.int64)
    for i in range(d):
        sigma = d - 1 - i
        oid = min(i, sigma)
        orbit_id[i] = oid
        if i != sigma:
            orbit_size[oid] = 2
    return orbit_id, orbit_size


def z2_symmetric_basis(d: int) -> dict:
    """Return data needed to reformulate the LP in symmetric variables.

    nu_i = mu_{orbit_id^{-1}(i)} for each orbit i.
    sum_i mu_i  =  sum_i orbit_size[i] * nu_i.
    """
    orbit_id, orbit_size = z2_index_map(d)
    return {
        "d": d,
        "n_sym": z2_dim(d),
        "orbit_id": orbit_id,
        "orbit_size": orbit_size,
    }


def project_M_to_z2(M: np.ndarray) -> np.ndarray:
    """Symmetrize d x d matrix M and project to symmetric basis.

    sum_{i,j} M[i,j] mu_i mu_j = sum_{a,b} M_sym[a,b] nu_a nu_b
    when mu is symmetric.
    """
    d = M.shape[0]
    orbit_id, _ = z2_index_map(d)
    n_sym = z2_dim(d)
    M_sym = np.zeros((n_sym, n_sym), dtype=M.dtype)
    np.add.at(M_sym, (orbit_id[:, None], orbit_id[None, :]), M)
    return M_sym


def project_window_set_to_z2(M_mats: List[np.ndarray]) -> Tuple[List[np.ndarray], List[int]]:
    """Project a list of window matrices to the symmetric basis (no rescaling).

    Two windows that are sigma-images of each other yield the same M_sym;
    de-duplicate via byte-hash (O(n_W * d^2) total).

    Note: returned matrices are NOT rescaled for the standard simplex.
    """
    if not M_mats:
        return [], []
    d = M_mats[0].shape[0]
    n_sym = z2_dim(d)
    orbit_id, _ = z2_index_map(d)

    # Vectorized projection: project all windows at once.
    M_arr = np.stack([np.asarray(M, dtype=np.float64) for M in M_mats], axis=0)  # (n_W, d, d)
    n_W = M_arr.shape[0]
    sym_arr = np.zeros((n_W, n_sym, n_sym), dtype=np.float64)
    # For each (i, j), accumulate into (orbit_id[i], orbit_id[j]).
    np.add.at(sym_arr, (slice(None), orbit_id[:, None], orbit_id[None, :]), M_arr)

    # Hash each n_sym x n_sym matrix to find duplicates.
    # Round to avoid floating-point noise; matrix entries are exact rationals 2*d/ell.
    rounded = np.round(sym_arr, decimals=10)
    keys = [r.tobytes() for r in rounded]

    seen: dict = {}
    unique: List[np.ndarray] = []
    counts: List[int] = []
    for k, key in enumerate(keys):
        if key in seen:
            idx = seen[key]
            counts[idx] += 1
        else:
            seen[key] = len(unique)
            unique.append(sym_arr[k])
            counts.append(1)
    return unique, counts


def rescale_for_standard_simplex(M_sym: np.ndarray, d: int) -> np.ndarray:
    """Rescale M_sym so that the LP can use the standard simplex sum nu = 1.

    The Z/2-symmetric problem lives on the weighted simplex
      sum_a orbit_size[a] * nu_a = 1.
    Substitute nu' = D nu (D = diag(orbit_size)) so sum nu' = 1.
    Then nu^T M_sym nu = (nu')^T D^{-1} M_sym D^{-1} nu', so the matrix
    we feed the LP is M_tilde = D^{-1} M_sym D^{-1}.
    """
    _, orbit_size = z2_index_map(d)
    Dinv = 1.0 / orbit_size.astype(np.float64)
    return (Dinv[:, None] * M_sym) * Dinv[None, :]


def project_window_set_to_z2_rescaled(M_mats: List[np.ndarray], d: int) -> Tuple[List[np.ndarray], List[int]]:
    """Convenience: project + rescale so caller can feed the LP directly."""
    sym_mats, counts = project_window_set_to_z2(M_mats)
    rescaled = [rescale_for_standard_simplex(M, d) for M in sym_mats]
    return rescaled, counts
