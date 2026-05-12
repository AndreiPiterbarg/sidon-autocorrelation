"""Saddle-KKT system for min_{mu in B} max_W TV_W(mu).

Given:
  * dimension d
  * windows: list of (ell, s_lo, scale_q=Fraction(2d, ell), pairs=list[(i,j)])
    where pairs are the (ordered) (i, j) with M_W[i,j] = scale_q.
    TV_W(mu) = scale_q * sum_{(i,j) in pairs} mu_i mu_j
  * box B = ∏_i [lo_i, hi_i] cap Delta_d (lo_q, hi_q as Fraction arrays)
  * active set (A_W: list[int] of window indices, A_plus, A_minus: list[int]
    of axis indices)

The reduced KKT system is square in unknowns

    x = [mu_F (n_F),  lambda (n_AW),  nu (1),  t (1)]

with n_F = d - |A_plus| - |A_minus|, n_AW = |A_W|, total dim
n = n_F + n_AW + 2.

Equations (n total):
  (E1) TV_W(mu) - t = 0                              for W in A_W   [n_AW eqs]
  (E2) sum_F mu_i  - (1 - sum_{A+} hi - sum_{A-} lo) = 0             [1 eq]
  (E3) sum_{W in A_W} lambda_W                       - 1 = 0         [1 eq]
  (E4) sum_{W in A_W} 2 * lambda_W * scale_W * (A_W mu)_i - nu = 0   for i in F
                                                                     [n_F eqs]

Derived (post-hoc check inequalities >= 0):
  beta_plus_i  = sum_W 2 lambda_W scale_W (A_W mu)_i - nu  for i in A_plus
  beta_minus_i = nu - sum_W 2 lambda_W scale_W (A_W mu)_i  for i in A_minus

Inequality conditions for a valid KKT point (post-hoc):
  lambda_W >= 0  (W in A_W)
  beta_plus_i  >= 0  (i in A_plus)
  beta_minus_i >= 0  (i in A_minus)
  TV_W(mu) <= t  (W not in A_W) -- "primal" feasibility
  mu_i in (lo_i, hi_i) (i in F)
  t < T  (the bad-branch hypothesis)

PROOF OBLIGATION FOR STEP 2:
  For every active set (A_W, A_plus, A_minus), if Krawczyk verifies that
  the equation system F(x) = 0 has *no* solution x in the prescribed box X
  (with t in [t_min, T - eps]), then there is no KKT critical point for
  this active set in B with value < T.  Combined with the requirement that
  the global minimum of max_W TV_W on B is *some* KKT point (Step 1
  cleared the vertex case), exhaustive enumeration over (A_W, A_plus,
  A_minus) gives a complete certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .iv_core import (IVMat, IVVec, iv_zero, rat_to_iv)


@dataclass
class WindowSpec:
    """Quadratic form TV_W(mu) = scale * mu^T A_W mu, encoded by support."""
    ell: int
    s_lo: int
    scale_q: Fraction               # = 2d/ell as exact rational
    pairs_all: Tuple[Tuple[int, int], ...]   # ordered (i, j) with A_W[i,j] = 1


@dataclass
class BoxSpec:
    d: int
    lo_q: Tuple[Fraction, ...]      # length d
    hi_q: Tuple[Fraction, ...]      # length d


@dataclass
class ActiveSet:
    A_W: Tuple[int, ...]            # indices into windows list
    A_plus: Tuple[int, ...]         # axis indices
    A_minus: Tuple[int, ...]        # axis indices
    target_T: Fraction              # threshold for bad branch


class KKTSystem:
    """Reduced KKT system for fixed active set on a box.

    Variables ordered as [mu_F, lambda_AW, nu, t].
    """

    def __init__(self, box: BoxSpec, windows: Sequence[WindowSpec],
                 active: ActiveSet):
        self.box = box
        self.windows = windows
        self.active = active

        d = box.d
        AP = set(active.A_plus)
        AM = set(active.A_minus)
        if AP & AM:
            raise ValueError("A_plus and A_minus must be disjoint")
        self.F: Tuple[int, ...] = tuple(i for i in range(d)
                                        if i not in AP and i not in AM)
        self.n_F = len(self.F)
        self.n_AW = len(active.A_W)
        self.n_AP = len(active.A_plus)
        self.n_AM = len(active.A_minus)
        # Variable indices
        self.idx_mu_F = list(range(self.n_F))
        off = self.n_F
        self.idx_lambda = list(range(off, off + self.n_AW))
        off += self.n_AW
        self.idx_nu = off
        off += 1
        self.idx_t = off
        off += 1
        self.n_vars = off

        # Equation indices
        # [E1: |A_W| TV eqs, E2: sum_mu_F=const, E3: sum_lambda=1, E4: |F| stationarity]
        self.n_eqs_E1 = self.n_AW
        self.n_eqs_E2 = 1
        self.n_eqs_E3 = 1
        self.n_eqs_E4 = self.n_F
        self.n_eqs = self.n_eqs_E1 + self.n_eqs_E2 + self.n_eqs_E3 + self.n_eqs_E4
        assert self.n_eqs == self.n_vars, \
            f"KKT system is non-square: {self.n_eqs} eqs vs {self.n_vars} vars"

        # Lookup: full-mu index -> (kind, slot)
        # kind: 'F' (slot is index into mu_F), 'P' (hi[slot]), 'M' (lo[slot])
        self.mu_lookup: List[Tuple[str, int]] = []
        F_pos = {i: k for k, i in enumerate(self.F)}
        for i in range(d):
            if i in AP:
                self.mu_lookup.append(('P', i))
            elif i in AM:
                self.mu_lookup.append(('M', i))
            else:
                self.mu_lookup.append(('F', F_pos[i]))

        # For each W in A_W, build a list of (i, j) pairs and a per-axis
        # adjacency (axis -> list of j with (axis, j) in pairs_all). Used
        # both for residual and Jacobian.
        self.A_W_pairs: List[Tuple[Tuple[int, int], ...]] = []
        self.A_W_adj: List[List[List[int]]] = []   # [w_local][i] = list of j
        for w_idx in active.A_W:
            w = windows[w_idx]
            self.A_W_pairs.append(w.pairs_all)
            adj = [[] for _ in range(d)]
            for (i, j) in w.pairs_all:
                adj[i].append(j)
            self.A_W_adj.append(adj)

        # Constants
        self.const_sum_F = (Fraction(1)
                            - sum((box.hi_q[i] for i in active.A_plus),
                                  Fraction(0))
                            - sum((box.lo_q[i] for i in active.A_minus),
                                  Fraction(0)))
        # The scale_q for each W in A_W
        self.A_W_scales_q: List[Fraction] = [windows[w].scale_q
                                             for w in active.A_W]

    # -----------------------------------------------------------------
    # Helpers: extract mu from x
    # -----------------------------------------------------------------

    def _full_mu(self, x_F, hi_q, lo_q):
        """Return list of length d with mu_i in {Fraction or interval}.
        x_F: list/array of size n_F (mu_F values).
        hi_q, lo_q: lists of d values for box endpoints."""
        d = self.box.d
        mu = [None] * d
        for i in range(d):
            kind, slot = self.mu_lookup[i]
            if kind == 'F':
                mu[i] = x_F[slot]
            elif kind == 'P':
                mu[i] = hi_q[slot]
            else:  # 'M'
                mu[i] = lo_q[slot]
        return mu

    # -----------------------------------------------------------------
    # Residual (works for Fraction, float, or iv.mpf inputs)
    # -----------------------------------------------------------------

    def residual(self, x_vec, hi_q=None, lo_q=None, scale_q_override=None):
        """Compute the residual vector F(x_vec) of length n_eqs.

        Inputs may be Fraction, float, or mpmath.iv intervals (or any
        consistent ring with +, -, *).  Returns a Python list of the same
        scalar type.

        hi_q/lo_q default to box.hi_q/lo_q (Fraction). Override to an
        interval/float vector if needed.
        scale_q_override: if provided, list of W-scales for A_W (overrides
        Fraction; useful when scales are converted to intervals).
        """
        if hi_q is None:
            hi_q = self.box.hi_q
        if lo_q is None:
            lo_q = self.box.lo_q
        scales = scale_q_override if scale_q_override is not None \
            else self.A_W_scales_q

        n_F = self.n_F
        n_AW = self.n_AW
        # Slice variables
        x_F = [x_vec[k] for k in self.idx_mu_F]
        lam = [x_vec[k] for k in self.idx_lambda]
        nu = x_vec[self.idx_nu]
        t = x_vec[self.idx_t]

        mu = self._full_mu(x_F, hi_q, lo_q)
        d = self.box.d

        eqs = []

        # (E1) TV_W(mu) - t = 0 for W in A_W
        for w_local in range(n_AW):
            pairs = self.A_W_pairs[w_local]
            scale = scales[w_local]
            tv = 0  # type-flexible zero (works with Fraction, float, iv)
            for (i, j) in pairs:
                tv = tv + mu[i] * mu[j]
            tv = scale * tv
            eqs.append(tv - t)

        # (E2) sum_F mu_F  - const = 0
        # Compute const dynamically from hi_q / lo_q so that if these are
        # iv.mpf intervals the result is also an interval (no mixing).
        const = 1
        for i in self.active.A_plus:
            const = const - hi_q[i]
        for i in self.active.A_minus:
            const = const - lo_q[i]
        sum_F = 0
        for k in range(n_F):
            sum_F = sum_F + x_F[k]
        eqs.append(sum_F - const)

        # (E3) sum lambda - 1 = 0
        s = 0
        for k in range(n_AW):
            s = s + lam[k]
        eqs.append(s - 1)

        # (E4) For i in F: sum_W 2 lambda_W scale_W (A_W mu)_i - nu = 0
        # (A_W mu)_i = sum_{j : (i,j) in pairs_W} mu_j
        for fi in range(n_F):
            i = self.F[fi]
            grad_i = 0
            for w_local in range(n_AW):
                adj = self.A_W_adj[w_local][i]
                if not adj:
                    continue
                ip_sum = 0
                for j in adj:
                    ip_sum = ip_sum + mu[j]
                grad_i = grad_i + 2 * lam[w_local] * scales[w_local] * ip_sum
            eqs.append(grad_i - nu)

        return eqs

    # -----------------------------------------------------------------
    # Jacobian (analytic, returns matrix of n_eqs x n_vars entries)
    # -----------------------------------------------------------------

    def jacobian(self, x_vec, hi_q=None, lo_q=None, scale_q_override=None):
        """Compute the Jacobian matrix dF/dx of shape (n_eqs, n_vars).

        Returns a list of lists. Entries follow the type of inputs.
        """
        if hi_q is None:
            hi_q = self.box.hi_q
        if lo_q is None:
            lo_q = self.box.lo_q
        scales = scale_q_override if scale_q_override is not None \
            else self.A_W_scales_q

        n_F = self.n_F
        n_AW = self.n_AW
        n = self.n_vars
        d = self.box.d

        x_F = [x_vec[k] for k in self.idx_mu_F]
        lam = [x_vec[k] for k in self.idx_lambda]
        # nu, t are also vars but t only appears linearly in E1 and nu in E4

        mu = self._full_mu(x_F, hi_q, lo_q)

        # Initialise zero matrix
        # Determine zero element matching input type
        z = 0  # works for Fraction, float; for iv.mpf will be coerced as needed
        # Build as list-of-lists
        J = [[z for _ in range(n)] for _ in range(self.n_eqs)]

        zero_const = 0
        F_pos = {i: k for k, i in enumerate(self.F)}

        # Row indices
        E1_off = 0
        E2_off = self.n_eqs_E1
        E3_off = E2_off + 1
        E4_off = E3_off + 1

        # ---- E1 rows: TV_W(mu) - t = 0
        # d/dmu_i of TV_W(mu) = scale_W * 2 * (A_W mu)_i  (since A_W is symmetric
        # via pairs_all duplicating off-diag entries)
        # d/dlambda_W = 0 (lambda doesn't appear in E1)
        # d/dnu = 0
        # d/dt = -1
        for w_local in range(n_AW):
            row = E1_off + w_local
            scale = scales[w_local]
            # d/dmu_F[fi]: scale * 2 * sum_j (in adj of axis F[fi]) mu_j
            adj = self.A_W_adj[w_local]
            for fi in range(n_F):
                i = self.F[fi]
                ip_sum = 0
                for j in adj[i]:
                    ip_sum = ip_sum + mu[j]
                J[row][self.idx_mu_F[fi]] = scale * 2 * ip_sum
            # d/dlambda: 0
            # d/dnu: 0
            # d/dt: -1
            J[row][self.idx_t] = -1

        # ---- E2 row: sum_F mu_F - const = 0
        # d/dmu_F[fi] = 1, others 0
        row = E2_off
        for fi in range(n_F):
            J[row][self.idx_mu_F[fi]] = 1
        # all others 0

        # ---- E3 row: sum lambda - 1 = 0
        # d/dlambda_W = 1
        row = E3_off
        for w_local in range(n_AW):
            J[row][self.idx_lambda[w_local]] = 1

        # ---- E4 rows: sum_W 2 lambda_W scale_W (A_W mu)_i - nu = 0  (i in F)
        # d/dmu_F[fj]:
        #    sum_{W in A_W} 2 lambda_W scale_W * d(A_W mu)_i / d mu_F[fj]
        #    (A_W mu)_i = sum_{j' in adj_W[i]} mu_{j'}
        #    d / d mu_F[fj] = #{occurrences of axis F[fj] in adj_W[i]}
        #                  = 1 if F[fj] in adj_W[i] else 0
        # d/dlambda_W = 2 scale_W (A_W mu)_i
        # d/dnu = -1
        # d/dt = 0
        for fi in range(n_F):
            i = self.F[fi]
            row = E4_off + fi
            # d/dmu_F[fj]
            for fj in range(n_F):
                j = self.F[fj]
                # sum over W of 2 lambda_W scale_W * 1{j in adj_W[i]}
                cell = 0
                for w_local in range(n_AW):
                    if j in self.A_W_adj[w_local][i]:
                        cell = cell + 2 * lam[w_local] * scales[w_local]
                J[row][self.idx_mu_F[fj]] = cell
            # d/dlambda_W: 2 scale_W * (A_W mu)_i
            for w_local in range(n_AW):
                ip_sum = 0
                for jj in self.A_W_adj[w_local][i]:
                    ip_sum = ip_sum + mu[jj]
                J[row][self.idx_lambda[w_local]] = 2 * scales[w_local] * ip_sum
            # d/dnu = -1
            J[row][self.idx_nu] = -1

        return J


# ---------------------------------------------------------------------
# Convenience: check derived inequalities
# ---------------------------------------------------------------------

def derived_quantities(system: KKTSystem, x_vec, hi_q=None, lo_q=None,
                       scale_q_override=None):
    """Compute beta_plus, beta_minus, and the "off-active" TV_W values.

    Returns dict with:
      beta_plus[i]   for i in A_plus    (must be >= 0)
      beta_minus[i]  for i in A_minus   (must be >= 0)
      tv_off[w_idx]  for w_idx not in A_W (must be <= t)
      tv_active[w_local]  for w_local in A_W (should equal t up to residual)
    """
    box = system.box
    if hi_q is None:
        hi_q = box.hi_q
    if lo_q is None:
        lo_q = box.lo_q
    scales = scale_q_override if scale_q_override is not None \
        else system.A_W_scales_q

    x_F = [x_vec[k] for k in system.idx_mu_F]
    lam = [x_vec[k] for k in system.idx_lambda]
    nu = x_vec[system.idx_nu]
    t = x_vec[system.idx_t]

    mu = system._full_mu(x_F, hi_q, lo_q)

    out = {"beta_plus": {}, "beta_minus": {}, "tv_off": {}, "tv_active": []}

    # tv_active
    for w_local in range(system.n_AW):
        pairs = system.A_W_pairs[w_local]
        scale = scales[w_local]
        tv = 0
        for (i, j) in pairs:
            tv = tv + mu[i] * mu[j]
        out["tv_active"].append(scale * tv)

    # beta_plus_i: sum_W 2 lambda_W scale_W (A_W mu)_i - nu
    for ap_local, i in enumerate(system.active.A_plus):
        grad_i = 0
        for w_local in range(system.n_AW):
            ip_sum = 0
            for jj in system.A_W_adj[w_local][i]:
                ip_sum = ip_sum + mu[jj]
            grad_i = grad_i + 2 * lam[w_local] * scales[w_local] * ip_sum
        out["beta_plus"][i] = grad_i - nu

    # beta_minus_i: nu - sum_W 2 lambda_W scale_W (A_W mu)_i
    for am_local, i in enumerate(system.active.A_minus):
        grad_i = 0
        for w_local in range(system.n_AW):
            ip_sum = 0
            for jj in system.A_W_adj[w_local][i]:
                ip_sum = ip_sum + mu[jj]
            grad_i = grad_i + 2 * lam[w_local] * scales[w_local] * ip_sum
        out["beta_minus"][i] = nu - grad_i

    # tv_off: for windows not in A_W
    A_W_set = set(system.active.A_W)
    for w_idx, w in enumerate(system.windows):
        if w_idx in A_W_set:
            continue
        scale = w.scale_q  # Fraction; for interval mode caller can pass own
        tv = 0
        for (i, j) in w.pairs_all:
            tv = tv + mu[i] * mu[j]
        out["tv_off"][w_idx] = scale * tv

    return out


# ---------------------------------------------------------------------
# Self-test on a tiny example
# ---------------------------------------------------------------------

if __name__ == "__main__":
    """Test residual and Jacobian on a hand-computable d=3 case.

    Take A_W = {W0} where W0 = (ell=2, s_lo=1) with
        TV_{W0}(mu) = scale * (mu_0 mu_1 + mu_1 mu_0) = 2 scale mu_0 mu_1
    where scale = 2*3/2 = 3.  So TV = 6 mu_0 mu_1.

    A_plus = empty, A_minus = empty (full simplex Delta_3).
    Box: lo=0, hi=1 trivially.
    KKT: minimise t s.t. 6 mu_0 mu_1 = t, sum mu = 1, sum lambda = 1.
         lambda_0 (the only lambda) = 1.
         Stationarity: for each i in F = {0,1,2}:
             sum_W 2 lam_W scale (A_W mu)_i - nu = 0
           For W0, (A_W0 mu)_0 = mu_1, (A_W0 mu)_1 = mu_0, (A_W0 mu)_2 = 0.
             i=0: 2 * 1 * 3 * mu_1 - nu = 6 mu_1 - nu = 0
             i=1: 6 mu_0 - nu = 0
             i=2: 0 - nu = 0  =>  nu = 0.
           From nu=0: mu_0 = 0 = mu_1.  But sum mu = 1 => mu_2 = 1.
           Then TV = 6*0*0 = 0 = t.
    So unique solution: mu = (0, 0, 1), nu = 0, t = 0, lambda = 1.
    Note this is a *boundary* point (mu_0=mu_1=0), so really i=0,1 should
    be in A_minus. This active set is *non-vertex* but degenerate; it
    illustrates the algebra. We test residual at the candidate solution.
    """
    from fractions import Fraction
    d = 3
    pairs01 = ((0, 1), (1, 0))
    W0 = WindowSpec(ell=2, s_lo=1, scale_q=Fraction(3), pairs_all=pairs01)
    box = BoxSpec(d=d, lo_q=(Fraction(0),) * d, hi_q=(Fraction(1),) * d)
    active = ActiveSet(A_W=(0,), A_plus=(), A_minus=(),
                       target_T=Fraction(1, 2))
    sys_kkt = KKTSystem(box, [W0], active)
    print(f"n_F={sys_kkt.n_F}, n_AW={sys_kkt.n_AW}, n_vars={sys_kkt.n_vars}, n_eqs={sys_kkt.n_eqs}")

    # Variables: [mu_F[0..2], lambda[0], nu, t]; n_vars = 6
    # Candidate solution: mu = (0, 0, 1), lambda=1, nu=0, t=0
    x = [Fraction(0), Fraction(0), Fraction(1),  # mu_F
         Fraction(1),                            # lambda
         Fraction(0),                            # nu
         Fraction(0)]                            # t
    res = sys_kkt.residual(x)
    print("residual at (0,0,1, 1, 0, 0) =", [str(r) for r in res])
    assert all(r == 0 for r in res), "Residual not zero!"

    # Jacobian shape
    J = sys_kkt.jacobian(x)
    print(f"Jacobian shape = ({len(J)} x {len(J[0])})")
    for row in J:
        print(" ", [str(c) for c in row])

    # Derived quantities (no A_plus or A_minus, so empty)
    der = derived_quantities(sys_kkt, x)
    print("derived: tv_active =", [str(v) for v in der["tv_active"]])

    print("\nself-test OK")
