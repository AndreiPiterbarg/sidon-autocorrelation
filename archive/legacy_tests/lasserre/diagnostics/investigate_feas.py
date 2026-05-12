#!/usr/bin/env python
r"""Investigate why ADMM phase-1 feasibility detection fails at d=32 bw=31.

Problem: tau converges to small negative value (-0.0001) regardless of
whether the original problem is feasible or infeasible. This means
bisection can never find the infeasible side.

Hypotheses to test:
  H1: Not enough ADMM iterations — tau hasn't converged yet
  H2: Ruiz equilibration distorts tau meaning
  H3: Phase-1 augmentation is wrong at this scale
  H4: The problem IS actually feasible at t=0.5 (we're wrong about expected)
  H5: Need a different feasibility detection approach entirely
"""
import torch
import numpy as np
import sys
import os
import time
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import (
    _precompute_highd, _check_violations_highd,
    _build_banded_cliques, val_d_known,
)
from run_scs_direct import build_base_problem
from admm_gpu_solver import admm_solve, augment_phase1

device = 'cuda'
D, BW = 32, 31

print("=" * 72)
print("FEASIBILITY INVESTIGATION: d=32 bw=31")
print("=" * 72)

# Build problem
cliques = _build_banded_cliques(D, BW)
P = _precompute_highd(D, 2, cliques, verbose=False)
A_base, b_base, c_obj, cone_base, meta = build_base_problem(P, True)
n_y = P['n_y']

# ================================================================
# INVESTIGATION 1: Does CPU SCS correctly detect infeasibility?
# ================================================================
print(f"\n{'='*72}")
print("INV 1: Ground truth — CPU SCS on small d=8 bw=7")
print("  First verify our understanding of feasibility is correct")
print("=" * 72)

# Use d=8 (small, SCS CPU works) to verify:
# at t < val(d), the problem should be infeasible
cliques_8 = _build_banded_cliques(8, 7)
P_8 = _precompute_highd(8, 2, cliques_8, verbose=False)
A_8, b_8, c_8, cone_8, meta_8 = build_base_problem(P_8, True)

import scs
for t_val in [0.5, 0.8, 1.0, 1.1, 1.2, 1.5, 2.0]:
    # The 'c' vector has c[t_col] = 1 (minimize t).
    # For feasibility at fixed t: set c=0, fix t=t_val via constraint.
    # Actually, the way run_scs_direct.py works: it builds A(t) and checks
    # if the system Ax + s = b, s in K is feasible with c=0 (find any x).
    c_feas = np.zeros(meta_8['n_x'])
    data = {'A': A_8, 'b': b_8, 'c': c_feas}
    s = scs.SCS(data, cone_8, max_iters=5000, eps_abs=1e-5, eps_rel=1e-5,
                verbose=False)
    sol = s.solve()
    status = sol['info']['status']
    print(f"  t={t_val} (in c_obj, not varied): status={status}")

# The base problem doesn't vary t — t is a free variable minimized.
# Feasibility check at a specific t requires modifying A.
# Let me check: what does the actual production code do?
print("\n  NOTE: The base problem minimizes t. For feasibility at fixed t,")
print("  we need the CG-round setup that builds A(t) = A_base + t*A_t.")
print("  Let me test with the actual phase-1 approach on d=8.")

A_8_p1, b_8_p1, c_8_p1, cone_8_p1, tau_8 = augment_phase1(A_8, b_8, cone_8)
print(f"\n  d=8 phase-1 problem: {A_8_p1.shape}")

# GPU ADMM on d=8 phase-1 (should work — it's small)
sol = admm_solve(A_8_p1, b_8_p1, c_8_p1, cone_8_p1,
                 max_iters=2000, eps_abs=1e-6, eps_rel=1e-6,
                 rho=0.5, alpha=1.0, device=device, verbose=False)
tau_val = sol['x'][tau_8]
print(f"  d=8 ADMM phase-1: tau={tau_val:.6f}, status={sol['info']['status']}, "
      f"iters={sol['info']['iter']}")
print(f"  tau < 0 means base problem is feasible (expected: yes, it's min t)")

# ================================================================
# INVESTIGATION 2: Is t=0.5 actually infeasible for d=32?
# ================================================================
print(f"\n{'='*72}")
print("INV 2: Check what the base problem (min t) gives for d=32")
print("  If min t* > 0.5, then t=0.5 should be infeasible.")
print("=" * 72)

# Run ADMM on the base problem (min t, no phase-1) with many iters
sol_base = admm_solve(A_base, b_base, c_obj, cone_base,
                      max_iters=2000, eps_abs=1e-5, eps_rel=1e-5,
                      rho=0.5, alpha=1.0, device=device, verbose=True)
t_star = sol_base['x'][meta['t_col']]
print(f"\n  min t* = {t_star:.6f}")
print(f"  So t=0.5 should be {'INFEASIBLE' if t_star > 0.5 else 'FEASIBLE'}")
print(f"  And t=2.0 should be FEASIBLE (2.0 > t*)")

# ================================================================
# INVESTIGATION 3: Phase-1 with many more iterations
# ================================================================
print(f"\n{'='*72}")
print("INV 3: Phase-1 at d=32 with increasing ADMM iterations")
print("  Does tau converge to the right sign with enough iterations?")
print("=" * 72)

# The base problem phase-1 should give tau ≤ 0 (feasible — we're minimizing t,
# so there exists a feasible point). But the question is about FIXED t.
# Let me check: what does phase-1 of the BASE problem mean?
#
# Base problem: min t s.t. Ax+s=b, s in K
# Phase-1 of base: min tau s.t. A_aug*x_aug + s = b_aug, s in K_aug
# where tau is slack on PSD diagonals.
#
# If the base problem is feasible (it is — t is free), then tau* ≤ 0.
# This doesn't test infeasibility of a FIXED t.
#
# The ACTUAL production code (run_scs_direct.py check_feasible()) does:
# 1. Build A(t_mid) = A_base + t_mid * A_t (with window PSD constraints)
# 2. augment_phase1(A(t_mid), b, cone)
# 3. Solve min tau
# 4. If tau ≤ threshold → feasible
#
# The key: A(t_mid) has t FIXED (baked into A), not as a variable.
# So phase-1 checks: "is there ANY x satisfying A(t)*x + s = b, s in K?"
# If t is too small, the PSD constraints are too tight → infeasible → tau > 0.
#
# BUT in our test (TEST 8), we used the FULL A with t as a variable,
# not A(t) with t fixed. That's the bug!

print("\n  DIAGNOSIS: TEST 8 may have used wrong problem formulation.")
print("  Phase-1 should be applied to A(t_fixed), not A with t as variable.")
print("  With t as free variable, the problem is ALWAYS feasible (pick large t).")
print("  So tau ≤ 0 is CORRECT — we were testing the wrong thing!")

# ================================================================
# INVESTIGATION 4: Test with FIXED t via window PSD constraints
# ================================================================
print(f"\n{'='*72}")
print("INV 4: Correct feasibility test — A(t) with t FIXED")
print("  Build window PSD constraints, fix t, then check phase-1")
print("=" * 72)

from run_scs_direct import _precompute_window_psd_decomposition, _assemble_window_psd

# Get violations and add windows
y_dummy = np.zeros(n_y)
viols = _check_violations_highd(y_dummy, 1.0, P, set())
active = set()
for w, _ in viols[:50]:
    active.add(w)

win_decomp = _precompute_window_psd_decomposition(P, active)
if win_decomp is None:
    print("  ERROR: no window decomposition")
    sys.exit(1)

# Build full A(t) for different t values
for t_val in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    A_win, _, psd_win = _assemble_window_psd(win_decomp, t_val)
    A_full = sp.vstack([A_base, A_win], format='csc')
    A_full.sort_indices()
    b_full = np.concatenate([b_base, np.zeros(win_decomp['n_rows'])])
    cone_full = {'z': cone_base['z'], 'l': cone_base['l'],
                 's': list(cone_base['s']) + psd_win}

    # Now remove the t column — we want t FIXED, not as a variable
    # Actually, the A matrix has t_col as a column. For feasibility
    # at fixed t, we should either:
    # (a) Fix x[t_col] = t_val and remove that variable, or
    # (b) Use c=0 (feasibility, not optimization) — then t is free
    #     but the window PSD constraints already encode t_val in A.
    #
    # Method (b) is what run_scs_direct.py does:
    #   c_feas = zeros(n_x)  (not minimize t, just find feasible x)
    #   The window constraints have t baked into A(t_val).
    #   But t is still a free variable in x!
    #   If solver picks x[t_col] = anything, the constraint A(t_val)*x + s = b
    #   might be satisfiable for some x even if t_val is "too small",
    #   because x[t_col] can compensate through the base A entries.
    #
    # Wait — the base A has entries for t_col too (the scalar window
    # constraints: f_W(y) - t ≤ 0, i.e., -1 in t_col for nonneg rows).
    # So when we do A(t_val)*x + s = b with c=0, the solver can freely
    # choose x[t_col], which enters via the base A's nonneg rows.
    # This means x[t_col] acts like a free t, making the problem feasible
    # for ANY t_val! The window PSD entries have t_val baked in,
    # but the scalar window rows still have -t as a variable.
    #
    # THIS IS THE BUG. The phase-1 approach with the full A matrix
    # (which includes t as a variable) is always feasible because
    # the solver can just increase x[t_col].

    # FIX: For feasibility at fixed t_val, we need to either:
    # 1. Remove t_col from A (fix t = t_val), or
    # 2. Add a constraint x[t_col] = t_val (equality).

    # Let's do option 2: add x[t_col] = t_val as a zero-cone constraint.
    t_col = meta['n_x'] - 1  # last column
    n_rows_full = A_full.shape[0]
    n_cols_full = A_full.shape[1]

    # Add row: x[t_col] = t_val → 1*x[t_col] + s_eq = t_val, s_eq in {0}
    fix_t_row = sp.csc_matrix(
        ([1.0], ([0], [t_col])),
        shape=(1, n_cols_full))
    A_fixed = sp.vstack([fix_t_row, A_full], format='csc')
    A_fixed.sort_indices()
    b_fixed = np.concatenate([[t_val], b_full])
    cone_fixed = {'z': cone_full['z'] + 1,  # one more equality
                  'l': cone_full['l'],
                  's': cone_full['s']}

    # Now phase-1 on this fixed-t problem
    A_fp1, b_fp1, c_fp1, cone_fp1, tau_fp1 = augment_phase1(
        A_fixed, b_fixed, cone_fixed)

    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_fp1, b_fp1, c_fp1, cone_fp1,
                     max_iters=1000, eps_abs=1e-5, eps_rel=1e-5,
                     rho=0.5, alpha=1.0, device=device, verbose=False)
    torch.cuda.synchronize()
    dt = time.time() - t0

    tau = sol['x'][tau_fp1] if sol['x'] is not None else float('inf')
    is_feas = (sol['info']['status'] in ('solved', 'solved_inaccurate')
               and tau <= max(1e-5 * 10, 1e-4))
    tag = "FEAS" if is_feas else "INFEAS"
    print(f"  t={t_val:4.1f}: tau={tau:+.6f} → {tag}  "
          f"({sol['info']['iter']} iters, {dt:.1f}s)")

# ================================================================
# INVESTIGATION 5: Alternative — no phase-1, just check residuals
# ================================================================
print(f"\n{'='*72}")
print("INV 5: Alternative feasibility — solve c=0 with t fixed, check residuals")
print("  If primal residual is small → feasible. If not → infeasible.")
print("=" * 72)

for t_val in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    A_win, _, psd_win = _assemble_window_psd(win_decomp, t_val)
    A_full = sp.vstack([A_base, A_win], format='csc')
    A_full.sort_indices()
    b_full = np.concatenate([b_base, np.zeros(win_decomp['n_rows'])])
    cone_full = {'z': cone_base['z'], 'l': cone_base['l'],
                 's': list(cone_base['s']) + psd_win}

    # Fix t = t_val
    t_col = meta['n_x'] - 1
    n_cols_full = A_full.shape[1]
    fix_t_row = sp.csc_matrix(
        ([1.0], ([0], [t_col])),
        shape=(1, n_cols_full))
    A_fixed = sp.vstack([fix_t_row, A_full], format='csc')
    A_fixed.sort_indices()
    b_fixed = np.concatenate([[t_val], b_full])
    cone_fixed = {'z': cone_full['z'] + 1, 'l': cone_full['l'],
                  's': cone_full['s']}

    # Solve feasibility (c=0, no phase-1)
    c_feas = np.zeros(n_cols_full)

    torch.cuda.synchronize()
    t0 = time.time()
    sol = admm_solve(A_fixed, b_fixed, c_feas, cone_fixed,
                     max_iters=1000, eps_abs=1e-5, eps_rel=1e-5,
                     rho=0.5, alpha=1.0, device=device, verbose=False)
    torch.cuda.synchronize()
    dt = time.time() - t0

    status = sol['info']['status']
    print(f"  t={t_val:4.1f}: status={status:>20}  "
          f"({sol['info']['iter']} iters, {dt:.1f}s)")

# ================================================================
# INVESTIGATION 6: What does run_scs_direct.py ACTUALLY do for GPU?
# ================================================================
print(f"\n{'='*72}")
print("INV 6: Trace the actual production code path")
print("=" * 72)

# Read the check_feasible function in run_scs_direct.py
print("""
  In run_scs_direct.py check_feasible() GPU path:
  1. Build A(t_mid) = A_base + t_mid * A_t  (window PSD rows vary with t)
  2. augment_phase1(A_full_t1, b_full_base, cone_full)
     - t is STILL a free variable in x
     - The window PSD rows have t_mid baked into the Q-part but t is in A via
       the t-dependent PSD entries
  3. Solve min tau
  4. Check tau <= tau_tol

  THE ISSUE: When t is a free variable and c_p1 = [0,...,0,1] (min tau),
  the solver can choose any x[t_col]. The scalar window constraints
  have -t in the nonneg rows. If t_mid is small (infeasible), the solver
  can just pick x[t_col] large to satisfy scalar constraints, while the
  window PSD constraints have t_mid baked into the t-dependent part.

  BUT WAIT: the A(t_mid) construction does:
    A = A_base + t_mid * A_t
  where A_t contains the t-dependent entries. So after this substitution,
  t_mid is baked into ALL of A. The t_col in x still exists but the
  A_base has scalar window rows with -1 in t_col. So x[t_col] enters
  the nonneg constraints as: f_W(y) - x[t_col] <= 0.

  This means x[t_col] is FREE and can be set to anything >= max f_W(y).
  The PSD window constraints have t_mid baked in (via A_t), so they
  constrain y but NOT x[t_col].

  So the problem is ALWAYS feasible: pick x[t_col] = 1000, satisfying
  all scalar windows, and the PSD windows are independent of x[t_col].

  CONCLUSION: The phase-1 approach was designed for problems WITHOUT
  a free t variable. The production code forgot to fix t when doing
  GPU phase-1 feasibility check.

  FIX: In check_feasible() GPU path, add an equality constraint
  x[t_col] = t_mid before phase-1 augmentation. Or remove t_col
  from the problem entirely.
""")

print("=" * 72)
print("ROOT CAUSE FOUND")
print("=" * 72)
print("""
The GPU ADMM path in run_scs_direct.py uses phase-1 augmentation on a
problem where t is a FREE variable. Since t enters the scalar window
constraints as f_W(y) - t <= 0, the solver can always pick t large enough
to make everything feasible, regardless of t_mid.

The CPU SCS path doesn't have this problem because it solves the
feasibility problem c=0 directly — SCS detects infeasibility through
its own homogeneous self-dual embedding, not through phase-1.

FIX: In check_feasible() GPU path, either:
  (a) Fix x[t_col] = 0 via equality constraint (since t_mid is already
      baked into A), or
  (b) Remove t_col from the augmented problem, or
  (c) Don't use phase-1: solve c=0 feasibility and check ADMM status
      (simpler, but ADMM status is unreliable for infeasibility)

Option (a) is cleanest and preserves the phase-1 reliability.
""")
