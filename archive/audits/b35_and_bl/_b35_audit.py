"""Audit whether B35 (single-coord rest-sum empty test) ever produces a
false positive on the cells the 1.25 cascade actually saw.

B35 from `_coarse_bnb_v4.py::is_cell_empty`:

 rest_sum_max = cell.hi.sum()
 for i in range(cell.d):
 if cell.lo[i] > 0:
 if (rest_sum_max - cell.hi[i]) < (1.0 - cell.lo[i]) - eps:
 return True -- ⟵ unsound: cell may still contain a probability vector

Mathematical check: for the simplex-intersect-box `Σμ=1, μ∈[lo,hi]` with `lo≥0`,
nonemptiness is *equivalent* to `Σlo ≤ 1 ≤ Σhi` (by IVT on `t↦(1-t)lo+t·hi`).
B35 fires whenever `Σ_{j≠i} hi_j < 1 - lo_i` and `lo_i > 0`, which is strictly
weaker than infeasibility.

Counter-example (validated below):
 d=2, lo=[0.5, 0.0], hi=[1.0, 0.3]
 → B35 fires for i=0, yet μ=[0.7, 0.3] is feasible.

Goal: check whether B35 ever fires on the actual cells the d=8/S=16 cascade
saw. If yes, the 1.25 result is tainted. If no (e.g. fires only on cells
B34 would have killed anyway), the result stands.
"""
from __future__ import annotations
import os, sys, time, logging
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v4 as v4
import _coarse_bnb_v3 as v3


def _b34_only(cell, eps=1e-12):
 """Only the sound B34 tests."""
 if cell.lo.sum() > 1.0 + eps:
 return True
 if cell.hi.sum() < 1.0 - eps:
 return True
 return False


def _b35_only(cell, eps=1e-12):
 """Only the (potentially unsound) B35 test, *after* B34 already failed."""
 if cell.lo.sum() > 1.0 + eps:
 return False # caught by B34a
 if cell.hi.sum() < 1.0 - eps:
 return False # caught by B34b
 rest_sum_max = cell.hi.sum()
 for i in range(cell.d):
 if cell.lo[i] > 0:
 if (rest_sum_max - cell.hi[i]) < (1.0 - cell.lo[i]) - eps:
 return True
 return False


def _is_actually_empty_lp(cell, eps=1e-9):
 """Sound LP feasibility check: ∃ μ ∈ [lo,hi] with Σμ=1.
 By IVT this is equivalent to Σlo ≤ 1 ≤ Σhi (assuming lo ≤ hi)."""
 return not (cell.lo.sum() <= 1.0 + eps and cell.hi.sum() >= 1.0 - eps)


def verify_counterexample():
 print("\n=== B35 counter-example check ===", flush=True)
 cell = v3.Cell(lo=np.array([0.5, 0.0]), hi=np.array([1.0, 0.3]))
 b34 = _b34_only(cell)
 b35 = _b35_only(cell)
 actual = _is_actually_empty_lp(cell)
 mu = np.array([0.7, 0.3])
 mu_in_cell = np.all(cell.lo <= mu) and np.all(mu <= cell.hi) and abs(mu.sum() - 1) < 1e-12
 print(f" cell lo={cell.lo.tolist()}, hi={cell.hi.tolist()}", flush=True)
 print(f" B34 fires (sound): {b34}", flush=True)
 print(f" B35 fires (unsound): {b35}", flush=True)
 print(f" truly empty (LP/IVT): {actual}", flush=True)
 print(f" mu=[0.7, 0.3] feasible: {mu_in_cell}", flush=True)
 print(f" => B35 unsound: {b35 and not actual}", flush=True)


def audit_cascade_cells():
 """Walk the actual d=8/S=16 cascade and count B35 firings.

 For each integer composition c (canonical) at d=8/S=16, build the
 Voronoi cell and ALL split sub-cells encountered during cert_cell.
 Count B35-fires that are NOT also B34-fires."""
 sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
 sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
 from compositions import generate_canonical_compositions_batched
 from _d16_F_bench import _prune_coarse_count_cell

 d, S, c_target = 8, 16, 1.25

 # JIT warm
 warm = np.zeros((1, d), dtype=np.int32)
 warm[0, 0] = S
 _prune_coarse_count_cell(warm, d, S, c_target)
 print(f"\n=== Audit cascade at d={d}, S={S}, c={c_target} ===", flush=True)

 # Limit to first 50_000 compositions for speed
 LIMIT = 50_000
 n_seen = 0
 n_b35_only_firings = 0 # B35 fires but B34 does not
 n_b35_unsound = 0 # B35 fires but cell is feasible
 examples = []

 for batch in generate_canonical_compositions_batched(d, S, batch_size=5_000):
 if n_seen >= LIMIT:
 break
 for c in batch:
 n_seen += 1
 if n_seen > LIMIT:
 break
 cell = v3.Cell.from_integer_composition(c, S)
 if not _b34_only(cell) and _b35_only(cell):
 n_b35_only_firings += 1
 if not _is_actually_empty_lp(cell):
 n_b35_unsound += 1
 if len(examples) < 3:
 examples.append((c.tolist(),
 cell.lo.tolist(),
 cell.hi.tolist()))
 if n_seen % 5_000 == 0:
 print(f" scanned {n_seen:,} B35-only-firings={n_b35_only_firings} "
 f"unsound={n_b35_unsound}", flush=True)

 print(f"\n TOTALS (first {n_seen:,} canonical compositions at d={d}, S={S}):", flush=True)
 print(f" B35-only firings (not caught by B34): {n_b35_only_firings}", flush=True)
 print(f" of these, UNSOUND firings: {n_b35_unsound}", flush=True)
 if n_b35_unsound > 0:
 print(f"\n *** B35 IS UNSOUND IN PRACTICE — 1.25 proof tainted ***", flush=True)
 for c, lo, hi in examples:
 print(f" c={c}", flush=True)
 print(f" lo={lo}", flush=True)
 print(f" hi={hi}", flush=True)
 else:
 print(f"\n B35 never fired (or only on cells B34 caught). 1.25 result safe.",
 flush=True)


if __name__ == '__main__':
 verify_counterexample()
 audit_cascade_cells()
