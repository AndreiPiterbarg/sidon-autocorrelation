"""Smoke test: verify level-2 reproduces Shor=1.0; check level-3 for d=4.

Should give level-3 > 1.0 (i.e. Parrilo r=1 strictly improves r=0).
"""
import os, sys, time
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.core import build_window_matrices
from probe import lasserre_y_lb, enum_monos

d = 4
windows, M_mats = build_window_matrices(d)
print(f"d={d} n_win={len(windows)}")

for level in (2, 3):
    n_basis = len(enum_monos(d, level))
    n_pseudo = len(enum_monos(d, 2*level))
    print(f"  Level {level}: M_{level} size {n_basis}, pseudo-moments {n_pseudo}")
    t0 = time.time()
    lb, status = lasserre_y_lb(d, M_mats, windows, level=level, solver="MOSEK", verbose=False)
    print(f"  Level {level}: lb={lb:.6f}  status={status}  time={time.time()-t0:.2f}s")
print(f"Expected val(4) = 1.102 (cell-true bound)")
print(f"Expected r=0 (Shor): trivial 1.0")
print(f"Expected r=1: maybe in between -- this is the test")
