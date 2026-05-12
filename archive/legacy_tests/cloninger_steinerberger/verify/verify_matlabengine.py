#!/usr/bin/env python
"""Verify matlab.engine + SDPNAL+ + our driver work end-to-end.

Run AFTER:
  1. MATLAB installed
  2. SDPNAL+ installed, MEX compiled (Installmex(1)), addpath saved
  3. `pip install matlabengine` (or: install from MATLAB's own setup.py)
  4. tests/verify_sdpnal_install.m has already passed inside MATLAB

Usage:
  python tests/verify_matlabengine.py

Exit code 0 = everything green.  Nonzero = fix the first failing step and rerun.
"""
from __future__ import annotations

import os
import sys
import time


def _fail(msg, hint=None):
    print(f"[FAIL] {msg}")
    if hint:
        print(f"       {hint}")
    return 1


def main():
    print("=== matlab.engine + SDPNAL+ end-to-end verification ===\n")

    # ------------------------------------------------------------
    # [1/5] Python-side import
    # ------------------------------------------------------------
    t0 = time.time()
    try:
        import matlab.engine  # noqa: F401
        import matlab
    except ImportError as exc:
        return _fail(
            f"matlab.engine import failed: {exc}",
            "pip install matlabengine   (version must match your MATLAB release)"
        )
    print(f"[1/5] matlab.engine import: OK   ({time.time() - t0:.2f}s)")

    # ------------------------------------------------------------
    # [2/5] Start a MATLAB engine
    # ------------------------------------------------------------
    t0 = time.time()
    print("[2/5] Starting MATLAB engine (cold start takes 10-30s)...",
          flush=True)
    try:
        eng = matlab.engine.start_matlab()
    except Exception as exc:
        return _fail(
            f"could not start MATLAB: {exc}",
            "is MATLAB installed and licensed? Launch `matlab` in a terminal; "
            "if that works, retry here."
        )
    ver = eng.version(nargout=1)
    print(f"       MATLAB engine started: {ver}   ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------
    # [3/5] SDPNAL+ on path
    # ------------------------------------------------------------
    try:
        exists = float(eng.exist('sdpnalplus', 'file'))
    except Exception as exc:
        eng.quit()
        return _fail(f"error probing SDPNAL+ path: {exc}")
    if exists == 0:
        eng.quit()
        return _fail(
            "sdpnalplus not on MATLAB path",
            "in MATLAB: addpath(genpath('C:\\path\\to\\SDPNALplus')); savepath;"
        )
    sdpnalplus_loc = eng.which('sdpnalplus', nargout=1)
    print(f"[3/5] sdpnalplus on path: OK   ({sdpnalplus_loc})")

    # ------------------------------------------------------------
    # [4/5] Run verify_sdpnal_install.m (2x2 test SDP)
    # ------------------------------------------------------------
    wrapper_dir = os.path.dirname(os.path.abspath(__file__))
    eng.addpath(wrapper_dir, nargout=0)
    print("[4/5] Running verify_sdpnal_install.m ...")
    t0 = time.time()
    try:
        eng.verify_sdpnal_install(nargout=0)
    except Exception as exc:
        eng.quit()
        return _fail(
            f"verify_sdpnal_install.m failed: {exc}",
            "open MATLAB manually, run `verify_sdpnal_install`, fix the error "
            "it reports, then rerun this script."
        )
    print(f"       2x2 test SDP OK   ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------
    # [5/5] End-to-end smoke: d=4 L2 through our driver
    # ------------------------------------------------------------
    print("[5/5] Running the full driver on d=4 L2 (smoke test, ~10s)...",
          flush=True)
    sys.path.insert(0, wrapper_dir)
    try:
        from lasserre_sdpnalplus import (
            build_problem, MatlabRunner,
        )
    except Exception as exc:
        eng.quit()
        return _fail(f"could not import driver: {exc}")

    data_dir = os.path.join(wrapper_dir, '..', 'data', 'sdpnal_verify')
    os.makedirs(data_dir, exist_ok=True)

    t0 = time.time()
    try:
        prob = build_problem(d=4, order=2, bandwidth=3,
                              add_upper_loc=True, use_z2=True,
                              include_all_windows=False,  # round-0 only
                              verbose=False)
    except Exception as exc:
        eng.quit()
        return _fail(f"build_problem failed: {exc}")

    try:
        runner = MatlabRunner(use_engine=True, verbose=False)
        runner._engine = eng            # reuse the already-started engine
        runner._a_loaded = False
        res = runner.solve(
            prob, mode='optimize', t_val=None,
            sdpnal_tol=1e-6, sdpnal_maxiter=2000,
            data_dir=data_dir, tag='verify_d4_l2', verbose=False,
        )
    except Exception as exc:
        eng.quit()
        return _fail(f"MatlabRunner.solve failed: {exc}")

    eng.quit()

    obj = res.get('obj')
    termcode = res.get('termcode')
    elapsed = time.time() - t0

    if termcode != 0:
        return _fail(
            f"d=4 L2 smoke test did not solve cleanly "
            f"(termcode={termcode}, obj={obj})",
            "inspect the MATLAB log in data/sdpnal_verify/ for details."
        )

    print(f"       d=4 L2 scalar lb = {obj:.6f}  "
          f"(termcode={termcode}, {elapsed:.1f}s)")
    print()
    print("=== ALL CHECKS PASSED ===")
    print("Ready for d=6 L3 acceptance:")
    print("  python tests/lasserre_sdpnalplus.py --d 6 --order 3 --bw 5 "
          "--mode bisection --sdpnal-tol 1e-7")
    return 0


if __name__ == '__main__':
    sys.exit(main())
