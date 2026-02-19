"""Safety regression tests for strict fail-closed proof behavior."""
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if "numba" not in sys.modules:
    def _njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _decorator(fn):
            return fn
        return _decorator

    sys.modules["numba"] = types.SimpleNamespace(
        njit=_njit,
        prange=range,
        boolean=bool,
    )
import run_proof


def _make_base_result(n_survivors, n_extracted, d=6):
    surv_shape = (n_extracted, d) if n_extracted > 0 else (0, d)
    return {
        "proven": n_survivors == 0,
        "proven_bound": 1.2 if n_survivors == 0 else None,
        "n_survivors": n_survivors,
        "survivors": np.zeros(surv_shape, dtype=np.int32),
        "survivor_file": None,
        "stats": {
            "n_pruned_asym": 0,
            "n_pruned_test": 0,
            "n_fp32_skipped": 0,
            "n_extracted": n_extracted,
            "streamed": False,
        },
    }


class TestRunProofSafety(unittest.TestCase):
    def _run_main(
        self,
        base_result,
        refine_result=None,
        max_levels=2,
        time_budget=3600,
        expect_exit=True,
    ):
        saved = []

        def save_capture(results, _path):
            saved.append(json.loads(json.dumps(results, default=str)))

        def refine_side_effect(*_args, **_kwargs):
            if refine_result is None:
                raise AssertionError("refine_parents should not have been called")
            return refine_result

        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "run_proof.py",
                "--target", "1.2",
                "--m", "20",
                "--n-start", "3",
                "--max-levels", str(max_levels),
                "--time-budget", str(time_budget),
                "--checkpoint-dir", tmp,
                "--force",
            ]
            with mock.patch.object(run_proof.sys, "argv", argv), \
                    mock.patch.object(run_proof, "save_results", side_effect=save_capture), \
                    mock.patch.object(run_proof, "_print_summary"), \
                    mock.patch.object(run_proof, "is_available", return_value=True), \
                    mock.patch.object(run_proof, "get_device_name", return_value="Mock GPU"), \
                    mock.patch.object(run_proof, "gpu_find_best_bound_direct", return_value=1.0), \
                    mock.patch.object(run_proof, "count_compositions", return_value=10), \
                    mock.patch.object(run_proof, "get_free_memory", return_value=80 * 1024**3), \
                    mock.patch.object(run_proof, "gpu_run_single_level", return_value=base_result), \
                    mock.patch.object(run_proof, "max_survivors_for_dim", return_value=1000), \
                    mock.patch.object(run_proof, "refine_parents", side_effect=refine_side_effect), \
                    mock.patch.object(run_proof.platform, "system", return_value="Linux"):
                if expect_exit:
                    with self.assertRaises(SystemExit) as cm:
                        run_proof.main()
                    code = cm.exception.code
                else:
                    try:
                        run_proof.main()
                        code = 0
                    except SystemExit as cm:
                        code = cm.code
        return code, saved

    def test_refinement_timeout_with_zero_survivors_is_inconclusive(self):
        base_result = _make_base_result(n_survivors=1, n_extracted=1, d=6)
        refine_result = {
            "total_asym": 0,
            "total_test": 1,
            "total_survivors": 0,
            "min_test_val": 1.0,
            "min_test_config": np.zeros(12, dtype=np.int32),
            "survivor_configs": np.empty((0, 12), dtype=np.int32),
            "n_extracted": 0,
            "timed_out": True,
        }
        code, saved = self._run_main(base_result, refine_result, max_levels=2)
        self.assertEqual(code, 2)
        self.assertEqual(saved[-1]["status"], "inconclusive")
        self.assertIn("timed_out", saved[-1]["inconclusive_reason"])

    def test_refinement_extraction_truncation_is_inconclusive(self):
        base_result = _make_base_result(n_survivors=1, n_extracted=1, d=6)
        refine_result = {
            "total_asym": 0,
            "total_test": 0,
            "total_survivors": 5,
            "min_test_val": 1.0,
            "min_test_config": np.zeros(12, dtype=np.int32),
            "survivor_configs": np.zeros((2, 12), dtype=np.int32),
            "n_extracted": 2,
            "timed_out": False,
        }
        code, saved = self._run_main(base_result, refine_result, max_levels=2)
        self.assertEqual(code, 2)
        self.assertEqual(saved[-1]["status"], "inconclusive")
        self.assertIn("extraction_truncated", saved[-1]["inconclusive_reason"])

    def test_refinement_extraction_exact_is_safe_to_continue(self):
        base_result = _make_base_result(n_survivors=1, n_extracted=1, d=6)
        refine_result = {
            "total_asym": 0,
            "total_test": 0,
            "total_survivors": 1,
            "min_test_val": 1.0,
            "min_test_config": np.zeros(12, dtype=np.int32),
            "survivor_configs": np.zeros((1, 12), dtype=np.int32),
            "n_extracted": 1,
            "timed_out": False,
        }
        code, saved = self._run_main(
            base_result,
            refine_result,
            max_levels=2,
            expect_exit=False,
        )
        self.assertEqual(code, 0)
        self.assertEqual(saved[-1]["status"], "not_proven")
        self.assertIsNone(saved[-1]["inconclusive_reason"])

    def test_base_extraction_truncation_is_inconclusive(self):
        base_result = _make_base_result(n_survivors=5, n_extracted=2, d=6)
        code, saved = self._run_main(base_result, refine_result=None, max_levels=1)
        self.assertEqual(code, 2)
        self.assertEqual(saved[-1]["status"], "inconclusive")
        self.assertEqual(saved[-1]["inconclusive_reason"], "base_extraction_truncated")

    def test_time_budget_exhausted_before_level_is_inconclusive(self):
        base_result = _make_base_result(n_survivors=0, n_extracted=0, d=6)
        code, saved = self._run_main(
            base_result,
            refine_result=None,
            max_levels=1,
            time_budget=1,
        )
        self.assertEqual(code, 2)
        self.assertEqual(saved[-1]["status"], "inconclusive")
        self.assertEqual(
            saved[-1]["inconclusive_reason"],
            "time_budget_exhausted_before_level_0",
        )


if __name__ == "__main__":
    unittest.main()
