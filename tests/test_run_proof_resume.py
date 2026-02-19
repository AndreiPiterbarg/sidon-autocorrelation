"""Regression tests for run_proof checkpoint resume semantics."""
import json
import os
import sys
import tempfile
import types
import unittest

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


class TestRunProofResume(unittest.TestCase):
    def test_resume_memory_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            lvl = 2
            survivors = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
            npy_path = os.path.join(tmp, f"level_{lvl}_survivors.npy")
            np.save(npy_path, survivors)
            meta = {
                "c_target": 1.2,
                "m": 50,
                "n_start": 3,
                "level": lvl,
                "storage_mode": "memory",
                "n_survivors": int(len(survivors)),
                "checkpoint_survivor_npy": npy_path,
            }
            with open(os.path.join(tmp, f"level_{lvl}_meta.json"), "w") as f:
                json.dump(meta, f)

            start_level, resumed_survivors, survivor_file = run_proof._try_resume(
                tmp, 1.2, 50, 3
            )
            self.assertEqual(start_level, lvl + 1)
            self.assertIsNone(survivor_file)
            np.testing.assert_array_equal(resumed_survivors, survivors)

    def test_resume_streamed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            lvl = 4
            # Backward-compat empty npy checkpoint.
            np.save(os.path.join(tmp, f"level_{lvl}_survivors.npy"),
                    np.empty((0,), dtype=np.int32))
            streamed = os.path.join(tmp, "survivors_streamed.bin")
            with open(streamed, "wb") as f:
                f.write(b"\x00" * 24)
            meta = {
                "c_target": 1.2,
                "m": 50,
                "n_start": 3,
                "level": lvl,
                "storage_mode": "file",
                "n_survivors": 10,
                "survivor_file_path": streamed,
            }
            with open(os.path.join(tmp, f"level_{lvl}_meta.json"), "w") as f:
                json.dump(meta, f)

            start_level, resumed_survivors, survivor_file = run_proof._try_resume(
                tmp, 1.2, 50, 3
            )
            self.assertEqual(start_level, lvl + 1)
            self.assertIsNone(resumed_survivors)
            self.assertEqual(survivor_file, streamed)

    def test_missing_streamed_file_falls_back_to_older_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Newer streamed checkpoint with missing file (stale).
            lvl_new = 5
            np.save(os.path.join(tmp, f"level_{lvl_new}_survivors.npy"),
                    np.empty((0,), dtype=np.int32))
            stale_meta = {
                "c_target": 1.2,
                "m": 50,
                "n_start": 3,
                "level": lvl_new,
                "storage_mode": "file",
                "n_survivors": 100,
                "survivor_file_path": os.path.join(tmp, "missing.bin"),
            }
            with open(os.path.join(tmp, f"level_{lvl_new}_meta.json"), "w") as f:
                json.dump(stale_meta, f)

            # Older valid memory checkpoint.
            lvl_old = 2
            survivors = np.array([[7, 8, 9]], dtype=np.int32)
            npy_path = os.path.join(tmp, f"level_{lvl_old}_survivors.npy")
            np.save(npy_path, survivors)
            good_meta = {
                "c_target": 1.2,
                "m": 50,
                "n_start": 3,
                "level": lvl_old,
                "storage_mode": "memory",
                "n_survivors": 1,
                "checkpoint_survivor_npy": npy_path,
            }
            with open(os.path.join(tmp, f"level_{lvl_old}_meta.json"), "w") as f:
                json.dump(good_meta, f)

            start_level, resumed_survivors, survivor_file = run_proof._try_resume(
                tmp, 1.2, 50, 3
            )
            self.assertEqual(start_level, lvl_old + 1)
            self.assertIsNone(survivor_file)
            np.testing.assert_array_equal(resumed_survivors, survivors)


if __name__ == "__main__":
    unittest.main()
