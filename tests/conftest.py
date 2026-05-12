"""Shared pytest configuration for the delsarte_dual test suite.

Ensures the project root is on ``sys.path`` so tests can import
``delsarte_dual`` regardless of the cwd pytest was invoked from.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
