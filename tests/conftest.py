"""tests/conftest.py

Shared fixtures and configuration for all tests.
"""
import pathlib
import sys

# Ensure src/ is on the Python path for all tests
_root = pathlib.Path(__file__).resolve().parent.parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
