"""Tier-A DOC-06 syntax lock-in — examples/api_usage.py compiles without errors."""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

import tls2dseg


@pytest.mark.tier_a
def test_api_usage_py_compiles() -> None:
    """examples/api_usage.py has valid Python syntax (no SyntaxError).

    Uses py_compile — no real imports, no GPU, no pchandler/pc2img.
    Tier A safe.
    """
    repo_root = Path(tls2dseg.__file__).resolve().parent.parent.parent
    api_usage = repo_root / "examples" / "api_usage.py"
    assert api_usage.exists(), f"examples/api_usage.py not found at {api_usage}"
    py_compile.compile(str(api_usage), doraise=True)
