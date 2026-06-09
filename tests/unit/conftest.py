"""Shared pytest fixtures for ``tls2dseg/tests/unit/``.

These fixtures support the Phase 3 ``config`` test suite. They are
specifically designed to be ``tier_a``-safe: no imports from ``pchandler``
or ``pc2img``, no GPU access, no network, no disk I/O beyond ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Minimal valid YAML — every required-no-default field present, defaults
# elsewhere. Used as a starting point that downstream tests mutate to
# exercise specific failure modes (typo'd keys, missing mode, etc.).
# ─────────────────────────────────────────────────────────────────────────────

MINIMAL_RUNCONFIG_YAML = """\
mode: single-view
io:
  input_path: /tmp/tls2dseg_test_input
  output_dir: /tmp/tls2dseg_test_output
prompt:
  text: wheat
preprocessing:
  output_resolution_m: 0.05
projection:
  features: [intensity, range]
inference:
  sam2_checkpoint: /tmp/tls2dseg_test_sam2.pt
d3d_extraction: {}
fusion: {}
"""


@pytest.fixture
def minimal_runconfig_yaml() -> str:
    """The smallest valid RunConfig YAML covering every required-no-default
    field. Tests can ``.replace()`` substrings to inject specific failure
    cases without rewriting the whole document.
    """
    return MINIMAL_RUNCONFIG_YAML


@pytest.fixture
def minimal_yaml_path(tmp_path: Path, minimal_runconfig_yaml: str) -> Path:
    """Path to a temp YAML file pre-populated with the minimal RunConfig."""
    p = tmp_path / "minimal.yaml"
    p.write_text(minimal_runconfig_yaml, encoding="utf-8")
    return p


@pytest.fixture
def tmp_yaml_path(tmp_path: Path) -> Path:
    """Empty Path inside tmp_path that tests can populate per case."""
    return tmp_path / "config.yaml"
