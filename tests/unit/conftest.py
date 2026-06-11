"""Shared pytest fixtures for ``tls2dseg/tests/unit/``.

These fixtures support the Phase 3 ``config`` test suite. They are
specifically designed to be ``tier_a``-safe: no imports from ``pchandler``
or ``pc2img``, no GPU access, no network, no disk I/O beyond ``tmp_path``.
"""

from __future__ import annotations

import os
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


# ─────────────────────────────────────────────────────────────────────────────
# NO_COLOR=1 autouse — disables Typer/Rich ANSI color emission in help/echo
# output so substring assertions in CLI-surface tests are stable across
# local + CI terminals. Per UAT 03 defect 3 (run 27271100913 job 80540474910):
# CI's runner emits ANSI codes via Rich, fragmenting `--log-level` into
# `\x1b[..]-\x1b[..]-log\x1b[..]-level` so `"--log-level" in stdout` fails.
# See https://no-color.org/ — Rich + Typer both honor this env var.
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _no_color_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set NO_COLOR=1 for every unit test to suppress Rich/Typer ANSI codes.

    Also defensively deletes BOTH force-color env vars Rich honors —
    FORCE_COLOR and PY_COLORS — because either one takes precedence over
    NO_COLOR in Rich's resolution order. Leaving them set (e.g. a CI runner
    that exports PY_COLORS=1) re-enables ANSI fragmentation of `--log-level`
    into `\\x1b[..]-\\x1b[..]-log..-level` despite NO_COLOR — exactly what made
    the cloud `tier_a` job red while the test passed locally (run 27286224192).
    FORCE_COLOR alone was handled before; PY_COLORS was the missed one.

    A test that legitimately needs to inspect colorized output should
    override this with `monkeypatch.delenv("NO_COLOR", raising=False)` and
    `monkeypatch.setenv("FORCE_COLOR", "1")` at the top of the test body —
    none exist today.
    """
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("PY_COLORS", raising=False)
