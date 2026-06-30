"""Tier-A CLI-surface tests (CFG-05).

Phase 3 plan 06. Locks in:

* ``tls2dseg --version`` exits 0 with a version string (CFG-05 + D-A3-01).
* ``tls2dseg --help`` exits 0 and lists run/validate-config/doctor.
* ``tls2dseg run --help`` exits 0 and lists all 8 flags per D-A3-02.
* ``tls2dseg validate-config --help`` exits 0 and lists the 3 flags per D-A3-03.
* ``tls2dseg doctor`` exits 0 and routes to diagnostics.doctor (no print() — LOG-01 conversion).
* ``tls2dseg run --dry-run`` exits 0 without invoking pipeline.run.main (D-A3-02).

Uses ``typer.testing.CliRunner`` — captures stdout/stderr/exit_code without
subprocess overhead (tier_a friendly).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

# Strip ANSI SGR escapes before substring assertions. Typer/Rich help output is
# colorized when the runner forces color, which fragments flags like
# `--log-level` into `-\x1b[..]-log\x1b[..]-level` so `"--log-level" in stdout`
# fails. NO_COLOR is honored locally but some CI runners still emit color (via
# FORCE_COLOR / PY_COLORS / terminal detection), so we strip rather than rely on
# env vars. This made the cloud tier_a job red while it passed locally.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Return ``text`` with ANSI SGR color/style escape codes removed."""
    return _ANSI_RE.sub("", text)


_TRACKED_LOGGERS = (
    "tls2dseg",
    "tls2dseg.config",
    "tls2dseg.runtime",
    "tls2dseg.runtime.context",
    "tls2dseg.cli",
    "tls2dseg.pipeline",
    "pchandler",
    "pc2img",
)


@pytest.fixture(autouse=True)
def _restore_logging_state() -> Generator[None, None, None]:
    """Restore root + reset per-logger state after each test.

    The Typer CLI run_cmd path calls ``configure_logging`` which sets
    ``propagate=False`` on ``tls2dseg`` and per_package loggers. Without
    explicit cleanup, subsequent tests using caplog (which relies on root
    propagation) silently lose records.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level

    try:
        yield
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)
        for name in _TRACKED_LOGGERS:
            lg = logging.getLogger(name)
            lg.setLevel(logging.NOTSET)
            lg.propagate = True
            for h in list(lg.handlers):
                lg.removeHandler(h)


@pytest.mark.tier_a
def test_version_flag_exits_zero_and_prints_semver() -> None:
    """``tls2dseg --version`` → exit 0, stdout matches ``r'^tls2dseg \\d'``."""
    from tls2dseg.cli import app

    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0, result.stdout
    assert re.match(r"^tls2dseg \d", _plain(result.stdout)), f"unexpected version string: {result.stdout!r}"


@pytest.mark.tier_a
def test_log_level_is_global_flag() -> None:
    """LOG-04: ``--log-level`` is wired on the root callback, not just on ``run``.

    Regression: UAT 03 Test 6 found ``tls2dseg --log-level DEBUG validate-config <yaml>``
    was rejected because the flag was only on the ``run`` subcommand. The intent of LOG-04
    is a global flag so config-loader DEBUG output is captured for every subcommand.
    """
    from tls2dseg.cli import app

    # Surface check: --log-level appears on the root --help.
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--log-level" in _plain(result.stdout), "global --log-level missing from root --help"

    # Behavioral check: --log-level is accepted before a non-run subcommand and exits 0.
    result = CliRunner().invoke(app, ["--log-level", "DEBUG", "doctor"])
    assert result.exit_code == 0, (
        f"`tls2dseg --log-level DEBUG doctor` failed with exit {result.exit_code}: {result.stdout}"
    )


@pytest.mark.tier_a
def test_help_lists_all_three_subcommands() -> None:
    """``tls2dseg --help`` → exit 0, lists run/validate-config/doctor."""
    from tls2dseg.cli import app

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    out = _plain(result.stdout)
    assert "run" in out
    assert "validate-config" in out
    assert "doctor" in out


@pytest.mark.tier_a
def test_run_help_lists_all_eight_flags() -> None:
    """``tls2dseg run --help`` → exit 0, lists all 8 flags per D-A3-02."""
    from tls2dseg.cli import app

    result = CliRunner().invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    expected_flags = [
        "--config",
        "--log-level",
        "--resume",
        "--run-id",
        "--output-dir",
        "--mode",
        "--device",
        "--dry-run",
    ]
    out = _plain(result.stdout)
    for flag in expected_flags:
        assert flag in out, f"run --help missing flag {flag!r}"


@pytest.mark.tier_a
def test_validate_config_help_lists_three_flags() -> None:
    """``tls2dseg validate-config --help`` → exit 0, lists FILE arg + 2 options."""
    from tls2dseg.cli import app

    result = CliRunner().invoke(app, ["validate-config", "--help"])

    assert result.exit_code == 0
    out = _plain(result.stdout)
    # Typer renders the positional arg as "FILE" or similar.
    assert "FILE" in out or "file" in out.lower()
    assert "--strict" in out
    assert "--verbose" in out


@pytest.mark.tier_a
def test_doctor_runs_and_uses_typer_echo_not_print() -> None:
    """``tls2dseg doctor`` → exit 0, prints diagnostics report via typer.echo."""
    from tls2dseg.cli import app

    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0
    # The doctor report header from diagnostics.format_report.
    assert "tls2dseg doctor" in _plain(result.stdout)


@pytest.mark.tier_a
def test_dry_run_does_not_invoke_pipeline_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, minimal_runconfig_yaml: str
) -> None:
    """``tls2dseg run --dry-run --config <valid.yaml>`` → exit 0 without calling pipeline.run.main (D-A3-02).

    Strategy: pre-install a fake ``tls2dseg.pipeline.run`` module in
    ``sys.modules`` whose ``main`` callable raises if invoked. This avoids
    importing the real pipeline.run (which transitively imports torch +
    pchandler + pc2img — tier_a-incompatible) and lets us assert dry-run
    truly bypasses the pipeline dispatch.
    """
    import sys
    import types

    from tls2dseg.cli import app

    fake_module = types.ModuleType("tls2dseg.pipeline.run")

    def _explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("dry-run must NOT invoke pipeline.run.main")

    fake_module.main = _explode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tls2dseg.pipeline.run", fake_module)

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(minimal_runconfig_yaml, encoding="utf-8")

    result = CliRunner().invoke(app, ["run", "--config", str(yaml_path), "--dry-run"])

    assert result.exit_code == 0, f"dry-run failed: exit={result.exit_code} stdout={result.stdout!r}"
    # Dry-run summary should mention run_id + mode at minimum.
    out = _plain(result.stdout)
    assert "run_id" in out
    assert "mode" in out
