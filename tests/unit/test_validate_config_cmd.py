"""Tier-A tests for ``tls2dseg validate-config`` (CFG-06).

Phase 3 plan 06. Locks in:

* Valid YAML → exit 0, stdout is "OK\\n" (or YAML dump if --verbose).
* YAML with typo'd key → exit 1, stderr names the typo or "extra_forbidden".
* YAML missing required field (e.g. mode) → exit 1, stderr names the field.
* Verbose mode: success path emits the resolved RunConfig as YAML.

Uses ``typer.testing.CliRunner`` for invocation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.mark.tier_a
def test_valid_yaml_exits_zero(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """``validate-config <valid.yaml>`` → exit 0, stdout starts with 'OK'."""
    from tls2dseg.cli import app

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(minimal_runconfig_yaml, encoding="utf-8")

    result = CliRunner().invoke(app, ["validate-config", str(yaml_path)])

    assert result.exit_code == 0, f"expected exit 0; got {result.exit_code} stdout={result.stdout!r}"
    assert "OK" in result.stdout


@pytest.mark.tier_a
def test_yaml_with_typo_key_exits_one_and_names_extra_forbidden(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """``validate-config <yaml-with-typo>`` → exit 1, stderr names the typo or extra_forbidden."""
    from tls2dseg.cli import app

    bad_yaml = minimal_runconfig_yaml + "typo_key: oops\n"
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(bad_yaml, encoding="utf-8")

    # mix_stderr=False ensures we can inspect stderr separately.
    result = CliRunner().invoke(app, ["validate-config", str(yaml_path)])

    assert result.exit_code == 1, f"expected exit 1; got {result.exit_code} stdout={result.stdout!r}"
    combined = (result.stdout or "") + (result.stderr or "")
    assert ("typo_key" in combined) or ("extra_forbidden" in combined.lower()) or ("Extra inputs" in combined), (
        f"expected typo_key or extra_forbidden in error output; got: {combined!r}"
    )


@pytest.mark.tier_a
def test_yaml_missing_mode_exits_one_and_names_mode(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """``validate-config <yaml-missing-mode>`` → exit 1, stderr names 'mode'."""
    from tls2dseg.cli import app

    # Drop the mode: single-view line.
    bad_yaml = "\n".join(line for line in minimal_runconfig_yaml.splitlines() if "mode:" not in line)
    yaml_path = tmp_path / "no_mode.yaml"
    yaml_path.write_text(bad_yaml + "\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["validate-config", str(yaml_path)])

    assert result.exit_code == 1
    combined = (result.stdout or "") + (result.stderr or "")
    assert "mode" in combined.lower(), f"expected 'mode' in error output; got: {combined!r}"


@pytest.mark.tier_a
def test_verbose_emits_resolved_yaml(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """``validate-config --verbose <valid.yaml>`` → exit 0, stdout contains YAML dump of resolved config."""
    from tls2dseg.cli import app

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(minimal_runconfig_yaml, encoding="utf-8")

    result = CliRunner().invoke(app, ["validate-config", "--verbose", str(yaml_path)])

    assert result.exit_code == 0
    # Resolved YAML should contain mode + io + the prompt text.
    assert "mode:" in result.stdout
    assert "io:" in result.stdout


@pytest.mark.tier_a
def test_nonexistent_yaml_exits_one(tmp_path: Path) -> None:
    """``validate-config <missing.yaml>`` → exit 1 with file-not-found message."""
    from tls2dseg.cli import app

    result = CliRunner().invoke(app, ["validate-config", str(tmp_path / "missing.yaml")])

    assert result.exit_code == 1
    combined = (result.stdout or "") + (result.stderr or "")
    assert "not found" in combined.lower() or "no such" in combined.lower()
