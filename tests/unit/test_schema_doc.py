"""Tier-A tests for :mod:`tls2dseg.config.schema_doc` — DOC-02 lock-in.

Covers:
- generate_schema_doc() returns a non-empty string (no NotImplementedError).
- Output contains all four tag group headings: primary, tuning, pipings, other.
- Output contains at least one field from each discriminated-union member
  (sam2_checkpoint from GroundedSAM2Config; sam2_hf_model_id from GroundedSAM2HFConfig).
- ``tls2dseg schema`` CLI subcommand exits 0 and prints content containing "primary".
"""

from __future__ import annotations

import logging
from collections.abc import Generator

import pytest

# ─── logging fixture (copied from test_cli_surface.py) ──────────────────────

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
    """Restore root + reset per-logger state after each test."""
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


# ─── generator tests ─────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_generate_schema_doc_returns_nonempty_string() -> None:
    """generate_schema_doc() returns a non-empty str — no NotImplementedError."""
    from tls2dseg.config.schema_doc import generate_schema_doc

    doc = generate_schema_doc()
    assert isinstance(doc, str)
    assert len(doc) > 0


@pytest.mark.tier_a
def test_generate_schema_doc_contains_all_tag_groups() -> None:
    """Output contains all four tag-group headings: primary, tuning, pipings, other."""
    from tls2dseg.config.schema_doc import generate_schema_doc

    doc = generate_schema_doc()
    for tag in ("primary", "tuning", "pipings", "other"):
        assert tag in doc, f"expected tag group '{tag}' in schema doc"


@pytest.mark.tier_a
def test_generate_schema_doc_union_both_members_present() -> None:
    """Output includes fields from both GroundedSAM2Config and GroundedSAM2HFConfig.

    sam2_checkpoint is unique to GroundedSAM2Config;
    sam2_hf_model_id is unique to GroundedSAM2HFConfig.
    """
    from tls2dseg.config.schema_doc import generate_schema_doc

    doc = generate_schema_doc()
    assert "sam2_checkpoint" in doc, "GroundedSAM2Config field 'sam2_checkpoint' not found"
    assert "sam2_hf_model_id" in doc, "GroundedSAM2HFConfig field 'sam2_hf_model_id' not found"


@pytest.mark.tier_a
def test_generate_schema_doc_no_duplicate_field_rows() -> None:
    """Shared InferenceSharedConfig fields must appear exactly once (WR-02 lock-in)."""
    from tls2dseg.config.schema_doc import generate_schema_doc

    doc = generate_schema_doc()
    assert doc.count("box_threshold") == 1, "box_threshold must appear exactly once (no inherited-field duplicates)"
    assert doc.count("text_threshold") == 1, "text_threshold must appear exactly once"


@pytest.mark.tier_a
def test_generate_schema_doc_no_pydantic_undefined() -> None:
    """No field default may render as the string 'PydanticUndefined' (WR-03 lock-in)."""
    from tls2dseg.config.schema_doc import generate_schema_doc

    doc = generate_schema_doc()
    assert "PydanticUndefined" not in doc, "PydanticUndefined must not appear in generated schema doc"


# ─── CLI test ────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_schema_cmd_exits_zero_and_prints_output() -> None:
    """``tls2dseg schema`` exits 0 and stdout contains 'primary'."""
    from typer.testing import CliRunner

    from tls2dseg.cli import app

    result = CliRunner().invoke(app, ["schema"])
    assert result.exit_code == 0, result.stdout
    assert "primary" in result.stdout
