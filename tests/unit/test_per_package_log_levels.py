"""Tier-A tests for per-package log level overrides (LOG-03).

Phase 3 plan 06 — ``configure_logging(per_package={...})`` sets per-logger
levels for pchandler/pc2img/anything-else the user names. Empty dict (D-A1-10
conservative default) means nothing is silenced.

The hardcoded ``logging.getLogger('pchandler').setLevel(logging.ERROR)`` at
pipeline/run.py:21 is removed in Task 3; this test locks in the
configuration-driven replacement.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _restore_logging_state() -> Generator[None, None, None]:
    """Restore root handlers + per-logger levels after each test."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_logger_states: dict[str, int] = {}
    for name in ("tls2dseg", "pchandler", "pc2img"):
        lg = logging.getLogger(name)
        saved_logger_states[name] = lg.level

    try:
        yield
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)
        for name, level in saved_logger_states.items():
            logging.getLogger(name).setLevel(level)


@pytest.mark.tier_a
def test_per_package_overrides_pchandler_and_pc2img() -> None:
    """``per_package={'pchandler':'WARNING','pc2img':'INFO'}`` → both loggers at the requested levels."""
    from tls2dseg.runtime.logging_setup import configure_logging

    configure_logging(level="INFO", per_package={"pchandler": "WARNING", "pc2img": "INFO"})

    assert logging.getLogger("pchandler").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("pc2img").getEffectiveLevel() == logging.INFO


@pytest.mark.tier_a
def test_per_package_empty_dict_leaves_loggers_at_parent_inherited_level() -> None:
    """Empty ``per_package={}`` (LOG-03 conservative default) — nothing is silenced.

    With no per-package override and root at WARNING (the dictConfig default
    root), pchandler/pc2img inherit the WARNING root — they are NOT raised
    to ERROR (the legacy hardcode behavior we're removing).
    """
    from tls2dseg.runtime.logging_setup import configure_logging

    configure_logging(level="INFO", per_package={})

    # No per-package config → inherits root (WARNING by default dictConfig)
    pch_level = logging.getLogger("pchandler").getEffectiveLevel()
    pc2_level = logging.getLogger("pc2img").getEffectiveLevel()
    # Critical: NOT silenced to ERROR (the legacy hardcode behavior).
    assert pch_level <= logging.WARNING, (
        f"pchandler must NOT be silenced to ERROR by default — got effective level {pch_level} "
        f"(LOG-03 removes the hardcoded ERROR setLevel)"
    )
    assert pc2_level <= logging.WARNING


@pytest.mark.tier_a
def test_per_package_overrides_emit_correct_records(caplog: pytest.LogCaptureFixture) -> None:
    """A WARNING from pchandler is emitted when its level is WARNING; INFO is not.

    Implementation detail: ``configure_logging`` sets ``propagate=False`` on the
    per-package loggers (avoid root-handler double-output). caplog uses root
    propagation, so we attach ``caplog.handler`` to the pchandler logger
    directly. We deliberately do NOT use ``caplog.at_level`` — that overrides
    the logger level and would defeat the per-package filtering we're trying
    to verify.
    """
    from tls2dseg.runtime.logging_setup import configure_logging

    configure_logging(level="INFO", per_package={"pchandler": "WARNING"})

    pch = logging.getLogger("pchandler")
    pch.addHandler(caplog.handler)
    try:
        pch.info("info-from-pchandler")
        pch.warning("warn-from-pchandler")
    finally:
        pch.removeHandler(caplog.handler)

    messages = [r.message for r in caplog.records if r.name == "pchandler"]
    assert "warn-from-pchandler" in messages
    # INFO should be filtered out by the per_package WARNING override.
    assert "info-from-pchandler" not in messages


@pytest.mark.tier_a
def test_per_package_none_treated_as_empty() -> None:
    """``per_package=None`` (default arg) → same as empty dict — no overrides."""
    from tls2dseg.runtime.logging_setup import configure_logging

    configure_logging(level="INFO", per_package=None)

    # No exception, sensible defaults.
    assert logging.getLogger("tls2dseg").getEffectiveLevel() == logging.INFO
