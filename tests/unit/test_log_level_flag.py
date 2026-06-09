"""Tier-A tests for the ``--log-level`` flag mechanism (LOG-04).

Phase 3 plan 06 — the CLI ``--log-level`` flag is wired in Task 2a via
``configure_logging(level=log_level, ...)``. This test locks the underlying
mechanism: changing the ``level`` arg on ``configure_logging`` actually
changes the effective level reported by ``tls2dseg.*`` loggers.

The CLI surface integration is exercised in ``test_cli_surface.py`` (Task
2b); this file isolates the mechanism so a regression in dictConfig wiring
is caught even when CliRunner exercise is unavailable.
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
    for name in ("tls2dseg", "tls2dseg.config", "tls2dseg.runtime", "tls2dseg.cli", "tls2dseg.pipeline"):
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
def test_log_level_debug_propagates_to_all_tls2dseg_subloggers() -> None:
    """``configure_logging('DEBUG')`` → all tls2dseg.* sub-loggers at DEBUG (LOG-04)."""
    from tls2dseg.runtime.logging_setup import configure_logging

    configure_logging(level="DEBUG")

    for name in ("tls2dseg", "tls2dseg.config", "tls2dseg.runtime", "tls2dseg.cli", "tls2dseg.pipeline"):
        assert logging.getLogger(name).getEffectiveLevel() == logging.DEBUG, (
            f"sub-logger {name!r} did not inherit DEBUG — LOG-04 mechanism broken"
        )


@pytest.mark.tier_a
def test_log_level_warning_filters_info_records(caplog: pytest.LogCaptureFixture) -> None:
    """``configure_logging('WARNING')`` filters out INFO records on tls2dseg.* loggers.

    Implementation detail: tls2dseg has ``propagate=False`` post-dictConfig; we
    attach ``caplog.handler`` to the child logger so the assertion records
    actually reach the caplog fixture. We deliberately do NOT use
    ``caplog.at_level`` — that overrides the logger level and would defeat
    the LOG-04 filtering we're trying to verify.
    """
    from tls2dseg.runtime.logging_setup import configure_logging

    configure_logging(level="WARNING")

    test_logger = logging.getLogger("tls2dseg.test")
    test_logger.addHandler(caplog.handler)
    try:
        test_logger.info("info-record")
        test_logger.warning("warn-record")
    finally:
        test_logger.removeHandler(caplog.handler)

    messages = [r.message for r in caplog.records if r.name == "tls2dseg.test"]
    assert "warn-record" in messages
    assert "info-record" not in messages


@pytest.mark.tier_a
def test_log_level_string_case_insensitive() -> None:
    """``configure_logging('debug')`` (lowercase) works the same as 'DEBUG'.

    The CLI may pass any casing; we normalize internally.
    """
    from tls2dseg.runtime.logging_setup import configure_logging

    configure_logging(level="debug")

    assert logging.getLogger("tls2dseg").getEffectiveLevel() == logging.DEBUG
