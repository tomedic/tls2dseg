"""Tier-A tests for :mod:`tls2dseg.runtime.logging_setup` — two-phase logging.

Phase 3 plan 06 (LOG-02 + LOG-04). Locks in:

* ``bootstrap_logger(level)`` (Phase 1 — basicConfig before load_config so
  config-loader DEBUG/INFO is captured).
* ``configure_logging(level, per_package, log_file)`` (Phase 2 — dictConfig
  with structured handlers, per-package overrides, optional file handler).
* ``bootstrap()`` — TOKENIZERS_PARALLELISM env var helper, moved from
  pipeline/run.py:18.
* The two-phase contract: DEBUG records emitted between Phase 1 and Phase 2
  are NOT lost; dictConfig cleanly replaces basicConfig handlers per RESEARCH
  §Pattern 8 ("dictConfig replaces root logger handlers as a unit").

Test isolation: a session-scoped fixture snapshots+restores root logger state
so tests don't bleed configured handlers into each other.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _restore_logging_state() -> Generator[None, None, None]:
    """Snapshot + restore root logger handlers + per-logger levels per test.

    Without this, ``configure_logging`` mutations leak across tests:
    dictConfig replaces root handlers, basicConfig adds another set, and
    subsequent tests see stale state.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_logger_states: dict[str, int] = {}
    for name in ("tls2dseg", "tls2dseg.config", "tls2dseg.config.loader", "tls2dseg.runtime", "pchandler", "pc2img"):
        lg = logging.getLogger(name)
        saved_logger_states[name] = lg.level

    try:
        yield
    finally:
        # Restore root handlers + level
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)
        # Restore per-logger levels
        for name, level in saved_logger_states.items():
            logging.getLogger(name).setLevel(level)


@pytest.mark.tier_a
def test_bootstrap_logger_sets_root_level_to_info() -> None:
    """After ``bootstrap_logger('INFO')`` the root logger is at INFO."""
    from tls2dseg.runtime.logging_setup import bootstrap_logger

    bootstrap_logger(level="INFO")

    assert logging.getLogger().getEffectiveLevel() == logging.INFO


@pytest.mark.tier_a
def test_bootstrap_logger_captures_records_before_dictconfig(caplog: pytest.LogCaptureFixture) -> None:
    """DEBUG record emitted BETWEEN bootstrap_logger and configure_logging is captured.

    This is the WARNING-5 fix lock-in: the two-phase split exists so that
    config/loader DEBUG output emitted DURING load_config (between phase 1
    and phase 2) is NOT silently dropped by dictConfig's later handler
    replacement.

    Implementation detail: ``configure_logging`` sets ``propagate=False`` on the
    ``tls2dseg`` logger (avoid root-handler double-output in production). The
    pytest ``caplog`` fixture attaches its handler to the root logger and
    relies on propagation, so we manually attach ``caplog.handler`` to the
    ``tls2dseg.config.loader`` logger after Phase 2 to capture phase-2 records.
    """
    from tls2dseg.runtime.logging_setup import bootstrap_logger, configure_logging

    loader_logger = logging.getLogger("tls2dseg.config.loader")

    bootstrap_logger(level="DEBUG")
    # PHASE 1: attach caplog handler to capture pre-dictConfig records.
    loader_logger.addHandler(caplog.handler)
    try:
        loader_logger.debug("config load starting")
    finally:
        loader_logger.removeHandler(caplog.handler)

    configure_logging(level="DEBUG")

    # PHASE 2: re-attach caplog handler after dictConfig replaces handlers
    # (per RESEARCH §Pattern 8 — dictConfig replaces root logger handlers as
    # a unit; tls2dseg.* gets propagate=False so caplog's root handler is
    # bypassed).
    loader_logger.addHandler(caplog.handler)
    try:
        loader_logger.debug("config load complete")
    finally:
        loader_logger.removeHandler(caplog.handler)

    messages = [r.message for r in caplog.records]
    assert "config load starting" in messages, (
        "phase-1 DEBUG record was dropped by phase-2 dictConfig — two-phase contract violated"
    )
    assert "config load complete" in messages


@pytest.mark.tier_a
def test_configure_logging_replaces_bootstrap_handlers_cleanly() -> None:
    """dictConfig cleanly replaces basicConfig handlers — no doubled output, no leftovers.

    Sequence: bootstrap_logger(INFO) → configure_logging(DEBUG, per_package={pchandler: WARNING}).
    After phase 2:
      - tls2dseg.config effective level = DEBUG
      - pchandler effective level = WARNING (per_package override)
    """
    from tls2dseg.runtime.logging_setup import bootstrap_logger, configure_logging

    bootstrap_logger(level="INFO")
    configure_logging(level="DEBUG", per_package={"pchandler": "WARNING"})

    assert logging.getLogger("tls2dseg.config").getEffectiveLevel() == logging.DEBUG
    assert logging.getLogger("pchandler").getEffectiveLevel() == logging.WARNING


@pytest.mark.tier_a
def test_configure_logging_info_sets_runtime_logger_level() -> None:
    """``configure_logging('INFO')`` → ``tls2dseg.runtime`` effective level is INFO (LOG-02)."""
    from tls2dseg.runtime.logging_setup import configure_logging

    configure_logging(level="INFO")

    assert logging.getLogger("tls2dseg.runtime").getEffectiveLevel() == logging.INFO


@pytest.mark.tier_a
def test_configure_logging_debug_propagates_to_children() -> None:
    """``configure_logging('DEBUG')`` → both ``tls2dseg.config`` AND ``tls2dseg.runtime`` are DEBUG.

    This is the LOG-04 wiring proof: the parent ``tls2dseg`` logger
    configuration propagates to children via Python's standard hierarchy.
    """
    from tls2dseg.runtime.logging_setup import configure_logging

    configure_logging(level="DEBUG")

    assert logging.getLogger("tls2dseg.config").getEffectiveLevel() == logging.DEBUG
    assert logging.getLogger("tls2dseg.runtime").getEffectiveLevel() == logging.DEBUG
    # Children of children also inherit.
    assert logging.getLogger("tls2dseg.config.models").getEffectiveLevel() == logging.DEBUG


@pytest.mark.tier_a
def test_configure_logging_can_be_called_twice_without_error() -> None:
    """Repeat ``configure_logging`` calls don't pile up handlers or crash."""
    from tls2dseg.runtime.logging_setup import configure_logging

    configure_logging(level="INFO")
    configure_logging(level="DEBUG")

    # Latest call wins.
    assert logging.getLogger("tls2dseg").getEffectiveLevel() == logging.DEBUG


@pytest.mark.tier_a
def test_configure_logging_writes_to_file(tmp_path) -> None:
    """``log_file`` arg produces a FileHandler on the ``tls2dseg`` logger."""
    from tls2dseg.runtime.logging_setup import configure_logging

    log_file = tmp_path / "run.log"
    configure_logging(level="INFO", log_file=str(log_file))

    logging.getLogger("tls2dseg.test").info("hello file handler")

    # File should exist + contain our record.
    assert log_file.is_file()
    text = log_file.read_text(encoding="utf-8")
    assert "hello file handler" in text


@pytest.mark.tier_a
def test_bootstrap_sets_tokenizers_parallelism_env_var() -> None:
    """``bootstrap()`` sets TOKENIZERS_PARALLELISM=true (moved from pipeline/run.py:18)."""
    from tls2dseg.runtime.logging_setup import bootstrap

    # Clear first (test isolation).
    os.environ.pop("TOKENIZERS_PARALLELISM", None)

    bootstrap()

    assert os.environ.get("TOKENIZERS_PARALLELISM") == "true"


@pytest.mark.tier_a
def test_bootstrap_respects_existing_env_var() -> None:
    """``bootstrap()`` uses setdefault — doesn't clobber user-set values."""
    from tls2dseg.runtime.logging_setup import bootstrap

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        bootstrap()
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    finally:
        os.environ.pop("TOKENIZERS_PARALLELISM", None)
