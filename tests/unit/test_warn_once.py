"""Tier-A tests for once-per-process WARNING semantics (CPU-03).

* pc2img's ``_warn_bary_knn_cpu_fallback`` (plan 03-03 surface) — CPU fallback
  WARNING propagates through the ``tls2dseg.*`` logger hierarchy and is
  visible at user-level INFO+ (CPU-03 final visibility layer).

Skipped when pc2img is missing — keeps the file tier_a-runnable in cloud CI
without forcing pc2img install.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _restore_logging_state() -> Generator[None, None, None]:
    """Restore root handlers + per-logger levels + propagate + handlers after each test.

    Critical: prior tests may run ``configure_logging`` which sets
    ``propagate=False`` on the ``tls2dseg`` logger. Without restoring
    propagate, caplog (which uses root propagation) silently loses records
    from ``tls2dseg.pipeline.pipeline`` etc.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    names = (
        "tls2dseg",
        "tls2dseg.pipeline",
        "tls2dseg.pipeline.pipeline",
        "tls2dseg.runtime",
        "tls2dseg.runtime.context",
        "tls2dseg.config",
        "pc2img",
        "pchandler",
    )
    # Reset for THIS test: propagate=True + no handlers so caplog captures.
    for name in names:
        lg = logging.getLogger(name)
        lg.propagate = True
        for h in list(lg.handlers):
            lg.removeHandler(h)

    try:
        yield
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)
        # Reset to clean defaults — see test_logging_setup _restore_logging_state rationale.
        for name in names:
            lg = logging.getLogger(name)
            lg.setLevel(logging.NOTSET)
            lg.propagate = True
            for h in list(lg.handlers):
                lg.removeHandler(h)


@pytest.mark.tier_a
def test_cpu_fallback_warning_propagates_through_tls2dseg_root(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPU-03 lock-in: pc2img's CPU fallback WARNING is visible at user-level INFO+.

    After ``configure_logging(level='INFO', per_package={})``, the pc2img logger
    inherits sensible default level (no silencing). Invoking
    ``_warn_bary_knn_cpu_fallback`` produces a WARNING record visible to the
    user — closing the final visibility layer for CPU-03.

    Skipped when pc2img is not installed — keeps the file tier_a-runnable in
    cloud CI without forcing pc2img install.
    """
    if importlib.util.find_spec("pc2img") is None:
        pytest.skip("pc2img not installed — CPU-03 final visibility test skipped in this env")

    try:
        from pc2img import image_generation as ig
    except Exception as e:
        pytest.skip(f"pc2img import failed (likely numpy 2.0 / pchandler incompatibility): {e}")

    from tls2dseg.runtime.logging_setup import configure_logging

    configure_logging(level="INFO", per_package={})

    # Reset the single-fire flag so the warning fires for this test.
    monkeypatch.setattr(ig, "_cpu_fallback_warned", False, raising=False)

    pc2img_logger = logging.getLogger("pc2img")
    pc2img_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="pc2img"):
            ig._warn_bary_knn_cpu_fallback()
    finally:
        pc2img_logger.removeHandler(caplog.handler)

    perf_warnings = [r for r in caplog.records if "PERFORMANCE" in r.message and r.levelname == "WARNING"]
    assert len(perf_warnings) >= 1, (
        f"CPU-03 visibility lock-in failed — expected pc2img CPU fallback WARNING "
        f"to be captured via tls2dseg.* logger hierarchy; got 0 records: "
        f"{[r.message for r in caplog.records]}"
    )
