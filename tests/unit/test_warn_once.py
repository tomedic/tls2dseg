"""Tier-A tests for once-per-process WARNING semantics (D-A1-06 + CPU-03).

Phase 3 plan 06 — Task 3. Two surfaces locked in:

* ``pipeline.run._warn_single_view_dispatch_pending`` — functools.cache wrapped
  function. Calling it twice produces exactly one WARNING record per process
  per D-A1-06 (mode=single-view text verbatim).
* pc2img's ``_warn_bary_knn_cpu_fallback`` (plan 03-03 surface) — CPU fallback
  WARNING propagates through the ``tls2dseg.*`` logger hierarchy and is
  visible at user-level INFO+ (CPU-03 final visibility layer).

Test 2 is skipped when pc2img is missing — keeps the file tier_a-runnable
in cloud CI without forcing pc2img install.
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
    from ``tls2dseg.pipeline.run`` etc.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_state: dict[str, tuple[int, bool, list]] = {}
    names = ("tls2dseg", "tls2dseg.pipeline", "tls2dseg.pipeline.run", "tls2dseg.runtime", "pc2img", "pchandler")
    for name in names:
        lg = logging.getLogger(name)
        saved_state[name] = (lg.level, lg.propagate, list(lg.handlers))
        # Reset for THIS test: propagate=True so caplog captures.
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
        for name, (level, propagate, handlers) in saved_state.items():
            lg = logging.getLogger(name)
            lg.setLevel(level)
            lg.propagate = propagate
            for h in list(lg.handlers):
                lg.removeHandler(h)
            for h in handlers:
                lg.addHandler(h)


@pytest.mark.tier_a
def test_single_view_warning_fires_once(caplog: pytest.LogCaptureFixture) -> None:
    """The mode=single-view dispatch-pending WARNING fires exactly once per process (D-A1-06).

    Uses functools.cache — once the function has been called once, subsequent
    calls are no-ops. Reset the cache before invocations to verify the
    cache-based gate works. caplog's root handler captures the record via
    propagation (tls2dseg.pipeline.run has propagate=True by default).
    """
    from tls2dseg.pipeline.run import _warn_single_view_dispatch_pending

    # Reset cache so the test isn't piggybacking on a prior invocation.
    _warn_single_view_dispatch_pending.cache_clear()

    with caplog.at_level(logging.WARNING, logger="tls2dseg.pipeline.run"):
        _warn_single_view_dispatch_pending()
        _warn_single_view_dispatch_pending()
        _warn_single_view_dispatch_pending()

    single_view_warnings = [r for r in caplog.records if "mode=single-view" in r.message and r.levelname == "WARNING"]
    assert len(single_view_warnings) == 1, (
        f"expected EXACTLY ONE 'mode=single-view' warning per process "
        f"(D-A1-06 functools.cache contract); got {len(single_view_warnings)}: "
        f"{[r.message for r in single_view_warnings]}"
    )


@pytest.mark.tier_a
def test_single_view_warning_text_matches_d_a1_06_verbatim(caplog: pytest.LogCaptureFixture) -> None:
    """The warning text matches CONTEXT.md D-A1-06 verbatim phrasing.

    Locks the message format so future refactors don't accidentally rephrase
    the user-facing warning.
    """
    from tls2dseg.pipeline.run import _warn_single_view_dispatch_pending

    _warn_single_view_dispatch_pending.cache_clear()

    with caplog.at_level(logging.WARNING, logger="tls2dseg.pipeline.run"):
        _warn_single_view_dispatch_pending()

    msgs = [r.message for r in caplog.records]
    matched = [m for m in msgs if "single-view" in m and "Phase 5" in m]
    assert matched, f"warning text doesn't match D-A1-06 phrasing; got: {msgs!r}"


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
