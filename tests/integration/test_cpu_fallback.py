"""Integration tests for CPU-01 + CPU-03 — pc2img importability without cuml +
single-fire fallback WARNING.

Locks Phase 3 plan 03-03 surgical fix at
``pc2img/src/pc2img/image_generation.py``: the module no longer hard-imports
``cuml.neighbors.NearestNeighbors`` at load time, and the fallback path emits
exactly one WARNING per process.

Marked ``tier_b_light`` per D-A4-01 — these tests import ``pc2img``, which
transitively pulls ``pchandler``. ``tier_a`` is reserved for tests that have
no heavy-dep imports (D-A4-01); a single ``import pc2img.image_generation``
disqualifies the file. ``tier_b_light`` runs on opt-in (label-triggered) cloud
CI and locally; not always-on.

Pre-fix failure mode (line 9 module-level cuml import):
    ``import pc2img.image_generation`` raised ``ImportError: No module named
    'cuml'`` on any machine without RAPIDS installed, blocking every importer.

Post-fix pass condition:
    Module imports cleanly; ``_cpu_fallback_warned`` flag + helper exist;
    helper emits one WARNING on first call and is silent thereafter.

Test isolation: the module-level ``_cpu_fallback_warned`` flag is reset via
``monkeypatch.setattr`` before any warning-fire assertion (Pitfall 9 in
03-RESEARCH.md). Without the reset, an earlier test in the same process that
already triggered the fallback would leave the flag ``True`` and silently
break the single-fire assertion here.
"""

from __future__ import annotations

import importlib.util
import logging

import pytest


def _import_pc2img_image_generation_or_skip():
    """Try to import ``pc2img.image_generation``; skip on missing/incompatible heavy deps.

    The dep-repo scope guard (CLAUDE.md + [[feedback_dep_repo_scope]]) forbids
    adding test infrastructure to pc2img. pc2img's ``__init__.py`` transitively
    imports heavy deps (``PIL``, ``imageio``, ``tifffile``, ``open3d``,
    ``laspy``, ``shapely``, etc.) that may not be installed in a minimal CI
    environment, and pchandler may break against newer NumPy due to API
    contract changes (e.g. ``np.float_`` removed in NumPy 2.0).

    We distinguish:

    - ``ModuleNotFoundError: No module named 'cuml'`` → **FAIL**: the surgical
      fix at ``pc2img/image_generation.py`` is regressed; this is exactly what
      CPU-01 forbids.
    - Any other ``ImportError`` / ``ModuleNotFoundError`` / ``AttributeError``
      / ``ValueError`` raised during ``pc2img.image_generation`` import →
      **SKIP**: the test environment is missing pc2img's runtime deps or has
      version-incompatible deps; out of scope for this plan.
    """
    if importlib.util.find_spec("pc2img") is None:
        pytest.skip("pc2img not installed (editable install required for integration tests)")
    try:
        import pc2img.image_generation as ig
    except ModuleNotFoundError as exc:
        if exc.name == "cuml":
            pytest.fail(
                "CPU-01 REGRESSION: importing pc2img.image_generation raised "
                f"ModuleNotFoundError on `cuml` — the surgical fix at "
                f"pc2img/src/pc2img/image_generation.py has been reverted or "
                f"a new module-level cuml import was reintroduced. ({exc!r})"
            )
        pytest.skip(
            f"pc2img runtime dep missing ({exc.name!r}); test env is not "
            "fully provisioned for pc2img import — out of scope for CPU-01"
        )
    except (ImportError, AttributeError, ValueError) as exc:
        pytest.skip(
            f"pc2img/pchandler dep stack import failed ({type(exc).__name__}: "
            f"{exc!s}); test env has incompatible runtime deps — out of scope "
            "for CPU-01"
        )
    return ig


@pytest.mark.tier_b_light
def test_pc2img_image_generation_imports_without_cuml() -> None:
    """``import pc2img.image_generation`` succeeds even when cuml is absent (CPU-01).

    Pre-fix code at ``pc2img/src/pc2img/image_generation.py:9`` had a hard
    top-level ``from cuml.neighbors import NearestNeighbors``. On any environment
    without RAPIDS/cuml installed, the import raised ``ImportError`` before
    the module body finished loading, breaking every transitive importer
    (including the whole tls2dseg pipeline). The surgical fix moves the
    import inside the ``'knn'`` case of ``calculate_triangulation`` with a
    try/except that falls back to ``sklearn.neighbors.NearestNeighbors``.

    The test skips when pc2img is not installed (CI without editable install)
    OR when any of pc2img's heavy runtime deps (PIL, imageio, tifffile, etc.)
    are missing — those are out of scope for CPU-01 per the dep-repo scope
    guard. The key assertion is in the helper: if ``cuml`` specifically is
    the missing module, the test FAILS — that's exactly the regression
    CPU-01 forbids. Any other missing module → skip.
    """
    ig = _import_pc2img_image_generation_or_skip()  # noqa: F841 — import is the assertion


@pytest.mark.tier_b_light
def test_cpu_fallback_warned_flag_exists() -> None:
    """``_cpu_fallback_warned`` module-level flag is a bool (CPU-03 mechanism shape).

    Locks the public surface of the surgical edit: tests need this exact
    attribute name to reset the flag via ``monkeypatch.setattr`` (Pitfall 9).
    If a future refactor renames or wraps the flag (e.g. into a class), this
    test catches it before the single-fire assertion silently breaks.
    """
    ig = _import_pc2img_image_generation_or_skip()

    assert hasattr(ig, "_cpu_fallback_warned"), (
        "module-level `_cpu_fallback_warned` flag missing — surgical fix "
        "at pc2img/image_generation.py is incomplete or has been refactored"
    )
    assert isinstance(ig._cpu_fallback_warned, bool), (
        f"`_cpu_fallback_warned` must be bool; got {type(ig._cpu_fallback_warned).__name__}"
    )


@pytest.mark.tier_b_light
def test_warn_bary_knn_cpu_fallback_fires_exactly_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``_warn_bary_knn_cpu_fallback()`` emits one WARNING per process (CPU-03).

    Calls the helper twice with the module flag pre-reset. Asserts the
    ``pc2img`` logger emits exactly one record matching the WARNING contract
    (``PERFORMANCE``, ``cuml.neighbors unavailable``, ``sklearn``, and
    ``once per process``). The second call must be silent — that's the
    single-fire guarantee Plan 06 (CPU-03 surfacing) depends on so the user
    doesn't get a hundred-line stderr wall on a multi-scan run.

    Why monkeypatch the flag: per Pitfall 9 in 03-RESEARCH.md, the
    module-level flag is process-state. If an earlier import or test in the
    same process already engaged the fallback, the flag is already ``True``
    and this test would silently assert zero records. Resetting the flag
    isolates the test from suite ordering.
    """
    ig = _import_pc2img_image_generation_or_skip()

    # Reset module-level flag — Pitfall 9.
    monkeypatch.setattr(ig, "_cpu_fallback_warned", False)

    with caplog.at_level(logging.WARNING, logger="pc2img"):
        ig._warn_bary_knn_cpu_fallback()
        ig._warn_bary_knn_cpu_fallback()  # second call must be silent

    fallback_records = [r for r in caplog.records if "fallback" in r.message.lower()]
    assert len(fallback_records) == 1, (
        f"expected exactly one fallback WARNING; got {len(fallback_records)} "
        f"(messages: {[r.message for r in fallback_records]!r}) — "
        "single-fire mechanism broken; CPU-03 contract violated"
    )

    # Verify the warning message contains all four contract tokens from
    # 03-RESEARCH.md §Pattern 4 (`PERFORMANCE`, `cuml.neighbors unavailable`,
    # `sklearn`, `once per process`). Loose substring matches — exact wording
    # is fixed by the surgical fix but tokens are what the CPU-03 plan-06
    # surfacing relies on for log filtering.
    msg = fallback_records[0].message
    for token in ("PERFORMANCE", "cuml.neighbors unavailable", "sklearn", "once per process"):
        assert token in msg, (
            f"WARNING message missing token {token!r}; got: {msg!r} — "
            "CPU-03 surfacing in plan 06 relies on these tokens"
        )

    # After the two calls, the flag must be True (so a future call in this
    # process also stays silent — the contract holds beyond the test window).
    assert ig._cpu_fallback_warned is True, (
        "`_cpu_fallback_warned` should be True after the helper fires; "
        "single-fire mechanism is broken if it remains False"
    )
