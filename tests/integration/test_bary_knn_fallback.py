"""Integration test for CPU-02 — bary_knn rasterization round-trip via sklearn
fallback when cuml is absent.

Locks Phase 3 plan 03-03 surgical fix at
``pc2img/src/pc2img/image_generation.py``: the ``'knn'`` case of
``calculate_triangulation`` (called from the ``'bary_knn'`` rasterization
method at lines ~247-252) falls back to ``sklearn.neighbors.NearestNeighbors``
on ``ImportError`` from cuml and produces non-empty, well-shaped output.

Marked ``tier_b_light`` per D-A4-01 — imports ``pc2img.image_generation``
which transitively pulls heavy deps (``pchandler``, etc.). ``tier_a`` is
reserved for tests with no heavy-dep imports.

Pre-fix failure mode:
    Could not even reach the bary_knn code path — module load failed on
    ``from cuml.neighbors import NearestNeighbors`` (line 9).

Post-fix pass condition:
    Calling ``ImageGenerator.calculate_triangulation(points, xi, method='knn')``
    on a synthetic point cloud returns a 3-tuple of equal-length non-empty
    arrays via the sklearn path, and the module-level ``_cpu_fallback_warned``
    flag transitions ``False -> True`` on first call.

The test uses ``ImageGenerator.calculate_triangulation`` as the public surface
(``@staticmethod``, lines 54-91 of image_generation.py) rather than the
higher-level ``project_and_rasterize`` flow — the latter requires a
``PointCloudData`` instance and full FoV/scalar-field setup, which is
out-of-scope heaviness for an integration test focused on the bary_knn
NN-lookup itself.
"""

from __future__ import annotations

import importlib.util

import numpy as np
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
            "fully provisioned for pc2img import — out of scope for CPU-02"
        )
    except (ImportError, AttributeError, ValueError) as exc:
        pytest.skip(
            f"pc2img/pchandler dep stack import failed ({type(exc).__name__}: "
            f"{exc!s}); test env has incompatible runtime deps — out of scope "
            "for CPU-02"
        )
    return ig


@pytest.mark.tier_b_light
def test_calculate_triangulation_knn_returns_valid_simplices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``calculate_triangulation(..., method='knn')`` returns 3 equal-length non-empty arrays (CPU-02).

    Constructs a small deterministic synthetic 2D point set and query set,
    runs the ``'knn'`` case of ``calculate_triangulation`` (which is the
    NN-lookup that underlies ``bary_knn`` rasterization), and asserts the
    returned ``simplices`` tuple is well-shaped — three arrays of length
    ``len(xi)``, each a ``(len(xi), 2)`` view into ``points``.

    Verifies the sklearn fallback actually executed (not some other branch)
    by checking the module-level ``_cpu_fallback_warned`` flag flipped to
    ``True`` after the call. The flag is reset via monkeypatch first so the
    transition is unambiguous regardless of test-suite ordering (Pitfall 9
    in 03-RESEARCH.md).

    Skips when:
        - pc2img is not installed (CI without editable install)
        - cuml is present (this test only verifies the sklearn fallback path;
          if cuml is present, the helper is never called and the flag never
          flips — a different test would be needed for the cuml path)
    """
    if importlib.util.find_spec("cuml") is not None:
        pytest.skip(
            "cuml is present in this environment — the sklearn fallback path "
            "is never exercised; rerun on a cuml-absent env to lock CPU-02"
        )

    ig = _import_pc2img_image_generation_or_skip()

    # Reset the single-fire flag so we can observe the transition.
    monkeypatch.setattr(ig, "_cpu_fallback_warned", False)

    # Deterministic synthetic 2D point cloud + query points.
    # 50 random points in [0, 1]^2; 30 random query points in the same range.
    rng = np.random.RandomState(seed=42)
    points = rng.rand(50, 2).astype(np.float32)
    xi = rng.rand(30, 2).astype(np.float32)

    simplices, indices, distances = ig.ImageGenerator.calculate_triangulation(points, xi, method="knn")

    # Shape contract: simplices is a 3-tuple of (m, 2) arrays where m = len(xi).
    assert isinstance(simplices, tuple), f"expected simplices to be a tuple; got {type(simplices).__name__}"
    assert len(simplices) == 3, f"expected 3-tuple of simplex vertex arrays; got len {len(simplices)}"
    for k, v in enumerate(simplices):
        assert isinstance(v, np.ndarray), f"simplex[{k}] is not ndarray; got {type(v).__name__}"
        assert v.shape == (len(xi), 2), (
            f"simplex[{k}] has shape {v.shape}; expected ({len(xi)}, 2) — indicates a bary_knn output shape regression"
        )
        assert v.size > 0, f"simplex[{k}] is empty — sklearn knn returned no neighbors"

    # indices: (m, 3) integer array — three nearest neighbors per query.
    assert indices.shape == (len(xi), 3), f"expected indices shape ({len(xi)}, 3); got {indices.shape}"

    # distances: (m, 3) float array — non-negative, finite.
    assert distances.shape == (len(xi), 3), f"expected distances shape ({len(xi)}, 3); got {distances.shape}"
    assert np.all(distances >= 0), "distances must be non-negative"
    assert np.all(np.isfinite(distances)), "distances must be finite (sklearn path produces real values)"

    # Sklearn-fallback executed: the single-fire flag must have flipped.
    assert ig._cpu_fallback_warned is True, (
        "`_cpu_fallback_warned` did not flip to True after calculate_triangulation "
        "with method='knn' — sklearn fallback path did NOT execute; either cuml is "
        "secretly available or the surgical fix's try/except branch is broken"
    )
