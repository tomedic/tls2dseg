"""TEST-03 wrappers-only: roi_mask_xy + roi_mask_xy_rectaware tier_a tests.

Phase 4 plan 05 Task 1. Locks the contract of the pure-numpy ROI mask helpers
extracted to ``tls2dseg.preprocessing.roi`` (ENG-07, D-D-04):

* ``roi_mask_xy`` correctly includes/excludes points for known polygon shapes.
* ``roi_mask_xy_rectaware`` fast-path for axis-aligned rectangles and convex
  quadrilaterals returns the same result as the general fallback.
* Both functions handle edge cases (empty input, on-boundary points).

Marked ``tier_a`` — pure numpy only, no pchandler/pc2img/torch imports needed.
Wrappers-only rule (D-D-04): tests tls2dseg-owned functions, NOT upstream PCHandler.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.tier_a
def test_roi_mask_xy_inside_square() -> None:
    """Points clearly inside a unit-square polygon are all True."""
    from tls2dseg.preprocessing.roi import roi_mask_xy

    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pts = np.array([[0.5, 0.5], [0.25, 0.75]])
    mask = roi_mask_xy(pts, poly, include_boundary=True)
    assert mask.shape == (2,)
    assert mask.all(), "Interior points should be inside the polygon"


@pytest.mark.tier_a
def test_roi_mask_xy_outside_square() -> None:
    """Points clearly outside the unit square are all False."""
    from tls2dseg.preprocessing.roi import roi_mask_xy

    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pts = np.array([[2.0, 2.0], [-1.0, 0.5], [0.5, 1.5]])
    mask = roi_mask_xy(pts, poly, include_boundary=True)
    assert not mask.any(), "Exterior points should be outside the polygon"


@pytest.mark.tier_a
def test_roi_mask_xy_mixed_inside_outside() -> None:
    """Mixed points: returns correct per-point boolean mask."""
    from tls2dseg.preprocessing.roi import roi_mask_xy

    poly = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    pts = np.array([[1.0, 1.0], [5.0, 5.0], [0.5, 0.5]])
    mask = roi_mask_xy(pts, poly, include_boundary=True)
    assert mask[0] is np.bool_(True), "Point (1,1) should be inside [0,2]x[0,2]"
    assert mask[1] is np.bool_(False), "Point (5,5) should be outside [0,2]x[0,2]"
    assert mask[2] is np.bool_(True), "Point (0.5,0.5) should be inside [0,2]x[0,2]"


@pytest.mark.tier_a
def test_roi_mask_xy_empty_input() -> None:
    """Empty point set returns empty mask without error."""
    from tls2dseg.preprocessing.roi import roi_mask_xy

    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pts = np.empty((0, 2))
    mask = roi_mask_xy(pts, poly)
    assert mask.shape == (0,)
    assert mask.dtype == bool


@pytest.mark.tier_a
def test_roi_mask_xy_rectaware_axis_aligned() -> None:
    """Axis-aligned rectangle fast path returns same result as general path."""
    from tls2dseg.preprocessing.roi import roi_mask_xy, roi_mask_xy_rectaware

    # Axis-aligned rectangle [0,3]x[0,3]
    poly = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]])
    rng = np.random.RandomState(42)
    pts = rng.uniform(-1, 4, size=(100, 2))

    mask_rectaware = roi_mask_xy_rectaware(pts, poly, include_boundary=True)
    mask_general = roi_mask_xy(pts, poly, include_boundary=True)

    assert np.array_equal(mask_rectaware, mask_general), (
        "roi_mask_xy_rectaware and roi_mask_xy should agree on axis-aligned rectangle"
    )


@pytest.mark.tier_a
def test_roi_mask_xy_rectaware_convex_quad() -> None:
    """Convex (non-axis-aligned) quadrilateral fast path agrees with general path."""
    from tls2dseg.preprocessing.roi import roi_mask_xy, roi_mask_xy_rectaware

    # Rotated square (diamond) — convex, not axis-aligned
    poly = np.array([[1.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 1.0]])
    rng = np.random.RandomState(99)
    pts = rng.uniform(-0.5, 2.5, size=(80, 2))

    mask_rectaware = roi_mask_xy_rectaware(pts, poly, include_boundary=True)
    mask_general = roi_mask_xy(pts, poly, include_boundary=True)

    assert np.array_equal(mask_rectaware, mask_general), (
        "roi_mask_xy_rectaware and roi_mask_xy should agree on convex quadrilateral"
    )
