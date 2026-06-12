"""Unit tests for AABB and OBB IoU helpers — TEST-06.

Phase 4 plan 03 Task 1 (TEST-06). Locks the contract of
``tls2dseg.engines.fusion.bboxes_iou``:

* ``compute_aabb_iou_vectorized`` — pure numpy; tier_a
* ``compute_obb_iou_naive`` / ``compute_obb_iou_parallel`` — trimesh; tier_b_light

Marked ``tier_a`` for AABB tests (imports ONLY numpy + tls2dseg.*).
Marked ``tier_b_light`` for OBB tests (trimesh required).
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# tier_a: AABB IoU — pure numpy, no heavy deps
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
@pytest.mark.parametrize(
    ("aabb_a", "aabb_b", "expected_iou"),
    [
        # two identical unit cubes → IoU = 1.0
        (
            np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            1.0,
        ),
        # non-overlapping cubes → IoU = 0.0
        (
            np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0]),
            0.0,
        ),
        # partial overlap: box A = [0..2]^3 (vol=8), box B = [1..3]^3 (vol=8)
        # intersection = [1..2]^3 (vol=1), union = 8+8-1 = 15 → IoU = 1/15
        (
            np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0]),
            np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),
            pytest.approx(1.0 / 15.0, abs=1e-5),
        ),
        # half-overlap in one axis only:
        # box A = [0..2, 0..2, 0..2] (vol=8)
        # box B = [1..3, 0..2, 0..2] (vol=8)
        # intersection = [1..2, 0..2, 0..2] (vol=4), union = 8+8-4 = 12 → IoU = 4/12 = 1/3
        (
            np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0]),
            np.array([1.0, 0.0, 0.0, 3.0, 2.0, 2.0]),
            pytest.approx(1.0 / 3.0, abs=1e-5),
        ),
        # touching at a face only → intersection vol = 0 → IoU = 0.0
        (
            np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 2.0, 1.0, 1.0]),
            0.0,
        ),
    ],
)
def test_aabb_iou_known_pairs(
    aabb_a: np.ndarray,
    aabb_b: np.ndarray,
    expected_iou: float,
) -> None:
    """compute_aabb_iou_vectorized produces exact IoU for known geometry pairs.

    Locks: TEST-06 AABB tier_a cases (identical=1.0, disjoint=0.0, partial fractions).
    """
    from tls2dseg.engines.fusion.bboxes_iou import compute_aabb_iou_vectorized

    aabb = np.vstack([aabb_a, aabb_b])
    pairs = np.array([[0, 1]])
    result = compute_aabb_iou_vectorized(aabb, pairs)

    assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
    assert result[0] == expected_iou


@pytest.mark.tier_a
def test_aabb_iou_multiple_pairs_vectorized() -> None:
    """compute_aabb_iou_vectorized handles multiple pairs in one call.

    Locks: vectorized shape contract — result has one entry per pair.
    """
    from tls2dseg.engines.fusion.bboxes_iou import compute_aabb_iou_vectorized

    # 3 boxes: identical (0,1), non-overlapping (0,2), overlapping (1,2)
    aabb = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0, 6.0, 6.0, 6.0],
        ]
    )
    pairs = np.array([[0, 1], [0, 2]])
    result = compute_aabb_iou_vectorized(aabb, pairs)
    assert result.shape == (2,)
    assert result[0] == pytest.approx(1.0, abs=1e-5)
    assert result[1] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# tier_b_light: OBB IoU — requires trimesh
# ---------------------------------------------------------------------------

try:
    import trimesh

    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False

# OBB boolean IoU requires a working backend: manifold3d package OR blender in PATH.
# Without a backend, trimesh.boolean.intersection raises ModuleNotFoundError and the
# function silently returns 0.0 — making IoU assertions meaningless.
try:
    import manifold3d

    _MANIFOLD_AVAILABLE = True
except ImportError:
    import shutil

    _MANIFOLD_AVAILABLE = shutil.which("blender") is not None

_OBB_BOOLEAN_AVAILABLE = _TRIMESH_AVAILABLE and _MANIFOLD_AVAILABLE


@pytest.mark.tier_b_light
@pytest.mark.skipif(not _OBB_BOOLEAN_AVAILABLE, reason="trimesh boolean engine (manifold3d or blender) not available")
def test_obb_iou_naive_identical_boxes() -> None:
    """Two identical OBBs (identity rotation) → IoU ≈ 1.0.

    Locks: TEST-06 OBB tier_b_light identical-box case.
    Requires a trimesh boolean engine (manifold3d or blender).
    """
    from tls2dseg.engines.fusion.bboxes_iou import compute_obb_iou_naive

    centers = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    extents = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    # identity quaternion [x, y, z, w] = [0, 0, 0, 1]
    quats = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    pairs = np.array([[0, 1]])

    result = compute_obb_iou_naive(centers, extents, quats, pairs)
    assert result.shape == (1,)
    assert result[0] == pytest.approx(1.0, abs=0.05), f"Expected IoU ≈ 1.0 for identical OBBs, got {result[0]}"


@pytest.mark.tier_b_light
@pytest.mark.skipif(not _OBB_BOOLEAN_AVAILABLE, reason="trimesh boolean engine (manifold3d or blender) not available")
def test_obb_iou_naive_non_overlapping_boxes() -> None:
    """Two far-apart OBBs → IoU ≈ 0.0.

    Locks: TEST-06 OBB tier_b_light disjoint-box case.
    Requires a trimesh boolean engine (manifold3d or blender).
    """
    from tls2dseg.engines.fusion.bboxes_iou import compute_obb_iou_naive

    centers = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
    extents = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    quats = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    pairs = np.array([[0, 1]])

    result = compute_obb_iou_naive(centers, extents, quats, pairs)
    assert result.shape == (1,)
    assert result[0] == pytest.approx(0.0, abs=0.01), f"Expected IoU ≈ 0.0 for non-overlapping OBBs, got {result[0]}"
