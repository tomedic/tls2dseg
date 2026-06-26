"""Unit tests for the scoring module (containment + correspondence + report).

Covers: TEST-16 tier_a — numpy+scipy only, no GPU/pchandler/torch.
All geometry tested on synthetic boxes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.tier_a

# ---------------------------------------------------------------------------
# OBB availability guard (for the OBB-path correspondence test)
# ---------------------------------------------------------------------------

try:
    import trimesh  # type: ignore[import-untyped]

    try:
        import manifold3d  # type: ignore[import-untyped]

        _OBB_BOOLEAN_AVAILABLE = True
        _ = (trimesh, manifold3d)  # keep references
    except ImportError:
        # trimesh present but no manifold3d/blender → boolean ops return 0
        _OBB_BOOLEAN_AVAILABLE = False
        _ = trimesh  # keep reference
except ImportError:
    _OBB_BOOLEAN_AVAILABLE = False


# ---------------------------------------------------------------------------
# point_in_obb
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_point_in_obb_center_is_inside() -> None:
    """A point at the OBB center is inside."""
    from tests.integration.scoring.containment import point_in_obb

    center = np.array([1.0, 2.0, 3.0])
    extent = np.array([2.0, 2.0, 2.0])
    quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity rotation
    assert point_in_obb(center, center, extent, quat) is True


@pytest.mark.tier_a
def test_point_in_obb_outside_face() -> None:
    """A point just outside one face of the OBB is outside."""
    from tests.integration.scoring.containment import point_in_obb

    center = np.array([0.0, 0.0, 0.0])
    extent = np.array([2.0, 2.0, 2.0])
    quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity
    # Just outside the +x face (center ± 1.0; put point at 1.01)
    p_outside = np.array([1.01, 0.0, 0.0])
    assert point_in_obb(p_outside, center, extent, quat) is False


@pytest.mark.tier_a
def test_point_in_obb_rotation_aware() -> None:
    """
    Rotation-awareness: a point inside the axis-aligned AABB of an OBB but
    outside the rotated OBB returns False from point_in_obb.

    Setup: unit cube centered at origin, rotated 45° around Z.
    The AABB of that rotated cube extends to ±sqrt(2)/2 ≈ ±0.707 in X and Y.
    A point at (0.6, 0.6, 0.0) is inside the AABB but outside the rotated box
    (the diagonal corner of the rotated square only reaches 0.5 along each axis).
    """
    from tests.integration.scoring.containment import point_in_aabb, point_in_obb

    center = np.array([0.0, 0.0, 0.0])
    extent = np.array([1.0, 1.0, 1.0])
    # 45° around Z: quat = [0, 0, sin(22.5°), cos(22.5°)]
    angle = math.pi / 4
    quat = np.array([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])

    # The rotated box's projection on X is ±sqrt(2)/2 ≈ 0.707
    # so (0.6, 0.6, 0) is INSIDE the AABB but OUTSIDE the rotated OBB
    p = np.array([0.6, 0.6, 0.0])

    aabb_row = np.array([-0.707, -0.707, -0.5, 0.707, 0.707, 0.5])
    assert point_in_aabb(p, aabb_row) is True
    assert point_in_obb(p, center, extent, quat) is False


# ---------------------------------------------------------------------------
# point_in_aabb
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_point_in_aabb_inside() -> None:
    """A point strictly inside the AABB returns True."""
    from tests.integration.scoring.containment import point_in_aabb

    aabb = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
    p = np.array([1.0, 1.0, 1.0])
    assert point_in_aabb(p, aabb) is True


@pytest.mark.tier_a
def test_point_in_aabb_outside() -> None:
    """A point outside the AABB returns False."""
    from tests.integration.scoring.containment import point_in_aabb

    aabb = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    p = np.array([2.0, 0.5, 0.5])
    assert point_in_aabb(p, aabb) is False


# ---------------------------------------------------------------------------
# ref_recall
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_ref_recall_perfect_aabb() -> None:
    """ref_recall returns 1.0 when every reference point is inside a matching detection."""
    from tests.integration.scoring.containment import ref_recall

    # Two reference points of class "plant", both inside the big box
    ref_xyz = np.array([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]])
    ref_cls = np.array(["plant", "plant"])

    # One detection: a big AABB covering [0..1]^3, class "plant"
    det_cls = np.array(["plant"])
    det_bboxes = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])

    rate = ref_recall(ref_xyz, ref_cls, det_cls, det_bboxes, "aabb")
    assert rate == pytest.approx(1.0)


@pytest.mark.tier_a
def test_ref_recall_zero_aabb() -> None:
    """ref_recall returns 0.0 when no reference point is inside any matching detection."""
    from tests.integration.scoring.containment import ref_recall

    # Reference point far outside the detection box
    ref_xyz = np.array([[5.0, 5.0, 5.0]])
    ref_cls = np.array(["plant"])

    det_cls = np.array(["plant"])
    det_bboxes = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])

    rate = ref_recall(ref_xyz, ref_cls, det_cls, det_bboxes, "aabb")
    assert rate == pytest.approx(0.0)


@pytest.mark.tier_a
def test_ref_recall_class_mismatch() -> None:
    """ref_recall only matches detections with the same class as the reference point."""
    from tests.integration.scoring.containment import ref_recall

    ref_xyz = np.array([[0.5, 0.5, 0.5]])
    ref_cls = np.array(["plant"])

    # Detection is spatially enclosing but wrong class
    det_cls = np.array(["cabinet"])
    det_bboxes = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])

    rate = ref_recall(ref_xyz, ref_cls, det_cls, det_bboxes, "aabb")
    assert rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# correspondence_rate (using AABB inputs to stay trimesh-free in tier_a)
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_correspondence_rate_identical_aabb() -> None:
    """Identical box sets produce correspondence_rate = 1.0."""
    from tests.integration.scoring.correspondence import correspondence_rate

    # Two identical AABB boxes
    aabb = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
        ]
    )
    rate, _matched_ious = correspondence_rate(aabb, aabb, bboxes_type="aabb")
    assert rate == pytest.approx(1.0)


@pytest.mark.tier_a
def test_correspondence_rate_disjoint_aabb() -> None:
    """Disjoint box sets produce correspondence_rate = 0.0."""
    from tests.integration.scoring.correspondence import correspondence_rate

    aabb1 = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    aabb2 = np.array([[10.0, 10.0, 10.0, 11.0, 11.0, 11.0]])
    rate, _matched_ious = correspondence_rate(aabb1, aabb2, bboxes_type="aabb")
    assert rate == pytest.approx(0.0)


@pytest.mark.tier_a
def test_correspondence_rate_returns_matched_ious() -> None:
    """correspondence_rate returns matched IoU values alongside the rate."""
    from tests.integration.scoring.correspondence import correspondence_rate

    aabb = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    rate, matched_ious = correspondence_rate(aabb, aabb, bboxes_type="aabb")
    assert rate == pytest.approx(1.0)
    assert len(matched_ious) == 1
    assert matched_ious[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# write_report round-trip
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_write_report_round_trip(tmp_path: Path) -> None:
    """write_report produces JSON + markdown; JSON contains expected keys."""
    from tests.integration.scoring.report import write_report

    metrics = {
        "iou_thr": 0.3,
        "datasets": {
            "office_small": {
                "recall": 0.85,
                "correspondence_rate": 0.75,
                "fp_rate": 0.10,
                "consistency_only": {"cabinet": {"detected": 2, "consistent": True}},
            }
        },
    }
    json_path, md_path = write_report(metrics, tmp_path)

    assert json_path.exists(), f"JSON report not found: {json_path}"
    assert md_path.exists(), f"Markdown report not found: {md_path}"

    data = json.loads(json_path.read_text())
    assert "iou_thr" in data
    assert "datasets" in data
    assert "office_small" in data["datasets"]
    ds = data["datasets"]["office_small"]
    assert "recall" in ds
    assert "correspondence_rate" in ds
    assert "fp_rate" in ds
