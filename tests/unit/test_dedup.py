"""Unit tests for engines/inference/dedup.py.

Covers: MZ-04 (cross-scale dedup), MZ-06 (cross-class keep_highest_confidence).
Tier: tier_a — stdlib + numpy + tls2dseg.types only; no pchandler/torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from tls2dseg.engines.inference.dedup import (
    _concat_detections,
    dedup_cross_class,
    dedup_same_class_ios,
    dedup_same_class_nms,
)
from tls2dseg.types import Detections2D

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_d2d(
    box: list[float],
    class_id: int = 1,
    confidence: float = 0.99,
    class_name: str = "fake_object",
) -> Detections2D:
    return Detections2D(
        masks=[np.array([[0, 0]], dtype=np.int32)],
        input_boxes=np.array([box], dtype=np.float32),
        confidences=np.array([confidence], dtype=np.float32),
        class_names=[class_name],
        class_ids=np.array([class_id], dtype=np.int32),
        mask_labels=[f"{class_name} {confidence:.2f}"],
    )


def _empty_d2d() -> Detections2D:
    return Detections2D(
        masks=[],
        input_boxes=np.empty((0, 4), dtype=np.float32),
        confidences=np.empty(0, dtype=np.float32),
        class_names=[],
        class_ids=np.empty(0, dtype=np.int32),
        mask_labels=[],
    )


# ---------------------------------------------------------------------------
# Task 1: concat + same-class IoU-NMS
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_concat_count_fields_extend() -> None:
    """Concatenating N1- and N2-detection sets yields N1+N2 entries."""
    d1 = _make_d2d([0.0, 0.0, 5.0, 5.0], class_id=1, confidence=0.9)
    d2 = _make_d2d([100.0, 100.0, 110.0, 110.0], class_id=1, confidence=0.8)
    d3 = _make_d2d([200.0, 200.0, 210.0, 210.0], class_id=2, confidence=0.7)

    result = _concat_detections([d1, d2, d3])

    assert len(result.input_boxes) == 3
    assert len(result.masks) == 3
    assert len(result.class_names) == 3
    assert len(result.mask_labels) == 3
    assert result.input_boxes.shape == (3, 4)
    assert result.confidences.shape == (3,)
    assert result.class_ids.shape == (3,)


@pytest.mark.tier_a
def test_concat_no_class_agnostic_collapse() -> None:
    """Different-class boxes with high IoU are BOTH kept by same-class NMS (D-D-05)."""
    # Two nearly-identical boxes, different classes
    box = [0.0, 0.0, 10.0, 10.0]
    d_wall = _make_d2d(box, class_id=1, confidence=0.9, class_name="wall")
    d_window = _make_d2d(box, class_id=2, confidence=0.8, class_name="window")

    combined = _concat_detections([d_wall, d_window])
    result = dedup_same_class_nms(combined, iou_threshold=0.5)

    assert len(result.input_boxes) == 2, (
        f"Different-class high-IoU pair must both survive same-class NMS, got {len(result.input_boxes)}"
    )


@pytest.mark.tier_a
def test_same_class_nms_collapses_duplicates() -> None:
    """Two same-class boxes at high IoU -> only higher-confidence survives."""
    box_a = [0.0, 0.0, 10.0, 10.0]
    box_b = [0.1, 0.1, 10.1, 10.1]
    d_hi = _make_d2d(box_a, class_id=1, confidence=0.9, class_name="tree")
    d_lo = _make_d2d(box_b, class_id=1, confidence=0.5, class_name="tree")

    combined = _concat_detections([d_hi, d_lo])
    result = dedup_same_class_nms(combined, iou_threshold=0.5)

    assert len(result.input_boxes) == 1
    assert abs(float(result.confidences[0]) - 0.9) < 1e-5, (
        f"Survivor should be the high-confidence box, got {result.confidences[0]}"
    )


@pytest.mark.tier_a
def test_empty_input_returns_valid_empty_detections() -> None:
    """Empty list / zero-detection input returns valid empty Detections2D."""
    result_empty_list = _concat_detections([])
    assert len(result_empty_list.input_boxes) == 0
    assert result_empty_list.input_boxes.shape == (0, 4)
    assert result_empty_list.input_boxes.dtype == np.float32
    assert result_empty_list.confidences.dtype == np.float32
    assert result_empty_list.class_ids.dtype == np.int32

    result_empty_d2d = _concat_detections([_empty_d2d()])
    assert len(result_empty_d2d.input_boxes) == 0
    assert result_empty_d2d.input_boxes.shape == (0, 4)

    result_nms = dedup_same_class_nms(_empty_d2d(), iou_threshold=0.5)
    assert len(result_nms.input_boxes) == 0
    assert result_nms.input_boxes.dtype == np.float32


# ---------------------------------------------------------------------------
# Task 2: cross-class keep_highest_confidence + optional same-class IoS
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_cross_class_dedup_keeps_higher_confidence() -> None:
    """Two different-class boxes with IoU > threshold -> only higher-confidence survives."""
    box = [0.0, 0.0, 10.0, 10.0]
    d_hi = _make_d2d(box, class_id=1, confidence=0.9, class_name="wall")
    d_lo = _make_d2d(box, class_id=2, confidence=0.6, class_name="door")

    combined = _concat_detections([d_hi, d_lo])
    result = dedup_cross_class(combined, iou_threshold=0.7)

    assert len(result.input_boxes) == 1, (
        f"High-IoU cross-class pair: lower-confidence must be dropped, got {len(result.input_boxes)}"
    )
    assert abs(float(result.confidences[0]) - 0.9) < 1e-5


@pytest.mark.tier_a
def test_nested_co_detection_survives_cross_class() -> None:
    """Window nested inside wall (low IoU due to size difference) — both survive D-D-04."""
    # wall: large box; window: small box nested inside
    wall_box = [0.0, 0.0, 100.0, 100.0]
    window_box = [10.0, 10.0, 30.0, 30.0]
    d_wall = _make_d2d(wall_box, class_id=1, confidence=0.9, class_name="wall")
    d_window = _make_d2d(window_box, class_id=2, confidence=0.85, class_name="window")

    # IoU = intersection / union
    # intersection = 20*20 = 400; union = 100*100 + 20*20 - 400 = 10000 - 400 + 400 = 10000
    # IoU ≈ 0.04 — well below the 0.7 threshold
    combined = _concat_detections([d_wall, d_window])
    result = dedup_cross_class(combined, iou_threshold=0.7)

    assert len(result.input_boxes) == 2, (
        f"Nested co-detections (low IoU) must both survive cross-class dedup, got {len(result.input_boxes)}"
    )


@pytest.mark.tier_a
def test_ios_same_class_only_collapses_fragment() -> None:
    """IoS=True: same-class fragment-vs-whole collapses; different-class nested pair does NOT."""
    # Same-class: small box (fragment) inside large box (whole) — both class_id=1
    whole_box = [0.0, 0.0, 100.0, 100.0]
    fragment_box = [10.0, 10.0, 30.0, 30.0]
    d_whole = _make_d2d(whole_box, class_id=1, confidence=0.9, class_name="tree")
    d_fragment = _make_d2d(fragment_box, class_id=1, confidence=0.7, class_name="tree")

    combined_same = _concat_detections([d_whole, d_fragment])
    result_same = dedup_same_class_ios(combined_same, ios_threshold=0.8, enabled=True)

    assert len(result_same.input_boxes) == 1, (
        f"Same-class fragment-vs-whole: fragment must be collapsed by IoS, got {len(result_same.input_boxes)}"
    )
    assert abs(float(result_same.confidences[0]) - 0.9) < 1e-5

    # Different-class: same geometry but class_id differs — must NOT collapse
    d_wall = _make_d2d(whole_box, class_id=1, confidence=0.9, class_name="wall")
    d_window = _make_d2d(fragment_box, class_id=2, confidence=0.7, class_name="window")

    combined_diff = _concat_detections([d_wall, d_window])
    result_diff = dedup_same_class_ios(combined_diff, ios_threshold=0.8, enabled=True)

    assert len(result_diff.input_boxes) == 2, (
        f"Different-class nested pair must NOT be collapsed by IoS, got {len(result_diff.input_boxes)}"
    )


@pytest.mark.tier_a
def test_ios_disabled_is_noop() -> None:
    """ios_enabled=False: IoS step is a no-op regardless of overlap."""
    whole_box = [0.0, 0.0, 100.0, 100.0]
    fragment_box = [10.0, 10.0, 30.0, 30.0]
    d_whole = _make_d2d(whole_box, class_id=1, confidence=0.9, class_name="tree")
    d_fragment = _make_d2d(fragment_box, class_id=1, confidence=0.7, class_name="tree")

    combined = _concat_detections([d_whole, d_fragment])
    result = dedup_same_class_ios(combined, ios_threshold=0.8, enabled=False)

    assert len(result.input_boxes) == 2, (
        f"disabled IoS must be a no-op (both detections survive), got {len(result.input_boxes)}"
    )
