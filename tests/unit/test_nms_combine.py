"""Unit tests for nms_combine_detections (preprocessing/nms_combine.py).

Covers: MODE-03 (2D-NMS combine deduplicates overlapping detections across features).
Tier: tier_a — uses only numpy + tls2dseg.types; no pchandler/torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from tls2dseg.preprocessing.nms_combine import _greedy_nms_indices, nms_combine_detections
from tls2dseg.types import Detections2D


def _make_d2d(box: list[float], class_id: int = 1, confidence: float = 0.99) -> Detections2D:
    return Detections2D(
        masks=[np.array([[0, 0]], dtype=np.int32)],
        input_boxes=np.array([box], dtype=np.float32),
        confidences=np.array([confidence], dtype=np.float32),
        class_names=["fake_object"],
        class_ids=np.array([class_id], dtype=np.int32),
        mask_labels=["fake_object 0.99"],
    )


@pytest.mark.tier_a
def test_nms_combine_suppresses_duplicate_detections() -> None:
    """Two identical boxes from different feature images -> one survives (highest confidence)."""
    box = [0.0, 0.0, 10.0, 10.0]
    d1 = _make_d2d(box, confidence=0.9)
    d2 = _make_d2d(box, confidence=0.7)

    result = nms_combine_detections([d1, d2], iou_threshold=0.80, overlap_filter_strategy="nms")

    assert len(result.input_boxes) == 1
    assert len(result.confidences) == 1
    assert abs(float(result.confidences[0]) - 0.9) < 1e-5, (
        f"Survivor should have confidence 0.9, got {result.confidences[0]}"
    )
    assert len(result.masks) == 1
    assert len(result.class_names) == 1
    assert len(result.class_ids) == 1
    assert len(result.mask_labels) == 1


@pytest.mark.tier_a
def test_nms_combine_keeps_non_overlapping_detections() -> None:
    """Two non-overlapping boxes -> both survive."""
    d1 = _make_d2d([0.0, 0.0, 5.0, 5.0], confidence=0.9)
    d2 = _make_d2d([100.0, 100.0, 110.0, 110.0], confidence=0.8)

    result = nms_combine_detections([d1, d2], iou_threshold=0.80, overlap_filter_strategy="nms")

    assert len(result.input_boxes) == 2
    assert len(result.confidences) == 2
    assert len(result.masks) == 2
    assert len(result.class_names) == 2


@pytest.mark.tier_a
def test_nms_combine_empty_list_returns_empty_detections() -> None:
    """Empty input list -> empty Detections2D (no crash)."""
    result = nms_combine_detections([], iou_threshold=0.80, overlap_filter_strategy="nms")

    assert len(result.input_boxes) == 0
    assert len(result.confidences) == 0
    assert len(result.masks) == 0
    assert len(result.class_names) == 0
    assert len(result.class_ids) == 0
    assert len(result.mask_labels) == 0


@pytest.mark.tier_a
def test_nms_combine_unknown_strategy_raises() -> None:
    """Unknown overlap_filter_strategy raises ValueError."""
    d1 = _make_d2d([0.0, 0.0, 5.0, 5.0])

    with pytest.raises(ValueError):
        nms_combine_detections([d1], iou_threshold=0.80, overlap_filter_strategy="greedymmm")


@pytest.mark.tier_a
def test_nms_combine_class_aware_keeps_different_classes() -> None:
    """class_agnostic=False: two heavily overlapping boxes of different classes both survive.

    Numpy path only (no supervision). Exercises the class-aware gate in
    _greedy_nms_indices (WR-01 fix).
    """
    # Two nearly-identical boxes, different classes
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.1, 0.1, 10.1, 10.1]], dtype=np.float32)
    confs = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([1, 2], dtype=np.int32)

    kept = _greedy_nms_indices(boxes, confs, class_ids, iou_threshold=0.50, class_agnostic=False)

    assert len(kept) == 2, f"Both different-class boxes should survive class-aware NMS, got {kept}"


@pytest.mark.tier_a
def test_nms_combine_class_agnostic_suppresses_different_classes() -> None:
    """class_agnostic=True: lower-confidence box suppressed even across classes.

    Numpy path only (no supervision). Exercises the class-agnostic path in
    _greedy_nms_indices.
    """
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.1, 0.1, 10.1, 10.1]], dtype=np.float32)
    confs = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([1, 2], dtype=np.int32)

    kept = _greedy_nms_indices(boxes, confs, class_ids, iou_threshold=0.50, class_agnostic=True)

    assert len(kept) == 1, f"class-agnostic NMS should suppress the lower-confidence box, got {kept}"
    assert 0 in kept, "Highest-confidence box (index 0) should be the survivor"


@pytest.mark.tier_a
def test_nms_combine_api_class_agnostic_flag_propagates() -> None:
    """nms_combine_detections forwards class_agnostic to the numpy path correctly.

    Two overlapping boxes with different class_ids:
    - class_agnostic=False  → both kept
    - class_agnostic=True   → only highest-confidence survives
    """
    box_a = [0.0, 0.0, 10.0, 10.0]
    box_b = [0.1, 0.1, 10.1, 10.1]
    d1 = _make_d2d(box_a, class_id=1, confidence=0.9)
    d2 = _make_d2d(box_b, class_id=2, confidence=0.8)

    result_aware = nms_combine_detections(
        [d1, d2], iou_threshold=0.50, overlap_filter_strategy="nms", class_agnostic=False
    )
    result_agnostic = nms_combine_detections(
        [d1, d2], iou_threshold=0.50, overlap_filter_strategy="nms", class_agnostic=True
    )

    assert len(result_aware.input_boxes) == 2, "Class-aware: different-class overlap should keep both"
    assert len(result_agnostic.input_boxes) == 1, "Class-agnostic: should suppress lower-confidence"
