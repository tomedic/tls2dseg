"""tier_b supervision-path tests for nms_combine.

Coverage split
--------------
- tier_a (tests/unit/test_nms_combine.py): numpy fallback path + config field +
  class-aware vs class-agnostic semantics on _greedy_nms_indices. NO supervision
  import — structurally cannot cover the supervision branch.
- tier_b (this file): supervision path only. Requires supervision to be installed
  (importorskip guards the whole module). Covers:
  1. No-crash for both class_agnostic=False and True on a multi-class fixture
     (would raise AssertionError against pre-fix code — CR-08).
  2. Path agreement: _nms_indices (supervision) and _greedy_nms_indices (numpy)
     return identical kept_indices on the same fixture for both flag values (WR-06).

IoU convention asserted: strict ``iou > iou_threshold``, area without ``+1`` pixel.
"""

from __future__ import annotations

import numpy as np
import pytest

supervision = pytest.importorskip("supervision", reason="supervision not installed; skipping tier_b NMS tests")

from tls2dseg.preprocessing.nms_combine import _greedy_nms_indices, _nms_indices  # noqa: E402

# ---------------------------------------------------------------------------
# Shared multi-class fixture
# ---------------------------------------------------------------------------

# Three boxes: box 0 and box 1 overlap heavily (same spatial region, different
# class). Box 2 is spatially separate.
_BOXES = np.array(
    [
        [0.0, 0.0, 10.0, 10.0],  # class 1, conf 0.9 — high confidence
        [0.5, 0.5, 10.5, 10.5],  # class 2, conf 0.8 — overlaps box 0, diff class
        [100.0, 100.0, 110.0, 110.0],  # class 1, conf 0.7 — no overlap
    ],
    dtype=np.float32,
)
_CONFS = np.array([0.9, 0.8, 0.7], dtype=np.float32)
_CLASS_IDS = np.array([1, 2, 1], dtype=np.int32)
_IOU_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# CR-08: supervision path must not crash for either class semantics
# ---------------------------------------------------------------------------


@pytest.mark.tier_b_light
def test_supervision_path_class_aware_no_crash() -> None:
    """supervision _nms_indices does not raise with class_agnostic=False on multi-class input.

    Would raise ``AssertionError: Detections class_id must be given`` against
    the pre-fix code (CR-08).
    """
    kept = _nms_indices(_BOXES, _CONFS, _CLASS_IDS, _IOU_THRESHOLD, class_agnostic=False)
    assert kept is not None
    assert len(kept) >= 1


@pytest.mark.tier_b_light
def test_supervision_path_class_agnostic_no_crash() -> None:
    """supervision _nms_indices does not raise with class_agnostic=True on multi-class input."""
    kept = _nms_indices(_BOXES, _CONFS, _CLASS_IDS, _IOU_THRESHOLD, class_agnostic=True)
    assert kept is not None
    assert len(kept) >= 1


# ---------------------------------------------------------------------------
# WR-06: path agreement — supervision and numpy must return identical indices
# ---------------------------------------------------------------------------


@pytest.mark.tier_b_light
def test_supervision_numpy_agreement_class_aware() -> None:
    """_nms_indices and _greedy_nms_indices return identical kept_indices (class_agnostic=False).

    Pins IoU convention: strict ``iou > threshold``, no ``+1`` area.
    Different-class overlaps are kept by both paths.
    """
    sv_kept = _nms_indices(_BOXES, _CONFS, _CLASS_IDS, _IOU_THRESHOLD, class_agnostic=False)
    np_kept = _greedy_nms_indices(_BOXES, _CONFS, _CLASS_IDS, _IOU_THRESHOLD, class_agnostic=False)

    np.testing.assert_array_equal(
        sv_kept,
        np_kept,
        err_msg=(
            f"supervision and numpy NMS disagree (class_agnostic=False): "
            f"sv={sv_kept}, numpy={np_kept}. "
            "Both paths must use strict iou > threshold, no +1 area."
        ),
    )


@pytest.mark.tier_b_light
def test_supervision_numpy_agreement_class_agnostic() -> None:
    """_nms_indices and _greedy_nms_indices return identical kept_indices (class_agnostic=True).

    Pins IoU convention: strict ``iou > threshold``, no ``+1`` area.
    Cross-class overlap suppression must be consistent across both paths.
    """
    sv_kept = _nms_indices(_BOXES, _CONFS, _CLASS_IDS, _IOU_THRESHOLD, class_agnostic=True)
    np_kept = _greedy_nms_indices(_BOXES, _CONFS, _CLASS_IDS, _IOU_THRESHOLD, class_agnostic=True)

    np.testing.assert_array_equal(
        sv_kept,
        np_kept,
        err_msg=(
            f"supervision and numpy NMS disagree (class_agnostic=True): "
            f"sv={sv_kept}, numpy={np_kept}. "
            "Both paths must use strict iou > threshold, no +1 area."
        ),
    )
