"""Post-concat deduplication helpers for multi-zoom inference output.

After the multi-zoom dispatcher concatenates N+1 per-pass Detections2D,
residual duplicates must be cleaned without collapsing legitimately
co-located different-class detections (windows-in-walls).

Composable pure functions — the dispatcher (multi_zoom_dispatch.py) calls
them in order: concat → same-class IoU-NMS → optional same-class IoS →
cross-class keep_highest_confidence.

No class-agnostic NMS path exists here (D-D-05).
supervision imported lazily inside function bodies; module is tier_a-safe.
IoU convention: strict iou > threshold, no +1 area (matches nms_combine.py).
"""

from __future__ import annotations

import logging

import numpy as np

from tls2dseg.types import Detections2D

logger = logging.getLogger("tls2dseg.engines.inference.dedup")


# ---------------------------------------------------------------------------
# Concat helper
# ---------------------------------------------------------------------------


def _concat_detections(d2d_list: list[Detections2D]) -> Detections2D:
    """Concatenate a list of Detections2D into one (no NMS applied).

    Returns a valid empty Detections2D for an empty or all-empty input list.
    """
    _empty = Detections2D(
        masks=[],
        input_boxes=np.empty((0, 4), dtype=np.float32),
        confidences=np.empty(0, dtype=np.float32),
        class_names=[],
        class_ids=np.empty(0, dtype=np.int32),
        mask_labels=[],
    )

    if len(d2d_list) == 0:
        return _empty

    all_boxes = np.concatenate([d.input_boxes for d in d2d_list], axis=0)
    all_confs = np.concatenate([d.confidences for d in d2d_list], axis=0)
    all_class_ids = np.concatenate([d.class_ids for d in d2d_list], axis=0)
    all_masks = [m for d in d2d_list for m in d.masks]
    all_class_names = [n for d in d2d_list for n in d.class_names]
    all_mask_labels = [lbl for d in d2d_list for lbl in d.mask_labels]

    if len(all_boxes) == 0:
        return _empty

    return Detections2D(
        masks=all_masks,
        input_boxes=all_boxes,
        confidences=all_confs,
        class_names=all_class_names,
        class_ids=all_class_ids,
        mask_labels=all_mask_labels,
    )


# ---------------------------------------------------------------------------
# Shared IoU helpers
# ---------------------------------------------------------------------------


def _box_areas(boxes: np.ndarray) -> np.ndarray:
    """Compute (N,) area array from (N, 4) xyxy boxes. No +1 pixel convention."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    """Compute (N, N) symmetric IoU matrix for xyxy boxes."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = _box_areas(boxes)

    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter_w = np.maximum(0.0, ix2 - ix1)
    inter_h = np.maximum(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    union = areas[:, None] + areas[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# Same-class IoU-NMS (D-D-03 / D-D-05)
# ---------------------------------------------------------------------------


def _nms_indices(
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return sorted indices that survive same-class IoU-NMS.

    Tries supervision with class_agnostic=False first; falls back to
    the pure-numpy greedy loop on any failure (CR-08 fix pattern).
    """
    try:
        import supervision as sv

        tracker = np.arange(len(boxes), dtype=np.int32)
        sv_dets = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids.astype(np.int32),
            tracker_id=tracker,
        )
        sv_filtered = sv_dets.with_nms(threshold=iou_threshold, class_agnostic=False)
        kept: np.ndarray = sv_filtered.tracker_id
        logger.debug("same-class NMS via supervision: %d -> %d", len(boxes), len(kept))
        return np.sort(kept)
    except (ImportError, Exception) as exc:
        if isinstance(exc, ImportError):
            logger.debug("supervision not available; using pure-numpy greedy NMS fallback")
        else:
            logger.debug("supervision NMS failed (%s); using pure-numpy greedy NMS fallback", exc)
        return _greedy_nms_indices(boxes, confidences, class_ids, iou_threshold)


def _greedy_nms_indices(
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Pure-numpy same-class greedy NMS."""
    order = np.argsort(confidences)[::-1]
    kept = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    for idx in order:
        if suppressed[idx]:
            continue
        kept.append(int(idx))
        ix1 = np.maximum(x1[idx], x1)
        iy1 = np.maximum(y1[idx], y1)
        ix2 = np.minimum(x2[idx], x2)
        iy2 = np.minimum(y2[idx], y2)
        inter_w = np.maximum(0.0, ix2 - ix1)
        inter_h = np.maximum(0.0, iy2 - iy1)
        inter = inter_w * inter_h
        union = areas[idx] + areas - inter
        iou = np.where(union > 0, inter / union, 0.0)
        suppress_mask = iou > iou_threshold
        suppress_mask[idx] = False
        # Same-class only — no class-agnostic collapse (D-D-05)
        suppress_mask &= class_ids == class_ids[idx]
        suppressed |= suppress_mask

    return np.sort(np.array(kept, dtype=np.int32))


def dedup_same_class_nms(d2d: Detections2D, iou_threshold: float) -> Detections2D:
    """Remove same-class cross-scale duplicates via IoU-NMS (D-D-03/D-D-05).

    Different-class detections are never suppressed by this step.
    """
    n = len(d2d.input_boxes)
    if n == 0:
        return Detections2D(
            masks=[],
            input_boxes=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty(0, dtype=np.float32),
            class_names=[],
            class_ids=np.empty(0, dtype=np.int32),
            mask_labels=[],
        )

    kept = _nms_indices(d2d.input_boxes, d2d.confidences, d2d.class_ids, iou_threshold)
    return Detections2D(
        masks=[d2d.masks[i] for i in kept],
        input_boxes=d2d.input_boxes[kept],
        confidences=d2d.confidences[kept],
        class_names=[d2d.class_names[i] for i in kept],
        class_ids=d2d.class_ids[kept],
        mask_labels=[d2d.mask_labels[i] for i in kept],
    )


# ---------------------------------------------------------------------------
# Cross-class keep_highest_confidence (D-D-04 / MZ-06)
# ---------------------------------------------------------------------------


def dedup_cross_class(d2d: Detections2D, iou_threshold: float) -> Detections2D:
    """Remove lower-confidence detection in cross-class pairs with IoU > threshold.

    Uses IoU (union denominator) so nested co-detections (window-in-wall)
    with inherently low IoU survive. Only applies to DIFFERENT-class pairs.
    Same-class pairs are untouched here (handled by dedup_same_class_nms).
    """
    n = len(d2d.input_boxes)
    if n == 0:
        return Detections2D(
            masks=[],
            input_boxes=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty(0, dtype=np.float32),
            class_names=[],
            class_ids=np.empty(0, dtype=np.int32),
            mask_labels=[],
        )

    iou_mat = _pairwise_iou(d2d.input_boxes)
    suppressed = np.zeros(n, dtype=bool)
    order = np.argsort(d2d.confidences)[::-1]

    for i in range(len(order)):
        idx = order[i]
        if suppressed[idx]:
            continue
        for j in range(i + 1, len(order)):
            jdx = order[j]
            if suppressed[jdx]:
                continue
            if d2d.class_ids[idx] == d2d.class_ids[jdx]:
                # Same-class pairs handled elsewhere
                continue
            if iou_mat[idx, jdx] > iou_threshold:
                # Drop the lower-confidence one (jdx — already ordered by confidence)
                suppressed[jdx] = True
                logger.debug(
                    "cross-class dedup: dropped %s (conf=%.3f) in favour of %s (conf=%.3f)",
                    d2d.class_names[jdx],
                    d2d.confidences[jdx],
                    d2d.class_names[idx],
                    d2d.confidences[idx],
                )

    kept = np.where(~suppressed)[0].astype(np.int32)
    return Detections2D(
        masks=[d2d.masks[i] for i in kept],
        input_boxes=d2d.input_boxes[kept],
        confidences=d2d.confidences[kept],
        class_names=[d2d.class_names[i] for i in kept],
        class_ids=d2d.class_ids[kept],
        mask_labels=[d2d.mask_labels[i] for i in kept],
    )


# ---------------------------------------------------------------------------
# Optional same-class IoS lever (D-D-03 fragment-vs-whole)
# ---------------------------------------------------------------------------


def dedup_same_class_ios(d2d: Detections2D, ios_threshold: float, enabled: bool) -> Detections2D:
    """Optional same-class-only IoS lever for fragment-vs-whole leftovers (D-D-03).

    When ``enabled=False`` (default), returns ``d2d`` unchanged.
    When ``enabled=True``, applies greedy IoS suppression restricted to
    SAME-class pairs only — different-class nested pairs are never touched.

    IoS = intersection / min(area_i, area_j): a fragment fully contained in
    a whole box yields IoS ≈ 1.0 even though their IoU is low.
    """
    if not enabled:
        return d2d

    n = len(d2d.input_boxes)
    if n == 0:
        return Detections2D(
            masks=[],
            input_boxes=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty(0, dtype=np.float32),
            class_names=[],
            class_ids=np.empty(0, dtype=np.int32),
            mask_labels=[],
        )

    boxes = d2d.input_boxes
    areas = _box_areas(boxes)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    suppressed = np.zeros(n, dtype=bool)
    order = np.argsort(d2d.confidences)[::-1]

    for i in range(len(order)):
        idx = order[i]
        if suppressed[idx]:
            continue
        for j in range(i + 1, len(order)):
            jdx = order[j]
            if suppressed[jdx]:
                continue
            if d2d.class_ids[idx] != d2d.class_ids[jdx]:
                # IoS restricted to same-class pairs (D-D-03)
                continue
            inter_w = max(0.0, min(x2[idx], x2[jdx]) - max(x1[idx], x1[jdx]))
            inter_h = max(0.0, min(y2[idx], y2[jdx]) - max(y1[idx], y1[jdx]))
            inter = inter_w * inter_h
            min_area = min(areas[idx], areas[jdx])
            ios = inter / min_area if min_area > 0 else 0.0
            if ios > ios_threshold:
                suppressed[jdx] = True
                logger.debug(
                    "same-class IoS dedup: dropped %s (conf=%.3f) IoS=%.3f",
                    d2d.class_names[jdx],
                    d2d.confidences[jdx],
                    ios,
                )

    kept = np.where(~suppressed)[0].astype(np.int32)
    return Detections2D(
        masks=[d2d.masks[i] for i in kept],
        input_boxes=d2d.input_boxes[kept],
        confidences=d2d.confidences[kept],
        class_names=[d2d.class_names[i] for i in kept],
        class_ids=d2d.class_ids[kept],
        mask_labels=[d2d.mask_labels[i] for i in kept],
    )
