"""Intra-scan multi-feature 2D-NMS combine helper.

Operates on a list of Detections2D objects from the same scan's feature
images (which share an identical spherical projection grid) and returns a
single deduplicated Detections2D via bounding-box IoU + NMS.

Single-view mode only. Does NOT enter the multi-view path.
Uses supervision's with_nms or a pure-numpy greedy NMS fallback.
No pchandler / torch / pc2img -- tier_a safe.
"""

from __future__ import annotations

import logging

import numpy as np

from tls2dseg.types import Detections2D

logger = logging.getLogger("tls2dseg.preprocessing.nms_combine")


def nms_combine_detections(
    d2d_list: list[Detections2D],
    iou_threshold: float,
    overlap_filter_strategy: str,
) -> Detections2D:
    """Concatenate N per-feature Detections2D and apply 2D box-IoU NMS.

    Parameters
    ----------
    d2d_list :
        Per-feature detection results from one scan; each shares the same
        spherical projection grid so 2D pixel IoU is a faithful 3D-overlap proxy.
    iou_threshold :
        Box-IoU threshold above which the lower-confidence detection is suppressed.
        Reuses cfg.inference.slicing.iou_threshold (default 0.80).
    overlap_filter_strategy :
        Currently always "nms". Reserved for future strategies.

    Returns
    -------
    Detections2D
        Single deduplicated detection set (surviving detections only).

    Raises
    ------
    ValueError
        If overlap_filter_strategy is not "nms".
    """
    if overlap_filter_strategy != "nms":
        raise ValueError(
            f"Unsupported overlap_filter_strategy: {overlap_filter_strategy!r}. Only 'nms' is currently supported."
        )

    if len(d2d_list) == 0:
        return Detections2D(
            masks=[],
            input_boxes=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty(0, dtype=np.float32),
            class_names=[],
            class_ids=np.empty(0, dtype=np.int32),
            mask_labels=[],
        )

    # Concatenate fields across all N Detections2D
    all_boxes = np.concatenate([d.input_boxes for d in d2d_list], axis=0)
    all_confs = np.concatenate([d.confidences for d in d2d_list], axis=0)
    all_class_ids = np.concatenate([d.class_ids for d in d2d_list], axis=0)
    all_masks = [m for d in d2d_list for m in d.masks]
    all_class_names = [n for d in d2d_list for n in d.class_names]
    all_mask_labels = [lbl for d in d2d_list for lbl in d.mask_labels]

    n_total = len(all_boxes)
    if n_total == 0:
        return Detections2D(
            masks=[],
            input_boxes=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty(0, dtype=np.float32),
            class_names=[],
            class_ids=np.empty(0, dtype=np.int32),
            mask_labels=[],
        )

    # Apply NMS using supervision with_nms (preferred) or pure-numpy fallback
    kept_indices = _nms_indices(all_boxes, all_confs, iou_threshold)

    return Detections2D(
        masks=[all_masks[i] for i in kept_indices],
        input_boxes=all_boxes[kept_indices],
        confidences=all_confs[kept_indices],
        class_names=[all_class_names[i] for i in kept_indices],
        class_ids=all_class_ids[kept_indices],
        mask_labels=[all_mask_labels[i] for i in kept_indices],
    )


def _nms_indices(
    boxes: np.ndarray,
    confidences: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return sorted indices of detections that survive NMS.

    Tries supervision first; falls back to a pure-numpy greedy NMS.
    """
    try:
        import supervision as sv

        # Embed original indices as tracker_id so they survive filtering
        tracker = np.arange(len(boxes), dtype=np.int32)
        sv_dets = sv.Detections(xyxy=boxes, confidence=confidences, tracker_id=tracker)
        sv_filtered = sv_dets.with_nms(threshold=iou_threshold)
        kept: np.ndarray = sv_filtered.tracker_id
        return np.sort(kept)
    except ImportError:
        logger.debug("supervision not available; using pure-numpy greedy NMS fallback")
        return _greedy_nms_indices(boxes, confidences, iou_threshold)


def _greedy_nms_indices(
    boxes: np.ndarray,
    confidences: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Pure-numpy greedy NMS: sort by confidence desc, suppress IoU > threshold."""
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
        # Compute IoU with all remaining boxes
        ix1 = np.maximum(x1[idx], x1)
        iy1 = np.maximum(y1[idx], y1)
        ix2 = np.minimum(x2[idx], x2)
        iy2 = np.minimum(y2[idx], y2)
        inter_w = np.maximum(0.0, ix2 - ix1)
        inter_h = np.maximum(0.0, iy2 - iy1)
        inter = inter_w * inter_h
        union = areas[idx] + areas - inter
        iou = np.where(union > 0, inter / union, 0.0)
        # Suppress all overlapping boxes (excluding the current box itself)
        suppress_mask = iou > iou_threshold
        suppress_mask[idx] = False
        suppressed |= suppress_mask

    return np.sort(np.array(kept, dtype=np.int32))
