"""Intra-scan multi-feature 2D-NMS combine helper.

Operates on a list of Detections2D objects from the same scan's feature
images (which share an identical spherical projection grid) and returns a
single deduplicated Detections2D via bounding-box IoU + NMS.

Single-view mode only. Does NOT enter the multi-view path.
Uses supervision's with_nms or a pure-numpy greedy NMS fallback.
No pchandler / torch / pc2img -- tier_a safe.

Class-suppression semantics
---------------------------
Controlled by the ``class_agnostic`` flag (sourced from
``cfg.inference.slicing.nms_combine_class_agnostic``, default ``False``):

- ``False`` (class-aware, default): overlapping detections of *different*
  classes are kept; only same-class duplicates are suppressed.
- ``True`` (class-agnostic): any overlapping detection is suppressed
  regardless of class.

Both the supervision path and the numpy fallback honour this flag
consistently so that pipeline output does not depend on whether supervision
is installed.

IoU convention
--------------
Strict greater-than: a box pair is suppressed when ``iou > iou_threshold``
(not ``>=``). Box area is computed without the ``+1`` pixel convention.
These conventions are asserted by the tier_b path-agreement test
(tests/pipeline/test_nms_combine_supervision.py).
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
    class_agnostic: bool = False,
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
    class_agnostic :
        If False (default, class-aware): only detections of the *same* class
        suppress each other. If True (class-agnostic): suppression ignores class.
        Sourced from ``cfg.inference.slicing.nms_combine_class_agnostic``.

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
    kept_indices = _nms_indices(all_boxes, all_confs, all_class_ids, iou_threshold, class_agnostic)

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
    class_ids: np.ndarray,
    iou_threshold: float,
    class_agnostic: bool,
) -> np.ndarray:
    """Return sorted indices of detections that survive NMS.

    Tries supervision first; falls back to the pure-numpy greedy NMS on any
    supervision-side failure (ImportError if not installed, or a runtime error
    such as AssertionError from a bad call).

    Both paths honour ``class_agnostic``:
    - False (class-aware): detections of different classes do not suppress each other.
    - True  (class-agnostic): suppression ignores class.

    IoU convention: strict ``iou > iou_threshold``, no ``+1`` pixel area.
    """
    try:
        import supervision as sv

        # Embed original indices as tracker_id so they survive filtering
        tracker = np.arange(len(boxes), dtype=np.int32)
        if class_agnostic:
            sv_dets = sv.Detections(xyxy=boxes, confidence=confidences, tracker_id=tracker)
            sv_filtered = sv_dets.with_nms(threshold=iou_threshold, class_agnostic=True)
        else:
            sv_dets = sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids.astype(np.int32),
                tracker_id=tracker,
            )
            sv_filtered = sv_dets.with_nms(threshold=iou_threshold, class_agnostic=False)
        kept: np.ndarray = sv_filtered.tracker_id
        logger.debug("NMS via supervision (class_agnostic=%s): %d -> %d", class_agnostic, len(boxes), len(kept))
        return np.sort(kept)
    except (ImportError, Exception) as exc:
        if isinstance(exc, ImportError):
            logger.debug("supervision not available; using pure-numpy greedy NMS fallback")
        else:
            logger.debug("supervision NMS failed (%s); using pure-numpy greedy NMS fallback", exc)
        return _greedy_nms_indices(boxes, confidences, class_ids, iou_threshold, class_agnostic)


def _greedy_nms_indices(
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    class_agnostic: bool,
) -> np.ndarray:
    """Pure-numpy greedy NMS: sort by confidence desc, suppress IoU > threshold.

    IoU convention: strict ``iou > iou_threshold``, area without ``+1`` pixel.
    Tie-break: equal-confidence boxes are ordered by their original index
    (argsort is stable on equal values when using ``[::-1]`` on sorted order).

    Class semantics (``class_agnostic``):
    - False: a box at index ``idx`` only suppresses other boxes that share the
      same ``class_ids[idx]`` (class-aware; different-class overlaps are kept).
    - True:  a box at index ``idx`` suppresses all overlapping boxes (classic
      class-agnostic NMS).
    """
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
        # Suppress overlapping boxes (strict threshold; exclude self)
        suppress_mask = iou > iou_threshold
        suppress_mask[idx] = False
        if not class_agnostic:
            # Only suppress boxes of the same class
            suppress_mask &= class_ids == class_ids[idx]
        suppressed |= suppress_mask

    return np.sort(np.array(kept, dtype=np.int32))
