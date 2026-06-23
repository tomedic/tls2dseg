"""Multi-zoom dispatcher — runs N+1 detect() passes and composes dedup.

Orchestrates per-class adaptive multi-zoom inference (MZ-03/04/05):
  1. One always-on full-image pass (no resize, no tiling).
  2. N tiled passes, each Lanczos3-downscaled per ZoomPass.resize_factor.
  3. Coord remap (resized→native) after each downscaled pass (RESEARCH Pitfall 3).
  4. class_id remap (pass-local→global) after every pass (RESEARCH Pitfall 5).
  5. Concat → same-class IoU-NMS → optional same-class IoS → cross-class dedup.
  6. Populate ctx.per_class_metadata in-place (MZ-05).

Tier-A import contract: only stdlib + numpy + tls2dseg.types at module level.
Heavy deps (pyvips) are imported inside function bodies (D-A-05).
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from tls2dseg.types import Detections2D, InferenceRequest

if TYPE_CHECKING:
    from tls2dseg.engines.inference.multi_zoom_plan import ZoomPass

logger = logging.getLogger("tls2dseg.engines.inference.multi_zoom_dispatch")

# Module-level warn-once flag for single-zoom fallback (MZ-10).
_warned_single_zoom: bool = False


# ---------------------------------------------------------------------------
# Public predicate — caller (stage1, 06-07) gates on this
# ---------------------------------------------------------------------------


def is_multi_zoom_active(mz_cfg: object) -> bool:
    """Return False when multi-zoom is not configured; warn once (MZ-10)."""
    mode = getattr(mz_cfg, "mode", None)
    classes = getattr(mz_cfg, "classes", None)
    if mode == "single-zoom" or classes is None:
        _warn_single_zoom_once()
        return False
    return True


# ---------------------------------------------------------------------------
# Warn-once helper (MZ-10)
# ---------------------------------------------------------------------------


def _warn_single_zoom_once() -> None:
    """Fire WARNING exactly once per process for single-zoom fallback (MZ-10)."""
    global _warned_single_zoom
    if not _warned_single_zoom:
        logger.warning(
            "multi-zoom mode not configured (no classes block or mode=single-zoom). "
            "Using single-zoom fallback. Set mode: multi-zoom and classes: [...] "
            "for per-class adaptive inference."
        )
        _warned_single_zoom = True


# ---------------------------------------------------------------------------
# Lanczos3 downscale helper (D-A-04)
# ---------------------------------------------------------------------------


def _lanczos3_resize(image: np.ndarray, factor: float) -> np.ndarray:
    """Lanczos3 downscale a 2-D float32 image via pyvips. factor < 1.0 only.

    Raw feature images (range, intensity) may exceed [0, 1]; clip is NOT
    applied here — preserving the original dynamic range for DINO.
    """
    import pyvips  # heavy dep; inside function body (D-A-05)

    img_c = np.ascontiguousarray(image, dtype=np.float32)
    nan_max = np.nanmax(img_c)
    img_c = np.nan_to_num(img_c, nan=nan_max, copy=False)
    h, w = img_c.shape[0], img_c.shape[1]
    vips_img = pyvips.Image.new_from_memory(img_c.data, w, h, 1, format="float")
    vips_img = vips_img.resize(factor, kernel="lanczos3")
    result = np.frombuffer(vips_img.write_to_memory(), dtype=np.float32)
    return result.reshape((vips_img.height, vips_img.width))


# ---------------------------------------------------------------------------
# Remap helpers
# ---------------------------------------------------------------------------


def _remap_to_native(det: Detections2D, resize_factor: float) -> Detections2D:
    """Remap bbox/mask coordinates from resized-image space to native space.

    Remap 1 (RESEARCH Pitfall 3 / Landmine 5):
      bbox: divide xyxy in float (no rounding).
      mask row/col: divide by resize_factor as int (±1 px acceptable).
    """
    inv = 1.0 / resize_factor
    boxes_native = det.input_boxes * inv  # float, shape (N, 4)
    masks_native = []
    for m in det.masks:
        # m is (K, 2) int32 sparse row/col pairs
        masks_native.append((m * inv).astype(np.int32))
    return Detections2D(
        masks=masks_native,
        input_boxes=boxes_native.astype(np.float32),
        confidences=det.confidences,
        class_names=det.class_names,
        class_ids=det.class_ids,
        mask_labels=det.mask_labels,
    )


def _remap_class_ids(det: Detections2D, class_id_map: dict[str, int]) -> Detections2D:
    """Replace pass-local class_ids with global project-level ids (Remap 2).

    RESEARCH Pitfall 5 / Landmine 4: each pass has its own local prompt order
    → its own local {name: id} mapping. After the pass, replace with the
    global class_id_map from RunContext.

    Labels not found as exact keys are resolved via substring fallback
    (mirrors resolve_class_names in shared.py). Unmatched labels get id 0
    (background) and are not dropped — dropping is caller's responsibility.
    """
    global_ids = []
    valid_keys = list(class_id_map.keys())
    for name in det.class_names:
        if name in class_id_map:
            global_ids.append(class_id_map[name])
        else:
            # substring fallback — same logic as resolve_class_names
            match = next((k for k in valid_keys if k in name), None)
            if match is not None:
                global_ids.append(class_id_map[match])
            else:
                logger.debug("class name %r not in class_id_map; assigning id 0", name)
                global_ids.append(0)

    return Detections2D(
        masks=det.masks,
        input_boxes=det.input_boxes,
        confidences=det.confidences,
        class_names=det.class_names,
        class_ids=np.array(global_ids, dtype=np.int32),
        mask_labels=det.mask_labels,
    )


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------


def run_multi_zoom(
    image_native: np.ndarray,
    inference_engine: object,
    base_request: InferenceRequest,
    zoom_passes: list[ZoomPass],
    ctx: object,
    *,
    mz_cfg: object,
    on_pass: Callable[[int, ZoomPass, np.ndarray, Detections2D], None] | None = None,
) -> Detections2D:
    """Run N+1 detect() passes, dedup, populate metadata, return Detections2D.

    Parameters
    ----------
    image_native :
        Full-resolution spherical feature image (2-D float32, H x W).
    inference_engine :
        Object conforming to InferenceEngine.detect(image, *, request) (D-C-02).
    base_request :
        InferenceRequest built by the caller with global inference settings.
        Per-pass fields (text_prompt, slicing_enabled, etc.) are replaced via
        dataclasses.replace — base_request is never mutated.
    zoom_passes :
        Ordered list[ZoomPass] from compute_zoom_passes (06-03). First entry is
        always the full-image pass (needs_tiling=False, resize_factor=1.0).
    ctx :
        RunContext supplying class_id_map and per_class_metadata (mutable dict).
    mz_cfg :
        MultiZoomConfig sub-block for IoS/threshold parameters.
    on_pass :
        Optional observer called after each detect() and BEFORE coordinate
        remap. Receives (pass_index, zoom_pass, image_input, det_i) where
        image_input is the resized image fed to the model and det_i contains
        detections in that resized coordinate space. Exceptions inside the
        callback are caught and logged at DEBUG level so they never affect
        inference results.
    """
    import gc

    from tls2dseg.engines.inference.dedup import (
        _concat_detections,
        dedup_cross_class,
        dedup_same_class_ios,
        dedup_same_class_nms,
    )

    class_id_map: dict[str, int] = getattr(ctx, "class_id_map", {})
    per_class_metadata: dict[str, object] = getattr(ctx, "per_class_metadata", {})

    all_detections: list[Detections2D] = []

    for pass_index, zoom_pass in enumerate(zoom_passes):
        # Resize: only when resize_factor < 1.0 (full-image pass stays native)
        if zoom_pass.resize_factor < 1.0:
            image_input = _lanczos3_resize(image_native, zoom_pass.resize_factor)
            logger.debug(
                "pass (resize=%.3f) input shape: %s -> %s",
                zoom_pass.resize_factor,
                image_native.shape,
                image_input.shape,
            )
        else:
            image_input = image_native

        # Build per-pass InferenceRequest (add-only fields per D-C-02)
        if zoom_pass.needs_tiling:
            tile_wh = (zoom_pass.tile_size_px, zoom_pass.tile_size_px)
            overlap_wh = (zoom_pass.overlap_px, zoom_pass.overlap_px)
        else:
            tile_wh = (0, 0)
            overlap_wh = (0, 0)

        per_pass_request = dataclasses.replace(
            base_request,
            text_prompt=zoom_pass.text_prompt,
            slicing_enabled=zoom_pass.needs_tiling,
            slice_width_height=tile_wh,
            overlap_width_height=overlap_wh,
            resize_factor=zoom_pass.resize_factor,
            is_full_image_pass=not zoom_pass.needs_tiling,
            pass_class_names=zoom_pass.class_names,
        )

        det_i: Detections2D = inference_engine.detect(image_input, request=per_pass_request)
        logger.debug(
            "pass (tiled=%s, resize=%.3f, classes=%s): %d detections",
            zoom_pass.needs_tiling,
            zoom_pass.resize_factor,
            zoom_pass.class_names,
            len(det_i.input_boxes),
        )

        if on_pass is not None:
            try:
                on_pass(pass_index, zoom_pass, image_input, det_i)
            except Exception:
                logger.debug("on_pass observer raised; ignoring", exc_info=True)

        # Remap 1: resized→native coordinates
        if zoom_pass.resize_factor < 1.0:
            det_i = _remap_to_native(det_i, zoom_pass.resize_factor)

        # Remap 2: pass-local class_ids → global project-level ids
        det_i = _remap_class_ids(det_i, class_id_map)

        all_detections.append(det_i)

        # VRAM hygiene between passes (T-06-10 / RESEARCH Pitfall 4)
        gc.collect()
        _torch = sys.modules.get("torch")
        if _torch is not None and _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    # Concatenate all passes
    combined = _concat_detections(all_detections)

    # Dedup composition (D-D-03/04): same-class NMS → optional IoS → cross-class
    combined = dedup_same_class_nms(combined, iou_threshold=base_request.iou_threshold)
    combined = dedup_same_class_ios(
        combined,
        ios_threshold=getattr(mz_cfg, "ios_threshold", 0.8),
        enabled=getattr(mz_cfg, "ios_enabled", False),
    )
    combined = dedup_cross_class(
        combined,
        iou_threshold=getattr(mz_cfg, "cross_class_iou_threshold", 0.7),
    )

    # Populate per-class metadata in-place (MZ-05)
    _populate_per_class_metadata(zoom_passes, per_class_metadata)

    return combined


def _populate_per_class_metadata(
    zoom_passes: list[ZoomPass],
    per_class_metadata: dict[str, object],
) -> None:
    """Fill per_class_metadata in-place from the resolved zoom passes.

    Called once per feature per scan. Each class is keyed by name; the value
    is a ClassMetadata TypedDict. Logged at INFO level once after all passes.
    """
    from tls2dseg.types import ClassMetadata

    for zoom_pass in zoom_passes:
        if not zoom_pass.needs_tiling:
            # Full-image pass — attributes are shared across all classes
            for class_name in zoom_pass.class_names:
                per_class_metadata[class_name] = ClassMetadata(
                    resize_factor=zoom_pass.resize_factor,
                    was_tiled=False,
                    tile_size_px=None,
                    grouped_with=tuple(n for n in zoom_pass.class_names if n != class_name),
                )
        else:
            for class_name in zoom_pass.class_names:
                per_class_metadata[class_name] = ClassMetadata(
                    resize_factor=zoom_pass.resize_factor,
                    was_tiled=True,
                    tile_size_px=zoom_pass.tile_size_px,
                    grouped_with=tuple(n for n in zoom_pass.class_names if n != class_name),
                )

    logger.info(
        "per-class zoom assignments: %s",
        {
            k: {
                "resize_factor": v["resize_factor"],
                "was_tiled": v["was_tiled"],
                "tile_size_px": v["tile_size_px"],
            }
            for k, v in per_class_metadata.items()
        },
    )
