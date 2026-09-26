"""Box-prompted SAM2 refinement on per-box crops of a large panorama."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("tls2dseg.labeling.sam2_refine")


def load_sam2_predictor(checkpoint: str, model_config: str, device: str) -> Any:
    """Load SAM2 alone (no Grounding-DINO) as a ``SAM2ImagePredictor``."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    logger.info("Loading SAM2 checkpoint: %s", checkpoint)
    return SAM2ImagePredictor(build_sam2(model_config, str(checkpoint), device=device))


def crop_window(
    box_xyxy: np.ndarray, img_hw: tuple[int, int], margin_frac: float = 0.25, min_px: int = 64
) -> tuple[int, int, int, int]:
    """Crop ``(r0, r1, c0, c1)`` (end-exclusive) around a box, padded by a margin and clipped to the image.

    The margin is ``margin_frac`` of the box size per side, and the crop is at
    least ``min_px`` wide/high where the image allows.
    """
    h, w = img_hw
    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    pad_x = max(margin_frac * bw, (min_px - bw) / 2.0, 0.0)
    pad_y = max(margin_frac * bh, (min_px - bh) / 2.0, 0.0)
    c0 = int(np.clip(np.floor(x0 - pad_x), 0, w - 1))
    c1 = int(np.clip(np.ceil(x1 + pad_x), c0 + 1, w))
    r0 = int(np.clip(np.floor(y0 - pad_y), 0, h - 1))
    r1 = int(np.clip(np.ceil(y1 + pad_y), r0 + 1, h))
    return r0, r1, c0, c1


def refine_boxes(
    predictor: Any,
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    device: str,
) -> list[np.ndarray]:
    """Run SAM2 per box on a crop around it; return sparse (M, 2) [row, col] masks in panorama coordinates.

    Cropping lets SAM2's internal 1024 px resize zoom in on small objects instead
    of shrinking a whole panorama. Each crop is contrast-stretched on its own,
    as in the pipeline's sliced inference.
    """
    import torch

    from tls2dseg.engines.inference.shared import convert_masks_to_sparse_masks, img_1to3_channels_encoding
    from tls2dseg.grounded_sam2_utils import run_sam2_bbox_prompt_inference_in_batches

    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    img_hw = image.shape[:2]
    sparse_masks: list[np.ndarray] = []
    for box in boxes_xyxy:
        r0, r1, c0, c1 = crop_window(box, img_hw)
        crop = img_1to3_channels_encoding(
            np.asarray(image[r0:r1, c0:c1], dtype=np.float32),
            normalize="0-1",
            output_dtype="float32",
            replace_nan_with="max",
            broadcast=False,
        )
        box_local = box - np.array([c0, r0, c0, r0], dtype=np.float32)
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            predictor.set_image(crop)
            masks = run_sam2_bbox_prompt_inference_in_batches(predictor, box_local[None, :], 1, [])
        sparse = convert_masks_to_sparse_masks(masks)[0]
        sparse[:, 0] += r0
        sparse[:, 1] += c0
        sparse_masks.append(sparse)
    if device == "cuda":
        torch.cuda.empty_cache()
    return sparse_masks
