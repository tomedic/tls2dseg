"""Grounded SAM2 utility functions.

Phase 4 plan 04-04 Task 3 (ENG-07):
Shared helpers have been moved to engines.inference.shared.
This module re-exports them for backward compatibility with
grounded_sam2.py and other callers.

The only function that remains here is run_sam2_bbox_prompt_inference_in_batches,
which is the direct sam2-package seam and belongs with the legacy engine.
"""

from __future__ import annotations

from tls2dseg.engines.inference.shared import (
    check_if_img_slice_complete,
    check_if_img_slice_empty,
    convert_masks_to_sparse_masks,
    mask_to_rle,
    parse_gdino_results,
    post_process_gdino_results,
    remove_detections_touching_image_edges,
    remove_too_large_detections,
    resolve_class_names,
    return_empty_detections,
    rle_to_mask,
)

__all__ = [
    "check_if_img_slice_complete",
    "check_if_img_slice_empty",
    "convert_masks_to_sparse_masks",
    "mask_to_rle",
    "parse_gdino_results",
    "post_process_gdino_results",
    "remove_detections_touching_image_edges",
    "remove_too_large_detections",
    "resolve_class_names",
    "return_empty_detections",
    "rle_to_mask",
    "run_sam2_bbox_prompt_inference_in_batches",
]


def run_sam2_bbox_prompt_inference_in_batches(sam2_predictor, input_boxes, sam_box_prompt_batch_size, masks) -> list:
    """Run SAM2 bbox-prompt inference in batches.

    This function stays here (not in shared.py) because it directly calls the
    sam2 package API — it is the seam between shared post-processing and the
    concrete GroundedSAM2Engine (direct sam2 install, D-C-01).
    """
    for batch_i in range(0, len(input_boxes), sam_box_prompt_batch_size):
        # Get batch
        batch_boxes = input_boxes[batch_i : batch_i + sam_box_prompt_batch_size]
        # Run SAM2
        masks_i, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=batch_boxes,
            multimask_output=False,
        )
        # Squeeze out unnecessary dimensions
        if masks_i.ndim == 4:
            masks_i = masks_i.squeeze(1)  # convert the shape to (n, H, W)
        # Append to list
        masks.append(masks_i)

    return masks
