"""2D detection helpers.

Phase 4 plan 04-04 Task 3 (ENG-07):
All functions have been moved to engines.inference.shared.
This module re-exports them for backward compatibility with pipeline.run
and other callers until they are re-pointed in Task 4.
"""

from __future__ import annotations

from tls2dseg.engines.inference.shared import (
    _compute_2d_mask_features,
    _compute_2d_mask_features_worker,
    d2d_outlier_removal,
    filter_out_samples_in_2d_detections,
    merge_list_of_2d_detections,
)

__all__ = [
    "_compute_2d_mask_features",
    "_compute_2d_mask_features_worker",
    "d2d_outlier_removal",
    "filter_out_samples_in_2d_detections",
    "merge_list_of_2d_detections",
]
