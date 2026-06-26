"""Cross-scan 3D-IoU correspondence scoring via Hungarian assignment.

Provides:
    correspondence_rate — fraction of instances that correspond between two
                          scans, matched optimally via linear_sum_assignment
                          on a 3D-IoU cost matrix.

Tier-a-importable at module level (numpy + scipy only).
compute_obb_iou_naive lazy-imports trimesh inside its body — safe here.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

from tls2dseg.engines.fusion.bboxes_iou import (
    compute_aabb_iou_vectorized,
    compute_obb_iou_naive,
)

logger = logging.getLogger("tls2dseg.tests.integration.scoring.correspondence")


def correspondence_rate(
    bboxes1: np.ndarray,
    bboxes2: np.ndarray,
    bboxes_type: str = "obb",
    iou_thr: float = 0.3,
) -> tuple[float, list[float]]:
    """Compute cross-scan correspondence rate via Hungarian IoU assignment.

    Parameters
    ----------
    bboxes1:
        (n1, 6) AABB or (n1, 10) OBB rows for scan 1.
    bboxes2:
        (n2, 6) AABB or (n2, 10) OBB rows for scan 2.
    bboxes_type:
        "obb" or "aabb".
    iou_thr:
        Per-pair IoU threshold to count a match (default 0.3, per research Q3).

    Returns
    -------
    rate : float
        matched_pairs / max(1, min(n1, n2)).
    matched_ious : list[float]
        IoU values for each matched pair (pairs with IoU >= iou_thr).
    """
    n1 = len(bboxes1)
    n2 = len(bboxes2)

    if n1 == 0 or n2 == 0:
        return 0.0, []

    # Build cross-scan pair index grid
    # Concatenate both sets so we can pass a single (N, *) array with offset indices
    pairs = np.array([(i, n1 + j) for i in range(n1) for j in range(n2)], dtype=np.int64)

    if bboxes_type == "obb":
        centers = np.vstack([bboxes1[:, :3], bboxes2[:, :3]])
        extents = np.vstack([bboxes1[:, 3:6], bboxes2[:, 3:6]])
        quats = np.vstack([bboxes1[:, 6:10], bboxes2[:, 6:10]])
        ious_flat = compute_obb_iou_naive(centers, extents, quats, pairs)
    else:
        aabb_all = np.vstack([bboxes1, bboxes2])
        ious_flat = compute_aabb_iou_vectorized(aabb_all, pairs)

    iou_matrix = ious_flat.reshape(n1, n2)

    # Hungarian assignment: maximize IoU ↔ minimize negative IoU
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched_ious = []
    for r, c in zip(row_ind, col_ind, strict=False):
        iou_val = float(iou_matrix[r, c])
        if iou_val >= iou_thr:
            matched_ious.append(iou_val)

    rate = len(matched_ious) / max(1, min(n1, n2))
    return rate, matched_ious
