"""Edge weight helpers — extracted verbatim from graph_clustering.py lines 137-235.

Phase 4 plan 03 Task 1 (ENG-07). Provides edge weight computation for the
fusion pipeline:

* ``compute_supporter_counts`` — shared neighbour counts for edge pairs
* ``get_edge_weights`` — IoU and/or supporter counts per pair
* ``count_significant_overlaps`` — per-detection significant overlap counts

Import-cycle rule (RESEARCH Pitfall 2): this module imports ONLY numpy,
typing, and tls2dseg.engines.fusion.bboxes_iou. It MUST NOT import
from engines/__init__.py or preprocessing/.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from tls2dseg.engines.fusion.bboxes_iou import compute_aabb_iou_vectorized, compute_obb_iou_parallel

if TYPE_CHECKING:
    from tls2dseg.detections_3d import Detections3D

logger = logging.getLogger("tls2dseg.engines.fusion.edge_weights")


def compute_supporter_counts(pairs: np.ndarray, bbox_overlap: np.ndarray, iou_threshold: float = 0.3) -> np.ndarray:
    M = pairs.shape[0]
    neigh: dict[Any, set[Any]] = {i: set() for i in np.unique(pairs)}
    for (i, j), iou in zip(pairs, bbox_overlap, strict=False):
        if iou >= iou_threshold:
            neigh[i].add(j)
            neigh[j].add(i)
    supporter_counts = np.zeros(M, dtype=np.int16)
    for idx, (i, j) in enumerate(pairs):
        supporter_counts[idx] = len(neigh[i].intersection(neigh[j]))
    return supporter_counts


def get_edge_weights(
    detections3d: Detections3D,
    pairs: np.ndarray,
    iou_threshold: float = 0.15,
    mode: Literal["iou", "supporters", "both"] = "both",
    obb_workers: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Compute edge weights for sparse pairs: IoU and/or supporter counts.

    Parameters
    ----------
    detections3d : Detections3D object
    pairs : (M,2) int array
    iou_threshold : float, supporters IoU threshold
    mode : 'iou','supporters','both'
    obb_workers : #workers for OBB IoU parallel

    Returns
    -------
    bbox_overlap : (M,) or None
    supporter_counts : (M,) or None
    """

    bboxes = detections3d.bboxes
    bboxes_type = detections3d.bboxes_type
    bbox_overlap = None
    supporter_counts = None

    # Compute 3D IoU (either for AABB or for OBB)
    # TODO: if statement left inside, because maybe I implement edge weights not based on 3D IoU in future
    # AABB
    if bboxes_type == "aabb":
        aabb = bboxes
        if mode in ("iou", "both", "supporters"):
            bbox_overlap = compute_aabb_iou_vectorized(aabb, pairs)
    # OBB
    elif bboxes_type == "obb":
        if mode in ("iou", "both", "supporters"):
            bbox_overlap = compute_obb_iou_parallel(
                centers=bboxes[:, :3],
                extents=bboxes[:, 3:6],
                quats=bboxes[:, 6:10],
                pairs=pairs,
                max_workers=obb_workers,
            )
    else:
        raise ValueError("bboxes must have 6 or 10 columns")

    if mode in ("supporters", "both"):
        supporter_counts = compute_supporter_counts(pairs, bbox_overlap, iou_threshold)

    if mode == "iou":
        return bbox_overlap, None
    elif mode == "supporters":
        return None, supporter_counts
    elif mode == "both":
        return bbox_overlap, supporter_counts
    else:
        raise ValueError(f"mode must be 'iou', 'supporters' or 'both', got {mode} instead")


def count_significant_overlaps(pairs: np.ndarray, bbox_overlap: np.ndarray, iou_threshold: float, N: int) -> np.ndarray:
    """
    Count, for each of N detections (its 3D bbox), with how many other detections (3D bboxes) it has a significant
     overlap with (overlaps > iou_threshold).

    Parameters
    ----------
    pairs : (M,2) int array of detection index pairs
    bbox_overlap : (M,) float IoUs for those pairs
    iou_threshold : float
    N : int total number of detections

    Returns
    -------
    counts : (N,) int array
        counts[k] = # of detections j where IoU(k,j) > threshold
    """
    counts = np.zeros(N, dtype=int)
    for (i, j), iou in zip(pairs, bbox_overlap, strict=False):
        if iou > iou_threshold:
            counts[i] += 1
            counts[j] += 1
    return counts
