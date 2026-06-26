"""Point-in-volume containment helpers for tier_b_heavy fixture scoring.

Provides:
    point_in_obb  — test whether a world-frame point lies inside an OBB
    point_in_aabb — test whether a world-frame point lies inside an AABB
    ref_recall    — fraction of reference rows whose XYZ falls inside a
                    class-matching detected instance

Tier-a-importable: only numpy and scipy at module level.
Quaternion convention is [x, y, z, w] — identical to bboxes_iou.py.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger("tls2dseg.tests.integration.scoring.containment")


def point_in_obb(
    p: np.ndarray,
    center: np.ndarray,
    extent: np.ndarray,
    quat_xyzw: np.ndarray,
) -> bool:
    """Return True if world-frame point *p* lies inside the OBB.

    Parameters
    ----------
    p:
        (3,) point in world frame.
    center:
        (3,) OBB center in world frame.
    extent:
        (3,) full box dimensions along each local axis.
    quat_xyzw:
        (4,) quaternion [x, y, z, w] — local-to-world rotation
        (same convention as Detections3D.bboxes OBB rows and bboxes_iou.py).
    """
    R_wc = R.from_quat(quat_xyzw).as_matrix()  # local -> world
    p_local = R_wc.T @ (np.asarray(p) - np.asarray(center))
    return bool(np.all(np.abs(p_local) <= np.asarray(extent) / 2.0))


def point_in_aabb(p: np.ndarray, aabb_row: np.ndarray) -> bool:
    """Return True if point *p* lies inside the axis-aligned box.

    Parameters
    ----------
    p:
        (3,) point in world frame.
    aabb_row:
        (6,) row [minx, miny, minz, maxx, maxy, maxz].
    """
    aabb_min = aabb_row[:3]
    aabb_max = aabb_row[3:]
    return bool(np.all(aabb_min <= p) and np.all(p <= aabb_max))


def ref_recall(
    ref_xyz: np.ndarray,
    ref_classes: np.ndarray,
    det_classes: np.ndarray,
    det_bboxes: np.ndarray,
    bboxes_type: str,
    iou_thr: float = 0.0,  # unused here; kept for API consistency with report
) -> float:
    """Fraction of reference rows whose XYZ falls inside a class-matching detection.

    Parameters
    ----------
    ref_xyz:
        (M, 3) reference point coordinates.
    ref_classes:
        (M,) class label strings for each reference point.
    det_classes:
        (N,) class label strings for each detected instance.
    det_bboxes:
        (N, 6) AABB rows or (N, 10) OBB rows depending on *bboxes_type*.
    bboxes_type:
        "obb" or "aabb".

    Returns
    -------
    float
        matched_count / total_reference_count; 0.0 if no reference rows.
    """
    total = len(ref_xyz)
    if total == 0:
        return 0.0

    matched = 0
    for p, cls in zip(ref_xyz, ref_classes, strict=False):
        # Consider only detections whose class matches the reference label
        mask = det_classes == cls
        if not np.any(mask):
            continue
        det_bboxes_cls = det_bboxes[mask]
        found = False
        for bbox_row in det_bboxes_cls:
            if bboxes_type == "obb":
                center = bbox_row[:3]
                extent = bbox_row[3:6]
                quat = bbox_row[6:10]
                if point_in_obb(p, center, extent, quat):
                    found = True
                    break
            else:
                if point_in_aabb(p, bbox_row):
                    found = True
                    break
        if found:
            matched += 1

    return matched / total
