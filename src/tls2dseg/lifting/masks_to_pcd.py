"""Standalone pure functions to back-project 2D segmentation masks to 3D scalar fields.

Phase 4 plan 02 Task 1 (ENG-07, D-B-02). Extracted verbatim from
``tls2dseg.pc2img_utils`` (functions ``project_masks2pcd_as_scalarfields`` and
``project_a_mask_2_pcd_as_scalarfield``), renamed per D-B-02:

* ``project_masks2pcd_as_scalarfields`` → ``lift_masks_to_pcd``
* ``project_a_mask_2_pcd_as_scalarfield`` → ``lift_mask_to_pcd``

The bodies are UNCHANGED from the source (PATTERNS.md: rename only). Only
imports and module-level logger are new.

Heavy-import rule (D-A-05): ``pchandler.geometry.PointCloudData`` is used only
at call-time (TYPE_CHECKING guard for annotation; real import deferred to
caller). Module-level imports are ``numpy`` only → this module is
tier_a-importable (no pchandler/pc2img/pyvips at module load).

Closest analog: ``tls2dseg.pc2img_utils`` lines 554-675.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pchandler.geometry import PointCloudData

logger = logging.getLogger("tls2dseg.lifting.masks_to_pcd")


def lift_masks_to_pcd(pcd: PointCloudData, instance_mask: np.ndarray, semantic_mask: np.ndarray) -> None:
    """Assign segmentation labels from an image to each point in the point cloud.

    Back-projects a paired instance + semantic mask (2D image coordinates)
    onto the 3D point cloud using spherical coordinates.  Writes two scalar
    fields: ``"instances"`` and ``"classes"``.

    Parameters
    ----------
    pcd :
        ``PointCloudData`` object with ``.spherical_coordinates`` and ``.fov``.
    instance_mask :
        ``np.ndarray`` of shape ``(H, W)`` with unique integer labels per
        instance (dtype ``np.int32`` or similar).
    semantic_mask :
        ``np.ndarray`` of shape ``(H, W)`` with semantic class labels.

    Modifies
    --------
    ``pcd.scalar_fields`` — adds/overwrites ``"instances"`` and ``"classes"``.

    Notes
    -----
    Extracted verbatim from ``pc2img_utils.project_masks2pcd_as_scalarfields``.
    The original function was removed from ``pc2img_utils`` in Plan 04-02.
    """
    # Unpack variables
    image_height, image_width = instance_mask.shape

    # 1: Extract spherical coordinates
    azimuth = pcd.spherical_coordinates[:, 2]
    elevation = pcd.spherical_coordinates[:, 1]

    # 2: Map spherical coordinates to normalized coordinates based on FOV
    # Extract FOV values in radians from image_stack.fov
    azimuth_min, elevation_min, azimuth_max, elevation_max = pcd.fov.as_numpy(unit="rad")

    # Normalize azimuth and elevation to [0, 1]
    normalized_azimuth = (azimuth - azimuth_min) / (azimuth_max - azimuth_min)
    normalized_elevation = (elevation - elevation_min) / (elevation_max - elevation_min)

    # 3: Map normalized coordinates to pixel coordinates
    u = normalized_azimuth * (image_width - 1)
    v = normalized_elevation * (image_height - 1)

    # Round and clip to valid indices
    u_int = np.clip(np.round(u).astype(int), 0, image_width - 1)
    v_int = np.clip(np.round(v).astype(int), 0, image_height - 1)

    # Handle edge cases: points outside FOV
    valid_indices = (
        (normalized_azimuth >= 0)
        & (normalized_azimuth <= 1)
        & (normalized_elevation >= 0)
        & (normalized_elevation <= 1)
    )

    # Initialize labels array with a default value (e.g., -1 for invalid points)
    all_labels_semantics = np.full(azimuth.shape, fill_value=-1, dtype=instance_mask.dtype)
    all_labels_instances = np.full(azimuth.shape, fill_value=-1, dtype=instance_mask.dtype)

    # Proceed only with valid points
    if np.any(valid_indices):
        valid_u_int = u_int[valid_indices]
        valid_v_int = v_int[valid_indices]
        labels_semantics = semantic_mask[valid_v_int, valid_u_int]
        labels_instances = instance_mask[valid_v_int, valid_u_int]

        all_labels_semantics[valid_indices] = labels_semantics
        all_labels_instances[valid_indices] = labels_instances

    # Step 5: Store instance and class labels into a scalar field
    pcd.scalar_fields.__setitem__("instances", all_labels_instances)
    pcd.scalar_fields.__setitem__("classes", all_labels_semantics)

    return None


def lift_mask_to_pcd(pcd: PointCloudData, mask: np.ndarray, mask_name: str) -> None:
    """Assign arbitrary values from an image to each point in the point cloud.

    Back-projects a single named mask (2D image coordinates) onto the 3D point
    cloud using spherical coordinates.  Writes one scalar field under
    ``mask_name``.

    Parameters
    ----------
    pcd :
        ``PointCloudData`` object with ``.spherical_coordinates`` and ``.fov``.
    mask :
        ``np.ndarray`` of shape ``(H, W)`` with arbitrary values.
    mask_name :
        Name of the scalar field to write on ``pcd``.

    Modifies
    --------
    ``pcd.scalar_fields`` — adds/overwrites the field named ``mask_name``.

    Notes
    -----
    Extracted verbatim from ``pc2img_utils.project_a_mask_2_pcd_as_scalarfield``.
    The original function was removed from ``pc2img_utils`` in Plan 04-02.
    """
    # Unpack variables
    image_height, image_width = mask.shape

    # 1: Extract spherical coordinates
    azimuth = pcd.spherical_coordinates[:, 2]
    elevation = pcd.spherical_coordinates[:, 1]

    # 2: Map spherical coordinates to normalized coordinates based on FOV
    # Extract FOV values in radians from image_stack.fov
    azimuth_min, elevation_min, azimuth_max, elevation_max = pcd.fov.as_numpy(unit="rad")

    # Normalize azimuth and elevation to [0, 1]
    normalized_azimuth = (azimuth - azimuth_min) / (azimuth_max - azimuth_min)
    normalized_elevation = (elevation - elevation_min) / (elevation_max - elevation_min)

    # 3: Map normalized coordinates to pixel coordinates
    u = normalized_azimuth * (image_width - 1)
    v = normalized_elevation * (image_height - 1)

    # Round and clip to valid indices
    u_int = np.clip(np.round(u).astype(int), 0, image_width - 1)
    v_int = np.clip(np.round(v).astype(int), 0, image_height - 1)

    # Handle edge cases: points outside FOV
    valid_indices = (
        (normalized_azimuth >= 0)
        & (normalized_azimuth <= 1)
        & (normalized_elevation >= 0)
        & (normalized_elevation <= 1)
    )

    # Initialize labels array with a default value (e.g., -1 for invalid points)
    all_values = np.full(azimuth.shape, fill_value=-1, dtype=mask.dtype)

    # Proceed only with valid points
    if np.any(valid_indices):
        valid_u_int = u_int[valid_indices]
        valid_v_int = v_int[valid_indices]

        values_i = mask[valid_v_int, valid_u_int]
        all_values[valid_indices] = values_i

    # Step 5: Store mask values as scalar field
    pcd.scalar_fields.__setitem__(mask_name, all_values)

    return None
