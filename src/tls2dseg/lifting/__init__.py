"""Public surface for the ``tls2dseg.lifting`` sub-package.

Phase 4 plan 02 Task 1 (ENG-07, D-B-02). Re-exports the two standalone pure
functions that back-project 2D segmentation masks to 3D scalar fields on a
PointCloudData instance.

Closest analog: ``tls2dseg.runtime.__init__`` (re-export surface pattern).
"""

from __future__ import annotations

import logging

from tls2dseg.lifting.masks_to_pcd import lift_mask_to_pcd, lift_masks_to_pcd

logger = logging.getLogger("tls2dseg.lifting")

__all__ = [
    "lift_mask_to_pcd",
    "lift_masks_to_pcd",
]
