import numpy as np
from typing import Tuple

def get_all_bbox_corners_from_min_max_corners(minimum_corner: np.ndarray, maximum_corner: np.ndarray) -> np.ndarray:
    # Gets all eight (8) corners of an axis-aligned bounding box (aabb) from 2 corners (min_xyz max_xyz)
    all_bbox_corners = np.stack(np.meshgrid(*zip(minimum_corner, maximum_corner), indexing='ij'), axis=-1) \
        .reshape(-1, 3)
    return all_bbox_corners

def get_min_max_corners_from_all_bbox_corners(corners: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    # Get min_xyz and max_xyz from all eight (8) corners of an axis-aligned bounding box (aabb)
    min_corner = np.min(corners, axis=0)
    max_corner = np.max(corners, axis=0)
    return min_corner, max_corner