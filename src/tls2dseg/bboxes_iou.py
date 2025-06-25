import numpy as np
from typing import Tuple, Optional, Union, List, Literal
import trimesh
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.transform import Rotation as R


def compute_obb_iou_naive(
        centers: np.ndarray,
        extents: np.ndarray,
        quats: np.ndarray,
        pairs: np.ndarray,
        engine: str = "scad"
) -> np.ndarray:
    """
    Compute 3D IoU for oriented bounding boxes (OBBs) using trimesh boolean intersection.

    Parameters
    ----------
    centers : (N,3) array
        OBB centers.
    extents : (N,3) array
        Full box dimensions along each local axis.
    quats : (N,4) array
        Quaternions in [x, y, z, w] order.
    pairs : (M,2) int array
        Each row [i,j] indicates a pair of boxes.
    engine : str
        Boolean engine for trimesh (e.g. 'scad', 'blender', 'cork').

    Returns
    -------
    ious : (M,) float array
        IoU for each OBB pair.
    """
    M = pairs.shape[0]
    ious = np.zeros(M, dtype=np.float32)

    for idx, (i, j) in enumerate(pairs):
        # Build box meshes
        box1 = trimesh.creation.box(extents=extents[i])
        box2 = trimesh.creation.box(extents=extents[j])

        # Apply transforms (rotation + translation)
        R1 = R.from_quat(quats[i, :]).as_matrix()
        T1 = np.eye(4)
        T1[:3, :3] = R1
        T1[:3, 3] = centers[i]
        box1.apply_transform(T1)

        R2 = R.from_quat(quats[j, :]).as_matrix()
        T2 = np.eye(4)
        T2[:3, :3] = R2
        T2[:3, 3] = centers[j]
        box2.apply_transform(T2)

        # Compute volumes
        vol1 = box1.volume
        vol2 = box2.volume

        # Boolean intersection
        try:
            inter = trimesh.boolean.intersection([box1, box2], engine=engine)
            inter_vol = inter.volume if inter is not None else 0.0
        except BaseException:
            inter_vol = 0.0

        union_vol = vol1 + vol2 - inter_vol
        ious[idx] = inter_vol / union_vol if union_vol > 0 else 0.0

    return ious


def _compute_iou_from_packed(
        args: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]
) -> float:
    ext_i, trans_i, ext_j, trans_j, engine = args
    box1 = trimesh.creation.box(extents=ext_i, transform=trans_i)
    box2 = trimesh.creation.box(extents=ext_j, transform=trans_j)
    v1, v2 = box1.volume, box2.volume
    try:
        inter = trimesh.boolean.intersection([box1, box2], engine=engine)
        iv = inter.volume if inter else 0.0
    except Exception:
        iv = 0.0
    union = v1 + v2 - iv
    return iv / union if union > 0 else 0.0


def compute_obb_iou_parallel(
        centers: np.ndarray,
        extents: np.ndarray,
        quats: np.ndarray,
        pairs: np.ndarray,
        engine: str = "scad",
        max_workers: Optional[int] = None
) -> np.ndarray:
    """
    Compute 3D IoU for OBBs by pre-packing only the per-pair extents/transforms.

    Parameters
    ----------
    centers : (N,3)  OBB centers
    extents : (N,3)  OBB dimensions
    quats   : (N,4)  [x,y,z,w] quaternions
    pairs   : (M,2)  integer index pairs
    engine  : bool engine for trimesh
    max_workers : # of processes

    Returns
    -------
    ious : (M,) float IoU per pair
    """
    # 1) Precompute all transforms
    r = R.from_quat(quat=quats)
    rots = r.as_matrix()  # (N,3,3)
    N = centers.shape[0]
    transforms = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    transforms[:, :3, :3] = rots
    transforms[:, :3, 3] = centers

    # 2) Pack per-pair arguments
    pack_args: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []
    for (i, j) in pairs:
        pack_args.append((extents[i], transforms[i],
                          extents[j], transforms[j],
                          engine))

    # 3) Parallel map
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        ious = list(exe.map(_compute_iou_from_packed, pack_args))

    return np.array(ious, dtype=np.float32)


def compute_aabb_iou_vectorized(
        aabb: np.ndarray,
        pairs: np.ndarray
) -> np.ndarray:
    """
    Vectorized IoU for axis-aligned bounding boxes.

    Parameters
    ----------
    aabb : (N,6) array [minx,miny,minz,maxx,maxy,maxz]
    pairs : (M,2) int array of index pairs

    Returns
    -------
    ious : (M,) float32 IoU per pair
    """
    # gather aabb corners
    mins = aabb[:, :3]
    maxs = aabb[:, 3:]
    # sort them according to pairs
    mins_i = mins[pairs[:, 0]]
    mins_j = mins[pairs[:, 1]]
    maxs_i = maxs[pairs[:, 0]]
    maxs_j = maxs[pairs[:, 1]]
    # intersection aabb and corresponding volume
    inter_min = np.maximum(mins_i, mins_j)
    inter_max = np.minimum(maxs_i, maxs_j)
    inter_dim = np.clip(inter_max - inter_min, 0, None)
    inter_vol = inter_dim[:, 0] * inter_dim[:, 1] * inter_dim[:, 2]
    # volumes (per each box in a pair and intersection)
    vol_i = np.prod(maxs_i - mins_i, axis=1)
    vol_j = np.prod(maxs_j - mins_j, axis=1)
    union_vol = vol_i + vol_j - inter_vol
    # 3D IoU
    return np.where(union_vol > 0, inter_vol / union_vol, 0.0).astype(np.float32)

